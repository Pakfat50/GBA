"""旧ハード自由振動で摩擦のみと摩擦＋バックラッシュを比較する。"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from backlash_model import information_criteria
from backlash_model import simulate_decay


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
WINDOWS_S = [(20.40, 32.00), (47.05, 59.05), (70.20, 82.00), (93.25, 105.25)]
TARGET_SAMPLE_PERIOD_S = 0.01
EPSILON_RAD_S = np.deg2rad(0.5)


def parse_arguments():
    parser = argparse.ArgumentParser(description="摩擦・バックラッシュ自由振動比較")
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def read_old_hardware_log(path):
    """旧ハードTXTから内軸角度を読み込む。"""

    rows = []
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) != 4:
            continue
        try:
            time_s = float(fields[0]) * 1e-6
            angle_deg = -0.0803317 * float(fields[2]) + 120.248
            rows.append([time_s, np.deg2rad(angle_deg)])
        except ValueError:
            pass
    data = np.asarray(rows, dtype=float)
    if len(data) == 0 or np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError("有効な単調増加時刻データを読み込めません")
    return data


def make_segments(raw_data):
    """従来と同じ4区間を100 Hzへ補間する。"""

    segments = []
    for start_s, end_s in WINDOWS_S:
        selected = raw_data[(raw_data[:, 0] >= start_s) & (raw_data[:, 0] <= end_s)]
        relative_raw_time = selected[:, 0] - selected[0, 0]
        time_s = np.arange(0.0, relative_raw_time[-1], TARGET_SAMPLE_PERIOD_S)
        angle_rad = np.interp(time_s, relative_raw_time, selected[:, 1])
        segments.append(
            {
                "time_s": time_s,
                "angle_rad": angle_rad,
                "device_start_s": float(selected[0, 0]),
            }
        )
    return segments


def fit_model(segment, include_backlash, starting_parameters=None):
    """1区間へVCまたはVCBモデルを最小二乗フィットする。"""

    time_s = segment["time_s"]
    measured = segment["angle_rad"]
    if starting_parameters is None:
        starting_parameters = np.asarray(
            [5.62, 0.083, 0.117, np.deg2rad(0.5), 0.0, measured[0], 0.0]
        )
    else:
        starting_parameters = np.asarray(starting_parameters, dtype=float).copy()

    lower = np.asarray([2.0, 0.0, 0.0, 0.0, -0.08, -1.2, -3.0])
    upper = np.asarray([10.0, 3.0, 1.0, np.deg2rad(5.0), 0.08, 1.2, 3.0])
    if include_backlash:
        fitted_indices = [0, 1, 2, 3, 4, 5, 6]
    else:
        fitted_indices = [0, 1, 2, 4, 5, 6]
        starting_parameters[3] = 0.0

    best = None
    starting_widths_deg = [0.0]
    if include_backlash:
        starting_widths_deg = [0.1, 0.5, 1.0, 2.0]

    for width_deg in starting_widths_deg:
        initial = starting_parameters.copy()
        initial[3] = np.deg2rad(width_deg)

        def unpack(values):
            parameters = initial.copy()
            parameters[fitted_indices] = values
            return parameters

        def residual(values):
            predicted, unused_play = simulate_decay(
                time_s, unpack(values), EPSILON_RAD_S
            )
            return predicted - measured

        optimization = least_squares(
            residual,
            initial[fitted_indices],
            bounds=(lower[fitted_indices], upper[fitted_indices]),
            diff_step=2e-3,
            max_nfev=160,
            xtol=2e-8,
            ftol=2e-8,
            gtol=2e-8,
        )
        parameters = unpack(optimization.x)
        predicted, play_state = simulate_decay(time_s, parameters, EPSILON_RAD_S)
        error = predicted - measured
        squared_error = float(np.sum(error * error))
        candidate = {
            "parameters": parameters,
            "prediction_rad": predicted,
            "play_state_rad": play_state,
            "residual_rad": error,
            "squared_error": squared_error,
            "success": bool(optimization.success),
            "message": str(optimization.message),
            "parameter_count": len(fitted_indices),
        }
        if best is None or squared_error < best["squared_error"]:
            best = candidate
    return best


def fit_all_segments(segments):
    """VCを先にフィットし、その結果をVCBの初期値へ使う。"""

    fits = {}
    metric_rows = []
    models = [("VC", False), ("VCB", True)]
    for model_name, include_backlash in models:
        index = 0
        while index < len(segments):
            start = None
            if include_backlash:
                start = fits[("VC", index)]["parameters"]
            result = fit_model(segments[index], include_backlash, start)
            fits[(model_name, index)] = result
            rmse_deg = float(np.sqrt(np.mean(np.rad2deg(result["residual_rad"]) ** 2)))
            aicc, bic = information_criteria(
                result["residual_rad"], result["parameter_count"]
            )
            parameters = result["parameters"]
            metric_rows.append(
                {
                    "model": model_name,
                    "segment": index + 1,
                    "rmse_deg": rmse_deg,
                    "aicc": aicc,
                    "bic": bic,
                    "k_over_i_per_s2": parameters[0],
                    "b_over_i_per_s": parameters[1],
                    "friction_over_i_rad_s2": parameters[2],
                    "backlash_half_width_deg": np.rad2deg(parameters[3]),
                    "backlash_total_width_deg": 2.0 * np.rad2deg(parameters[3]),
                    "offset_deg": np.rad2deg(parameters[4]),
                    "success": int(result["success"]),
                }
            )
            print(model_name + " Trial " + str(index + 1) + " RMSE=" + format(rmse_deg, ".5f"))
            index += 1
    return fits, metric_rows


def cross_validate(segments, fits):
    """他3区間の動力学係数中央値を残り1区間へ移植する。"""

    rows = []
    predictions = []
    for model_name in ["VC", "VCB"]:
        segment_index = 0
        while segment_index < len(segments):
            other_parameters = []
            other_index = 0
            while other_index < len(segments):
                if other_index != segment_index:
                    other_parameters.append(fits[(model_name, other_index)]["parameters"])
                other_index += 1
            parameters = np.median(np.asarray(other_parameters), axis=0)
            segment = segments[segment_index]
            time_s = segment["time_s"]
            measured = segment["angle_rad"]
            initial_mask = time_s <= 1.0

            def unpack_initial(values):
                trial_parameters = parameters.copy()
                trial_parameters[4:7] = values
                return trial_parameters

            def initial_residual(values):
                predicted, unused_play = simulate_decay(
                    time_s[initial_mask], unpack_initial(values), EPSILON_RAD_S
                )
                return predicted - measured[initial_mask]

            optimization = least_squares(
                initial_residual,
                parameters[4:7],
                bounds=([-0.08, -1.2, -3.0], [0.08, 1.2, 3.0]),
                max_nfev=100,
                diff_step=2e-3,
            )
            trial_parameters = unpack_initial(optimization.x)
            predicted, unused_play = simulate_decay(time_s, trial_parameters, EPSILON_RAD_S)
            valid = time_s > 1.0
            error_deg = np.rad2deg(predicted[valid] - measured[valid])
            rows.append(
                {
                    "model": model_name,
                    "segment": segment_index + 1,
                    "rmse_deg": float(np.sqrt(np.mean(error_deg * error_deg))),
                    "mae_deg": float(np.mean(np.abs(error_deg))),
                    "transferred_backlash_half_width_deg": np.rad2deg(parameters[3]),
                }
            )
            predictions.append(
                {
                    "model": model_name,
                    "segment": segment_index + 1,
                    "time_s": time_s,
                    "measured_rad": measured,
                    "predicted_rad": predicted,
                }
            )
            segment_index += 1
    return rows, predictions


def make_plots(segments, fits, metric_table, validation_table):
    """波形、残差、モデル比較、バックラッシュ幅を図にする。"""

    colors = {"VC": "#d62728", "VCB": "#0072b2"}
    figure, axes = plt.subplots(4, 2, figsize=(14, 11), sharex="row", sharey="row")
    segment_index = 0
    while segment_index < len(segments):
        model_index = 0
        for model_name in ["VC", "VCB"]:
            axis = axes[segment_index, model_index]
            segment = segments[segment_index]
            result = fits[(model_name, segment_index)]
            elapsed = segment["time_s"] + segment["device_start_s"]
            axis.plot(elapsed, np.rad2deg(segment["angle_rad"]), color="black", alpha=0.55, label="Measured")
            axis.plot(elapsed, np.rad2deg(result["prediction_rad"]), color=colors[model_name], linewidth=1.2, label=model_name)
            if segment_index == 0:
                axis.set_title(model_name)
            if model_index == 0:
                axis.set_ylabel("Trial " + str(segment_index + 1) + "\nAngle [deg]")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)
            model_index += 1
        segment_index += 1
    axes[-1, 0].set_xlabel("Device elapsed time [s]")
    axes[-1, 1].set_xlabel("Device elapsed time [s]")
    figure.suptitle("Free-decay fit: friction versus friction plus backlash")
    figure.tight_layout()
    figure.savefig(SCRIPT_DIRECTORY / "fits.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(4, 2, figsize=(14, 11), sharex="row", sharey="row")
    segment_index = 0
    while segment_index < len(segments):
        model_index = 0
        for model_name in ["VC", "VCB"]:
            axis = axes[segment_index, model_index]
            segment = segments[segment_index]
            result = fits[(model_name, segment_index)]
            elapsed = segment["time_s"] + segment["device_start_s"]
            residual_deg = np.rad2deg(result["prediction_rad"] - segment["angle_rad"])
            axis.plot(elapsed, residual_deg, color=colors[model_name], linewidth=0.9)
            axis.axhline(0.0, color="black", linewidth=0.6)
            if segment_index == 0:
                axis.set_title(model_name)
            if model_index == 0:
                axis.set_ylabel("Trial " + str(segment_index + 1) + "\nResidual [deg]")
            axis.grid(True, alpha=0.25)
            model_index += 1
        segment_index += 1
    axes[-1, 0].set_xlabel("Device elapsed time [s]")
    axes[-1, 1].set_xlabel("Device elapsed time [s]")
    figure.suptitle("Fit residuals")
    figure.tight_layout()
    figure.savefig(SCRIPT_DIRECTORY / "residuals.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(4)
    width = 0.34
    for model_index, model_name in enumerate(["VC", "VCB"]):
        fit_values = metric_table[metric_table["model"] == model_name]["rmse_deg"].to_numpy()
        validation_values = validation_table[validation_table["model"] == model_name]["rmse_deg"].to_numpy()
        axes[0].bar(x + model_index * width, fit_values, width=width, color=colors[model_name], label=model_name)
        axes[1].bar(x + model_index * width, validation_values, width=width, color=colors[model_name], label=model_name)
    backlash_values = metric_table[metric_table["model"] == "VCB"]["backlash_total_width_deg"].to_numpy()
    axes[2].bar(x, backlash_values, color=colors["VCB"])
    for axis in axes[:2]:
        axis.set_xticks(x + 0.5 * width, ["Trial 1", "Trial 2", "Trial 3", "Trial 4"])
        axis.set_ylabel("RMSE [deg]")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("In-trial fit")
    axes[1].set_title("Held-out dynamics")
    axes[2].set_xticks(x, ["Trial 1", "Trial 2", "Trial 3", "Trial 4"])
    axes[2].set_ylabel("Total backlash width [deg]")
    axes[2].set_title("Identified backlash")
    axes[2].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(SCRIPT_DIRECTORY / "comparison.png", dpi=170)
    plt.close(figure)


def main():
    arguments = parse_arguments()
    raw_data = read_old_hardware_log(arguments.input)
    segments = make_segments(raw_data)
    fits, metric_rows = fit_all_segments(segments)
    validation_rows, unused_predictions = cross_validate(segments, fits)
    metric_table = pd.DataFrame(metric_rows)
    validation_table = pd.DataFrame(validation_rows)
    metric_table.to_csv(SCRIPT_DIRECTORY / "fits.csv", index=False)
    validation_table.to_csv(SCRIPT_DIRECTORY / "validation.csv", index=False)
    make_plots(segments, fits, metric_table, validation_table)

    provenance = {
        "input": str(arguments.input),
        "sha256": hashlib.sha256(arguments.input.read_bytes()).hexdigest(),
        "windows_s": WINDOWS_S,
        "resampled_rate_hz": 1.0 / TARGET_SAMPLE_PERIOD_S,
        "friction_epsilon_deg_s": np.rad2deg(EPSILON_RAD_S),
        "models": {
            "VC": "viscous plus smoothed Coulomb friction",
            "VCB": "VC plus restoring-torque play backlash",
        },
        "backlash_parameter": "half width; reported total width is twice this value",
    }
    with open(SCRIPT_DIRECTORY / "provenance.json", "w", encoding="utf-8") as file_object:
        json.dump(provenance, file_object, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

