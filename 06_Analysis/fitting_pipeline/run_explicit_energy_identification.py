"""周期同定と陽な頂点間エネルギー式で自由減衰係数を同定する。

従来の全波形同時フィットからI、K、b、c、tauを切り離し、次の順に求める。

1. 同符号頂点間の周期と有限振幅補正からK/Iを求める。
2. 既知スペーサーの慣性・復元係数増分からI0、K0を求める。
3. 頂点間エネルギー差を陽な式へフィットし、b、c、tauを求める。
4. 固定した全係数で元の角度波形を数値積分し、偏差を評価する。
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from fitting_tools import component_increments
from fitting_tools import ensure_directory
from fitting_tools import explicit_half_cycle_energy_basis
from fitting_tools import extract_decay_turning_points
from fitting_tools import fit_quality_statistics
from fitting_tools import identify_base_inertia_and_restoring
from fitting_tools import identify_k_over_i_from_turning_points
from fitting_tools import read_angle_log
from fitting_tools import save_csv
from fitting_tools import simulate_normalized_decay
from run_calibration import FRICTION_EPSILON_DEG_S
from run_calibration import read_manifest
from run_calibration import read_waveform_selection
from run_calibration import resolve_physical_inputs


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "04_Data" / "05_Fitting"
DEFAULT_RESULT_ROOT = SCRIPT_DIRECTORY / "results"
MINIMUM_FREQUENCY_AMPLITUDE_DEG = 5.0
MINIMUM_ENERGY_AMPLITUDE_DEG = 4.0
BALL_ROD_QUADRATIC_RATIO = (0.129 / 0.229) ** 4


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="周期と陽な頂点間エネルギー式による自由減衰同定"
    )
    parser.add_argument("--date", required=True, help="試験日フォルダ。例: 20260921")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args()


def load_waveform_records(date_directory, selection):
    """確認済み範囲を読み、独立な静止基準から頂点を抽出する。"""

    records = []
    cache = {}
    for unused_index, row in selection.iterrows():
        if int(row["use_for_fitting"]) != 1:
            continue
        key = (str(row["data_file"]), str(row["angle_column"]))
        if key not in cache:
            cache[key] = read_angle_log(
                date_directory / key[0], key[1]
            )
        complete_time, complete_angle = cache[key]
        start = int(row["start_index"])
        end = int(row["end_index"]) + 1
        time_s = complete_time[start:end]
        angle_rad = complete_angle[start:end]
        center_rad = np.deg2rad(float(row["baseline_deg"]))
        turning = extract_decay_turning_points(
            time_s,
            angle_rad,
            center_rad,
        )
        frequency = identify_k_over_i_from_turning_points(
            turning["time_s"],
            turning["amplitude_rad"],
            MINIMUM_FREQUENCY_AMPLITUDE_DEG,
        )
        records.append(
            {
                "segment_id": str(row["segment_id"]),
                "axis": str(row["axis"]).strip().upper(),
                "configuration": str(row["configuration"]).strip().upper(),
                "direction": str(row["direction"]).strip().upper(),
                "repetition": int(row["repetition"]),
                "center_rad": center_rad,
                "time_s": time_s,
                "angle_rad": angle_rad,
                "turning": turning,
                "frequency": frequency,
            }
        )
    return records


def summarize_frequency(records, manifest):
    """波形別K/Iを形態ごとに集約し、既知の追加量を付ける。"""

    segment_rows = []
    for record in records:
        frequency = record["frequency"]
        segment_rows.append(
            {
                "segment_id": record["segment_id"],
                "axis": record["axis"],
                "configuration": record["configuration"],
                "direction": record["direction"],
                "repetition": record["repetition"],
                "center_deg": float(np.rad2deg(record["center_rad"])),
                "turning_points": len(record["turning"]["indices"]),
                "periods_used": frequency["number_of_periods"],
                "k_over_i_per_s2": frequency["k_over_i_per_s2"],
                "period_sample_std_per_s2": frequency["sample_std_per_s2"],
                "period_robust_sigma_per_s2": frequency["robust_sigma_per_s2"],
            }
        )

    segment_table = pd.DataFrame(segment_rows)
    level_rows = []
    for unused_index, manifest_row in manifest.iterrows():
        if int(manifest_row["valid"]) != 1:
            continue
        axis = str(manifest_row["axis"]).strip().upper()
        configuration = str(manifest_row["configuration"]).strip().upper()
        matching = segment_table[
            (segment_table["axis"] == axis)
            & (segment_table["configuration"] == configuration)
        ]
        values = matching["k_over_i_per_s2"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        uncertainty = max(
            standard_deviation / np.sqrt(max(len(values), 1)),
            0.001 * mean,
        )
        delta_inertia, delta_restoring = component_increments(manifest_row)
        level_rows.append(
            {
                "axis": axis,
                "configuration": configuration,
                "spacer_count": int(manifest_row["spacer_count"]),
                "use_for_calibration": int(manifest_row["use_for_calibration"]),
                "number_of_segments": len(values),
                "k_over_i_mean": mean,
                "k_over_i_std": standard_deviation,
                "k_over_i_uncertainty": uncertainty,
                "delta_inertia": float(delta_inertia),
                "delta_restoring": float(delta_restoring),
                "force_lever_m": float(manifest_row["force_lever_m"]),
            }
        )
    return segment_rows, level_rows


def identify_inertia_and_restoring(level_rows):
    """周期由来K/Iだけを使い、各形態のI、Kを確定する。"""

    axes = sorted({row["axis"] for row in level_rows})
    base_rows = []
    checked_level_rows = []
    physical = {}
    for axis in axes:
        axis_levels = [row for row in level_rows if row["axis"] == axis]
        calibration = [row for row in axis_levels if row["use_for_calibration"] == 1]
        inertia_zero, restoring_zero, checked = identify_base_inertia_and_restoring(
            calibration
        )
        measured = np.asarray([row["k_over_i_mean"] for row in checked])
        predicted = np.asarray([row["k_over_i_predicted"] for row in checked])
        rmse, correlation, r_squared = fit_quality_statistics(measured, predicted)
        base_rows.append(
            {
                "axis": axis,
                "base_inertia_kg_m2": inertia_zero,
                "base_restoring_n_m_per_rad": restoring_zero,
                "calibration_levels": len(calibration),
                "calibration_rmse_per_s2": rmse,
                "calibration_r": correlation,
                "calibration_r_squared": r_squared,
            }
        )

        checked_by_configuration = {
            row["configuration"]: row for row in checked
        }
        for row in axis_levels:
            output = dict(row)
            if row["configuration"] in checked_by_configuration:
                output = checked_by_configuration[row["configuration"]]
            else:
                output["k_over_i_predicted"] = ""
                output["k_over_i_residual"] = ""
            checked_level_rows.append(output)

            restoring_from_gravity = restoring_zero + row["delta_restoring"]
            geometric_inertia = inertia_zero + row["delta_inertia"]
            inertia = geometric_inertia
            if row["configuration"] == "BALL":
                # 球の空気付加慣性を含む実効Iを周期から逆算する。
                inertia = restoring_from_gravity / row["k_over_i_mean"]
                restoring = restoring_from_gravity
            else:
                # スペーサーの追加Iは既知として、各形態の実測周期K/IをODEへ
                # 直接反映する。これにより、較正曲線の残差と減衰誤差を分離する。
                restoring = inertia * row["k_over_i_mean"]
            physical[(axis, row["configuration"])] = {
                "inertia_kg_m2": float(inertia),
                "geometric_inertia_kg_m2": float(geometric_inertia),
                "restoring_n_m_per_rad": float(restoring),
                "gravity_model_restoring_n_m_per_rad": float(
                    restoring_from_gravity
                ),
                "k_over_i_per_s2": float(row["k_over_i_mean"]),
                "force_lever_m": float(row["force_lever_m"]),
            }
    return base_rows, checked_level_rows, physical


def make_energy_system(records, physical):
    """頂点間エネルギー差を共有係数の線形系へ変換する。"""

    parameter_names = [
        "b_IN",
        "b_OUT",
        "tau_IN",
        "tau_OUT",
        "c_rod",
        "c_sphere",
    ]
    matrix = []
    target = []
    scales = []
    interval_rows = []
    minimum_amplitude = np.deg2rad(MINIMUM_ENERGY_AMPLITUDE_DEG)

    for record in records:
        key = (record["axis"], record["configuration"])
        inertia = physical[key]["inertia_kg_m2"]
        restoring = physical[key]["restoring_n_m_per_rad"]
        turning = record["turning"]
        amplitude = turning["amplitude_rad"]
        for index in range(len(amplitude) - 1):
            first = float(amplitude[index])
            last = float(amplitude[index + 1])
            if min(first, last) < minimum_amplitude:
                continue
            measured_loss = restoring * (np.cos(last) - np.cos(first))
            if measured_loss <= 0.0:
                continue
            basis = explicit_half_cycle_energy_basis(first, inertia, restoring)
            row = np.zeros(len(parameter_names), dtype=float)
            if record["axis"] == "IN":
                row[0] = basis["viscous_basis"]
                row[2] = basis["friction_basis"]
            else:
                row[1] = basis["viscous_basis"]
                row[3] = basis["friction_basis"]
            if record["configuration"] == "BALL":
                row[4] = BALL_ROD_QUADRATIC_RATIO * basis["quadratic_basis"]
                row[5] = basis["quadratic_basis"]
            else:
                row[4] = basis["quadratic_basis"]

            # エネルギー残差を、おおむね角度残差へ換算して均等化する。
            representative = 0.5 * (first + last)
            scale = restoring * max(
                np.sin(representative),
                np.sin(minimum_amplitude),
            )
            matrix.append(row)
            target.append(measured_loss)
            scales.append(scale)
            interval_rows.append(
                {
                    "segment_id": record["segment_id"],
                    "axis": record["axis"],
                    "configuration": record["configuration"],
                    "transition": (
                        "+to-" if turning["sign"][index] > 0 else "-to+"
                    ),
                    "interval_number": index + 1,
                    "start_time_s": float(turning["time_s"][index]),
                    "end_time_s": float(turning["time_s"][index + 1]),
                    "start_amplitude_deg": float(np.rad2deg(first)),
                    "end_amplitude_deg": float(np.rad2deg(last)),
                    "measured_energy_loss_j": float(measured_loss),
                    "viscous_basis": basis["viscous_basis"],
                    "quadratic_basis": basis["quadratic_basis"],
                    "friction_basis": basis["friction_basis"],
                    "theoretical_half_period_s": basis["half_period_s"],
                    "measured_half_period_s": float(
                        turning["time_s"][index + 1]
                        - turning["time_s"][index]
                    ),
                }
            )
    return (
        parameter_names,
        np.asarray(matrix),
        np.asarray(target),
        np.asarray(scales),
        interval_rows,
    )


def fit_energy_coefficients(parameter_names, matrix, target, scales, interval_rows):
    """共有制約付きでb、c、tauを非負ロバスト最小二乗同定する。"""

    weighted_matrix = matrix / scales[:, None]
    weighted_target = target / scales
    initial = np.asarray([1e-5, 3e-5, 1e-4, 1e-4, 1e-6, 1e-5])
    result = least_squares(
        lambda parameters: weighted_matrix @ parameters - weighted_target,
        initial,
        bounds=(0.0, np.inf),
        loss="soft_l1",
        f_scale=np.deg2rad(0.05),
        x_scale="jac",
        max_nfev=5000,
    )
    if not result.success:
        raise RuntimeError("陽なエネルギー係数同定に失敗しました: " + result.message)
    coefficients = dict(zip(parameter_names, result.x))
    predicted = matrix @ result.x
    angle_equivalent_residual = (predicted - target) / scales

    for index, row in enumerate(interval_rows):
        row["predicted_energy_loss_j"] = float(predicted[index])
        row["energy_loss_residual_j"] = float(predicted[index] - target[index])
        row["angle_equivalent_residual_deg"] = float(
            np.rad2deg(angle_equivalent_residual[index])
        )

    normalized = weighted_matrix / np.maximum(
        np.linalg.norm(weighted_matrix, axis=0), 1e-30
    )
    diagnostics = {
        "success": int(result.success),
        "message": str(result.message),
        "intervals": len(target),
        "angle_equivalent_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(angle_equivalent_residual**2)))
        ),
        "normalized_design_condition_number": float(np.linalg.cond(normalized)),
    }
    return coefficients, diagnostics, interval_rows


def make_parameter_rows(physical, coefficients):
    rows = []
    for axis, configuration in sorted(physical):
        values = physical[(axis, configuration)]
        damping = coefficients["b_" + axis]
        friction = coefficients["tau_" + axis]
        if configuration == "BALL":
            quadratic = coefficients["c_sphere"]
            quadratic += BALL_ROD_QUADRATIC_RATIO * coefficients["c_rod"]
        else:
            quadratic = coefficients["c_rod"]
        row = {
            "axis": axis,
            "configuration": configuration,
            **values,
            "damping_n_m_s_per_rad": float(damping),
            "quadratic_n_m_s2_per_rad2": float(quadratic),
            "friction_n_m": float(friction),
            "b_over_i_per_s": float(damping / values["inertia_kg_m2"]),
            "c_over_i_per_rad": float(quadratic / values["inertia_kg_m2"]),
            "tau_over_i_rad_s2": float(friction / values["inertia_kg_m2"]),
        }
        rows.append(row)
    return rows


def validate_waveforms(records, physical, coefficients, previous_fit_path):
    """固定係数で元時系列を再計算し、波形偏差と包絡線偏差を求める。"""

    previous = None
    if previous_fit_path.exists():
        previous = pd.read_csv(previous_fit_path, encoding="utf-8-sig").set_index(
            "segment_id"
        )
    metrics = []
    plot_records = []
    minimum_amplitude = np.deg2rad(MINIMUM_ENERGY_AMPLITUDE_DEG)
    epsilon = np.deg2rad(FRICTION_EPSILON_DEG_S)

    for record in records:
        key = (record["axis"], record["configuration"])
        values = physical[key]
        inertia = values["inertia_kg_m2"]
        restoring = values["restoring_n_m_per_rad"]
        damping = coefficients["b_" + record["axis"]]
        friction = coefficients["tau_" + record["axis"]]
        if record["configuration"] == "BALL":
            quadratic = coefficients["c_sphere"]
            quadratic += BALL_ROD_QUADRATIC_RATIO * coefficients["c_rod"]
        else:
            quadratic = coefficients["c_rod"]

        turning = record["turning"]
        valid_turning = np.flatnonzero(turning["amplitude_rad"] >= minimum_amplitude)
        if len(valid_turning) < 3:
            continue
        first_turning = valid_turning[0]
        last_turning = valid_turning[-1]
        start_index = int(turning["indices"][first_turning])
        end_index = int(turning["indices"][last_turning]) + 1
        time_s = record["time_s"][start_index:end_index]
        relative_time = time_s - time_s[0]
        measured = record["angle_rad"][start_index:end_index] - record["center_rad"]
        initial_angle = (
            turning["smoothed_angle_rad"][start_index] - record["center_rad"]
        )
        parameters = np.asarray(
            [
                restoring / inertia,
                damping / inertia,
                quadratic / inertia,
                friction / inertia,
                0.0,
                initial_angle,
                0.0,
            ]
        )
        predicted = simulate_normalized_decay(relative_time, parameters, epsilon)
        rmse_rad, correlation, r_squared = fit_quality_statistics(measured, predicted)

        # 陽な式を再帰的に使って、頂点包絡線そのものの累積誤差も求める。
        measured_amplitude = turning["amplitude_rad"][
            first_turning : last_turning + 1
        ]
        predicted_amplitude = [float(measured_amplitude[0])]
        current_energy = restoring * (1.0 - np.cos(predicted_amplitude[0]))
        for unused_index in range(len(measured_amplitude) - 1):
            if current_energy <= 0.0 or predicted_amplitude[-1] <= 0.0:
                predicted_amplitude.append(0.0)
                continue
            basis = explicit_half_cycle_energy_basis(
                predicted_amplitude[-1], inertia, restoring
            )
            loss = damping * basis["viscous_basis"]
            loss += quadratic * basis["quadratic_basis"]
            loss += friction * basis["friction_basis"]
            current_energy = max(0.0, current_energy - loss)
            next_amplitude = np.arccos(
                np.clip(1.0 - current_energy / restoring, -1.0, 1.0)
            )
            predicted_amplitude.append(float(next_amplitude))
        predicted_amplitude = np.asarray(predicted_amplitude)
        envelope_rmse = float(
            np.rad2deg(
                np.sqrt(np.mean((predicted_amplitude - measured_amplitude) ** 2))
            )
        )

        previous_rmse = float("nan")
        if previous is not None and record["segment_id"] in previous.index:
            previous_rmse = float(previous.loc[record["segment_id"], "rmse_deg"])
        metrics.append(
            {
                "segment_id": record["segment_id"],
                "axis": record["axis"],
                "configuration": record["configuration"],
                "direction": record["direction"],
                "samples": len(measured),
                "turning_points": len(measured_amplitude),
                "waveform_rmse_deg": float(np.rad2deg(rmse_rad)),
                "waveform_r": correlation,
                "waveform_r_squared": r_squared,
                "envelope_rmse_deg": envelope_rmse,
                "previous_full_fit_rmse_deg": previous_rmse,
            }
        )
        plot_records.append(
            {
                "segment_id": record["segment_id"],
                "time_s": relative_time,
                "measured_rad": measured,
                "predicted_rad": predicted,
                "rmse_deg": float(np.rad2deg(rmse_rad)),
                "r_squared": r_squared,
            }
        )
    return metrics, plot_records


def plot_frequency_calibration(level_rows, base_rows, output_path):
    table = pd.DataFrame(level_rows)
    base = pd.DataFrame(base_rows).set_index("axis")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    for column, axis in enumerate(["IN", "OUT"]):
        plot_axis = axes[0, column]
        levels = table[
            (table["axis"] == axis) & (table["use_for_calibration"] == 1)
        ].sort_values("spacer_count")
        plot_axis.errorbar(
            levels["spacer_count"],
            levels["k_over_i_mean"],
            yerr=levels["k_over_i_uncertainty"],
            fmt="o",
            capsize=3,
            label="period measurement",
        )
        plot_axis.plot(
            levels["spacer_count"],
            levels["k_over_i_predicted"],
            "--",
            label="I0, K0 calibration",
        )
        values = base.loc[axis]
        plot_axis.set_title(
            f"{axis}: R$^2$={values['calibration_r_squared']:.5f}"
        )
        plot_axis.set_xlabel("spacer count")
        plot_axis.set_ylabel("K/I [s$^{-2}$]")
        plot_axis.grid(True, alpha=0.3)
        plot_axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def plot_energy_fit(interval_rows, output_path):
    table = pd.DataFrame(interval_rows)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, color in [("IN", "#1f77b4"), ("OUT", "#d62728")]:
        values = table[table["axis"] == axis]
        axes[0].scatter(
            values["measured_energy_loss_j"],
            values["predicted_energy_loss_j"],
            s=8,
            alpha=0.35,
            label=axis,
            color=color,
        )
        axes[1].scatter(
            values["start_amplitude_deg"],
            values["angle_equivalent_residual_deg"],
            s=8,
            alpha=0.35,
            label=axis,
            color=color,
        )
    maximum = max(
        table["measured_energy_loss_j"].max(),
        table["predicted_energy_loss_j"].max(),
    )
    axes[0].plot([0.0, maximum], [0.0, maximum], "k--", linewidth=1)
    axes[0].set_xlabel("measured half-cycle loss [J]")
    axes[0].set_ylabel("predicted half-cycle loss [J]")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("previous peak amplitude [deg]")
    axes[1].set_ylabel("angle-equivalent residual [deg]")
    for plot_axis in axes:
        plot_axis.grid(True, alpha=0.3)
        plot_axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def plot_waveform_overview(plot_records, output_path):
    columns = 4
    rows = int(np.ceil(len(plot_records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 2.5 * rows), squeeze=False)
    for index, record in enumerate(plot_records):
        plot_axis = axes[index // columns, index % columns]
        plot_axis.plot(
            record["time_s"], np.rad2deg(record["measured_rad"]), linewidth=0.8
        )
        plot_axis.plot(
            record["time_s"],
            np.rad2deg(record["predicted_rad"]),
            "--",
            linewidth=0.8,
        )
        plot_axis.set_title(
            f"{record['segment_id']}\nRMSE={record['rmse_deg']:.2f} deg, "
            f"R2={record['r_squared']:.3f}",
            fontsize=8,
        )
        plot_axis.grid(True, alpha=0.2)
    for index in range(len(plot_records), rows * columns):
        axes[index // columns, index % columns].axis("off")
    figure.supxlabel("time from first turning point [s]")
    figure.supylabel("angle about independently measured center [deg]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=100)
    plt.close(figure)


def write_report(
    output_path,
    records,
    base_rows,
    parameter_rows,
    coefficients,
    diagnostics,
    metrics,
    direction_rows,
):
    metric_table = pd.DataFrame(metrics)
    base_table = pd.DataFrame(base_rows).set_index("axis")
    parameter_table = pd.DataFrame(parameter_rows)
    direction_table = pd.DataFrame(direction_rows)
    ball = parameter_table[parameter_table["configuration"] == "BALL"].set_index("axis")
    lines = [
        "# 周期・陽エネルギー式による係数同定評価",
        "",
        "## 方法",
        "",
        "- `I`, `K`: 同符号頂点間の周期に有限振幅楕円積分補正を適用し、スペーサ既知増分から分離",
        "- `b`, `c`, `tau`: 前の頂点振幅だけで明示した半周期エネルギー損失へ非負ロバスト最小二乗フィット",
        "- `c_rod`: 球なし全形態・IN/OUTで共通",
        "- `c_sphere`: IN/OUTで共通。球ありは `c_sphere + 0.101*c_rod`",
        "- `b`, `tau`: 軸ごとに共通",
        "- 評価: 固定係数を元の非線形ODEへ入れ、最初の有効頂点から実波形を再計算",
        "",
        "```text",
        "Delta_E[n] = b*B(A[n]) + c*C(A[n]) + tau*R(A[n])",
        "B(A) = 8*sqrt(K/I)*{E(k) - (1-k^2)*K(k)}",
        "C(A) = 4*(K/I)*(sin(A) - A*cos(A))",
        "R(A) = 2*A,  k = sin(A/2)",
        "```",
        "",
        "## 周期から求めた基準I、K",
        "",
        "| 軸 | I0 [kg m^2] | K0 [N m/rad] | 較正R^2 |",
        "|---|---:|---:|---:|",
    ]
    for axis in ["IN", "OUT"]:
        row = base_table.loc[axis]
        lines.append(
            f"| {axis} | {row['base_inertia_kg_m2']:.8e} | "
            f"{row['base_restoring_n_m_per_rad']:.8e} | "
            f"{row['calibration_r_squared']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 陽なエネルギー式の共有係数",
            "",
            "| 係数 | 値 |",
            "|---|---:|",
        ]
    )
    for name in ["b_IN", "b_OUT", "tau_IN", "tau_OUT", "c_rod", "c_sphere"]:
        lines.append(f"| `{name}` | {coefficients[name]:.8e} |")
    lines.extend(
        [
            "",
            f"使用した頂点間隔は{diagnostics['intervals']}個、1区間先の"
            f"角度換算RMSEは{diagnostics['angle_equivalent_rmse_deg']:.3f} degである。",
            "",
            "## 球あり最終係数",
            "",
            "| 軸 | I [kg m^2] | K [N m/rad] | b [N m s/rad] | c [N m s^2/rad^2] | tau [N m] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for axis in ["IN", "OUT"]:
        row = ball.loc[axis]
        lines.append(
            f"| {axis} | {row['inertia_kg_m2']:.8e} | "
            f"{row['restoring_n_m_per_rad']:.8e} | "
            f"{row['damping_n_m_s_per_rad']:.8e} | "
            f"{row['quadratic_n_m_s2_per_rad2']:.8e} | "
            f"{row['friction_n_m']:.8e} |"
        )
    lines.extend(
        [
            "",
            "## 元波形による評価",
            "",
            "| 指標 | 平均 | 中央値 | 最大 |",
            "|---|---:|---:|---:|",
            f"| 波形RMSE [deg] | {metric_table['waveform_rmse_deg'].mean():.3f} | "
            f"{metric_table['waveform_rmse_deg'].median():.3f} | "
            f"{metric_table['waveform_rmse_deg'].max():.3f} |",
            f"| 頂点包絡線RMSE [deg] | {metric_table['envelope_rmse_deg'].mean():.3f} | "
            f"{metric_table['envelope_rmse_deg'].median():.3f} | "
            f"{metric_table['envelope_rmse_deg'].max():.3f} |",
            f"| 従来の波形別同時フィットRMSE [deg] | "
            f"{metric_table['previous_full_fit_rmse_deg'].mean():.3f} | "
            f"{metric_table['previous_full_fit_rmse_deg'].median():.3f} | "
            f"{metric_table['previous_full_fit_rmse_deg'].max():.3f} |",
            "",
            "陽な方法は1区間先の損失を直接説明するが、再帰包絡線と元波形では誤差が累積する。"
            "特に波形RMSEには減衰誤差だけでなく、共通K/Iと各試行固有周期の微小差による位相ずれも含まれる。",
            "",
            "## 折り返し方向別の残差",
            "",
            "| 軸 | 遷移 | 平均残差 [deg] | 標準偏差 [deg] | 区間数 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for unused_index, row in direction_table.iterrows():
        lines.append(
            f"| {row['axis']} | {row['transition']} | "
            f"{row['mean_angle_equivalent_residual_deg']:.3f} | "
            f"{row['std_angle_equivalent_residual_deg']:.3f} | "
            f"{int(row['intervals'])} |"
        )
    lines.extend(
        [
            "",
            "OUT軸では `+to-` と `-to+` の平均残差が反対符号となる。これは方向依存摩擦、"
            "平衡角誤差、センサ非対称性のいずれでも生じ得るため、現段階ではヒステリシスの"
            "候補として扱い、方向別tauモデルとの比較を次段階とする。",
            "",
            "## 図とCSV",
            "",
            "- [周期較正](frequency_calibration.png)",
            "- [頂点間エネルギー損失](energy_fit.png)",
            "- [全波形の固定係数再現](waveform_validation.png)",
            "- [波形別周期](frequency_segments.csv)",
            "- [頂点間損失](energy_intervals.csv)",
            "- [折り返し方向別残差](direction_residuals.csv)",
            "- [波形別評価値](waveform_metrics.csv)",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    arguments = parse_arguments()
    date_directory = arguments.data_root / arguments.date
    if not date_directory.is_dir():
        raise FileNotFoundError("試験日フォルダがありません: " + str(date_directory))
    parent_result = arguments.result_root / arguments.date
    output_directory = parent_result / "explicit_energy"
    ensure_directory(output_directory)

    manifest, unused_manifest_path = read_manifest(date_directory)
    manifest = resolve_physical_inputs(manifest, parent_result)
    selection, selection_path = read_waveform_selection(parent_result)
    if selection is None:
        raise FileNotFoundError("確認済み波形表がありません: " + str(selection_path))

    print("1/4 頂点周期からK/Iを同定")
    records = load_waveform_records(date_directory, selection)
    frequency_segments, level_rows = summarize_frequency(records, manifest)
    base_rows, checked_levels, physical = identify_inertia_and_restoring(level_rows)
    save_csv(output_directory / "frequency_segments.csv", frequency_segments)
    save_csv(output_directory / "frequency_levels.csv", checked_levels)
    save_csv(output_directory / "base_parameters.csv", base_rows)

    print("2/4 陽な頂点間エネルギー式からb、c、tauを同定")
    names, matrix, target, scales, interval_rows = make_energy_system(records, physical)
    coefficients, diagnostics, interval_rows = fit_energy_coefficients(
        names, matrix, target, scales, interval_rows
    )
    parameter_rows = make_parameter_rows(physical, coefficients)
    save_csv(output_directory / "energy_intervals.csv", interval_rows)
    interval_table = pd.DataFrame(interval_rows)
    direction_rows = []
    for (axis, transition), group in interval_table.groupby(["axis", "transition"]):
        residual = group["angle_equivalent_residual_deg"].to_numpy(dtype=float)
        direction_rows.append(
            {
                "axis": axis,
                "transition": transition,
                "mean_angle_equivalent_residual_deg": float(np.mean(residual)),
                "std_angle_equivalent_residual_deg": float(np.std(residual, ddof=1)),
                "intervals": len(residual),
            }
        )
    save_csv(output_directory / "direction_residuals.csv", direction_rows)
    save_csv(output_directory / "identified_parameters.csv", parameter_rows)
    save_csv(
        output_directory / "energy_fit_summary.csv",
        [{**coefficients, **diagnostics}],
    )

    print("3/4 固定係数で元波形を再計算")
    metrics, plot_records = validate_waveforms(
        records,
        physical,
        coefficients,
        parent_result / "segment_fits.csv",
    )
    save_csv(output_directory / "waveform_metrics.csv", metrics)

    print("4/4 図とレポートを作成")
    plot_frequency_calibration(
        checked_levels, base_rows, output_directory / "frequency_calibration.png"
    )
    plot_energy_fit(interval_rows, output_directory / "energy_fit.png")
    plot_waveform_overview(
        plot_records, output_directory / "waveform_validation.png"
    )
    write_report(
        output_directory / "FITTING_REPORT.md",
        records,
        base_rows,
        parameter_rows,
        coefficients,
        diagnostics,
        metrics,
        direction_rows,
    )
    print("結果: " + str(output_directory))


if __name__ == "__main__":
    main()
