"""OW-03: tune observers on one wind record and validate on new records.

This script uses the adopted nonlinear HBK coefficients for the BALL setup.
The plant and observer use the same mechanical coefficients (model match),
and the simulated angle sensor is ideal.  Observer settings are selected on
one Kaimal training wind record only.  They are then frozen and evaluated on
different calm, steady, turbulent, and gust inputs.

Run at the repository root:
    python 06_Analysis/simulation/src/run_ow03_hbk_observer_tuning.py

The output folder receives the full parameter scan, evaluation tables, a
JSON summary, comparison plots, and a Japanese report.  Sensor noise,
quantization, and delay are intentionally left for OW-05.  If a simulated
angle exceeds the physical +/-60 degree range, that case is marked invalid
for the hardware and is not silently clipped.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from hbk_model_coefficients import select_coefficients, simulate_hbk_plant
from hbk_nonlinear_estimators import (
    calculate_luenberger_gain,
    nonlinear_ekf_rts_force,
    nonlinear_luenberger_force,
)
from run_real_wind_doe import causal_lowpass, wind_speed_from_force
from wind import drag_force_from_speed, synthesize_kaimal_wind


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "ow03_hbk_observer_tuning.json"
DEFAULT_OUTPUT = ROOT / "results" / "observer_wind" / "ow03_tuning"
METHODS = (
    "静的換算", "因果LPF", "ESO 3状態", "ESO 4状態",
    "RTS 3状態（オフライン）", "RTS 4状態（オフライン）",
)
CASE_LABELS = {
    "無風": "Calm",
    "一定風 2 m/s": "Steady 2 m/s",
    "Kaimal乱流 平均3.75 m/s TI20% 最大6 m/s": "Kaimal 3.75 m/s, TI 20%, max 6 m/s",
    "独立Kaimal乱流 平均2 m/s TI20%": "Kaimal 2 m/s, TI 20%",
    "ガスト 2→6 m/s": "Gust 2 to 6 m/s",
}
COLORS = {
    "静的換算": "#777777",
    "因果LPF": "#8e6bbd",
    "ESO 3状態": "#2776bc",
    "ESO 4状態": "#e67e22",
    "RTS 3状態（オフライン）": "#159477",
    "RTS 4状態（オフライン）": "#006b4f",
}
LABELS = {
    "静的換算": "Static",
    "因果LPF": "Causal LPF",
    "ESO 3状態": "ESO 3-state",
    "ESO 4状態": "ESO 4-state",
    "RTS 3状態（オフライン）": "RTS 3-state (offline)",
    "RTS 4状態（オフライン）": "RTS 4-state (offline)",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a table with its column names; UTF-8 BOM opens well in Excel."""
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_wind(config: dict, description: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create one of the documented, repeatable wind records."""
    fs, duration = config["sample_rate_hz"], config["duration_s"]
    count = int(round(fs * duration))
    time = np.arange(count, dtype=float) / fs
    kind = description["kind"]
    if kind == "calm":
        speed = np.zeros(count)
        info = {"mean_m_s": 0.0, "realized_ti": 0.0, "minimum_m_s": 0.0,
                "maximum_m_s": 0.0, "seed": description["seed"]}
    elif kind == "steady":
        speed = np.full(count, description["mean_wind_speed_m_s"])
        info = {"mean_m_s": float(np.mean(speed)), "realized_ti": 0.0,
                "minimum_m_s": float(np.min(speed)), "maximum_m_s": float(np.max(speed)),
                "seed": description["seed"]}
    elif kind == "kaimal":
        time, speed, info = synthesize_kaimal_wind(
            fs, duration, description["mean_wind_speed_m_s"],
            description["target_turbulence_intensity"],
            config["kaimal_integral_scale_m"], description["seed"],
            description["maximum_wind_speed_m_s"],
        )
        info["mean_m_s"] = float(np.mean(speed))
        info["seed"] = description["seed"]
    elif kind == "gust":
        # Two smooth Gaussian gusts exercise both rising and falling response.
        speed = np.full(count, description["base_wind_speed_m_s"], dtype=float)
        for center_s, width_s in ((35.0, 5.0), (78.0, 7.0)):
            pulse = np.exp(-0.5 * ((time - center_s) / width_s) ** 2)
            speed += (description["gust_peak_m_s"] - description["base_wind_speed_m_s"]) * pulse
        speed = np.minimum(speed, description["gust_peak_m_s"])
        info = {"mean_m_s": float(np.mean(speed)), "realized_ti": float(np.std(speed) / np.mean(speed)),
                "minimum_m_s": float(np.min(speed)), "maximum_m_s": float(np.max(speed)),
                "seed": description["seed"]}
    else:
        raise ValueError(f"Unknown wind kind: {kind}")
    return time, speed, info


def applied_force(config: dict, speed: np.ndarray) -> np.ndarray:
    """Convert the known wind speed into the force applied to the plant."""
    return drag_force_from_speed(
        speed, config["air_density_kg_m3"], config["drag_coefficient"],
        config["projected_area_m2"],
    )


def metrics(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Calculate error measures on the stated evaluation interval."""
    error = np.asarray(estimate)[mask] - np.asarray(truth)[mask]
    if not np.all(np.isfinite(error)):
        return {"rmse_m_s": None, "bias_m_s": None, "mae_m_s": None,
                "p95_abs_error_m_s": None, "max_abs_error_m_s": None}
    absolute = np.abs(error)
    return {
        "rmse_m_s": float(np.sqrt(np.mean(error**2))),
        "bias_m_s": float(np.mean(error)),
        "mae_m_s": float(np.mean(absolute)),
        "p95_abs_error_m_s": float(np.quantile(absolute, 0.95)),
        "max_abs_error_m_s": float(np.max(absolute)),
    }


def peak_delay_s(time: np.ndarray, truth: np.ndarray, estimate: np.ndarray,
                 mask: np.ndarray) -> float | None:
    """Return estimated-minus-true peak time; zero-wind inputs have no peak."""
    indices = np.flatnonzero(mask)
    if not len(indices) or np.ptp(truth[mask]) < 1e-8:
        return None
    if not np.all(np.isfinite(estimate[mask])):
        return None
    truth_peak = indices[int(np.argmax(truth[mask]))]
    estimate_peak = indices[int(np.argmax(estimate[mask]))]
    return float(time[estimate_peak] - time[truth_peak])


def static_force(angle: np.ndarray, coefficients: dict, lever_m: float) -> np.ndarray:
    """Use the static restoring-torque balance as a simple baseline."""
    return coefficients["restoring_n_m_per_rad"] / lever_m * np.tan(angle)


def estimate_methods(angle: np.ndarray, coefficients: dict, config: dict,
                     tuning: dict, dt: float) -> dict[str, np.ndarray]:
    """Run all comparison methods with the currently selected settings."""
    static = static_force(angle, coefficients, config["force_lever_m"])
    def safe_force_estimate(function) -> np.ndarray:
        try:
            estimate = function()
            if np.all(np.isfinite(estimate)):
                return estimate
        except (ValueError, OverflowError, FloatingPointError):
            pass
        return np.full_like(angle, np.nan, dtype=float)

    rts3 = safe_force_estimate(lambda: nonlinear_ekf_rts_force(
        angle, coefficients, dt, config["force_lever_m"],
        np.deg2rad(config["assumed_angle_noise_deg"]), tuning["rts3_q"], 0,
        config["friction_epsilon_deg_s"],
    )[1])
    rts4 = safe_force_estimate(lambda: nonlinear_ekf_rts_force(
        angle, coefficients, dt, config["force_lever_m"],
        np.deg2rad(config["assumed_angle_noise_deg"]), tuning["rts4_q"], 1,
        config["friction_epsilon_deg_s"],
    )[1])
    eso3 = safe_force_estimate(lambda: nonlinear_luenberger_force(
        angle, coefficients, dt, config["force_lever_m"],
        tuning["eso3_hz"], 0, config["friction_epsilon_deg_s"],
    ))
    eso4 = safe_force_estimate(lambda: nonlinear_luenberger_force(
        angle, coefficients, dt, config["force_lever_m"],
        tuning["eso4_hz"], 1, config["friction_epsilon_deg_s"],
    ))
    result = {
        "静的換算": static,
        "因果LPF": causal_lowpass(static, tuning["lpf_hz"], dt),
        "ESO 3状態": eso3,
        "ESO 4状態": eso4,
        "RTS 3状態（オフライン）": rts3,
        "RTS 4状態（オフライン）": rts4,
    }
    # Preserve an unstable observer as NaNs so it is reported as a failure,
    # rather than aborting the evaluation of other estimators.
    return result


def force_to_speed(config: dict, force: np.ndarray) -> np.ndarray:
    return wind_speed_from_force(
        force, config["air_density_kg_m3"], config["drag_coefficient"],
        config["projected_area_m2"],
    )


def tune_axis(angle: np.ndarray, true_speed: np.ndarray, true_force: np.ndarray,
              coefficients: dict, config: dict, mask: np.ndarray,
              dt: float) -> tuple[dict, list[dict]]:
    """Select each tunable setting by wind-speed RMSE on training data only."""
    rows: list[dict] = []
    q_noise = np.deg2rad(config["assumed_angle_noise_deg"])
    tuning_grid = (
        ("ESO 3状態", "極周波数", "eso_pole_grid_hz", "eso3_hz", 0),
        ("ESO 4状態", "極周波数", "eso_pole_grid_hz", "eso4_hz", 1),
        ("RTS 3状態（オフライン）", "風力プロセス雑音", "rts3_force_random_walk_grid_n_per_sample", "rts3_q", 0),
        ("RTS 4状態（オフライン）", "力変化率プロセス雑音", "rts4_force_rate_random_walk_grid_n_per_s_per_sample", "rts4_q", 1),
        ("因果LPF", "遮断周波数", "lpf_cutoff_grid_hz", "lpf_hz", -1),
    )
    static_estimate = force_to_speed(config, static_force(angle, coefficients, config["force_lever_m"]))
    rows.append({"method": "静的換算", "parameter_name": "なし", "parameter": 0.0,
                 "stable": True, **metrics(true_speed, static_estimate, mask)})

    for method, parameter_name, grid_key, setting_key, order in tuning_grid:
        for value in config[grid_key]:
            try:
                if order == -1:
                    estimated_force = causal_lowpass(
                        static_force(angle, coefficients, config["force_lever_m"]), value, dt
                    )
                elif method.startswith("ESO"):
                    estimated_force = nonlinear_luenberger_force(
                        angle, coefficients, dt, config["force_lever_m"], value,
                        order, config["friction_epsilon_deg_s"],
                    )
                else:
                    _, estimated_force = nonlinear_ekf_rts_force(
                        angle, coefficients, dt, config["force_lever_m"], q_noise,
                        value, order, config["friction_epsilon_deg_s"],
                    )
                estimate = force_to_speed(config, estimated_force)
            except (ValueError, OverflowError, FloatingPointError):
                # Treat numerical blow-up as an unstable tuning candidate.
                estimate = np.full_like(true_speed, np.nan, dtype=float)
            finite = bool(np.all(np.isfinite(estimate)))
            stable = bool(finite and np.max(np.abs(estimate)) <= 20.0)
            # Large finite runaway estimates are kept in the scan as failures,
            # but cannot win parameter selection.
            row_metrics = metrics(true_speed, estimate, mask) if finite else {
                "rmse_m_s": None, "bias_m_s": None,
                "mae_m_s": None, "p95_abs_error_m_s": None,
                "max_abs_error_m_s": None,
            }
            rows.append({"method": method, "parameter_name": parameter_name,
                         "parameter": float(value), "stable": stable, **row_metrics})
    best = {"静的換算": {"parameter": None}}
    for method, _, _, setting_key, _ in tuning_grid:
        candidates = [row for row in rows if row["method"] == method]
        stable_candidates = [row for row in candidates if row["stable"]]
        pool = stable_candidates or [row for row in candidates if row["rmse_m_s"] is not None]
        if not pool:
            # The sweep had only numerical blow-ups. Keep a documented
            # diagnostic starting point to show the failure in validation.
            pool = candidates
            selected = pool[0]
        else:
            selected = min(pool, key=lambda row: row["rmse_m_s"] if row["rmse_m_s"] is not None else float("inf"))
        best[method] = {"parameter": selected["parameter"], "parameter_name": selected["parameter_name"],
                        "setting_key": setting_key, "training_rmse_m_s": selected["rmse_m_s"],
                        "stable_candidate_found": bool(stable_candidates)}
    return best, rows


def best_parameter_values(best: dict) -> dict[str, float]:
    return {
        "eso3_hz": best["ESO 3状態"]["parameter"],
        "eso4_hz": best["ESO 4状態"]["parameter"],
        "rts3_q": best["RTS 3状態（オフライン）"]["parameter"],
        "rts4_q": best["RTS 4状態（オフライン）"]["parameter"],
        "lpf_hz": best["因果LPF"]["parameter"],
    }


def save_plots(output: Path, config: dict, train_time: np.ndarray,
               train_speed: np.ndarray, scan_rows: dict[str, list[dict]],
               tuning_by_axis: dict, evaluation: dict,
               metrics_rows: list[dict]) -> None:
    """Create tuning curves plus overview and zoomed validation figures.

    An axis/observer pair that failed the tuning boundedness screen is not
    plotted as if it were a usable estimate. Its failure is called out in the
    panel and the report instead.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, method in zip(axes.flat, (
        "ESO 3状態", "ESO 4状態", "RTS 3状態（オフライン）",
        "RTS 4状態（オフライン）", "因果LPF", "静的換算",
    )):
        for axis_name, line_style in (("IN", "-"), ("OUT", "--")):
            rows = sorted((r for r in scan_rows[axis_name]
                           if r["method"] == method and r["parameter"] > 0 and r["stable"]),
                          key=lambda row: row["parameter"])
            if rows:
                ax.plot([r["parameter"] for r in rows], [r["rmse_m_s"] for r in rows],
                        marker="o", linestyle=line_style, label=f"{axis_name} axis")
        if method != "静的換算":
            ax.set_xscale("log")
        else:
            ax.text(0.5, 0.5, "No tuning parameter", ha="center", va="center", transform=ax.transAxes)
        ax.set(title=LABELS[method], xlabel="Tuning parameter (log scale)", ylabel="Training wind RMSE [m/s]")
        ax.grid(alpha=0.25)
        if method == "ESO 4状態" and not tuning_by_axis["OUT"][method].get("stable_candidate_found", True):
            ax.text(0.98, 0.96, "OUT: usable tuning candidate none\nexcluded from later evaluation",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color="#9b2c2c", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        if method != "静的換算":
            ax.legend(fontsize=8)
    fig.suptitle("OW-03 tuning scan (training wind only)")
    fig.savefig(output / "ow03_tuning_scan.png", dpi=160)
    plt.close(fig)

    case_names = list(evaluation)

    def plot_waveform_figure(filename: str, time_windows: dict[str, tuple[float, float]], title: str) -> None:
        """Draw each case over its own interval measured from evaluation start."""
        fig, axes = plt.subplots(len(case_names), 2, figsize=(15, 3.0 * len(case_names)),
                                 sharex=False, layout="constrained")
        for row_idx, case_name in enumerate(case_names):
            case = evaluation[case_name]
            window_start, window_end = time_windows[case_name]
            relative_time = case["time"] - config["evaluation_start_s"]
            selected = (case["mask"] & (relative_time >= window_start)
                        & (relative_time <= window_end))
            for col, axis_name in enumerate(("IN", "OUT")):
                ax = axes[row_idx, col]
                ax.plot(relative_time[selected], case["speed"][selected], color="black",
                        linewidth=1.6, label="True wind")
                for method, estimate_speed in case["speed_estimates"][axis_name].items():
                    # OUT 4-state ESO has no tuning candidate that passed the
                    # boundedness screen. Do not let its diagnostic runaway
                    # dominate the useful comparisons in any waveform plot.
                    if (method == "ESO 4状態" and axis_name == "OUT"
                            and not tuning_by_axis[axis_name][method].get("stable_candidate_found", True)):
                        continue
                    ax.plot(relative_time[selected], estimate_speed[selected], color=COLORS[method],
                            linewidth=0.85, alpha=0.9, label=LABELS[method])
                if (axis_name == "OUT"
                        and not tuning_by_axis[axis_name]["ESO 4状態"].get("stable_candidate_found", True)):
                    ax.text(0.99, 0.96, "4-state ESO: no bounded tuning candidate; omitted",
                            transform=ax.transAxes, ha="right", va="top", fontsize=8,
                            color="#9b2c2c", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
                ax.set_ylabel("Wind speed [m/s]")
                ax.set_title(f"{CASE_LABELS[case_name]} / {axis_name} axis ({window_start:g}–{window_end:g} s)")
                ax.grid(alpha=0.2)
                if row_idx == 0 and col == 0:
                    ax.legend(ncol=3, fontsize=7)
                if row_idx == len(case_names) - 1:
                    ax.set_xlabel("Time from evaluation-window start [s]")
        fig.suptitle(title)
        fig.savefig(output / filename, dpi=110)
        plt.close(fig)

    # Overview: show the first 45 s after the initial transient for every case.
    overview_windows = {name: (0.0, 45.0) for name in case_names}
    plot_waveform_figure(
        "ow03_validation_waveforms.png", overview_windows,
        "OW-03 independent wind validation (overview; first 45 s of evaluation interval)",
    )
    # Zoom: show the first 5 s of each case. For the gust case, center the view
    # on its first pulse (plant time 35 s; evaluation-relative time 20 s).
    zoom_windows = {name: (0.0, 5.0) for name in case_names}
    gust_case = next((name for name in case_names if "ガスト" in name), None)
    if gust_case:
        zoom_windows[gust_case] = (16.0, 24.0)
    plot_waveform_figure(
        "ow03_validation_waveforms_zoom.png", zoom_windows,
        "OW-03 independent wind validation (time-axis zoom; gust shown around first pulse)",
    )

    # A concise, axis-by-case RMSE comparison complements the waveform panels.
    cases = list(dict.fromkeys(row["case"] for row in metrics_rows))
    methods = [m for m in METHODS if m != "静的換算"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, layout="constrained")
    x = np.arange(len(cases))
    width = 0.12
    for ax, axis_name in zip(axes, ("IN", "OUT")):
        for index, method in enumerate(methods):
            if (axis_name == "OUT" and method == "ESO 4状態"
                    and not tuning_by_axis[axis_name][method].get("stable_candidate_found", True)):
                continue
            values = []
            for case in cases:
                found = next(r for r in metrics_rows if r["axis"] == axis_name and r["case"] == case and r["method"] == method)
                values.append(np.nan if found["rmse_m_s"] is None else found["rmse_m_s"])
            ax.bar(x + (index - len(methods) / 2) * width, values, width, color=COLORS[method], label=LABELS[method])
        ax.set_xticks(x, [CASE_LABELS[case] for case in cases], rotation=25, ha="right")
        ax.set_ylabel("Wind-speed RMSE [m/s]")
        ax.set_title(f"{axis_name} axis")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    if not tuning_by_axis["OUT"]["ESO 4状態"].get("stable_candidate_found", True):
        axes[1].text(0.99, 0.98, "OUT 4-state ESO omitted:\nno bounded tuning candidate",
                     transform=axes[1].transAxes, ha="right", va="top", fontsize=8,
                     color="#9b2c2c", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    fig.suptitle("Independent validation errors by wind case")
    fig.savefig(output / "ow03_validation_rmse.png", dpi=160)
    plt.close(fig)


def write_report(path: Path, config: dict, training_info: dict,
                 tuning_by_axis: dict, scan_rows: dict,
                 validation_info: dict, metrics_rows: list[dict],
                 max_angle: dict, gains: dict) -> None:
    """Write a Japanese report, including enough detail to reproduce tuning."""
    lines = [
        "# OW-03 新ハード係数でのオブザーバー調整と独立評価", "",
        "## 目的と結論", "",
        "OW-02で旧調整値のまま比較したESOの極周波数等を、新しいHBK係数に合わせて再調整した。調整に使う風系列と、性能を確かめる風系列は独立させた。プラントと推定モデルの機械係数は一致させ、角度センサーは理想値とした。", "",
        "調整では推定風速RMSEが小さくなるパラメーターを各軸・各方式ごとに選んだ。評価用風系列で各方式の性能を比較し、ESO極周波数を旧設定から変更した場合の発散・追従性を確認した。RTSは将来データを用いるオフライン方式のため、オンラインESOとは分けて解釈する。", "",
  