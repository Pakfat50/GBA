"""Validate OW-05 observer output with measured BALL free-decay records.

The measured angle trace is used directly. A zero-external-wind free-decay
trajectory is integrated from the observed initial angle/rate; the angle
residual to that trajectory captures sensor variation and plant mismatch.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from hbk_model_coefficients import select_coefficients
from hbk_nonlinear_estimators import (
    _rk4_step,
    nonlinear_ekf_rts_force,
    nonlinear_luenberger_force,
)
from run_ow03_hbk_observer_tuning import force_to_speed


METHODS = ("ESO 3状態", "RTS 3状態（オフライン）")


def _free_decay_reference(angle0: float, rate0: float, n: int, dt: float,
                          coefficients: dict, lever_m: float,
                          epsilon_deg_s: float) -> np.ndarray:
    """Integrate the adopted plant at zero applied force from observed ICs."""
    state = np.array([angle0, rate0, 0.0], dtype=float)
    epsilon = np.deg2rad(epsilon_deg_s)
    y = np.empty(n, dtype=float)
    for i in range(n):
        y[i] = state[0]
        if i + 1 < n:
            state = _rk4_step(state, dt, coefficients, lever_m, epsilon)
    return y


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def evaluate_free_decay(cfg: dict, tunings: dict, out: Path, repo: Path) -> dict:
    """Run zero-wind validation on approved, measured BALL free-decay traces."""
    out.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / float(cfg["sample_rate_hz"])
    data_root = repo / "04_Data/05_Fitting/20260921"
    selection_path = repo / "06_Analysis/fitting_pipeline/results/20260921/waveform_review/waveform_selection.csv"
    selection = pd.read_csv(selection_path, encoding="utf-8-sig")
    selection = selection[
        (selection["use_for_fitting"] == 1)
        & (selection["review_status"].astype(str).str.upper() == "APPROVED")
        & (selection["configuration"].astype(str).str.upper() == "BALL")
    ]
    tables: dict[str, pd.DataFrame] = {}
    noise_reference = pd.read_csv(out / "ow05_noise_amplification.csv")
    noise_reference = noise_reference[noise_reference["sensor_case"] == "static_window_sigma"]

    records = []
    series = {}
    margin_start_s, margin_end_s = 3.0, 0.5
    rate_hz = float(cfg["sample_rate_hz"])
    angle_noise_rad = np.deg2rad(float(cfg["assumed_angle_noise_deg"]))
    for _, row in selection.iterrows():
        relative = str(row["data_file"])
        if relative not in tables:
            tables[relative] = pd.read_csv(data_root / relative)
        table = tables[relative]
        angle_col = "angle0[deg]" if str(row["axis"]) == "IN" else "angle1[deg]"
        y_deg = pd.to_numeric(table[angle_col], errors="coerce").to_numpy(float)
        y_deg = (y_deg + 180.0) % 360.0 - 180.0
        start, end = int(row["start_index"]), int(row["end_index"])
        angle_deg = y_deg[start:end]
        if len(angle_deg) < int((margin_start_s + margin_end_s + 1.0) * rate_hz):
            continue
        angle = np.deg2rad(angle_deg)
        n = len(angle)
        t = np.arange(n, dtype=float) * dt
        initial_count = min(n, max(5, int(round(0.20 * rate_hz))))
        initial_rate = float(np.polyfit(t[:initial_count], angle[:initial_count], 1)[0])
        initial_state = np.array([angle[0], initial_rate, 0.0], dtype=float)
        axis = str(row["axis"])
        coefficients = select_coefficients(axis, "BALL")
        reference = _free_decay_reference(
            angle[0], initial_rate, n, dt, coefficients,
            cfg["force_lever_m"], cfg["friction_epsilon_deg_s"],
        )
        score = (t >= margin_start_s) & (t <= t[-1] - margin_end_s)
        residual_rad = angle - reference
        residual_deg = np.rad2deg(residual_rad)

        estimated = {}
        clean = {}
        for method in METHODS:
            parameter = tunings[(axis, method)]
            if method.startswith("ESO"):
                estimated_force = nonlinear_luenberger_force(
                    angle, coefficients, dt, cfg["force_lever_m"], parameter, 0,
                    cfg["friction_epsilon_deg_s"], initial_state=initial_state,
                )
                clean_force = nonlinear_luenberger_force(
                    reference, coefficients, dt, cfg["force_lever_m"], parameter, 0,
                    cfg["friction_epsilon_deg_s"], initial_state=initial_state,
                )
            else:
                _, estimated_force = nonlinear_ekf_rts_force(
                    angle, coefficients, dt, cfg["force_lever_m"], angle_noise_rad,
                    parameter, 0, cfg["friction_epsilon_deg_s"], initial_state=initial_state,
                )
                _, clean_force = nonlinear_ekf_rts_force(
                    reference, coefficients, dt, cfg["force_lever_m"], angle_noise_rad,
                    parameter, 0, cfg["friction_epsilon_deg_s"], initial_state=initial_state,
                )
            estimated[method] = force_to_speed(cfg, estimated_force)
            clean[method] = force_to_speed(cfg, clean_force)

            wind_zero_rmse = _rms(estimated[method][score])
            response = estimated[method] - clean[method]
            response_rmse = _rms(response[score])
            angle_resid_rmse = _rms(residual_deg[score])
            centered_residual = residual_deg[score] - float(np.mean(residual_deg[score]))
            residual_ss = float(np.dot(centered_residual, centered_residual))
            residual_acf_lag1 = float(np.dot(centered_residual[:-1], centered_residual[1:]) / residual_ss) if residual_ss > 0 else 0.0
            records.append({
                "segment_id": str(row["segment_id"]),
                "axis": axis,
                "direction": str(row["direction"]),
                "repetition": int(row["repetition"]),
                "start_index": start,
                "end_index_exclusive": end,
                "samples_scored": int(score.sum()),
                "duration_s": float(t[-1]),
                "method": method,
                "wind_zero_rmse_m_s": wind_zero_rmse,
                "wind_zero_bias_m_s": float(np.mean(estimated[method][score])),
                "wind_zero_mae_m_s": float(np.mean(np.abs(estimated[method][score]))),
                "wind_zero_p95_abs_m_s": float(np.quantile(np.abs(estimated[method][score]), 0.95)),
                "wind_zero_max_abs_m_s": float(np.max(np.abs(estimated[method][score]))),
                "clean_reference_wind_rmse_m_s": _rms(clean[method][score]),
                "measured_angle_residual_rmse_deg": angle_resid_rmse,
                "measured_angle_residual_acf_lag1": residual_acf_lag1,
                "measured_angle_residual_ss_deg2": residual_ss,
                "measured_response_rmse_m_s": response_rmse,
                "measured_amplification_m_s_per_deg": response_rmse / angle_resid_rmse if angle_resid_rmse > 0 else None,
            })
        series[str(row["segment_id"])] = {
            "axis": axis, "time": t, "angle_deg": angle_deg, "reference_deg": np.rad2deg(reference),
            "score": score, "estimated": estimated, "clean": clean, "residual_deg": residual_deg,
        }

    if not records:
        raise RuntimeError("No approved BALL free-decay records were evaluated.")
    detail = pd.DataFrame(records)
    detail.to_csv(out / "ow05_free_decay_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.10g")

    summary = []
    for axis in ("IN", "OUT"):
        for method in METHODS:
            group = detail[(detail["axis"] == axis) & (detail["method"] == method)]
            ref_group = noise_reference[(noise_reference["axis"] == axis) & (noise_reference["method"] == method)]
            nscore = group["samples_scored"].to_numpy(float)
            angle_rms = float(np.sqrt(np.average(group["measured_angle_residual_rmse_deg"] ** 2, weights=nscore)))
            response_rms = float(np.sqrt(np.average(group["measured_response_rmse_m_s"] ** 2, weights=nscore)))
            zero_rms = float(np.sqrt(np.average(group["wind_zero_rmse_m_s"] ** 2, weights=nscore)))
            zero_bias = float(np.average(group["wind_zero_bias_m_s"], weights=nscore))
            residual_ss = float(group["measured_angle_residual_ss_deg2"].sum())
            residual_acf = float(np.sum(group["measured_angle_residual_acf_lag1"] * group["measured_angle_residual_ss_deg2"]) / residual_ss) if residual_ss > 0 else 0.0
            synthetic_angle = float(np.sqrt(np.mean(ref_group["angle_noise_rmse_deg"].to_numpy(float) ** 2)))
            synthetic_response = float(np.sqrt(np.mean(ref_group["observer_noise_rmse_m_s"].to_numpy(float) ** 2)))
            measured_gain = response_rms / angle_rms if angle_rms > 0 else float("nan")
            white_gain = synthetic_response / synthetic_angle if synthetic_angle > 0 else float("nan")
            summary.append({
                "axis": axis, "method": method, "segments": int(len(group)),
                "scored_samples": int(group["samples_scored"].sum()),
                "wind_zero_rmse_m_s": zero_rms,
                "wind_zero_bias_m_s": zero_bias,
                "measured_angle_residual_rmse_deg": angle_rms,
                "measured_angle_residual_acf_lag1": residual_acf,
                "measured_response_rmse_m_s": response_rms,
                "measured_amplification_m_s_per_deg": measured_gain,
                "white_sigma_observer_response_rmse_m_s": synthetic_response,
                "white_sigma_amplification_m_s_per_deg": white_gain,
                "measured_to_white_gain_ratio": measured_gain / white_gain if white_gain > 0 else None,
                "zero_wind_to_white_response_ratio": zero_rms / synthetic_response if synthetic_response > 0 else None,
            })
    pd.DataFrame(summary).to_csv(out / "ow05_free_decay_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout="constrained")
    representative = {}
    for col, axis in enumerate(("IN", "OUT")):
        axis_group = detail[detail["axis"] == axis]
        eso = axis_group[axis_group["method"] == METHODS[0]].sort_values("wind_zero_rmse_m_s")
        segment_id = str(eso.iloc[len(eso) // 2]["segment_id"])
        representative[axis] = segment_id
        item = series[segment_id]
        ax = axes[0, col]
        ax.plot(item["time"], item["angle_deg"], color="#222222", lw=1.0, label="Measured angle")
        ax.plot(item["time"], item["reference_deg"], color="#4c9f70", lw=1.0, ls="--", label="Zero-wind model")
        ax.axvspan(0, margin_start_s, color="#999999", alpha=.15)
        ax.axvspan(item["time"][-1] - margin_end_s, item["time"][-1], color="#999999", alpha=.15)
        ax.set_title(f"{axis}: {segment_id} | measured free decay")
        ax.set_ylabel("Angle [deg]")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        ax = axes[1, col]
        for method, color in zip(METHODS, ("#2673a8", "#dd8452")):
            ax.plot(item["time"], item["estimated"][method], color=color, lw=1.0,
                    label="ESO measured" if method.startswith("ESO") else "RTS measured")
        ax.axhline(0, color="#222222", ls="--", lw=.9, label="True wind: 0")
        ax.axvspan(0, margin_start_s, color="#999999", alpha=.15)
        ax.axvspan(item["time"][-1] - margin_end_s, item["time"][-1], color="#999999", alpha=.15)
        ax.set_xlabel("Time from selected waveform start [s]")
        ax.set_ylabel("Inferred wind speed [m/s]")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("OW-05 observer on measured free-decay angles | zero external wind", fontsize=14)
    waveform_png = "ow05_free_decay_zero_wind.png"
    fig.savefig(out / waveform_png, dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    colors = {METHODS[0]: "#2673a8", METHODS[1]: "#dd8452"}
    for ax, axis in zip(axes, ("IN", "OUT")):
        for xpos, method in enumerate(METHODS):
            group = detail[(detail["axis"] == axis) & (detail["method"] == method)]
            x = np.full(len(group), xpos, dtype=float)
            ax.scatter(x, group["measured_amplification_m_s_per_deg"], color=colors[method],
                       alpha=.65, s=28, label="Measured segments" if xpos == 0 else None)
            srow = next(r for r in summary if r["axis"] == axis and r["method"] == method)
            ax.scatter([xpos], [srow["white_sigma_amplification_m_s_per_deg"]],
                       marker="D", s=60, facecolor="none", edgecolor="#b34b35",
                       linewidth=1.5, label="Static-window σ white model" if xpos == 0 else None)
        ax.set_xticks(range(len(METHODS)), ["ESO 3-state", "RTS 3-state\n(offline)"])
        ax.set_title(axis)
        ax.set_ylabel("Observer response / angle residual [m/s per deg]")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Measured free-decay response compared with the white-noise model")
    gain_png = "ow05_free_decay_gain_comparison.png"
    fig.savefig(out / gain_png, dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    xbase = np.arange(len(METHODS), dtype=float)
    width = 0.36
    for ax, axis in zip(axes, ("IN", "OUT")):
        actual = [next(r["wind_zero_rmse_m_s"] for r in summary if r["axis"] == axis and r["method"] == method) for method in METHODS]
        simulated = [next(r["white_sigma_observer_response_rmse_m_s"] for r in summary if r["axis"] == axis and r["method"] == method) for method in METHODS]
        ax.bar(xbase - width / 2, actual, width, color="#2673a8", label="Measured free decay, true wind 0")
        ax.bar(xbase + width / 2, simulated, width, color="#dd8452", label="White σ model, noise-only response")
        ax.set_xticks(xbase, ["ESO 3-state", "RTS 3-state (offline)"])
        ax.set_title(axis)
        ax.set_ylabel("Wind-speed RMSE [m/s]")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Zero-wind measured estimate vs simulated noise-only response")
    response_png = "ow05_free_decay_zero_vs_white.png"
    fig.savefig(out / response_png, dpi=170)
    plt.close(fig)

    return {
        "segments": int(detail["segment_id"].nunique()),
        "source": "04_Data/05_Fitting/20260921/Raw/球/LOG00012_ANGLE.csv",
        "configuration": "BALL",
        "zero_wind_assumption": True,
        "initialization": "Measured first angle and 0.2 s linear-fit angular rate; external force initialized to zero.",
        "score_margins_s": {"start": margin_start_s, "end": margin_end_s},
        "note": "Measured angle residual to a zero-force free-decay simulation includes sensor variation and plant/model mismatch; it is not pure sensor noise.",
        "representative_segments": representative,
        "summary": summary,
        "figures": [waveform_png, gain_png, response_png],
    }
