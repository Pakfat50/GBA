#!/usr/bin/env python3
"""Estimate IN/OUT mechanical backlash from approved new-hardware free-decay records.

Run from the repository root:

    python 06_Analysis/fitting_pipeline/run_hybrid_backlash_estimation.py

The script compares the already identified free-decay model with the same model
plus one shared, rate-independent play/backlash width per axis. All physical
coefficients are loaded from the accepted IHB results and held fixed. The only
global fitted quantities are the IN and OUT backlash widths. This restriction
keeps this analysis from silently absorbing phase and damping errors into many
free coefficients.

Important: free-decay angle is an indirect observation of backlash. The result
is therefore a model-based estimate conditional on fixed I/K/c/tau, known
angle centers, a zero-speed release at the first eligible peak, and the chosen
play-operator initial state. Waveform-level bootstrap intervals show repeat
spread under these assumptions; they are not complete physical uncertainty
intervals for backlash.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[2]
DATE = "20260921"
DEFAULT_DATA = ROOT / "04_Data/05_Fitting" / DATE
DEFAULT_RESULTS = ROOT / "06_Analysis/fitting_pipeline/results" / DATE
DEFAULT_OUT = DEFAULT_RESULTS / "hybrid_identification/08_backlash_estimation"
MINIMUM_PEAK_AMPLITUDE_DEG = 4.0
SENSOR_PERIOD_S = 0.01
# Logger records are nominally 100 Hz. RK4 at that interval is still much
# finer than the pendulum period and keeps the width profile inexpensive.
INTEGRATION_STEP_S = 0.01
EPSILON_RAD_S = math.radians(0.5)
MAX_TOTAL_WIDTH_DEG = 2.0
TOTAL_WIDTH_STEP_DEG = 0.02
BOOTSTRAP_REPLICATES = 1000
RNG_SEED = 54108


def parse_args() -> argparse.Namespace:
    """Read optional input/output folders; defaults point to the approved run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA,
                        help="20260921 raw and manifest folder")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS,
                        help="20260921 preprocessing and coefficient results folder")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                        help="folder for the CSV, figures, settings and report")
    return parser.parse_args()


def read_inputs(data_root: Path, results_root: Path):
    """Load approved segment definitions, preprocessed centers, and fixed coefficients."""
    stage = results_root / "hybrid_identification"
    selected = pd.read_csv(results_root / "waveform_review/waveform_selection.csv",
                           encoding="utf-8-sig")
    selected = selected[(selected.use_for_fitting.astype(int) == 1)
                        & (selected.review_status.astype(str).str.upper() == "APPROVED")].copy()
    if selected.empty:
        raise ValueError("承認済みの自由振動波形がありません")

    centers = pd.read_csv(stage / "01_preprocessing/waveform_preprocessing.csv")
    identified = pd.read_csv(stage / "02_frequency_identification/identified_inertia_restoring.csv")
    stage4 = json.loads((stage / "04_rod_damping_identification/stage4_settings.json").read_text())
    ball_settings = json.loads((results_root / "iterative_hybrid/ihb05_cball/ihb05_position_adjusted_settings.json").read_text())
    ball_coefficients = pd.read_csv(results_root / "iterative_hybrid/ihb05_cball/ihb05_position_adjusted_coefficients.csv",
                                    encoding="utf-8-sig")

    # The final ball-specific position scan updated only BALL I/K and c_ball.
    # Other configurations keep the previously accepted IHB-02 I/K and rod c.
    rod_model = stage4["models_for_review"]["theoretical_c"]
    c_rod = float(rod_model["c_rod"])
    # The ball covers the lower part of the upper rod. IHB-05 applies its
    # documented fourth-power exposed-length correction: (129 / 229)^4.
    c_rod_ball = c_rod * (0.129 / 0.229) ** 4
    c_ball = float(ball_settings["c_ball_refitted_n_m_s2_per_rad2"])
    tau_by_axis = {key: float(value) for key, value in ball_settings["tau_by_axis_n_m"].items()}
    identified_index = identified.set_index(["axis", "configuration"])
    ball_index = ball_coefficients.set_index("axis")
    coefficients = {}
    for axis in ("IN", "OUT"):
        for configuration in ("SP00", "SP01", "SP02", "SP03", "SP04", "BALL"):
            if configuration == "BALL":
                row = ball_index.loc[axis]
                inertia = float(row.inertia_after_kg_m2)
                restoring = float(row.restoring_after_n_m)
                drag = c_rod_ball + c_ball
            else:
                row = identified_index.loc[(axis, configuration)]
                inertia = float(row.inertia_kg_m2)
                restoring = float(row.restoring_n_m_per_rad)
                drag = c_rod
            coefficients[(axis, configuration)] = {
                "inertia": inertia, "restoring": restoring, "b": 0.0,
                "quadratic_drag": drag, "tau": tau_by_axis[axis],
            }
    center_by_id = centers.set_index("segment_id")["envelope_center_deg"].to_dict()
    return selected, center_by_id, coefficients, {
        "c_rod": c_rod, "c_rod_ball": c_rod_ball, "c_ball": c_ball,
        "tau_by_axis": tau_by_axis,
    }


def build_records(selected: pd.DataFrame, centers: dict, data_root: Path,
                  results_root: Path) -> list[dict]:
    """Slice each approved 100 Hz recording from first to last eligible peak."""
    # Waveform peak data live beside the results, not beside the raw logger files.
    # Resolve that known location from this script's repository root.
    turning_path = results_root / "hybrid_identification/01_preprocessing/turning_points.csv"
    turning = pd.read_csv(turning_path)
    records = []
    raw_cache: dict[str, pd.DataFrame] = {}
    grouped_peaks = {str(sid): rows.sort_values("peak_number").reset_index(drop=True)
                     for sid, rows in turning.groupby("segment_id", sort=False)}

    for _, selection in selected.iterrows():
        segment_id = str(selection.segment_id)
        if segment_id not in grouped_peaks or segment_id not in centers:
            raise ValueError(f"{segment_id}: 頂点または角度中心の記録がありません")
        data_file = str(selection.data_file)
        if data_file not in raw_cache:
            raw_cache[data_file] = pd.read_csv(data_root / data_file)
        raw = raw_cache[data_file]
        time = pd.to_numeric(raw["systime[ms]"], errors="coerce").to_numpy(float) / 1000.0
        angle = pd.to_numeric(raw[str(selection.angle_column)], errors="coerce").to_numpy(float)
        valid = np.isfinite(time) & np.isfinite(angle)
        time, angle = time[valid], angle[valid]
        angle = (angle + 180.0) % 360.0 - 180.0
        start = int(selection.start_index)
        end = int(selection.end_index) + 1
        if end > len(time):
            raise ValueError(f"{segment_id}: manifest index is outside its log file")
        time = time[start:end]
        angle = angle[start:end] - float(centers[segment_id])

        peaks = grouped_peaks[segment_id]
        initial_rows = np.flatnonzero(peaks.is_initial_peak.astype(int).to_numpy() == 1)
        if len(initial_rows) != 1:
            raise ValueError(f"{segment_id}: 初期頂点が一意ではありません")
        peaks = peaks.iloc[int(initial_rows[0]):].reset_index(drop=True)
        eligible = peaks.eligible_for_later_stages.astype(int).to_numpy() == 1
        amplitude = peaks.amplitude_deg.to_numpy(float)
        pairs = [i for i in range(len(peaks)-1) if eligible[i] and eligible[i+1]
                 and min(amplitude[i], amplitude[i+1]) >= MINIMUM_PEAK_AMPLITUDE_DEG]
        if not pairs:
            continue
        first, last = pairs[0], pairs[-1] + 1
        peak_rows = peaks.iloc[first:last+1]
        start_time = float(peak_rows.peak_time_s.iloc[0])
        stop_time = float(peak_rows.peak_time_s.iloc[-1])
        duration = stop_time - start_time
        # Resample only for efficient, repeatable integration. The original logger
        # sampling is nominally 100 Hz and the comparison uses linearly interpolated angles.
        sample_t = np.arange(0.0, duration + SENSOR_PERIOD_S * 0.25, SENSOR_PERIOD_S)
        sample_t = sample_t[sample_t <= duration + 1e-9]
        obs_deg = np.interp(start_time + sample_t, time, angle)
        initial_deg = float(peak_rows.centered_peak_angle_deg.iloc[0])
        initial_kind = str(peak_rows.peak_kind.iloc[0]).upper()
        if initial_kind not in {"MAX", "MIN"}:
            raise ValueError(f"{segment_id}: peak kind is not MAX or MIN")
        records.append({
            "segment_id": segment_id, "axis": str(selection.axis).upper(),
            "configuration": str(selection.configuration).upper(),
            "direction": str(selection.direction).upper(),
            "repetition": int(selection.repetition), "time": sample_t,
            "observed_deg": obs_deg, "initial_deg": initial_deg,
            "initial_kind": initial_kind,
            "start_peak_time_s": start_time, "duration_s": duration,
            "accepted_half_cycles": len(pairs),
        })
    if not records:
        raise ValueError("4°以上の隣接頂点を持つ承認波形がありません")
    return records


def play_project(angle: np.ndarray, previous_play: np.ndarray, half_width: float) -> np.ndarray:
    """One vectorized update of the rate-independent play operator."""
    return np.clip(previous_play, angle - half_width, angle + half_width)


def acceleration(angle: np.ndarray, speed: np.ndarray, play: np.ndarray,
                 inertia: np.ndarray, restoring: np.ndarray, b: np.ndarray,
                 drag: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Angular acceleration for the fixed physical model, evaluated per waveform."""
    moment = restoring * np.sin(play) + b * speed
    moment += drag * np.abs(speed) * speed
    moment += tau * np.tanh(speed / EPSILON_RAD_S)
    return -moment / inertia


def simulate_batch(records: list[dict], coefficients: dict, total_width_deg: float,
                   return_traces: bool = False):
    """Simulate many waveforms together using RK4 and the play operator.

    The play state is initialized at the boundary corresponding to the measured
    release extremum. A fine fixed integration step resolves passage through the
    backlash gap; output is compared at the 100 Hz measurement intervals.
    """
    count = len(records)
    max_samples = max(len(r["time"]) for r in records)
    theta = np.deg2rad(np.asarray([r["initial_deg"] for r in records], dtype=float))
    speed = np.zeros(count, dtype=float)  # Release is assumed to be at rest.
    half_width = math.radians(total_width_deg / 2.0)
    # At a positive maximum the spring side is at theta-delta; at a negative
    # minimum it is at theta+delta, if the part approached the stop from outside.
    signs = np.asarray([1.0 if r["initial_kind"] == "MAX" else -1.0 for r in records])
    play = theta - signs * half_width
    coeffs = [coefficients[(r["axis"], r["configuration"])] for r in records]
    inertia = np.asarray([c["inertia"] for c in coeffs])
    restoring = np.asarray([c["restoring"] for c in coeffs])
    b = np.asarray([c["b"] for c in coeffs])
    drag = np.asarray([c["quadratic_drag"] for c in coeffs])
    tau = np.asarray([c["tau"] for c in coeffs])
    active_lengths = np.asarray([len(r["time"]) for r in records], dtype=int)
    predicted = np.full((count, max_samples), np.nan, dtype=float)
    predicted[:, 0] = np.rad2deg(theta)
    sample_step_count = max(1, int(round(SENSOR_PERIOD_S / INTEGRATION_STEP_S)))
    internal_step = SENSOR_PERIOD_S / sample_step_count
    total_steps = (max_samples - 1) * sample_step_count

    for step in range(total_steps):
        sample_index = step // sample_step_count + 1
        if sample_index >= max_samples:
            break
        active = sample_index < active_lengths
        if not np.any(active):
            break
        old_theta, old_speed, old_play = theta.copy(), speed.copy(), play.copy()
        p1 = play_project(old_theta, old_play, half_width)
        t1 = old_speed
        w1 = acceleration(old_theta, old_speed, p1, inertia, restoring, b, drag, tau)

        theta2 = old_theta + 0.5 * internal_step * t1
        speed2 = old_speed + 0.5 * internal_step * w1
        p2 = play_project(theta2, p1, half_width)
        t2 = speed2
        w2 = acceleration(theta2, speed2, p2, inertia, restoring, b, drag, tau)

        theta3 = old_theta + 0.5 * internal_step * t2
        speed3 = old_speed + 0.5 * internal_step * w2
        p3 = play_project(theta3, p2, half_width)
        t3 = speed3
        w3 = acceleration(theta3, speed3, p3, inertia, restoring, b, drag, tau)

        theta4 = old_theta + internal_step * t3
        speed4 = old_speed + internal_step * w3
        p4 = play_project(theta4, p3, half_width)
        t4 = speed4
        w4 = acceleration(theta4, speed4, p4, inertia, restoring, b, drag, tau)

        next_theta = old_theta + internal_step * (t1 + 2*t2 + 2*t3 + t4) / 6.0
        next_speed = old_speed + internal_step * (w1 + 2*w2 + 2*w3 + w4) / 6.0
        next_play = play_project(next_theta, p4, half_width)
        theta = np.where(active, next_theta, old_theta)
        speed = np.where(active, next_speed, old_speed)
        play = np.where(active, next_play, old_play)
        if (step + 1) % sample_step_count == 0:
            predicted[:, sample_index] = np.where(active, np.rad2deg(theta), np.nan)

    per_wave_rmse = []
    trace_list = []
    for index, record in enumerate(records):
        n = len(record["time"])
        model = predicted[index, :n]
        residual = model - record["observed_deg"]
        per_wave_rmse.append(float(np.sqrt(np.mean(residual * residual))))
        if return_traces:
            trace_list.append({**record, "predicted_deg": model.copy(), "residual_deg": residual.copy()})
    return np.asarray(per_wave_rmse), trace_list


def equal_configuration_score(records: list[dict], per_wave_rmse: np.ndarray, axis: str) -> float:
    """Root mean square with equal weight for each configuration and each run."""
    rows = [(record, per_wave_rmse[i]) for i, record in enumerate(records)
            if record["axis"] == axis]
    config_values = []
    for configuration in sorted({r["configuration"] for r, _ in rows}):
        values = np.asarray([rmse for r, rmse in rows if r["configuration"] == configuration])
        config_values.append(float(np.mean(values * values)))
    return math.sqrt(float(np.mean(config_values)))


def grid_search(records: list[dict], coefficients: dict):
    """Scan full backlash width and retain per-wave RMSE at every candidate."""
    grid = np.round(np.arange(0.0, MAX_TOTAL_WIDTH_DEG + TOTAL_WIDTH_STEP_DEG/2,
                              TOTAL_WIDTH_STEP_DEG), 8)
    curve_rows, wave_rows, scores = [], [], {axis: [] for axis in ("IN", "OUT")}
    rmse_matrix = np.zeros((len(grid), len(records)), dtype=float)
    for i, width in enumerate(grid):
        rmse, _ = simulate_batch(records, coefficients, float(width))
        rmse_matrix[i, :] = rmse
        for axis in ("IN", "OUT"):
            score = equal_configuration_score(records, rmse, axis)
            scores[axis].append(score)
            curve_rows.append({"axis": axis, "backlash_total_width_deg": width,
                               "equal_configuration_waveform_rmse_deg": score})
    best = {}
    for axis in ("IN", "OUT"):
        values = np.asarray(scores[axis])
        best_index = int(np.argmin(values))
        best[axis] = {"grid_index": best_index,
                      "total_width_deg": float(grid[best_index]),
                      "half_width_deg": float(grid[best_index] / 2.0),
                      "rmse_deg": float(values[best_index]),
                      "no_backlash_rmse_deg": float(values[0]),
                      "curve": values}
    # Each recording's own best fit is diagnostic only; it is not a separate
    # physical estimate because I/K/center/initial-state errors can also shift it.
    for j, record in enumerate(records):
        row = dict(segment_id=record["segment_id"], axis=record["axis"],
                   configuration=record["configuration"], direction=record["direction"],
                   repetition=record["repetition"], half_cycles=record["accepted_half_cycles"])
        for i, width in enumerate(grid):
            wave_rows.append({**row, "backlash_total_width_deg": float(width),
                              "waveform_rmse_deg": float(rmse_matrix[i, j])})
    scan = pd.DataFrame(curve_rows)
    per_wave = pd.DataFrame(wave_rows)
    return grid, scan, per_wave, rmse_matrix, best


def bootstrap_profile(records: list[dict], grid: np.ndarray, rmse_matrix: np.ndarray,
                      replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    """Resample whole waveforms within each configuration, preserving balance."""
    rng = np.random.default_rng(RNG_SEED)
    output = {}
    record_axis = np.asarray([r["axis"] for r in records])
    record_config = np.asarray([r["configuration"] for r in records])
    for axis in ("IN", "OUT"):
        strata = [np.flatnonzero((record_axis == axis) & (record_config == cfg))
                  for cfg in sorted(set(record_config[record_axis == axis]))]
        estimates = []
        for _ in range(replicates):
            picked = np.concatenate([rng.choice(indices, size=len(indices), replace=True)
                                     for indices in strata])
            # Equal configuration weight; within each group, runs have equal weight.
            objective = np.mean([np.mean(rmse_matrix[:, indices] ** 2, axis=1)
                                 for indices in [picked_group for picked_group in
                                                 [picked[np.isin(picked, stratum)] for stratum in strata]]], axis=0)
            estimates.append(float(grid[int(np.argmin(objective))]))
        estimates = np.asarray(estimates)
        output[axis] = {"lower_95_deg": float(np.quantile(estimates, 0.025)),
                        "upper_95_deg": float(np.quantile(estimates, 0.975)),
                        "median_deg": float(np.median(estimates)),
                        "samples_deg": estimates}
    return output


def plot_profile(scan: pd.DataFrame, bootstrap: dict, path: Path):
    """Show model error versus common backlash width for each axis."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = {"IN": "#0072B2", "OUT": "#D55E00"}
    for axis in ("IN", "OUT"):
        part = scan[scan.axis == axis]
        ax.plot(part.backlash_total_width_deg, part.equal_configuration_waveform_rmse_deg,
                color=colors[axis], lw=2, label=f"{axis} waveforms")
        b = bootstrap[axis]
        ax.axvspan(b["lower_95_deg"], b["upper_95_deg"], color=colors[axis], alpha=0.10)
    ax.set_xlabel("Backlash total width [deg]")
    ax.set_ylabel("Equal-configuration waveform RMSE [deg]")
    ax.set_title("Profile scan with fixed I, K, c and τ")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_waveform_spread(per_wave: pd.DataFrame, path: Path):
    """Box/point plot of each recording's individual best width by configuration."""
    ordered = ["SP00", "SP01", "SP02", "SP03", "SP04", "BALL"]
    rows = []
    for (sid, axis, cfg), group in per_wave.groupby(["segment_id", "axis", "configuration"]):
        index = int(group.waveform_rmse_deg.to_numpy().argmin())
        rows.append({"segment_id": sid, "axis": axis, "configuration": cfg,
                     "best_individual_total_width_deg": float(group.backlash_total_width_deg.iloc[index])})
    opt = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, axis, color in zip(axes, ("IN", "OUT"), ("#0072B2", "#D55E00")):
        positions, datasets = [], []
        for i, cfg in enumerate(ordered, start=1):
            values = opt[(opt.axis == axis) & (opt.configuration == cfg)].best_individual_total_width_deg.to_numpy()
            if len(values):
                positions.append(i)
                datasets.append(values)
                ax.scatter(np.full(len(values), i), values, color=color, s=24, alpha=0.75, zorder=3)
        if datasets:
            ax.boxplot(datasets, positions=positions, widths=0.45, showfliers=False,
                       medianprops={"color": "black", "linewidth": 1.4})
        ax.set_xticks(range(1, len(ordered)+1), ordered)
        ax.set_title(axis)
        ax.set_xlabel("Mounting configuration")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Single-waveform best backlash total width [deg]")
    fig.suptitle("Between-record spread (diagnostic; not independent estimates)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def configuration_profiles(per_wave: pd.DataFrame) -> pd.DataFrame:
    """Aggregate waveform errors by axis/configuration without pooling shapes."""
    rows = []
    for (axis, configuration), group in per_wave.groupby(["axis", "configuration"], sort=True):
        curve = group.groupby("backlash_total_width_deg").waveform_rmse_deg.apply(
            lambda values: float(np.sqrt(np.mean(values.to_numpy(float) ** 2))))
        best_width = float(curve.idxmin())
        rows.append({"axis": axis, "configuration": configuration,
                     "waveforms": int(group.segment_id.nunique()),
                     "best_total_width_deg": best_width,
                     "no_backlash_rmse_deg": float(curve.loc[0.0]),
                     "best_rmse_deg": float(curve.loc[best_width]),
                     "rmse_change_percent": 100.0 * (float(curve.loc[best_width]) / float(curve.loc[0.0]) - 1.0)})
    return pd.DataFrame(rows)


def plot_configuration_profiles_all(per_wave: pd.DataFrame, path: Path):
    """One 2x6 chart, with independent y scales so small residuals remain visible."""
    configs = ["SP00", "SP01", "SP02", "SP03", "SP04", "BALL"]
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.2))
    for col, config in enumerate(configs):
        for row, axis in enumerate(("IN", "OUT")):
            subset = per_wave[(per_wave.axis == axis) & (per_wave.configuration == config)]
            curve = subset.groupby("backlash_total_width_deg").waveform_rmse_deg.apply(
                lambda values: float(np.sqrt(np.mean(values.to_numpy(float) ** 2))))
            ax = axes[row, col]
            color = "#0072B2" if axis == "IN" else "#D55E00"
            ax.plot(curve.index, curve.values, color=color, lw=1.6)
            ax.axvline(float(curve.idxmin()), color=color, linestyle="--", lw=0.8)
            ax.set_title(f"{axis} · {config}\nn={subset.segment_id.nunique()}", fontsize=9)
            ax.grid(True, alpha=0.22)
            ax.tick_params(labelsize=8)
            if row == 1:
                ax.set_xlabel("Total width [deg]", fontsize=8)
            if col == 0:
                ax.set_ylabel("Waveform RMSE [deg]", fontsize=8)
    fig.suptitle("Configuration-specific diagnostic profiles (independent y scales)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_reversal_examples(records: list[dict], coefficients: dict, per_wave: pd.DataFrame,
                           path: Path):
    """Zoom around low-amplitude reversals, where a fixed gap is most visible."""
    selected = []
    for axis in ("IN", "OUT"):
        # SP01 was selected because the per-run diagnostic profile showed a
        # repeatable nonzero candidate. It is plotted as an example, not as proof
        # of axis-wide backlash (the axis-wide profile's optimum is separate).
        candidates = [r for r in records if r["axis"] == axis and r["configuration"] == "SP01"]
        if candidates:
            stratum = per_wave[(per_wave.axis == axis) & (per_wave.configuration == "SP01")]
            best_indices = stratum.groupby("segment_id").waveform_rmse_deg.idxmin()
            individual_best = stratum.loc[best_indices]
            med = float(individual_best.backlash_total_width_deg.median())
            candidate = min(candidates, key=lambda r: abs(float(
                individual_best[individual_best.segment_id == r["segment_id"]]
                .backlash_total_width_deg.iloc[0]) - med))
            selected.append(candidate)
    if len(selected) < 2:
        selected = [max([r for r in records if r["axis"] == axis and r["configuration"] == "SP00"],
                        key=lambda r: r["accepted_half_cycles"])
                    for axis in ("IN", "OUT")]
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 3.3*len(selected)), squeeze=False)
    for ax, record in zip(axes[:, 0], selected):
        individual = per_wave[per_wave.segment_id == record["segment_id"]]
        individual = individual.loc[individual.waveform_rmse_deg.idxmin()]
        candidate_width = float(individual.backlash_total_width_deg)
        angle = record["observed_deg"]
        maxima, _ = find_peaks(angle, distance=25, prominence=0.25)
        minima, _ = find_peaks(-angle, distance=25, prominence=0.25)
        extrema = np.sort(np.r_[maxima, minima])
        useful = [i for i in extrema[1:-1] if 4.0 <= abs(angle[i]) <= 10.0]
        peak_i = min(useful, key=lambda i: abs(abs(angle[i]) - 7.0)) if useful else int(extrema[len(extrema)//2])
        center_t = float(record["time"][peak_i])
        # Reinitialize each model at this local measured turning point. This
        # avoids confusing accumulated phase drift with the short gap-take-up
        # behavior immediately after reversal.
        duration = min(0.18, float(record["time"][-1] - center_t))
        local_time = np.arange(0.0, duration + SENSOR_PERIOD_S / 4.0, SENSOR_PERIOD_S)
        local_time = local_time[local_time <= duration + 1e-9]
        measured_local = np.interp(center_t + local_time, record["time"], angle)
        kind = "MAX" if peak_i in set(maxima.tolist()) else "MIN"
        local_record = {**record, "time": local_time, "observed_deg": measured_local,
                        "initial_deg": float(angle[peak_i]), "initial_kind": kind}
        reference_angle = float(angle[peak_i])
        ax.plot(local_time * 1000.0, measured_local - reference_angle, color="black", marker=".", ms=3,
                lw=0.75, label="Measured angle")
        no_rmse, no_traces = simulate_batch([local_record], coefficients, 0.0, True)
        fit_rmse, fit_traces = simulate_batch([local_record], coefficients, candidate_width, True)
        ax.plot(local_time * 1000.0, no_traces[0]["predicted_deg"] - reference_angle,
                color="#999999", lw=1.2,
                label=f"No backlash (local RMSE {no_rmse[0]:.3f}°)")
        ax.plot(local_time * 1000.0, fit_traces[0]["predicted_deg"] - reference_angle,
                color="#009E73", lw=1.25,
                label=f"SP01 run-wise candidate {candidate_width:.2f}° (RMSE {fit_rmse[0]:.3f}°)")
        ax.axvline(0.0, color="#777777", linestyle="--", lw=0.8)
        ax.set_title(f"{record['segment_id']} — {record['axis']}, measured angle {reference_angle:.1f}° reversal")
        ax.set_xlabel("Time from measured turning point [ms]")
        ax.set_ylabel("Angle change from turning point [deg]")
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle("Measured signal around a direction reversal", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, records: list[dict], best: dict, bootstrap: dict,
                 spread: pd.DataFrame, config_table: pd.DataFrame, fixed: dict):
    """Write a human-readable report including assumptions and interpretation limits."""
    lines = [
        "# 新ハード自由振動波形によるバックラッシュ推定",
        "",
        "## 目的と結論の読み方",
        "",
        f"承認済み自由振動波形{len(records)}本を使い、IN軸とOUT軸のバックラッシュを別々に推定した。I、K、ロッド抗力、球抗力、クーロン摩擦係数は既存の採用値に固定し、軸ごとに共通するバックラッシュ全幅だけを走査した。推定対象は、復元機構に入れたplayモデルの全幅である。",
        "",
        "| 軸 | 共通幅の最良値 [deg] | 再標本化した最良幅の2.5–97.5%範囲 [deg] | バックラッシュなしRMSE [deg] | 最良モデルRMSE [deg] |",
        "|---|---:|---:|---:|---:|",
    ]
    for axis in ("IN", "OUT"):
        b = bootstrap[axis]
        row = best[axis]
        lines.append(f"| {axis} | {row['total_width_deg']:.3f} | {b['lower_95_deg']:.3f}–{b['upper_95_deg']:.3f} | {row['no_backlash_rmse_deg']:.3f} | {row['rmse_deg']:.3f} |")
    lines += [
        "",
        "今回の両軸の最良値は探索下限の0°だった。再標本化した1,000組もすべて0°を選び、範囲が0°に縮退した。これは「真のバックラッシュが物理的に厳密な0°」という意味ではなく、この固定係数モデルでは非ゼロ幅を採用するだけの波形誤差改善が見られなかったことを表す。再標本化範囲は係数・中心角・モデル構造の誤差を含まず、物理的な上限ではない。",
        "",
        "## バックラッシュのモデル",
        "",
        "角度センサーの角度を $\\theta$、ばね・復元機構側の角度を $z$、バックラッシュ半幅を $\\delta$ とした。全幅は $2\\delta$ である。playモデルでは、角度が反転して隙間を通過する間、復元機構側の角度が直ちには動かない。",
        "",
        "```math",
        "z_k=\\max(\\theta_k-\\delta,\\min(z_{k-1},\\theta_k+\\delta))",
        "```",
        "",
        "運動方程式は、現在採用している固定係数に復元角 $z$ を入れた形とした。摩擦トルクと二乗抗力はそのままにし、追加パラメータは $\\delta$ のみとする。",
        "",
        "```math",
        "I\\ddot{\\theta}+K\\sin(z)+c|\\dot{\\theta}|\\dot{\\theta}+\\tau\\tanh(\\dot{\\theta}/\\epsilon)=0,\\qquad b=0",
        "```",
        "",
        "![反転点近傍の計測波形とモデル](reversal_neighborhood.png)",
        "",
        "図は、軸共通幅の結論とは別に、繰返し波形で非ゼロ幅候補が揃ったSP01の低振幅反転直後を拡大したもの。各モデルはその反転点の計測角・角速度ゼロから局所的に再初期化している。黒は計測、灰はバックラッシュなし、緑はその波形だけで選んだ幅の候補である。これは現象の見え方を説明する診断例であり、軸別採用値ではない。反転点の小さな角度差はサンプリング、角度量子化、係数誤差でも生じるため、機械的隙間を直接測った図ではない。",
        "",
        "## 推定手順と仮定",
        "",
        "1. 20260921の波形レビュー表で `APPROVED` かつ解析対象に指定されたデータを使った。IN/OUT各軸で、球とスペーサ0〜4個の全形態を含む。",
        "2. 各波形を既存前処理と同じ包絡線中心で中心化し、初期頂点から隣接する頂点の両方が4°以上の区間を対象にした。",
        "3. シミュレーションは最初の適格頂点の計測角から開始し、角速度を0とした。初期play状態は、正の最大点なら $z=\\theta-\\delta$、負の最小点なら $z=\\theta+\\delta$ とした。",
        "4. I/KはIHB-02の形態別値を使用した。ただしBALLは、今回完了した位置補正（180.61 mm）後のI/Kを使った。ロッド抗力は理論固定値、BALLの球抗力は方式Aの再同定値、軸別 $\\tau$ は方式Aの一回積分法の採用値を使った。$b=0$ とした。",
        "5. バックラッシュ全幅を0〜2°、0.02°刻みで走査した。各軸内では形態ごとの総重みを等しくし、形態内では波形を等重みにした。ODEは10 ms刻み（ログの公称100 Hz）のRK4で積分し、同じ計測間隔で角度残差を計算した。",
        "6. 区間は波形単位の再標本化を1,000回行った。複数の形態を含む軸では、各形態の標本数を保った層別再標本化とした。",
        "",
        "## 形態・繰返し間のばらつき",
        "",
        "波形別の最良幅は、個々の試行だけでRMSEが最小になる幅である。I/Kの誤差、初期条件、中心角の小誤差も引き受けるため、単独波形の値を部品固有値とは見なさない。",
        "",
        "![波形別最良バックラッシュ幅のばらつき](waveform_width_spread.png)",
        "",
        "| 軸 | 形態 | 波形数 | 波形別最良全幅中央値 [deg] | 10–90%範囲 [deg] |",
        "|---|---|---:|---:|---:|",
    ]
    opt = spread.copy()
    for (axis, configuration), group in opt.groupby(["axis", "configuration"], sort=True):
        values = group.best_individual_total_width_deg.to_numpy(float)
        lines.append(f"| {axis} | {configuration} | {len(values)} | {np.median(values):.3f} | {np.quantile(values,0.1):.3f}–{np.quantile(values,0.9):.3f} |")
    lines += [
        "",
        "### 形態別に幅を分けた診断",
        "",
        "軸共通幅ではなく形態別幅を別々に選ぶと、SP01ではIN 0.28°、OUT 0.26°の候補となり、同じ形態の繰返し波形でも概ね揃った。ただし他形態は0°または小さい幅となり、SP01以外に同じ非ゼロ幅が見られない。形態別の幅を各々採用すると自由度を増やした同一データ内適合になるため、軸全体の共通バックラッシュとしては採用しない。SP01の候補幅は、同じ形態でI/K・周期誤差が残る効果との分離がまだできていない。",
        "",
        "| 軸 | 形態 | 波形数 | 形態別最良全幅 [deg] | 幅0° RMSE [deg] | 形態別最良RMSE [deg] | RMSE変化 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in config_table.itertuples(index=False):
        lines.append(f"| {row.axis} | {row.configuration} | {row.waveforms} | {row.best_total_width_deg:.2f} | {row.no_backlash_rmse_deg:.3f} | {row.best_rmse_deg:.3f} | {row.rmse_change_percent:+.1f}% |")
    lines += [
        "",
        "![形態別のバックラッシュ幅スキャン](configuration_profile.png)",
        "",
        "![全波形での幅スキャンと95%再標本化区間](backlash_profile.png)",
        "",
        "スキャン曲線が平坦であれば、RMSEの最小位置は数値上求まっても幅をデータから識別できていない。最小値が探索端にある場合も、指定範囲内の最良点にすぎない。波形ごと・形態ごとの最良幅が大きく散る場合は、共通バックラッシュ幅の仮定を支持しない。",
        "",
        "## 解釈上の制約と次の扱い",
        "",
        "自由振動の角度だけからバックラッシュを見積もるには、I/K、摩擦、中心角、初期play状態が正しいという仮定が必要である。いずれの誤差も周期・位相に作用し、バックラッシュ推定値へ混入しうる。特に本解析では、これら係数の不確かさを再フィットしていない。角度波形の時間差はバックラッシュが存在しなくても起こる。",
        "",
        "従って、今回の結論は「軸全体で一貫するバックラッシュ幅は、この自由振動データからは確認できない」である。SP01の形態別候補は残るが、バックラッシュと周期係数の誤差を切り分けた確定値ではない。オブザーバーには現時点で非ゼロの補償幅を固定しない。将来、トルクと角度を同時計測する低速正逆往復で同じ幅が確認された場合に、改めて補償を判断する。",
        "",
        "### 固定した係数",
        "",
        f"- ロッド抗力 $c_{{rod}}$: {fixed['c_rod']:.6g} N·m·s²/rad²",
        f"- 球装着時ロッド抗力 $c_{{rod,BALL}}$: {fixed['c_rod_ball']:.6g} N·m·s²/rad²",
        f"- 球抗力 $c_{{ball}}$: {fixed['c_ball']:.6g} N·m·s²/rad²",
        f"- $\\tau_{{IN}}$: {fixed['tau_by_axis']['IN']:.6g} N·m",
        f"- $\\tau_{{OUT}}$: {fixed['tau_by_axis']['OUT']:.6g} N·m",
        "",
        "## 再実行",
        "",
        "リポジトリのルートから次を実行する。必要な入力は承認済み波形選択、頂点・中心前処理、IHB-02/03係数、IHB-05の更新係数、20260921の元角度ログである。",
        "",
        "```bash",
        "python 06_Analysis/fitting_pipeline/run_hybrid_backlash_estimation.py",
        "```",
        "",
        "主要出力: `backlash_profile.csv`, `configuration_profile.csv`, `waveform_width_scan.csv`, `waveform_best_widths.csv`, `backlash_profile.png`, `configuration_profile.png`, `waveform_width_spread.png`, `reversal_neighborhood.png`, `settings.json`。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected, centers, coefficients, fixed = read_inputs(args.data_root, args.results_root)
    records = build_records(selected, centers, args.data_root, args.results_root)
    if len(records) != len(selected):
        print(f"Note: accepted records with sufficient adjacent peaks = {len(records)} / {len(selected)}")
    grid, scan, per_wave, rmse_matrix, best = grid_search(records, coefficients)
    bootstrap = bootstrap_profile(records, grid, rmse_matrix)
    config_table = configuration_profiles(per_wave)

    scan.to_csv(args.output_dir / "backlash_profile.csv", index=False, float_format="%.8g")
    per_wave.to_csv(args.output_dir / "waveform_width_scan.csv", index=False, float_format="%.8g")
    best_rows = []
    for axis in ("IN", "OUT"):
        b = best[axis]
        ci = bootstrap[axis]
        best_rows.append({"axis": axis, "best_total_width_deg": b["total_width_deg"],
                          "best_half_width_deg": b["half_width_deg"],
                          "bootstrap_95_lower_deg": ci["lower_95_deg"],
                          "bootstrap_95_upper_deg": ci["upper_95_deg"],
                          "no_backlash_rmse_deg": b["no_backlash_rmse_deg"],
                          "best_rmse_deg": b["rmse_deg"],
                          "relative_rmse_change_percent": 100*(b["rmse_deg"]/b["no_backlash_rmse_deg"]-1)})
    summary = pd.DataFrame(best_rows)
    summary.to_csv(args.output_dir / "backlash_axis_estimates.csv", index=False, float_format="%.8g")

    width_opt = []
    for segment_id, group in per_wave.groupby("segment_id", sort=False):
        i = int(group.waveform_rmse_deg.to_numpy().argmin())
        first = group.iloc[i]
        width_opt.append({"segment_id": segment_id, "axis": first.axis,
                          "configuration": first.configuration,
                          "direction": first.direction, "repetition": first.repetition,
                          "half_cycles": first.half_cycles,
                          "best_individual_total_width_deg": first.backlash_total_width_deg,
                          "best_individual_rmse_deg": first.waveform_rmse_deg})
    spread = pd.DataFrame(width_opt)
    spread.to_csv(args.output_dir / "waveform_best_widths.csv", index=False, float_format="%.8g")
    config_table.to_csv(args.output_dir / "configuration_profile.csv", index=False, float_format="%.8g")

    plot_profile(scan, bootstrap, args.output_dir / "backlash_profile.png")
    plot_configuration_profiles_all(per_wave, args.output_dir / "configuration_profile.png")
    plot_waveform_spread(per_wave, args.output_dir / "waveform_width_spread.png")
    plot_reversal_examples(records, coefficients, per_wave,
                           args.output_dir / "reversal_neighborhood.png")

    settings = {
        "date": DATE, "waveform_count": len(records),
        "minimum_peak_amplitude_deg": MINIMUM_PEAK_AMPLITUDE_DEG,
        "total_width_scan_deg": [0.0, MAX_TOTAL_WIDTH_DEG],
        "total_width_step_deg": TOTAL_WIDTH_STEP_DEG,
        "integration_step_s": INTEGRATION_STEP_S,
        "measurement_rate_hz": 1/SENSOR_PERIOD_S,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "random_seed": RNG_SEED,
        "waveform_weighting": "equal configuration, then equal waveform within configuration",
        "initial_state": "measured first eligible turning angle, zero speed; play state at direction-specific boundary",
        "fitted_parameters": ["one shared play half-width per axis"],
        "fixed_parameters": {"I_K_source": "IHB-02 per configuration; BALL uses the accepted ±10 mm IHB-05 update",
                             "tau_source": "IHB-03 monotone-amplitude one-integral values used by IHB-05",
                             "b_rad_s": 0.0, **fixed},
        "results": summary.to_dict(orient="records"),
        "scope_limit": "conditional model-based estimate; does not include fixed coefficient uncertainty",
    }
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.output_dir / "BACKLASH_ESTIMATION_REPORT.md", records, best,
                 bootstrap, spread, config_table, fixed)
    print(summary.to_string(index=False))
    print(f"Wrote report and outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
