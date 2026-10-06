"""Robust 0-10 Hz force-estimation sweep over damper and RTS settings.

The true BALL plant and the angle-only RTS are simulated nonlinearly at 100 Hz.
The study compares the adopted zero-wind damping model against three added
rotary-viscous-damper levels, six RTS force random-walk settings, and the
OW-04 100% coefficient-error corner. It is a design sweep, not a hardware
validation; the input is force-domain sinusoidal loading about 2 m/s equivalent.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(HERE))
from hbk_model_coefficients import select_coefficients, simulate_hbk_plant
from hbk_nonlinear_estimators import (
    _continuous_jacobian, _rk4_step, nonlinear_ekf_rts_force,
)
from run_ow05_force_gain_frequency_response import (
    SCENARIO, applied_force, fundamental, load_config, mismatch_coefficients,
)

OUT = SIM / "results/observer_wind/ow05_robust_bandwidth_damper_sweep"
FREQUENCIES_HZ = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    0.85, 0.9, 0.925, 0.95, 0.975, 0.98, 0.99, 0.995,
    1.0, 1.005, 1.01, 1.02, 1.025, 1.05, 1.075, 1.1,
    1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0,
)
AMPLITUDE_RATIOS = (0.35,)  # worst tested amplitude in the preceding sweep
LOW_AMPLITUDE_RATIO = 0.1
LOW_AMPLITUDE_CHECK_FREQUENCIES_HZ = (0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 10.0)
DAMPER_B_VALUES = (0.0, 1.0e-4, 3.0e-4)  # N m s/rad, added viscous damping
Q_MULTIPLIERS = (1.0, 10.0, 30.0)
HIGH_Q_MULTIPLIERS = (100.0, 300.0, 1000.0)
HIGH_Q_CHECK_FREQUENCIES_HZ = (0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 3.0, 5.0, 7.5, 10.0)
HIGH_Q_LOW_AMPLITUDE_CHECK_HZ = (0.95, 1.0, 1.05, 10.0)
PLOT_Q_MULTIPLIERS = Q_MULTIPLIERS + HIGH_Q_MULTIPLIERS
BASE_Q = {"IN": 3.0e-5, "OUT": 3.0e-4}
OBSERVER_CASES = ("matched", "OW-04 100%")
COLORS_B = {0.0: "#444444", 1.0e-4: "#2776bc", 3.0e-4: "#d46a25"}
COLORS_Q = {1.0: "#2776bc", 10.0: "#e67e22", 30.0: "#159477",
            100.0: "#8e5ab5", 300.0: "#cf4d76", 1000.0: "#704214"}


def _score_mask(time_s: np.ndarray, f_hz: float) -> np.ndarray:
    start = max(5.0, 3.0 / f_hz)
    return (time_s >= start) & (time_s <= time_s[-1] - max(1.0 / f_hz, 0.5))


def _fit_metrics(force: np.ndarray, estimate: np.ndarray, time_s: np.ndarray,
                 f_hz: float, score: np.ndarray) -> dict:
    truth_amp, _ = fundamental(force[score], time_s[score], f_hz)
    estimate_amp, phase_deg = fundamental(estimate[score], time_s[score], f_hz)
    phase_rad = math.radians(phase_deg)
    lag_s = -phase_rad / (2.0 * math.pi * f_hz)
    # A single phase-derived shift is allowed for the shape metric. It leaves
    # amplitude error and nonlinear harmonics visible; it does not warp time.
    valid_t = time_s[score] - phase_rad / (2.0 * math.pi * f_hz)
    shifted = np.interp(valid_t, time_s, estimate)
    true_ac = force[score] - np.mean(force[score])
    ac_scale = float(np.sqrt(np.mean(true_ac ** 2)))
    raw_err = estimate[score] - force[score]
    shifted_err = shifted - force[score]
    estimate_bias = float(np.mean(estimate[score]) - np.mean(force[score]))
    return {
        "gain_estimated_over_input": estimate_amp / truth_amp,
        "phase_estimate_minus_input_deg": phase_deg,
        "estimated_lag_s_positive_means_lag": lag_s,
        "input_fundamental_n": truth_amp,
        "estimate_fundamental_n": estimate_amp,
        "dc_bias_n": estimate_bias,
        "raw_ac_rmse_normalized": float(np.sqrt(np.mean(raw_err ** 2)) / ac_scale),
        "lag_corrected_shape_nrmse": float(np.sqrt(np.mean(shifted_err ** 2)) / ac_scale),
    }


def _fast_transition(jacobian: np.ndarray, dt: float) -> np.ndarray:
    """Exact expm for the estimator's [angle, rate, constant-force] Jacobian.

    The upper-left block is 2x2 and the force state is constant. This closed
    form is algebraically identical to expm(J*dt), but avoids a general 3x3
    matrix exponential at every sample. It is checked against scipy expm in
    the script's self-check before a sweep is run.
    """
    a, b, c = float(jacobian[1, 0]), float(jacobian[1, 1]), float(jacobian[1, 2])
    half_trace = 0.5 * b
    discriminant = half_trace * half_trace + a
    centered = np.array([[-half_trace, 1.0], [a, half_trace]])
    identity2 = np.eye(2)
    if discriminant > 1e-14:
        root = math.sqrt(discriminant)
        scalar_i = math.cosh(root * dt)
        scalar_c = math.sinh(root * dt) / root
    elif discriminant < -1e-14:
        root = math.sqrt(-discriminant)
        scalar_i = math.cos(root * dt)
        scalar_c = math.sin(root * dt) / root
    else:
        scalar_i = 1.0
        scalar_c = dt
    block = math.exp(half_trace * dt) * (scalar_i * identity2 + scalar_c * centered)
    bmat = np.array([[0.0, 1.0], [a, b]])
    force_col = np.linalg.solve(bmat, (block - identity2) @ np.array([0.0, c]))
    transition = np.eye(3)
    transition[:2, :2] = block
    transition[:2, 2] = force_col
    return transition


def _fast_nonlinear_rts(angle_rad: np.ndarray, coefficients: dict, dt: float,
                        force_lever_m: float, angle_noise_rad: float,
                        q_force_std: float, epsilon_deg_s: float,
                        initial_state: np.ndarray) -> np.ndarray:
    """Same order-0 EKF/RTS as the project implementation, with fast expm."""
    n = 3
    measurement = np.array([[1.0, 0.0, 0.0]])
    r = angle_noise_rad ** 2
    q = np.zeros((n, n))
    q[2, 2] = q_force_std ** 2
    initial_cov = np.diag([math.radians(5.0) ** 2, 1.0, 0.5 ** 2])
    eye = np.eye(n)
    count = len(angle_rad)
    filt = np.zeros((count, n))
    pred = np.zeros_like(filt)
    filt_cov = np.zeros((count, n, n))
    pred_cov = np.zeros_like(filt_cov)
    state = np.asarray(initial_state, dtype=float).copy()
    covariance = initial_cov
    epsilon = math.radians(epsilon_deg_s)
    for k, obs in enumerate(angle_rad):
        if k:
            jac = _continuous_jacobian(filt[k - 1], coefficients, force_lever_m, epsilon)
            transition = _fast_transition(jac, dt)
            state = _rk4_step(filt[k - 1], dt, coefficients, force_lever_m, epsilon)
            covariance = transition @ filt_cov[k - 1] @ transition.T + q
        else:
            transition = np.eye(n)
        pred[k] = state
        pred_cov[k] = covariance
        innovation = float(obs - (measurement @ state).item())
        variance = float((measurement @ covariance @ measurement.T).item() + r)
        gain = covariance @ measurement.T / variance
        state = state + gain[:, 0] * innovation
        correction = eye - gain @ measurement
        covariance = correction @ covariance @ correction.T + gain * r @ gain.T
        covariance = 0.5 * (covariance + covariance.T)
        filt[k] = state
        filt_cov[k] = covariance
    smooth = filt.copy()
    for k in range(count - 2, -1, -1):
        jac = _continuous_jacobian(filt[k], coefficients, force_lever_m, epsilon)
        transition = _fast_transition(jac, dt)
        cross = filt_cov[k] @ transition.T
        smoother_gain = np.linalg.solve(pred_cov[k + 1].T, cross.T).T
        smooth[k] += smoother_gain @ (smooth[k + 1] - pred[k + 1])
    return smooth[:, 2].copy()


def _check_fast_transition() -> None:
    rng = np.random.default_rng(20261006)
    for _ in range(30):
        a = -rng.uniform(10.0, 1000.0)
        b = -rng.uniform(0.0, 10.0)
        c = rng.uniform(1.0, 1000.0)
        jac = np.array([[0.0, 1.0, 0.0], [a, b, c], [0.0, 0.0, 0.0]])
        fast = _fast_transition(jac, 0.01)
        exact = expm(jac * 0.01)
        if not np.allclose(fast, exact, rtol=2e-11, atol=2e-12):
            raise AssertionError(f"specialized transition mismatch: {np.max(np.abs(fast-exact))}")


def run_sinusoid(axis: str, f_hz: float, amplitude_ratio: float, b_added: float,
                 q_multiplier: float, cfg: dict, plant_nominal: dict,
                 observer_nominal: dict, observer_mismatch: dict) -> list[dict]:
    fs = float(cfg["sample_rate_hz"])
    dt = 1.0 / fs
    f0 = float(applied_force(np.array([2.0]), cfg)[0])
    amp_force = amplitude_ratio * f0
    duration_s = max(20.0, 8.0 / f_hz)
    n = int(round(duration_s * fs))
    time_s = np.arange(n) * dt
    force = f0 + amp_force * np.sin(2.0 * math.pi * f_hz * time_s)
    theta0 = math.atan(cfg["force_lever_m"] * f0 / plant_nominal["restoring_n_m_per_rad"])

    plant = plant_nominal.copy()
    plant["viscous_damping_n_m_s_per_rad"] = b_added
    angle = simulate_hbk_plant(
        force, plant, dt, theta0, 0.0, cfg["force_lever_m"], cfg["friction_epsilon_deg_s"]
    )[:, 0]
    max_angle = float(np.max(np.abs(np.rad2deg(angle))))
    mask = _score_mask(time_s, f_hz)
    q = BASE_Q[axis] * q_multiplier
    initial_state = np.array([angle[0], 0.0, f0])
    out = []
    for observer_case, base in (("matched", observer_nominal), ("OW-04 100%", observer_mismatch)):
        observer = base.copy()
        # The damper value itself is assumed known in this first design sweep;
        # the separate OW-04 corner perturbs I/K/c/tau as before.
        observer["viscous_damping_n_m_s_per_rad"] = b_added
        estimate = _fast_nonlinear_rts(
            angle, observer, dt, cfg["force_lever_m"],
            np.deg2rad(cfg["assumed_angle_noise_deg"]), q,
            cfg["friction_epsilon_deg_s"], initial_state,
        )
        metrics = _fit_metrics(force, estimate, time_s, f_hz, mask)
        out.append({
            "axis": axis, "frequency_hz": f_hz,
            "force_mean_n": f0, "force_amplitude_n": amp_force,
            "amplitude_ratio_A_over_F0": amplitude_ratio,
            "plant_added_damper_b_n_m_s_rad": b_added,
            "added_damping_ratio_linear_reference": b_added / (2.0 * math.sqrt(
                plant_nominal["inertia_kg_m2"] * plant_nominal["restoring_n_m_per_rad"])),
            "observer_case": observer_case,
            "observer_b_assumed_n_m_s_rad": b_added,
            "q_multiplier_vs_adopted": q_multiplier,
            "rts_q_n_per_sample": q,
            "max_abs_angle_deg": max_angle,
            "within_60_deg_limit": max_angle <= cfg["mechanical_angle_limit_deg"],
            **metrics,
        })
    return out


def run_dc(axis: str, amplitude_ratio: float, b_added: float, q_multiplier: float,
           cfg: dict, plant_nominal: dict, observer_nominal: dict,
           observer_mismatch: dict) -> list[dict]:
    """Steady-state force offset response; DC has no sinusoidal phase/gain fit."""
    fs = float(cfg["sample_rate_hz"])
    dt = 1.0 / fs
    f0 = float(applied_force(np.array([2.0]), cfg)[0])
    delta = amplitude_ratio * f0
    applied = f0 + delta
    duration_s = 30.0
    n = int(duration_s * fs)
    time_s = np.arange(n) * dt
    force = np.full(n, applied)
    theta0 = math.atan(cfg["force_lever_m"] * f0 / plant_nominal["restoring_n_m_per_rad"])
    plant = plant_nominal.copy()
    plant["viscous_damping_n_m_s_per_rad"] = b_added
    angle = simulate_hbk_plant(force, plant, dt, theta0, 0.0,
                               cfg["force_lever_m"], cfg["friction_epsilon_deg_s"])[:, 0]
    score = time_s >= 20.0
    rows = []
    q = BASE_Q[axis] * q_multiplier
    initial = np.array([angle[0], 0.0, f0])
    for observer_case, base in (("matched", observer_nominal), ("OW-04 100%", observer_mismatch)):
        observer = base.copy()
        observer["viscous_damping_n_m_s_per_rad"] = b_added
        estimate = _fast_nonlinear_rts(
            angle, observer, dt, cfg["force_lever_m"],
            np.deg2rad(cfg["assumed_angle_noise_deg"]), q,
            cfg["friction_epsilon_deg_s"], initial,
        )
        estimate_delta = float(np.mean(estimate[score]) - f0)
        rows.append({
            "axis": axis, "frequency_hz": 0.0,
            "force_mean_n": f0, "force_amplitude_n": delta,
            "amplitude_ratio_A_over_F0": amplitude_ratio,
            "plant_added_damper_b_n_m_s_rad": b_added,
            "added_damping_ratio_linear_reference": b_added / (2.0 * math.sqrt(
                plant_nominal["inertia_kg_m2"] * plant_nominal["restoring_n_m_per_rad"])),
            "observer_case": observer_case, "observer_b_assumed_n_m_s_rad": b_added,
            "q_multiplier_vs_adopted": q_multiplier, "rts_q_n_per_sample": q,
            "max_abs_angle_deg": float(np.max(np.abs(np.rad2deg(angle)))),
            "within_60_deg_limit": bool(np.max(np.abs(np.rad2deg(angle))) <= cfg["mechanical_angle_limit_deg"]),
            "gain_estimated_over_input": estimate_delta / delta,
            "phase_estimate_minus_input_deg": float("nan"),
            "estimated_lag_s_positive_means_lag": float("nan"),
            "input_fundamental_n": float("nan"), "estimate_fundamental_n": float("nan"),
            "dc_bias_n": float(np.mean(estimate[score]) - applied),
            "raw_ac_rmse_normalized": float("nan"),
            "lag_corrected_shape_nrmse": float("nan"),
            "steady_force_estimate_n": float(np.mean(estimate[score])),
        })
    return rows


def make_plots(frame: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    colors = {1.0: "#2776bc", 10.0: "#e67e22", 30.0: "#159477",
              100.0: "#8055a6", 300.0: "#cb4b4b", 1000.0: "#52616b"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey="row", layout="constrained")
    for row, axis in enumerate(("IN", "OUT")):
        for col, b in enumerate(DAMPER_B_VALUES):
            ax = axes[row, col]
            sub = frame[(frame.axis == axis) & (frame.plant_added_damper_b_n_m_s_rad == b)
                        & (frame.amplitude_ratio_A_over_F0 == 0.35)]
            for q_mult in PLOT_Q_MULTIPLIERS:
                for case, ls in (("matched", "-"), ("OW-04 100%", "--")):
                    curve = sub[(sub.q_multiplier_vs_adopted == q_mult)
                                & (sub.observer_case == case) & (sub.frequency_hz > 0)].sort_values("frequency_hz")
                    ax.plot(curve.frequency_hz, curve.gain_estimated_over_input,
                            color=colors[q_mult], ls=ls, lw=1.35,
                            label=f"q×{q_mult:g} {case}")
            ax.axhline(1.0, color="#333", ls=":", lw=0.8)
            ax.axvline(1.0, color="#888", ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlim(0.09, 11)
            ax.grid(True, which="both", alpha=.25)
            ax.set_title(f"{axis} | added b={b:.0e} N m s/rad")
            if row == 1:
                ax.set_xlabel("Applied force frequency [Hz]")
            if col == 0:
                ax.set_ylabel("Estimated / applied amplitude gain")
            if row == 0 and col == 2:
                ax.legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Nonlinear RTS force gain, 0-10 Hz sweep | A/F0=0.35")
    fig.savefig(OUT / "gain_vs_frequency_b_q.png", dpi=180)
    fig.savefig(OUT / "gain_vs_frequency_b_q.svg")
    plt.close(fig)

    # Worst-case amplitude gain error by frequency for each damper and q.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    for ax, axis in zip(axes, ("IN", "OUT")):
        sub = frame[(frame.axis == axis) & (frame.frequency_hz > 0)]
        for b in DAMPER_B_VALUES:
            for q_mult in PLOT_Q_MULTIPLIERS:
                group = sub[(sub.plant_added_damper_b_n_m_s_rad == b)
                            & (sub.q_multiplier_vs_adopted == q_mult)]
                worst = group.groupby("frequency_hz").gain_estimated_over_input.apply(
                    lambda x: float(np.max(np.abs(np.asarray(x, float) - 1.0))))
                ax.plot(worst.index, worst.values, color=colors[q_mult],
                        ls={0.0: ":", 1e-4: "-", 3e-4: "--"}[b], lw=1.4,
                        label=f"b={b:.0e}, q×{q_mult:g}")
        ax.set_xscale("log")
        ax.set_xlim(.09, 11)
        ax.set_ylim(bottom=0)
        ax.axvline(1.0, color="#888", ls="--", lw=.8)
        ax.grid(True, which="both", alpha=.25)
        ax.set_title(f"{axis}: worst |gain−1| across amplitudes/models")
        ax.set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("Absolute amplitude gain error")
    axes[1].legend(fontsize=7.3, ncol=2, loc="best")
    fig.suptitle("Robustness envelope across amplitude and OW-04 coefficient mismatch")
    fig.savefig(OUT / "worst_gain_error_vs_frequency.png", dpi=180)
    fig.savefig(OUT / "worst_gain_error_vs_frequency.svg")
    plt.close(fig)

    # Heatmaps summarize the in-band worst-case gain deviation for each b/q.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    bands = ((0.8, 1.2, "Near natural frequency 0.8-1.2 Hz"),
             (5.0, 10.0, "Upper band 5-10 Hz"))
    for row, axis in enumerate(("IN", "OUT")):
        sub = frame[(frame.axis == axis) & (frame.frequency_hz > 0)]
        for col, (lo, hi, title) in enumerate(bands):
            matrix = np.zeros((len(DAMPER_B_VALUES), len(PLOT_Q_MULTIPLIERS)))
            for i, b in enumerate(DAMPER_B_VALUES):
                for j, q_mult in enumerate(PLOT_Q_MULTIPLIERS):
                    c = sub[(sub.plant_added_damper_b_n_m_s_rad == b)
                            & (sub.q_multiplier_vs_adopted == q_mult)
                            & (sub.frequency_hz >= lo) & (sub.frequency_hz <= hi)]
                    matrix[i, j] = float(np.max(np.abs(c.gain_estimated_over_input - 1.0)))
            im = axes[row, col].imshow(matrix, aspect="auto", cmap="magma", origin="lower")
            axes[row, col].set_xticks(range(len(PLOT_Q_MULTIPLIERS)), [f"×{x:g}" for x in PLOT_Q_MULTIPLIERS])
            axes[row, col].set_yticks(range(len(DAMPER_B_VALUES)), [f"{x:.0e}" for x in DAMPER_B_VALUES])
            axes[row, col].set_xlabel("RTS q / adopted q")
            axes[row, col].set_ylabel("Added b [N m s/rad]")
            axes[row, col].set_title(f"{axis} | {title}")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axes[row, col].text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white", fontsize=9)
            fig.colorbar(im, ax=axes[row, col], label="Worst |gain−1|")
    fig.suptitle("In-band worst-case amplitude error (all tested amplitudes and models)")
    fig.savefig(OUT / "inband_robustness_heatmap.png", dpi=180)
    fig.savefig(OUT / "inband_robustness_heatmap.svg")
    plt.close(fig)

    # Phase response for the coefficient-mismatch corner, comparing adopted q
    # with the higher setting that restored the simulated 5-10 Hz gain.
    phase = frame[(frame.observer_case == "OW-04 100%")
                  & (frame.amplitude_ratio_A_over_F0 == 0.35)
                  & (frame.frequency_hz > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, layout="constrained")
    damper_colors = {0.0: "#444444", 1.0e-4: "#2776bc", 3.0e-4: "#d46a25"}
    for ax, axis in zip(axes, ("IN", "OUT")):
        for b in DAMPER_B_VALUES:
            for q_mult, ls in ((1.0, "--"), (300.0, "-")):
                curve = phase[(phase.axis == axis)
                              & (phase.plant_added_damper_b_n_m_s_rad == b)
                              & (phase.q_multiplier_vs_adopted == q_mult)].sort_values("frequency_hz")
                ax.plot(curve.frequency_hz, curve.phase_estimate_minus_input_deg,
                        color=damper_colors[b], ls=ls, lw=1.5,
                        label=f"b={b:.0e}, q×{q_mult:g}")
        ax.axhline(0, color="#333", lw=0.8)
        ax.axvline(1, color="#888", ls=":", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlim(0.09, 11)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(axis)
        ax.set_xlabel("Applied force frequency [Hz]")
    axes[0].set_ylabel("Estimated force phase − applied force phase [deg]")
    axes[1].legend(fontsize=8, ncol=2, loc="best")
    fig.suptitle("Force estimation phase response | OW-04 mismatch, A/F0=0.35")
    fig.savefig(OUT / "phase_vs_frequency_b_q.png", dpi=180)
    fig.savefig(OUT / "phase_vs_frequency_b_q.svg")
    plt.close(fig)


def main() -> None:
    _check_fast_transition()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    runs = 0
    condition_count = (len(FREQUENCIES_HZ) + len(LOW_AMPLITUDE_CHECK_FREQUENCIES_HZ)
                       + 2)  # two DC amplitudes
    total = len(("IN", "OUT")) * len(DAMPER_B_VALUES) * condition_count * len(Q_MULTIPLIERS) * len(OBSERVER_CASES)
    for axis in ("IN", "OUT"):
        nominal = select_coefficients(axis, "BALL")
        mismatch = mismatch_coefficients(axis, nominal)
        for b in DAMPER_B_VALUES:
            for ratio in AMPLITUDE_RATIOS:
                for f_hz in FREQUENCIES_HZ:
                    # Plant trajectory is independent of observer q/model; reuse it
                    # inside run_sinusoid's paired observer calculation.
                    for q_mult in Q_MULTIPLIERS:
                        rows.extend(run_sinusoid(axis, f_hz, ratio, b, q_mult, cfg,
                                                 nominal, nominal, mismatch))
                        runs += len(OBSERVER_CASES)
                        if runs % 100 == 0:
                            print(f"observer runs {runs}/{total}", flush=True)
                for f_hz in LOW_AMPLITUDE_CHECK_FREQUENCIES_HZ:
                    for q_mult in Q_MULTIPLIERS:
                        rows.extend(run_sinusoid(axis, f_hz, LOW_AMPLITUDE_RATIO, b, q_mult,
                                                 cfg, nominal, nominal, mismatch))
                        runs += len(OBSERVER_CASES)
                        if runs % 100 == 0:
                            print(f"observer runs {runs}/{total}", flush=True)
                for ratio in (0.1, 0.35):
                    for q_mult in Q_MULTIPLIERS:
                        rows.extend(run_dc(axis, ratio, b, q_mult, cfg, nominal, nominal, mismatch))
                        runs += len(OBSERVER_CASES)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "bandwidth_damper_sweep_metrics.csv", index=False,
                 encoding="utf-8-sig", float_format="%.9g")
    settings = {
        "target_force_band_hz": [0.0, 10.0],
        "sample_rate_hz": cfg["sample_rate_hz"],
        "mean_wind_equivalent_m_s": 2.0,
        "mean_force_n": float(applied_force(np.array([2.0]), cfg)[0]),
        "force_amplitude_ratios_A_over_F0": [0.1, 0.35],
        "frequency_sweep_amplitude_A_over_F0": 0.35,
        "low_amplitude_resonance_check_A_over_F0": LOW_AMPLITUDE_RATIO,
        "low_amplitude_check_frequencies_hz": LOW_AMPLITUDE_CHECK_FREQUENCIES_HZ,
        "frequency_hz_nonzero": FREQUENCIES_HZ,
        "zero_frequency_method": "constant force offset +A_F; evaluate final 10 s as DC gain",
        "plant": "full nonlinear adopted BALL model, with added linear viscous damper b",
        "added_damper_b_n_m_s_per_rad": DAMPER_B_VALUES,
        "observer": "nonlinear RTS, order 0; angle-only; ideal simulated angle; assumed angle sigma 0.02 deg",
        "observer_q": {a: [BASE_Q[a] * m for m in Q_MULTIPLIERS] for a in ("IN", "OUT")},
        "q_multipliers": Q_MULTIPLIERS,
        "observer_models": ["matched plant coefficients", f"OW-04 full coefficient-error corner: {SCENARIO}"],
        "damper_uncertainty": "not included; observer is given the exact b used by plant; isolates effect of damping and RTS q",
        "gain": "fitted estimated-force fundamental amplitude / applied-force fundamental amplitude after startup",
        "shape_error": "NRMSE after one phase-derived time shift; no time warping",
        "mechanical_limit": "report max angle and whether it stays within +/-60 deg",
    }
    (OUT / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_plots(frame)
    print(f"Wrote {len(frame)} rows to {OUT}", flush=True)
    for axis in ("IN", "OUT"):
        for b in DAMPER_B_VALUES:
            for q_mult in Q_MULTIPLIERS:
                sub = frame[(frame.axis == axis) & (frame.plant_added_damper_b_n_m_s_rad == b)
                            & (frame.q_multiplier_vs_adopted == q_mult) & (frame.frequency_hz > 0)]
                worst = sub.loc[(sub.gain_estimated_over_input - 1).abs().idxmax()]
                print(axis, f"b={b:.0e}", f"q×{q_mult:g}",
                      f"worst gain={worst.gain_estimated_over_input:.3f} at {worst.frequency_hz:g} Hz",
                      f"case={worst.observer_case}, A/F0={worst.amplitude_ratio_A_over_F0:g}")


def extend_high_q() -> None:
    """Target higher q values at resonance and upper-band points.

    The first sweep showed that q<=30x the adopted setting cannot preserve
    the IN-axis 5-10 Hz response. This extension tests higher q only where it
    informs the 1 Hz robustness peak and the upper passband, avoiding a costly
    full-band rerun for q values that already imply substantial noise gain.
    """
    _check_fast_transition()
    cfg = load_config()
    base_path = OUT / "bandwidth_damper_sweep_metrics.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Run the base sweep first: {base_path}")
    base_frame = pd.read_csv(base_path)
    extension_rows: list[dict] = []
    n_done = 0
    total = len(("IN", "OUT")) * len(DAMPER_B_VALUES) * len(HIGH_Q_MULTIPLIERS) * (
        len(HIGH_Q_CHECK_FREQUENCIES_HZ) + len(HIGH_Q_LOW_AMPLITUDE_CHECK_HZ) + 2
    ) * len(OBSERVER_CASES)
    for axis in ("IN", "OUT"):
        nominal = select_coefficients(axis, "BALL")
        mismatch = mismatch_coefficients(axis, nominal)
        for b in DAMPER_B_VALUES:
            for f_hz in HIGH_Q_CHECK_FREQUENCIES_HZ:
                for q_mult in HIGH_Q_MULTIPLIERS:
                    extension_rows.extend(run_sinusoid(axis, f_hz, 0.35, b, q_mult,
                                                       cfg, nominal, nominal, mismatch))
                    n_done += len(OBSERVER_CASES)
                    if n_done % 100 == 0:
                        print(f"high-q observer runs {n_done}/{total}", flush=True)
            for f_hz in HIGH_Q_LOW_AMPLITUDE_CHECK_HZ:
                for q_mult in HIGH_Q_MULTIPLIERS:
                    extension_rows.extend(run_sinusoid(axis, f_hz, 0.1, b, q_mult,
                                                       cfg, nominal, nominal, mismatch))
                    n_done += len(OBSERVER_CASES)
            for ratio in (0.1, 0.35):
                for q_mult in HIGH_Q_MULTIPLIERS:
                    extension_rows.extend(run_dc(axis, ratio, b, q_mult, cfg,
                                                 nominal, nominal, mismatch))
                    n_done += len(OBSERVER_CASES)
    extension = pd.DataFrame(extension_rows)
    extension.to_csv(OUT / "high_q_extension_metrics.csv", index=False,
                     encoding="utf-8-sig", float_format="%.9g")
    combined = pd.concat([base_frame, extension], ignore_index=True)
    combined.to_csv(OUT / "bandwidth_damper_sweep_all_metrics.csv", index=False,
                    encoding="utf-8-sig", float_format="%.9g")
    extension_settings = {
        "purpose": "test observer q values high enough to approach a 10 Hz force-estimation passband",
        "q_multipliers_vs_adopted": HIGH_Q_MULTIPLIERS,
        "q_values_n_per_sample": {a: [BASE_Q[a] * m for m in HIGH_Q_MULTIPLIERS] for a in ("IN", "OUT")},
        "frequencies_hz_A_over_F0_0_35": HIGH_Q_CHECK_FREQUENCIES_HZ,
        "frequencies_hz_A_over_F0_0_1": HIGH_Q_LOW_AMPLITUDE_CHECK_HZ,
        "sensor_noise": "none added; optimistic bandwidth upper-bound check",
    }
    (OUT / "high_q_extension_settings.json").write_text(
        json.dumps(extension_settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_plots(combined)
    print(f"Wrote {len(extension)} high-q rows and {len(combined)} combined rows", flush=True)
    for axis in ("IN", "OUT"):
        for b in DAMPER_B_VALUES:
            for q_mult in HIGH_Q_MULTIPLIERS:
                sub = combined[(combined.axis == axis)
                               & (combined.plant_added_damper_b_n_m_s_rad == b)
                               & (combined.q_multiplier_vs_adopted == q_mult)
                               & (combined.observer_case == "OW-04 100%")
                               & (combined.frequency_hz >= 5.0)]
                if len(sub):
                    worst = sub.loc[(sub.gain_estimated_over_input - 1).abs().idxmax()]
                    at10 = sub[sub.frequency_hz == 10.0].iloc[0]
                    print(axis, f"b={b:.0e}", f"q×{q_mult:g}",
                          f"10Hz gain={at10.gain_estimated_over_input:.3f}",
                          f"upper-band worst={worst.gain_estimated_over_input:.3f} at {worst.frequency_hz:g}Hz")


if __name__ == "__main__":
    if "--extend-high-q" in sys.argv:
        extend_high_q()
    else:
        main()
