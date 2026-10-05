"""Controlled OW-05 study separating wind fluctuation amplitude and frequency.

Compares matched and OW-04 100%-mismatch offline RTS, static conversion,
OW-03 causal low-pass, and centered 1 s / 3 s averages of static conversion.
All methods see the same noise-free plant angle for each wind input.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

from hbk_model_coefficients import select_coefficients, simulate_hbk_plant
from hbk_nonlinear_estimators import nonlinear_ekf_rts_force
from wind import drag_force_from_speed

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "ow03_hbk_observer_tuning.json"
OW04_CSV = ROOT / "results" / "observer_wind" / "ow04_coefficient_sensitivity" / "ow04_joint_scenarios.csv"
OUT = ROOT / "results" / "observer_wind" / "ow05_model_error_slew_comparison"
SCENARIO = "joint_mass_high_arm_high_tau-high_c-high"
METHODS = ("RTS matched", "RTS OW-04 100%", "Static", "LPF", "Static mean 1 s", "Static mean 3 s")
COLORS = {"RTS matched":"#1b9e77", "RTS OW-04 100%":"#d95f02", "Static":"#777777", "LPF":"#7570b3", "Static mean 1 s":"#66a61e", "Static mean 3 s":"#e6ab02"}


def load_config():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["rts_q"] = {"IN": 3e-5, "OUT": 3e-4}
    cfg["lpf_hz"] = {"IN": 0.5, "OUT": 0.75}
    return cfg


def mismatch_coefficients(axis, nominal):
    with OW04_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        row = next(r for r in rows if r["axis"] == axis and r["scenario"] == SCENARIO)
    result = nominal.copy()
    result["inertia_kg_m2"] = float(row["observer_I_kg_m2"])
    result["restoring_n_m_per_rad"] = float(row["observer_K_n_m_per_rad"])
    result["tau_n_m"] = float(row["observer_tau_n_m"])
    result["ball_quadratic_drag_n_m_s2_per_rad2"] = float(row["observer_c_ball_n_m_s2_per_rad2"])
    result["total_quadratic_drag_n_m_s2_per_rad2"] = result["rod_quadratic_drag_n_m_s2_per_rad2"] + result["ball_quadratic_drag_n_m_s2_per_rad2"]
    return result


def moving_average_centered(x, seconds, fs):
    width = int(round(seconds * fs))
    if width % 2 == 0:
        width += 1
    return np.convolve(x, np.ones(width) / width, mode="same")


def signed_wind(force, cfg):
    aero = cfg["air_density_kg_m3"] * cfg["drag_coefficient"] * cfg["projected_area_m2"]
    force = np.asarray(force, dtype=float)
    return np.sign(force) * np.sqrt(2.0 * np.abs(force) / aero)


def metrics(truth, estimate, start, dt, frequency_hz):
    i0 = int(round(start / dt))
    y, x = truth[i0:], estimate[i0:]
    error = x - y
    best = (-np.inf, 0, 0.0)
    max_lag = min(int(round(min(2.0, 0.5 / frequency_hz) / dt)), len(y) // 4)
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            yt, xt = y[:-lag], x[lag:]
        elif lag < 0:
            yt, xt = y[-lag:], x[:lag]
        else:
            yt, xt = y, x
        yt0, xt0 = yt - np.mean(yt), xt - np.mean(xt)
        denom = np.linalg.norm(yt0) * np.linalg.norm(xt0)
        corr = float(np.dot(yt0, xt0) / denom) if denom > 1e-15 else 0.0
        if corr > best[0] + 1e-10 or (abs(corr - best[0]) <= 1e-10 and abs(lag) < abs(best[1])):
            best = (corr, lag, float(np.sqrt(np.mean((xt0 - yt0) ** 2))))
    truth_std = float(np.std(y))
    return {
        "same_time_rmse_m_s": float(np.sqrt(np.mean(error**2))),
        "bias_m_s": float(np.mean(error)),
        "aligned_lag_s_estimate_later_positive": float(best[1] * dt),
        "shape_correlation_after_shift": float(best[0]),
        "shift_aligned_fluctuation_rmse_m_s": float(best[2]),
        "shift_aligned_nrmse_fluctuation": float(best[2] / truth_std) if truth_std > 1e-12 else float("nan"),
        "fluctuation_amplitude_ratio": float(np.std(x) / truth_std) if truth_std > 1e-12 else float("nan"),
    }


def run_case(axis, mean, amp, freq, kind, label, q, cfg, nominal, mismatch):
    fs = cfg["sample_rate_hz"]
    dt = 1.0 / fs
    duration = max(16.0, 6.0 / freq)
    n = int(round(duration * fs))
    t = np.arange(n) * dt
    speed = mean + amp * np.sin(2 * np.pi * freq * t)
    force = drag_force_from_speed(speed, cfg["air_density_kg_m3"], cfg["drag_coefficient"], cfg["projected_area_m2"])
    theta0 = np.arctan(cfg["force_lever_m"] * force[0] / nominal["restoring_n_m_per_rad"])
    states = simulate_hbk_plant(force, nominal, dt, theta0, 0.0, cfg["force_lever_m"], cfg["friction_epsilon_deg_s"])
    angle = states[:, 0]
    max_angle_deg = float(np.max(np.abs(np.rad2deg(angle))))
    static_force = nominal["restoring_n_m_per_rad"] / cfg["force_lever_m"] * np.tan(angle)
    static_speed = signed_wind(static_force, cfg)
    lpf = static_speed.copy()
    pole = np.exp(-2.0 * np.pi * cfg["lpf_hz"][axis] * dt)
    for _ in range(3):
        lpf = signal.lfilter([0.0, 1.0-pole], [1.0, -pole], lpf)
    estimates = {
        "Static": static_speed,
        "LPF": lpf,
        "Static mean 1 s": moving_average_centered(static_speed, 1.0, fs),
        "Static mean 3 s": moving_average_centered(static_speed, 3.0, fs),
    }
    # Initialize with the measured angle and a zero disturbance prior.  The
    # true applied wind force is not supplied to either RTS observer.
    initial = np.array([angle[0], 0.0, 0.0])
    for key, coeff in (("RTS matched", nominal), ("RTS OW-04 100%", mismatch)):
        _, estimate_force = nonlinear_ekf_rts_force(
            angle, coeff, dt, cfg["force_lever_m"], np.deg2rad(cfg["assumed_angle_noise_deg"]),
            q, 0, cfg["friction_epsilon_deg_s"], initial_state=initial,
        )
        estimates[key] = signed_wind(estimate_force, cfg)
    # Score after two seconds, excluding initialization and centered-window edges.
    eval_start = max(3.0, 1.5 / freq)
    rows = []
    for method in METHODS:
        rows.append({"axis":axis,"sweep":kind,"case":label,"mean_m_s":mean,"amplitude_m_s":amp,"frequency_hz":freq,"max_abs_angle_deg":max_angle_deg,"within_mechanical_limit":max_angle_deg <= cfg["mechanical_angle_limit_deg"],"rts_q_n_per_sample":q,"method":method,**metrics(speed, estimates[method],eval_start,dt,freq)})
    return t, speed, estimates, rows, eval_start


def _lpf_coeffs(cutoff, dt):
    pole = np.exp(-2 * np.pi * cutoff * dt)
    return ([0.0, 1.0-pole], [1.0, -pole])


def make_plots(rows, traces):
    OUT.mkdir(parents=True, exist_ok=True)
    # Frequency response: each method and axis; shift-aligned shape error is the primary shape metric.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    for ax, axis in zip(axes, ("IN", "OUT")):
        selected = [r for r in rows if r["axis"] == axis and r["sweep"] == "frequency"]
        for method in METHODS:
            ss = [r for r in selected if r["method"] == method]
            ax.plot([r["frequency_hz"] for r in ss], [r["shift_aligned_nrmse_fluctuation"] for r in ss], marker="o", label=method, color=COLORS[method])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="both", alpha=.25)
        ax.set(title=f"{axis}: frequency sweep", xlabel="Input wind frequency [Hz]")
    axes[0].set_ylabel("Shift-aligned fluctuation NRMSE")
    axes[1].legend(fontsize=8, loc="best")
    fig.savefig(OUT / "frequency_sweep_shape_error.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    for ax, axis in zip(axes, ("IN", "OUT")):
        selected = [r for r in rows if r["axis"] == axis and r["sweep"] == "amplitude"]
        for method in METHODS:
            ss = [r for r in selected if r["method"] == method]
            ax.plot([r["amplitude_m_s"] for r in ss], [r["shift_aligned_nrmse_fluctuation"] for r in ss], marker="o", label=method, color=COLORS[method])
        ax.grid(True, alpha=.25); ax.set(title=f"{axis}: amplitude sweep", xlabel="Wind fluctuation amplitude [m/s]")
    axes[0].set_ylabel("Shift-aligned fluctuation NRMSE")
    axes[1].legend(fontsize=8, loc="best")
    fig.savefig(OUT / "amplitude_sweep_shape_error.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    for col, axis in enumerate(("IN", "OUT")):
        for row, kind in enumerate(("amplitude", "frequency")):
            choices = [item for item in traces if item["axis"] == axis and item["sweep"] == kind]
            item = choices[len(choices)//2]
            t, speed, estimates, _, start = item["data"]
            idx = (t >= start) & (t <= start + min(12.0, t[-1]-start))
            ax = axes[row, col]
            ax.plot(t[idx], speed[idx], color="black", lw=2, label="True wind")
            for method in METHODS:
                ax.plot(t[idx], estimates[method][idx], color=COLORS[method], lw=1, label=method)
            ax.set(title=f"{axis}: {kind} sweep representative", xlabel="Time [s]", ylabel="Wind speed [m/s]")
            ax.grid(alpha=.2)
    axes[0,1].legend(fontsize=7, ncol=2)
    fig.savefig(OUT / "representative_waveforms.png", dpi=180); plt.close(fig)


def main():
    cfg = load_config(); OUT.mkdir(parents=True, exist_ok=True)
    all_rows, traces = [], []
    amp_levels = (0.25, 0.5, 1.0)
    freq_levels = (0.1, 0.25, 0.5, 1.0, 2.0)
    freq_amplitude = 0.25
    mean = 2.0
    for axis in ("IN", "OUT"):
        nominal = select_coefficients(axis, "BALL")
        mismatch = mismatch_coefficients(axis, nominal)
        # Same fixed frequency, varying only amplitude.
        for amp in amp_levels:
            case = run_case(axis, mean, amp, 0.5, "amplitude", f"amp_{amp:g}", cfg["rts_q"][axis], cfg, nominal, mismatch)
            t, speed, est, rows, start = case; all_rows.extend(rows)
            if amp == 0.5: traces.append({"axis":axis,"sweep":"amplitude","data":(t,speed,est,rows,start)})
        # Same mean and amplitude, varying only frequency.
        for freq in freq_levels:
            case = run_case(axis, mean, freq_amplitude, freq, "frequency", f"freq_{freq:g}", cfg["rts_q"][axis], cfg, nominal, mismatch)
            t, speed, est, rows, start = case; all_rows.extend(rows)
            if freq == 0.5: traces.append({"axis":axis,"sweep":"frequency","data":(t,speed,est,rows,start)})
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    settings = {"mean_speed_m_s":mean,"amplitude_sweep_m_s":amp_levels,"amplitude_sweep_frequency_hz":0.5,"frequency_sweep_hz":freq_levels,"frequency_sweep_amplitude_m_s":freq_amplitude,"RTS_order":0,"RTS_q_n_per_sample":cfg["rts_q"],"assumed_angle_noise_deg":cfg["assumed_angle_noise_deg"],"LPF_cutoff_hz":cfg["lpf_hz"],"window_average":"centered moving average after static force-to-speed conversion","OW04_mismatch_corner":SCENARIO,"sensor_noise":"none; deterministic model-isolation study","evaluation":"same-time RMSE, plus lag-optimized fluctuation shape metrics; lag search +/-min(2 s, half input period); initial interval excluded","validity":"maximum plant angle and +/-60 deg limit included per case"}
    (OUT / "settings.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    make_plots(all_rows,traces)
    print(f"Wrote {len(all_rows)} metric rows and 3 figures to {OUT}")


if __name__ == "__main__":
    main()
