"""Force-domain frequency response of the OW-05 matched/mismatched RTS.

The plant receives a sinusoidal force ripple around the force equivalent of
2 m/s wind. The reported gain is the fitted fundamental amplitude of the RTS
force estimate divided by the applied force amplitude. This is an amplitude-
dependent nonlinear frequency response, not a single LTI Bode transfer.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from hbk_model_coefficients import select_coefficients, simulate_hbk_plant
from hbk_nonlinear_estimators import nonlinear_ekf_rts_force
from wind import drag_force_from_speed

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "ow03_hbk_observer_tuning.json"
OW04_CSV = ROOT / "results" / "observer_wind" / "ow04_coefficient_sensitivity" / "ow04_joint_scenarios.csv"
OUT = ROOT / "results" / "observer_wind" / "ow05_force_gain_frequency_response"
SCENARIO = "joint_mass_high_arm_high_tau-high_c-high"
FREQUENCIES_HZ = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
                  0.925, 0.95, 0.975, 0.98, 0.99, 0.995, 1.0, 1.005,
                  1.01, 1.02, 1.025, 1.05, 1.075, 1.1, 1.2, 1.3, 1.4,
                  1.6, 1.8, 2.0)
AMPLITUDE_RATIOS = (0.1, 0.25, 0.35)
METHODS = ("RTS matched", "RTS OW-04 100%")
COLORS = {0.1: "#2776bc", 0.25: "#e67e22", 0.35: "#159477"}
STYLES = {"RTS matched": "-", "RTS OW-04 100%": "--"}


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def applied_force(speed, cfg):
    return drag_force_from_speed(speed, cfg["air_density_kg_m3"], cfg["drag_coefficient"], cfg["projected_area_m2"])


def mismatch_coefficients(axis, nominal):
    with OW04_CSV.open(encoding="utf-8-sig", newline="") as stream:
        row = next(r for r in csv.DictReader(stream)
                   if r["axis"] == axis and r["scenario"] == SCENARIO)
    result = nominal.copy()
    result["inertia_kg_m2"] = float(row["observer_I_kg_m2"])
    result["restoring_n_m_per_rad"] = float(row["observer_K_n_m_per_rad"])
    result["tau_n_m"] = float(row["observer_tau_n_m"])
    result["ball_quadratic_drag_n_m_s2_per_rad2"] = float(row["observer_c_ball_n_m_s2_per_rad2"])
    result["total_quadratic_drag_n_m_s2_per_rad2"] = result["rod_quadratic_drag_n_m_s2_per_rad2"] + result["ball_quadratic_drag_n_m_s2_per_rad2"]
    return result


def fundamental(signal_values, time_s, frequency_hz):
    phase = 2.0 * np.pi * frequency_hz * time_s
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones_like(phase)))
    coeff, *_ = np.linalg.lstsq(design, signal_values, rcond=None)
    amp = float(np.hypot(coeff[0], coeff[1]))
    phase_deg = float(np.rad2deg(np.arctan2(coeff[1], coeff[0])))
    return amp, phase_deg


def run_case(axis, f_hz, amplitude_ratio, cfg, nominal, mismatched):
    fs = cfg["sample_rate_hz"]
    dt = 1.0 / fs
    # Mean load is the steady aerodynamic force at 2 m/s.  A/F0 stays below
    # 0.5 so the applied force remains positive and all tested plant angles
    # stay within the +/-60 deg hardware range.
    f0 = float(applied_force(np.array([2.0]), cfg)[0])
    amp_f = amplitude_ratio * f0
    duration_s = max(20.0, 8.0 / f_hz)
    count = int(round(duration_s * fs))
    time_s = np.arange(count, dtype=float) * dt
    force = f0 + amp_f * np.sin(2.0 * np.pi * f_hz * time_s)
    theta0 = float(np.arctan(cfg["force_lever_m"] * f0 / nominal["restoring_n_m_per_rad"]))
    angle = simulate_hbk_plant(
        force, nominal, dt, theta0, 0.0, cfg["force_lever_m"], cfg["friction_epsilon_deg_s"]
    )[:, 0]
    max_angle_deg = float(np.max(np.abs(np.rad2deg(angle))))

    # Start both observers at the same mean operating point; neither is given
    # the sinusoidal disturbance.  Initial cycles are dropped before fitting.
    initial_state = np.array([angle[0], 0.0, f0])
    q = {"IN": 3e-5, "OUT": 3e-4}[axis]
    scoring_start_s = max(5.0, 3.0 / f_hz)
    mask = time_s >= scoring_start_s
    rows = []
    for method, coefficients in (("RTS matched", nominal), ("RTS OW-04 100%", mismatched)):
        _, estimate = nonlinear_ekf_rts_force(
            angle, coefficients, dt, cfg["force_lever_m"],
            np.deg2rad(cfg["assumed_angle_noise_deg"]), q, 0,
            cfg["friction_epsilon_deg_s"], initial_state=initial_state,
        )
        estimate_amp, estimate_phase = fundamental(estimate[mask], time_s[mask], f_hz)
        truth_amp, _ = fundamental(force[mask], time_s[mask], f_hz)
        residual = estimate[mask] - np.mean(estimate[mask])
        total_ac_rms_gain = float(np.sqrt(np.mean(residual**2)) / (amp_f / np.sqrt(2.0)))
        rows.append({
            "axis": axis, "frequency_hz": f_hz, "force_mean_n": f0,
            "force_amplitude_n": amp_f, "amplitude_ratio_A_over_F0": amplitude_ratio,
            "method": method, "rts_q_n_per_sample": q,
            "true_force_fundamental_n": truth_amp,
            "estimated_force_fundamental_n": estimate_amp,
            "fundamental_gain_estimated_over_true": estimate_amp / truth_amp,
            "phase_estimate_minus_input_deg": estimate_phase,
            "total_ac_rms_gain": total_ac_rms_gain,
            "max_abs_angle_deg": max_angle_deg,
            "within_60_deg_limit": max_angle_deg <= cfg["mechanical_angle_limit_deg"],
            "scoring_start_s": scoring_start_s,
        })
    return rows


def plot_gain(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True, layout="constrained")
    for ax, axis in zip(axes, ("IN", "OUT")):
        for ratio in AMPLITUDE_RATIOS:
            for method in METHODS:
                subset = sorted((r for r in rows if r.get("axis", r.get("\ufeffaxis")) == axis and
                                 float(r["amplitude_ratio_A_over_F0"]) == ratio and r["method"] == method),
                                key=lambda r: r["frequency_hz"])
                ax.plot([float(r["frequency_hz"]) for r in subset],
                        [float(r["fundamental_gain_estimated_over_true"]) for r in subset],
                        color=COLORS[ratio], linestyle=STYLES[method], marker="o", markersize=3,
                        linewidth=1.5, label=f"A/F0={ratio:g}, {method}")
        ax.axhline(1.0, color="#444444", linestyle=":", linewidth=1)
        ax.axvline(1.0, color="#888888", linestyle="--", linewidth=0.8)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set(title=f"{axis} axis", xlabel="Applied force frequency [Hz]")
    axes[0].set_ylabel("Estimated / applied force amplitude")
    axes[1].legend(fontsize=7.5, ncol=1, loc="best")
    fig.suptitle("OW-05 force-domain frequency response at different force amplitudes")
    fig.savefig(OUT / "estimated_force_gain_vs_frequency.png", dpi=180)
    fig.savefig(OUT / "estimated_force_gain_vs_frequency.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True, layout="constrained")
    for ax, axis in zip(axes, ("IN", "OUT")):
        for ratio in AMPLITUDE_RATIOS:
            for method in METHODS:
                subset = sorted((r for r in rows if r.get("axis", r.get("\ufeffaxis")) == axis and
                                 float(r["amplitude_ratio_A_over_F0"]) == ratio and r["method"] == method),
                                key=lambda r: r["frequency_hz"])
                ax.plot([float(r["frequency_hz"]) for r in subset],
                        [float(r["phase_estimate_minus_input_deg"]) for r in subset],
                        color=COLORS[ratio], linestyle=STYLES[method], marker="o", markersize=3,
                        linewidth=1.5, label=f"A/F0={ratio:g}, {method}")
        ax.axhline(0.0, color="#444444", linestyle=":", linewidth=1)
        ax.axvline(1.0, color="#888888", linestyle="--", linewidth=0.8)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set(title=f"{axis} axis", xlabel="Applied force frequency [Hz]")
    axes[0].set_ylabel("Estimated force phase relative to input [deg]")
    axes[1].legend(fontsize=7.5, ncol=1, loc="best")
    fig.suptitle("OW-05 force-estimate phase response")
    fig.savefig(OUT / "estimated_force_phase_vs_frequency.png", dpi=180)
    plt.close(fig)


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for axis in ("IN", "OUT"):
        nominal = select_coefficients(axis, "BALL")
        mismatched = mismatch_coefficients(axis, nominal)
        for ratio in AMPLITUDE_RATIOS:
            for f_hz in FREQUENCIES_HZ:
                rows.extend(run_case(axis, f_hz, ratio, cfg, nominal, mismatched))
    with (OUT / "force_gain_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = {
        "mean_wind_equivalent_m_s": 2.0,
        "mean_force_n": float(applied_force(np.array([2.0]), cfg)[0]),
        "force_amplitude_ratios_A_over_F0": AMPLITUDE_RATIOS,
        "frequency_hz": FREQUENCIES_HZ,
        "plant_configuration": "BALL; exact nominal OW-03 coefficients",
        "observer_cases": ["matched", f"OW-04 100% mismatch corner: {SCENARIO}"],
        "observer_q_n_per_sample": {"IN": 3e-5, "OUT": 3e-4},
        "angle_measurement_noise": "none added to simulated angle; RTS assumed sigma=0.02 deg",
        "force_gain_definition": "fitted fundamental amplitude of estimated force divided by fitted fundamental amplitude of applied force after startup cycles",
        "input": "F(t)=F0+A_F sin(2*pi*f*t); direct force-domain test, not sinusoidal wind-speed input",
        "mechanical_limit": "maximum angle reported; selected amplitudes intended to remain within +/-60 deg",
        "nonlinear_response_note": "amplitude-dependent frequency response; not a single LTI Bode transfer function",
    }
    (OUT / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_gain(rows)
    print(f"Wrote {len(rows)} metric rows to {OUT}")


if __name__ == "__main__":
    main()
