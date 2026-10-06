"""Compare exact-sign nonlinear angle response with OW-06 classical LTI model.

Case selection follows the OW-05 mismatched-RTS peak for IN, A/F0=0.35.
The physical plant is driven by the same force tone; its Coulomb torque is
-tau*sign(theta_dot), evaluated directly (no tanh smoothing). The comparison
LTI model uses the OW-06 local small-signal damping c_local=b+tau/epsilon.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

HERE = Path(__file__).resolve()
SIM = HERE.parents[1]
OUT = SIM / "results" / "observer_wind" / "ow06_frequency_aware_observer_design"
CFG = json.loads((SIM / "config" / "ow03_hbk_observer_tuning.json").read_text(encoding="utf-8"))
REGISTRY = json.loads((SIM / "config" / "hbk_model_coefficients.json").read_text(encoding="utf-8"))
MEAN_SPEED = 2.0
A_OVER_F0 = 0.35
CASES = {"IN": 0.995, "OUT": 0.980}
DURATION = 30.0
OUTPUT_FS = 1000.0
DT_INTERNAL = float(os.environ.get("OW06_DT_INTERNAL", "5.0e-5"))


def coeff_for(axis):
    return next(r.copy() for r in REGISTRY["records"]
                if r["axis"] == axis and r["configuration"] == "BALL")


def force_at_speed(v):
    return 0.5 * CFG["air_density_kg_m3"] * CFG["drag_coefficient"] * CFG["projected_area_m2"] * v * abs(v)


def fit_fundamental(t, y, f):
    phase = 2 * np.pi * f * t
    mat = np.column_stack([np.sin(phase), np.cos(phase), np.ones_like(t)])
    fit, *_ = np.linalg.lstsq(mat, y, rcond=None)
    return float(np.hypot(fit[0], fit[1])), float(np.rad2deg(np.arctan2(fit[1], fit[0]))), fit


def run_case(axis, f_hz):
    p = coeff_for(axis)
    I = p["inertia_kg_m2"]
    K = p["restoring_n_m_per_rad"]
    b = p["viscous_damping_n_m_s_per_rad"]
    c_q = p["total_quadratic_drag_n_m_s2_per_rad2"]
    tau = p["tau_n_m"]
    lever = CFG["force_lever_m"]
    F0 = force_at_speed(MEAN_SPEED)
    AF = A_OVER_F0 * F0
    theta0 = math.atan(lever * F0 / K)
    k_eff = K * math.cos(theta0) + lever * F0 * math.sin(theta0)
    c_local = b + tau / math.radians(CFG["friction_epsilon_deg_s"])
    w = 2 * np.pi * f_hz
    duration = DURATION
    t_eval = np.arange(0, duration + 0.5 / OUTPUT_FS, 1 / OUTPUT_FS)
    # Fixed-step RK4. The discontinuous Coulomb law is evaluated as sign(rate)
    # at each RK stage. Internal dt is ten times finer than the exported trace.
    substeps = int(round((1.0 / OUTPUT_FS) / DT_INTERNAL))
    if abs(substeps * DT_INTERNAL - 1.0 / OUTPUT_FS) > 1e-12:
        raise ValueError("DT_INTERNAL must divide the output sample period")
    def integrate(use_smooth_friction):
        state = np.array([theta0, 0.0], dtype=float)
        trace = np.empty(len(t_eval), dtype=float)
        trace[0] = state[0]

        def rhs(t, y):
            theta, rate = y
            friction = (tau * math.tanh(rate / math.radians(CFG["friction_epsilon_deg_s"]))
                        if use_smooth_friction else tau * np.sign(rate))
            accel = (lever * (F0 + AF * math.sin(w * t)) * math.cos(theta)
                     - K * math.sin(theta) - b * rate - c_q * abs(rate) * rate - friction) / I
            return np.array([rate, accel])

        dt = DT_INTERNAL
        for j in range(1, len(t_eval)):
            t0 = t_eval[j - 1]
            for n in range(substeps):
                tn = t0 + n * dt
                k1 = rhs(tn, state)
                k2 = rhs(tn + dt / 2, state + dt * k1 / 2)
                k3 = rhs(tn + dt / 2, state + dt * k2 / 2)
                k4 = rhs(tn + dt, state + dt * k3)
                state += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            trace[j] = state[0]
        return trace

    # Both nonlinear trajectories retain the exact trigonometric plant terms;
    # they differ only in Coulomb friction: sign() versus the prior tanh law.
    theta_exact = integrate(use_smooth_friction=False)
    theta_tanh = integrate(use_smooth_friction=True)

    # Classical OW-06 LTI small-signal response around the same wind equilibrium.
    dF = AF
    numerator = lever * math.cos(theta0) * dF
    den = complex(k_eff - I * w * w, c_local * w)
    amp = abs(numerator / den)
    phase = -np.angle(den)
    theta_lti = theta0 + amp * np.sin(w * t_eval + phase)

    # Compare only the final six periods, by fitting each waveform's fundamental.
    start = duration - 6.0 / f_hz
    mask = t_eval >= start
    amp_exact, phase_exact_deg, fit_exact = fit_fundamental(t_eval[mask], theta_exact[mask] - theta0, f_hz)
    amp_lti_fit, phase_lti_deg, _ = fit_fundamental(t_eval[mask], theta_lti[mask] - theta0, f_hz)
    amp_tanh, phase_tanh_deg, _ = fit_fundamental(t_eval[mask], theta_tanh[mask] - theta0, f_hz)
    max_abs_deg = float(np.max(np.abs(np.rad2deg(theta_exact))))
    if max_abs_deg > CFG["mechanical_angle_limit_deg"]:
        raise RuntimeError(f"{axis} angle limit exceeded: {max_abs_deg:.3f} deg")
    metrics = {
        "axis": axis, "frequency_hz": f_hz, "mean_wind_equivalent_m_s": MEAN_SPEED,
        "mean_force_n": F0, "force_amplitude_n": AF, "force_amplitude_ratio": A_OVER_F0,
        "equilibrium_angle_deg": math.degrees(theta0), "exact_sgn_fundamental_angle_amplitude_deg": math.degrees(amp_exact),
        "exact_sgn_fundamental_phase_deg": phase_exact_deg,
        "nonlinear_tanh_fundamental_angle_amplitude_deg": math.degrees(amp_tanh),
        "nonlinear_tanh_fundamental_phase_deg": phase_tanh_deg,
        "classical_lti_fundamental_angle_amplitude_deg": math.degrees(amp_lti_fit),
        "classical_lti_fundamental_phase_deg": phase_lti_deg,
        "classical_lti_dc_damping_ratio": c_local / (2 * math.sqrt(I * k_eff)),
        "exact_sgn_max_absolute_angle_deg": max_abs_deg,
        "friction_model": "tau*sign(theta_dot), numpy sign(0)=0; no tanh smoothing",
        "comparison_window_s": f"{start:.6f} to {duration:.6f}",
    }
    return t_eval, theta_exact, theta_tanh, theta_lti, theta0, metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), layout="constrained")
    rows = []
    for ax, axis in zip(axes, ("IN", "OUT")):
        t, exact, tanh, lti, theta0, m = run_case(axis, CASES[axis])
        rows.append(m)
        # Plot the final six periods, angle deviation from equilibrium.
        start = DURATION - 6.0 / CASES[axis]
        mask = t >= start
        ax.plot(t[mask], np.rad2deg(exact[mask] - theta0), color="#1565c0", lw=1.5,
                label="Nonlinear exact sgn friction")
        ax.plot(t[mask], np.rad2deg(tanh[mask] - theta0), color="#159477", lw=1.2, ls=":",
                label="Nonlinear tanh friction")
        ax.plot(t[mask], np.rad2deg(lti[mask] - theta0), color="#e65100", lw=1.5, ls="--",
                label="Classical LTI: c=b+τ/ε")
        ax.set_title(f"{axis} axis, f={CASES[axis]:.3f} Hz, A_F/F0=0.35")
        ax.set_ylabel("Angle deviation from equilibrium [deg]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        zoom = inset_axes(ax, width="32%", height="42%", loc="lower right", borderpad=1.5)
        zoom.plot(t[mask], np.rad2deg(lti[mask] - theta0), color="#e65100", lw=1.25, ls="--")
        zoom.set_title("Classical LTI detail", fontsize=7)
        zoom.set_ylim(-1.0 if axis == "IN" else -0.5, 1.0 if axis == "IN" else 0.5)
        zoom.grid(True, alpha=0.25)
        zoom.tick_params(labelsize=6)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Exact-sign nonlinear plant, tanh nonlinear plant, and classical LTI\nSame input; final six cycles")
    fig.savefig(OUT / "ow06_exact_sgn_vs_classical_lti_peak.png", dpi=180)
    fig.savefig(OUT / "ow06_exact_sgn_vs_classical_lti_peak.svg")
    plt.close(fig)
    with (OUT / "ow06_exact_sgn_vs_classical_lti_peak.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0]))
        wri.writeheader(); wri.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print("Wrote plot and metrics to", OUT)

if __name__ == "__main__":
    main()
