"""Plot the three WBS 1.2 frequency sensitivities.

A: exact-sign nonlinear plant integrated in time, force-to-angle fundamental gain.
B: angle measurement tone to estimated-wind sensitivity from the stored RTS run.
C: coefficient-mismatch relative force-gain error from the stored RTS run.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
SIM = HERE.parents[1]
RESULTS = SIM / "results" / "observer_wind" / "ow06_frequency_aware_observer_design"
OW05 = SIM / "results" / "observer_wind" / "ow05_force_gain_frequency_response"
OUT = RESULTS / "ow06_wbs12_sensitivity_plots"
CFG = json.loads((SIM / "config" / "ow03_hbk_observer_tuning.json").read_text(encoding="utf-8"))
COEFFS = json.loads((SIM / "config" / "hbk_model_coefficients.json").read_text(encoding="utf-8"))["records"]
OW05_SETTINGS = json.loads((OW05 / "settings.json").read_text(encoding="utf-8"))
MEAN_WIND = float(OW05_SETTINGS["mean_wind_equivalent_m_s"])
FREQUENCIES = [float(x) for x in OW05_SETTINGS["frequency_hz"]]
AMPLITUDE_RATIOS = [float(x) for x in OW05_SETTINGS["force_amplitude_ratios_A_over_F0"]]
COLORS = {0.10: "#2878B5", 0.25: "#E07B24", 0.35: "#188977"}
DT_DEFAULT = 0.002  # RK4 internal step; 500 Hz, with a 100 Hz fitting/output trace.


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def applied_force(speed):
    return (0.5 * CFG["air_density_kg_m3"] * CFG["drag_coefficient"]
            * CFG["projected_area_m2"] * speed * abs(speed))


def select_coeff(axis):
    return next(x.copy() for x in COEFFS
                if x["axis"] == axis and x["configuration"] == "BALL")


def simulate_physical_case(axis, frequency_hz, amplitude_ratio, dt=DT_DEFAULT):
    """Integrate full nonlinear plant with kinetic Coulomb sign friction.

    The idealized moving-friction term is tau*sign(rate), with sign(0)=0.
    Static friction/sticking is not included. All trig terms remain nonlinear.
    """
    p = select_coeff(axis)
    I = p["inertia_kg_m2"]
    K = p["restoring_n_m_per_rad"]
    b = p["viscous_damping_n_m_s_per_rad"]
    c_q = p["total_quadratic_drag_n_m_s2_per_rad2"]
    tau = p["tau_n_m"]
    L = CFG["force_lever_m"]
    F0 = applied_force(MEAN_WIND)
    AF = amplitude_ratio * F0
    w = 2.0 * math.pi * frequency_hz
    theta0 = math.atan(L * F0 / K)
    duration = max(20.0, 8.0 / frequency_hz)
    output_dt = 0.01
    substeps = int(round(output_dt / dt))
    if substeps < 1 or abs(substeps * dt - output_dt) > 1e-10:
        raise ValueError("dt must divide the 0.01 s output interval")
    n_output = int(round(duration / output_dt)) + 1
    t_out = np.arange(n_output, dtype=float) * output_dt
    angle_out = np.empty(n_output, dtype=float)
    angle = theta0
    rate = 0.0
    angle_out[0] = angle

    def rhs(t, th, om):
        friction = tau if om > 0.0 else (-tau if om < 0.0 else 0.0)
        force = F0 + AF * math.sin(w * t)
        acc = (L * force * math.cos(th) - K * math.sin(th)
               - b * om - c_q * abs(om) * om - friction) / I
        return om, acc

    for k in range(1, n_output):
        t0 = t_out[k - 1]
        for j in range(substeps):
            t = t0 + j * dt
            k1t, k1v = rhs(t, angle, rate)
            k2t, k2v = rhs(t + dt/2, angle + dt*k1t/2, rate + dt*k1v/2)
            k3t, k3v = rhs(t + dt/2, angle + dt*k2t/2, rate + dt*k2v/2)
            k4t, k4v = rhs(t + dt, angle + dt*k3t, rate + dt*k3v)
            angle += dt * (k1t + 2*k2t + 2*k3t + k4t) / 6
            rate += dt * (k1v + 2*k2v + 2*k3v + k4v) / 6
        angle_out[k] = angle

    score_start = max(5.0, 3.0 / frequency_hz)
    mask = t_out >= score_start
    t = t_out[mask]
    y = angle_out[mask] - theta0
    phase = w * t
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones_like(t)))
    fit, *_ = np.linalg.lstsq(design, y, rcond=None)
    amp_rad = float(np.hypot(fit[0], fit[1]))
    phase_deg = float(np.rad2deg(np.arctan2(fit[1], fit[0])))
    max_angle = float(np.max(np.abs(np.rad2deg(angle_out))))
    return {
        "axis": axis, "mean_wind_m_s": MEAN_WIND, "frequency_hz": frequency_hz,
        "force_amplitude_ratio": amplitude_ratio, "force_amplitude_n": AF,
        "angle_fundamental_amplitude_deg": math.degrees(amp_rad),
        "physical_sensitivity_deg_per_n": math.degrees(amp_rad) / AF,
        "angle_fundamental_phase_deg": phase_deg,
        "max_absolute_angle_deg": max_angle,
        "within_60_deg": max_angle <= CFG["mechanical_angle_limit_deg"],
        "integration_dt_s": dt, "friction": "tau*sign(rate), sign(0)=0; no static friction",
    }


def plot_physical(rows):
    ratio = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), sharey=True, layout="constrained")
    for ax, axis, color in zip(axes, ("IN", "OUT"), ("#2878B5", "#188977")):
        subset = sorted((r for r in rows if r["axis"] == axis
                         and math.isclose(float(r["force_amplitude_ratio"]), ratio)),
                        key=lambda r: float(r["frequency_hz"]))
        ax.plot([float(r["frequency_hz"]) for r in subset],
                [float(r["physical_sensitivity_deg_per_n"]) for r in subset],
                marker="o", ms=3, lw=1.7, color=color,
                label=f"Exact sgn, A_F/F0={ratio:g}")
        ax.set_xscale("log")
        ax.set_title(f"{axis} axis")
        ax.set_xlabel("Force frequency [Hz]")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend()
    axes[0].set_ylabel(r"Fundamental angle / force amplitude $|\theta_1|/|F_1|$ [deg/N]")
    fig.suptitle("Physical sensitivity from nonlinear time integration\n2 m/s mean force; full sin/cos dynamics; exact kinetic Coulomb sgn friction")
    fig.savefig(OUT / "physical_sensitivity_vs_frequency.png", dpi=180)
    fig.savefig(OUT / "physical_sensitivity_vs_frequency.svg")
    plt.close(fig)


def plot_noise():
    rows = read_csv(RESULTS / "ow06_rts_angle_noise_transfer.csv")
    fig, ax = plt.subplots(figsize=(8.6, 5.0), layout="constrained")
    for axis, color in (("IN", COLORS[0.10]), ("OUT", COLORS[0.35])):
        subset = sorted((r for r in rows if r["axis"] == axis), key=lambda r: float(r["frequency_hz"]))
        ax.plot([float(r["frequency_hz"]) for r in subset],
                [float(r["wind_amplitude_per_angle_error_m_s_per_deg"]) for r in subset],
                marker="o", lw=1.8, color=color,
                label=f"{axis} (q={float(subset[0]['q_n_per_sample']):g} N/sample)")
    ax.set_xscale("log")
    ax.set_xlabel("Angle-noise frequency [Hz]")
    ax.set_ylabel("Estimated-wind amplitude / angle-error amplitude [m/s/deg]")
    ax.set_title("Noise sensitivity: angle measurement tone to estimated wind")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend()
    fig.suptitle("Noise sensitivity: angle tone to estimated wind\n2 m/s mean; 0.02 deg tone; nonlinear 3-state RTS at 100 Hz; assumed model/noise settings")
    fig.savefig(OUT / "noise_sensitivity_vs_frequency.png", dpi=180)
    fig.savefig(OUT / "noise_sensitivity_vs_frequency.svg")
    plt.close(fig)


def plot_coefficient_error():
    rows = read_csv(OW05 / "force_gain_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), sharey=True, layout="constrained")
    out_rows = []
    for ax, axis in zip(axes, ("IN", "OUT")):
        for ratio in AMPLITUDE_RATIOS:
            subset = sorted((r for r in rows if r["axis"] == axis
                             and math.isclose(float(r["amplitude_ratio_A_over_F0"]), ratio)
                             and r["method"] == "RTS OW-04 100%"),
                            key=lambda r: float(r["frequency_hz"]))
            freq = [float(r["frequency_hz"]) for r in subset]
            error = [100 * abs(float(r["fundamental_gain_estimated_over_true"]) - 1.0)
                     for r in subset]
            ax.plot(freq, error, marker="o", ms=3, lw=1.6, color=COLORS[ratio],
                    label=f"A_F/F0={ratio:g}")
            for r, e in zip(subset, error):
                out_rows.append({"axis": axis, "frequency_hz": r["frequency_hz"],
                                 "force_amplitude_ratio": ratio,
                                 "relative_estimated_force_gain_error_percent": e,
                                 "observer_case": r["method"]})
        ax.set_xscale("log")
        ax.set_title(f"{axis} axis")
        ax.set_xlabel("Input-force frequency [Hz]")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(title="Force amplitude")
    axes[0].set_ylabel("Relative estimated-force gain error [%]")
    fig.suptitle("Coefficient-error sensitivity: OW-04 mismatch corner\n100% coefficient-mismatch case; error = 100×|estimated/applied force gain − 1|; RTS simulation")
    fig.savefig(OUT / "coefficient_error_sensitivity_vs_frequency.png", dpi=180)
    fig.savefig(OUT / "coefficient_error_sensitivity_vs_frequency.svg")
    plt.close(fig)
    write_csv(OUT / "coefficient_error_sensitivity_plot_data.csv", out_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    physical = []
    for axis in ("IN", "OUT"):
        for ratio in AMPLITUDE_RATIOS:
            for freq in FREQUENCIES:
                physical.append(simulate_physical_case(axis, freq, ratio))
    write_csv(OUT / "physical_sensitivity_plot_data.csv", physical)
    plot_physical(physical)
    plot_noise()
    plot_coefficient_error()
    print(f"Wrote {len(physical)} exact-sign physical-response cases and three sensitivity figures to {OUT}")

if __name__ == "__main__":
    main()
