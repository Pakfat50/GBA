#!/usr/bin/env python3
"""OW-07 WBS2.1: nonlinear free decay with a grease shear damper surrogate."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "06_Analysis/simulation/config/hbk_model_coefficients.json"
OUT_DIR = ROOT / "06_Analysis/simulation/results/observer_wind/ow07_hardware_bandwidth_design"

# Shin-Etsu G-331 reference test: 4 mm shaft, 8 mm grease contact length,
# 35 um clearance, 10 rpm running torque 34 x 10^-4 N m.
SHAFT_DIAMETER_MM = 4.0
REFERENCE_CONTACT_MM = 8.0
REFERENCE_TORQUE_NM = 34e-4
REFERENCE_RPM = 10.0
CONTACT_LENGTHS_MM = (0.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
STATIC_RATIOS = (1.0, 1.5, 2.0)
INITIAL_ANGLE_DEG = 60.0
PLOT_DURATION_S = 8.0
MAX_STEP_S = 0.005
MAX_HALF_CYCLE_S = 10.0
TURN_SPEED_TOL = 1e-5


def load_ball_records():
    records = json.loads(CONFIG.read_text(encoding="utf-8"))["records"]
    return {r["axis"]: r for r in records if r["configuration"] == "BALL"}


def equivalent_b(length_mm: float) -> float:
    """Linear b surrogate using the catalog torque at 10 rpm and length scaling."""
    omega = REFERENCE_RPM * 2.0 * math.pi / 60.0
    return (REFERENCE_TORQUE_NM / omega
            * length_mm / REFERENCE_CONTACT_MM)


def half_cycle(I, K, c_quad, tau_dyn, b, theta, t_start):
    direction = -1.0 if theta > 0 else 1.0
    y0 = [theta, direction * 1e-4]

    def rhs(_t, y):
        angle, omega = y
        torque = (K * math.sin(angle) + b * omega
                  + c_quad * abs(omega) * omega + tau_dyn * direction)
        return [omega, -torque / I]

    def turning_point(_t, y):
        return y[1] * direction - TURN_SPEED_TOL

    turning_point.terminal = True
    turning_point.direction = -1
    sol = solve_ivp(rhs, (t_start, t_start + MAX_HALF_CYCLE_S), y0,
                    events=turning_point, rtol=2e-10, atol=1e-12,
                    max_step=MAX_STEP_S, dense_output=True)
    if not len(sol.t_events[0]):
        return sol, float(sol.t[-1]), float(sol.y[0, -1]), False
    return sol, float(sol.t_events[0][0]), float(sol.y_events[0][0][0]), True


def free_decay(record, b, tau_static_ratio):
    I = record["inertia_kg_m2"]
    K = record["restoring_n_m_per_rad"]
    c_quad = record["total_quadratic_drag_n_m_s2_per_rad2"]
    tau_dyn = record["tau_n_m"]
    tau_static = tau_dyn * tau_static_ratio
    theta = math.radians(INITIAL_ANGLE_DEG)
    t = 0.0
    segments = []
    extrema = [(t, theta)]
    status = "max_half_cycles"

    if abs(K * math.sin(theta)) <= tau_static:
        return {"status": "stuck_initial", "segments": segments,
                "extrema": extrema, "stop_time_s": 0.0,
                "stop_angle_deg": math.degrees(theta), "zero_crossings": 0,
                "first_opposite_peak_deg": None}

    first_opposite_peak = None
    zero_crossings = 0
    for _ in range(40):
        previous_theta = theta
        sol, t, theta, turned = half_cycle(
            I, K, c_quad, tau_dyn, b, theta, t)
        segments.append(sol)
        if np.sign(previous_theta) != np.sign(theta):
            zero_crossings += 1
        if not turned:
            status = "monotonic_no_turn"
            break
        extrema.append((t, theta))
        if first_opposite_peak is None and theta < 0:
            first_opposite_peak = math.degrees(theta)
        if abs(K * math.sin(theta)) <= tau_static:
            status = "stuck_after_turn"
            break

    return {"status": status, "segments": segments, "extrema": extrema,
            "stop_time_s": t, "stop_angle_deg": math.degrees(theta),
            "zero_crossings": zero_crossings,
            "first_opposite_peak_deg": first_opposite_peak}


def trajectory(result):
    ts, angles = [], []
    for sol in result["segments"]:
        grid = np.linspace(sol.t[0], sol.t[-1],
                           max(2, int((sol.t[-1] - sol.t[0]) / 0.002) + 1))
        y = sol.sol(grid)
        ts.extend(grid.tolist())
        angles.extend(np.degrees(y[0]).tolist())
    if result["stop_time_s"] is not None:
        ts.extend([result["stop_time_s"], PLOT_DURATION_S])
        angles.extend([result["stop_angle_deg"], result["stop_angle_deg"]])
    return np.asarray(ts), np.asarray(angles)


def write_results(records):
    rows = []
    for length in CONTACT_LENGTHS_MM:
        b = equivalent_b(length)
        for axis, record in records.items():
            I = record["inertia_kg_m2"]
            K = record["restoring_n_m_per_rad"]
            b_critical = 2.0 * math.sqrt(I * K)
            for ratio in STATIC_RATIOS:
                result = free_decay(record, b, ratio)
                rows.append({
                    "shaft_diameter_mm": SHAFT_DIAMETER_MM,
                    "grease_contact_length_mm": length,
                    "grease": "Shin-Etsu G-331",
                    "axis": axis,
                    "tau_static_over_tau_dynamic": ratio,
                    "equivalent_b_Nm_s_per_rad": b,
                    "b_critical_Nm_s_per_rad": b_critical,
                    "zeta_equivalent": b / b_critical,
                    "I_kg_m2": I,
                    "K_Nm_per_rad": K,
                    "status": result["status"],
                    "zero_crossings": result["zero_crossings"],
                    "first_opposite_peak_deg": result["first_opposite_peak_deg"],
                    "stop_angle_deg": result["stop_angle_deg"],
                    "stop_time_s": result["stop_time_s"],
                })
    out = OUT_DIR / "wbs21_grease_damper_free_decay.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows([{k: ("NA" if v is None else v) for k, v in row.items()}
                           for row in rows])
    return rows


def plot_timeseries(records):
    shown_lengths = (0.0, 8.0, 12.0, 20.0, 30.0)
    colors = dict(zip(shown_lengths, plt.cm.viridis(np.linspace(.05, .95, len(shown_lengths)))))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, axis in zip(axes, ("IN", "OUT")):
        for length in shown_lengths:
            b = equivalent_b(length)
            result = free_decay(records[axis], b, 2.0)
            t, theta = trajectory(result)
            ax.plot(t, theta, color=colors[length], linewidth=1.7,
                    label=f"L={length:g} mm (ζeq={b/(2*math.sqrt(records[axis]['inertia_kg_m2']*records[axis]['restoring_n_m_per_rad'])):.2f})")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(60.0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(f"{axis} axis — release from +60°, τs/τdyn=2")
        ax.set_ylabel("Angle [deg]")
        ax.set_ylim(-25, 65)
        ax.grid(True, alpha=.3)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(0, PLOT_DURATION_S)
    fig.suptitle("WBS2.1: G-331 grease damper — 4 mm shaft nonlinear free decay")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wbs21_grease_damper_free_decay.png", dpi=180)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_ball_records()
    rows = write_results(records)
    plot_timeseries(records)
    print(f"Wrote {len(rows)} nonlinear free-decay cases")
    for length in (8.0, 12.0, 20.0, 30.0):
        b = equivalent_b(length)
        print(f"L={length:>4.0f} mm: b_eq={b*1e3:.3f} mN m s/rad")
        for axis, record in records.items():
            r = free_decay(record, b, 2.0)
            print(f"  {axis}: zeta_eq={b/(2*math.sqrt(record['inertia_kg_m2']*record['restoring_n_m_per_rad'])):.3f}, "
                  f"crossings={r['zero_crossings']}, first opposite peak={r['first_opposite_peak_deg']}, "
                  f"stop={r['stop_angle_deg']:.3f} deg, t={r['stop_time_s']:.3f} s")


if __name__ == "__main__":
    main()
