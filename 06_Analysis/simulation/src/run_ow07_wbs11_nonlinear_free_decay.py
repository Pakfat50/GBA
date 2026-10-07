#!/usr/bin/env python3
"""Nonlinear free-decay screening for OW-07 WBS 1.1 candidate hardware sets."""
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
G = 9.80665
BALL_M0 = 0.0039
BALL_R0 = -0.174  # signed arm under the existing restoring-torque convention
BALL_D0 = 0.100
BALLAST_M0 = 0.05072
BALLAST_R0 = 0.055
STATIC_RATIOS = (1.0, 1.5, 2.0)
INITIAL_ANGLE_DEG = 60.0

# WBS 1.1 screen. Ball mass follows the same-density D^3 approximation;
# ballast masses and radii are the nominal exchange/placement scenarios.
CANDIDATES = {
    "P0": {"D_m": 0.070, "ballast_kg": 0.05072, "ballast_r_m": 0.055},
    "P1": {"D_m": 0.068, "ballast_kg": 0.047832, "ballast_r_m": 0.055},
    "P2": {"D_m": 0.065, "ballast_kg": 0.05072, "ballast_r_m": 0.050},
    "P3": {"D_m": 0.055, "ballast_kg": 0.035874, "ballast_r_m": 0.055},
    "P4": {"D_m": 0.060, "ballast_kg": 0.05072, "ballast_r_m": 0.045},
    # Extreme downsizing case: OUT K is set near the 8 m/s, 60 deg angle-limit floor.
    "P5": {"D_m": 0.012, "ballast_kg": 0.012963855238988241, "ballast_r_m": 0.055},
}


def load_coefficients():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    return {r["axis"]: r for r in data["records"] if r["configuration"] == "BALL"}


def candidate_parameters(axis_record, candidate):
    d = candidate["D_m"]
    m_ball = BALL_M0 * (d / BALL_D0) ** 3
    r_ball = BALL_R0
    m_weight = candidate["ballast_kg"]
    r_weight = candidate["ballast_r_m"]
    # Solid-sphere centroid inertia plus parallel-axis term; ballast is treated as a point mass.
    I = axis_record["inertia_kg_m2"]
    I += m_ball * r_ball**2 + 0.4 * m_ball * (d / 2) ** 2
    I -= BALL_M0 * BALL_R0**2 + 0.4 * BALL_M0 * (BALL_D0 / 2) ** 2
    I += m_weight * r_weight**2 - BALLAST_M0 * BALLAST_R0**2
    K = axis_record["restoring_n_m_per_rad"]
    K += G * r_ball * (m_ball - BALL_M0)
    K += G * (m_weight * r_weight - BALLAST_M0 * BALLAST_R0)
    c = axis_record["rod_quadratic_drag_n_m_s2_per_rad2"]
    c += axis_record["ball_quadratic_drag_n_m_s2_per_rad2"] * (d / BALL_D0) ** 2
    tau = axis_record["tau_n_m"]
    return {"I": I, "K": K, "c": c, "tau_dyn": tau,
            "ball_mass_kg": m_ball, "D_m": d, "ballast_kg": m_weight,
            "ballast_r_m": r_weight}


def half_cycle(I, K, c, tau_dyn, theta, t_start, max_step=0.001):
    direction = -1.0 if theta > 0 else 1.0
    y0 = [theta, direction * 1e-12]

    def rhs(_t, y):
        angle, omega = y
        return [omega, (-K * math.sin(angle) - c * abs(omega) * omega
                        - tau_dyn * direction) / I]

    def turn(_t, y):
        return y[1]

    turn.terminal = True
    turn.direction = 1 if direction < 0 else -1
    sol = solve_ivp(rhs, (t_start, t_start + 30.0), y0, events=turn,
                    rtol=2e-10, atol=1e-12, max_step=max_step, dense_output=True)
    if not len(sol.t_events[0]):
        return sol, float(sol.t[-1]), float(sol.y[0, -1]), False
    return sol, float(sol.t_events[0][0]), float(sol.y_events[0][0][0]), True


def free_decay(I, K, c, tau_dyn, tau_static, theta0, max_step=0.001):
    theta = theta0
    t = 0.0
    extrema = [(0.0, theta0)]
    segments = []
    half_count = 0
    # At rest, static friction balances the spring moment until its threshold is exceeded.
    if abs(K * math.sin(theta)) <= tau_static:
        return {"status": "stuck_initial", "period_s": None, "next_peak_deg": None,
                "stop_angle_deg": math.degrees(theta), "extrema": extrema,
                "segments": segments, "stop_time_s": 0.0}
    for _ in range(20):
        sol, t, theta, turned = half_cycle(I, K, c, tau_dyn, theta, t, max_step)
        segments.append(sol)
        if not turned:
            return {"status": "monotonic_no_turn", "period_s": None,
                    "next_peak_deg": None, "stop_angle_deg": math.degrees(theta),
                    "extrema": extrema, "segments": segments, "stop_time_s": t}
        extrema.append((t, theta))
        half_count += 1
        if abs(K * math.sin(theta)) <= tau_static:
            return {"status": "stuck_after_half_cycle", "period_s": None,
                    "next_peak_deg": None, "stop_angle_deg": math.degrees(theta),
                    "extrema": extrema, "segments": segments, "stop_time_s": t}
        if theta > 0 and half_count >= 2:
            return {"status": "oscillatory", "period_s": t,
                    "next_peak_deg": math.degrees(theta), "stop_angle_deg": None,
                    "extrema": extrema, "segments": segments, "stop_time_s": None}
    raise RuntimeError("Cycle classification did not converge")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows([{k: ("NA" if v is None else v) for k, v in row.items()}
                           for row in rows])


def result_rows(parameters):
    rows = []
    for name, cand in CANDIDATES.items():
        for axis in ("IN", "OUT"):
            p = parameters[name, axis]
            for ratio in STATIC_RATIOS:
                r = free_decay(p["I"], p["K"], p["c"], p["tau_dyn"],
                               p["tau_dyn"] * ratio, math.radians(INITIAL_ANGLE_DEG))
                rows.append({
                        "candidate": name, "axis": axis, "initial_angle_deg": INITIAL_ANGLE_DEG,
                        "tau_static_over_tau_dynamic": ratio, "I_kg_m2": p["I"],
                        "K_Nm_per_rad": p["K"], "c_quadratic_Nm_s2_per_rad2": p["c"],
                        "tau_dynamic_Nm": p["tau_dyn"], "status": r["status"],
                        "period_s": r["period_s"], "next_same_side_peak_deg": r["next_peak_deg"],
                        "amplitude_retention_pct": (100 * r["next_peak_deg"] / INITIAL_ANGLE_DEG
                                                      if r["next_peak_deg"] is not None else None),
                        "stop_angle_deg": r["stop_angle_deg"], "stop_time_s": r["stop_time_s"],
                })
    return rows


def trajectory(result, duration_s):
    t_values, theta_values = [], []
    for sol in result["segments"]:
        t0, t1 = sol.t[0], sol.t[-1]
        grid = np.linspace(t0, t1, max(2, int((t1 - t0) / 0.001) + 1))
        y = sol.sol(grid)
        t_values.extend(grid.tolist())
        theta_values.extend(np.degrees(y[0]).tolist())
    end = result["stop_time_s"]
    if end is not None:
        t_values.extend([end, duration_s])
        theta_values.extend([result["stop_angle_deg"], result["stop_angle_deg"]])
    return np.asarray(t_values), np.asarray(theta_values)


def make_plots(parameters):
    colors = dict(zip(CANDIDATES, plt.cm.tab10.colors[:len(CANDIDATES)]))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    for ax, axis in zip(axes, ("IN", "OUT")):
        for name in CANDIDATES:
            p = parameters[name, axis]
            r = free_decay(p["I"], p["K"], p["c"], p["tau_dyn"],
                           2 * p["tau_dyn"], math.radians(INITIAL_ANGLE_DEG))
            t, th = trajectory(r, r["period_s"] or 1.0)
            ax.plot(t, th, label=name, color=colors[name], linewidth=1.6)
        ax.set_title(f"{axis}: release from +60°, τs/τdyn = 2")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle [deg]")
        ax.grid(True, alpha=.3)
        ax.legend(ncol=5, loc="upper right")
    fig.suptitle("OW-07 WBS1.1 nonlinear free decay from the angle limit")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wbs11_free_decay_60deg.png", dpi=180)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_coefficients()
    parameters = {(name, axis): candidate_parameters(records[axis], cand)
                  for name, cand in CANDIDATES.items() for axis in ("IN", "OUT")}
    rows = result_rows(parameters)
    write_csv(OUT_DIR / "wbs11_nonlinear_free_decay.csv", rows)
    parameter_rows = []
    for (name, axis), p in parameters.items():
        parameter_rows.append({"candidate": name, "axis": axis, **p})
    write_csv(OUT_DIR / "wbs11_candidate_dynamic_parameters.csv", parameter_rows)
    make_plots(parameters)
    print(f"Wrote {len(rows)} simulation cases to {OUT_DIR}")
    stuck = sum(r["status"] != "oscillatory" for r in rows)
    print(f"60-degree release: {stuck}/{len(rows)} cases stick before a full cycle")


if __name__ == "__main__":
    main()
