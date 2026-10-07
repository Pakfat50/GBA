#!/usr/bin/env python3
"""OW-07 WBS2.2: package envelope and 70 mm sphere under selected damper geometry."""
from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "06_Analysis/simulation/results/observer_wind/ow07_hardware_bandwidth_design"
WBS1_PATH = Path(__file__).resolve().parent / "run_ow07_wbs11_nonlinear_free_decay.py"
WBS21_PATH = Path(__file__).resolve().parent / "run_ow07_wbs21_grease_damper_free_decay.py"
SELECTED_SHAFT_DIAMETER_MM = 8.0
SELECTED_GAP_MM = 1.0
SELECTED_LENGTH_MM = 30.0
AIR_DENSITY = 1.225
CD_SPHERE = 0.544
BALL_ARM_M = 0.174
U_MAX = 8.0
BALL_DIAMETERS_M = (0.100, 0.070)
STATIC_RATIOS = (1.0, 1.5, 2.0)


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def wind_torque(diameter_m):
    area = math.pi * diameter_m**2 / 4.0
    return 0.5 * AIR_DENSITY * CD_SPHERE * area * BALL_ARM_M * U_MAX**2


def static_angle_deg(torque, stiffness):
    ratio = torque / stiffness
    return math.degrees(math.asin(ratio)) if 0 <= ratio <= 1 else None


def critical_wind_speed(diameter_m, tau_static):
    coeff = 0.5 * AIR_DENSITY * CD_SPHERE * math.pi * diameter_m**2 / 4.0 * BALL_ARM_M
    return math.sqrt(tau_static / coeff)


def main():
    wbs1 = import_module("ow07_wbs1_core", WBS1_PATH)
    wbs21 = import_module("ow07_wbs21_core", WBS21_PATH)
    records = wbs1.load_coefficients()
    ballast = wbs1.CANDIDATES["P0"]
    b_eq = wbs21.equivalent_b(
        SELECTED_LENGTH_MM, SELECTED_SHAFT_DIAMETER_MM, SELECTED_GAP_MM)

    rows = []
    candidate_records = {}
    for diameter_m in BALL_DIAMETERS_M:
        label = "current_100mm" if diameter_m == 0.100 else "P0_70mm"
        candidate = dict(ballast, D_m=diameter_m)
        for axis in ("IN", "OUT"):
            params = wbs1.candidate_parameters(records[axis], candidate)
            rec = records[axis].copy()
            rec["inertia_kg_m2"] = params["I"]
            rec["restoring_n_m_per_rad"] = params["K"]
            rec["total_quadratic_drag_n_m_s2_per_rad2"] = params["c"]
            rec["tau_n_m"] = params["tau_dyn"]
            candidate_records[label, axis] = rec
            torque_8 = wind_torque(diameter_m)
            theta_static = static_angle_deg(torque_8, params["K"])
            mass_g = wbs1.BALL_M0 * (diameter_m / wbs1.BALL_D0)**3 * 1000.0
            bcrit = 2 * math.sqrt(params["I"] * params["K"])
            for ratio in STATIC_RATIOS:
                result = wbs21.free_decay(rec, b_eq, ratio)
                rows.append({
                    "candidate": label,
                    "shaft_diameter_mm": SELECTED_SHAFT_DIAMETER_MM,
                    "radial_gap_mm": SELECTED_GAP_MM,
                    "contact_length_mm": SELECTED_LENGTH_MM,
                    "sphere_diameter_mm": diameter_m * 1000,
                    "sphere_mass_g_same_density": mass_g,
                    "ball_arm_mm": BALL_ARM_M * 1000,
                    "ballast_mass_g": ballast["ballast_kg"] * 1000,
                    "ballast_arm_mm": ballast["ballast_r_m"] * 1000,
                    "axis": axis,
                    "I_kg_m2": params["I"],
                    "K_Nm_per_rad": params["K"],
                    "b_eq_Nm_s_per_rad": b_eq,
                    "b_critical_Nm_s_per_rad": bcrit,
                    "zeta_equivalent": b_eq / bcrit,
                    "wind_torque_at_8m_s_Nm": torque_8,
                    "static_angle_at_8m_s_deg": "no_equilibrium" if theta_static is None else theta_static,
                    "tau_static_over_tau_dynamic": ratio,
                    "breakaway_wind_m_s": critical_wind_speed(diameter_m, params["tau_dyn"] * ratio),
                    "free_decay_status": result["status"],
                    "zero_crossings": result["zero_crossings"],
                    "first_opposite_peak_deg": result["first_opposite_peak_deg"],
                    "stop_angle_deg": result["stop_angle_deg"],
                    "stop_time_s": result["stop_time_s"],
                })

    out_csv = OUT_DIR / "wbs22_selected_candidate.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows([{k: ("NA" if v is None else v) for k, v in row.items()}
                          for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {"current_100mm": "#666666", "P0_70mm": "#0072B2"}
    for ax, axis in zip(axes, ("IN", "OUT")):
        for label, name in (("current_100mm", "100 mm sphere"), ("P0_70mm", "70 mm sphere")):
            rec = candidate_records[label, axis]
            result = wbs21.free_decay(rec, b_eq, 2.0)
            t, theta = wbs21.trajectory(result)
            zeta = b_eq / (2 * math.sqrt(rec["inertia_kg_m2"]
                                         * rec["restoring_n_m_per_rad"]))
            ax.plot(t, theta, label=f"{name} (ζeq={zeta:.3f})",
                    color=colors[label], linewidth=2)
        ax.axhline(0, color="black", linewidth=.8)
        ax.axhline(60, color="gray", linestyle="--", linewidth=.7)
        ax.set_title(f"{axis}: +60° release, 8 mm shaft / 1 mm gap / 30 mm length")
        ax.set_ylabel("Angle [deg]")
        ax.set_ylim(-65, 65)
        ax.set_xlim(0, 8)
        ax.grid(True, alpha=.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("WBS2.2 selected damper geometry: sphere static-range comparison")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wbs22_selected_candidate_timeseries.png", dpi=160)
    plt.close(fig)

    annular_volume_ml = math.pi * (((8/2 + 1)**2 - (8/2)**2)) * 30 / 1000
    grease_density_g_ml = 1.15
    print(f"b_eq={b_eq*1e3:.3f} mN m s/rad; annular volume={annular_volume_ml:.3f} mL/axis; "
          f"G-331 fill mass={annular_volume_ml*grease_density_g_ml:.3f} g/axis")
    print(f"two shafts: {2*annular_volume_ml:.3f} mL, "
          f"approximately {2*annular_volume_ml*grease_density_g_ml:.3f} g grease")
    for row in rows:
        if row["tau_static_over_tau_dynamic"] == 2.0:
            print(row["candidate"], row["axis"],
                  "I", f"{row['I_kg_m2']:.6e}", "K", f"{row['K_Nm_per_rad']:.6f}",
                  "zeta", f"{row['zeta_equivalent']:.3f}",
                  "angle8", row["static_angle_at_8m_s_deg"],
                  "start", f"{row['breakaway_wind_m_s']:.3f} m/s",
                  "cross", row["zero_crossings"],
                  "stop", row["stop_angle_deg"], row["stop_time_s"])


if __name__ == "__main__":
    main()
