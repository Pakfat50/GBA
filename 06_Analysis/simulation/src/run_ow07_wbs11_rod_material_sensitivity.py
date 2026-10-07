#!/usr/bin/env python3
"""No-damper rod-material sensitivity for the extreme P5 hardware case."""
from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parent / "run_ow07_wbs11_nonlinear_free_decay.py"
OUT = ROOT / "06_Analysis/simulation/results/observer_wind/ow07_hardware_bandwidth_design"
RHO_AL = 2700.0
BASE_ROD_LENGTH_M = 0.229
OD_M = 0.005
ID_M = 0.004
BALL_ARM_M = 0.174
BALL_D_M = 0.012
END_CLEARANCE_M = 0.005
ROD_LENGTH_M = BALL_ARM_M + BALL_D_M / 2 + END_CLEARANCE_M
BALLAST_ARM_M = 0.055
P5_OUT_K_NM = 0.0005
STATIC_RATIO = 2.0
MATERIALS = {"Aluminum (baseline)": RHO_AL, "CFRP": 1600.0, "POM-C": 1410.0}


def load_core():
    spec = importlib.util.spec_from_file_location("free_decay_core", SRC)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def main():
    core = load_core()
    records = core.load_coefficients()
    base = {axis: core.candidate_parameters(records[axis], core.CANDIDATES["P5"])
            for axis in ("IN", "OUT")}
    area = math.pi / 4 * (OD_M**2 - ID_M**2)
    rod_volume = area * ROD_LENGTH_M
    base_rod_volume = area * BASE_ROD_LENGTH_M
    # Existing geometry/sign convention treats the 229 mm tube as lying on the
    # ball side, so its gravitational contribution is destabilizing (negative K).
    rows = []
    results = {}
    for material, rho in MATERIALS.items():
        rod_mass = rho * rod_volume
        # Compare the shortened, selected-material tube with the current
        # 229 mm aluminum tube already included in the fitted baseline I and K.
        delta_I_rod = (rho * rod_volume * ROD_LENGTH_M**2 / 3
                       - RHO_AL * base_rod_volume * BASE_ROD_LENGTH_M**2 / 3)
        # The tube is on the destabilizing ball side (negative K contribution).
        # A positive change recovers K and is compensated by reducing ballast.
        delta_K_recovered = (RHO_AL * base_rod_volume * 9.80665 * BASE_ROD_LENGTH_M / 2
                             - rho * rod_volume * 9.80665 * ROD_LENGTH_M / 2)
        ballast_delta = delta_K_recovered / (9.80665 * BALLAST_ARM_M)
        ballast_mass = core.CANDIDATES["P5"]["ballast_kg"] - ballast_delta
        results[material] = {}
        for axis in ("IN", "OUT"):
            p = base[axis]
            inertia = p["I"] + delta_I_rod - ballast_delta * BALLAST_ARM_M**2
            stiffness = p["K"]  # ballast reduction keeps K_OUT at 0.500 mN m/rad
            decay = core.free_decay(inertia, stiffness, p["c"], p["tau_dyn"],
                                    STATIC_RATIO * p["tau_dyn"], math.radians(60))
            results[material][axis] = (inertia, stiffness, p, decay)
            rows.append({
                "material": material, "rod_density_kg_m3": rho,
                "rod_mass_g": rod_mass * 1000,
                "ball_diameter_mm": BALL_D_M * 1000,
                "ball_center_arm_mm": BALL_ARM_M * 1000,
                "rod_length_mm": ROD_LENGTH_M * 1000,
                "ball_mass_g_same_density": p["ball_mass_kg"] * 1000,
                "ballast_mass_g": ballast_mass * 1000, "ballast_arm_mm": BALLAST_ARM_M * 1000,
                "axis": axis, "I_kg_m2": inertia, "K_Nm_per_rad": stiffness,
                "tau_dynamic_Nm": p["tau_dyn"], "tau_static_ratio": STATIC_RATIO,
                "status": decay["status"], "period_s": decay["period_s"],
                "next_same_side_peak_deg": decay["next_peak_deg"],
                "stop_angle_deg": decay["stop_angle_deg"],
            })
    path = OUT / "wbs11_rod_material_sensitivity.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows([{k: ("NA" if v is None else v) for k, v in row.items()}
                     for row in rows])

    colors = {"Aluminum (baseline)": "#1f77b4", "CFRP": "#ff7f0e", "POM-C": "#2ca02c"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    for ax, axis in zip(axes, ("IN", "OUT")):
        for material in MATERIALS:
            inertia, stiffness, p, decay = results[material][axis]
            ts, ys = [], []
            for sol in decay["segments"]:
                grid = np.linspace(sol.t[0], sol.t[-1], max(2, int((sol.t[-1]-sol.t[0])/.001)+1))
                y = sol.sol(grid)
                ts.extend(grid.tolist())
                ys.extend(np.degrees(y[0]).tolist())
            if decay["stop_time_s"] is not None:
                t_stop = decay["stop_time_s"]
                theta_stop = decay["stop_angle_deg"]
                ts.extend([t_stop, max(2.5, t_stop)])
                ys.extend([theta_stop, theta_stop])
            ax.plot(ts, ys, label=material, color=colors[material], linewidth=1.7)
        ax.set_title(f"{axis}: release from +60°, no damper, static friction = 2τdyn")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle [deg]")
        ax.grid(True, alpha=.3)
        ax.legend()
    fig.suptitle("OW-07 WBS1.1: rod material sensitivity, P5 geometry")
    fig.tight_layout()
    fig.savefig(OUT / "wbs11_rod_material_sensitivity.png", dpi=180)
    plt.close(fig)
    print(path)
    print(OUT / "wbs11_rod_material_sensitivity.png")


if __name__ == "__main__":
    main()
