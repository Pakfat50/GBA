#!/usr/bin/env python3
"""No-damper geometry scan with ball/ballast arm ratio and rod length variable."""
from __future__ import annotations

import csv
import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
CORE_PATH = Path(__file__).resolve().parent / "run_ow07_wbs11_nonlinear_free_decay.py"
OUT = ROOT / "06_Analysis/simulation/results/observer_wind/ow07_hardware_bandwidth_design"
G = 9.80665
RHO_AIR = 1.225
CD = 0.544
BASE_D = 0.100
BASE_BALL_M = 0.0039
BASE_BALL_R = 0.174
BASE_WEIGHT_M = 0.05072
BASE_WEIGHT_R = 0.055
BASE_ROD_L = 0.229
BASE_ROD_RHO = 2700.0
OD = 0.005
ID = 0.004
END_CLEARANCE = 0.005
STATIC_RATIO = 2.0
ANGLE_LIMIT = math.radians(60)
MATERIALS = {"CFRP": 1600.0, "POM-C": 1410.0}
BALL_DIAMETERS = (0.008, 0.010, 0.012, 0.016, 0.020, 0.025, 0.030, 0.040)
BALL_ARMS = (0.050, 0.070, 0.100, 0.120, 0.150, 0.174, 0.220, 0.300, 0.400, 0.500)
WEIGHT_ARMS = (0.020, 0.040, 0.055, 0.070)
K_MARGIN = 1.05


def load_core():
    spec = importlib.util.spec_from_file_location("free_decay_core", CORE_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def main():
    core = load_core()
    coeff = core.load_coefficients()
    tube_area = math.pi / 4 * (OD**2 - ID**2)
    base_rod_m = BASE_ROD_RHO * tube_area * BASE_ROD_L
    # Actual wind torque from a sphere: 0.5*rho*Cd*A*arm*U^2.
    bw_base = 0.5 * RHO_AIR * CD * (math.pi * BASE_D**2 / 4) * BASE_BALL_R
    rows = []
    feasible = []
    for D, rball, rweight, (material, rho_rod) in itertools.product(
            BALL_DIAMETERS, BALL_ARMS, WEIGHT_ARMS, MATERIALS.items()):
        mball = BASE_BALL_M * (D / BASE_D) ** 3
        Lrod = rball + D / 2 + END_CLEARANCE
        mrod = rho_rod * tube_area * Lrod
        # Relative to the current 229 mm aluminum tube, current sphere and weight.
        dK_ball = -G * mball * rball - (-G * BASE_BALL_M * BASE_BALL_R)
        dI_ball = (mball * rball**2 + 0.4 * mball * (D / 2)**2
                   - BASE_BALL_M * BASE_BALL_R**2
                   - 0.4 * BASE_BALL_M * (BASE_D / 2)**2)
        dK_rod = -G * mrod * (Lrod / 2) - (-G * base_rod_m * BASE_ROD_L / 2)
        dI_rod = mrod * Lrod**2 / 3 - base_rod_m * BASE_ROD_L**2 / 3
        # 5% restoring-moment margin over the wind torque at 8 m/s and 60 degrees.
        bw = bw_base * (D / BASE_D)**2 * (rball / BASE_BALL_R)
        wind_torque_8 = bw * 8.0**2
        kout_target = K_MARGIN * wind_torque_8 / math.sin(ANGLE_LIMIT)
        rec_out = coeff["OUT"]
        rec_in = coeff["IN"]
        mweight = BASE_WEIGHT_M + (kout_target - rec_out["restoring_n_m_per_rad"]
                                   - dK_ball - dK_rod) / (G * rweight)
        dK_weight = G * (mweight * rweight - BASE_WEIGHT_M * BASE_WEIGHT_R)
        dI_weight = mweight * rweight**2 - BASE_WEIGHT_M * BASE_WEIGHT_R**2
        k_out = rec_out["restoring_n_m_per_rad"] + dK_ball + dK_rod + dK_weight
        k_in = rec_in["restoring_n_m_per_rad"] + dK_ball + dK_rod + dK_weight
        i_out = rec_out["inertia_kg_m2"] + dI_ball + dI_rod + dI_weight
        i_in = rec_in["inertia_kg_m2"] + dI_ball + dI_rod + dI_weight
        exposed_length = max(1e-6, Lrod - D)
        rows_for_candidate = []
        results = {}
        for axis, rec, inertia, stiffness in (("IN", rec_in, i_in, k_in),
                                               ("OUT", rec_out, i_out, k_out)):
            c_rod = rec["rod_quadratic_drag_n_m_s2_per_rad2"] * (exposed_length / 0.129)**4
            c_ball = (rec["ball_quadratic_drag_n_m_s2_per_rad2"] * (D / BASE_D)**2
                      * (rball / BASE_BALL_R)**2)
            c = c_rod + c_ball
            tau = rec["tau_n_m"]
            physically_valid = mweight > 0 and inertia > 0 and stiffness > 0
            if physically_valid:
                decay = core.free_decay(inertia, stiffness, c, tau,
                                        tau * STATIC_RATIO, ANGLE_LIMIT)
            else:
                decay = {"status": "invalid_physical_parameters", "period_s": None,
                         "next_peak_deg": None, "stop_angle_deg": None,
                         "stop_time_s": None, "segments": []}
            results[axis] = decay
            theta_ratio = wind_torque_8 / stiffness if stiffness > 0 else float("inf")
            breakaway = math.sqrt((STATIC_RATIO * tau) / bw) if bw > 0 else float("inf")
            static_angle = (math.degrees(math.asin(theta_ratio))
                            if 0 <= theta_ratio <= 1 else None)
            rows_for_candidate.append({
                "material": material, "ball_diameter_mm": D * 1000,
                "ball_mass_g_same_density": mball * 1000, "ball_arm_mm": rball * 1000,
                "weight_mass_g": mweight * 1000, "weight_arm_mm": rweight * 1000,
                "rod_length_mm": Lrod * 1000, "rod_mass_g": mrod * 1000,
                "axis": axis, "I_kg_m2": inertia, "K_Nm_per_rad": stiffness,
                "c_quadratic_Nm_s2_per_rad2": c, "8m_s_static_angle_deg":
                    static_angle,
                "breakaway_wind_m_s": breakaway, "tau_static_ratio": STATIC_RATIO,
                "free_decay_status": decay["status"], "next_same_side_peak_deg": decay["next_peak_deg"],
                "stop_angle_deg": decay["stop_angle_deg"], "period_s": decay["period_s"],
            })
        for row in rows_for_candidate:
            rows.append(row)
        ok = (mweight > 0 and i_in > 0 and i_out > 0 and k_in > 0 and k_out > 0
              and all(row["8m_s_static_angle_deg"] is not None
                      and row["8m_s_static_angle_deg"] <= 60.0 + 1e-9
                      for row in rows_for_candidate)
              and all(row["breakaway_wind_m_s"] < 8.0 for row in rows_for_candidate))
        if ok:
            no_cross = all(results[a]["status"] in ("stuck_after_half_cycle", "monotonic_no_turn")
                           and results[a]["stop_angle_deg"] >= 0 for a in ("IN", "OUT"))
            worst_turn = max((results[a]["stop_angle_deg"] or 0.0) for a in ("IN", "OUT"))
            feasible.append((no_cross, worst_turn, material, D, rball, rweight, mweight,
                             results["IN"], results["OUT"], i_in, i_out, k_in, k_out))

    path = OUT / "wbs11_arm_ratio_scan.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows([{k: ("NA" if v is None else v) for k, v in row.items()}
                     for row in rows])
    feasible.sort(key=lambda x: (not x[0],
                                 x[7]["next_peak_deg"] if x[7]["next_peak_deg"] is not None else -1.0))
    print(f"wrote {len(rows)} axis cases; feasible common geometries={len(feasible)}")
    for c in feasible[:10]:
        print(f"noncross={c[0]} {c[2]} D={c[3]*1e3:.1f}mm rball={c[4]*1e3:.0f}mm "
              f"rweight={c[5]*1e3:.0f}mm mweight={c[6]*1e3:.2f}g "
              f"I=({c[9]:.3e},{c[10]:.3e}) K=({c[11]*1e3:.3f},{c[12]*1e3:.3f})mNm "
              f"IN={c[7]['status']} {c[7]['stop_angle_deg']} OUT={c[8]['status']} {c[8]['stop_angle_deg']}")

    best = next((c for c in feasible if c[7]["next_peak_deg"] is not None), None)
    if best:
        _, _, material, D, rball, rweight, mweight, in_result, out_result, *_ = best
        fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
        for ax, axis, result in ((axes[0], "IN", in_result), (axes[1], "OUT", out_result)):
            ts, ys = [], []
            for sol in result["segments"]:
                grid = np.linspace(sol.t[0], sol.t[-1], max(2, int((sol.t[-1]-sol.t[0])/.001)+1))
                y = sol.sol(grid)
                ts.extend(grid.tolist())
                ys.extend(np.degrees(y[0]).tolist())
            if result["stop_time_s"] is not None:
                ts.extend([result["stop_time_s"], 5.0])
                ys.extend([result["stop_angle_deg"], result["stop_angle_deg"]])
            ax.plot(ts, ys, linewidth=1.8, color="#1f77b4")
            ax.set_title(f"{axis}: {material}, D={D*1000:.1f} mm, ball arm={rball*1000:.0f} mm, "
                         f"weight={mweight*1000:.1f} g @ {rweight*1000:.0f} mm")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Angle [deg]")
            ax.grid(True, alpha=.3)
        fig.suptitle("Best screened geometry: no damper, release from +60 deg")
        fig.tight_layout()
        fig.savefig(OUT / "wbs11_arm_ratio_best_timeseries.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
