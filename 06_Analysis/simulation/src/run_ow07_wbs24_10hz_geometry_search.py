#!/usr/bin/env python3
"""OW-07 WBS2.4: compact geometry search for 10 Hz wind response.

This is an exploratory model. It preserves the WBS2.3 grease b and axis-specific
tau, samples candidate geometry, then validates minimum-sphere, minimum-ballast
and best-linear-signal candidates with the nonlinear plant, sensor and RTS.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
OUT = SIM / "results/observer_wind/ow07_hardware_bandwidth_design/wbs24_10hz_geometry_search"
sys.path.insert(0, str(HERE))

import run_ow07_wbs23_damper_observer as wbs23
from run_ow05_robust_bandwidth_damper_sweep import _fast_nonlinear_rts
from run_ow07_wbs21_grease_damper_free_decay import equivalent_b
from sensor_model import AngleSensorParameters, apply_angle_sensor_model
from wind import drag_force_from_speed

G = 9.80665
RHO_AIR = 1.225
CD = 0.544
RHO_FOAM = 1050.0 / 60.0
RHO_REFERENCE_SOLID_PS = 1050.0
RHO_ROD = 1600.0  # provisional CFRP tube equivalent density
ROD_WALL_M = 0.0005
ROD_CLEARANCE_M = 0.005
ROD_REFERENCE_D_M = 0.005
ROD_REFERENCE_LENGTH_M = 0.229
ROD_REFERENCE_WALL_M = 0.0005
ROD_REFERENCE_RHO = 2700.0  # baseline WBS11 aluminum reference
D0_M, MB0_KG, RB0_M = 0.100, 0.0039, 0.174
MW0_KG, RW0_M = 0.05072, 0.055
F_HZ, DV_M_S, VMEAN_M_S = 10.0, 0.5, 2.0
GAIN_MIN = 0.5
ZETA_MIN = 0.70  # about 4.6% overshoot in a linear 2nd-order step response
BALL_D_MIN_M, BALL_D_MAX_M = 0.030, 0.100
BALL_ARM_MAX_M = 0.500
BALLAST_ARM_MIN_M, BALLAST_ARM_MAX_M = 0.002, 0.070
BALLAST_MASS_MIN_KG, BALLAST_MASS_MAX_KG = 0.05, 1000.0
ROD_D_MIN_M, ROD_D_MAX_M = 0.002, 0.008
SAMPLES = 2_000_000
SEED = 20261008

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_baseline() -> dict[str, dict]:
    return wbs23.load_coefficients()


def reference_residuals(records: dict[str, dict]) -> dict[str, tuple[float, float]]:
    """Subtract modeled baseline ball, ballast and aluminum rod from WBS2.3 I/K."""
    area = math.pi * ((ROD_REFERENCE_D_M / 2) ** 2
                      - (ROD_REFERENCE_D_M / 2 - ROD_REFERENCE_WALL_M) ** 2)
    mrod = area * ROD_REFERENCE_LENGTH_M * ROD_REFERENCE_RHO
    Irod = mrod * ROD_REFERENCE_LENGTH_M ** 2 / 3
    Krod = -G * mrod * ROD_REFERENCE_LENGTH_M / 2
    Iball = 0.4 * MB0_KG * (D0_M / 2) ** 2 + MB0_KG * RB0_M ** 2
    Iweight = MW0_KG * RW0_M ** 2
    Kball_weight_rod = G * MW0_KG * RW0_M - G * MB0_KG * RB0_M + Krod
    return {
        axis: (rec["inertia_kg_m2"] - Iball - Iweight - Irod,
               rec["restoring_n_m_per_rad"] - Kball_weight_rod)
        for axis, rec in records.items()
    }


def tube_terms(rod_od: np.ndarray | float, length: np.ndarray | float,
               density: float = RHO_ROD):
    area = math.pi * ((np.asarray(rod_od) / 2) ** 2
                       - np.maximum(np.asarray(rod_od) / 2 - ROD_WALL_M, 0) ** 2)
    mass = area * length * density
    return mass * length ** 2 / 3, -G * mass * length / 2


def sample_screen(records: dict[str, dict], residuals: dict[str, tuple[float, float]]):
    rng = np.random.default_rng(SEED)
    n = SAMPLES
    d = rng.uniform(BALL_D_MIN_M, BALL_D_MAX_M, n)
    arm = np.maximum(rng.uniform(BALL_D_MIN_M / 2 + ROD_CLEARANCE_M,
                                 BALL_ARM_MAX_M, n), d / 2 + ROD_CLEARANCE_M)
    rw = rng.uniform(BALLAST_ARM_MIN_M, BALLAST_ARM_MAX_M, n)
    mw = 10 ** rng.uniform(math.log10(BALLAST_MASS_MIN_KG),
                           math.log10(BALLAST_MASS_MAX_KG), n)
    rod_od = rng.uniform(ROD_D_MIN_M, ROD_D_MAX_M, n)
    rod_len = arm + d / 2 + ROD_CLEARANCE_M
    Irod, Krod = tube_terms(rod_od, rod_len)
    mb = RHO_FOAM * math.pi * d ** 3 / 6
    Iball = 0.4 * mb * (d / 2) ** 2 + mb * arm ** 2
    Iweight = mw * rw ** 2
    mean_force = 0.5 * RHO_AIR * CD * math.pi * d ** 2 / 4 * VMEAN_M_S ** 2
    force8 = 0.5 * RHO_AIR * CD * math.pi * d ** 2 / 4 * 8.0 ** 2
    B = equivalent_b(30.0, 8.0, 1.0)
    common_geometry = dict(ball_d_m=d, ball_arm_m=arm, ballast_mass_kg=mw,
                           ballast_arm_m=rw, rod_od_m=rod_od,
                           ball_mass_kg=mb, rod_length_m=rod_len)
    axis_passes = []
    axis_metrics = {}
    omega = 2 * math.pi * F_HZ
    for axis in ("IN", "OUT"):
        ifr, kfr = residuals[axis]
        I = ifr + Iball + Iweight + Irod
        K = kfr + G * mw * rw - G * mb * arm + Krod
        th_mean = np.arctan2(mean_force * arm, K)
        keff = K * np.cos(th_mean) + mean_force * arm * np.sin(th_mean)
        zeta = B / (2 * np.sqrt(np.maximum(I * keff, 1e-300)))
        denom = np.sqrt((keff - I * omega ** 2) ** 2 + (B * omega) ** 2)
        gain = keff / denom
        torque_gain = arm * np.cos(th_mean) * RHO_AIR * CD * math.pi * d ** 2 / 4 * VMEAN_M_S
        angle_amp = torque_gain * DV_M_S / denom
        th8 = np.arctan2(force8 * arm, K)
        fn = np.sqrt(np.maximum(keff / I, 0)) / (2 * math.pi)
        valid = (K > 0) & (th8 <= math.radians(60)) & (zeta >= ZETA_MIN) & (gain >= GAIN_MIN)
        axis_passes.append(valid)
        axis_metrics[axis] = dict(I=I, K=K, keff=keff, zeta=zeta, gain=gain,
                                  angle_amp=angle_amp, th8=th8, fn=fn)
    joint = axis_passes[0] & axis_passes[1]
    strict_joint = joint.copy()
    for axis in ("IN", "OUT"):
        strict_joint &= axis_metrics[axis]["zeta"] >= 1.0
    # Preserve all jointly feasible points plus a compact shortlist by package
    # envelope, then ballast mass. Geometry feasibility is only a first screen.
    idx = np.flatnonzero(joint)
    if idx.size:
        envelope = arm[idx] + d[idx] / 2
        order = np.lexsort((mw[idx], envelope))
        idx = idx[order[:250]]
    rows = []
    for i in idx:
        row = {k: float(v[i]) for k, v in common_geometry.items()}
        row["package_envelope_m"] = float(arm[i] + d[i] / 2)
        for axis in ("IN", "OUT"):
            for key in ("I", "K", "zeta", "gain", "angle_amp", "th8", "fn"):
                value = axis_metrics[axis][key][i]
                suffix = "_rad" if key in ("angle_amp", "th8") else ""
                row[f"{axis}_{key}{suffix}"] = float(value)
        rows.append(row)
    return rows, int(joint.sum()), int(axis_passes[0].sum()), int(axis_passes[1].sum()), int(strict_joint.sum()), axis_metrics, common_geometry


def evaluate_fixed_candidate(records, residuals, candidate, candidate_name):
    c = candidate.copy()
    d, arm, mw, rw, rod_od = (c[k] for k in ("ball_d_m", "ball_arm_m", "ballast_mass_kg", "ballast_arm_m", "rod_od_m"))
    rod_len = arm + d / 2 + ROD_CLEARANCE_M
    mball = RHO_FOAM * math.pi * d ** 3 / 6
    Irod, Krod = tube_terms(rod_od, rod_len)
    Irod, Krod = float(Irod), float(Krod)
    Iball = 0.4 * mball * (d / 2) ** 2 + mball * arm ** 2
    Iweight = mw * rw ** 2
    b = equivalent_b(30.0, 8.0, 1.0)
    wbs23.DT = 0.01
    wbs23.FS = 100.0
    wbs23.LEVER_M = arm
    wbs23.AREA_M2 = math.pi * d ** 2 / 4
    wbs23.DRAG_FACTOR = 0.5 * RHO_AIR * CD * wbs23.AREA_M2
    sensor = AngleSensorParameters(sample_rate_hz=100, resolution_bits=14,
                                   full_scale_deg=360, white_noise_std_deg=0.015,
                                   coloured_noise_std_deg=0.010,
                                   coloured_noise_time_constant_s=0.475,
                                   fixed_delay_s=0.010)
    t = np.arange(3001) * 0.01
    wind = 2.0 + DV_M_S * np.sin(2 * math.pi * F_HZ * t)
    force = drag_force_from_speed(wind, RHO_AIR, CD, wbs23.AREA_M2)
    Xmask = (t >= 3.0) & (t <= 27.0)
    X = np.column_stack((np.ones(Xmask.sum()),
                         np.sin(2 * math.pi * F_HZ * t[Xmask]),
                         np.cos(2 * math.pi * F_HZ * t[Xmask])))
    rows = []
    metrics = {}
    angle_data = {}
    for axis in ("IN", "OUT"):
        ifr, kfr = residuals[axis]
        I = ifr + Iball + Iweight + Irod
        K = kfr + G * mw * rw - G * mball * arm + Krod
        base = records[axis]
        cdrag = (base["rod_quadratic_drag_n_m_s2_per_rad2"] * (rod_len / ROD_REFERENCE_LENGTH_M) ** 4
                 + base["ball_quadratic_drag_n_m_s2_per_rad2"] * (d / D0_M) ** 2)
        coeff = dict(inertia_kg_m2=I, restoring_n_m_per_rad=K,
                     total_quadratic_drag_n_m_s2_per_rad2=cdrag,
                     viscous_damping_n_m_s_per_rad=b,
                     tau_n_m=base["tau_n_m"])
        theta0 = math.atan2(arm * force[0], K)
        force_mean = 0.5 * RHO_AIR * CD * math.pi * d ** 2 / 4 * VMEAN_M_S ** 2
        keff = K * math.cos(theta0) + arm * force[0] * math.sin(theta0)
        omega = 2 * math.pi * F_HZ
        denom = math.hypot(keff - I * omega ** 2, b * omega)
        theta8 = math.atan2(0.5 * RHO_AIR * CD * math.pi * d ** 2 / 4 * 8.0 ** 2 * arm, K)
        torque_gain = arm * math.cos(theta0) * RHO_AIR * CD * math.pi * d ** 2 / 4 * VMEAN_M_S
        metrics[axis] = {"I_kg_m2": I, "K_Nm_per_rad": K,
                         "zeta_linearized": b / (2 * math.sqrt(I * keff)),
                         "normalized_mechanical_gain_10Hz": keff / denom,
                         "natural_frequency_hz": math.sqrt(keff / I) / (2 * math.pi),
                         "static_angle_8m_s_deg": math.degrees(theta8),
                         "linear_angle_amplitude_10Hz_deg": math.degrees(torque_gain * DV_M_S / denom)}
        theta, _ = wbs23.rk4_plant(t, force, coeff, theta0)
        angle_fit = np.linalg.lstsq(X, theta[Xmask], rcond=None)[0]
        angle_amp_deg = math.degrees(math.hypot(angle_fit[1], angle_fit[2]))
        angle_data[axis] = theta
        for q in (0.003, 0.01, 0.03):
            gains, phases, rmses = [], [], []
            for seed in (20261007, 20261008, 20261009):
                observed, _ = apply_angle_sensor_model(t, theta, sensor, seed)
                f_est = _fast_nonlinear_rts(
                    observed, coeff, 0.01, arm, math.radians(0.02), q, 0.5,
                    np.array([theta0, 0.0, force[0]]))
                wind_est = np.sign(f_est) * np.sqrt(np.abs(f_est) / wbs23.DRAG_FACTOR)
                fit = np.linalg.lstsq(X, wind_est[Xmask], rcond=None)[0]
                gains.append(float(math.hypot(fit[1], fit[2]) / DV_M_S))
                phases.append(float(math.degrees(math.atan2(fit[2], fit[1]))))
                rmses.append(float(np.sqrt(np.mean((wind_est[Xmask] - wind[Xmask]) ** 2))))
            rows.append({"candidate": candidate_name, "axis": axis, "Q_N_per_sample": q,
                         "tau_fixed_Nm": base["tau_n_m"], "b_fixed_Nms_per_rad": b,
                         "theta_10Hz_amp_deg": angle_amp_deg,
                         "angle_amp_over_assumed_sigma": angle_amp_deg / 0.02,
                         "RTS_gain_mean_3seeds": float(np.mean(gains)),
                         "RTS_gain_seed_min": min(gains), "RTS_gain_seed_max": max(gains),
                         "RTS_phase_deg_seeds": ";".join(f"{x:.2f}" for x in phases),
                         "wind_RMSE_m_s_mean": float(np.mean(rmses)),
                         "wind_RMSE_m_s_seeds": ";".join(f"{x:.4f}" for x in rmses)})
    return rows, angle_data, dict(c, rod_length_m=rod_len, ball_mass_kg=mball, metrics=metrics,
                                  tau_IN=records["IN"]["tau_n_m"], tau_OUT=records["OUT"]["tau_n_m"],
                                  b_Nms_per_rad=b)


def plot_results(screen_rows, rts_rows, metadata):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    if screen_rows:
        mass = np.array([r["ballast_mass_kg"] for r in screen_rows])
        diam = np.array([r["ball_d_m"] * 1000 for r in screen_rows])
        gain = np.array([min(r["IN_gain"], r["OUT_gain"]) for r in screen_rows])
        sc = ax.scatter(diam, mass, c=gain, s=35, cmap="viridis", edgecolor="black", linewidth=.3)
        fig.colorbar(sc, ax=ax, label="min axis 10 Hz normalized mechanical gain")
    else:
        ax.text(.5, .6, "No jointly feasible sampled geometry", ha="center", transform=ax.transAxes)
    ax.scatter([metadata["ball_d_m"] * 1000], [metadata["ballast_mass_kg"]],
               marker="*", s=180, color="#d62728", edgecolor="black", label="minimum ballast candidate")
    ax.set_yscale("log")
    ax.set_xlabel("Sphere diameter [mm]")
    ax.set_ylabel("Counterweight mass [kg], log scale")
    ax.set_title("10 Hz first-pass geometry screen: ζ ≥ 0.70 and gain ≥ 0.5")
    ax.grid(True, which="both", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "geometry_search_screen.png", dpi=180)
    fig.savefig(OUT / "geometry_search_screen.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for col, axis in enumerate(("IN", "OUT")):
        ax = axes[0, col]
        for candidate_name, label, marker in (("minimum_sphere", "minimum sphere", "o"),
                                               ("minimum_ballast", "minimum ballast", "s"),
                                               ("best_signal", "best linear signal", "^")):
            rs = sorted((r for r in rts_rows if r["axis"] == axis and r["candidate"] == candidate_name),
                        key=lambda r: r["Q_N_per_sample"])
            ax.plot([r["Q_N_per_sample"] for r in rs], [r["RTS_gain_mean_3seeds"] for r in rs],
                    marker + "-", label=label)
        ax.axhline(GAIN_MIN, color="black", linestyle="--", label="gain threshold 0.5")
        ax.set_xscale("log")
        ax.set_title(axis)
        ax.grid(True, which="both", alpha=.3)
        ax = axes[1, col]
        for candidate_name, label, marker in (("minimum_sphere", "minimum sphere", "o"),
                                               ("minimum_ballast", "minimum ballast", "s"),
                                               ("best_signal", "best linear signal", "^")):
            rs = sorted((r for r in rts_rows if r["axis"] == axis and r["candidate"] == candidate_name),
                        key=lambda r: r["Q_N_per_sample"])
            ax.plot([r["Q_N_per_sample"] for r in rs], [r["wind_RMSE_m_s_mean"] for r in rs],
                    marker + "-", label=label)
        ax.set_xscale("log")
        ax.set_xlabel("RTS process noise Q [N/sample]")
        ax.grid(True, which="both", alpha=.3)
    axes[0, 0].set_ylabel("Estimated wind amplitude gain")
    axes[1, 0].set_ylabel("Wind speed RMSE [m/s]")
    axes[0, 1].legend(loc="best", fontsize=8)
    fig.suptitle("Raising RTS Q may recover apparent gain while degrading wind RMSE")
    fig.tight_layout()
    fig.savefig(OUT / "rts_q_sensitivity.png", dpi=180)
    fig.savefig(OUT / "rts_q_sensitivity.svg")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_baseline()
    residuals = reference_residuals(records)
    shortlist, joint_count, in_count, out_count, strict_count, _, _ = sample_screen(records, residuals)
    write_csv(OUT / "jointly_feasible_geometry_shortlist.csv", shortlist)
    min_sphere_row = min(shortlist, key=lambda r: r["ball_d_m"])
    min_mass_row = min(shortlist, key=lambda r: r["ballast_mass_kg"])
    best_signal = max(shortlist, key=lambda r: min(r["IN_angle_amp_rad"], r["OUT_angle_amp_rad"]))
    candidate_specs = {
        "minimum_sphere": {k: min_sphere_row[k] for k in ("ball_d_m", "ball_arm_m", "ballast_mass_kg", "ballast_arm_m", "rod_od_m")},
        "minimum_ballast": {k: min_mass_row[k] for k in ("ball_d_m", "ball_arm_m", "ballast_mass_kg", "ballast_arm_m", "rod_od_m")},
        "best_signal": {k: best_signal[k] for k in ("ball_d_m", "ball_arm_m", "ballast_mass_kg", "ballast_arm_m", "rod_od_m")},
    }
    rts_rows = []
    candidate_meta = {}
    for name, candidate in candidate_specs.items():
        rows, _, metadata = evaluate_fixed_candidate(records, residuals, candidate, name)
        rts_rows.extend(rows)
        candidate_meta[name] = metadata
    write_csv(OUT / "candidate_rts_noise_validation.csv", rts_rows)
    summary = {
        "sample_count": SAMPLES, "seed": SEED,
        "single_axis_pass_counts": {"IN": in_count, "OUT": out_count},
        "joint_pass_count": joint_count,
        "ball_density_kg_m3": RHO_FOAM,
        "foam_density_basis": "1050 kg/m3 solid polystyrene divided by expansion ratio 60",
        "fixed_tau_Nm": {axis: records[axis]["tau_n_m"] for axis in ("IN", "OUT")},
        "fixed_b_Nms_per_rad": equivalent_b(30.0, 8.0, 1.0),
        "nonlinear_rts_candidates": candidate_meta,
        "strict_overdamped_joint_passes_in_sample": strict_count,
        "best_screened_angle_amplitude_candidate": best_signal,
        "screen_thresholds": {"zeta_min_mild_overshoot": ZETA_MIN,
                              "strict_overdamped_zeta": 1.0,
                              "normalized_mechanical_gain_min": GAIN_MIN,
                              "static_angle_at_8m_s_max_deg": 60.0},
        "sampling_bounds": {"ball_D_m": [BALL_D_MIN_M, BALL_D_MAX_M],
                            "ball_center_arm_m": [None, BALL_ARM_MAX_M],
                            "counterweight_mass_kg": [BALLAST_MASS_MIN_KG, BALLAST_MASS_MAX_KG],
                            "counterweight_arm_m": [BALLAST_ARM_MIN_M, BALLAST_ARM_MAX_M],
                            "rod_OD_m": [ROD_D_MIN_M, ROD_D_MAX_M],
                            "rod_wall_m": ROD_WALL_M,
                            "rod_material_density_kg_m3": RHO_ROD},
    }
    (OUT / "search_settings_and_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_results(shortlist, rts_rows, candidate_meta["minimum_ballast"])
    print(json.dumps({"output": str(OUT), "joint_pass_count": joint_count,
                      "shortlist_rows": len(shortlist), "candidates": candidate_meta,
                      "rts_validation_rows": len(rts_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
