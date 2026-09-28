"""Stage 5: 球あり自由減衰波形から球抗力係数を同定する。"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from peak_to_peak_solver import solve_next_turning_point
from run_hybrid_rod_damping_identification import AIR_DENSITY_KG_M3
from run_hybrid_rod_damping_identification import AIR_DYNAMIC_VISCOSITY_PA_S
from run_hybrid_rod_damping_identification import FRICTION_EPSILON_DEG_S
from run_hybrid_rod_damping_identification import MAX_STEP_PERIOD_FRACTION
from run_hybrid_rod_damping_identification import MAXIMUM_SEARCH_PERIODS
from run_hybrid_rod_damping_identification import MINIMUM_AMPLITUDE_DEG
from run_hybrid_rod_damping_identification import ROD_DIAMETER_M
from run_hybrid_rod_damping_identification import ROD_DRAG_COEFFICIENT
from run_hybrid_rod_damping_identification import ROD_LOWER_LENGTH_M
from run_hybrid_rod_damping_identification import ROD_UPPER_LENGTH_M
from run_hybrid_rod_damping_identification import THEORETICAL_ROD_C
from run_hybrid_rod_damping_identification import DEFAULT_ANGLE_SPEED_ATOL
from run_hybrid_rod_damping_identification import DEFAULT_RTOL
from run_hybrid_rod_damping_identification import explicit_energy_basis


HERE = Path(__file__).resolve().parent
DATE = "20260921"
SPHERE_MASS_KG = 0.0039
SPHERE_DIAMETER_M = 0.100
SPHERE_RADIUS_M = SPHERE_DIAMETER_M / 2.0
SPHERE_CENTER_ARM_M = 0.174
SPHERE_AREA_M2 = math.pi * SPHERE_DIAMETER_M**2 / 4.0
ROD_EXPOSED_UPPER_M = 0.129
ROD_ALPHA_BALL = (ROD_EXPOSED_UPPER_M / ROD_UPPER_LENGTH_M) ** 4
MAXIMUM_SPHERE_C = 2.0e-4


def read_stage_inputs(root, date):
    parent = root / date
    stage = parent / "hybrid_identification"
    turning = pd.read_csv(stage / "01_preprocessing" / "turning_points.csv")
    waveforms = pd.read_csv(stage / "01_preprocessing" / "waveform_preprocessing.csv")
    parameters = pd.read_csv(
        stage / "02_frequency_identification" / "identified_inertia_restoring.csv"
    )
    selection = pd.read_csv(parent / "waveform_review" / "waveform_selection.csv", encoding="utf-8-sig")
    settings = json.loads(
        (stage / "04_rod_damping_identification" / "stage4_settings.json").read_text(encoding="utf-8")
    )
    return turning, waveforms, parameters, selection, settings


def build_ball_intervals(turning, parameters, selection):
    approved = selection[
        (pd.to_numeric(selection["use_for_fitting"], errors="raise") == 1)
        & (selection["configuration"].astype(str).str.upper() == "BALL")
    ]
    allowed = set(approved["segment_id"].astype(str))
    parameter_index = parameters.set_index(["axis", "configuration"])
    rows = []
    ball = turning[turning["segment_id"].astype(str).isin(allowed)]
    for segment_id, group in ball.groupby("segment_id", sort=False):
        group = group.sort_values("peak_number").reset_index(drop=True)
        initial = np.flatnonzero(group["is_initial_peak"].to_numpy(dtype=int) == 1)
        if len(initial) != 1:
            raise ValueError(f"{segment_id}: initial peak is not unique")
        group = group.iloc[int(initial[0]):].reset_index(drop=True)
        axis = str(group.loc[0, "axis"]).upper()
        physical = parameter_index.loc[(axis, "BALL")]
        for index in range(len(group) - 1):
            first, last = group.iloc[index], group.iloc[index + 1]
            if int(first["eligible_for_later_stages"]) != 1 or int(last["eligible_for_later_stages"]) != 1:
                continue
            if min(float(first["amplitude_deg"]), float(last["amplitude_deg"])) < MINIMUM_AMPLITUDE_DEG:
                continue
            a = float(first["centered_peak_angle_deg"])
            z = float(last["centered_peak_angle_deg"])
            if a * z >= 0.0:
                raise ValueError(f"{segment_id}: successive peak signs are not alternating")
            rows.append({
                "interval_id": f"{segment_id}_H{index + 1:03d}",
                "segment_id": str(segment_id), "axis": axis,
                "direction": str(group.loc[0, "direction"]).upper(),
                "interval_number": index + 1,
                "start_peak_number": int(first["peak_number"]),
                "end_peak_number": int(last["peak_number"]),
                "start_time_s": float(first["peak_time_s"]),
                "end_time_s": float(last["peak_time_s"]),
                "start_angle_rad": float(np.deg2rad(a)),
                "measured_next_angle_rad": float(np.deg2rad(z)),
                "start_amplitude_deg": abs(a), "end_amplitude_deg": abs(z),
                "inertia_kg_m2": float(physical["inertia_kg_m2"]),
                "restoring_n_m_per_rad": float(physical["restoring_n_m_per_rad"]),
            })
    if not rows:
        raise ValueError("No eligible BALL half cycles")
    counts = pd.Series([r["segment_id"] for r in rows]).value_counts().to_dict()
    for row in rows:
        row["waveform_interval_count"] = int(counts[row["segment_id"]])
    return rows


def simulate(interval, sphere_c, stage4, rtol=DEFAULT_RTOL, atol=DEFAULT_ANGLE_SPEED_ATOL):
    axis = interval["axis"]
    rod_c = float(stage4["rod_drag_theory"]["theoretical_c_n_m_s2_per_rad2"])
    friction = float(stage4["models_for_review"]["theoretical_c"]["tau_" + axis])
    ball_rod_c = rod_c * ROD_ALPHA_BALL
    result = solve_next_turning_point(
        interval["start_angle_rad"], interval["inertia_kg_m2"],
        interval["restoring_n_m_per_rad"], 0.0, ball_rod_c + sphere_c,
        friction, np.deg2rad(FRICTION_EPSILON_DEG_S),
        rtol=rtol, angle_speed_atol=atol,
        max_step_fraction=MAX_STEP_PERIOD_FRACTION,
        max_periods=MAXIMUM_SEARCH_PERIODS,
    )
    return float(result["next_angle_rad"])


def fit_c(intervals, stage4):
    groups = {}
    for row in intervals:
        groups.setdefault(row["segment_id"], []).append(row)
    rod_ball = THEORETICAL_ROD_C * ROD_ALPHA_BALL
    tau_by_axis = stage4["models_for_review"]["theoretical_c"]
    numerators, denominators = {}, {}
    prepared = {}
    wave_stats = {}
    for segment_id, rows in groups.items():
        numerator = denominator = 0.0
        for row in rows:
            basis = explicit_energy_basis(row["start_angle_rad"], row["inertia_kg_m2"], row["restoring_n_m_per_rad"])
            start_e = row["restoring_n_m_per_rad"] * (1.0 - math.cos(row["start_angle_rad"]))
            end_e = row["restoring_n_m_per_rad"] * (1.0 - math.cos(row["measured_next_angle_rad"]))
            target = start_e - end_e - float(tau_by_axis["tau_" + row["axis"]]) * basis[2] - rod_ball * basis[1]
            weight2 = 1.0 / len(rows)
            numerator += weight2 * basis[1] * target
            denominator += weight2 * basis[1] ** 2
            row["_sphere_basis"] = float(basis[1])
            row["_sphere_target"] = float(target)
            row["_weight2"] = weight2
        numerators[segment_id], denominators[segment_id] = numerator, denominator
        wave_stats[segment_id] = (numerator, denominator)
        prepared[segment_id] = rows
    c = max(0.0, sum(numerators.values()) / sum(denominators.values()))

    def objective(value):
        sums = []
        for rows in groups.values():
            sums.extend(row["_weight2"] * (row["_sphere_target"] - value * row["_sphere_basis"]) ** 2 for row in rows)
        return float(sum(sums) / len(groups))

    for segment_id, rows in groups.items():
        row_count = sum(r["_weight2"] * r["_sphere_basis"] ** 2 for r in rows)
        waveform_c = max(0.0, numerators[segment_id] / row_count)
        rows[0]["_waveform_c"] = waveform_c
    return c, objective, groups, wave_stats


def brown_lawler_cd(reynolds):
    # Brown & Lawler corrected-sphere-data correlation, valid through Re < 2e5.
    re = max(float(reynolds), 1e-12)
    return (24.0 / re) * (1.0 + 0.15 * re**0.681) + 0.407 / (1.0 + 8710.0 / re)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=DATE)
    parser.add_argument("--result-root", type=Path, default=HERE / "results")
    args = parser.parse_args()
    turning, waveforms, parameters, selection, stage4 = read_stage_inputs(args.result_root, args.date)
    intervals = build_ball_intervals(turning, parameters, selection)
    c_sphere, objective, groups, wave_stats = fit_c(intervals, stage4)
    predictions = []
    for row in intervals:
        next_angle = simulate(row, c_sphere, stage4)
        row = dict(row)
        row["predicted_next_angle_deg"] = float(np.rad2deg(next_angle))
        row["residual_deg"] = row["predicted_next_angle_deg"] - float(np.rad2deg(row["measured_next_angle_rad"]))
        row["rod_c_ball"] = THEORETICAL_ROD_C * ROD_ALPHA_BALL
        row["sphere_c"] = c_sphere
        predictions.append(row)
    interval_table = pd.DataFrame(predictions)

    # Per-waveform estimates expose repeatability; resample waveforms for a cluster bootstrap CI.
    waveform_rows = []
    for segment_id, rows in groups.items():
        waveform_rows.append({"segment_id": segment_id, "axis": rows[0]["axis"], "c_sphere": rows[0]["_waveform_c"], "intervals": len(rows), "rmse_deg": float(np.sqrt(np.mean(interval_table.loc[interval_table.segment_id == segment_id, "residual_deg"]**2)))})
    rng = np.random.default_rng(20260921)
    boot = []
    ids = list(groups)
    for unused in range(1000):
        sampled = rng.integers(0, len(ids), size=len(ids))
        numerator = sum(wave_stats[ids[i]][0] for i in sampled)
        denominator = sum(wave_stats[ids[i]][1] for i in sampled)
        boot.append(float(max(0.0, numerator / denominator)))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    omega = np.array([
        math.sqrt(2.0 * row["restoring_n_m_per_rad"] * (1.0 - math.cos(row["start_angle_rad"])) / row["inertia_kg_m2"])
        for row in intervals
    ])
    velocity = SPHERE_CENTER_ARM_M * omega
    reynolds = AIR_DENSITY_KG_M3 * velocity * SPHERE_DIAMETER_M / AIR_DYNAMIC_VISCOSITY_PA_S
    reynolds_table = interval_table[["interval_id", "segment_id", "axis", "start_amplitude_deg"]].copy()
    reynolds_table["omega_conservative_rad_s"] = omega
    reynolds_table["sphere_speed_m_s"] = velocity
    reynolds_table["reynolds_number"] = reynolds
    reynolds_table["brown_lawler_cd"] = [brown_lawler_cd(re) for re in reynolds]
    reynolds_table["sphere_c_implied_cd"] = 2.0 * c_sphere / (AIR_DENSITY_KG_M3 * SPHERE_AREA_M2 * SPHERE_CENTER_ARM_M**3)
    # Flow-dependent drag correlation integrated over each conservative half-cycle.
    # The energy loss from quadratic drag is proportional to integral(Cd * omega^2 dtheta).
    theory_cd = []
    for row in intervals:
        amplitude = abs(row["start_angle_rad"])
        theta = np.linspace(-amplitude, amplitude, 401)
        omega2 = 2.0 * row["restoring_n_m_per_rad"] / row["inertia_kg_m2"] * (np.cos(theta) - math.cos(amplitude))
        local_re = AIR_DENSITY_KG_M3 * SPHERE_CENTER_ARM_M * np.sqrt(np.maximum(omega2, 0.0)) * SPHERE_DIAMETER_M / AIR_DYNAMIC_VISCOSITY_PA_S
        local_cd = np.asarray([brown_lawler_cd(re) for re in local_re])
        denominator = float(np.trapezoid(omega2, theta))
        theory_cd.append(float(np.trapezoid(local_cd * omega2, theta) / denominator))
    reynolds_table["brown_lawler_cd_energy_weighted"] = theory_cd
    reynolds_table["theory_reynolds_max"] = reynolds

    out = args.result_root / args.date / "hybrid_identification" / "05_sphere_drag_identification"
    out.mkdir(parents=True, exist_ok=True)
    interval_table.to_csv(out / "interval_predictions.csv", index=False, float_format="%.10g")
    pd.DataFrame(waveform_rows).to_csv(out / "waveform_estimates.csv", index=False, float_format="%.10g")
    reynolds_table.to_csv(out / "reynolds_assessment.csv", index=False, float_format="%.10g")
    cd = 2.0 * c_sphere / (AIR_DENSITY_KG_M3 * SPHERE_AREA_M2 * SPHERE_CENTER_ARM_M**3)
    summary = {
        "stage": 5, "date": args.date, "sphere_c_n_m_s2_per_rad2": c_sphere,
        "sphere_c_cluster_bootstrap_95pct": [float(ci_low), float(ci_high)],
        "sphere_cd_equivalent": cd,
        "sphere_cd_cluster_bootstrap_95pct": [float(2*ci_low/(AIR_DENSITY_KG_M3*SPHERE_AREA_M2*SPHERE_CENTER_ARM_M**3)), float(2*ci_high/(AIR_DENSITY_KG_M3*SPHERE_AREA_M2*SPHERE_CENTER_ARM_M**3))],
        "fit_rmse_half_cycle_deg": float(np.sqrt(np.mean(interval_table.residual_deg**2))),
        "objective_equal_waveform_mse_rad2": objective(c_sphere),
        "waveforms": len(groups), "intervals": len(intervals),
        "tau_Nm_IN": stage4["models_for_review"]["theoretical_c"]["tau_IN"],
        "tau_Nm_OUT": stage4["models_for_review"]["theoretical_c"]["tau_OUT"],
        "b_IN": 0.0, "b_OUT": 0.0,
        "rod_c_n_m_s2_per_rad2_no_ball": THEORETICAL_ROD_C,
        "rod_exposed_upper_m_ball": ROD_EXPOSED_UPPER_M,
        "rod_alpha_ball": ROD_ALPHA_BALL,
        "rod_c_n_m_s2_per_rad2_ball": THEORETICAL_ROD_C * ROD_ALPHA_BALL,
        "sphere_mass_kg": SPHERE_MASS_KG, "sphere_diameter_m": SPHERE_DIAMETER_M,
        "sphere_center_arm_m": SPHERE_CENTER_ARM_M, "sphere_area_m2": SPHERE_AREA_M2,
        "sphere_reynolds_min": float(reynolds.min()), "sphere_reynolds_median": float(np.median(reynolds)), "sphere_reynolds_max": float(reynolds.max()),
        "brown_lawler_cd_min": float(np.min(reynolds_table.brown_lawler_cd)), "brown_lawler_cd_median": float(np.median(reynolds_table.brown_lawler_cd)), "brown_lawler_cd_max": float(np.max(reynolds_table.brown_lawler_cd)),
        "brown_lawler_cd_energy_weighted": float(np.average(theory_cd, weights=[sum(row["_weight2"] * row["_sphere_basis"]**2 for row in groups[sid]) for sid in groups for row in groups[sid]])),
        "fit_algorithm": "explicit half-cycle dissipated-energy balance with equal waveform weight; shared nonnegative sphere c; exact nonlinear ODE endpoint prediction used for validation",
        "bootstrap": "1000 cluster resamples of waveform-level weighted least-squares numerator/denominator; deterministic seed 20260921",
        "rod_reynolds_note": "Stage 4 rod theoretical c scaled by exposed-rod rotational drag length^4 ratio alpha=(129/229)^4",
    }
    (out / "stage5_settings.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
