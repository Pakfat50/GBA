#!/usr/bin/env python3
"""OW-07 WBS2.3: offline RTS wind estimation with the selected grease b.

The plant is the 70 mm sphere candidate from WBS2.2.  The grease b is the
ideal-Couette extrapolation from WBS2.1 and is therefore an assumed value.
This script compares offline RTS against the historical static/LPF baselines
under matched, OW-04 coefficient, b-only, and combined model mismatch cases.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from PIL import Image

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
ROOT = SIM.parents[1]
OUT = SIM / "results/observer_wind/ow07_hardware_bandwidth_design/wbs23_observer"
sys.path.insert(0, str(HERE))

from hbk_model_coefficients import select_coefficients
from hbk_nonlinear_estimators import _continuous_jacobian, _rk4_step
from run_ow05_robust_bandwidth_damper_sweep import _fast_nonlinear_rts
from run_ow07_wbs21_grease_damper_free_decay import equivalent_b
from run_ow07_wbs11_nonlinear_free_decay import (
    BALL_D0, BALL_M0, BALL_R0, BALLAST_M0, BALLAST_R0, CANDIDATES, G,
    candidate_parameters, load_coefficients,
)
from sensor_model import AngleSensorParameters, apply_angle_sensor_model
from wind import drag_force_from_speed, synthesize_kaimal_wind

DT = 0.01
FS = 1.0 / DT
LEVER_M = 0.174  # geometric support-to-sphere center arm used by WBS2.2
BALL_D_M = 0.070
BALL_CD = 0.544
AIR_DENSITY = 1.225
AREA_M2 = math.pi * BALL_D_M**2 / 4.0
DRAG_FACTOR = 0.5 * AIR_DENSITY * BALL_CD * AREA_M2
EPSILON_DEG_S = 0.5
ANGLE_LIMIT = math.radians(60.0)
ANGLE_NOISE_ASSUMED_RAD = math.radians(0.02)
Q_BY_AXIS = {"IN": 3.0e-5, "OUT": 3.0e-4}
LPF_CUTOFF_HZ = {"IN": 0.5, "OUT": 0.75}
NOISE_SEEDS = (20261007, 20261008, 20261009)
SENSOR = AngleSensorParameters(
    sample_rate_hz=FS, resolution_bits=14, full_scale_deg=360.0,
    white_noise_std_deg=0.015, coloured_noise_std_deg=0.010,
    coloured_noise_time_constant_s=0.475, fixed_delay_s=0.010,
)
METHODS = ("RTS", "Static", "LPF", "Static mean 1 s", "Static mean 3 s")
METHOD_COLORS = {
    "RTS": "#0072B2", "Static": "#777777", "LPF": "#D55E00",
    "Static mean 1 s": "#009E73", "Static mean 3 s": "#CC79A7",
}
CASE_COLORS = {
    "matched": "#0072B2", "OW04_joint": "#D55E00",
    "b_low_0.5x": "#009E73", "b_high_1.5x": "#CC79A7",
    "OW04_plus_b_low": "#E69F00", "OW04_plus_b_high": "#7A68A6",
}
OW04_CSV = SIM / "results/observer_wind/ow04_coefficient_sensitivity/ow04_joint_scenarios.csv"
OW04_SCENARIO = "joint_mass_high_arm_high_tau-high_c-high"


def candidate_coefficients(axis: str) -> dict:
    base = load_coefficients()[axis]
    candidate = dict(CANDIDATES["P0"], D_m=BALL_D_M)
    values = candidate_parameters(base, candidate)
    result = base.copy()
    result["inertia_kg_m2"] = values["I"]
    result["restoring_n_m_per_rad"] = values["K"]
    result["total_quadratic_drag_n_m_s2_per_rad2"] = values["c"]
    result["ball_quadratic_drag_n_m_s2_per_rad2"] = base["ball_quadratic_drag_n_m_s2_per_rad2"] * (BALL_D_M / BALL_D0) ** 2
    result["tau_n_m"] = values["tau_dyn"]
    result["viscous_damping_n_m_s_per_rad"] = equivalent_b(30.0, 8.0, 1.0)
    return result


def coefficient_ratios(axis: str) -> dict[str, float]:
    """Ratios of the documented OW-04 high-mass/arm/tau/c corner to BALL."""
    with OW04_CSV.open(encoding="utf-8-sig", newline="") as stream:
        row = next(r for r in csv.DictReader(stream)
                   if r["axis"] == axis and r["scenario"] == OW04_SCENARIO)
    old = select_coefficients(axis, "BALL")
    return {
        "inertia_kg_m2": float(row["observer_I_kg_m2"]) / old["inertia_kg_m2"],
        "restoring_n_m_per_rad": float(row["observer_K_n_m_per_rad"]) / old["restoring_n_m_per_rad"],
        "tau_n_m": float(row["observer_tau_n_m"]) / old["tau_n_m"],
        "ball_quadratic_drag_n_m_s2_per_rad2": (
            float(row["observer_c_ball_n_m_s2_per_rad2"])
            / old["ball_quadratic_drag_n_m_s2_per_rad2"]
        ),
    }


def varied_coefficients(nominal: dict, ratios: dict, b_ratio: float = 1.0,
                        apply_ow04: bool = False) -> dict:
    result = nominal.copy()
    if apply_ow04:
        for key, ratio in ratios.items():
            result[key] = nominal[key] * ratio
        result["total_quadratic_drag_n_m_s2_per_rad2"] = (
            nominal["rod_quadratic_drag_n_m_s2_per_rad2"]
            + result["ball_quadratic_drag_n_m_s2_per_rad2"]
        )
    result["viscous_damping_n_m_s_per_rad"] = (
        nominal["viscous_damping_n_m_s_per_rad"] * b_ratio
    )
    return result


def model_cases(nominal: dict, ratios: dict) -> dict[str, dict]:
    return {
        "matched": varied_coefficients(nominal, ratios),
        "OW04_joint": varied_coefficients(nominal, ratios, apply_ow04=True),
        "b_low_0.5x": varied_coefficients(nominal, ratios, b_ratio=0.5),
        "b_high_1.5x": varied_coefficients(nominal, ratios, b_ratio=1.5),
        "OW04_plus_b_low": varied_coefficients(nominal, ratios, b_ratio=0.5, apply_ow04=True),
        "OW04_plus_b_high": varied_coefficients(nominal, ratios, b_ratio=1.5, apply_ow04=True),
    }


def static_wind_from_angle(angle: np.ndarray, coeff: dict) -> np.ndarray:
    force = coeff["restoring_n_m_per_rad"] * np.tan(np.clip(angle, -ANGLE_LIMIT, ANGLE_LIMIT)) / LEVER_M
    return np.sign(force) * np.sqrt(np.abs(force) / DRAG_FACTOR)


def centered_average(values: np.ndarray, seconds: float) -> np.ndarray:
    width = int(round(seconds * FS))
    if width % 2 == 0:
        width += 1
    return np.convolve(values, np.ones(width) / width, mode="same")


def causal_lpf(values: np.ndarray, cutoff_hz: float) -> np.ndarray:
    pole = math.exp(-2.0 * math.pi * cutoff_hz * DT)
    result = np.asarray(values, dtype=float).copy()
    for _ in range(3):
        result = signal.lfilter([0.0, 1.0 - pole], [1.0, -pole], result)
    return result


def rk4_plant(time_s: np.ndarray, force_n: np.ndarray, coeff: dict,
              initial_angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Integrate 100 Hz output at 5 internal steps and enforce +/-60 deg stops."""
    h = DT / 5.0
    theta, rate = float(initial_angle_rad), 0.0
    angles = np.empty(len(time_s)); angles[0] = theta
    contact = np.zeros(len(time_s), dtype=bool)
    inertia = coeff["inertia_kg_m2"]
    stiffness = coeff["restoring_n_m_per_rad"]
    b = coeff["viscous_damping_n_m_s_per_rad"]
    c = coeff["total_quadratic_drag_n_m_s2_per_rad2"]
    tau = coeff["tau_n_m"]
    eps = math.radians(EPSILON_DEG_S)

    def rhs(th, w, f):
        loss = b*w + c*abs(w)*w + tau*math.tanh(w/eps)
        return w, (LEVER_M*f*math.cos(th) - stiffness*math.sin(th) - loss) / inertia

    for k in range(1, len(time_s)):
        f = float(force_n[k-1])
        for j in range(5):
            tm = time_s[k-1] + j*h
            f0 = float(np.interp(tm, [time_s[k-1], time_s[k]], [force_n[k-1], force_n[k]]))
            f1 = float(np.interp(tm+h/2, [time_s[k-1], time_s[k]], [force_n[k-1], force_n[k]]))
            f2 = float(np.interp(tm+h, [time_s[k-1], time_s[k]], [force_n[k-1], force_n[k]]))
            u1,a1=rhs(theta,rate,f0)
            u2,a2=rhs(theta+h*u1/2,rate+h*a1/2,f1)
            u3,a3=rhs(theta+h*u2/2,rate+h*a2/2,f1)
            u4,a4=rhs(theta+h*u3,rate+h*a3,f2)
            theta += h*(u1+2*u2+2*u3+u4)/6
            rate += h*(a1+2*a2+2*a3+a4)/6
            if abs(theta) >= ANGLE_LIMIT:
                theta = math.copysign(ANGLE_LIMIT, theta)
                if theta*rate > 0:
                    rate = 0.0
                contact[k] = True
        angles[k] = theta
    return angles, contact


def build_inputs() -> dict[str, dict]:
    dt = DT
    result = {}
    # The wind record follows OW-03/05: independent Kaimal, mean 2 m/s, TI 20%.
    t, v, wind_meta = synthesize_kaimal_wind(
        FS, 120.0, 2.0, 0.20, 11.34, 20261013, maximum_speed_m_s=4.0,
    )
    result["wind_model_Kaimal"] = {"time": t, "speed": v, "eval_start": 15.0,
                                    "kind": "turbulent", "meta": wind_meta}
    for f in (1.0, 10.0):
        duration = 30.0
        t = np.arange(int(duration*FS)+1)*dt
        v = 2.0 + 0.5*np.sin(2*np.pi*f*t)
        result[f"sine_{f:g}Hz"] = {"time": t, "speed": v, "eval_start": 3.0,
                                   "kind": "sine", "frequency_hz": f,
                                   "meta": {"mean_m_s": 2.0, "amplitude_m_s": 0.5}}
    t = np.arange(int(120.0*FS)+1)*dt
    v = np.full(len(t), 2.0)
    for center_s, width_s in ((35.0, 5.0), (78.0, 7.0)):
        v += 4.0*np.exp(-0.5*((t-center_s)/width_s)**2)
    v = np.minimum(v, 6.0)
    result["gust_2_to_6m_s"] = {"time": t, "speed": v, "eval_start": 15.0,
                                "kind": "gust", "meta": {"base_m_s": 2.0, "peak_m_s": 6.0}}
    t = np.arange(int(30.0*FS)+1)*dt
    v = np.zeros(len(t))
    result["release_60deg_zero_wind"] = {"time": t, "speed": v, "eval_start": 0.0,
                                         "kind": "release", "initial_angle_deg": 60.0,
                                         "meta": {"initial_angle_deg": 60.0, "post_release_wind_m_s": 0.0}}
    for item in result.values():
        item["force"] = drag_force_from_speed(item["speed"], AIR_DENSITY, BALL_CD, AREA_M2)
    return result


def score(truth: np.ndarray, estimate: np.ndarray, t: np.ndarray, mask: np.ndarray,
          frequency_hz: float | None) -> dict:
    y, x = np.asarray(truth)[mask], np.asarray(estimate)[mask]
    err = x-y
    raw = {"rmse_m_s": float(np.sqrt(np.mean(err**2))),
           "bias_m_s": float(np.mean(err)),
           "p95_abs_error_m_s": float(np.quantile(np.abs(err), .95)),
           "max_abs_error_m_s": float(np.max(np.abs(err)))}
    if np.std(y) < 1e-10:
        return {**raw, "best_lag_s": 0.0, "shape_corr_after_shift": float("nan"),
                "lag_aligned_nrmse": float("nan"), "amplitude_ratio": float("nan")}
    max_lag_s = 2.0 if frequency_hz is None else min(2.0, .5/frequency_hz)
    max_lag = int(round(max_lag_s*FS))
    yc, xc = y-y.mean(), x-x.mean()
    best = (-np.inf, 0, None, None)
    for lag in range(-max_lag, max_lag+1):
        if lag > 0:
            ya, xa = yc[:-lag], xc[lag:]
        elif lag < 0:
            ya, xa = yc[-lag:], xc[:lag]
        else:
            ya, xa = yc, xc
        den = np.linalg.norm(ya)*np.linalg.norm(xa)
        corr = float(np.dot(ya, xa)/den) if den > 0 else -np.inf
        if corr > best[0]:
            best = (corr, lag, ya, xa)
    corr, lag, ya, xa = best
    denom = float(np.std(y))
    aligned_nrmse = float(np.sqrt(np.mean((xa-ya)**2))/denom) if denom else float("nan")
    if frequency_hz is not None:
        phase = 2.0*np.pi*frequency_hz*t[mask]
        design = np.column_stack((np.sin(phase), np.cos(phase), np.ones(len(phase))))
        truth_fit = np.linalg.lstsq(design, y, rcond=None)[0]
        estimate_fit = np.linalg.lstsq(design, x, rcond=None)[0]
        truth_amplitude = float(np.hypot(truth_fit[0], truth_fit[1]))
        estimate_amplitude = float(np.hypot(estimate_fit[0], estimate_fit[1]))
        amplitude_ratio = estimate_amplitude/truth_amplitude if truth_amplitude else float("nan")
    else:
        amplitude_ratio = float(np.std(x)/denom) if denom else float("nan")
    return {**raw, "best_lag_s": float(lag*DT), "shape_corr_after_shift": float(corr),
            "lag_aligned_nrmse": aligned_nrmse,
            "amplitude_ratio": amplitude_ratio}


def run() -> tuple[pd.DataFrame, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs()
    all_rows = []
    plots = {}
    coeff_report = {}
    for axis in ("IN", "OUT"):
        nominal = candidate_coefficients(axis)
        ratios = coefficient_ratios(axis)
        cases = model_cases(nominal, ratios)
        coeff_report[axis] = {"nominal": nominal, "ow04_ratios": ratios,
                              "case_b_Nm_s_rad": {k:v["viscous_damping_n_m_s_per_rad"] for k,v in cases.items()}}
        q = Q_BY_AXIS[axis]
        for input_name, data in inputs.items():
            t, true_speed, force = data["time"], data["speed"], data["force"]
            case_traces = {}
            for case_name, true_coeff in cases.items():
                if data["kind"] == "release":
                    initial_angle = math.radians(60.0)
                else:
                    initial_angle = math.atan(LEVER_M*float(force[0])/true_coeff["restoring_n_m_per_rad"])
                true_angle, contact = rk4_plant(t, force, true_coeff, initial_angle)
                measurements = {"ideal": true_angle}
                for seed in NOISE_SEEDS:
                    measurements[f"noise_seed_{seed}"] = apply_angle_sensor_model(
                        t, true_angle, SENSOR, seed,
                    )[0]
                for sensor_case, measured in measurements.items():
                    outputs = {}
                    static = static_wind_from_angle(measured, nominal)
                    outputs["Static"] = static
                    outputs["LPF"] = causal_lpf(static, LPF_CUTOFF_HZ[axis])
                    outputs["Static mean 1 s"] = centered_average(static, 1.0)
                    outputs["Static mean 3 s"] = centered_average(static, 3.0)
                    initial_state = np.array([measured[0], 0.0, 0.0])
                    force_hat = _fast_nonlinear_rts(
                        measured, nominal, DT, LEVER_M, ANGLE_NOISE_ASSUMED_RAD,
                        q, EPSILON_DEG_S, initial_state,
                    )
                    outputs["RTS"] = np.sign(force_hat)*np.sqrt(np.abs(force_hat)/DRAG_FACTOR)
                    mask = t >= data["eval_start"]
                    if data["kind"] == "sine":
                        freq = data["frequency_hz"]
                    else:
                        freq = None
                    for method in METHODS:
                        metrics = score(true_speed, outputs[method], t, mask, freq)
                        row = {
                            "axis": axis, "input": input_name, "plant_case": case_name,
                            "sensor_case": sensor_case, "sensor_seed": sensor_case.replace("noise_seed_", "") if sensor_case != "ideal" else "",
                            "method": method, "b_true_Nm_s_rad": true_coeff["viscous_damping_n_m_s_per_rad"],
                            "b_assumed_Nm_s_rad": nominal["viscous_damping_n_m_s_per_rad"],
                            "evaluation_start_s": data["eval_start"], "max_abs_angle_deg": float(np.max(np.abs(np.rad2deg(true_angle)))),
                            "hard_stop_contact_samples": int(contact.sum()), "hard_stop_contact": bool(contact.any()),
                            **metrics,
                        }
                        all_rows.append(row)
                    if sensor_case in ("ideal", f"noise_seed_{NOISE_SEEDS[0]}"):
                        case_traces[(case_name, sensor_case)] = {
                            "truth": true_speed, "angle": np.rad2deg(measured), **outputs,
                        }
            plots[axis, input_name] = (t, data, case_traces)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(OUT/"wbs23_observer_metrics.csv", index=False, float_format="%.9g", encoding="utf-8-sig")
    (OUT/"wbs23_coefficients_and_settings.json").write_text(json.dumps({
        "task": "OW-07 WBS2.3", "sphere_diameter_mm": BALL_D_M*1000,
        "sphere_mass_same_density_g": BALL_M0*(BALL_D_M/BALL_D0)**3*1000,
        "shaft_diameter_mm": 8.0, "grease_contact_length_mm": 30.0,
        "radial_gap_mm": 1.0, "sample_rate_hz": FS,
        "force_lever_m": LEVER_M, "aerodynamic_cd": BALL_CD,
        "assumed_angle_noise_deg": math.degrees(ANGLE_NOISE_ASSUMED_RAD),
        "sensor": {"bits":14,"full_scale_deg":360,"white_noise_std_deg":.015,
                   "coloured_noise_std_deg":.010,"coloured_time_constant_s":.475,"delay_s":.010,
                   "seeds":NOISE_SEEDS},
        "q_force_rw_N_per_sample":Q_BY_AXIS,"lpf_cutoff_hz":LPF_CUTOFF_HZ,
        "ow04_scenario":OW04_SCENARIO,"coefficient_cases":coeff_report,
        "input_meta":{k:{kk:vv for kk,vv in v.items() if kk not in ("time","speed","force")} for k,v in inputs.items()},
    }, indent=2