"""OW-04: adopted係数のずれが風速推定へ与える影響を調べる。

このスクリプトは最適化を行わない。プラントはOW-03で採用した係数の
まま固定し、推定側の係数だけを、根拠のある幅で端点まで動かす。

比較は二段階で行う。
  1. 一因子ずつ変える（OAT）。I、K、τ、球抗力係数を個別に動かし、
     どの誤差が結果へ効きやすいかを切り分ける。
  2. 現実的な組合せを調べる。球質量と重心腕長からI/Kを同時に計算し、
     そこへτと球抗力係数の上下端を組み合わせる。IとKの相関を保つ。

比較対象は、OW-03で引き続き候補となった3状態ESO（オンライン）と
3状態RTS（オフライン）である。OUT軸4状態ESOはOW-03の結論どおり除外。
静的換算と因果LPFはKずれだけが作用する参考ベースラインとしてOATに加える。

実行方法（リポジトリのルートから）:
    python 06_Analysis/simulation/src/run_ow04_hbk_coefficient_sensitivity.py

生成物は results/observer_wind/ow04_coefficient_sensitivity/ に保存する。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from hbk_model_coefficients import select_coefficients, simulate_hbk_plant
from hbk_nonlinear_estimators import nonlinear_ekf_rts_force, nonlinear_luenberger_force
from run_ow03_hbk_observer_tuning import (
    CASE_LABELS,
    applied_force,
    best_parameter_values,
    force_to_speed,
    make_wind,
    metrics as calculate_metrics,
    peak_delay_s,
    static_force,
)
from run_real_wind_doe import causal_lowpass


SIM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SIM_ROOT.parents[1]
DEFAULT_CONFIG = SIM_ROOT / "config" / "ow04_hbk_coefficient_sensitivity.json"
DEFAULT_OUTPUT = SIM_ROOT / "results" / "observer_wind" / "ow04_coefficient_sensitivity"
OW03_CONFIG = SIM_ROOT / "config" / "ow03_hbk_observer_tuning.json"
OW03_SUMMARY = SIM_ROOT / "results" / "observer_wind" / "ow03_tuning" / "ow03_summary.json"

METHOD_COLORS = {"ESO 3状態": "#2776bc", "RTS 3状態（オフライン）": "#159477"}
METHOD_PLOT_LABELS = {"ESO 3状態": "ESO 3-state", "RTS 3状態（オフライン）": "RTS 3-state (offline)"}
CASE_PLOT_LABELS = {
    "独立Kaimal乱流 平均2 m/s TI20%": "Independent Kaimal (2 m/s)",
    "ガスト 2→6 m/s": "Gust (2 to 6 m/s)",
}
PARAMETERS = ("I", "K", "tau", "c_ball")
PARAMETER_LABELS = {"I": "I", "K": "K", "tau": "τ", "c_ball": "c_ball"}
GUST_NAME = "ガスト 2→6 m/s"


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write CSV with UTF-8 BOM so Japanese Excel installations open it cleanly."""
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_tau_leave_out_bounds(csv_path: Path) -> dict[str, tuple[float, float]]:
    """Read the IHB-03 leave-one-spacer-out sensitivity envelope.

    The percentages in that file are deviations from the adopted IHB-03
    one-integral estimate. OW-04 applies the same relative envelope around
    the corresponding coefficient in the OW-03 registry.
    """
    values: dict[str, list[float]] = {"IN": [], "OUT": []}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case"].startswith("leave_out_"):
                values[row["axis"]].append(float(row["change_percent"]) / 100.0)
    if any(not values[axis] for axis in values):
        raise ValueError("IHB-03 leave-out sensitivity is missing an axis")
    return {axis: (min(items), max(items)) for axis, items in values.items()}


def derive_parameter_bounds(config: dict, tau_relative: dict[str, tuple[float, float]],
                            nominal_by_axis: dict[str, dict]) -> tuple[dict, list[dict]]:
    """Turn measurement/fitting uncertainty into per-axis I/K/tau/c limits.

    A ball is treated as a solid sphere. Its contribution to pivot inertia is
    m*(ell**2 + 2*r**2/5). The sign for K follows the adopted IHB-05 equation:
    increasing the downward ball arm reduces restoring stiffness.
    """
    mass0 = config["ball_mass_nominal_g"] / 1000.0
    mass_error = config["ball_mass_error_g"] / 1000.0
    arm0 = config["ball_center_arm_nominal_mm"] / 1000.0
    arm_error = config["ball_center_arm_error_mm"] / 1000.0
    radius = config["ball_radius_m"]
    gravity = config["gravity_m_s2"]
    c_low, c_high = config["c_ball_bootstrap_95ci_n_m_s2_per_rad2"]
    bounds: dict[str, dict] = {}
    rows: list[dict] = []
    physical_corners = []

    # Four corners retain the physical correlation between mass and position.
    for mass_label, mass in (("mass_low", mass0 - mass_error), ("mass_high", mass0 + mass_error)):
        for arm_label, arm in (("arm_low", arm0 - arm_error), ("arm_high", arm0 + arm_error)):
            physical_corners.append({"name": f"{mass_label}_{arm_label}", "mass_kg": mass, "arm_m": arm})

    for axis, nominal in nominal_by_axis.items():
        inertia_base = nominal["inertia_kg_m2"] - mass0 * (arm0**2 + 2.0 * radius**2 / 5.0)
        stiffness_base = nominal["restoring_n_m_per_rad"] + mass0 * gravity * arm0
        physical = []
        for corner in physical_corners:
            inertia = inertia_base + corner["mass_kg"] * (corner["arm_m"]**2 + 2.0 * radius**2 / 5.0)
            stiffness = stiffness_base - corner["mass_kg"] * gravity * corner["arm_m"]
            physical.append({**corner, "I": inertia, "K": stiffness})
        i_values = [item["I"] for item in physical]
        k_values = [item["K"] for item in physical]
        tau_delta = tau_relative[axis]
        tau0 = nominal["tau_n_m"]
        bounds[axis] = {
            "I": (min(i_values), max(i_values)),
            "K": (min(k_values), max(k_values)),
            "tau": (tau0 * (1.0 + tau_delta[0]), tau0 * (1.0 + tau_delta[1])),
            "c_ball": (c_low, c_high),
            "physical_corners": physical,
        }
        for name in PARAMETERS:
            low, high = bounds[axis][name]
            nominal_value = {
                "I": nominal["inertia_kg_m2"],
                "K": nominal["restoring_n_m_per_rad"],
                "tau": tau0,
                "c_ball": nominal["ball_quadratic_drag_n_m_s2_per_rad2"],
            }[name]
            rows.append({
                "axis": axis, "parameter": name,
                "nominal": nominal_value, "lower": low, "upper": high,
                "lower_change_percent": 100.0 * (low / nominal_value - 1.0),
                "upper_change_percent": 100.0 * (high / nominal_value - 1.0),
            })
    return bounds, rows


def apply_parameter_changes(nominal: dict, changes: dict) -> dict:
    """Copy the adopted axis coefficients and replace only stated estimates."""
    model = deepcopy(nominal)
    if "I" in changes:
        model["inertia_kg_m2"] = changes["I"]
    if "K" in changes:
        model["restoring_n_m_per_rad"] = changes["K"]
    if "tau" in changes:
        model["tau_n_m"] = changes["tau"]
    if "c_ball" in changes:
        model["ball_quadratic_drag_n_m_s2_per_rad2"] = changes["c_ball"]
        model["total_quadratic_drag_n_m_s2_per_rad2"] = (
            model["rod_quadratic_drag_n_m_s2_per_rad2"] + changes["c_ball"]
        )
    return model


def make_oat_scenarios(axis: str, bounds: dict, nominal: dict) -> list[dict]:
    """Create low/high one-factor-at-a-time coefficient variants."""
    scenarios = []
    for parameter in PARAMETERS:
        for side, value in zip(("low", "high"), bounds[axis][parameter]):
            scenarios.append({
                "scenario": f"OAT_{parameter}_{side}",
                "scenario_type": "OAT",
                "parameter": parameter,
                "side": side,
                "changes": {parameter: value},
            })
    return scenarios


def make_joint_scenarios(axis: str, bounds: dict) -> list[dict]:
    """Create the 16 endpoint combinations without decoupling I and K.

    Four (mass, arm) corners give paired I/K estimates. Each is crossed with
    the two empirical tau endpoints and the two c_ball bootstrap endpoints.
    These are bounded sensitivity cases, not 95% joint confidence limits.
    """
    scenarios = []
    for corner in bounds[axis]["physical_corners"]:
        for tau_side, tau in zip(("low", "high"), bounds[axis]["tau"]):
            for c_side, c_ball in zip(("low", "high"), bounds[axis]["c_ball"]):
                label = (f"joint_{corner['name']}_tau-{tau_side}_c-{c_side}")
                scenarios.append({
                    "scenario": label,
                    "scenario_type": "joint_boundary",
                    "parameter": "I,K,tau,c_ball",
                    "side": "combined",
                    "mass_kg": corner["mass_kg"],
                    "arm_m": corner["arm_m"],
                    "changes": {"I": corner["I"], "K": corner["K"],
                                "tau": tau, "c_ball": c_ball},
                })
    return scenarios


def estimate_wind(method: str, angle: np.ndarray, model: dict, settings: dict,
                  config: dict, dt: float) -> np.ndarray:
    """Estimate wind speed with fixed OW-03 tuning and a supplied model copy."""
    lever = config["force_lever_m"]
    if method in ("静的換算", "因果LPF"):
        estimated_force = static_force(angle, model, lever)
        if method == "因果LPF":
            estimated_force = causal_lowpass(estimated_force, settings["lpf_hz"], dt)
    elif method == "ESO 3状態":
        estimated_force = nonlinear_luenberger_force(
            angle, model, dt, lever, settings["eso3_hz"], 0,
            config["friction_epsilon_deg_s"],
        )
    elif method == "RTS 3状態（オフライン）":
        _, estimated_force = nonlinear_ekf_rts_force(
            angle, model, dt, lever,
            np.deg2rad(config["assumed_angle_noise_deg"]),
            settings["rts3_q"], 0, config["friction_epsilon_deg_s"],
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    return force_to_speed(config, estimated_force)


def score_estimate(case_name: str, axis: str, method: str, scenario: str,
                   estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                   time_s: np.ndarray, max_angle: float, limit_deg: float,
                   bounded_limit: float, baseline_rmse: float | None = None) -> dict:
    """Compute common metrics and label estimates that ran away numerically."""
    metric = calculate_metrics(truth, estimate, mask)
    finite = bool(np.all(np.isfinite(estimate[mask])))
    max_estimate = float(np.max(np.abs(estimate[mask]))) if finite else math.inf
    bounded = finite and max_estimate <= bounded_limit
    rmse = metric["rmse_m_s"]
    if not finite:
        status = "非有限値・発散"
    elif not bounded:
        status = f"有界判定超過（>{bounded_limit:g} m/s）"
    else:
        status = "有界"
    if max_angle > limit_deg:
        status += "／実機角度範囲外"
    delay = peak_delay_s(time_s, truth, estimate, mask)
    delta_pct = None
    if baseline_rmse is not None and baseline_rmse > 1e-12 and rmse is not None:
        delta_pct = 100.0 * (rmse / baseline_rmse - 1.0)
    return {
        "case": case_name, "axis": axis, "method": method, "scenario": scenario,
        "status": status, "estimate_max_abs_m_s": max_estimate if finite else None,
        "rmse_m_s": rmse, "delta_rmse_percent": delta_pct,
        "bias_m_s": metric["bias_m_s"], "mae_m_s": metric["mae_m_s"],
        "p95_abs_error_m_s": metric["p95_abs_error_m_s"],
        "max_abs_error_m_s": metric["max_abs_error_m_s"],
        "peak_delay_s": delay, "max_angle_deg": max_angle,
        "within_angle_limit": max_angle <= limit_deg,
        "bounded_estimate": bounded,
    }


def build_plants(config: dict, ow03_config: dict) -> dict:
    """Generate each common OW-03 plant record once; never alter the plant."""
    dt = 1.0 / config["sample_rate_hz"]
    mask = np.arange(int(round(config["duration_s"] * config["sample_rate_hz"]))) / config["sample_rate_hz"] >= config["evaluation_start_s"]
    plants = {}
    for description in ow03_config["validation_winds"]:
        case = description["name"]
        time_s, speed, _ = make_wind(ow03_config, description)
        force = applied_force(ow03_config, speed)
        for axis in ow03_config["axis_names"]:
            coeff = select_coefficients(axis, config["physical_configuration"])
            state = simulate_hbk_plant(
                force, coeff, dt, force_lever_m=config["force_lever_m"],
                friction_epsilon_deg_s=config["friction_epsilon_deg_s"],
            )
            angle = state[:, 0]
            maximum_angle = float(np.max(np.abs(np.rad2deg(angle[mask]))))
            plants[(case, axis)] = {
                "time": time_s, "speed": speed, "force": force, "angle": angle,
                "mask": mask, "maximum_angle_deg": maximum_angle,
            }
    return plants


def evaluate_record(case: str, axis: str, method: str, scenario: dict,
                    nominal: dict, plant: dict, settings: dict,
                    config: dict, dt: float, limit: float,
                    baseline_rmse: float | None = None) -> dict:
    """Run one observer model variant on one unchanged plant data record."""
    observer_model = apply_parameter_changes(nominal, scenario["changes"])
    try:
        estimate = estimate_wind(method, plant["angle"], observer_model,
                                 settings, config, dt)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        # Keep one numerically problematic corner from aborting the full sweep.
        estimate = np.full_like(plant["speed"], np.nan)
    row = score_estimate(
        case, axis, method, scenario["scenario"], estimate, plant["speed"],
        plant["mask"], plant["time"], plant["maximum_angle_deg"], limit,
        config["bounded_estimate_limit_m_s"], baseline_rmse,
    )
    row.update({
        "scenario_type": scenario["scenario_type"],
        "parameter": scenario["parameter"], "side": scenario["side"],
        "observer_I_kg_m2": observer_model["inertia_kg_m2"],
        "observer_K_n_m_per_rad": observer_model["restoring_n_m_per_rad"],
        "observer_tau_n_m": observer_model["tau_n_m"],
        "observer_c_ball_n_m_s2_per_rad2": observer_model["ball_quadratic_drag_n_m_s2_per_rad2"],
        "mass_g": scenario.get("mass_kg", None) * 1000.0 if scenario.get("mass_kg") else None,
        "arm_mm": scenario.get("arm_m", None) * 1000.0 if scenario.get("arm_m") else None,
    })
    return row


def summarize_oat(rows: list[dict]) -> list[dict]:
    """Reduce low/high OAT endpoint pairs to their RMSE change interval."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["scenario_type"] == "OAT":
            key = (row["case"], row["axis"], row["method"], row["parameter"])
            groups.setdefault(key, []).append(row)
    summaries = []
    for (case, axis, method, parameter), items in groups.items():
        valid = [r for r in items if r["bounded_estimate"] and r["within_angle_limit"]
                 and r["delta_rmse_percent"] is not None]
        changes = [r["delta_rmse_percent"] for r in valid]
        absolute_changes = [r["rmse_m_s"] - r["baseline_rmse_m_s"] for r in valid]
        max_errors = [r["max_abs_error_m_s"] for r in valid if r["max_abs_error_m_s"] is not None]
        summaries.append({
            "case": case, "axis": axis, "method": method, "parameter": parameter,
            "valid_endpoints": len(valid),
            "delta_rmse_min_percent": min(changes) if changes else None,
            "delta_rmse_max_percent": max(changes) if changes else None,
            "most_adverse_abs_percent": max((abs(x) for x in changes), default=None),
            "delta_rmse_min_m_s": min(absolute_changes) if absolute_changes else None,
            "delta_rmse_max_m_s": max(absolute_changes) if absolute_changes else None,
            "max_abs_error_min_m_s": min(max_errors) if max_errors else None,
            "max_abs_error_max_m_s": max(max_errors) if max_errors else None,
            "baseline_rmse_m_s": items[0].get("baseline_rmse_m_s"),
        })
    return summaries


def summarize_joint(rows: list[dict]) -> list[dict]:
    """Report the envelope across the finite joint endpoint combinations."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["scenario_type"] == "joint_boundary":
            key = (row["case"], row["axis"], row["method"])
            groups.setdefault(key, []).append(row)
    summaries = []
    for (case, axis, method), items in groups.items():
        valid = [r for r in items if r["bounded_estimate"] and r["within_angle_limit"]]
        rmse_values = [r["rmse_m_s"] for r in valid if r["rmse_m_s"] is not None]
        delta_values = [r["delta_rmse_percent"] for r in valid if r["delta_rmse_percent"] is not None]
        max_errors = [r["max_abs_error_m_s"] for r in valid if r["max_abs_error_m_s"] is not None]
        summaries.append({
            "case": case, "axis": axis, "method": method,
            "scenarios_total": len(items), "scenarios_valid": len(valid),
            "baseline_rmse_m_s": items[0].get("baseline_rmse_m_s"),
            "rmse_min_m_s": min(rmse_values) if rmse_values else None,
            "rmse_max_m_s": max(rmse_values) if rmse_values else None,
            "rmse_median_m_s": float(np.median(rmse_values)) if rmse_values else None,
            "max_abs_error_min_m_s": min(max_errors) if max_errors else None,
            "max_abs_error_max_m_s": max(max_errors) if max_errors else None,
            "max_abs_error_median_m_s": float(np.median(max_errors)) if max_errors else None,
            "delta_rmse_min_percent": min(delta_values) if delta_values else None,
            "delta_rmse_max_percent": max(delta_values) if delta_values else None,
            "delta_rmse_median_percent": float(np.median(delta_values)) if delta_values else None,
        })
    return summaries


def save_plots(output: Path, oat_summary: list[dict], joint_summary: list[dict],
               sensitivity_cases: list[str]) -> None:
    """Save OAT influence and joint-boundary range plots."""
    primary_case = sensitivity_cases[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, layout="constrained")
    for ax, axis in zip(axes, ("IN", "OUT")):
        x = np.arange(len(PARAMETERS), dtype=float)
        for method, offset in (("ESO 3状態", -0.09), ("RTS 3状態（オフライン）", 0.09)):
            selected = [r for r in oat_summary if r["axis"] == axis and r["method"] == method
                        and r["case"] == primary_case]
            y_low, y_high = [], []
            for parameter in PARAMETERS:
                match = next((r for r in selected if r["parameter"] == parameter), None)
                y_low.append(np.nan if match is None else match["delta_rmse_min_m_s"])
                y_high.append(np.nan if match is None else match["delta_rmse_max_m_s"])
            lows, highs = np.asarray(y_low), np.asarray(y_high)
            center = (lows + highs) / 2.0
            yerr = np.vstack((center - lows, highs - center))
            ax.errorbar(x + offset, center, yerr=yerr, fmt="o", capsize=4,
                        color=METHOD_COLORS[method], label=METHOD_PLOT_LABELS[method])
        ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
        ax.set_xticks(x, [PARAMETER_LABELS[p] for p in PARAMETERS])
        ax.set_title(f"{axis} axis")
        ax.set_ylabel("Wind RMSE change from nominal [m/s]")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle(f"OW-04 one-factor-at-a-time effect ({CASE_PLOT_LABELS.get(primary_case, primary_case)})")
    fig.savefig(output / "ow04_oat_rmse_sensitivity.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, layout="constrained")
    for row_idx, axis in enumerate(("IN", "OUT")):
        for col_idx, method in enumerate(("ESO 3状態", "RTS 3状態（オフライン）")):
            ax = axes[row_idx, col_idx]
            items = [r for r in joint_summary if r["axis"] == axis and r["method"] == method]
            cases = list(dict.fromkeys(r["case"] for r in items))
            x = np.arange(len(cases))
            for case_idx, case in enumerate(cases):
                summary = next(r for r in items if r["case"] == case)
                low, high = summary["rmse_min_m_s"], summary["rmse_max_m_s"]
                median = summary["rmse_median_m_s"]
                if low is None or high is None or median is None:
                    continue
                ax.errorbar(case_idx, median, yerr=[[median - low], [high - median]],
                            fmt="o", capsize=5, color=METHOD_COLORS[method])
            ax.set_xticks(x, [CASE_PLOT_LABELS.get(c, c) for c in cases], rotation=15, ha="right")
            ax.set_title(f"{axis} / {METHOD_PLOT_LABELS[method]}")
            ax.set_ylabel("Wind RMSE [m/s]\n(min–median–max of 16 corners)")
            baseline = items[0]["baseline_rmse_m_s"] if items else None
            if baseline is not None:
                ax.axhline(baseline, color="#b34b35", linewidth=1.2, linestyle=":",
                           label="Nominal coefficients")
                ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("OW-04 combined coefficient-uncertainty envelope")
    fig.savefig(output / "ow04_joint_rmse_envelope.png", dpi=160)
    plt.close(fig)

def save_worst_case_waveforms(output: Path, config: dict, plants: dict,
                              nominal_by_axis: dict, settings_by_axis: dict,
                              joint_rows: list[dict], bounds: dict,
                              ow03_config: dict) -> list[dict]:
    """Plot each wind case's worst valid joint-boundary RMSE near its peak error."""
    dt = 1.0 / config["sample_rate_hz"]
    limit = ow03_config["mechanical_angle_limit_deg"]
    results = []
    safe_case_names = {
        "独立Kaimal乱流 平均2 m/s TI20%": "kaimal",
        "ガスト 2→6 m/s": "gust",
    }
    for case in config["sensitivity_cases"]:
        eligible = [r for r in joint_rows if r["case"] == case
                    and r["bounded_estimate"] and r["within_angle_limit"]
                    and r["rmse_m_s"] is not None]
        if not eligible:
            continue
        worst = max(eligible, key=lambda row: row["rmse_m_s"])
        axis, method = worst["axis"], worst["method"]
        plant = plants[(case, axis)]
        scenario = next(s for s in make_joint_scenarios(axis, bounds)
                        if s["scenario"] == worst["scenario"])
        model = apply_parameter_changes(nominal_by_axis[axis], scenario["changes"])
        estimate = estimate_wind(method, plant["angle"], model, settings_by_axis[axis],
                                 config, dt)
        error = estimate - plant["speed"]
        valid_indices = np.flatnonzero(plant["mask"] & np.isfinite(error))
        peak_index = int(valid_indices[np.argmax(np.abs(error[valid_indices]))])
        peak_time = float(plant["time"][peak_index])
        window_s = 2.0
        view = (plant["time"] >= peak_time - window_s) & (plant["time"] <= peak_time + window_s)
        stem = safe_case_names.get(case, f"case_{len(results)+1}")
        fig, (ax, err_ax) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1]},
                                         layout="constrained")
        ax.plot(plant["time"][view], plant["speed"][view], color="#222222", lw=1.5,
                label="True wind speed")
        ax.plot(plant["time"][view], estimate[view], color=METHOD_COLORS[method], lw=1.2,
                label=METHOD_PLOT_LABELS[method])
        ax.axvline(peak_time, color="#b34b35", ls="--", lw=1,
                   label=f"Peak |error| at {peak_time:.2f} s")
        ax.set_ylabel("Wind speed [m/s]")
        ax.set_title(f"Worst joint-boundary case: {CASE_PLOT_LABELS.get(case, case)}\n"
                     f"{axis} / {METHOD_PLOT_LABELS[method]} / RMSE {worst['rmse_m_s']:.4f} m/s")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        err_ax.plot(plant["time"][view], error[view], color=METHOD_COLORS[method], lw=1.0)
        err_ax.axhline(0, color="#555555", lw=0.8)
        err_ax.axvline(peak_time, color="#b34b35", ls="--", lw=1)
        err_ax.scatter([peak_time], [error[peak_index]], color="#b34b35", zorder=3)
        err_ax.set_xlabel("Time [s]")
        err_ax.set_ylabel("Estimate − true\n[m/s]")
        err_ax.grid(alpha=0.25)
        fig.savefig(output / f"ow04_worst_{stem}_timeseries.png", dpi=180)
        plt.close(fig)
        results.append({
            "case": case, "axis": axis, "method": method,
            "scenario": worst["scenario"], "rmse_m_s": worst["rmse_m_s"],
            "max_abs_error_m_s": worst["max_abs_error_m_s"],
            "error_at_peak_m_s": float(error[peak_index]),
            "mass_g": worst.get("mass_g"),
            "arm_mm": worst.get("arm_mm"),
            "coefficients_used": {
                "I_kg_m2": model["inertia_kg_m2"],
                "K_n_m_per_rad": model["restoring_n_m_per_rad"],
                "c_ball_n_m_s2_per_rad2": model["ball_quadratic_drag_n_m_s2_per_rad2"],
                "c_total_n_m_s2_per_rad2": model["total_quadratic_drag_n_m_s2_per_rad2"],
                "tau_n_m": model["tau_n_m"],
            },
            "coefficient_offsets_percent": {
                "I": 100.0 * (model["inertia_kg_m2"] / nominal_by_axis[axis]["inertia_kg_m2"] - 1.0),
                "K": 100.0 * (model["restoring_n_m_per_rad"] / nominal_by_axis[axis]["restoring_n_m_per_rad"] - 1.0),
                "c_ball": 100.0 * (model["ball_quadratic_drag_n_m_s2_per_rad2"] / nominal_by_axis[axis]["ball_quadratic_drag_n_m_s2_per_rad2"] - 1.0),
                "c_total": 100.0 * (model["total_quadratic_drag_n_m_s2_per_rad2"] / nominal_by_axis[axis]["total_quadratic_drag_n_m_s2_per_rad2"] - 1.0),
                "tau": 100.0 * (model["tau_n_m"] / nominal_by_axis[axis]["tau_n_m"] - 1.0),
            },
            "peak_error_time_s": peak_time, "plot_window_start_s": max(0.0, peak_time-window_s),
            "plot_window_end_s": peak_time+window_s,
        })
    return results

def write_report(path: Path, config: dict, bounds: dict, bound_rows: list[dict],
                 baseline_rows: list[dict], oat_summary: list[dict],
                 joint_summary: list[dict], worst_waveforms: list[dict],
                 nominal: dict, elapsed_s: float) -> None:
    """Write the Japanese OW-04 report from the generated tables."""
    lines = [
        "# OW-04 係数ずれに対する感度評価", "",
        "## 目的と結論", "",
        "OW-03で採用したBALL係数を真のプラントに固定し、推定側だけの係数をずらした。したがって今回の差は機械そのものの変化ではなく、推定モデルがI、K、τ、球抗力係数を正確に知らないときの影響である。OW-03で決めたゲイン・プロセス雑音・LPF遮断周波数は固定し、係数ずれに合わせた再調整や最適化は行っていない。センサーはOW-03同様に理想値である。", "",
        "OW-03の結果を引き継ぎ、OUT軸4状態ESOは比較対象から除外した。主要比較はオンラインの3状態ESOと、OW-03で最小誤差だった3状態RTS（オフライン）で行い、Kずれに限り静的換算・因果LPFも参考比較した。", "",
        "本レポートの誤差範囲は今回指定した風速モデルごとに算出している。具体的には、独立Kaimal乱流（平均2 m/s、TI 20%）と2→6 m/sガストについて、それぞれ同じ固定プラント波形を使い、推定側の係数端点シナリオ間でRMSEを比較した範囲である。したがって別の乱流系列、風速条件、実測風に対する範囲ではない。また係数誤差の確率分布や将来の実機RMSEを保証する信頼区間でもない。RTSは未来角度を使うため、実装時に利用できる方式であるESOとの順位をRMSEだけで決めない。", "",
        "## 係数のずれ幅と根拠", "",
        "IとKは、IHB-05の公称球質量3.9 g・有効重心腕180.61 mmに、許容した質量誤差±0.5 gと位置誤差±10 mmを与えて再計算した。この腕長は実測で確定した寸法ではなく、IHB-05の周期適合で得た有効値である。球の質量・位置はIとKを同時に変えるため、組合せ解析では同じ質量・位置からI/Kを一緒に算出した。", "",
        "τの上下端は、IHB-03でスペーサー条件を一つずつ外したときの同定値変化率を採用値へ適用した。球抗力係数c_ballは、IHB-05方式Aの波形クラスタ・ブートストラップ95%区間を使った。この区間は波形間のばらつきのみを表し、他係数や質量・寸法の不確かさを含まない。c_rodの理論値は固定した。", "",
        "| 軸 | 係数 | 採用値 | 下側端 | 変化率 | 上側端 | 変化率 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    units = {"I": "kg·m²", "K": "N·m/rad", "tau": "N·m", "c_ball": "N·m·s²/rad²"}
    nominal_index = {"I": "inertia_kg_m2", "K": "restoring_n_m_per_rad", "tau": "tau_n_m",
                     "c_ball": "ball_quadratic_drag_n_m_s2_per_rad2"}
    for row in bound_rows:
        lines.append(
            f"| {row['axis']} | {PARAMETER_LABELS[row['parameter']]} [{units[row['parameter']]}] | "
            f"{row['nominal']:.6g} | {row['lower']:.6g} | {row['lower_change_percent']:+.3f}% | "
            f"{row['upper']:.6g} | {row['upper_change_percent']:+.3f}% |"
        )
    lines += [
        "", "I/Kの物理的な組合せは、各軸について4つの質量・腕長端点（質量low/high × 腕長low/high）で作った。これをτのlow/high、c_ballのlow/highと組み合わせ、軸ごとに計16ケースを評価した。これは端点を組み合わせた保守的な感度範囲であり、同時確率95%区間ではない。", "",
        "## 比較条件", "",
        f"- プラント：OW-03の採用BALL係数で固定。係数、入力風、角度波形を感度ケース間で変更しない。",
        f"- センサー：理想角度。サンプリング {config['sample_rate_hz']:.0f} Hz、記録 {config['duration_s']:.0f} s、先頭{config['evaluation_start_s']:.0f} sは過渡として評価外。",
        "- 調整値：OW-03のESO極周波数およびRTSプロセス雑音設定を固定。係数ずれに対する再調整なし。",
        "- 風条件：OW-03の全5条件で公称基準を再確認。係数ずれのOAT・組合せ解析は、±60°内だった独立Kaimal平均2 m/sと2→6 m/sガストに限定。",
        "- 指標：風速RMSE、偏り、MAE、絶対誤差95パーセンタイル・最大値、ピーク時刻差。±60°外のプラント波形は実機性能の判断から除外。",
        "- 数値暴走：評価区間に非有限値がある、または推定絶対風速が20 m/sを越える場合は有界判定不合格とした。これはOW-03と同じ暴走除外目安で、数学的な安定性証明ではない。", "",
        "## 公称モデルでの基準値", "",
        "以下は今回使った独立Kaimal 2 m/sとガスト条件の公称モデル結果である。高風速Kaimal系列（平均3.75 m/s、最大6 m/s）は±60°を越えたため、他ケースの参考値とともに付属CSVへ残すが、性能比較からは除外した。", "",
        "| ケース | 軸 | 方式 | RMSE [m/s] | 最大絶対誤差 [m/s] | 偏り [m/s] | 95%絶対誤差 [m/s] | 最大角度 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    relevant_methods = ("ESO 3状態", "RTS 3状態（オフライン）", "静的換算", "因果LPF")
    for row in baseline_rows:
        if row["case"] not in config["sensitivity_cases"] or row["method"] not in relevant_methods:
            continue
        if row["rmse_m_s"] is None:
            vals = ("発散", "—", "—", "—")
        else:
            vals = (f"{row['rmse_m_s']:.5f}", f"{row['max_abs_error_m_s']:.5f}", f"{row['bias_m_s']:.5f}", f"{row['p95_abs_error_m_s']:.5f}")
        lines.append(f"| {row['case']} | {row['axis']} | {row['method']} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {row['max_angle_deg']:.2f}° |")
    lines += [
        "", "## 一因子ずつ動かした結果（OAT）", "",
        "OATでは対象の係数だけを下側端または上側端へ動かし、残りの係数を公称値に固定した。静的換算・因果LPFは式にKしか含まれないため、Kずれだけを比較した。表と図のRMSE変化率は、公称モデルに対する増減を表す。マイナスはその端点で偶然RMSEが小さくなったことを示すが、ずれた値を採用すべきという意味ではない。", "",
        "![係数一つずつの端点変化によるRMSE感度](ow04_oat_rmse_sensitivity.png)", "",
        "変化率は公称RMSEが小さい場合に過大表示となるため、表には変化率とRMSEの絶対変化を併記した。判断には絶対変化も見る。", "",
        "| ケース | 軸 | 方式 | 係数 | RMSE変化率 | RMSE絶対変化 [m/s] | 最大絶対誤差の範囲 [m/s] |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in oat_summary:
        if row["case"] != config["sensitivity_cases"][0]:
            continue
        value = "算出不可" if row["delta_rmse_min_percent"] is None else f"{row['delta_rmse_min_percent']:+.2f}% ～ {row['delta_rmse_max_percent']:+.2f}%"
        abs_value = "算出不可" if row["delta_rmse_min_m_s"] is None else f"{row['delta_rmse_min_m_s']:+.5f} ～ {row['delta_rmse_max_m_s']:+.5f}"
        peak_value = "算出不可" if row["max_abs_error_min_m_s"] is None else f"{row['max_abs_error_min_m_s']:.5f} ～ {row['max_abs_error_max_m_s']:.5f}"
        lines.append(f"| {row['case']} | {row['axis']} | {row['method']} | {PARAMETER_LABELS.get(row['parameter'], row['parameter'])} | {value} | {abs_value} | {peak_value} |")
    lines += [
        "", "## 複数係数を組み合わせた結果", "",
        "各ケースの16端点シナリオから得たRMSEの最小値・中央値・最大値を、公称係数のRMSE（赤い点線）と比較する。公称ガスト誤差がほぼ0のため、複合条件の主評価は不安定な変化率ではなく絶対RMSEとした。端点の組み合わせであるため、区間内のすべての結果を覆う保証や確率的な解釈はない。", "",
        "![物理量由来の係数ずれを組み合わせたときのRMSE範囲](ow04_joint_rmse_envelope.png)", "",
        "| ケース | 軸 | 方式 | 有効/16 | 公称RMSE [m/s] | 組合せRMSE min/中央値/max [m/s] | 最大絶対誤差 min/中央値/max [m/s] |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in joint_summary:
        rmse_text = "算出不可" if row["rmse_min_m_s"] is None else f"{row['rmse_min_m_s']:.5f} ～ {row['rmse_max_m_s']:.5f}"
        base = "—" if row["baseline_rmse_m_s"] is None else f"{row['baseline_rmse_m_s']:.5f}"
        if row["rmse_min_m_s"] is None:
            rmse_text = "算出不可"
        else:
            rmse_text = f"{row['rmse_min_m_s']:.5f} / {row['rmse_median_m_s']:.5f} / {row['rmse_max_m_s']:.5f}"
        max_error_text = "算出不可" if row["max_abs_error_min_m_s"] is None else f"{row['max_abs_error_min_m_s']:.5f} / {row['max_abs_error_median_m_s']:.5f} / {row['max_abs_error_max_m_s']:.5f}"
        lines.append(f"| {row['case']} | {row['axis']} | {row['method']} | {row['scenarios_valid']}/{row['scenarios_total']} | {base} | {rmse_text} | {max_error_text} |")
    lines += [
        "", "## 考察と次段階", "",
        "OATでは、I/KずれがESO 3状態に対しても明確に誤差を増やし、c_ballのずれは今回の範囲ではほとんど影響しなかった。RTS 3状態は特にI/Kに敏感で、独立Kaimal条件の非常に小さい公称RMSEを基準にすると変化率が大きく見えるため、絶対RMSEと合わせて読む必要がある。τの影響は主にOUT軸に現れた。", "",
        "組合せ端点では、係数ずれがあると両方式とも公称条件よりRMSEが増え、RTS 3状態の大きな公称優位は縮まった。よって、RTS 3状態がOW-03で優勢だった結論は係数一致・理想センサー条件では維持されるが、係数ずれを含む実環境での優位は未確定である。RTSは未来サンプルを使うオフライン方式なので、オンライン選定にはESO 3状態との比較が必要である。", "",
        "今回の端点は機械係数同定と既知の球質量・腕長条件から定めた。実際のI/K/τ/c_ballの誤差がこの幅を越えていないかは、係数の再計測で直接証明したわけではない。またc_ballのブートストラップ区間には質量・腕長の誤差が含まれない。このため複合端点の範囲は、感度を見落とさないためのシナリオであり、統計的な信頼区間としては解釈しない。", "",
        "## 風モデルごとの最大誤差条件と時系列", "",
        "次の図は、各風モデルで16個の複合係数端点のうちRMSEが最大となった有界・角度範囲内の条件を選び、その条件で絶対誤差が最大となる時刻を中心に前後2秒を拡大したものである。上段は真の風速と推定風速、下段は推定誤差を示す。対象にした風モデルは本解析で係数感度評価を行った2条件である。", "",
    ]
    for row in worst_waveforms:
        stem = "kaimal" if row["case"] == "独立Kaimal乱流 平均2 m/s TI20%" else "gust"
        lines += [
            f"### {row['case']}", "",
            f"最大RMSE条件は **{row['axis']}軸・{row['method']}・{row['scenario']}** で、RMSEは **{row['rmse_m_s']:.5f} m/s**。評価区間内の最大絶対誤差は **{row['max_abs_error_m_s']:.5f} m/s**、発生時刻は **{row['peak_error_time_s']:.2f} s**。"
            + (f"この時刻の符号付き誤差（推定値−真値）は **{row['error_at_peak_m_s']:+.5f} m/s** で、真値との差の向きも下段に示す。推定器に設定したOUT軸係数はI={row['coefficients_used']['I_kg_m2']:.6g} kg·m²（公称比{row['coefficient_offsets_percent']['I']:+.3f}%）、K={row['coefficients_used']['K_n_m_per_rad']:.6g} N·m/rad（{row['coefficient_offsets_percent']['K']:+.3f}%）、球抗力係数c_ball={row['coefficients_used']['c_ball_n_m_s2_per_rad2']:.6g} N·m·s²/rad²（{row['coefficient_offsets_percent']['c_ball']:+.3f}%）、τ={row['coefficients_used']['tau_n_m']:.6g} N·m（{row['coefficient_offsets_percent']['tau']:+.3f}%）だった。総二乗抗力係数cは{row['coefficients_used']['c_total_n_m_s2_per_rad2']:.6g} N·m·s²/rad²（{row['coefficient_offsets_percent']['c_total']:+.3f}%）。I/Kは球質量{row['mass_g']:.2f} g・有効腕長{row['arm_mm']:.2f} mmから同時算出し、ロッド抗力は理論値で固定した。プラント側は公称係数のまま。" if row["case"] == "独立Kaimal乱流 平均2 m/s TI20%" else "")
            + f"図は **{row['plot_window_start_s']:.2f}–{row['plot_window_end_s']:.2f} s** を表示する。", "",
            f"![{row['case']}で最大誤差となった条件の時系列拡大](ow04_worst_{stem}_timeseries.png)", "",
        ]
    lines += [
        "OW-05ではセンサーノイズ、量子化、遅延を加える。今回のOW-04は係数ずれだけを扱ったので、OW-05で係数ずれとセンサー非理想性を同時に加え、RTS 3状態（オフライン）とESO 3状態（オンライン）を実装条件も含めて最終比較する。", "",
        f"計算時間（この実行環境）：{elapsed_s:.1f} s。全ケースで同じ固定風波形・固定オブザーバー調整値を再利用し、反復最適化を行わず、OAT端点と16個の組合せだけを評価した。", "",
        "## 再実行", "",
        "リポジトリルートで次を実行する。入力誤差幅、使用ケース、評価対象方式は `config/ow04_hbk_coefficient_sensitivity.json` に記録している。", "",
        "```bash", "python 06_Analysis/simulation/src/run_ow04_hbk_coefficient_sensitivity.py", "```", "",
        "詳細な全シナリオ表は `ow04_oat_scenarios.csv`、`ow04_joint_scenarios.csv`、全条件公称値は `ow04_nominal_baseline.csv` を参照。", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, output_dir: Path) -> dict:
    """Execute baseline, OAT and finite joint-endpoint sensitivity evaluations."""
    started = time.perf_counter()
    config = read_json(config_path)
    ow03_config = read_json(OW03_CONFIG)
    ow03_summary = read_json(OW03_SUMMARY)
    # The sensitivity file owns uncertainty-specific settings; aerodynamic
    # inputs and validation wind definitions remain the reviewed OW-03 values.
    analysis_config = {**ow03_config, **config}
    dt = 1.0 / config["sample_rate_hz"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load physical nominal coefficients once. All plant simulations use these.
    nominal_by_axis = {
        axis: select_coefficients(axis, config["physical_configuration"])
        for axis in ow03_config["axis_names"]
    }
    tau_csv = (SIM_ROOT / config["tau_leave_one_configuration_out_csv"]).resolve()
    tau_relative = load_tau_leave_out_bounds(tau_csv)
    bounds, bound_rows = derive_parameter_bounds(config, tau_relative, nominal_by_axis)
    plants = build_plants(analysis_config, ow03_config)

    # Copy the OW-03 estimator settings. No settings are re-tuned in OW-04.
    settings_by_axis = {
        axis: best_parameter_values(ow03_summary["selected_tunings_by_axis"][axis])
        for axis in ow03_config["axis_names"]
    }
    max_angle_limit = ow03_config["mechanical_angle_limit_deg"]
    all_methods = ["静的換算", "因果LPF", *config["observer_methods"]]
    baseline_rows: list[dict] = []
    baseline_rmse: dict[tuple, float | None] = {}

    # 公称係数ですべてのOW-03 validation caseを再計算し、基準値を確認する。
    for case in (d["name"] for d in ow03_config["validation_winds"]):
        for axis in ow03_config["axis_names"]:
            plant = plants[(case, axis)]
            nominal = nominal_by_axis[axis]
            for method in all_methods:
                try:
                    estimate = estimate_wind(method, plant["angle"], nominal,
                                             settings_by_axis[axis], analysis_config, dt)
                except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                    estimate = np.full_like(plant["speed"], np.nan)
                row = score_estimate(
                    case, axis, method, "nominal", estimate, plant["speed"],
                    plant["mask"], plant["time"], plant["maximum_angle_deg"],
                    max_angle_limit, analysis_config["bounded_estimate_limit_m_s"],
                )
                baseline_rmse[(case, axis, method)] = row["rmse_m_s"]
                row["baseline_rmse_m_s"] = row["rmse_m_s"]
                baseline_rows.append(row)

    oat_rows: list[dict] = []
    joint_rows: list[dict] = []
    cases = config["sensitivity_cases"]
    for case in cases:
        for axis in ow03_config["axis_names"]:
            plant = plants[(case, axis)]
            nominal = nominal_by_axis[axis]
            base_scenarios = [{"scenario": "nominal", "scenario_type": "nominal",
                               "parameter": "none", "side": "nominal", "changes": {}}]
            scenarios = base_scenarios + make_oat_scenarios(axis, bounds, nominal)
            # Static conversion and LPF depend on K only. Their extra I/tau/c
            # runs would repeat an unchanged equation, so only K endpoints enter.
            static_methods = config["static_baselines_for_K_only"]
            k_only = [s for s in scenarios if s["parameter"] == "K"]
            for method in [*config["observer_methods"], *static_methods]:
                method_scenarios = (scenarios if method in config["observer_methods"]
                                    else base_scenarios + k_only)
                for scenario in method_scenarios:
                    row = evaluate_record(
                        case, axis, method, scenario, nominal, plant,
                        settings_by_axis[axis], analysis_config, dt, max_angle_limit,
                        baseline_rmse.get((case, axis, method)),
                    )
                    row["baseline_rmse_m_s"] = baseline_rmse.get((case, axis, method))
                    oat_rows.append(row)

            # Cross four physical m/arm corners with low/high tau and c_ball.
            for method in config["observer_methods"]:
                for scenario in make_joint_scenarios(axis, bounds):
                    row = evaluate_record(
                        case, axis, method, scenario, nominal, plant,
                        settings_by_axis[axis], analysis_config, dt, max_angle_limit,
                        baseline_rmse.get((case, axis, method)),
                    )
                    row["baseline_rmse_m_s"] = baseline_rmse.get((case, axis, method))
                    joint_rows.append(row)

    oat_summary = summarize_oat(oat_rows)
    joint_summary = summarize_joint(joint_rows)
    worst_waveforms = save_worst_case_waveforms(
        output_dir, analysis_config, plants, nominal_by_axis, settings_by_axis,
        joint_rows, bounds, ow03_config,
    )
    write_csv(output_dir / "ow04_coefficient_bounds.csv", bound_rows)
    write_csv(output_dir / "ow04_nominal_baseline.csv", baseline_rows)
    write_csv(output_dir / "ow04_oat_scenarios.csv", oat_rows)
    write_csv(output_dir / "ow04_oat_summary.csv", oat_summary)
    write_csv(output_dir / "ow04_joint_scenarios.csv", joint_rows)
    write_csv(output_dir / "ow04_joint_summary.csv", joint_summary)
    write_csv(output_dir / "ow04_worst_case_timeseries_summary.csv", worst_waveforms)
    save_plots(output_dir, oat_summary, joint_summary, cases)
    elapsed_s = time.perf_counter() - started
    summary = {
        "task_id": "OW-04", "configuration": config["physical_configuration"],
        "plant_coefficients_fixed": True, "observer_tuning_unchanged_from_ow03": True,
        "ideal_sensor": True, "methods": config["observer_methods"],
        "sensitivity_cases": cases, "uncertainty_bounds_by_axis": bounds,
        "oat_scenarios_count": len(oat_rows), "joint_scenarios_count": len(joint_rows),
        "nominal_baseline": baseline_rows, "oat_summary": oat_summary,
        "joint_summary": joint_summary, "worst_case_waveforms": worst_waveforms,
        "elapsed_s": elapsed_s,
    }
    (output_dir / "ow04_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir / "OW-04_REPORT.md", config, bounds, bound_rows,
        baseline_rows, oat_summary, joint_summary, worst_waveforms,
        nominal_by_axis, elapsed_s,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OW-04 coefficient mismatch sensitivity analysis.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    print(f"OAT rows: {result['oat_scenarios_count']}; joint rows: {result['joint_scenarios_count']}")
    print(f"Elapsed: {result['elapsed_s']:.1f} s")
    print(f"Report: {args.output_dir / 'OW-04_REPORT.md'}")


if __name__ == "__main__":
    main()
