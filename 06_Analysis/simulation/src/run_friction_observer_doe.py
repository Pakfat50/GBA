"""Evaluate Coulomb-friction compensation and five-factor sensitivity."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from friction_estimators import (
    causal_luenberger_with_friction,
    kalman_rts_force_with_friction,
)
from model import PendulumParameters
from run_real_wind_doe import (
    CANDIDATES,
    derive_hardware_parameters,
    force_metrics,
    sensitivity_tables,
    wind_case,
    wind_speed_metrics,
    wind_factor_characteristics,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "ESO 3-state": "#2776bc",
    "ESO 4-state": "#e67e22",
    "RTS 3-state": "#159477",
    "RTS 4-state": "#006b4f",
}


def friction_plant(force_n, parameters, dt, friction_torque, epsilon):
    """Linear pendulum with smoothed Coulomb friction, integrated by RK4."""
    force_n = np.asarray(force_n, dtype=float)
    states = np.zeros((len(force_n), 2))

    def derivative(state, applied_force):
        theta, omega = state
        friction = friction_torque * np.tanh(omega / epsilon)
        acceleration = (
            parameters.force_lever_m * applied_force
            - parameters.damping_n_m_s_per_rad * omega
            - parameters.restoring_n_m_per_rad * theta
            - friction
        ) / parameters.inertia_kg_m2
        return np.array([omega, acceleration])

    for index in range(len(force_n) - 1):
        state = states[index]
        applied_force = force_n[index]
        k1 = derivative(state, applied_force)
        k2 = derivative(state + 0.5 * dt * k1, applied_force)
        k3 = derivative(state + 0.5 * dt * k2, applied_force)
        k4 = derivative(state + dt * k3, applied_force)
        states[index + 1] = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return states


def four_candidates(angle, parameters, dt, tuning, noise, friction, epsilon):
    _, rts3 = kalman_rts_force_with_friction(
        angle, parameters, dt, noise, tuning["rts3_process_noise"],
        friction, epsilon, 0,
    )
    _, rts4 = kalman_rts_force_with_friction(
        angle, parameters, dt, noise, tuning["rts4_process_noise"],
        friction, epsilon, 1,
    )
    return {
        "ESO 3-state": causal_luenberger_with_friction(
            angle, parameters, dt, tuning["eso3_pole_hz"], friction, epsilon, 0
        ),
        "ESO 4-state": causal_luenberger_with_friction(
            angle, parameters, dt, tuning["eso4_pole_hz"], friction, epsilon, 1
        ),
        "RTS 3-state": rts3,
        "RTS 4-state": rts4,
    }


def decode(nominal, nominal_friction, coded, half_ranges):
    ratios = 1.0 + np.asarray(coded) * half_ranges
    parameters = PendulumParameters(
        nominal.inertia_kg_m2 * ratios[0],
        nominal.damping_n_m_s_per_rad * ratios[1],
        nominal.restoring_n_m_per_rad * ratios[2],
        nominal.force_lever_m * ratios[3],
    )
    return parameters, nominal_friction * ratios[4], ratios


def read_old_effects(path):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (row["method"], row["factor"]): float(row["main_effect_range_m_s"])
        for row in rows
    }


def make_plots(output, nominal_rows, main_rows, factor_rows, factor_names):
    factor_labels = {"I": "I", "b": "b", "K": "K", "l": "l (lever)", "tau_f": "τf"}
    model_labels = {
        "without friction": "compensation OFF",
        "with friction": "compensation ON",
    }
    x = np.arange(len(CANDIDATES))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, awareness in enumerate(("without friction", "with friction")):
        rows = [r for r in nominal_rows if r["observer_model"] == awareness]
        force_values = [
            next(r["nrmse_fluctuation"] for r in rows if r["method"] == method)
            for method in CANDIDATES
        ]
        wind_values = [
            next(r["rmse_m_s"] for r in rows if r["method"] == method)
            for method in CANDIDATES
        ]
        axes[0].bar(
            x + (index - 0.5)*width, force_values, width,
            label=model_labels[awareness],
        )
        axes[1].bar(
            x + (index - 0.5)*width, wind_values, width,
            label=model_labels[awareness],
        )
    for axis, ylabel in zip(
        axes, ("Force fluctuation NRMSE", "Wind-speed RMSE [m/s]")
    ):
        axis.set_xticks(x, CANDIDATES, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Nominal V1.0-Light plant with Coulomb friction")
    fig.tight_layout()
    fig.savefig(output / "nominal_friction_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(factor_names))
    width = 0.19
    for method_index, method in enumerate(CANDIDATES):
        values = [
            next(
                r["main_effect_range"] for r in main_rows
                if r["method"] == method and r["factor"] == factor
            )
            for factor in factor_names
        ]
        axis.bar(
            x + (method_index - 1.5)*width, values, width,
            label=method, color=COLORS[method],
        )
    axis.set_xticks(x, [factor_labels[name] for name in factor_names])
    axis.set_ylabel("Main-effect range of force NRMSE")
    axis.set_title("Five-factor three-level DOE")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "doe_main_effects.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, factor in zip(axes.flat, factor_names):
        for method in CANDIDATES:
            rows = sorted(
                (r for r in factor_rows if r["method"] == method and r["factor"] == factor),
                key=lambda r: r["ratio"],
            )
            ratio = np.array([r["ratio"] for r in rows])
            mean = np.array([r["mean_wind_rmse_m_s"] for r in rows])
            low = np.array([r["p10_wind_rmse_m_s"] for r in rows])
            high = np.array([r["p90_wind_rmse_m_s"] for r in rows])
            axis.plot(ratio, mean, marker="o", label=method, color=COLORS[method])
            axis.fill_between(ratio, low, high, color=COLORS[method], alpha=0.10)
        axis.set_title(factor_labels[factor])
        axis.set_xlabel("Assumed / nominal coefficient")
        axis.set_ylabel("Wind-speed RMSE [m/s]")
        axis.grid(alpha=0.25)
    axes[1, 2].axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Wind-speed error; bands are conditional 10-90% ranges")
    fig.tight_layout()
    fig.savefig(output / "wind_rmse_factor_characteristics.png", dpi=180)
    plt.close(fig)


def run(config_path, output):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    derived = derive_hardware_parameters(config)
    nominal = PendulumParameters(**config["nominal_parameters"])
    for key in (
        "inertia_kg_m2", "damping_n_m_s_per_rad",
        "restoring_n_m_per_rad", "force_lever_m",
    ):
        if not np.isclose(getattr(nominal, key), derived[key], rtol=1e-10, atol=1e-14):
            raise ValueError(f"Nominal {key} is inconsistent with hardware derivation")
    median_acceleration = float(
        np.median(config["old_hardware_vc_friction_over_inertia_rad_s2"])
    )
    nominal_friction = config["nominal_friction_torque_n_m"]
    if not np.isclose(
        median_acceleration, config["nominal_friction_acceleration_rad_s2"], rtol=1e-12
    ):
        raise ValueError("Configured friction acceleration is not the VC median")
    if not np.isclose(
        median_acceleration * nominal.inertia_kg_m2, nominal_friction, rtol=1e-12
    ):
        raise ValueError("Configured friction torque is not I times tau_f/I")

    output.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / config["sample_rate_hz"]
    epsilon = np.deg2rad(config["friction_epsilon_deg_s"])
    time, speed, force, wind_metadata = wind_case(config, config["evaluation_seed"])
    state = friction_plant(force, nominal, dt, nominal_friction, epsilon)
    angle = state[:, 0]
    mask = (time >= config["evaluation_start_s"]) & (
        time <= config["evaluation_end_s"]
    )
    tuning = config["fixed_tuning_from_real_wind_doe"]
    noise = np.deg2rad(config["assumed_angle_noise_deg"])

    nominal_rows = []
    for label, assumed_friction in (
        ("without friction", 0.0),
        ("with friction", nominal_friction),
    ):
        estimates = four_candidates(
            angle, nominal, dt, tuning, noise, assumed_friction, epsilon
        )
        for method, estimate in estimates.items():
            _, wind_metrics = wind_speed_metrics(time, speed, estimate, mask, config)
            nominal_rows.append({
                "observer_model": label,
                "method": method,
                **force_metrics(force, estimate, mask),
                **wind_metrics,
            })
    write_csv(output / "nominal_comparison.csv", nominal_rows)

    factor_names = config["factor_names"]
    half_ranges = np.asarray(config["factor_half_ranges"], dtype=float)
    design = np.asarray(list(product((-1.0, 0.0, 1.0), repeat=len(factor_names))))
    design_rows = []
    for run_index, coded in enumerate(design):
        parameters, assumed_friction, ratios = decode(
            nominal, nominal_friction, coded, half_ranges
        )
        estimates = four_candidates(
            angle, parameters, dt, tuning, noise, assumed_friction, epsilon
        )
        for method, estimate in estimates.items():
            _, wind_metrics = wind_speed_metrics(time, speed, estimate, mask, config)
            design_rows.append({
                "run": run_index,
                "method": method,
                **{f"coded_{n}": float(v) for n, v in zip(factor_names, coded)},
                **{f"ratio_{n}": float(v) for n, v in zip(factor_names, ratios)},
                **force_metrics(force, estimate, mask),
                "wind_rmse_m_s": wind_metrics["rmse_m_s"],
            })
        if (run_index + 1) % 10 == 0 or run_index + 1 == len(design):
            print(f"DOE progress: {run_index + 1}/{len(design)}", flush=True)

    write_csv(output / "doe_results.csv", design_rows)
    main_rows, interaction_rows = sensitivity_tables(design_rows, factor_names)
    factor_rows = wind_factor_characteristics(design_rows, factor_names)
    write_csv(output / "doe_main_effects.csv", main_rows)
    write_csv(output / "doe_interactions.csv", interaction_rows)
    write_csv(output / "wind_rmse_factor_characteristics.csv", factor_rows)

    method_summary = []
    for method in CANDIDATES:
        rows = [r for r in design_rows if r["method"] == method]
        worst = max(rows, key=lambda r: r["nrmse_fluctuation"])
        method_summary.append({
            "method": method,
            "minimum_nrmse": min(r["nrmse_fluctuation"] for r in rows),
            "median_nrmse": float(np.median([r["nrmse_fluctuation"] for r in rows])),
            "maximum_nrmse": worst["nrmse_fluctuation"],
            "worst_parameter_ratios": {
                name: worst[f"ratio_{name}"] for name in factor_names
            },
        })

    old_effects = read_old_effects(
        ROOT / "results/real_wind_doe/wind_rmse_factor_characteristics.csv"
    )
    comparison = []
    for method in CANDIDATES:
        for factor in ("I", "b", "K", "l"):
            new = next(
                r["main_effect_range_m_s"] for r in factor_rows
                if r["method"] == method and r["factor"] == factor
            )
            old = old_effects[(method, factor)]
            comparison.append({
                "method": method, "factor": factor,
                "old_effect_m_s": old, "new_effect_m_s": new,
                "difference_m_s": new - old,
            })
    write_csv(output / "comparison_with_original_doe.csv", comparison)
    make_plots(output, nominal_rows, main_rows, factor_rows, factor_names)
    summary = {
        "scope": "V1.0-Light friction compensation and five-factor DOE",
        "status": "COMPLETE_AWAITING_USER_REVIEW",
        "friction_assumption": {
            "old_hardware_median_tau_over_I_rad_s2": median_acceleration,
            "new_hardware_nominal_torque_n_m": nominal_friction,
            "epsilon_deg_s": config["friction_epsilon_deg_s"],
            "observer_uncertainty_range_ratio": [0.0, 2.0],
        },
        "wind": wind_metadata,
        "fixed_tuning": tuning,
        "nominal_comparison": nominal_rows,
        "doe": {
            "design": "five-factor three-level full factorial",
            "points": len(design),
            "factors": factor_names,
            "half_ranges": config["factor_half_ranges"],
            "method_summary": method_summary,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/friction_observer_doe.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/friction_observer_doe",
    )
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
