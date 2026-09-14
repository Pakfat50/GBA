"""Compare four angle-only force estimators under Kaimal-spectrum wind."""
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
from scipy import signal
from scipy.stats import qmc

from doe import features
from estimators import causal_luenberger, kalman_rts_force
from model import PendulumParameters, natural_characteristics
from stage2_system import nonlinear_plant, plant
from wind import drag_force_from_speed, kaimal_longitudinal_psd, synthesize_kaimal_wind


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("ESO 3-state", "ESO 4-state", "RTS 3-state", "RTS 4-state")
BASELINES = ("Raw static", "Causal LPF")
COLORS = {
    "True force": "#111111",
    "Raw static": "#9e9e9e",
    "Causal LPF": "#8e6bbd",
    "ESO 3-state": "#2776bc",
    "ESO 4-state": "#e67e22",
    "RTS 3-state": "#159477",
    "RTS 4-state": "#006b4f",
}


def derive_hardware_parameters(config: dict) -> dict[str, float]:
    """Derive the dynamic coefficients from Sheet1 column E inputs."""
    source = config["hardware_parameter_source"]
    ball_lever = abs(source["ball_lever_m"])
    weight_lever = source["weight_lever_m"]
    rod_center = source["rod_center_m"]
    ball_mass = source["ball_mass_kg"]
    weight_mass = source["weight_mass_kg"]
    rod_mass = source["rod_mass_kg"]
    ball_radius = source["ball_diameter_m"] / 2.0
    rod_length = ball_lever + weight_lever
    gravity = source["gravity_m_s2"]

    inertia = (
        ball_mass * ball_lever**2
        + (2.0 / 5.0) * ball_mass * ball_radius**2
        + weight_mass * weight_lever**2
        + rod_mass * (rod_length**2 / 12.0 + rod_center**2)
    )
    restoring = gravity * (
        weight_mass * weight_lever
        + ball_mass * source["ball_lever_m"]
        + rod_mass * rod_center
    )
    damping = 2.0 * source["damping_ratio_assumption"] * np.sqrt(inertia * restoring)
    area = np.pi * source["ball_diameter_m"] ** 2 / 4.0
    return {
        "inertia_kg_m2": float(inertia),
        "damping_n_m_s_per_rad": float(damping),
        "restoring_n_m_per_rad": float(restoring),
        "force_lever_m": float(ball_lever),
        "projected_area_m2": float(area),
        "rod_length_assumption_m": float(rod_length),
        "ball_radius_used_for_inertia_m": float(ball_radius),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decode_parameters(
    nominal: PendulumParameters, coded: np.ndarray, half_ranges: np.ndarray
) -> PendulumParameters:
    ratio = 1.0 + np.asarray(coded, dtype=float) * half_ranges
    return PendulumParameters(
        nominal.inertia_kg_m2 * ratio[0],
        nominal.damping_n_m_s_per_rad * ratio[1],
        nominal.restoring_n_m_per_rad * ratio[2],
        nominal.force_lever_m * ratio[3],
    )


def force_metrics(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    error = estimate[mask] - truth[mask]
    truth_std = float(np.std(truth[mask]))
    return {
        "rmse_N": float(np.sqrt(np.mean(error**2))),
        "nrmse_fluctuation": float(np.sqrt(np.mean(error**2)) / truth_std),
        "bias_N": float(np.mean(error)),
        "mae_N": float(np.mean(np.abs(error))),
        "p95_abs_error_N": float(np.quantile(np.abs(error), 0.95)),
        "max_abs_error_N": float(np.max(np.abs(error))),
    }


def causal_lowpass(values: np.ndarray, cutoff_hz: float, dt: float) -> np.ndarray:
    pole = np.exp(-2.0 * np.pi * cutoff_hz * dt)
    result = np.asarray(values, dtype=float)
    for _ in range(3):
        result = signal.lfilter([0.0, 1.0 - pole], [1.0, -pole], result)
    return result


def wind_case(config: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    time, speed, metadata = synthesize_kaimal_wind(
        config["sample_rate_hz"],
        config["duration_s"],
        config["mean_wind_speed_m_s"],
        config["target_turbulence_intensity"],
        config["kaimal_integral_scale_m"],
        seed,
        config["maximum_wind_speed_m_s"],
    )
    force = drag_force_from_speed(
        speed,
        config["air_density_kg_m3"],
        config["drag_coefficient"],
        config["projected_area_m2"],
    )
    return time, speed, force, metadata


def four_candidates(
    angle: np.ndarray,
    parameters: PendulumParameters,
    dt: float,
    tuning: dict[str, float],
    angle_noise_rad: float,
) -> dict[str, np.ndarray]:
    _, rts3 = kalman_rts_force(
        angle, parameters, dt, angle_noise_rad, tuning["rts3_process_noise"], 0
    )
    _, rts4 = kalman_rts_force(
        angle, parameters, dt, angle_noise_rad, tuning["rts4_process_noise"], 1
    )
    return {
        "ESO 3-state": causal_luenberger(angle, parameters, dt, tuning["eso3_pole_hz"], 0),
        "ESO 4-state": causal_luenberger(angle, parameters, dt, tuning["eso4_pole_hz"], 1),
        "RTS 3-state": rts3,
        "RTS 4-state": rts4,
    }


def tune_estimators(
    angle: np.ndarray,
    force: np.ndarray,
    parameters: PendulumParameters,
    dt: float,
    mask: np.ndarray,
    config: dict,
) -> tuple[dict[str, float], list[dict]]:
    rows: list[dict] = []
    for order, label in ((0, "ESO 3-state"), (1, "ESO 4-state")):
        for value in config["eso_pole_grid_hz"]:
            estimate = causal_luenberger(angle, parameters, dt, value, order)
            rows.append(
                {
                    "method": label,
                    "parameter": value,
                    "parameter_name": "repeated_pole_hz",
                    **force_metrics(force, estimate, mask),
                }
            )
    angle_noise = np.deg2rad(config["assumed_angle_noise_deg"])
    for order, label, grid in (
        (0, "RTS 3-state", config["rts3_force_random_walk_grid_N_per_sample"]),
        (1, "RTS 4-state", config["rts4_force_rate_random_walk_grid_N_per_s_per_sample"]),
    ):
        for value in grid:
            _, estimate = kalman_rts_force(angle, parameters, dt, angle_noise, value, order)
            rows.append(
                {
                    "method": label,
                    "parameter": value,
                    "parameter_name": "force_process_noise",
                    **force_metrics(force, estimate, mask),
                }
            )
    for value in config["lpf_cutoff_grid_hz"]:
        raw = parameters.restoring_n_m_per_rad / parameters.force_lever_m * angle
        estimate = causal_lowpass(raw, value, dt)
        rows.append(
            {
                "method": "Causal LPF",
                "parameter": value,
                "parameter_name": "cutoff_hz",
                **force_metrics(force, estimate, mask),
            }
        )

    best = {}
    mapping = {
        "ESO 3-state": "eso3_pole_hz",
        "ESO 4-state": "eso4_pole_hz",
        "RTS 3-state": "rts3_process_noise",
        "RTS 4-state": "rts4_process_noise",
        "Causal LPF": "lpf_cutoff_hz",
    }
    for method, key in mapping.items():
        selected = min((row for row in rows if row["method"] == method), key=lambda r: r["nrmse_fluctuation"])
        best[key] = float(selected["parameter"])
    return best, rows


def sensitivity_tables(
    rows: list[dict], factor_names: list[str]
) -> tuple[list[dict], list[dict]]:
    main_rows: list[dict] = []
    interaction_rows: list[dict] = []
    for method in CANDIDATES:
        selected = [row for row in rows if row["method"] == method]
        overall = float(np.mean([row["nrmse_fluctuation"] for row in selected]))
        level_means: dict[tuple[int, int], float] = {}
        for index, factor in enumerate(factor_names):
            means = []
            for level in (-1, 0, 1):
                mean = float(
                    np.mean(
                        [row["nrmse_fluctuation"] for row in selected if int(row[f"coded_{factor}"]) == level]
                    )
                )
                level_means[(index, level)] = mean
                means.append(mean)
            main_rows.append(
                {
                    "method": method,
                    "factor": factor,
                    "mean_at_low": means[0],
                    "mean_at_center": means[1],
                    "mean_at_high": means[2],
                    "main_effect_range": max(means) - min(means),
                    "worst_level_mean": max(means),
                }
            )
        for first in range(len(factor_names)):
            for second in range(first + 1, len(factor_names)):
                residual = []
                for level_first in (-1, 0, 1):
                    for level_second in (-1, 0, 1):
                        cell = float(
                            np.mean(
                                [
                                    row["nrmse_fluctuation"]
                                    for row in selected
                                    if int(row[f"coded_{factor_names[first]}"]) == level_first
                                    and int(row[f"coded_{factor_names[second]}"]) == level_second
                                ]
                            )
                        )
                        additive = (
                            level_means[(first, level_first)]
                            + level_means[(second, level_second)]
                            - overall
                        )
                        residual.append(cell - additive)
                interaction_rows.append(
                    {
                        "method": method,
                        "interaction": f"{factor_names[first]}:{factor_names[second]}",
                        "interaction_rms": float(np.sqrt(np.mean(np.square(residual)))),
                        "interaction_max_abs": float(np.max(np.abs(residual))),
                    }
                )
    return main_rows, interaction_rows


def response_surface_validation(
    design: np.ndarray,
    design_rows: list[dict],
    validation: np.ndarray,
    validation_rows: list[dict],
) -> list[dict]:
    train_x, _ = features(design)
    test_x, _ = features(validation)
    output = []
    for method in CANDIDATES:
        train_y = np.log10(
            [row["rmse_N"] for row in design_rows if row["method"] == method]
        )
        test_y = np.array([row["rmse_N"] for row in validation_rows if row["method"] == method])
        coefficient = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
        train_prediction = train_x @ coefficient
        prediction = 10.0 ** (test_x @ coefficient)
        relative = np.abs(prediction - test_y) / test_y
        ss_res = float(np.sum((train_y - train_prediction) ** 2))
        ss_total = float(np.sum((train_y - np.mean(train_y)) ** 2))
        output.append(
            {
                "method": method,
                "train_log_rmse_r2": 1.0 - ss_res / ss_total,
                "validation_median_relative_error": float(np.median(relative)),
                "validation_max_relative_error": float(np.max(relative)),
                "gate_10_percent": bool(np.max(relative) <= 0.1),
            }
        )
    return output


def make_plots(
    output: Path,
    config: dict,
    time: np.ndarray,
    speed: np.ndarray,
    force: np.ndarray,
    estimates: dict[str, np.ndarray],
    mask: np.ndarray,
    wind_metadata: dict,
    main_rows: list[dict],
    linear_angle: np.ndarray,
    nonlinear_angle: np.ndarray,
    nonlinear_estimates: dict[str, np.ndarray],
) -> None:
    fs = config["sample_rate_hz"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    selected = mask & (time <= config["evaluation_start_s"] + 60.0)
    axes[0].plot(time[selected], speed[selected], color="#2776bc", linewidth=1.0)
    axes[0].set(ylabel="Wind speed [m/s]", title="Bounded Kaimal-spectrum wind")
    axes[1].plot(time[selected], 1000.0 * force[selected], color="black", linewidth=2.0, label="True force")
    for method in BASELINES + CANDIDATES:
        axes[1].plot(
            time[selected],
            1000.0 * estimates[method][selected],
            color=COLORS[method],
            linewidth=1.0,
            label=method,
        )
    axes[1].set(xlabel="Time [s]", ylabel="Force [mN]")
    axes[1].legend(ncol=3, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "real_wind_timeseries.png", dpi=180)
    plt.close(fig)

    frequency, measured_psd = signal.welch(
        speed - np.mean(speed), fs=fs, window="hann", nperseg=min(len(speed), 8192)
    )
    target_sigma = config["target_turbulence_intensity"] * config["mean_wind_speed_m_s"]
    target_psd = kaimal_longitudinal_psd(
        frequency,
        config["mean_wind_speed_m_s"],
        target_sigma,
        config["kaimal_integral_scale_m"],
    ) * wind_metadata["fluctuation_scale_for_bounds"] ** 2
    keep = frequency > 0.0
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.loglog(frequency[keep], target_psd[keep], "k--", linewidth=2, label="Kaimal target")
    ax.loglog(frequency[keep], measured_psd[keep], color="#2776bc", alpha=0.8, label="Generated record (Welch)")
    ax.axvline(50.0, color="#c23b73", linestyle=":", label="100 Hz Nyquist")
    ax.set(xlabel="Frequency [Hz]", ylabel="One-sided PSD [(m/s)^2/Hz]", title="Wind spectrum check")
    ax.grid(which="both", alpha=0.2)
    ax.legend()
    fig.savefig(output / "wind_spectrum.png", dpi=180)
    plt.close(fig)

    factor_names = config["factor_names"]
    matrix = np.array(
        [
            [next(row["main_effect_range"] for row in main_rows if row["method"] == method and row["factor"] == factor) for factor in factor_names]
            for method in CANDIDATES
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(factor_names)), factor_names)
    ax.set_yticks(range(len(CANDIDATES)), CANDIDATES)
    ax.set_title("DOE main-effect range of fluctuation-normalized RMSE")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="NRMSE range")
    fig.savefig(output / "doe_main_effects.png", dpi=180)
    plt.close(fig)

    selected = mask & (time <= config["evaluation_start_s"] + 60.0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    axes[0].plot(time[selected], np.rad2deg(linear_angle[selected]), color="#7f7f7f", label="Linear plant")
    axes[0].plot(time[selected], np.rad2deg(nonlinear_angle[selected]), color="#2776bc", label="Nonlinear plant")
    limit = config["mechanical_angle_limit_deg"]
    axes[0].axhline(limit, color="#c23b73", linestyle="--", label=f"Mechanical range {limit:g} deg")
    axes[0].axhline(-limit, color="#c23b73", linestyle="--")
    axes[0].set(ylabel="Angle [deg]", title="Linear and nonlinear plant check")
    axes[0].legend(ncol=3)
    axes[1].plot(time[selected], 1000.0 * force[selected], color="black", linewidth=2.0, label="True force")
    for method in BASELINES + CANDIDATES:
        axes[1].plot(
            time[selected],
            1000.0 * nonlinear_estimates[method][selected],
            color=COLORS[method],
            linewidth=1.0,
            label=method,
        )
    axes[1].set(xlabel="Time [s]", ylabel="Force [mN]")
    axes[1].legend(ncol=3, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "nonlinear_plant_check.png", dpi=180)
    plt.close(fig)


def make_worst_case_plot(
    output: Path,
    time: np.ndarray,
    force: np.ndarray,
    estimates: dict[str, np.ndarray],
    mask: np.ndarray,
    method_summary: list[dict],
) -> None:
    """Plot each estimator at its own worst DOE parameter combination."""
    selected_time = time[mask]
    truth_mn = 1000.0 * force[mask]
    fig, axes = plt.subplots(
        len(CANDIDATES), 1, figsize=(14, 12), sharex=True, sharey=True,
        layout="constrained"
    )
    summaries = {row["method"]: row for row in method_summary}
    all_values = [truth_mn]
    all_values.extend(1000.0 * estimates[method][mask] for method in CANDIDATES)
    y_min = min(float(np.min(values)) for values in all_values)
    y_max = max(float(np.max(values)) for values in all_values)
    padding = 0.05 * (y_max - y_min)

    for index, (axis, method) in enumerate(zip(axes, CANDIDATES)):
        estimate_mn = 1000.0 * estimates[method][mask]
        row = summaries[method]
        ratios = row["worst_parameter_ratios"]
        axis.fill_between(
            selected_time, truth_mn, estimate_mn, color=COLORS[method], alpha=0.12
        )
        axis.plot(selected_time, truth_mn, color="black", linewidth=1.6, label="True force")
        axis.plot(
            selected_time, estimate_mn, color=COLORS[method], linewidth=0.9,
            label="Worst-case estimate"
        )
        axis.set_ylabel("Force [mN]")
        axis.set_ylim(y_min - padding, y_max + padding)
        axis.set_title(
            f"{method}: worst DOE case, NRMSE={row['maximum_nrmse']:.3f}",
            loc="left"
        )
        axis.text(
            0.995, 0.92,
            f"I={ratios['I']:.2f}, b={ratios['b']:.2f}, "
            f"K={ratios['K']:.2f}, l={ratios['l']:.2f}",
            transform=axis.transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(ncol=2, loc="upper left")
    axes[-1].set_xlabel("Evaluation time [s]")
    fig.suptitle("Time histories at each estimator's worst DOE parameter case")
    fig.savefig(output / "worst_case_timeseries.png", dpi=100)
    plt.close(fig)


def run(config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    derived = derive_hardware_parameters(config)
    for key in ("inertia_kg_m2", "damping_n_m_s_per_rad", "restoring_n_m_per_rad", "force_lever_m"):
        if not np.isclose(config["nominal_parameters"][key], derived[key], rtol=1e-10, atol=1e-14):
            raise ValueError(f"nominal_parameters.{key} is inconsistent with Sheet1-E derivation")
    if not np.isclose(config["projected_area_m2"], derived["projected_area_m2"], rtol=1e-10):
        raise ValueError("projected_area_m2 is inconsistent with Sheet1-E ball diameter")
    nominal = PendulumParameters(**config["nominal_parameters"])
    dt = 1.0 / config["sample_rate_hz"]
    training_time, training_speed, training_force, training_wind = wind_case(config, config["training_seed"])
    evaluation_time, evaluation_speed, evaluation_force, evaluation_wind = wind_case(config, config["evaluation_seed"])
    training_angle = plant(training_force, nominal, dt)[:, 0]
    evaluation_state = plant(evaluation_force, nominal, dt)
    evaluation_angle = evaluation_state[:, 0]
    training_mask = (training_time >= config["evaluation_start_s"]) & (training_time <= config["evaluation_end_s"])
    evaluation_mask = (evaluation_time >= config["evaluation_start_s"]) & (evaluation_time <= config["evaluation_end_s"])
    tuning, tuning_rows = tune_estimators(
        training_angle, training_force, nominal, dt, training_mask, config
    )
    write_csv(output / "tuning_scan.csv", tuning_rows)
    angle_noise = np.deg2rad(config["assumed_angle_noise_deg"])
    candidate_nominal = four_candidates(evaluation_angle, nominal, dt, tuning, angle_noise)
    raw = nominal.restoring_n_m_per_rad / nominal.force_lever_m * evaluation_angle
    nominal_estimates = {
        "Raw static": raw,
        "Causal LPF": causal_lowpass(raw, tuning["lpf_cutoff_hz"], dt),
        **candidate_nominal,
    }
    nominal_rows = [
        {"method": method, **force_metrics(evaluation_force, values, evaluation_mask)}
        for method, values in nominal_estimates.items()
    ]
    write_csv(output / "nominal_metrics.csv", nominal_rows)

    nonlinear_state = nonlinear_plant(evaluation_force, nominal, dt)
    nonlinear_angle = nonlinear_state[:, 0]
    nonlinear_candidates = four_candidates(
        nonlinear_angle, nominal, dt, tuning, angle_noise
    )
    nonlinear_static = (
        nominal.restoring_n_m_per_rad
        / nominal.force_lever_m
        * np.tan(nonlinear_angle)
    )
    nonlinear_estimates = {
        "Raw static": nonlinear_static,
        "Causal LPF": causal_lowpass(nonlinear_static, tuning["lpf_cutoff_hz"], dt),
        **nonlinear_candidates,
    }
    nonlinear_rows = [
        {"method": method, **force_metrics(evaluation_force, values, evaluation_mask)}
        for method, values in nonlinear_estimates.items()
    ]
    write_csv(output / "nonlinear_nominal_metrics.csv", nonlinear_rows)

    half_ranges = np.asarray(config["factor_half_ranges"], dtype=float)
    factor_names = config["factor_names"]
    design = np.asarray(list(product((-1.0, 0.0, 1.0), repeat=4)))
    latin = 2.0 * qmc.LatinHypercube(d=4, seed=config["evaluation_seed"] + 1).random(
        config["validation_points"]
    ) - 1.0

    def evaluate_points(points: np.ndarray, kind: str) -> list[dict]:
        rows: list[dict] = []
        for run_index, coded in enumerate(points):
            parameters = decode_parameters(nominal, coded, half_ranges)
            estimates = four_candidates(evaluation_angle, parameters, dt, tuning, angle_noise)
            coded_columns = {f"coded_{name}": float(value) for name, value in zip(factor_names, coded)}
            ratio_columns = {
                f"ratio_{name}": float(1.0 + value * width)
                for name, value, width in zip(factor_names, coded, half_ranges)
            }
            for method, values in estimates.items():
                rows.append(
                    {
                        "set": kind,
                        "run": run_index,
                        "method": method,
                        **coded_columns,
                        **ratio_columns,
                        **force_metrics(evaluation_force, values, evaluation_mask),
                    }
                )
            if kind == "design" and (run_index + 1) % 10 == 0:
                print(f"DOE progress: {run_index + 1}/{len(points)}")
        return rows

    design_rows = evaluate_points(design, "design")
    validation_rows = evaluate_points(latin, "validation")
    write_csv(output / "doe_results.csv", design_rows)
    write_csv(output / "doe_validation_points.csv", validation_rows)
    main_rows, interaction_rows = sensitivity_tables(design_rows, factor_names)
    write_csv(output / "doe_main_effects.csv", main_rows)
    write_csv(output / "doe_interactions.csv", interaction_rows)
    surface_rows = response_surface_validation(design, design_rows, latin, validation_rows)
    write_csv(output / "response_surface_validation.csv", surface_rows)
    method_summary = []
    for method in CANDIDATES:
        selected = [row for row in design_rows if row["method"] == method]
        values = np.array([row["nrmse_fluctuation"] for row in selected])
        worst = max(selected, key=lambda row: row["nrmse_fluctuation"])
        method_summary.append(
            {
                "method": method,
                "minimum_nrmse": float(np.min(values)),
                "median_nrmse": float(np.median(values)),
                "maximum_nrmse": float(np.max(values)),
                "worst_parameter_ratios": {
                    name: worst[f"ratio_{name}"] for name in factor_names
                },
                "worst_case_metrics": {
                    key: worst[key]
                    for key in (
                        "rmse_N", "nrmse_fluctuation", "bias_N", "mae_N",
                        "p95_abs_error_N", "max_abs_error_N"
                    )
                },
            }
        )

    worst_case_estimates: dict[str, np.ndarray] = {}
    for row in method_summary:
        ratios = row["worst_parameter_ratios"]
        parameters = PendulumParameters(
            nominal.inertia_kg_m2 * ratios["I"],
            nominal.damping_n_m_s_per_rad * ratios["b"],
            nominal.restoring_n_m_per_rad * ratios["K"],
            nominal.force_lever_m * ratios["l"],
        )
        worst_case_estimates[row["method"]] = four_candidates(
            evaluation_angle, parameters, dt, tuning, angle_noise
        )[row["method"]]
        reproduced = force_metrics(
            evaluation_force, worst_case_estimates[row["method"]], evaluation_mask
        )
        if not np.isclose(
            reproduced["nrmse_fluctuation"], row["maximum_nrmse"], rtol=1e-12
        ):
            raise RuntimeError(f"Failed to reproduce worst DOE case for {row['method']}")

    aero = config["air_density_kg_m3"] * config["drag_coefficient"] * config["projected_area_m2"]
    force_at_maximum = 0.5 * aero * config["maximum_wind_speed_m_s"] ** 2
    angle_limit = np.deg2rad(config["mechanical_angle_limit_deg"])
    linear_force_limit = nominal.restoring_n_m_per_rad * angle_limit / nominal.force_lever_m
    nonlinear_force_limit = nominal.restoring_n_m_per_rad * np.tan(angle_limit) / nominal.force_lever_m
    limits = {
        "force_at_maximum_wind_N": float(force_at_maximum),
        "linear_static_angle_at_maximum_wind_deg": float(
            np.rad2deg(nominal.force_lever_m * force_at_maximum / nominal.restoring_n_m_per_rad)
        ),
        "nonlinear_static_angle_at_maximum_wind_deg": float(
            np.rad2deg(np.arctan(nominal.force_lever_m * force_at_maximum / nominal.restoring_n_m_per_rad))
        ),
        "linear_force_limit_at_mechanical_angle_N": float(linear_force_limit),
        "nonlinear_force_limit_at_mechanical_angle_N": float(nonlinear_force_limit),
        "linear_wind_limit_m_s": float(np.sqrt(2.0 * linear_force_limit / aero)),
        "nonlinear_wind_limit_m_s": float(np.sqrt(2.0 * nonlinear_force_limit / aero)),
    }
    max_linear_angle = float(np.max(np.abs(np.rad2deg(evaluation_angle[evaluation_mask]))))
    max_nonlinear_angle = float(np.max(np.abs(np.rad2deg(nonlinear_angle[evaluation_mask]))))
    summary = {
        "scope": "Stage 2 extension: ideal angle, no sensor model, angle-only estimators",
        "wind_model": "one-sided IEC-style Kaimal along-wind spectrum with bounded random-phase realization",
        "training_wind": training_wind,
        "evaluation_wind": evaluation_wind,
        "tuning": tuning,
        "hardware_parameter_source": config["hardware_parameter_source"],
        "hardware_parameter_derivation": derived,
        "tuning_at_search_boundary": {
            "ESO 3-state": tuning["eso3_pole_hz"]
            in (min(config["eso_pole_grid_hz"]), max(config["eso_pole_grid_hz"])),
            "ESO 4-state": tuning["eso4_pole_hz"]
            in (min(config["eso_pole_grid_hz"]), max(config["eso_pole_grid_hz"])),
            "RTS 3-state": tuning["rts3_process_noise"]
            in (
                min(config["rts3_force_random_walk_grid_N_per_sample"]),
                max(config["rts3_force_random_walk_grid_N_per_sample"]),
            ),
            "RTS 4-state": tuning["rts4_process_noise"]
            in (
                min(config["rts4_force_rate_random_walk_grid_N_per_s_per_sample"]),
                max(config["rts4_force_rate_random_walk_grid_N_per_s_per_sample"]),
            ),
        },
        "nominal_metrics": nominal_rows,
        "nonlinear_nominal_metrics": nonlinear_rows,
        "doe": {
            "design": "four-factor three-level full factorial",
            "design_points": len(design),
            "validation_points": len(latin),
            "factor_names": factor_names,
            "factor_half_ranges": config["factor_half_ranges"],
            "method_summary": method_summary,
            "response_surface_validation": surface_rows,
            "worst_case_timeseries": {
                "evaluation_start_s": config["evaluation_start_s"],
                "evaluation_end_s": config["evaluation_end_s"],
                "figure": "results/real_wind_doe/worst_case_timeseries.png",
                "note": "Each estimator uses its own maximum-NRMSE DOE case.",
            },
        },
        "plant": {
            "parameters": config["nominal_parameters"],
            "characteristics": natural_characteristics(nominal),
            "maximum_linear_dynamic_angle_deg_in_evaluation": max_linear_angle,
            "maximum_nonlinear_dynamic_angle_deg_in_evaluation": max_nonlinear_angle,
            "mechanical_angle_limit_deg": config["mechanical_angle_limit_deg"],
            "limits": limits,
        },
        "status": "COMPLETE_AWAITING_USER_REVIEW",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_plots(
        output,
        config,
        evaluation_time,
        evaluation_speed,
        evaluation_force,
        nominal_estimates,
        evaluation_mask,
        evaluation_wind,
        main_rows,
        evaluation_angle,
        nonlinear_angle,
        nonlinear_estimates,
    )
    make_worst_case_plot(
        output,
        evaluation_time,
        evaluation_force,
        worst_case_estimates,
        evaluation_mask,
        method_summary,
    )
    write_report(ROOT / "Real_Wind_DOE_Report.md", config, summary, main_rows, interaction_rows)
    return summary


def write_report(
    path: Path,
    config: dict,
    summary: dict,
    main_rows: list[dict],
    interaction_rows: list[dict],
) -> None:
    metrics = {row["method"]: row for row in summary["nominal_metrics"]}
    nonlinear_metrics = {
        row["method"]: row for row in summary["nonlinear_nominal_metrics"]
    }
    source = summary["hardware_parameter_source"]
    derived = summary["hardware_parameter_derivation"]
    limits = summary["plant"]["limits"]
    lines = [
        "# Stage 2追加評価：実風スペクトルと4推定器の係数感度",
        "",
        "## 結論の読み方",
        "",
        "この評価は、Kaimalスペクトルに従う風速を準定常抗力へ変換し、角度だけから外力を推定するPoCである。センサノイズ、機械ストッパー、非定常Cdはまだ含めていない。4方式の比較対象は3状態ESO、4状態ESO、3状態RTSスムーザー、4状態RTSスムーザーである。生の静的換算と因果LPFを基準として併記する。",
        "",
        "## 実風スペクトル",
        "",
        "地表層の実測から整理されたKaimalスペクトルを採用した。NREL資料に記載されたIEC形式の一方向スペクトルを使う。",
        "",
        "```math",
        "S_u(f)=\\frac{4\\sigma_u^2 L_u/U}{\\left(1+6fL_u/U\\right)^{5/3}}",
        "```",
        "",
        f"ここで、$`U`$は平均風速、$`\\sigma_u`$は風速変動の標準偏差、$`L_u`$は積分長さスケールである。今回は高さ2 mの代表値として$`L_u=8.1\\times0.7z=11.34`$ m、平均{config['mean_wind_speed_m_s']:.2f} m/s、目標乱流強度20%、上限{config['maximum_wind_speed_m_s']:.1f} m/sとした。平均値は、以前の5/8という平均／上限比を保つため3.75 m/sへ変更した。上限を守るため変動全体へ一つの倍率を掛け、スペクトル形状を保った。評価波形で実現した乱流強度は{100*summary['evaluation_wind']['realized_turbulence_intensity']:.2f}%である。",
        "",
        "- [Kaimal et al. (1972), Spectral characteristics of surface-layer turbulence](https://doi.org/10.1002/qj.49709841707)",
        "- [NREL, Sensitivity Analysis of Wind Characteristics and Wind Turbine Properties](https://www.nrel.gov/docs/fy19osti/74876.pdf)",
        "",
        "![風速スペクトル](results/real_wind_doe/wind_spectrum.png)",
        "",
        "抗力は次の準定常式で真値を作った。Cdは入力生成だけに用い、オブザーバーはCdや風速を知らない。",
        "",
        "```math",
        "F(t)=\\frac{1}{2}\\rho C_d A U(t)\\lvert U(t)\\rvert",
        "```",
        "",
        "## V1.0-Lightの装置パラメータ",
        "",
        f"入力値は `{source['workbook']}` の `{source['sheet']}`、{source['column']}列 `{source['variant']}` から読み取った。シートの `r=0.1 m` は名称上は半径だが、面積式が $`S=\\pi r^2/4`$ なので直径として扱った。元ブック自体は変更していない。",
        "",
        "| 量 | 使用値 | 根拠 |",
        "|---|---:|---|",
        f"| 球側作用距離 $`l`$ | {derived['force_lever_m']:.6f} m | E2の絶対値 |",
        f"| 投影面積 $`S`$ | {derived['projected_area_m2']:.9f} m² | E10を直径として$`\\pi d^2/4`$ |",
        f"| 復元係数 $`K`$ | {derived['restoring_n_m_per_rad']:.9f} N m/rad | E列の質量・重心位置から導出 |",
        f"| 慣性モーメント $`I`$ | {derived['inertia_kg_m2']:.9f} kg m² | 中実球＋一様棒＋錘の剛体近似 |",
        f"| 減衰係数 $`b`$ | {derived['damping_n_m_s_per_rad']:.9f} N m s/rad | 旧機の暫定減衰比$`\\zeta={source['damping_ratio_assumption']:.4f}`$を維持 |",
        "",
        "導出式は次の通りである。ロッド長はE列の球側・錘側距離の和と仮定した。",
        "",
        "```math",
        "K=g(m_w l_w+m_b l_b+m_l l_l)",
        "```",
        "",
        "```math",
        "I=m_b|l_b|^2+\\frac{2}{5}m_b\\left(\\frac{d}{2}\\right)^2+m_wl_w^2+m_l\\left(\\frac{(|l_b|+l_w)^2}{12}+l_l^2\\right)",
        "```",
        "",
        "```math",
        "b=2\\zeta\\sqrt{IK}",
        "```",
        "",
        "このIとbはシートに直接記載された値ではない。Iは部材を剛体近似した暫定値、bは旧機の減衰比を引き継いだ暫定値である。次号機の自由減衰試験またはCAD慣性値が得られたら更新する。",
        "",
        "## 6 m/sと45度の範囲確認",
        "",
        f"6 m/sでの抗力は{1000*limits['force_at_maximum_wind_N']:.3f} mNである。シートと同じ非線形静力学では角度は{limits['nonlinear_static_angle_at_maximum_wind_deg']:.2f}度となり、約45度という設計意図と一致する。一方、小角度線形モデルでは{limits['linear_static_angle_at_maximum_wind_deg']:.2f}度となる。45度域では線形近似誤差を無視できない。45度に対応する風速は、非線形式で{limits['nonlinear_wind_limit_m_s']:.2f} m/s、線形式で{limits['linear_wind_limit_m_s']:.2f} m/sである。",
        "",
        f"係数感度DOEは従来との比較性を保つため、プラントと推定器が同じ線形モデルを使う条件を維持した。別途、$`I\\ddot{{\\theta}}+b\\dot{{\\theta}}+K\\sin\\theta=lF\\cos\\theta`$ の非線形プラントを計算した。評価区間の最大動的角度は線形モデルで{summary['plant']['maximum_linear_dynamic_angle_deg_in_evaluation']:.2f}度、非線形モデルで{summary['plant']['maximum_nonlinear_dynamic_angle_deg_in_evaluation']:.2f}度である。非線形モデルでも約45度を{summary['plant']['maximum_nonlinear_dynamic_angle_deg_in_evaluation']-config['mechanical_angle_limit_deg']:.2f}度上回ったため、6 m/sを静的に44度へ合わせるだけでは過渡余裕がない。実機仕様ではストッパー余裕、最大運用風速、または減衰の見直しが必要である。",
        "",
        "![非線形プラント確認](results/real_wind_doe/nonlinear_plant_check.png)",
        "",
        "## 4方式と調整",
        "",
        "全方式の観測は角度だけであり、角速度は入力しない。ESOは公称学習波形で反復極を探索する。RTSは同じ公称学習波形で外力または外力変化率へ与えるプロセス雑音を探索する。評価波形とDOEでは選択値を固定した。",
        "",
        "| 方式 | 状態 | 外力仮定 | 因果性 | 選択値 |",
        "|---|---|---|---|---:|",
        f"| ESO 3-state | 角度・角速度・トルク | 区間内で一定 | 因果 | {summary['tuning']['eso3_pole_hz']:.6g} Hz |",
        f"| ESO 4-state | 上記＋トルク変化率 | 区間内で一定勾配 | 因果 | {summary['tuning']['eso4_pole_hz']:.6g} Hz |",
        f"| RTS 3-state | 角度・角速度・トルク | トルクrandom walk | 非因果 | {summary['tuning']['rts3_process_noise']:.6g} N/sample |",
        f"| RTS 4-state | 上記＋トルク変化率 | 変化率random walk | 非因果 | {summary['tuning']['rts4_process_noise']:.6g} (N/s)/sample |",
        "",
        "探索上限に達した方式がある場合、理想角度・ノイズなしでは高帯域化の罰則が十分に現れていないことを意味する。この選択値を実機の推奨値とは扱わない。センサノイズを有効にしたStage 3で再調整する。",
        "",
        "## 公称条件の結果",
        "",
        "RMSEは全外力の誤差、NRMSEは外力変動の標準偏差でRMSEを割った値である。平均抗力が大きい条件でも動的誤差を過小評価しにくい後者を主指標にする。",
        "",
        "| 方式 | RMSE [mN] | 変動基準NRMSE | Bias [mN] | 95%絶対誤差 [mN] |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in BASELINES + CANDIDATES:
        row = metrics[method]
        lines.append(
            f"| {method} | {1000*row['rmse_N']:.4f} | {row['nrmse_fluctuation']:.4f} | {1000*row['bias_N']:.4f} | {1000*row['p95_abs_error_N']:.4f} |"
        )
    lines += [
        "",
        "![時系列](results/real_wind_doe/real_wind_timeseries.png)",
        "",
        "## 非線形プラントでの補助確認",
        "",
        "非線形プラントに対しては、静的換算だけ$`F=K\\tan\\theta/l`$を用いた。ESOとRTSは線形モデルのままであり、下表には45度域の構造的モデル差も含まれる。DOEの係数感度とは別の確認である。",
        "",
        "| 方式 | RMSE [mN] | 変動基準NRMSE | Bias [mN] | 95%絶対誤差 [mN] |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in BASELINES + CANDIDATES:
        row = nonlinear_metrics[method]
        lines.append(
            f"| {method} | {1000*row['rmse_N']:.4f} | {row['nrmse_fluctuation']:.4f} | {1000*row['bias_N']:.4f} | {1000*row['p95_abs_error_N']:.4f} |"
        )
    lines += [
        "",
        "## 係数誤差DOE",
        "",
        "I ±25%、b ±50%、K ±10%、作用距離l ±5%を低・中央・高の3水準とし、3の4乗=81条件を直接計算した。推定側の係数だけを変更し、プラント、風入力、調整値は固定した。各因子の感度値は、その因子の3水準ごとに他因子全条件を平均したNRMSEの最大値と最小値の差である。",
        "",
        "| 方式 | 81条件の最小NRMSE | 中央値 | 最大値 | 最大値となるI / b / K / l倍率 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["doe"]["method_summary"]:
        ratios = row["worst_parameter_ratios"]
        lines.append(
            f"| {row['method']} | {row['minimum_nrmse']:.4f} | {row['median_nrmse']:.4f} | {row['maximum_nrmse']:.4f} | {ratios['I']:.2f} / {ratios['b']:.2f} / {ratios['K']:.2f} / {ratios['l']:.2f} |"
        )
    lines += [
        "",
        "## ワースト係数条件の時系列",
        "",
        f"各推定器について、81条件のうちNRMSEが最大となった係数組合せを個別に再現した。横軸は評価に使った{config['evaluation_start_s']:.0f}～{config['evaluation_end_s']:.0f}秒であり、黒線が真の外力、色線が推定外力、塗りつぶしが両者の差である。各方式はそれぞれ異なるワースト係数条件なので、単一の共通ハードウェア条件を表す図ではない。",
        "",
        "| 方式 | I / b / K / l倍率 | RMSE [mN] | NRMSE | 最大絶対誤差 [mN] |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["doe"]["method_summary"]:
        ratios = row["worst_parameter_ratios"]
        metrics_worst = row["worst_case_metrics"]
        lines.append(
            f"| {row['method']} | {ratios['I']:.2f} / {ratios['b']:.2f} / {ratios['K']:.2f} / {ratios['l']:.2f} | "
            f"{1000*metrics_worst['rmse_N']:.4f} | {metrics_worst['nrmse_fluctuation']:.4f} | "
            f"{1000*metrics_worst['max_abs_error_N']:.4f} |"
        )
    lines += [
        "",
        "![ワースト係数条件の時系列](results/real_wind_doe/worst_case_timeseries.png)",
        "",
        "## 主効果感度順位",
        "",
        "| 方式 | 1位 | 2位 | 3位 | 4位 |",
        "|---|---|---|---|---|",
    ]
    for method in CANDIDATES:
        ranked = sorted(
            (row for row in main_rows if row["method"] == method),
            key=lambda row: row["main_effect_range"],
            reverse=True,
        )
        cells = [f"{row['factor']} ({row['main_effect_range']:.4f})" for row in ranked]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "![主効果](results/real_wind_doe/doe_main_effects.png)",
        "",
        "交互作用は、2因子の各3×3セル平均から加法的な主効果を引いた残差で評価した。値はdoe_interactions.csvに保存した。二次応答曲面は独立Latin Hypercube点で確認し、10%最大相対誤差ゲートを満たした場合だけ補間用に使える。直接計算した81条件の感度順位は、このゲートに依存しない。",
        "",
        "| 方式 | 学習log(RMSE) R2 | 確認中央値誤差 | 確認最大誤差 | 10%ゲート |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["doe"]["response_surface_validation"]:
        lines.append(
            f"| {row['method']} | {row['train_log_rmse_r2']:.4f} | {100*row['validation_median_relative_error']:.2f}% | {100*row['validation_max_relative_error']:.2f}% | {'PASS' if row['gate_10_percent'] else 'NOT_MET'} |"
        )
    strongest = sorted(interaction_rows, key=lambda row: row["interaction_rms"], reverse=True)[:4]
    lines += [
        "",
        "強い交互作用の上位4件は次の通りである。",
        "",
        "| 方式 | 交互作用 | RMS | 最大絶対値 |",
        "|---|---|---:|---:|",
    ]
    for row in strongest:
        lines.append(
            f"| {row['method']} | {row['interaction']} | {row['interaction_rms']:.5f} | {row['interaction_max_abs']:.5f} |"
        )
    lines += [
        "",
        "## 再現方法",
        "",
        "```bash",
        "python 06_Analysis/simulation/src/run_real_wind_doe.py",
        "python -m unittest discover -s 06_Analysis/simulation/tests -v",
        "```",
        "",
        "設定はsimulation/config/real_wind_doe.json、数値結果はsimulation/results/real_wind_doeに保存する。Iとbの暫定導出式、E列の転記値、平均・最大風速も設定ファイルに保存した。Stage 3では独立したセンサモデルを追加し、同じ4方式を再調整して比較する。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/real_wind_doe.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/real_wind_doe")
    args = parser.parse_args()
    summary = run(args.config, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
