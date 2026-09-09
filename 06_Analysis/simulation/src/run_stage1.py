"""Run the Stage 1 matched-model verification and generate review artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from model import (
    PendulumParameters,
    continuous_matrices,
    exact_discretization,
    natural_characteristics,
    observability_matrix,
)
from observer import (
    ackermann_observer_gain,
    continuous_repeated_pole_gain,
    normalized_state_error,
    simulate_matched_model,
)


HERE = Path(__file__).resolve().parent
SIMULATION_ROOT = HERE.parent


def _box(ax, xy, width, height, text, color):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="#263238",
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10)


def _arrow(ax, start, end, label=""):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#37474f"))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, label, ha="center", fontsize=9)


def plot_block_diagram(output_path: Path) -> None:
    """Save a renderer-independent PNG of the Stage 1 signal flow."""

    fig, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.03, 0.65), 0.18, 0.18, "Constant torque state\ntau dot = 0", "#fff3cd")
    _box(ax, (0.32, 0.65), 0.21, 0.18, "Matched linear plant\nx[k+1] = Ad x[k]", "#d9edf7")
    _box(ax, (0.68, 0.65), 0.20, 0.18, "Ideal angle\ny[k] = C x[k]", "#e2f0d9")
    _box(ax, (0.57, 0.18), 0.32, 0.20, "Discrete observer\nxhat[k+1] = Ad xhat[k]\n+ Ld (y[k] - C xhat[k])", "#e8dff5")
    _box(ax, (0.13, 0.18), 0.23, 0.20, "State comparison\nx[k] - xhat[k]", "#f8d7da")
    _arrow(ax, (0.21, 0.74), (0.32, 0.74), "tau")
    _arrow(ax, (0.53, 0.74), (0.68, 0.74), "theta")
    _arrow(ax, (0.78, 0.65), (0.74, 0.38), "measured angle")
    _arrow(ax, (0.57, 0.28), (0.36, 0.28), "xhat")
    _arrow(ax, (0.42, 0.65), (0.25, 0.38), "true x")
    ax.text(0.5, 0.94, "GBA Stage 1: matched-model verification", ha="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.04, "No sensor model, no noise, no parameter mismatch, one linear axis", ha="center", fontsize=10, color="#455a64")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def first_permanent_below(time: np.ndarray, error: np.ndarray, threshold: float, start_index: int = 0) -> float | None:
    """Return the first time after which every error sample stays below threshold."""

    within = np.max(error, axis=1) <= threshold
    suffix = np.logical_and.accumulate(within[::-1])[::-1]
    indices = np.flatnonzero(suffix & (np.arange(len(time)) >= start_index))
    return None if len(indices) == 0 else float(time[indices[0]])


def run(config_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    p = config["parameters"]
    parameters = PendulumParameters(
        inertia_kg_m2=p["inertia_kg_m2"],
        damping_n_m_s_per_rad=p["damping_n_m_s_per_rad"],
        restoring_n_m_per_rad=p["restoring_n_m_per_rad"],
        force_lever_m=p["force_lever_m"],
    )
    fs = float(config["sample_rate_hz"])
    dt = 1.0 / fs
    samples = int(round(config["duration_s"] * fs)) + 1
    time = np.arange(samples) * dt
    a, c = continuous_matrices(parameters)
    ad = exact_discretization(a, dt)
    observability_rank = int(np.linalg.matrix_rank(observability_matrix(a, c)))

    pole_rad_s = 2.0 * np.pi * float(config["observer"]["repeated_pole_hz"])
    target_z = float(np.exp(-pole_rad_s * dt))
    desired_discrete_poles = np.full(3, target_z)
    ld = ackermann_observer_gain(ad, c, desired_discrete_poles)
    lc = continuous_repeated_pole_gain(
        parameters.inertia_kg_m2,
        parameters.damping_n_m_s_per_rad,
        parameters.restoring_n_m_per_rad,
        pole_rad_s,
    )
    actual_poles = np.linalg.eigvals(ad - ld @ c)
    pole_cluster_spread = float(np.max(np.abs(actual_poles - target_z)))
    actual_characteristic = np.poly(ad - ld @ c)
    target_characteristic = np.poly(desired_discrete_poles)
    characteristic_error = float(np.max(np.abs(actual_characteristic - target_characteristic)))

    initial_true = np.array([np.deg2rad(7.0), 0.15, 0.0020])
    identity_x, identity_xhat, _ = simulate_matched_model(
        ad, c, ld, initial_true, initial_true.copy(), samples
    )
    scales = np.array([np.deg2rad(20.0), 1.0, 0.005])
    identity_error = normalized_state_error(identity_x, identity_xhat, scales)
    identity_max_error = float(np.max(identity_error))

    initial_estimate = np.array([np.deg2rad(-4.0), -0.20, 0.0])
    truth, estimate, measured_angle = simulate_matched_model(
        ad, c, ld, initial_true, initial_estimate, samples
    )
    convergence_error = normalized_state_error(truth, estimate, scales)
    final_error = float(np.max(convergence_error[-1]))
    settling_time = first_permanent_below(time, convergence_error, 0.01)
    torque_relative_error = float(abs(estimate[-1, 2] - truth[-1, 2]) / abs(truth[-1, 2]))

    step_time_s = 2.0
    step_torque = 0.003
    torque_schedule = np.where(time >= step_time_s, step_torque, 0.0)
    step_truth, step_estimate, _ = simulate_matched_model(
        ad, c, ld, np.zeros(3), np.zeros(3), samples, torque_schedule
    )
    step_scales = np.array([step_torque / parameters.restoring_n_m_per_rad, 1.0, step_torque])
    step_error = normalized_state_error(step_truth, step_estimate, step_scales)
    step_index = int(round(step_time_s * fs))
    step_settling_absolute = first_permanent_below(time, step_error, 0.01, step_index)
    step_settling_after_step = None if step_settling_absolute is None else step_settling_absolute - step_time_s

    limits = config["acceptance"]
    checks = {
        "observable_rank_is_three": observability_rank == 3,
        "matched_identity_error_pass": identity_max_error <= limits["identity_normalized_max_error"],
        "characteristic_coefficient_error_pass": characteristic_error
        <= limits["characteristic_coefficient_max_error"],
        "initial_error_convergence_pass": final_error <= limits["final_normalized_state_error"],
        "steady_torque_error_pass": torque_relative_error <= limits["steady_torque_relative_error"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "stage": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "one-axis linear matched model; ideal angle measurement; constant-disturbance observer",
        "checks": checks,
        "metrics": {
            "observability_rank": observability_rank,
            "identity_normalized_max_error": identity_max_error,
            "characteristic_coefficient_max_error": characteristic_error,
            "repeated_pole_numeric_cluster_spread": pole_cluster_spread,
            "final_normalized_state_error": final_error,
            "steady_torque_relative_error": torque_relative_error,
            "one_percent_settling_time_s": settling_time,
            "step_one_percent_settling_time_after_step_s": step_settling_after_step,
        },
        "parameters": p,
        "natural_characteristics": natural_characteristics(parameters),
        "observer": {
            "continuous_repeated_pole_rad_s": pole_rad_s,
            "target_discrete_repeated_pole": target_z,
            "actual_discrete_poles": [
                {"real": float(v.real), "imag": float(v.imag)} for v in actual_poles
            ],
            "target_characteristic_coefficients": target_characteristic.tolist(),
            "actual_characteristic_coefficients": actual_characteristic.tolist(),
            "continuous_gain": lc[:, 0].tolist(),
            "discrete_gain": ld[:, 0].tolist(),
        },
        "acceptance": limits,
    }
    (output_dir / "stage1_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    labels = ("Angle [deg]", "Angular rate [deg/s]", "Disturbance torque [mN m]")
    scales_for_plot = (180.0 / np.pi, 180.0 / np.pi, 1000.0)
    for index, (label, factor) in enumerate(zip(labels, scales_for_plot)):
        axes[index].plot(time, truth[:, index] * factor, color="#1f77b4", linewidth=2.0, label="Plant truth")
        axes[index].plot(time, estimate[:, index] * factor, color="#e67e22", linestyle="--", linewidth=1.5, label="Observer")
        axes[index].set_ylabel(label)
        axes[index].grid(alpha=0.2)
        axes[index].legend(loc="upper right")
    axes[3].semilogy(time, np.maximum(np.max(convergence_error, axis=1), 1e-16), color="#7b1fa2")
    axes[3].axhline(0.01, color="#c62828", linestyle="--", label="1% threshold")
    axes[3].set_ylabel("Max normalized error")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(alpha=0.2)
    axes[3].legend(loc="upper right")
    fig.suptitle("Stage 1 matched-model observer convergence", fontsize=15)
    fig.savefig(output_dir / "stage1_convergence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for index, (label, factor) in enumerate(zip(labels, scales_for_plot)):
        axes[index].plot(time, step_truth[:, index] * factor, color="#1f77b4", linewidth=2.0, label="Plant truth")
        axes[index].plot(time, step_estimate[:, index] * factor, color="#e67e22", linestyle="--", linewidth=1.5, label="Observer")
        axes[index].axvline(step_time_s, color="#555555", linestyle=":", label="Torque step" if index == 0 else None)
        axes[index].set_ylabel(label)
        axes[index].grid(alpha=0.2)
        axes[index].legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Stage 1 response to an unannounced torque step", fontsize=15)
    fig.savefig(output_dir / "stage1_step_response.png", dpi=180)
    plt.close(fig)
    plot_block_diagram(output_dir / "stage1_block_diagram.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SIMULATION_ROOT / "config" / "stage1_nominal.json")
    parser.add_argument("--output", type=Path, default=SIMULATION_ROOT / "results" / "stage1")
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
