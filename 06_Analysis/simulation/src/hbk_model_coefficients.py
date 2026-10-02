"""Load adopted HBK mechanical coefficients for observer simulations.

Fitted parameters are separate from legacy DOE files because the adopted
values differ by axis and hardware configuration. Callers must choose both.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = SIMULATION_ROOT / "config" / "hbk_model_coefficients.json"


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Read the checked-in registry and validate its basic schema and values."""
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported HBK coefficient registry schema")
    records = registry.get("records", [])
    keys = [(item["axis"], item["configuration"]) for item in records]
    if len(keys) != 12 or len(set(keys)) != 12:
        raise ValueError("Registry must contain one record for each of 2 axes x 6 configurations")
    for item in records:
        for name in ("inertia_kg_m2", "restoring_n_m_per_rad", "tau_n_m"):
            if item[name] <= 0:
                raise ValueError(f"{name} must be positive for {item['axis']}/{item['configuration']}")
        if item["viscous_damping_n_m_s_per_rad"] != 0:
            raise ValueError("Adopted model requires b=0")
    return registry


def select_coefficients(axis: str, configuration: str, path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Return one record, requiring the analyst to state axis and setup."""
    axis, configuration = axis.upper(), configuration.upper()
    for item in load_registry(path)["records"]:
        if item["axis"] == axis and item["configuration"] == configuration:
            return item.copy()
    raise ValueError(f"No adopted coefficients for {axis}/{configuration}")


def simulate_hbk_plant(
    force_n, coefficients: dict[str, Any], sample_period_s: float,
    initial_angle_rad: float = 0.0, initial_rate_rad_s: float = 0.0,
    force_lever_m: float | None = None, friction_epsilon_deg_s: float = 0.5,
):
    """Integrate the adopted nonlinear one-axis plant with RK4.

    ``force_n`` is the applied force at each sample. The equation uses the
    adopted zero-viscous-damping model, quadratic rod/ball drag, and smooth
    Coulomb friction. This is the plant-side input path for wind simulations;
    linear observers can still use their existing linearized state model.
    """
    import numpy as np

    force = np.asarray(force_n, dtype=float)
    if force.ndim != 1 or len(force) == 0 or not np.all(np.isfinite(force)):
        raise ValueError("force_n must be a non-empty finite one-dimensional array")
    if sample_period_s <= 0 or friction_epsilon_deg_s <= 0:
        raise ValueError("sample period and friction smoothing speed must be positive")
    if force_lever_m is None or not math.isfinite(force_lever_m) or force_lever_m <= 0:
        raise ValueError("Supply the measured/defined positive force_lever_m explicitly")
    if coefficients["viscous_damping_n_m_s_per_rad"] != 0:
        raise ValueError("Adopted HBK plant requires b=0")

    inertia = coefficients["inertia_kg_m2"]
    stiffness = coefficients["restoring_n_m_per_rad"]
    drag = coefficients["total_quadratic_drag_n_m_s2_per_rad2"]
    tau = coefficients["tau_n_m"]
    lever = force_lever_m
    epsilon = math.radians(friction_epsilon_deg_s)
    states = np.zeros((len(force), 2), dtype=float)
    states[0] = (initial_angle_rad, initial_rate_rad_s)

    def derivative(state, applied_force):
        angle, rate = state
        acceleration = (
            lever * applied_force * math.cos(angle)
            - stiffness * math.sin(angle)
            - drag * abs(rate) * rate
            - tau * math.tanh(rate / epsilon)
        ) / inertia
        return np.array([rate, acceleration])

    for index in range(len(force) - 1):
        state, applied = states[index], force[index]
        k1 = derivative(state, applied)
        k2 = derivative(state + 0.5 * sample_period_s * k1, applied)
        k3 = derivative(state + 0.5 * sample_period_s * k2, applied)
        k4 = derivative(state + sample_period_s * k3, applied)
        states[index + 1] = state + sample_period_s * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return states


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print adopted HBK coefficients for one observer simulation case.")
    parser.add_argument("--axis", required=True, choices=("IN", "OUT"), help="IN or OUT motion axis")
    parser.add_argument("--configuration", required=True,
                        choices=("SP00", "SP01", "SP02", "SP03", "SP04", "BALL"),
                        help="mechanical setup used in free-decay identification")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    print(json.dumps(select_coefficients(args.axis, args.configuration, args.registry), indent=2, ensure_ascii=False))
