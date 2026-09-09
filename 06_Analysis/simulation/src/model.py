"""Common one-axis linear model used by the Stage 1 plant and observer.

The augmented state is [angle_rad, angular_rate_rad_s, torque_Nm].
Stage 1 deliberately gives the plant and observer the same model.  Parameter
mismatch, nonlinear mechanics and sensor effects belong to later stages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class PendulumParameters:
    """Physical parameters of the linearized one-axis pendulum."""

    inertia_kg_m2: float
    damping_n_m_s_per_rad: float
    restoring_n_m_per_rad: float
    force_lever_m: float

    def validate(self) -> None:
        values = asdict(self)
        if not all(np.isfinite(v) and v > 0.0 for v in values.values()):
            raise ValueError(f"All physical parameters must be positive: {values}")


def continuous_matrices(parameters: PendulumParameters) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous A and C for x=[theta, omega, tau]."""

    parameters.validate()
    inertia = parameters.inertia_kg_m2
    a = np.array(
        [
            [0.0, 1.0, 0.0],
            [
                -parameters.restoring_n_m_per_rad / inertia,
                -parameters.damping_n_m_s_per_rad / inertia,
                1.0 / inertia,
            ],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    c = np.array([[1.0, 0.0, 0.0]], dtype=float)
    return a, c


def exact_discretization(a: np.ndarray, sample_period_s: float) -> np.ndarray:
    """Exact zero-input discretization of the autonomous augmented model."""

    if not np.isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be positive")
    return expm(np.asarray(a, dtype=float) * sample_period_s)


def observability_matrix(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return [C; C A; C A^2] for the three-state model."""

    return np.vstack((c, c @ a, c @ a @ a))


def natural_characteristics(parameters: PendulumParameters) -> dict[str, float]:
    """Return undamped/damped natural frequency and decay rate."""

    parameters.validate()
    inertia = parameters.inertia_kg_m2
    decay = parameters.damping_n_m_s_per_rad / (2.0 * inertia)
    omega_n = np.sqrt(parameters.restoring_n_m_per_rad / inertia)
    omega_d = np.sqrt(max(omega_n * omega_n - decay * decay, 0.0))
    return {
        "omega_n_rad_s": float(omega_n),
        "natural_frequency_hz": float(omega_n / (2.0 * np.pi)),
        "decay_rate_per_s": float(decay),
        "damped_frequency_hz": float(omega_d / (2.0 * np.pi)),
        "damping_ratio": float(decay / omega_n),
    }
