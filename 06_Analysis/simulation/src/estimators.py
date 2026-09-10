"""Angle-only estimators used in the Stage 2 implementation comparison.

All public functions accept angle samples as their only observation. Angular
rate is always a hidden state. The RTS result is non-causal because its backward
pass uses measurements later than the estimated time.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from model import PendulumParameters
from observer import ackermann_observer_gain


def augmented_matrices(
    parameters: PendulumParameters, disturbance_order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return A,C for constant-torque (0) or torque-ramp (1) augmentation."""

    parameters.validate()
    if disturbance_order not in (0, 1):
        raise ValueError("disturbance_order must be 0 or 1")
    n = 3 + disturbance_order
    a = np.zeros((n, n), dtype=float)
    a[0, 1] = 1.0
    a[1, 0] = -parameters.restoring_n_m_per_rad / parameters.inertia_kg_m2
    a[1, 1] = -parameters.damping_n_m_s_per_rad / parameters.inertia_kg_m2
    a[1, 2] = 1.0 / parameters.inertia_kg_m2
    if disturbance_order == 1:
        a[2, 3] = 1.0
    c = np.zeros((1, n), dtype=float)
    c[0, 0] = 1.0
    return a, c


def causal_luenberger(
    angle_rad: np.ndarray,
    parameters: PendulumParameters,
    sample_period_s: float,
    repeated_pole_hz: float,
    disturbance_order: int = 0,
) -> np.ndarray:
    """Estimate force with the current or torque-ramp Luenberger observer.

    The output at sample k is based on measurements through k-1, matching the
    observer update convention already validated in Stage 1.
    """

    if repeated_pole_hz <= 0.0:
        raise ValueError("repeated_pole_hz must be positive")
    a, c = augmented_matrices(parameters, disturbance_order)
    ad = expm(a * sample_period_s)
    target = np.exp(-2.0 * np.pi * repeated_pole_hz * sample_period_s)
    ld = ackermann_observer_gain(ad, c, np.full(len(a), target))
    transition = ad - ld @ c
    state = np.zeros(len(a), dtype=float)
    force = np.zeros(len(angle_rad), dtype=float)
    for k, angle in enumerate(np.asarray(angle_rad, dtype=float)):
        force[k] = state[2] / parameters.force_lever_m
        state = transition @ state + ld[:, 0] * angle
    return force


def kalman_filter_and_rts_smoother(
    angle_rad: np.ndarray,
    parameters: PendulumParameters,
    sample_period_s: float,
    assumed_angle_noise_rad: float,
    force_random_walk_std_n_per_sample: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal Kalman-filter and full-interval RTS-smoothed force.

    The state is [angle, angular rate, torque]. The stochastic model treats
    torque as a random walk. The RTS estimate at k uses the complete record,
    including angle samples after k.
    """

    if assumed_angle_noise_rad <= 0.0 or force_random_walk_std_n_per_sample <= 0.0:
        raise ValueError("Noise assumptions must be positive")
    a, c = augmented_matrices(parameters, 0)
    ad = expm(a * sample_period_s)
    n_samples = len(angle_rad)
    q_tau = parameters.force_lever_m * force_random_walk_std_n_per_sample
    q = np.diag([1e-24, 1e-24, q_tau * q_tau])
    r = assumed_angle_noise_rad**2
    initial_force_std = 0.005
    covariance = np.diag(
        [np.deg2rad(5.0) ** 2, 1.0**2, (parameters.force_lever_m * initial_force_std) ** 2]
    )
    identity = np.eye(3)
    filtered = np.zeros((n_samples, 3))
    predicted = np.zeros_like(filtered)
    filtered_covariance = np.zeros((n_samples, 3, 3))
    predicted_covariance = np.zeros_like(filtered_covariance)
    state = np.zeros(3)

    for k, measurement in enumerate(np.asarray(angle_rad, dtype=float)):
        if k:
            state = ad @ filtered[k - 1]
            covariance = ad @ filtered_covariance[k - 1] @ ad.T + q
        predicted[k] = state
        predicted_covariance[k] = covariance
        innovation_variance = (c @ covariance @ c.T).item() + r
        gain = covariance @ c.T / innovation_variance
        state = state + gain[:, 0] * (measurement - (c @ state).item())
        # Joseph form preserves symmetry and positive semidefiniteness.
        correction = identity - gain @ c
        covariance = correction @ covariance @ correction.T + gain * r @ gain.T
        covariance = 0.5 * (covariance + covariance.T)
        filtered[k] = state
        filtered_covariance[k] = covariance

    smoothed = filtered.copy()
    for k in range(n_samples - 2, -1, -1):
        cross = filtered_covariance[k] @ ad.T
        smoother_gain = np.linalg.solve(predicted_covariance[k + 1].T, cross.T).T
        smoothed[k] += smoother_gain @ (smoothed[k + 1] - predicted[k + 1])

    scale = parameters.force_lever_m
    return filtered[:, 2] / scale, smoothed[:, 2] / scale
