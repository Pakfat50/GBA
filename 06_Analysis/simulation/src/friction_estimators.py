"""Angle-only estimators with known Coulomb-friction compensation.

The estimated disturbance state is aerodynamic torque.  Mechanical friction is
handled as a known nonlinear input computed from the estimated angular rate.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from estimators import augmented_matrices
from model import PendulumParameters
from observer import ackermann_observer_gain


def smoothed_coulomb_torque(
    angular_rate_rad_s: float, friction_torque_n_m: float, epsilon_rad_s: float
) -> float:
    """Return the signed torque applied to the mechanism by friction."""
    if friction_torque_n_m < 0.0 or epsilon_rad_s <= 0.0:
        raise ValueError("Friction magnitude must be non-negative and epsilon positive")
    return -friction_torque_n_m * np.tanh(angular_rate_rad_s / epsilon_rad_s)


def _zoh_matrices(
    parameters: PendulumParameters, disturbance_order: int, sample_period_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize the augmented linear part and a mechanical-torque input."""
    a, c = augmented_matrices(parameters, disturbance_order)
    n_state = len(a)
    torque_input = np.zeros((n_state, 1))
    torque_input[1, 0] = 1.0 / parameters.inertia_kg_m2
    block = np.zeros((n_state + 1, n_state + 1))
    block[:n_state, :n_state] = a
    block[:n_state, n_state:] = torque_input
    discrete = expm(block * sample_period_s)
    return discrete[:n_state, :n_state], discrete[:n_state, n_state:], c


def causal_luenberger_with_friction(
    angle_rad: np.ndarray,
    parameters: PendulumParameters,
    sample_period_s: float,
    repeated_pole_hz: float,
    friction_torque_n_m: float,
    epsilon_rad_s: float,
    disturbance_order: int = 0,
) -> np.ndarray:
    """Estimate aerodynamic force with nonlinear friction feed-forward."""
    if repeated_pole_hz <= 0.0:
        raise ValueError("repeated_pole_hz must be positive")
    ad, bd, c = _zoh_matrices(parameters, disturbance_order, sample_period_s)
    target = np.exp(-2.0 * np.pi * repeated_pole_hz * sample_period_s)
    ld = ackermann_observer_gain(ad, c, np.full(len(ad), target))
    transition = ad - ld @ c
    state = np.zeros(len(ad))
    force = np.zeros(len(angle_rad))
    for index, angle in enumerate(np.asarray(angle_rad, dtype=float)):
        force[index] = state[2] / parameters.force_lever_m
        friction = smoothed_coulomb_torque(
            state[1], friction_torque_n_m, epsilon_rad_s
        )
        state = transition @ state + ld[:, 0] * angle + bd[:, 0] * friction
    return force


def kalman_rts_force_with_friction(
    angle_rad: np.ndarray,
    parameters: PendulumParameters,
    sample_period_s: float,
    assumed_angle_noise_rad: float,
    disturbance_noise_std: float,
    friction_torque_n_m: float,
    epsilon_rad_s: float,
    disturbance_order: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return EKF and extended-RTS aerodynamic-force estimates.

    The friction mean and its angular-rate Jacobian are included in prediction.
    """
    if assumed_angle_noise_rad <= 0.0 or disturbance_noise_std <= 0.0:
        raise ValueError("Noise assumptions must be positive")
    ad, bd, c = _zoh_matrices(parameters, disturbance_order, sample_period_s)
    n_state = len(ad)
    n_samples = len(angle_rad)
    disturbance_scale = parameters.force_lever_m * disturbance_noise_std
    process_covariance = np.eye(n_state) * 1e-24
    process_covariance[-1, -1] = disturbance_scale**2
    measurement_variance = assumed_angle_noise_rad**2

    covariance = np.eye(n_state)
    covariance[0, 0] = np.deg2rad(5.0) ** 2
    covariance[1, 1] = 1.0
    covariance[2, 2] = (parameters.force_lever_m * 0.5) ** 2
    if disturbance_order == 1:
        covariance[3, 3] = (parameters.force_lever_m * 0.5) ** 2

    identity = np.eye(n_state)
    filtered = np.zeros((n_samples, n_state))
    predicted = np.zeros_like(filtered)
    filtered_covariance = np.zeros((n_samples, n_state, n_state))
    predicted_covariance = np.zeros_like(filtered_covariance)
    transition_jacobian = np.repeat(ad[None, :, :], n_samples, axis=0)
    state = np.zeros(n_state)

    for index, measurement in enumerate(np.asarray(angle_rad, dtype=float)):
        if index:
            previous = filtered[index - 1]
            normalized_rate = previous[1] / epsilon_rad_s
            tanh_rate = np.tanh(normalized_rate)
            friction = -friction_torque_n_m * tanh_rate
            derivative = (
                -friction_torque_n_m
                / epsilon_rad_s
                * (1.0 - tanh_rate * tanh_rate)
            )
            jacobian = ad.copy()
            jacobian[:, 1] += bd[:, 0] * derivative
            transition_jacobian[index] = jacobian
            state = ad @ previous + bd[:, 0] * friction
            covariance = (
                jacobian @ filtered_covariance[index - 1] @ jacobian.T
                + process_covariance
            )
        predicted[index] = state
        predicted_covariance[index] = covariance
        innovation_variance = (c @ covariance @ c.T).item() + measurement_variance
        gain = covariance @ c.T / innovation_variance
        state = state + gain[:, 0] * (measurement - (c @ state).item())
        correction = identity - gain @ c
        covariance = (
            correction @ covariance @ correction.T
            + gain * measurement_variance @ gain.T
        )
        covariance = 0.5 * (covariance + covariance.T)
        filtered[index] = state
        filtered_covariance[index] = covariance

    smoothed = filtered.copy()
    for index in range(n_samples - 2, -1, -1):
        jacobian = transition_jacobian[index + 1]
        cross = filtered_covariance[index] @ jacobian.T
        smoother_gain = np.linalg.solve(
            predicted_covariance[index + 1].T, cross.T
        ).T
        smoothed[index] += smoother_gain @ (
            smoothed[index + 1] - predicted[index + 1]
        )

    scale = parameters.force_lever_m
    return filtered[:, 2] / scale, smoothed[:, 2] / scale
