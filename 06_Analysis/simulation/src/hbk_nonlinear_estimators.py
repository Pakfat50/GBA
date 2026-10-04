"""Angle-only estimators using the adopted nonlinear HBK plant model.

This module mirrors the observer families used by the earlier wind study:
nonlinear Luenberger/ESO-style observers and extended Kalman RTS smoothers.
The estimated disturbance state is aerodynamic force in newtons.  The model
uses the selected I, K, c, and tau coefficients as known quantities.

For a causal observer, the force returned at sample k is the state estimate
before the angle at k is applied.  This preserves the repository's existing
one-sample observer convention. RTS uses the complete record and is offline.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm

from observer import ackermann_observer_gain


def _model_derivative(state: np.ndarray, coefficients: dict, force_lever_m: float,
                      epsilon_rad_s: float) -> np.ndarray:
    """State derivative for [angle, rate, force] or [angle, rate, force, force_rate]."""
    angle, rate, force = state[:3]
    force_rate = state[3] if len(state) == 4 else 0.0
    inertia = coefficients["inertia_kg_m2"]
    loss_torque = (
        coefficients["viscous_damping_n_m_s_per_rad"] * rate
        + coefficients["total_quadratic_drag_n_m_s2_per_rad2"] * abs(rate) * rate
        + coefficients["tau_n_m"] * math.tanh(rate / epsilon_rad_s)
    )
    acceleration = (
        force_lever_m * force * math.cos(angle)
        - coefficients["restoring_n_m_per_rad"] * math.sin(angle)
        - loss_torque
    ) / inertia
    if len(state) == 3:
        return np.array([rate, acceleration, 0.0])
    return np.array([rate, acceleration, force_rate, 0.0])


def _rk4_step(state: np.ndarray, dt: float, coefficients: dict,
              force_lever_m: float, epsilon_rad_s: float) -> np.ndarray:
    """Advance one state interval with fourth-order Runge–Kutta integration."""
    f = lambda x: _model_derivative(x, coefficients, force_lever_m, epsilon_rad_s)
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def _continuous_jacobian(state: np.ndarray, coefficients: dict,
                         force_lever_m: float, epsilon_rad_s: float) -> np.ndarray:
    """Linearize the nonlinear state model at one state for observer design."""
    angle, rate, force = state[:3]
    inertia = coefficients["inertia_kg_m2"]
    jacobian = np.zeros((len(state), len(state)), dtype=float)
    jacobian[0, 1] = 1.0
    jacobian[1, 0] = (
        -force_lever_m * force * math.sin(angle)
        - coefficients["restoring_n_m_per_rad"] * math.cos(angle)
    ) / inertia
    jacobian[1, 1] = (
        -coefficients["viscous_damping_n_m_s_per_rad"]
        -2.0 * coefficients["total_quadratic_drag_n_m_s2_per_rad2"] * abs(rate)
        - coefficients["tau_n_m"] / epsilon_rad_s
          * (1.0 - math.tanh(rate / epsilon_rad_s) ** 2)
    ) / inertia
    jacobian[1, 2] = force_lever_m * math.cos(angle) / inertia
    if len(state) == 4:
        jacobian[2, 3] = 1.0
    return jacobian


def calculate_luenberger_gain(coefficients: dict, force_lever_m: float, dt: float,
                              pole_hz: float, order: int,
                              epsilon_rad_s: float) -> np.ndarray:
    """Calculate the ESO correction gains for the selected pole bandwidth.

    The matrix is linearized at rest, discretized at ``dt``, then assigned
    ``3 + order`` identical discrete error poles.  The returned vector is
    used in ``x_next = model_prediction + gain * angle_residual``.
    """
    n = 3 + order
    origin = np.zeros(n)
    continuous = _continuous_jacobian(origin, coefficients, force_lever_m, epsilon_rad_s)
    discrete = expm(continuous * dt)
    measurement = np.zeros((1, n))
    measurement[0, 0] = 1.0
    target = np.exp(-2.0 * np.pi * pole_hz * dt)
    return ackermann_observer_gain(discrete, measurement, np.full(n, target))


def nonlinear_luenberger_force(angle_rad: np.ndarray, coefficients: dict,
                               sample_period_s: float, force_lever_m: float,
                               repeated_pole_hz: float, disturbance_order: int = 0,
                               friction_epsilon_deg_s: float = 0.5,
                               initial_state: np.ndarray | None = None) -> np.ndarray:
    """Estimate wind force with an angle-only nonlinear Luenberger observer.

    ``disturbance_order=0`` treats force as constant between updates; order 1
    adds an unknown force-rate state.  Observer gains are designed from the
    adopted model linearized around rest, while each prediction uses the full
    nonlinear equation, including the axis/configuration's I, K, c and tau.
    """
    angle = np.asarray(angle_rad, dtype=float)
    if angle.ndim != 1 or len(angle) < 2 or not np.all(np.isfinite(angle)):
        raise ValueError("angle_rad must be a finite one-dimensional signal")
    if sample_period_s <= 0 or force_lever_m <= 0 or repeated_pole_hz <= 0:
        raise ValueError("sample period, force lever, and observer pole must be positive")
    if disturbance_order not in (0, 1):
        raise ValueError("disturbance_order must be 0 or 1")
    epsilon = math.radians(friction_epsilon_deg_s)
    gain = calculate_luenberger_gain(
        coefficients, force_lever_m, sample_period_s,
        repeated_pole_hz, disturbance_order, epsilon,
    )
    state = np.zeros(3 + disturbance_order)
    if initial_state is not None:
        initial = np.asarray(initial_state, dtype=float)
        if initial.shape != state.shape or not np.all(np.isfinite(initial)):
            raise ValueError(f"initial_state must be a finite vector with shape {state.shape}")
        state = initial.copy()
    force = np.zeros(len(angle))
    for index, measurement in enumerate(angle):
        # Preserve existing convention: output before consuming this sample.
        force[index] = state[2]
        if index + 1 == len(angle):
            break
        predicted = _rk4_step(state, sample_period_s, coefficients, force_lever_m, epsilon)
        innovation = measurement - state[0]
        state = predicted + gain[:, 0] * innovation
    return force


def nonlinear_ekf_rts_force(angle_rad: np.ndarray, coefficients: dict,
                            sample_period_s: float, force_lever_m: float,
                            assumed_angle_noise_rad: float,
                            disturbance_noise_std: float, disturbance_order: int = 0,
                            friction_epsilon_deg_s: float = 0.5,
                            initial_state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return causal EKF and offline extended RTS force estimates.

    Process noise is applied to force (order 0) or force rate (order 1).  The
    RTS output uses future angle samples and is not an online estimator.
    """
    angle = np.asarray(angle_rad, dtype=float)
    if angle.ndim != 1 or len(angle) < 2 or not np.all(np.isfinite(angle)):
        raise ValueError("angle_rad must be a finite one-dimensional signal")
    if min(sample_period_s, force_lever_m, assumed_angle_noise_rad,
           disturbance_noise_std) <= 0:
        raise ValueError("All periods, scales, and noise assumptions must be positive")
    if disturbance_order not in (0, 1):
        raise ValueError("disturbance_order must be 0 or 1")
    epsilon = math.radians(friction_epsilon_deg_s)
    n = 3 + disturbance_order
    measurement = np.zeros((1, n))
    measurement[0, 0] = 1.0
    r = assumed_angle_noise_rad ** 2
    q = np.eye(n) * 1e-24
    if disturbance_order == 0:
        q[2, 2] = disturbance_noise_std ** 2
    else:
        q[3, 3] = disturbance_noise_std ** 2
    initial_covariance = np.diag([math.radians(5.0)**2, 1.0, 0.5**2]
                                 + ([0.5**2] if disturbance_order else []))
    identity = np.eye(n)
    filtered = np.zeros((len(angle), n))
    predicted = np.zeros_like(filtered)
    filtered_cov = np.zeros((len(angle), n, n))
    predicted_cov = np.zeros_like(filtered_cov)
    state = np.zeros(n)
    if initial_state is not None:
        initial = np.asarray(initial_state, dtype=float)
        if initial.shape != state.shape or not np.all(np.isfinite(initial)):
            raise ValueError(f"initial_state must be a finite vector with shape {state.shape}")
        state = initial.copy()
    covariance = initial_covariance

    for index, value in enumerate(angle):
        if index:
            jacobian = _continuous_jacobian(filtered[index - 1], coefficients,
                                            force_lever_m, epsilon)
            transition = expm(jacobian * sample_period_s)
            state = _rk4_step(filtered[index - 1], sample_period_s,
                              coefficients, force_lever_m, epsilon)
            covariance = transition @ filtered_cov[index - 1] @ transition.T + q
        predicted[index] = state
        predicted_cov[index] = covariance
        innovation = value - (measurement @ state).item()
        variance = (measurement @ covariance @ measurement.T).item() + r
        kalman_gain = covariance @ measurement.T / variance
        state = state + kalman_gain[:, 0] * innovation
        correction = identity - kalman_gain @ measurement
        covariance = correction @ covariance @ correction.T + kalman_gain * r @ kalman_gain.T
        covariance = 0.5 * (covariance + covariance.T)
        filtered[index] = state
        filtered_cov[index] = covariance

    smoothed = filtered.copy()
    for index in range(len(angle) - 2, -1, -1):
        jacobian = _continuous_jacobian(filtered[index], coefficients,
                                        force_lever_m, epsilon)
        transition = expm(jacobian * sample_period_s)
        cross = filtered_cov[index] @ transition.T
        smoother_gain = np.linalg.solve(predicted_cov[index + 1].T, cross.T).T
        smoothed[index] += smoother_gain @ (smoothed[index + 1] - predicted[index + 1])
    return filtered[:, 2].copy(), smoothed[:, 2].copy()
