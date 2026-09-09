"""Observer-gain derivation and deterministic Stage 1 simulation."""
from __future__ import annotations

import numpy as np


def continuous_repeated_pole_gain(
    inertia_kg_m2: float,
    damping_n_m_s_per_rad: float,
    restoring_n_m_per_rad: float,
    pole_rad_s: float,
) -> np.ndarray:
    """Return L that places all continuous observer-error poles at -pole_rad_s.

    This formula includes the restoring term in A.  It differs from the earlier
    measured-gravity-compensation form, whose A[1,0] is zero.
    """

    if min(inertia_kg_m2, damping_n_m_s_per_rad, restoring_n_m_per_rad, pole_rad_s) <= 0:
        raise ValueError("Observer and physical parameters must be positive")
    a = restoring_n_m_per_rad / inertia_kg_m2
    d = damping_n_m_s_per_rad / inertia_kg_m2
    l1 = 3.0 * pole_rad_s - d
    l2 = 3.0 * pole_rad_s**2 - a - d * l1
    l3 = inertia_kg_m2 * pole_rad_s**3
    return np.array([[l1], [l2], [l3]], dtype=float)


def ackermann_observer_gain(
    ad: np.ndarray, c: np.ndarray, desired_poles: np.ndarray
) -> np.ndarray:
    """Place discrete observer poles, including an exactly repeated SISO pole.

    scipy.signal.place_poles intentionally rejects a pole repeated more often
    than rank(C.T).  Ackermann's formula is suitable here because this fixed
    three-state pair is observable and Stage 1 needs a true repeated pole.
    """

    ad = np.asarray(ad, dtype=float)
    c = np.asarray(c, dtype=float)
    poles = np.asarray(desired_poles, dtype=complex)
    n = ad.shape[0]
    if ad.shape != (n, n) or c.shape != (1, n) or poles.shape != (n,):
        raise ValueError("Expected square Ad, one-row C and one pole per state")

    dual_a = ad.T
    dual_b = c.T
    controllability = np.column_stack(
        [np.linalg.matrix_power(dual_a, i) @ dual_b for i in range(n)]
    )
    if np.linalg.matrix_rank(controllability) != n:
        raise ValueError("The discrete model is not observable")

    coefficients = np.poly(poles)
    phi = np.linalg.matrix_power(dual_a, n).astype(complex)
    for power, coefficient in enumerate(coefficients[1:]):
        phi += coefficient * np.linalg.matrix_power(dual_a, n - 1 - power)
    selector = np.zeros((1, n))
    selector[0, -1] = 1.0
    gain = (selector @ np.linalg.inv(controllability) @ phi).T
    return np.real_if_close(gain, tol=1000).astype(float)


def simulate_matched_model(
    ad: np.ndarray,
    c: np.ndarray,
    observer_gain: np.ndarray,
    initial_state: np.ndarray,
    initial_estimate: np.ndarray,
    samples: int,
    torque_schedule: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate matched discrete plant and innovation observer.

    A torque schedule, when present, overwrites the plant's augmented torque
    state before the measurement at each sample.  This allows a step test while
    making explicit that the constant-disturbance model is violated at the step.
    """

    if samples < 2:
        raise ValueError("At least two samples are required")
    x = np.zeros((samples, 3), dtype=float)
    xhat = np.zeros_like(x)
    y = np.zeros(samples, dtype=float)
    x[0] = np.asarray(initial_state, dtype=float)
    xhat[0] = np.asarray(initial_estimate, dtype=float)
    if torque_schedule is not None and len(torque_schedule) != samples:
        raise ValueError("torque_schedule length must equal samples")

    for k in range(samples):
        if torque_schedule is not None:
            x[k, 2] = torque_schedule[k]
        y[k] = (c @ x[k]).item()
        if k == samples - 1:
            break
        innovation = y[k] - (c @ xhat[k]).item()
        xhat[k + 1] = ad @ xhat[k] + observer_gain[:, 0] * innovation
        x[k + 1] = ad @ x[k]
    return x, xhat, y


def normalized_state_error(
    truth: np.ndarray, estimate: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """Absolute error normalized by explicit engineering scales."""

    scales = np.asarray(scales, dtype=float)
    if scales.shape != (3,) or np.any(scales <= 0.0):
        raise ValueError("Three positive normalization scales are required")
    return np.abs(np.asarray(estimate) - np.asarray(truth)) / scales
