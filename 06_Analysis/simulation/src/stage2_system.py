"""Separate physical plant, ideal angle observation and estimators for Stage 2.

Force is an external input to the two-state plant. The observer only receives
angle samples, never true force, velocity or state. Every output at k uses
measurements strictly before k, matching the Stage 1 innovation convention.
LPF and static output use angle[k]; there is no fitted shift or amplitude gain.
"""
import numpy as np
from scipy import signal
from model import continuous_matrices, exact_discretization, PendulumParameters
from observer import ackermann_observer_gain


def plant_matrices(p, dt):
    a = np.array([[0., 1.], [-p.restoring_n_m_per_rad/p.inertia_kg_m2,
                            -p.damping_n_m_s_per_rad/p.inertia_kg_m2]])
    b = np.array([[0.], [p.force_lever_m/p.inertia_kg_m2]])
    ad, bd, _, _, _ = signal.cont2discrete((a, b, np.eye(2), np.zeros((2, 1))), dt)
    return ad, bd


def plant(force, p, dt):
    """Exact ZOH plant with zero initial state and force held per sample."""
    ad, bd = plant_matrices(p, dt)
    _, _, states = signal.dlsim((ad, bd, np.eye(2), np.zeros((2, 1)), dt), force)
    return states


def nonlinear_plant(force, p, dt):
    """Nonlinear pendulum with horizontal force, integrated by RK4.

    I*theta_ddot + b*theta_dot + K*sin(theta) = l*F*cos(theta).
    The force is held constant over each sample.  This model is used only as
    a physical-range check; the Stage 2 DOE remains the matched linear model.
    """
    force = np.asarray(force, dtype=float)
    states = np.zeros((len(force), 2), dtype=float)

    def derivative(state, applied_force):
        theta, omega = state
        acceleration = (
            p.force_lever_m * applied_force * np.cos(theta)
            - p.damping_n_m_s_per_rad * omega
            - p.restoring_n_m_per_rad * np.sin(theta)
        ) / p.inertia_kg_m2
        return np.array([omega, acceleration])

    for index in range(len(force) - 1):
        current = states[index]
        applied_force = force[index]
        k1 = derivative(current, applied_force)
        k2 = derivative(current + 0.5 * dt * k1, applied_force)
        k3 = derivative(current + 0.5 * dt * k2, applied_force)
        k4 = derivative(current + dt * k3, applied_force)
        states[index + 1] = current + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return states


def observer_system(p, pole_hz, dt):
    a, c = continuous_matrices(p)
    ad = exact_discretization(a, dt)
    pole = np.exp(-2*np.pi*pole_hz*dt)
    ld = ackermann_observer_gain(ad, c, np.full(3, pole))
    m = ad - ld @ c
    h = np.array([[0., 0., 1/p.force_lever_m]])
    return m, ld, h, np.zeros((1, 1))


def estimate(angle, nominal, ratios, dt):
    p = PendulumParameters(nominal.inertia_kg_m2*ratios[0],
                           nominal.damping_n_m_s_per_rad*ratios[1],
                           nominal.restoring_n_m_per_rad*ratios[2],
                           nominal.force_lever_m*ratios[3])
    sys = observer_system(p, ratios[4], dt)
    num, den = signal.ss2tf(*sys)
    observer = signal.lfilter(num[0], den, angle)
    static = p.restoring_n_m_per_rad / p.force_lever_m * angle
    # Three identical causal first-order poles. ZOH; DC gain exactly one.
    z = np.exp(-2*np.pi*ratios[4]*dt)
    low = static.copy()
    for _ in range(3):
        low = signal.lfilter([0., 1-z], [1., -z], low)
    return {'Static': static, 'Low-pass': low, 'Observer': observer}


def force_frequency_response(f_hz, nominal, ratios, dt):
    """Discrete exact transfers, including observer sample timing."""
    ad, bd = plant_matrices(nominal, dt)
    p = PendulumParameters(nominal.inertia_kg_m2*ratios[0],
                           nominal.damping_n_m_s_per_rad*ratios[1],
                           nominal.restoring_n_m_per_rad*ratios[2],
                           nominal.force_lever_m*ratios[3])
    m, ld, h, _ = observer_system(p, ratios[4], dt)
    z = np.exp(2j*np.pi*np.asarray(f_hz)*dt)
    plant_tf = np.array([(np.array([[1., 0.]]) @ np.linalg.solve(zz*np.eye(2)-ad, bd)).item() for zz in z])
    obs_tf = np.array([(h @ np.linalg.solve(zz*np.eye(3)-m, ld)).item() for zz in z])
    alpha = np.exp(-2*np.pi*ratios[4]*dt)
    static_tf = plant_tf*p.restoring_n_m_per_rad/p.force_lever_m
    return {'Static': static_tf, 'Low-pass': static_tf*((1-alpha)/(z-alpha))**3,
            'Observer': plant_tf*obs_tf}
