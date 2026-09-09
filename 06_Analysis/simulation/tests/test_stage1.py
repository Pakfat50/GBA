"""Automated tests for the Stage 1 matched-model gate."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import PendulumParameters, continuous_matrices, exact_discretization, observability_matrix
from observer import ackermann_observer_gain, continuous_repeated_pole_gain, simulate_matched_model


class Stage1ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = PendulumParameters(0.00215194497, 0.00059676098, 0.01178181, 0.245)
        self.a, self.c = continuous_matrices(self.parameters)
        self.dt = 0.01
        self.ad = exact_discretization(self.a, self.dt)
        self.p = 2.0 * np.pi
        self.z = np.exp(-self.p * self.dt)
        self.ld = ackermann_observer_gain(self.ad, self.c, np.full(3, self.z))

    def test_augmented_model_is_observable(self) -> None:
        self.assertEqual(np.linalg.matrix_rank(observability_matrix(self.a, self.c)), 3)

    def test_discrete_observer_has_requested_repeated_pole(self) -> None:
        actual = np.poly(self.ad - self.ld @ self.c)
        target = np.poly(np.full(3, self.z))
        self.assertLess(np.max(np.abs(actual - target)), 1e-10)

    def test_identical_state_remains_identical(self) -> None:
        x0 = np.array([0.1, 0.2, 0.002])
        truth, estimate, _ = simulate_matched_model(self.ad, self.c, self.ld, x0, x0.copy(), 801)
        self.assertLess(np.max(np.abs(truth - estimate)), 1e-12)

    def test_initial_error_converges(self) -> None:
        truth, estimate, _ = simulate_matched_model(
            self.ad,
            self.c,
            self.ld,
            np.array([0.1, 0.2, 0.002]),
            np.array([-0.1, -0.2, 0.0]),
            801,
        )
        scales = np.array([0.35, 1.0, 0.005])
        self.assertLess(np.max(np.abs(truth[-1] - estimate[-1]) / scales), 0.01)

    def test_unannounced_torque_step_reconverges(self) -> None:
        samples = 801
        torque = np.zeros(samples)
        torque[200:] = 0.003
        truth, estimate, _ = simulate_matched_model(
            self.ad, self.c, self.ld, np.zeros(3), np.zeros(3), samples, torque
        )
        scales = np.array([torque[-1] / self.parameters.restoring_n_m_per_rad, 1.0, torque[-1]])
        self.assertLess(np.max(np.abs(truth[-1] - estimate[-1]) / scales), 0.01)

    def test_continuous_gain_places_requested_poles(self) -> None:
        lc = continuous_repeated_pole_gain(
            self.parameters.inertia_kg_m2,
            self.parameters.damping_n_m_s_per_rad,
            self.parameters.restoring_n_m_per_rad,
            self.p,
        )
        actual = np.poly(self.a - lc @ self.c)
        target = np.poly(np.full(3, -self.p))
        self.assertLess(np.max(np.abs(actual - target)), 1e-10)

    def test_exact_discretization_is_step_size_consistent(self) -> None:
        ad_half = exact_discretization(self.a, self.dt / 2.0)
        self.assertLess(np.max(np.abs(self.ad - ad_half @ ad_half)), 1e-12)


if __name__ == "__main__":
    unittest.main()
