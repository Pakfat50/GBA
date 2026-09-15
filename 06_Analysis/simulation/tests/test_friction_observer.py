"""Checks for the friction plant and compensated estimators."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from estimators import causal_luenberger, kalman_rts_force
from friction_estimators import (
    causal_luenberger_with_friction,
    kalman_rts_force_with_friction,
    smoothed_coulomb_torque,
)
from model import PendulumParameters
from run_friction_observer_doe import friction_plant


class FrictionObserverTests(unittest.TestCase):
    def setUp(self):
        self.parameters = PendulumParameters(
            0.00042796350277537,
            0.00031709627284592847,
            0.016726993199666917,
            0.17,
        )
        self.dt = 0.01
        self.epsilon = np.deg2rad(0.5)

    def test_friction_torque_opposes_motion(self):
        magnitude = 5.0e-5
        self.assertLess(
            smoothed_coulomb_torque(1.0, magnitude, self.epsilon), 0.0
        )
        self.assertGreater(
            smoothed_coulomb_torque(-1.0, magnitude, self.epsilon), 0.0
        )
        self.assertEqual(
            smoothed_coulomb_torque(0.0, magnitude, self.epsilon), 0.0
        )

    def test_zero_friction_matches_existing_eso(self):
        rng = np.random.default_rng(4)
        angle = rng.normal(0.0, 0.01, 400)
        for order, pole in ((0, 5.0), (1, 3.0)):
            expected = causal_luenberger(
                angle, self.parameters, self.dt, pole, order
            )
            actual = causal_luenberger_with_friction(
                angle, self.parameters, self.dt, pole, 0.0,
                self.epsilon, order,
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

    def test_zero_friction_matches_existing_rts(self):
        rng = np.random.default_rng(5)
        angle = rng.normal(0.0, 0.01, 300)
        for order, noise in ((0, 0.03), (1, 0.03)):
            expected = kalman_rts_force(
                angle, self.parameters, self.dt, np.deg2rad(0.02), noise, order
            )
            actual = kalman_rts_force_with_friction(
                angle, self.parameters, self.dt, np.deg2rad(0.02), noise,
                0.0, self.epsilon, order,
            )
            np.testing.assert_allclose(actual[0], expected[0], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(actual[1], expected[1], rtol=1e-9, atol=1e-11)

    def test_friction_reduces_free_decay_energy(self):
        force = np.zeros(501)
        # A short impulse creates the same initial motion in both simulations.
        force[0] = 0.05
        without = friction_plant(
            force, self.parameters, self.dt, 0.0, self.epsilon
        )
        with_friction = friction_plant(
            force, self.parameters, self.dt, 5.0e-5, self.epsilon
        )
        tail_without = np.max(np.abs(without[-100:, 0]))
        tail_with = np.max(np.abs(with_friction[-100:, 0]))
        self.assertLess(tail_with, tail_without)


if __name__ == "__main__":
    unittest.main()
