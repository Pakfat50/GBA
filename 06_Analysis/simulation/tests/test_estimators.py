"""Behavioral tests for causal and future-data estimators."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from estimators import augmented_matrices, causal_luenberger, kalman_filter_and_rts_smoother
from model import PendulumParameters


class EstimatorComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.p = PendulumParameters(0.00215194497, 0.00059676098, 0.01178181, 0.245)
        self.dt = 0.01

    def test_ramp_observer_tracks_matching_four_state_model(self) -> None:
        a, c = augmented_matrices(self.p, 1)
        ad = expm(a * self.dt)
        state = np.array([0.0, 0.0, 0.0, 2e-6])
        angles = np.zeros(2001)
        truth = np.zeros(2001)
        for k in range(len(angles)):
            angles[k] = (c @ state).item()
            truth[k] = state[2] / self.p.force_lever_m
            state = ad @ state
        estimate = causal_luenberger(angles, self.p, self.dt, 1.0, 1)
        self.assertLess(abs(estimate[-1] - truth[-1]), 1e-8)

    def test_causal_estimators_do_not_use_future_angle(self) -> None:
        first = np.zeros(300)
        second = first.copy()
        second[200] = 0.01
        for order in (0, 1):
            a = causal_luenberger(first, self.p, self.dt, 1.0, order)
            b = causal_luenberger(second, self.p, self.dt, 1.0, order)
            np.testing.assert_array_equal(a[:201], b[:201])
        kf_a, _ = kalman_filter_and_rts_smoother(first, self.p, self.dt, np.deg2rad(0.02), 1e-5)
        kf_b, _ = kalman_filter_and_rts_smoother(second, self.p, self.dt, np.deg2rad(0.02), 1e-5)
        np.testing.assert_array_equal(kf_a[:200], kf_b[:200])

    def test_rts_smoother_uses_future_angle(self) -> None:
        first = np.zeros(300)
        second = first.copy()
        second[200] = 0.01
        _, smooth_a = kalman_filter_and_rts_smoother(first, self.p, self.dt, np.deg2rad(0.02), 1e-5)
        _, smooth_b = kalman_filter_and_rts_smoother(second, self.p, self.dt, np.deg2rad(0.02), 1e-5)
        self.assertGreater(np.max(np.abs(smooth_a[150:200] - smooth_b[150:200])), 1e-8)

    def test_constant_force_converges_for_all_estimators(self) -> None:
        force = 0.0008
        angle = np.full(3001, self.p.force_lever_m * force / self.p.restoring_n_m_per_rad)
        estimates = [
            causal_luenberger(angle, self.p, self.dt, 1.0, 0),
            causal_luenberger(angle, self.p, self.dt, 1.0, 1),
        ]
        estimates.extend(kalman_filter_and_rts_smoother(angle, self.p, self.dt, np.deg2rad(0.02), 1e-5))
        for estimate in estimates:
            self.assertLess(abs(estimate[2000] / force - 1.0), 1e-4)


if __name__ == "__main__":
    unittest.main()
