"""Stage 3の頂点間非線形ソルバーを合成条件で検証する。"""

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from peak_to_peak_solver import conservative_half_period_s
from peak_to_peak_solver import solve_next_turning_point


class PeakToPeakSolverTest(unittest.TestCase):
    def setUp(self):
        self.inertia = 3.7e-4
        self.restoring = 1.5e-2
        self.epsilon = np.deg2rad(0.5)

    def solve(self, angle_deg, damping=0.0, quadratic=0.0, friction=0.0, **kwargs):
        return solve_next_turning_point(
            np.deg2rad(angle_deg),
            self.inertia,
            self.restoring,
            damping,
            quadratic,
            friction,
            self.epsilon,
            **kwargs,
        )

    def test_conservative_peak_and_elliptic_half_period(self):
        for angle_deg in [5.0, 30.0, 60.0]:
            with self.subTest(angle_deg=angle_deg):
                result = self.solve(angle_deg)
                exact_time = conservative_half_period_s(
                    np.deg2rad(angle_deg), self.inertia, self.restoring
                )
                self.assertAlmostEqual(
                    np.rad2deg(result["next_angle_rad"]), -angle_deg, delta=1.0e-8
                )
                self.assertAlmostEqual(result["half_period_s"], exact_time, delta=1.0e-9)

    def test_event_direction_alternates_for_both_initial_signs(self):
        positive = self.solve(45.0, 3.8e-5, 1.2e-5, 8.0e-5)
        negative = self.solve(-45.0, 3.8e-5, 1.2e-5, 8.0e-5)
        self.assertLess(positive["next_angle_rad"], 0.0)
        self.assertGreater(negative["next_angle_rad"], 0.0)
        self.assertLess(abs(positive["next_speed_rad_s"]), 1.0e-10)
        self.assertLess(abs(negative["next_speed_rad_s"]), 1.0e-10)

    def test_dissipated_work_closes_energy_balance(self):
        result = self.solve(60.0, 3.8e-5, 1.2e-5, 8.0e-5)
        relative_closure = abs(result["energy_closure_error_j"]) / result[
            "initial_energy_j"
        ]
        self.assertGreater(result["mechanical_loss_j"], 0.0)
        self.assertLess(relative_closure, 1.0e-9)

    def test_default_solution_matches_strict_reference(self):
        default = self.solve(60.0, 3.8e-5, 1.2e-5, 8.0e-5)
        reference = self.solve(
            60.0,
            3.8e-5,
            1.2e-5,
            8.0e-5,
            rtol=1.0e-12,
            angle_speed_atol=1.0e-14,
            energy_atol=1.0e-16,
            max_step_fraction=1.0 / 640.0,
        )
        self.assertLess(
            abs(np.rad2deg(default["next_angle_rad"] - reference["next_angle_rad"])),
            1.0e-7,
        )
        self.assertLess(abs(default["half_period_s"] - reference["half_period_s"]), 1.0e-8)

    def test_invalid_physical_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            solve_next_turning_point(
                np.deg2rad(30.0),
                0.0,
                self.restoring,
                0.0,
                0.0,
                0.0,
                self.epsilon,
            )
        with self.assertRaises(ValueError):
            self.solve(30.0, friction=-1.0e-5)


if __name__ == "__main__":
    unittest.main()
