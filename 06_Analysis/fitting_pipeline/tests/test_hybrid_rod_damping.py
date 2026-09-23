"""Stage 4の共有ロッド減衰同定を合成条件で検証する。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from peak_to_peak_solver import solve_next_turning_point
from run_hybrid_rod_damping_identification import FREE_PARAMETER_NAMES
from run_hybrid_rod_damping_identification import build_intervals
from run_hybrid_rod_damping_identification import fit_model
from run_hybrid_rod_damping_identification import fit_model_gauss_newton
from run_hybrid_rod_damping_identification import parameter_scales
from run_hybrid_rod_damping_identification import solve_interval_with_sensitivities


class HybridRodDampingTest(unittest.TestCase):
    def make_interval(self, axis="IN", angle_deg=40.0):
        return {
            "interval_id": axis + "_TEST",
            "segment_id": axis + "_TEST",
            "axis": axis,
            "configuration": "SP00",
            "direction": "P",
            "start_angle_rad": np.deg2rad(angle_deg),
            "measured_next_angle_rad": np.deg2rad(-30.0),
            "start_angle_deg": angle_deg,
            "measured_next_angle_deg": -30.0,
            "start_amplitude_deg": abs(angle_deg),
            "end_amplitude_deg": 30.0,
            "inertia_kg_m2": 3.0e-4,
            "restoring_n_m_per_rad": 1.8e-2,
            "waveform_interval_count": 1,
        }

    def test_augmented_solver_matches_stage3_solver(self):
        interval = self.make_interval()
        physical = {
            "b_IN": 3.0e-5,
            "b_OUT": 4.0e-5,
            "c_rod": 2.0e-6,
            "tau_IN": 7.0e-5,
            "tau_OUT": 8.0e-5,
        }
        augmented = solve_interval_with_sensitivities(interval, physical)
        direct = solve_next_turning_point(
            interval["start_angle_rad"],
            interval["inertia_kg_m2"],
            interval["restoring_n_m_per_rad"],
            physical["b_IN"],
            physical["c_rod"],
            physical["tau_IN"],
            np.deg2rad(0.5),
        )
        self.assertAlmostEqual(
            augmented["predicted_next_angle_rad"],
            direct["next_angle_rad"],
            delta=1.0e-10,
        )

    def test_sensitivities_match_central_difference(self):
        interval = self.make_interval()
        physical = {
            "b_IN": 3.0e-5,
            "b_OUT": 4.0e-5,
            "c_rod": 2.0e-6,
            "tau_IN": 7.0e-5,
            "tau_OUT": 8.0e-5,
        }
        result = solve_interval_with_sensitivities(interval, physical)
        for name, sensitivity_name, step in [
            ("b_IN", "sensitivity_b", 1.0e-8),
            ("c_rod", "sensitivity_c", 1.0e-9),
            ("tau_IN", "sensitivity_tau", 1.0e-8),
        ]:
            plus = dict(physical)
            minus = dict(physical)
            plus[name] += step
            minus[name] -= step
            plus_angle = solve_interval_with_sensitivities(interval, plus)[
                "predicted_next_angle_rad"
            ]
            minus_angle = solve_interval_with_sensitivities(interval, minus)[
                "predicted_next_angle_rad"
            ]
            finite_difference = (plus_angle - minus_angle) / (2.0 * step)
            self.assertAlmostEqual(
                result[sensitivity_name] / finite_difference, 1.0, delta=2.0e-4
            )

    def test_interval_selection_excludes_ball_and_below_four_degrees(self):
        turning = pd.DataFrame(
            [
                ["IN_SP00_P_R01", "IN", "SP00", "P", 0, 0.0, 10.0, 10.0, 1, 1],
                ["IN_SP00_P_R01", "IN", "SP00", "P", 1, 0.5, -8.0, 8.0, 0, 1],
                ["IN_SP00_P_R01", "IN", "SP00", "P", 2, 1.0, -3.5, 3.5, 0, 0],
                ["IN_SP00_P_R01", "IN", "SP00", "P", 3, 1.5, 5.0, 5.0, 0, 1],
                ["IN_BALL_P_R01", "IN", "BALL", "P", 0, 0.0, 10.0, 10.0, 1, 1],
                ["IN_BALL_P_R01", "IN", "BALL", "P", 1, 0.5, -8.0, 8.0, 0, 1],
            ],
            columns=[
                "segment_id",
                "axis",
                "configuration",
                "direction",
                "peak_number",
                "peak_time_s",
                "centered_peak_angle_deg",
                "amplitude_deg",
                "is_initial_peak",
                "eligible_for_later_stages",
            ],
        )
        parameters = pd.DataFrame(
            [
                ["IN", "SP00", 3.0e-4, 1.8e-2],
                ["IN", "BALL", 4.0e-4, 1.5e-2],
            ],
            columns=[
                "axis",
                "configuration",
                "inertia_kg_m2",
                "restoring_n_m_per_rad",
            ],
        )
        intervals = build_intervals(turning, parameters)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["configuration"], "SP00")

    def test_shared_fit_recovers_synthetic_coefficients(self):
        truth = {
            "b_IN": 3.0e-5,
            "b_OUT": 4.0e-5,
            "c_rod": 2.0e-6,
            "tau_IN": 7.0e-5,
            "tau_OUT": 8.0e-5,
        }
        intervals = []
        for axis in ["IN", "OUT"]:
            for waveform in range(2):
                segment_id = axis + "_SYN_" + str(waveform)
                for angle_deg in [15.0, 25.0, 40.0, 55.0]:
                    interval = self.make_interval(axis, angle_deg)
                    interval["interval_id"] = segment_id + "_" + str(angle_deg)
                    interval["segment_id"] = segment_id
                    interval["waveform_interval_count"] = 4
                    prediction = solve_interval_with_sensitivities(interval, truth)
                    interval["measured_next_angle_rad"] = prediction[
                        "predicted_next_angle_rad"
                    ]
                    intervals.append(interval)
        initial = {name: truth[name] * 1.25 for name in FREE_PARAMETER_NAMES}
        scales = parameter_scales(initial)
        fit = fit_model(
            intervals,
            FREE_PARAMETER_NAMES,
            scales,
            robust_scale_deg=0.2,
            initial_physical=initial,
            max_iterations=60,
        )
        self.assertTrue(fit["success"])
        for name in FREE_PARAMETER_NAMES:
            self.assertAlmostEqual(fit["parameters"][name] / truth[name], 1.0, delta=0.02)
        gauss_newton = fit_model_gauss_newton(
            intervals,
            FREE_PARAMETER_NAMES,
            scales,
            robust_scale_deg=0.2,
            initial_physical=initial,
            max_iterations=8,
        )
        self.assertTrue(gauss_newton["success"])
        for name in FREE_PARAMETER_NAMES:
            self.assertAlmostEqual(
                gauss_newton["parameters"][name] / truth[name], 1.0, delta=0.02
            )


if __name__ == "__main__":
    unittest.main()
