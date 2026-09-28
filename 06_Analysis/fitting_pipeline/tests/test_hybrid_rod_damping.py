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
from run_hybrid_rod_damping_identification import THEORETICAL_ROD_C
from run_hybrid_rod_damping_identification import build_intervals
from run_hybrid_rod_damping_identification import calculate_reynolds_assessment
from run_hybrid_rod_damping_identification import energy_fit_with_fixed_c
from run_hybrid_rod_damping_identification import explicit_energy_basis
from run_hybrid_rod_damping_identification import fit_model
from run_hybrid_rod_damping_identification import fit_model_gauss_newton
from run_hybrid_rod_damping_identification import fit_tau_only_then_aggregate
from run_hybrid_rod_damping_identification import parameter_scales
from run_hybrid_rod_damping_identification import solve_interval_trajectory
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
            delta=np.deg2rad(1.0e-2),
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

    def test_plot_trajectory_ends_at_same_turning_point(self):
        interval = self.make_interval()
        physical = {
            "b_IN": 3.0e-5,
            "b_OUT": 4.0e-5,
            "c_rod": 2.0e-6,
            "tau_IN": 7.0e-5,
            "tau_OUT": 8.0e-5,
        }
        trajectory_time, trajectory_angle = solve_interval_trajectory(
            interval, physical
        )
        augmented = solve_interval_with_sensitivities(interval, physical)
        self.assertEqual(len(trajectory_time), 61)
        self.assertAlmostEqual(
            trajectory_time[-1],
            augmented["predicted_half_period_s"],
            delta=1.0e-4,
        )
        self.assertAlmostEqual(
            trajectory_angle[-1],
            augmented["predicted_next_angle_rad"],
            delta=np.deg2rad(1.0e-2),
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

    def test_fixed_c_energy_fit_recovers_constructed_coefficients(self):
        truth = {"b_IN": 2.0e-5, "c_rod": 3.0e-6, "tau_IN": 7.0e-5}
        intervals = []
        for angle_deg in [12.0, 20.0, 30.0, 45.0, 60.0]:
            interval = self.make_interval("IN", angle_deg)
            basis = explicit_energy_basis(
                interval["start_angle_rad"],
                interval["inertia_kg_m2"],
                interval["restoring_n_m_per_rad"],
            )
            initial_energy = interval["restoring_n_m_per_rad"] * (
                1.0 - np.cos(interval["start_angle_rad"])
            )
            loss = (
                truth["b_IN"] * basis[0]
                + truth["c_rod"] * basis[1]
                + truth["tau_IN"] * basis[2]
            )
            final_energy = initial_energy - loss
            next_amplitude = np.arccos(
                1.0 - final_energy / interval["restoring_n_m_per_rad"]
            )
            interval["measured_next_angle_rad"] = -next_amplitude
            interval["waveform_interval_count"] = 5
            intervals.append(interval)
        fitted = energy_fit_with_fixed_c(
            intervals, ["b_IN", "tau_IN"], truth["c_rod"]
        )
        self.assertAlmostEqual(fitted["b_IN"] / truth["b_IN"], 1.0, delta=1.0e-8)
        self.assertAlmostEqual(
            fitted["tau_IN"] / truth["tau_IN"], 1.0, delta=1.0e-8
        )

    def test_theoretical_rod_c_uses_confirmed_dimensions(self):
        expected = 1.225 * 1.2 * 0.005 / 8.0 * (0.229**4 + 0.070**4)
        self.assertAlmostEqual(THEORETICAL_ROD_C, expected, delta=1.0e-18)

    def test_tau_only_fit_and_reynolds_assessment(self):
        truth_tau = 8.0e-5
        intervals = []
        for number, angle_deg in enumerate([15.0, 25.0, 40.0, 55.0]):
            interval = self.make_interval("IN", angle_deg)
            interval["interval_id"] = "IN_SYN_H" + str(number)
            interval["segment_id"] = "IN_SYN"
            interval["waveform_interval_count"] = 4
            physical = {
                "b_IN": 0.0,
                "b_OUT": 0.0,
                "c_rod": THEORETICAL_ROD_C,
                "tau_IN": truth_tau,
                "tau_OUT": 0.0,
            }
            prediction = solve_interval_with_sensitivities(interval, physical)
            interval["measured_next_angle_rad"] = prediction[
                "predicted_next_angle_rad"
            ]
            intervals.append(interval)
        result = fit_tau_only_then_aggregate(intervals, THEORETICAL_ROD_C)
        self.assertAlmostEqual(
            result["fit"]["parameters"]["tau_IN"] / truth_tau,
            1.0,
            # 保存系軌道を使うエネルギー近似からtauを初期同定するため、
            # 減衰ODEで作った合成波形に対して数%の差を許容する。
            delta=0.05,
        )
        reynolds = calculate_reynolds_assessment(intervals)
        self.assertEqual(len(reynolds), 4)
        self.assertGreater(
            reynolds[-1]["upper_tip_reynolds"],
            reynolds[0]["upper_tip_reynolds"],
        )
        self.assertLess(
            reynolds[-1]["drag_weighted_reynolds"],
            reynolds[-1]["upper_tip_reynolds"],
        )


if __name__ == "__main__":
    unittest.main()
