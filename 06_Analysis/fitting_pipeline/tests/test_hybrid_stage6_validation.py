"""Stage 6 configuration mapping and fixed-coefficient checks."""

import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from run_hybrid_stage6_validation import configuration_coefficients
from run_hybrid_stage6_validation import simulate_continuous


class HybridStage6ValidationTest(unittest.TestCase):
    def setUp(self):
        self.parameters = pd.DataFrame([
            {"axis": "IN", "configuration": "SP00", "inertia_kg_m2": 2.4e-4, "restoring_n_m_per_rad": 2.2e-2},
            {"axis": "IN", "configuration": "BALL", "inertia_kg_m2": 3.7e-4, "restoring_n_m_per_rad": 1.6e-2},
            {"axis": "OUT", "configuration": "SP00", "inertia_kg_m2": 2.5e-4, "restoring_n_m_per_rad": 2.1e-2},
            {"axis": "OUT", "configuration": "BALL", "inertia_kg_m2": 3.8e-4, "restoring_n_m_per_rad": 1.5e-2},
        ])
        self.stage4 = {"models_for_review": {"theoretical_c": {"c_rod": 2.5e-6, "tau_IN": 8.7e-5, "tau_OUT": 1.7e-4}}}
        self.stage5 = {"sphere_c_n_m_s2_per_rad2": 1.46e-5, "rod_c_n_m_s2_per_rad2_ball": 2.6e-7}

    def test_configuration_switches_physical_inertia_stiffness_and_drag(self):
        sp00 = configuration_coefficients("IN", "SP00", self.parameters, self.stage4, self.stage5)
        ball = configuration_coefficients("IN", "BALL", self.parameters, self.stage4, self.stage5)
        self.assertEqual(sp00["inertia_kg_m2"], 2.4e-4)
        self.assertEqual(ball["inertia_kg_m2"], 3.7e-4)
        self.assertEqual(sp00["restoring_n_m_per_rad"], 2.2e-2)
        self.assertEqual(ball["restoring_n_m_per_rad"], 1.6e-2)
        self.assertEqual(sp00["c_n_m_s2_per_rad2"], 2.5e-6)
        self.assertEqual(ball["c_n_m_s2_per_rad2"], 1.46e-5 + 2.6e-7)
        self.assertEqual(ball["b_n_m_s_per_rad"], 0.0)

    def test_tau_is_axis_specific_and_not_changed_by_configuration(self):
        in_ball = configuration_coefficients("IN", "BALL", self.parameters, self.stage4, self.stage5)
        out_ball = configuration_coefficients("OUT", "BALL", self.parameters, self.stage4, self.stage5)
        self.assertEqual(in_ball["tau_n_m"], 8.7e-5)
        self.assertEqual(out_ball["tau_n_m"], 1.7e-4)

    def test_continuous_turning_detection_counts_both_peak_directions(self):
        half_period = 2.0 * 3.141592653589793 * (2.4e-4 / 2.2e-2) ** 0.5 / 2.0
        record = {
            "segment_id": "SYNTHETIC",
            "time_s": pd.Series([i / 100.0 for i in range(501)]).to_numpy(),
            "centered_angle_deg": pd.Series([0.0 for _ in range(501)]).to_numpy(),
            "peak_times_s": pd.Series([i * half_period for i in range(16)]).to_numpy(),
            "peak_angles_deg": pd.Series([20.0 if i % 2 == 0 else -20.0 for i in range(16)]).to_numpy(),
        }
        coefficients = {
            "inertia_kg_m2": 2.4e-4,
            "restoring_n_m_per_rad": 2.2e-2,
            "b_n_m_s_per_rad": 0.0,
            "c_n_m_s2_per_rad2": 0.0,
            "tau_n_m": 0.0,
        }
        result = simulate_continuous(record, coefficients)
        self.assertGreaterEqual(result["predicted_peak_count"], 14)
        self.assertEqual(result["matched_half_period_count"], 15)


if __name__ == "__main__":
    unittest.main()
