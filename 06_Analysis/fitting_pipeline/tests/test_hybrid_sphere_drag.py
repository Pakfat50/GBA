"""Stage 5 球抗力同定の物理式・係数回帰テスト。"""

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from run_hybrid_sphere_drag_identification import ROD_ALPHA_BALL
from run_hybrid_sphere_drag_identification import build_ball_intervals
from run_hybrid_sphere_drag_identification import fit_c
from run_hybrid_sphere_drag_identification import brown_lawler_cd
from run_hybrid_rod_damping_identification import explicit_energy_basis
from run_hybrid_rod_damping_identification import THEORETICAL_ROD_C


class HybridSphereDragTest(unittest.TestCase):
    def test_rod_mask_factor_uses_exposed_length_fourth_power(self):
        self.assertAlmostEqual(ROD_ALPHA_BALL, (0.129 / 0.229) ** 4)
        self.assertLess(ROD_ALPHA_BALL, 0.11)

    def test_energy_fit_recovers_known_sphere_c_with_fixed_tau_and_rod(self):
        c_sphere = 1.4e-5
        row = {
            "segment_id": "IN_BALL_TEST", "axis": "IN", "start_angle_rad": np.deg2rad(55.0),
            "measured_next_angle_rad": np.deg2rad(-45.0), "inertia_kg_m2": 3.66e-4,
            "restoring_n_m_per_rad": 1.57e-2,
        }
        basis = explicit_energy_basis(row["start_angle_rad"], row["inertia_kg_m2"], row["restoring_n_m_per_rad"])
        tau = 8.65e-5
        rod_ball = THEORETICAL_ROD_C * ROD_ALPHA_BALL
        energy_loss = (c_sphere + rod_ball) * basis[1] + tau * basis[2]
        start_energy = row["restoring_n_m_per_rad"] * (1.0 - np.cos(row["start_angle_rad"]))
        end_energy = start_energy - energy_loss
        row["measured_next_angle_rad"] = -np.arccos(1.0 - end_energy / row["restoring_n_m_per_rad"])
        row["waveform_interval_count"] = 1
        stage4 = {"rod_drag_theory": {"theoretical_c_n_m_s2_per_rad2": THEORETICAL_ROD_C}, "models_for_review": {"theoretical_c": {"tau_IN": tau, "tau_OUT": tau}}}
        fitted, unused_objective, unused_groups, unused_stats = fit_c([row], stage4)
        self.assertAlmostEqual(fitted, c_sphere, delta=1e-15)

    def test_brown_lawler_cd_is_finite_in_target_re_range(self):
        cd = brown_lawler_cd(3000.0)
        self.assertTrue(0.35 < cd < 0.5)


if __name__ == "__main__":
    unittest.main()
