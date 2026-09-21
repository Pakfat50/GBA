"""playバックラッシュモデルの基本動作を確認する。"""

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from backlash_model import information_criteria
from backlash_model import simulate_decay
from backlash_model import update_play_state


class BacklashModelTest(unittest.TestCase):
    def test_play_state_stays_during_small_reversal(self):
        half_width = np.deg2rad(1.0)
        state = update_play_state(np.deg2rad(10.0), np.deg2rad(9.0), half_width)
        reversed_state = update_play_state(np.deg2rad(9.5), state, half_width)
        self.assertAlmostEqual(reversed_state, state)

    def test_play_state_follows_after_gap_is_taken_up(self):
        half_width = np.deg2rad(1.0)
        state = np.deg2rad(9.0)
        state = update_play_state(np.deg2rad(7.0), state, half_width)
        self.assertAlmostEqual(state, np.deg2rad(8.0))

    def test_zero_backlash_simulation_is_finite(self):
        time_s = np.arange(0.0, 5.0, 0.01)
        parameters = [5.62, 0.08, 0.117, 0.0, 0.0, np.deg2rad(45.0), 0.0]
        angle, play = simulate_decay(time_s, parameters, np.deg2rad(0.5))
        self.assertTrue(np.all(np.isfinite(angle)))
        self.assertTrue(np.allclose(angle, play))

    def test_backlash_changes_free_decay(self):
        time_s = np.arange(0.0, 5.0, 0.01)
        no_backlash = [5.62, 0.08, 0.117, 0.0, 0.0, np.deg2rad(45.0), 0.0]
        with_backlash = no_backlash.copy()
        with_backlash[3] = np.deg2rad(1.0)
        angle_0, unused = simulate_decay(time_s, no_backlash, np.deg2rad(0.5))
        angle_1, unused = simulate_decay(time_s, with_backlash, np.deg2rad(0.5))
        difference = np.sqrt(np.mean((angle_1 - angle_0) ** 2))
        self.assertGreater(difference, np.deg2rad(0.05))

    def test_information_criteria_penalize_extra_parameter(self):
        residual = np.ones(100) * 0.01
        aicc_6, bic_6 = information_criteria(residual, 6)
        aicc_7, bic_7 = information_criteria(residual, 7)
        self.assertGreater(aicc_7, aicc_6)
        self.assertGreater(bic_7, bic_6)


if __name__ == "__main__":
    unittest.main()

