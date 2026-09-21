"""Stage 4バックラッシュプラントの基本動作を確認する。"""

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(MODULE_DIRECTORY))

from model import PendulumParameters
from run_backlash_stage4 import backlash_friction_plant
from run_friction_observer_doe import friction_plant


class BacklashStage4Test(unittest.TestCase):
    def setUp(self):
        self.parameters = PendulumParameters(0.00043, 0.0003, 0.0167, 0.17)
        self.dt = 0.01
        self.force = 0.01 * np.sin(2.0 * np.pi * 0.4 * np.arange(0.0, 5.0, self.dt))
        self.friction = 0.00005
        self.epsilon = np.deg2rad(0.5)

    def test_zero_backlash_matches_existing_friction_plant(self):
        expected = friction_plant(
            self.force, self.parameters, self.dt, self.friction, self.epsilon
        )
        actual, play = backlash_friction_plant(
            self.force,
            self.parameters,
            self.dt,
            self.friction,
            self.epsilon,
            0.0,
        )
        self.assertTrue(np.allclose(actual, expected, atol=1e-12, rtol=1e-10))
        self.assertTrue(np.allclose(actual[:, 0], play))

    def test_nonzero_backlash_bounds_angle_difference(self):
        half_width = np.deg2rad(0.5)
        state, play = backlash_friction_plant(
            self.force,
            self.parameters,
            self.dt,
            self.friction,
            self.epsilon,
            half_width,
        )
        difference = np.abs(state[:, 0] - play)
        self.assertLessEqual(float(np.max(difference)), half_width + 1e-12)


if __name__ == "__main__":
    unittest.main()

