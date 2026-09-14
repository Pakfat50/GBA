"""Tests for the literature-based wind input generator."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind import drag_force_from_speed, kaimal_longitudinal_psd, synthesize_kaimal_wind


class KaimalWindTest(unittest.TestCase):
    def test_psd_integrates_to_requested_variance(self) -> None:
        frequency = np.geomspace(1e-7, 1e4, 500000)
        spectrum = kaimal_longitudinal_psd(frequency, 5.0, 1.0, 11.34)
        self.assertAlmostEqual(np.trapezoid(spectrum, frequency), 1.0, places=3)

    def test_synthesis_is_reproducible_and_bounded(self) -> None:
        args = (100.0, 60.0, 5.0, 0.2, 11.34, 1234, 8.0)
        t1, u1, metadata = synthesize_kaimal_wind(*args)
        t2, u2, _ = synthesize_kaimal_wind(*args)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(u1, u2)
        self.assertLessEqual(u1.max(), 8.0 + 1e-12)
        self.assertGreaterEqual(u1.min(), -1e-12)
        self.assertAlmostEqual(u1.mean(), 5.0, places=12)
        self.assertGreater(metadata["realized_turbulence_intensity"], 0.1)

    def test_drag_is_signed_and_quadratic(self) -> None:
        force = drag_force_from_speed(np.array([-2.0, 0.0, 2.0]), 1.2, 0.5, 0.1)
        np.testing.assert_allclose(force, [-0.12, 0.0, 0.12])


if __name__ == "__main__":
    unittest.main()
