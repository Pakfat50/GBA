"""Tests for the switchable Stage 3 angle-sensor model."""
import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sensor_model import (  # noqa: E402
    AngleSensorParameters,
    apply_angle_sensor_model,
    effective_uncorrelated_noise_std_deg,
)


class SensorModelTests(unittest.TestCase):
    def test_mt6701_quantisation_grid(self):
        parameters = AngleSensorParameters(sample_rate_hz=100.0)
        time = np.arange(5) / 100.0
        angle = np.deg2rad(np.array([0.0, 0.01, 0.02, 0.03, 0.04]))
        measured, _ = apply_angle_sensor_model(time, angle, parameters, seed=1)
        counts = np.rad2deg(measured) / parameters.quantisation_step_deg
        np.testing.assert_allclose(counts, np.round(counts), atol=1e-12)
        self.assertAlmostEqual(parameters.quantisation_step_deg, 360.0 / 16384.0)

    def test_fixed_delay_uses_previous_angle(self):
        parameters = AngleSensorParameters(
            sample_rate_hz=100.0,
            resolution_bits=53,
            fixed_delay_s=0.02,
        )
        time = np.arange(10) / 100.0
        angle = time.copy()
        measured, components = apply_angle_sensor_model(time, angle, parameters, seed=2)
        np.testing.assert_allclose(measured[2:], angle[:-2], atol=1e-12)
        np.testing.assert_allclose(components["sample_time_s"], time - 0.02)

    def test_seed_reproduces_noise_and_jitter(self):
        parameters = AngleSensorParameters(
            sample_rate_hz=100.0,
            white_noise_std_deg=0.02,
            coloured_noise_std_deg=0.01,
            coloured_noise_time_constant_s=0.5,
            sampling_jitter_std_s=0.001,
        )
        time = np.arange(1000) / 100.0
        angle = np.sin(time)
        first, first_parts = apply_angle_sensor_model(time, angle, parameters, seed=3)
        second, second_parts = apply_angle_sensor_model(time, angle, parameters, seed=3)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_parts["jitter_s"], second_parts["jitter_s"])

    def test_effective_noise_includes_uniform_quantisation(self):
        parameters = AngleSensorParameters(
            sample_rate_hz=100.0,
            white_noise_std_deg=0.015,
        )
        expected = np.hypot(0.015, (360.0 / 16384.0) / np.sqrt(12.0))
        self.assertAlmostEqual(effective_uncorrelated_noise_std_deg(parameters), expected)

    def test_coloured_noise_requires_positive_time_constant(self):
        parameters = AngleSensorParameters(
            sample_rate_hz=100.0,
            coloured_noise_std_deg=0.01,
        )
        with self.assertRaises(ValueError):
            parameters.validate()


if __name__ == "__main__":
    unittest.main()
