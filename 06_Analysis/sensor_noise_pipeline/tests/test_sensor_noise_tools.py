"""センサーノイズ解析の主要処理を合成データで確認する。"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from sensor_noise_tools import calculate_segment_metrics
from sensor_noise_tools import quantization_grid_match
from sensor_noise_tools import read_angle_log
from sensor_noise_tools import split_stationary_segments


class SensorNoiseToolsTest(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "stationary_window_s": 0.5,
            "stationary_std_max_deg": 0.08,
            "target_tolerance_deg": 0.8,
            "minimum_segment_s": 5.0,
            "edge_trim_s": 0.5,
        }

    def test_logger_decoder_format_is_read_and_wrapped(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "LOG00008_ANGLE.csv"
            table = pd.DataFrame(
                {
                    "systime[ms]": np.arange(30) * 10,
                    "angle0[deg]": np.full(30, 359.0),
                    "angle1[deg]": np.zeros(30),
                }
            )
            table.to_csv(path, index=False)
            time_s, angle_deg = read_angle_log(path, "angle0[deg]")
            self.assertAlmostEqual(time_s[1] - time_s[0], 0.01)
            self.assertAlmostEqual(angle_deg[0], -1.0)

    def test_three_stationary_repetitions_are_split(self):
        dt = 0.01
        time_s = np.arange(0.0, 25.0, dt)
        angle = np.zeros(len(time_s), dtype=float)
        first_move = (time_s >= 7.0) & (time_s < 9.0)
        second_move = (time_s >= 16.0) & (time_s < 18.0)
        angle[first_move] = 3.0 * np.sin(np.pi * (time_s[first_move] - 7.0) / 2.0)
        angle[second_move] = -3.0 * np.sin(np.pi * (time_s[second_move] - 16.0) / 2.0)
        segments = split_stationary_segments(time_s, angle, 0.0, self.settings)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[2]["repetition"], 3)

    def test_mt6701_quantization_grid_is_detected(self):
        quantum = 360.0 / 16384.0
        values = np.asarray([0, 1, -1, 2, -2, 1], dtype=float) * quantum + 0.04
        ratio = quantization_grid_match(values, quantum)
        self.assertGreater(ratio, 0.99)

    def test_time_quality_and_total_noise_are_measured(self):
        random_generator = np.random.default_rng(123)
        quantum = 360.0 / 16384.0
        time_s = np.arange(0.0, 20.0, 0.01)
        angle = random_generator.normal(0.0, 0.02, len(time_s))
        angle = np.round(angle / quantum) * quantum
        metrics, details = calculate_segment_metrics(time_s, angle, quantum, 2.0)
        self.assertAlmostEqual(metrics["sample_rate_hz"], 100.0, places=6)
        self.assertEqual(metrics["missing_interval_count"], 0)
        self.assertGreater(metrics["noise_std_deg"], 0.015)
        self.assertEqual(len(details["residual_deg"]), len(time_s))

    def test_colored_noise_has_nonzero_time_constant(self):
        random_generator = np.random.default_rng(456)
        time_s = np.arange(0.0, 40.0, 0.01)
        coefficient = np.exp(-0.01 / 0.5)
        colored = np.zeros(len(time_s), dtype=float)
        index = 1
        while index < len(time_s):
            colored[index] = coefficient * colored[index - 1]
            colored[index] += random_generator.normal(0.0, 0.003)
            index += 1
        white = random_generator.normal(0.0, 0.01, len(time_s))
        quantum = 360.0 / 16384.0
        angle = np.round((colored + white) / quantum) * quantum
        metrics, unused_details = calculate_segment_metrics(time_s, angle, quantum, 2.0)
        self.assertGreater(metrics["colored_noise_std_deg"], 0.0)
        self.assertGreater(metrics["colored_time_constant_s"], 0.05)


if __name__ == "__main__":
    unittest.main()

