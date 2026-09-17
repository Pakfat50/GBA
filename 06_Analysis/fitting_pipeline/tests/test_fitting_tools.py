"""fitting_pipelineの主要処理を合成データで確認する。"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from fitting_tools import estimate_force_eso
from fitting_tools import estimate_force_rts
from fitting_tools import fit_one_decay
from fitting_tools import identify_base_inertia_and_restoring
from fitting_tools import read_angle_log
from fitting_tools import simulate_forced_motion
from fitting_tools import simulate_normalized_decay
from fitting_tools import split_free_decay


class FittingToolsTest(unittest.TestCase):
    def setUp(self):
        self.segment_settings = {
            "smooth_time_s": 0.15,
            "start_angle_min_deg": 35.0,
            "hold_duration_s": 1.5,
            "hold_std_max_deg": 0.35,
            "hold_speed_max_deg_s": 3.0,
            "release_drop_deg": 0.7,
            "minimum_decay_time_s": 4.0,
            "maximum_decay_time_s": 30.0,
            "settle_duration_s": 3.0,
            "settle_center_max_deg": 3.0,
            "settle_std_max_deg": 0.25,
        }

    def test_logger_decoder_csv_is_read_and_wrapped(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "LOG00008_ANGLE.csv"
            table = pd.DataFrame(
                {
                    "systime[ms]": np.arange(20) * 10,
                    "angle0[deg]": np.linspace(350.0, 355.0, 20),
                    "angle1[deg]": np.linspace(5.0, 10.0, 20),
                }
            )
            table.to_csv(path, index=False)
            time_s, angle_rad = read_angle_log(path, "angle0[deg]")
            self.assertAlmostEqual(time_s[1] - time_s[0], 0.01)
            self.assertLess(np.rad2deg(angle_rad[0]), 0.0)
            self.assertAlmostEqual(np.rad2deg(angle_rad[0]), -10.0)

    def test_multiple_decays_are_split_from_one_file(self):
        dt = 0.01
        complete_time = []
        complete_angle = []
        current_time = 0.0
        directions = [1.0, -1.0]
        for direction in directions:
            hold_time = np.arange(0.0, 2.5, dt)
            hold_angle = np.full(len(hold_time), direction * np.deg2rad(45.0))
            decay_time = np.arange(0.0, 10.0, dt)
            parameters = np.array([5.6, 0.15, 0.03, 0.10, 0.0, hold_angle[0], 0.0])
            decay_angle = simulate_normalized_decay(decay_time, parameters, np.deg2rad(0.5))
            rest_time = np.arange(0.0, 5.0, dt)
            rest_angle = np.zeros(len(rest_time))
            pieces_time = [hold_time, decay_time + hold_time[-1] + dt]
            pieces_time.append(rest_time + hold_time[-1] + decay_time[-1] + 2.0 * dt)
            pieces_angle = [hold_angle, decay_angle, rest_angle]
            local_time = np.concatenate(pieces_time) + current_time
            local_angle = np.concatenate(pieces_angle)
            complete_time.extend(local_time.tolist())
            complete_angle.extend(local_angle.tolist())
            current_time = local_time[-1] + dt

        segments = split_free_decay(
            np.asarray(complete_time), np.asarray(complete_angle), self.segment_settings
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["direction"], "P45")
        self.assertEqual(segments[1]["direction"], "N45")

    def test_waveform_fit_reproduces_synthetic_decay(self):
        time_s = np.arange(0.0, 10.0, 0.02)
        truth = np.array([5.6, 0.10, 0.04, 0.11, 0.005, np.deg2rad(45.0), 0.0])
        angle = simulate_normalized_decay(time_s, truth, np.deg2rad(0.5))
        fit = fit_one_decay(time_s, angle, np.deg2rad(0.5), 15.0, 140)
        self.assertEqual(fit["success"], 1)
        self.assertLess(fit["rmse_deg"], 0.15)
        self.assertAlmostEqual(fit["k_over_i_per_s2"], truth[0], delta=0.3)

    def test_multiple_levels_identify_absolute_i_and_k(self):
        base_inertia = 0.00050
        base_restoring = 0.0200
        rows = []
        level = 0
        while level < 5:
            delta_inertia = level * 0.000035
            delta_restoring = -level * 0.0016
            q_value = (base_restoring + delta_restoring) / (
                base_inertia + delta_inertia
            )
            rows.append(
                {
                    "configuration": "SP" + str(level).zfill(2),
                    "k_over_i_mean": q_value,
                    "k_over_i_uncertainty": 0.01,
                    "delta_inertia": delta_inertia,
                    "delta_restoring": delta_restoring,
                }
            )
            level += 1
        inertia, restoring, predictions = identify_base_inertia_and_restoring(rows)
        self.assertAlmostEqual(inertia, base_inertia, delta=1e-8)
        self.assertAlmostEqual(restoring, base_restoring, delta=1e-7)
        self.assertEqual(len(predictions), 5)

    def test_force_estimators_return_finite_arrays(self):
        parameters = {
            "inertia_kg_m2": 0.00043,
            "damping_n_m_s_per_rad": 0.00015,
            "restoring_n_m_per_rad": 0.0167,
            "quadratic_n_m_s2_per_rad2": 0.00001,
            "friction_n_m": 0.00005,
            "force_lever_m": 0.17,
            "epsilon_rad_s": np.deg2rad(0.5),
        }
        settings = {
            "eso_pole_hz": 20.0,
            "rts_force_random_walk_n_per_sample": 1.0,
            "angle_noise_rad": np.deg2rad(0.02),
            "epsilon_rad_s": np.deg2rad(0.5),
        }
        dt = 0.01
        time_s = np.arange(0.0, 5.0, dt)
        force = 0.005 * np.sin(2.0 * np.pi * 0.4 * time_s)
        states = simulate_forced_motion(force, dt, parameters)
        eso = estimate_force_eso(states[:, 0], dt, parameters, settings)
        rts = estimate_force_rts(states[:, 0], dt, parameters, settings)
        self.assertEqual(len(eso), len(force))
        self.assertEqual(len(rts), len(force))
        self.assertTrue(np.all(np.isfinite(eso)))
        self.assertTrue(np.all(np.isfinite(rts)))

    def test_force_estimators_report_near_zero_during_exact_free_decay(self):
        """同定モデルと完全に一致する自由減衰では外力がほぼ0 Nになる。"""

        parameters = {
            "inertia_kg_m2": 0.00043,
            "damping_n_m_s_per_rad": 0.00015,
            "restoring_n_m_per_rad": 0.0167,
            "quadratic_n_m_s2_per_rad2": 0.00001,
            "friction_n_m": 0.00005,
            "force_lever_m": 0.17,
            "epsilon_rad_s": np.deg2rad(0.5),
        }
        settings = {
            "eso_pole_hz": 45.0,
            "rts_force_random_walk_n_per_sample": 1.0,
            "angle_noise_rad": np.deg2rad(0.02),
            "epsilon_rad_s": np.deg2rad(0.5),
        }
        dt = 0.01
        time_s = np.arange(0.0, 8.0, dt)
        force = np.zeros(len(time_s))
        initial_state = [np.deg2rad(45.0), 0.0]
        states = simulate_forced_motion(force, dt, parameters, initial_state)
        eso = estimate_force_eso(states[:, 0], dt, parameters, settings)
        rts = estimate_force_rts(states[:, 0], dt, parameters, settings)
        evaluation = time_s >= 1.0
        eso_rmse = np.sqrt(np.mean(eso[evaluation] * eso[evaluation]))
        rts_rmse = np.sqrt(np.mean(rts[evaluation] * rts[evaluation]))
        self.assertLess(eso_rmse, 0.0001)
        self.assertLess(rts_rmse, 0.000001)


if __name__ == "__main__":
    unittest.main()
