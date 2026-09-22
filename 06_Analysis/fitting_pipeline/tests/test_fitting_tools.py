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
from fitting_tools import detect_free_decay_candidates
from fitting_tools import estimate_envelope_center
from fitting_tools import explicit_half_cycle_energy_basis
from fitting_tools import extract_decay_extrema
from fitting_tools import extract_decay_turning_points
from fitting_tools import fit_one_decay
from fitting_tools import fit_quality_statistics
from fitting_tools import identify_base_inertia_and_restoring
from fitting_tools import identify_k_over_i_from_turning_points
from fitting_tools import nonlinear_period_samples
from fitting_tools import preprocess_free_decay
from fitting_tools import read_angle_log
from fitting_tools import refine_turning_point_quadratic
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
        self.candidate_settings = {
            "smooth_time_s": 0.05,
            "baseline_time_s": 3.0,
            "candidate_peak_min_deg": 35.0,
            "candidate_peak_prominence_deg": 0.5,
            "candidate_gap_s": 8.0,
            "peak_distance_s": 0.15,
            "zero_crossing_search_s": 15.0,
            "release_drop_deg": 0.7,
            "release_speed_min_deg_s": 5.0,
            "release_speed_sustain_s": 0.05,
            "minimum_decay_time_s": 3.0,
            "maximum_decay_time_s": 30.0,
            "settle_duration_s": 2.0,
            "settle_center_max_deg": 3.0,
            "settle_std_max_deg": 0.35,
            "quality_peak_prominence_deg": 0.2,
            "quality_growth_min_deg": 3.0,
            "quality_growth_ratio": 0.12,
            "quality_max_angle_deg": 75.0,
            "quality_max_jump_deg": 10.0,
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

    def test_continuous_retry_is_split_into_separate_candidates(self):
        """異常試行の直後に再試験しても、二つの候補として残す。"""

        dt = 0.01
        short_time = np.arange(0.0, 2.0, dt)
        long_time = np.arange(0.0, 10.0, dt)
        rest_time = np.arange(0.0, 4.0, dt)
        parameters = np.array(
            [5.6, 0.16, 0.03, 0.10, 0.0, np.deg2rad(60.0), 0.0]
        )
        first = simulate_normalized_decay(short_time, parameters, np.deg2rad(0.5))
        second = simulate_normalized_decay(long_time, parameters, np.deg2rad(0.5))
        angle = np.concatenate([first, second, np.zeros(len(rest_time))])
        time_s = np.arange(len(angle), dtype=float) * dt

        candidates = detect_free_decay_candidates(
            time_s, angle, self.candidate_settings
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["suggested_use"], 0)
        self.assertEqual(candidates[1]["suggested_use"], 1)

    def test_flat_holding_region_is_removed_before_fitting(self):
        """解放前の端部保持を候補波形の先頭へ含めない。"""

        dt = 0.01
        hold_time = np.arange(0.0, 6.0, dt)
        hold_deg = 60.0 + 0.6 * np.sin(2.0 * np.pi * 0.4 * hold_time)
        decay_time = np.arange(0.0, 10.0, dt)
        parameters = np.array(
            [5.6, 0.16, 0.03, 0.10, 0.0, np.deg2rad(60.0), 0.0]
        )
        decay = simulate_normalized_decay(
            decay_time, parameters, np.deg2rad(0.5)
        )
        rest_time = np.arange(0.0, 4.0, dt)
        angle = np.concatenate(
            [np.deg2rad(hold_deg), decay, np.zeros(len(rest_time))]
        )
        time_s = np.arange(len(angle), dtype=float) * dt

        candidates = detect_free_decay_candidates(
            time_s, angle, self.candidate_settings
        )
        self.assertEqual(len(candidates), 1)
        self.assertGreater(candidates[0]["start_time_s"], 5.9)
        self.assertLess(candidates[0]["start_time_s"], 6.2)

    def test_waveform_fit_reproduces_synthetic_decay(self):
        time_s = np.arange(0.0, 10.0, 0.02)
        truth = np.array([5.6, 0.10, 0.04, 0.11, 0.005, np.deg2rad(45.0), 0.0])
        angle = simulate_normalized_decay(time_s, truth, np.deg2rad(0.5))
        fit = fit_one_decay(time_s, angle, np.deg2rad(0.5), 15.0, 140)
        self.assertEqual(fit["success"], 1)
        self.assertLess(fit["rmse_deg"], 0.15)
        self.assertGreater(fit["r_value"], 0.999)
        self.assertGreater(fit["r_squared"], 0.999)
        self.assertAlmostEqual(fit["k_over_i_per_s2"], truth[0], delta=0.3)

    def test_fit_quality_statistics_reports_r_and_r_squared(self):
        measured = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8])
        rmse, r_value, r_squared = fit_quality_statistics(measured, predicted)
        self.assertAlmostEqual(rmse, np.sqrt(0.025), places=12)
        self.assertGreater(r_value, 0.99)
        self.assertAlmostEqual(r_squared, 0.98, places=12)

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

    def test_frequency_is_identified_from_nonlinear_turning_points(self):
        time_s = np.arange(0.0, 12.0, 0.005)
        truth = np.array([40.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(60.0), 0.0])
        angle = simulate_normalized_decay(time_s, truth, np.deg2rad(0.5))
        turning = extract_decay_turning_points(time_s, angle, 0.0)
        identified = identify_k_over_i_from_turning_points(
            turning["time_s"], turning["amplitude_rad"]
        )
        self.assertAlmostEqual(identified["k_over_i_per_s2"], truth[0], delta=0.15)

    def test_nonlinear_period_sample_reports_finite_amplitude_correction(self):
        time_s = np.arange(0.0, 10.0, 0.002)
        truth = np.array([36.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(60.0), 0.0])
        angle = simulate_normalized_decay(time_s, truth, np.deg2rad(0.5))
        turning = extract_decay_turning_points(time_s, angle, 0.0)
        samples = nonlinear_period_samples(
            turning["time_s"], turning["amplitude_rad"], 5.0
        )
        self.assertGreater(len(samples), 2)
        self.assertGreater(samples[0]["finite_amplitude_correction_ratio"], 1.1)
        self.assertAlmostEqual(samples[0]["k_over_i_per_s2"], truth[0], delta=0.1)

    def test_quadratic_turning_point_recovers_subsample_vertex(self):
        time_s = np.array([0.8, 1.0, 1.2])
        truth_time = 1.07
        values = 3.0 - 2.5 * (time_s - truth_time) ** 2
        peak_time, peak_value, used = refine_turning_point_quadratic(
            time_s, values, 1
        )
        self.assertEqual(used, 1)
        self.assertAlmostEqual(peak_time, truth_time, places=12)
        self.assertAlmostEqual(peak_value, 3.0, places=12)

    def test_envelope_center_recovers_damped_oscillation_center(self):
        time_s = np.arange(0.0, 14.0, 0.013)
        center = np.deg2rad(1.7)
        amplitude = np.deg2rad(58.0) * np.exp(-0.075 * time_s)
        angle = center + amplitude * np.cos(2.0 * np.pi * 0.72 * time_s)
        extrema = extract_decay_extrema(time_s, angle)
        result = estimate_envelope_center(
            extrema["time_s"], extrema["angle_rad"], extrema["kind"]
        )
        self.assertAlmostEqual(
            result["center_rad"], float(np.mean(result["midpoint_rad"])), places=14
        )
        np.testing.assert_allclose(
            result["midpoint_rad"],
            0.5 * (
                result["upper_envelope_rad"] + result["lower_envelope_rad"]
            ),
        )
        self.assertAlmostEqual(
            np.rad2deg(result["center_rad"]), np.rad2deg(center), delta=0.05
        )
        first_peak_time = extrema["time_s"][0] - time_s[0]
        self.assertGreater(first_peak_time, 0.6)
        self.assertLess(first_peak_time, 0.8)
        self.assertEqual(extrema["kind"][0], -1)

        preprocessing = preprocess_free_decay(
            time_s, angle, existing_center_rad=center
        )
        self.assertEqual(preprocessing["initial_speed_rad_s"], 0.0)
        self.assertGreater(preprocessing["ignored_initial_half_cycle_s"], 0.6)
        self.assertLess(preprocessing["ignored_initial_half_cycle_s"], 0.8)
        self.assertLess(preprocessing["initial_angle_rad"], 0.0)

    def test_envelope_center_uses_arithmetic_mean_not_median(self):
        time_s = np.arange(6, dtype=float)
        angle_deg = np.array([3.0, -1.0, 3.0, -1.0, 11.0, -1.0])
        kind = np.array([1, -1, 1, -1, 1, -1])
        result = estimate_envelope_center(
            time_s,
            np.deg2rad(angle_deg),
            kind,
            minimum_amplitude_deg=0.0,
        )
        midpoint_deg = np.rad2deg(result["midpoint_rad"])
        self.assertAlmostEqual(np.rad2deg(result["center_rad"]), 2.5)
        self.assertNotAlmostEqual(
            np.rad2deg(result["center_rad"]), float(np.median(midpoint_deg))
        )

    def test_explicit_energy_basis_has_small_angle_limits(self):
        amplitude = 0.05
        inertia = 0.0004
        restoring = 0.016
        omega = np.sqrt(restoring / inertia)
        basis = explicit_half_cycle_energy_basis(amplitude, inertia, restoring)
        viscous_limit = 0.5 * np.pi * amplitude**2 * omega
        quadratic_limit = 4.0 / 3.0 * amplitude**3 * omega**2
        friction_limit = 2.0 * amplitude
        self.assertAlmostEqual(
            basis["viscous_basis"] / viscous_limit, 1.0, delta=0.002
        )
        self.assertAlmostEqual(
            basis["quadratic_basis"] / quadratic_limit, 1.0, delta=0.002
        )
        self.assertAlmostEqual(basis["friction_basis"], friction_limit, places=12)

    def test_explicit_energy_linear_system_recovers_coefficients(self):
        inertia = 0.0004
        restoring = 0.016
        truth = np.array([3.0e-5, 1.2e-5, 8.0e-5])
        matrix = []
        for amplitude_deg in [8.0, 15.0, 25.0, 40.0, 60.0]:
            basis = explicit_half_cycle_energy_basis(
                np.deg2rad(amplitude_deg), inertia, restoring
            )
            matrix.append(
                [
                    basis["viscous_basis"],
                    basis["quadratic_basis"],
                    basis["friction_basis"],
                ]
            )
        matrix = np.asarray(matrix)
        measured_loss = matrix @ truth
        identified, unused_residual, unused_rank, unused_singular = np.linalg.lstsq(
            matrix, measured_loss, rcond=None
        )
        np.testing.assert_allclose(identified, truth, rtol=1e-9, atol=1e-12)

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
