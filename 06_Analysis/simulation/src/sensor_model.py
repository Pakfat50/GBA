"""Switchable angle-sensor model for Stage 3 observer evaluation.

The physical plant stays on a uniform time grid.  This module samples that
continuous-looking record with delay and clock jitter, adds white and coloured
noise, then applies the MT6701 quantisation.  Each effect can be disabled by
setting its coefficient to zero, so one-factor sensitivity checks use exactly
the same implementation as the nominal case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AngleSensorParameters:
    """Parameters of the complete angle-measurement path."""

    sample_rate_hz: float
    resolution_bits: int = 14
    full_scale_deg: float = 360.0
    white_noise_std_deg: float = 0.0
    coloured_noise_std_deg: float = 0.0
    coloured_noise_time_constant_s: float = 0.0
    fixed_delay_s: float = 0.0
    sampling_jitter_std_s: float = 0.0
    gain_error_fraction: float = 0.0
    offset_deg: float = 0.0

    def validate(self) -> None:
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if self.resolution_bits <= 0:
            raise ValueError("resolution_bits must be positive")
        if self.full_scale_deg <= 0.0:
            raise ValueError("full_scale_deg must be positive")
        if min(
            self.white_noise_std_deg,
            self.coloured_noise_std_deg,
            self.coloured_noise_time_constant_s,
            self.fixed_delay_s,
            self.sampling_jitter_std_s,
        ) < 0.0:
            raise ValueError("Noise, time constants, delay and jitter must be non-negative")
        if self.coloured_noise_std_deg > 0.0 and self.coloured_noise_time_constant_s <= 0.0:
            raise ValueError("coloured noise requires a positive time constant")

    @property
    def quantisation_step_deg(self) -> float:
        """One least-significant bit of the absolute-angle output."""
        return self.full_scale_deg / float(2**self.resolution_bits)


def _stationary_coloured_noise(
    count: int,
    sample_period_s: float,
    standard_deviation_deg: float,
    time_constant_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a stationary first-order Gauss-Markov sequence."""
    if standard_deviation_deg == 0.0:
        return np.zeros(count)
    pole = np.exp(-sample_period_s / time_constant_s)
    driving_std = standard_deviation_deg * np.sqrt(1.0 - pole * pole)
    values = np.zeros(count)
    values[0] = rng.normal(0.0, standard_deviation_deg)
    for index in range(1, count):
        values[index] = pole * values[index - 1] + rng.normal(0.0, driving_std)
    return values


def apply_angle_sensor_model(
    time_s: np.ndarray,
    true_angle_rad: np.ndarray,
    parameters: AngleSensorParameters,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return measured angle in radians and individual error components.

    Delayed or jittered sampling uses linear interpolation.  Values requested
    before the first physical sample are held at the initial angle, matching a
    causal acquisition path at start-up.  Quantisation is performed last.
    """
    parameters.validate()
    time_s = np.asarray(time_s, dtype=float)
    true_angle_rad = np.asarray(true_angle_rad, dtype=float)
    if time_s.ndim != 1 or true_angle_rad.shape != time_s.shape:
        raise ValueError("time_s and true_angle_rad must be one-dimensional and equal length")
    if len(time_s) < 2 or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time_s must be strictly increasing")

    rng = np.random.default_rng(seed)
    jitter_s = rng.normal(0.0, parameters.sampling_jitter_std_s, len(time_s))
    sample_time_s = time_s - parameters.fixed_delay_s + jitter_s
    delayed_rad = np.interp(
        sample_time_s,
        time_s,
        true_angle_rad,
        left=true_angle_rad[0],
        right=true_angle_rad[-1],
    )
    delayed_deg = np.rad2deg(delayed_rad)
    calibrated_deg = (
        (1.0 + parameters.gain_error_fraction) * delayed_deg
        + parameters.offset_deg
    )
    white_deg = rng.normal(0.0, parameters.white_noise_std_deg, len(time_s))
    coloured_deg = _stationary_coloured_noise(
        len(time_s),
        1.0 / parameters.sample_rate_hz,
        parameters.coloured_noise_std_deg,
        parameters.coloured_noise_time_constant_s,
        rng,
    )
    before_quantisation_deg = calibrated_deg + white_deg + coloured_deg
    step_deg = parameters.quantisation_step_deg
    measured_deg = np.round(before_quantisation_deg / step_deg) * step_deg

    components = {
        "sample_time_s": sample_time_s,
        "jitter_s": jitter_s,
        "delayed_angle_rad": delayed_rad,
        "white_noise_rad": np.deg2rad(white_deg),
        "coloured_noise_rad": np.deg2rad(coloured_deg),
        "quantisation_error_rad": np.deg2rad(measured_deg - before_quantisation_deg),
    }
    return np.deg2rad(measured_deg), components


def effective_uncorrelated_noise_std_deg(parameters: AngleSensorParameters) -> float:
    """RSS of white noise and ideal uniform quantisation noise."""
    parameters.validate()
    quantisation_std = parameters.quantisation_step_deg / np.sqrt(12.0)
    return float(np.hypot(parameters.white_noise_std_deg, quantisation_std))
