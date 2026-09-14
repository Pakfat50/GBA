"""Kaimal-spectrum wind synthesis for the Stage 2 force-estimation PoC."""
from __future__ import annotations

import numpy as np


def kaimal_longitudinal_psd(
    frequency_hz: np.ndarray,
    mean_speed_m_s: float,
    sigma_speed_m_s: float,
    integral_scale_m: float,
) -> np.ndarray:
    """One-sided IEC-style Kaimal spectrum for along-wind velocity.

    S_u(f) = 4 sigma_u^2 L_u / U / (1 + 6 f L_u / U)^(5/3)
    """

    if min(mean_speed_m_s, sigma_speed_m_s, integral_scale_m) <= 0.0:
        raise ValueError("Kaimal parameters must be positive")
    frequency = np.asarray(frequency_hz, dtype=float)
    if np.any(frequency < 0.0):
        raise ValueError("Frequency must be non-negative")
    reduced = frequency * integral_scale_m / mean_speed_m_s
    return (
        4.0
        * sigma_speed_m_s**2
        * integral_scale_m
        / mean_speed_m_s
        / (1.0 + 6.0 * reduced) ** (5.0 / 3.0)
    )


def synthesize_kaimal_wind(
    sample_rate_hz: float,
    duration_s: float,
    mean_speed_m_s: float,
    turbulence_intensity: float,
    integral_scale_m: float,
    seed: int,
    maximum_speed_m_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Generate a stationary along-wind record by random-phase synthesis.

    If ``maximum_speed_m_s`` is supplied, one scalar is applied to all
    fluctuations so the record remains between zero and the maximum.  This
    preserves spectral shape while reducing the realised turbulence intensity.
    """

    if min(sample_rate_hz, duration_s, mean_speed_m_s, turbulence_intensity) <= 0.0:
        raise ValueError("Wind synthesis parameters must be positive")
    n = int(round(sample_rate_hz * duration_s))
    if n < 4:
        raise ValueError("Wind record is too short")
    dt = 1.0 / sample_rate_hz
    time = np.arange(n, dtype=float) * dt
    frequency = np.fft.rfftfreq(n, dt)
    df = frequency[1] - frequency[0]
    target_sigma = turbulence_intensity * mean_speed_m_s
    spectrum = kaimal_longitudinal_psd(frequency, mean_speed_m_s, target_sigma, integral_scale_m)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, len(frequency))
    coefficient = n * np.sqrt(spectrum * df / 2.0) * np.exp(1j * phase)
    coefficient[0] = 0.0
    if n % 2 == 0:
        coefficient[-1] = 0.0
    fluctuation = np.fft.irfft(coefficient, n=n)
    fluctuation *= target_sigma / np.std(fluctuation)

    scale = 1.0
    negative_extent = -float(np.min(fluctuation))
    if negative_extent > 0.0:
        scale = min(scale, mean_speed_m_s / negative_extent)
    if maximum_speed_m_s is not None:
        if maximum_speed_m_s <= mean_speed_m_s:
            raise ValueError("Maximum wind speed must exceed the mean")
        positive_extent = float(np.max(fluctuation))
        if positive_extent > 0.0:
            scale = min(scale, (maximum_speed_m_s - mean_speed_m_s) / positive_extent)
    fluctuation *= scale
    speed = mean_speed_m_s + fluctuation
    metadata = {
        "target_turbulence_intensity": float(turbulence_intensity),
        "realized_turbulence_intensity": float(np.std(speed) / np.mean(speed)),
        "fluctuation_scale_for_bounds": float(scale),
        "minimum_speed_m_s": float(np.min(speed)),
        "maximum_speed_m_s": float(np.max(speed)),
    }
    return time, speed, metadata


def drag_force_from_speed(
    speed_m_s: np.ndarray,
    air_density_kg_m3: float,
    drag_coefficient: float,
    projected_area_m2: float,
) -> np.ndarray:
    """Return signed quasi-steady drag from signed along-wind speed."""

    if min(air_density_kg_m3, drag_coefficient, projected_area_m2) <= 0.0:
        raise ValueError("Aerodynamic parameters must be positive")
    speed = np.asarray(speed_m_s, dtype=float)
    return 0.5 * air_density_kg_m3 * drag_coefficient * projected_area_m2 * speed * np.abs(speed)
