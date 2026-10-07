#!/usr/bin/env python3
"""Matched nonlinear WBS2.3 RTS wind-speed gain sweep.

This is an amplitude-dependent empirical frequency response about 2 m/s,
using the WBS2.3 70 mm sphere and the selected, assumed grease damping b.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
OUT = SIM / "results/observer_wind/ow07_hardware_bandwidth_design/wbs23_observer"
sys.path.insert(0, str(HERE))

import run_ow07_wbs23_damper_observer as wbs23
from run_ow05_robust_bandwidth_damper_sweep import _fast_nonlinear_rts
from wind import drag_force_from_speed

FREQUENCIES_HZ = np.unique(np.append(np.geomspace(0.05, 15.0, 41), [1.0, 10.0]))
MEAN_WIND_M_S = 2.0
WIND_TONE_AMPLITUDE_M_S = 0.2
CYCLES_TOTAL = 10
DROP_CYCLES_EACH_END = 2


def tone_fit(time_s: np.ndarray, values: np.ndarray, frequency_hz: float,
             mask: np.ndarray) -> tuple[float, complex]:
    """Fit a + b cos(wt) + c sin(wt); return tone amplitude and phasor."""
    phase = 2.0 * np.pi * frequency_hz * time_s[mask]
    matrix = np.column_stack((np.ones(np.count_nonzero(mask)),
                              np.cos(phase), np.sin(phase)))
    beta, *_ = np.linalg.lstsq(matrix, values[mask], rcond=None)
    phasor = complex(float(beta[1]), -float(beta[2]))
    return abs(phasor), phasor


def cutoff_3db(group: pd.DataFrame) -> tuple[float | None, float, float]:
    """First -3 dB crossing relative to the low-frequency gain plateau."""
    group = group.sort_values("frequency_hz").reset_index(drop=True)
    plateau_n = min(3, len(group))
    plateau_db = float(group.loc[:plateau_n - 1, "gain_db"].mean())
    target_db = plateau_db - 3.0
    cutoff = None
    for i in range(1, len(group)):
        y0 = float(group.loc[i - 1, "gain_db"])
        y1 = float(group.loc[i, "gain_db"])
        if y0 > target_db >= y1:
            x0 = math.log10(float(group.loc[i - 1, "frequency_hz"]))
            x1 = math.log10(float(group.loc[i, "frequency_hz"]))
            cutoff = 10 ** (x0 + (target_db - y0) * (x1 - x0) / (y1 - y0))
            break
    return cutoff, plateau_db, target_db


def run() -> tuple[pd.DataFrame, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    cutoff_summary: dict[str, dict] = {}

    for axis in ("IN", "OUT"):
        coeff = wbs23.candidate_coefficients(axis)
        b_value = float(coeff["viscous_damping_n_m_s_per_rad"])
        q_std = float(wbs23.Q_BY_AXIS[axis])
        rts_angle_sigma = float(wbs23.ANGLE_NOISE_ASSUMED_RAD)
        drag_factor = wbs23.DRAG_FACTOR
        mean_force = float(drag_force_from_speed(
            np.array([MEAN_WIND_M_S]), wbs23.AIR_DENSITY,
            wbs23.BALL_CD, wbs23.AREA_M2)[0])
        equilibrium_angle = math.atan(
            wbs23.LEVER_M * mean_force / coeff["restoring_n_m_per_rad"])

        for frequency in FREQUENCIES_HZ:
            duration_s = CYCLES_TOTAL / float(frequency)
            count = int(round(duration_s * wbs23.FS)) + 1
            time_s = np.arange(count, dtype=float) / wbs23.FS
            true_wind = (MEAN_WIND_M_S + WIND_TONE_AMPLITUDE_M_S
                         * np.sin(2 * np.pi * frequency * time_s))
            force = drag_force_from_speed(
                true_wind, wbs23.AIR_DENSITY, wbs23.BALL_CD, wbs23.AREA_M2)

            # The WBS2.3 RK4 plant includes the selected grease b.  The RTS
            # receives the same coefficient dictionary: this is the matched case.
            angle, contact = wbs23.rk4_plant(
                time_s, force, coeff, equilibrium_angle)
            initial_state = np.array([equilibrium_angle, 0.0, mean_force])
            estimated_force = _fast_nonlinear_rts(
                angle, coeff, 1.0 / wbs23.FS, wbs23.LEVER_M,
                rts_angle_sigma, q_std, wbs23.EPSILON_DEG_S, initial_state)
            estimated_wind = (np.sign(estimated_force)
                              * np.sqrt(np.abs(estimated_force) / drag_factor))

            # Exclude startup/end cycles because RTS is an offline smoother and
            # finite-record edges otherwise affect the fitted tone.
            mask = ((time_s >= DROP_CYCLES_EACH_END / frequency)
                    & (time_s <= time_s[-1] - DROP_CYCLES_EACH_END / frequency))
            input_amp, input_phasor = tone_fit(time_s, true_wind, frequency, mask)
            output_amp, output_phasor = tone_fit(time_s, estimated_wind, frequency, mask)
            transfer = output_phasor / input_phasor
            gain = float(abs(transfer))
            records.append({
                "axis": axis,
                "frequency_hz": float(frequency),
                "mean_wind_m_s": MEAN_WIND_M_S,
                "input_tone_amplitude_m_s": input_amp,
                "estimated_tone_amplitude_m_s": output_amp,
                "gain_ratio_estimated_over_true": gain,
                "gain_db": float(20 * np.log10(max(gain, 1e-15))),
                "phase_deg_estimated_minus_true": float(np.rad2deg(np.angle(transfer))),
                "b_matched_Nm_s_rad": b_value,
                "q_force_rw_N_per_sample": q_std,
                "max_angle_deg": float(np.max(np.abs(np.rad2deg(angle)))),
                "hard_stop_contact": bool(np.any(contact)),
                "cycles_total": CYCLES_TOTAL,
                "cycles_scored": CYCLES_TOTAL - 2 * DROP_CYCLES_EACH_END,
                "angle_noise_added": False,
                "rts_assumed_angle_sigma_deg": float(np.rad2deg(rts_angle_sigma)),
            })

    frame = pd.DataFrame(records)
    for axis in ("IN", "OUT"):
        group = frame[frame.axis == axis]
        cutoff, plateau_db, target_db = cutoff_3db(group)
        cutoff_summary[axis] = {
            "low_frequency_gain_db": plateau_db,
            "minus_3db_target_db": target_db,
            "minus_3db_gain_ratio": 10 ** (target_db / 20),
            "minus_3db_cutoff_hz": cutoff,
        }

    csv_path = OUT / "rts_wind_speed_gain_vs_frequency.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.10g")
    settings = {
        "task": "OW-07 WBS2.3 matched RTS wind-speed frequency response",
        "frequency_hz": [float(x) for x in FREQUENCIES_HZ],
        "mean_wind_speed_m_s": MEAN_WIND_M_S,
        "wind_speed_tone_amplitude_m_s": WIND_TONE_AMPLITUDE_M_S,
        "plant_and_observer": "WBS2.3 matched 70 mm sphere; b=4.613e-3 N m s/rad in both plant and RTS",
        "sampling_hz": wbs23.FS,
        "sensor_noise_added": False,
        "rts_assumed_angle_sigma_deg": float(np.rad2deg(rts_angle_sigma)),
        "rts_q_force_rw_N_per_sample": wbs23.Q_BY_AXIS,
        "tone_fit": "least-squares constant + sin + cos at applied frequency; drop first/last two cycles",
        "gain_definition": "estimated wind-speed fundamental amplitude / true wind-speed fundamental amplitude",
        "cutoff_definition": "first downward -3 dB crossing relative to mean gain in the lowest three frequency points",
        "cutoff_summary": cutoff_summary,
        "nonlinear_response_note": "Empirical local nonlinear response about 2 m/s; amplitude and operating-point dependent, not a global LTI Bode transfer.",
    }
    (OUT / "rts_wind_speed_gain_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col",
                             gridspec_kw={"height_ratios": [1.15, 0.85]},
                             layout="constrained")
    colors = {"IN": "#2673a8", "OUT": "#dd8452"}
    for col, axis in enumerate(("IN", "OUT")):
        group = frame[frame.axis == axis].sort_values("frequency_hz")
        color = colors[axis]
        gain_ax, phase_ax = axes[0, col], axes[1, col]
        gain_ax.semilogx(group.frequency_hz, group.gain_ratio_estimated_over_true,
                         "o-", color=color, linewidth=1.7, markersize=4,
                         label=f"{axis} matched RTS")
        gain_ax.axhline(1.0, color="#444444", linestyle=":", linewidth=1,
                        label="unity gain")
        summary = cutoff_summary[axis]
        gain_ax.axhline(summary["minus_3db_gain_ratio"], color=color,
                        linestyle=":", alpha=0.65, linewidth=1,
                        label="−3 dB from low-frequency gain")
        cutoff = summary["minus_3db_cutoff_hz"]
        if cutoff is not None:
            gain_ax.axvline(cutoff, color=color, linestyle="--", alpha=0.7,
                            linewidth=1, label=f"−3 dB: {cutoff:.2f} Hz")
        gain_ax.set_title(f"{axis} axis")
        gain_ax.set_ylabel("Wind-speed amplitude gain [ratio]")
        gain_ax.set_ylim(bottom=0)
        gain_ax.grid(True, which="both", alpha=0.25)
        gain_ax.legend(fontsize=8, loc="best")

        # Once the estimated tone is almost extinguished, its phase becomes
        # numerically sensitive and is not useful as a design indicator.
        phase_display = group.phase_deg_estimated_minus_true.where(
            group.gain_ratio_estimated_over_true >= 0.05)
        phase_ax.semilogx(group.frequency_hz,
                          phase_display,
                          "o-", color=color, linewidth=1.5, markersize=3.5)
        phase_ax.axhline(0.0, color="#444444", linestyle=":", linewidth=1)
        phase_ax.set_xlabel("Input wind sine frequency [Hz]")
        phase_ax.set_ylabel("Phase difference [deg]\n(shown where gain ≥ 0.05)")
        phase_ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("WBS2.3 matched nonlinear RTS: wind-speed frequency response\n"
                 "2.0 m/s mean + 0.2 m/s sine; 70 mm sphere; selected grease b")
    png = OUT / "rts_wind_speed_gain_vs_frequency.png"
    svg = OUT / "rts_wind_speed_gain_vs_frequency.svg"
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    return frame, settings


if __name__ == "__main__":
    result, config = run()
    print(f"Wrote {len(result)} sweep rows")
    print(json.dumps(config["cutoff_summary"], indent=2))
