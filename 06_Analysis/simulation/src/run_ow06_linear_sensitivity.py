"""WBS 1.1-1.2 local linearization and sensitivity reference for OW-06.

Uses the adopted new-hardware BALL coefficients. Each axis is modeled as an
independent 1-DOF plant. The local transfer calculations are small-signal
references; the smoothed Coulomb-friction equivalent damping is separately
computed for finite sinusoidal angle amplitudes. RTS noise-tone response is
measured by driving a pure angle-measurement perturbation through the existing
nonlinear EKF-RTS implementation.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
SIM = HERE.parents[1]
ROOT = SIM.parent
sys.path.insert(0, str(HERE.parent))
from hbk_model_coefficients import select_coefficients  # noqa: E402
from hbk_nonlinear_estimators import nonlinear_ekf_rts_force  # noqa: E402

CFG = json.loads((SIM / "config" / "ow03_hbk_observer_tuning.json").read_text(encoding="utf-8"))
REGISTRY = SIM / "config" / "hbk_model_coefficients.json"
OW04 = SIM / "results" / "observer_wind" / "ow04_coefficient_sensitivity" / "ow04_joint_scenarios.csv"
OUT = SIM / "results" / "observer_wind" / "ow06_frequency_aware_observer_design"
SCENARIO = "joint_mass_high_arm_high_tau-high_c-high"
MEAN_WIND = 2.0
FREQS = (0.2, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0, 10.0)
ANGLE_AMPLITUDES_DEG = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0)
WIND_LEVELS = (0.0, 1.0, 2.0, 4.0, 6.0)
EPS = np.deg2rad(CFG["friction_epsilon_deg_s"])
DT = 1.0 / CFG["sample_rate_hz"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def applied_force(speed: float) -> float:
    return (0.5 * CFG["air_density_kg_m3"] * CFG["drag_coefficient"]
            * CFG["projected_area_m2"] * speed * abs(speed))


def local_properties(coeff: dict, force: float) -> dict:
    lever = CFG["force_lever_m"]
    theta = float(np.arctan(lever * force / coeff["restoring_n_m_per_rad"]))
    k_eff = (coeff["restoring_n_m_per_rad"] * np.cos(theta)
             + lever * force * np.sin(theta))
    c_local = (coeff["viscous_damping_n_m_s_per_rad"]
               + coeff["tau_n_m"] / EPS)
    inertia = coeff["inertia_kg_m2"]
    omega_n = float(np.sqrt(k_eff / inertia))
    zeta = float(c_local / (2.0 * np.sqrt(inertia * k_eff)))
    disc = c_local**2 - 4.0 * inertia * k_eff
    poles = [(-c_local + np.sqrt(disc)) / (2.0 * inertia),
             (-c_local - np.sqrt(disc)) / (2.0 * inertia)] if disc >= 0 else [None, None]
    return {"theta": theta, "k_eff": float(k_eff), "c_local": float(c_local),
            "omega_n": omega_n, "zeta": zeta, "poles": poles,
            "dc_gain_rad_per_n": float(lever * np.cos(theta) / k_eff)}


def equivalent_damping(coeff: dict, omega: float, amplitude_rad: float) -> tuple[float, float]:
    # Energy-equivalent viscous coefficient for one sinusoidal velocity cycle:
    # c_eq = <(tau_f*tanh(v/eps) + c_q*|v|v) v> / <v^2>.
    phase = np.linspace(0.0, 2.0 * np.pi, 16384, endpoint=False)
    v = omega * amplitude_rad * np.cos(phase)
    friction = coeff["tau_n_m"] * np.tanh(v / EPS)
    quadratic = coeff["total_quadratic_drag_n_m_s2_per_rad2"] * np.abs(v) * v
    c_eq = (coeff["viscous_damping_n_m_s_per_rad"]
            + float(np.mean((friction + quadratic) * v) / np.mean(v**2)))
    zeta_eq = c_eq / (2.0 * np.sqrt(coeff["inertia_kg_m2"]
                                    * (coeff["restoring_n_m_per_rad"])))
    return c_eq, float(zeta_eq)


def mismatch_coeff(axis: str, nominal: dict) -> dict:
    with OW04.open(encoding="utf-8-sig", newline="") as f:
        row = next(r for r in csv.DictReader(f)
                   if r["axis"] == axis and r["scenario"] == SCENARIO)
    result = nominal.copy()
    result["inertia_kg_m2"] = float(row["observer_I_kg_m2"])
    result["restoring_n_m_per_rad"] = float(row["observer_K_n_m_per_rad"])
    result["tau_n_m"] = float(row["observer_tau_n_m"])
    result["ball_quadratic_drag_n_m_s2_per_rad2"] = float(row["observer_c_ball_n_m_s2_per_rad2"])
    result["total_quadratic_drag_n_m_s2_per_rad2"] = (
        result["rod_quadratic_drag_n_m_s2_per_rad2"]
        + result["ball_quadratic_drag_n_m_s2_per_rad2"])
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    equilibrium_rows, response_rows, damping_rows, mismatch_rows, noise_rows = [], [], [], [], []
    mean_force = applied_force(MEAN_WIND)
    for axis, q in (("IN", 3e-5), ("OUT", 3e-4)):
        coeff = select_coefficients(axis, "BALL", REGISTRY)
        mismatch = mismatch_coeff(axis, coeff)
        for wind in WIND_LEVELS:
            force = applied_force(wind)
            p = local_properties(coeff, force)
            equilibrium_rows.append({
                "axis": axis, "mean_wind_m_s": wind, "mean_force_n": force,
                "equilibrium_angle_deg": np.rad2deg(p["theta"]),
                "effective_stiffness_n_m_per_rad": p["k_eff"],
                "local_damping_n_m_s_per_rad": p["c_local"],
                "inertial_frequency_scale_hz": p["omega_n"] / (2*np.pi),
                "local_damping_ratio": p["zeta"],
                "local_pole_1_per_s": p["poles"][0], "local_pole_2_per_s": p["poles"][1],
                "dc_angle_sensitivity_rad_per_n": p["dc_gain_rad_per_n"],
                "classical_resonance_peak": p["zeta"] < 1/np.sqrt(2),
            })
        nominal = local_properties(coeff, mean_force)
        alt = local_properties(mismatch, mean_force)
        for freq in FREQS:
            w = 2*np.pi*freq
            d = nominal["k_eff"] - coeff["inertia_kg_m2"]*w*w + 1j*nominal["c_local"]*w
            plant_gain = abs(CFG["force_lever_m"]*np.cos(nominal["theta"]) / d)
            inv_gain = 1.0 / plant_gain
            speed_derivative = MEAN_WIND/(2*mean_force)
            response_rows.append({
                "axis": axis, "mean_wind_m_s": MEAN_WIND, "frequency_hz": freq,
                "plant_angle_gain_rad_per_n": plant_gain,
                "gain_relative_to_dc": plant_gain/nominal["dc_gain_rad_per_n"],
                "unregularized_force_gain_n_per_rad_angle": inv_gain,
                "unregularized_wind_gain_m_s_per_rad_angle": inv_gain*speed_derivative,
                "unregularized_wind_error_for_0p02deg_m_s": inv_gain*np.deg2rad(0.02)*speed_derivative,
            })
            dhat = (alt["k_eff"] - mismatch["inertia_kg_m2"]*w*w
                    + 1j*alt["c_local"]*w)
            dtrue = (nominal["k_eff"] - coeff["inertia_kg_m2"]*w*w
                     + 1j*nominal["c_local"]*w)
            ratio = (dhat/dtrue) * np.cos(nominal["theta"])/np.cos(alt["theta"])
            mismatch_rows.append({
                "axis": axis, "mean_wind_m_s": MEAN_WIND, "frequency_hz": freq,
                "observer_model_to_plant_force_ratio_unregularized": abs(ratio),
                "phase_error_deg": np.rad2deg(np.angle(ratio)),
                "relative_force_error_amplitude": abs(ratio-1.0),
                "scope_note": "linearized inverse reference; excludes RTS/Kalman regularization",
            })
        for amp_deg in ANGLE_AMPLITUDES_DEG:
            c_eq, z_eq = equivalent_damping(coeff, nominal["omega_n"], np.deg2rad(amp_deg))
            damping_rows.append({
                "axis": axis, "mean_wind_m_s": MEAN_WIND,
                "oscillation_amplitude_deg": amp_deg,
                "rate_amplitude_rad_s": nominal["omega_n"]*np.deg2rad(amp_deg),
                "energy_equivalent_damping_n_m_s_per_rad": c_eq,
                "equivalent_damping_ratio": z_eq,
                "frequency_used_hz": nominal["omega_n"]/(2*np.pi),
            })
        # Pure sinusoidal angle-measurement perturbation; true plant angle is fixed.
        theta0 = nominal["theta"]
        time = np.arange(int(30.0/DT))*DT
        angle_amp = np.deg2rad(0.02)
        dVdF = MEAN_WIND/(2*mean_force)
        for freq in FREQS:
            measured = theta0 + angle_amp*np.sin(2*np.pi*freq*time)
            _, estimate = nonlinear_ekf_rts_force(
                measured, coeff, DT, CFG["force_lever_m"], np.deg2rad(0.02), q, 0,
                CFG["friction_epsilon_deg_s"],
                initial_state=np.array([theta0, 0.0, mean_force]),
            )
            mask = time >= 5.0
            phase = 2*np.pi*freq*time[mask]
            design = np.column_stack((np.sin(phase), np.cos(phase), np.ones(mask.sum())))
            fit, *_ = np.linalg.lstsq(design, estimate[mask], rcond=None)
            force_amp = float(np.hypot(fit[0], fit[1]))
            wind_amp = force_amp*dVdF
            noise_rows.append({
                "axis": axis, "mean_wind_m_s": MEAN_WIND, "frequency_hz": freq,
                "angle_error_tone_amplitude_deg": 0.02,
                "estimated_force_tone_amplitude_n": force_amp,
                "equivalent_wind_tone_amplitude_m_s": wind_amp,
                "wind_amplitude_per_angle_error_m_s_per_deg": wind_amp/0.02,
                "q_n_per_sample": q,
                "note": "single-tone angle error; nonlinear 3-state RTS, 30 s record, last 25 s fitted",
            })
    write_csv(OUT/"ow06_equilibrium_linearization.csv", equilibrium_rows)
    write_csv(OUT/"ow06_small_signal_frequency_response.csv", response_rows)
    write_csv(OUT/"ow06_amplitude_equivalent_damping.csv", damping_rows)
    write_csv(OUT/"ow06_unregularized_mismatch_reference.csv", mismatch_rows)
    write_csv(OUT/"ow06_rts_angle_noise_transfer.csv", noise_rows)
    print("Wrote", len(equilibrium_rows), "operating points,", len(response_rows),
          "frequency points,", len(damping_rows), "amplitude damping points,",
          len(mismatch_rows), "mismatch points, and", len(noise_rows), "RTS noise points to", OUT)


if __name__ == "__main__":
    main()
