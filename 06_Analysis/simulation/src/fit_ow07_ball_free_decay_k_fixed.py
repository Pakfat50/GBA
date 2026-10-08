"""Fit BALL free-decay coefficients with K fixed, then test zero-wind RTS output.

Uses every approved BALL record (12 total, six per axis). K remains at the
physical/registry value. I, tau, and total quadratic drag c are fit per axis;
b stays fixed at the adopted zero value. This is a batch, prior-initialized
nonlinear trajectory fit, not an augmented-state Kalman parameter tracker.

Run from repository root:
    python 06_Analysis/simulation/src/fit_ow07_ball_free_decay_k_fixed.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import least_squares

from hbk_model_coefficients import select_coefficients
from hbk_nonlinear_estimators import nonlinear_ekf_rts_force
from run_ow03_hbk_observer_tuning import force_to_speed

REPO = Path(__file__).resolve().parents[3]
SIM = REPO / "06_Analysis/simulation"
OUT = SIM / "results/observer_wind/ow07_ball_coeff_refit_k_fixed"
DATA = REPO / "04_Data/05_Fitting/20260921"
SELECTION = REPO / "06_Analysis/fitting_pipeline/results/20260921/waveform_review/waveform_selection.csv"
CONFIG = json.loads((SIM / "config/ow03_hbk_observer_tuning.json").read_text(encoding="utf-8"))
FS = float(CONFIG["sample_rate_hz"])
DT = 1.0 / FS
EPSILON = math.radians(float(CONFIG["friction_epsilon_deg_s"]))
ANGLE_SIGMA = math.radians(float(CONFIG["assumed_angle_noise_deg"]))
Q_CURRENT = 1.0e-3
Q_SELECTED = {"IN": 3.0e-5, "OUT": 3.0e-4}
SCORE_START_S = 3.0
SCORE_END_MARGIN_S = 0.5
FIT_START_S = 0.2
FIT_END_MARGIN_S = 0.5
FIT_STRIDE = 2  # 50 Hz objective; final observer remains at 100 Hz


def rhs(state: np.ndarray, inertia: float, stiffness: float,
        quadratic: float, coulomb: float) -> np.ndarray:
    angle, rate = state
    acceleration = (
        -stiffness * math.sin(angle)
        -quadratic * abs(rate) * rate
        -coulomb * math.tanh(rate / EPSILON)
    ) / inertia
    return np.array([rate, acceleration])


def integrate(angle0: float, rate0: float, n: int, dt: float,
              inertia: float, stiffness: float, quadratic: float,
              coulomb: float) -> np.ndarray:
    state = np.array([angle0, rate0], dtype=float)
    output = np.empty(n, dtype=float)
    for i in range(n):
        output[i] = state[0]
        if i + 1 == n:
            break
        k1 = rhs(state, inertia, stiffness, quadratic, coulomb)
        k2 = rhs(state + 0.5*dt*k1, inertia, stiffness, quadratic, coulomb)
        k3 = rhs(state + 0.5*dt*k2, inertia, stiffness, quadratic, coulomb)
        k4 = rhs(state + dt*k3, inertia, stiffness, quadratic, coulomb)
        state += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return output


def load_records() -> list[dict]:
    selected = pd.read_csv(SELECTION, encoding="utf-8-sig")
    selected = selected[
        (pd.to_numeric(selected["use_for_fitting"], errors="coerce") == 1)
        & (selected["review_status"].astype(str).str.upper() == "APPROVED")
        & (selected["configuration"].astype(str).str.upper() == "BALL")
    ]
    cache: dict[str, pd.DataFrame] = {}
    records = []
    for row in selected.to_dict("records"):
        relative = str(row["data_file"])
        if relative not in cache:
            cache[relative] = pd.read_csv(DATA / relative)
        frame = cache[relative]
        axis = str(row["axis"])
        column = "angle0[deg]" if axis == "IN" else "angle1[deg]"
        full = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(full)
        if not np.all(finite):
            ids = np.arange(len(full))
            full[~finite] = np.interp(ids[~finite], ids[finite], full[finite])
        full = (full + 180.0) % 360.0 - 180.0
        baseline = float(row.get("baseline_deg", 0.0))
        full -= baseline
        start, end = int(row["start_index"]), int(row["end_index"])
        direction = str(row["direction"]).upper()
        direction_sign = 1.0 if direction == "P" else -1.0
        signed = direction_sign * full[start:end]
        search_n = min(len(signed), max(10, int(round(0.8 * FS))))
        peak_local = int(np.argmax(signed[:search_n]))
        held_peak = float(signed[peak_local])
        release_local = peak_local
        for k in range(peak_local + 1, len(signed) - 2):
            if np.all(signed[k:k+3] < held_peak - 0.15):
                release_local = max(peak_local, k - 1)
                break
        release_index = start + release_local
        angle_deg = full[release_index:end].copy()
        angle = np.deg2rad(angle_deg)
        t = np.arange(len(angle), dtype=float) * DT
        lo = max(0, release_index - 5)
        hi = min(len(full), release_index + 6)
        local_t = (np.arange(lo, hi, dtype=float) - release_index) * DT
        local_angle = np.deg2rad(full[lo:hi])
        rate0 = float(np.polyfit(local_t, local_angle, 2)[1]) if len(local_t) >= 3 else 0.0
        fit_mask = (t >= FIT_START_S) & (t <= t[-1] - FIT_END_MARGIN_S)
        score_mask = (t >= SCORE_START_S) & (t <= t[-1] - SCORE_END_MARGIN_S)
        if fit_mask.sum() < 100 or score_mask.sum() < 100:
            continue
        records.append({
            **row, "axis": axis, "direction": direction,
            "segment_id": str(row["segment_id"]), "baseline_deg": baseline,
            "time": t, "angle": angle, "angle_deg": angle_deg,
            "angle0": float(angle[0]), "rate0": rate0,
            "fit_mask": fit_mask, "score_mask": score_mask,
        })
    if len(records) != 12:
        raise RuntimeError(f"Expected 12 approved BALL records, found {len(records)}")
    return records


def unpack(z: np.ndarray, initial: np.ndarray) -> np.ndarray:
    return initial * np.exp(z)  # [I, tau, c], all strictly positive


def fit_axis(axis: str, records: list[dict], initial: np.ndarray,
             stiffness: float, exclude: str | None = None):
    subset = [r for r in records if r["axis"] == axis and r["segment_id"] != exclude]
    if len(subset) < 4:
        raise ValueError("At least four training waveforms are required per axis")

    def residual(z: np.ndarray) -> np.ndarray:
        inertia, coulomb, quadratic = unpack(z, initial)
        parts = []
        for r in subset:
            ids = np.arange(0, len(r["time"]), FIT_STRIDE)
            valid = r["fit_mask"][ids]
            model = integrate(r["angle0"], r["rate0"], len(ids), DT*FIT_STRIDE,
                              inertia, stiffness, quadratic, coulomb)
            err = np.rad2deg(model[valid] - r["angle"][ids[valid]])
            parts.append(err)
        return np.concatenate(parts)

    # Broad but finite bounds expose large required corrections without
    # allowing the optimizer to cross nonphysical zero/negative coefficients.
    lower = np.log(np.array([0.5, 0.1, 0.1]))
    upper = np.log(np.array([1.5, 8.0, 8.0]))
    result = least_squares(
        residual, np.zeros(3), bounds=(lower, upper), x_scale="jac",
        loss="soft_l1", f_scale=0.4, max_nfev=260,
        ftol=2e-9, xtol=2e-9, gtol=2e-9,
    )
    if not result.success and result.status <= 0:
        raise RuntimeError(f"{axis} fit failed: {result.message}")
    return unpack(result.x, initial), result


def make_coefficients(base: dict, fit: np.ndarray) -> dict:
    inertia, coulomb, quadratic = map(float, fit)
    result = base.copy()
    result["inertia_kg_m2"] = inertia
    # K is explicitly fixed at the current physical/registry value.
    result["viscous_damping_n_m_s_per_rad"] = 0.0
    result["tau_n_m"] = coulomb
    result["total_quadratic_drag_n_m_s2_per_rad2"] = quadratic
    # Do not imply that the fitted effective c retains the old rod/ball split.
    result["rod_quadratic_drag_n_m_s2_per_rad2"] = None
    result["ball_quadratic_drag_n_m_s2_per_rad2"] = None
    return result


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def observer_run(angle: np.ndarray, coeff: dict, q: float, r: dict) -> np.ndarray:
    initial = np.array([r["angle0"], r["rate0"], 0.0])
    _, force = nonlinear_ekf_rts_force(
        angle, coeff, DT, float(CONFIG["force_lever_m"]), ANGLE_SIGMA,
        q, 0, float(CONFIG["friction_epsilon_deg_s"]), initial_state=initial,
    )
    return force_to_speed(CONFIG, force)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    initial: dict[str, np.ndarray] = {}
    fixed_k: dict[str, float] = {}
    old: dict[str, dict] = {}
    fitted: dict[str, dict] = {}
    fit_rows = []
    angle_rows = []
    loo_rows = []
    for axis in ("IN", "OUT"):
        old[axis] = select_coefficients(axis, "BALL")
        fixed_k[axis] = float(old[axis]["restoring_n_m_per_rad"])
        initial[axis] = np.array([
            float(old[axis]["inertia_kg_m2"]),
            float(old[axis]["tau_n_m"]),
            float(old[axis]["total_quadratic_drag_n_m_s2_per_rad2"]),
        ])
        estimate, result = fit_axis(axis, records, initial[axis], fixed_k[axis])
        fitted[axis] = make_coefficients(old[axis], estimate)
        names = ("I", "tau", "c")
        units = ("kg m^2", "N m", "N m s^2/rad^2")
        keys = ("inertia_kg_m2", "tau_n_m", "total_quadratic_drag_n_m_s2_per_rad2")
        for i, (name, unit, key) in enumerate(zip(names, units, keys)):
            fit_rows.append({
                "axis": axis, "parameter": name, "unit": unit,
                "K_fixed_Nm_per_rad": fixed_k[axis],
                "initial": initial[axis][i], "fitted": estimate[i],
                "change_percent": 100.0*(estimate[i]/initial[axis][i]-1.0),
                "lower_bound_ratio": (0.5, 0.1, 0.1)[i],
                "upper_bound_ratio": (1.5, 8.0, 8.0)[i],
            })
        # Parameter identifiability diagnostic from the local robust Jacobian.
        singular = np.linalg.svd(result.jac, compute_uv=False)
        rank = int(np.linalg.matrix_rank(result.jac))
        condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else float("inf")
        fit_rows.append({"axis": axis, "parameter": "fit_diagnostic", "unit": "",
                         "K_fixed_Nm_per_rad": fixed_k[axis], "initial": None,
                         "fitted": None, "change_percent": None,
                         "lower_bound_ratio": None, "upper_bound_ratio": None,
                         "nfev": int(result.nfev), "cost": float(result.cost),
                         "jacobian_rank": rank, "jacobian_condition_number": condition,
                         "optimizer_message": result.message})

    # Forward angle validation, including one-waveform-out refits.
    for r in records:
        axis = r["axis"]
        for label, coeff in (("current", old[axis]), ("K_fixed_refit", fitted[axis])):
            y = integrate(r["angle0"], r["rate0"], len(r["time"]), DT,
                          float(coeff["inertia_kg_m2"]), fixed_k[axis],
                          float(coeff["total_quadratic_drag_n_m_s2_per_rad2"]),
                          float(coeff["tau_n_m"]))
            error = np.rad2deg(y-r["angle"])
            angle_rows.append({"segment_id": r["segment_id"], "axis": axis,
                               "direction": r["direction"], "model": label,
                               "score_samples": int(r["score_mask"].sum()),
                               "angle_rmse_deg": rms(error[r["score_mask"]]),
                               "angle_mae_deg": float(np.mean(np.abs(error[r["score_mask"]]))),
                               "angle_bias_deg": float(np.mean(error[r["score_mask"]]))})
        held, _ = fit_axis(axis, records, initial[axis], fixed_k[axis], exclude=r["segment_id"])
        yloo = integrate(r["angle0"], r["rate0"], len(r["time"]), DT,
                         held[0], fixed_k[axis], held[2], held[1])
        error = np.rad2deg(yloo-r["angle"])
        loo_rows.append({"segment_id": r["segment_id"], "axis": axis,
                         "training_records": 5,
                         "heldout_angle_rmse_deg": rms(error[r["score_mask"]]),
                         "heldout_angle_mae_deg": float(np.mean(np.abs(error[r["score_mask"]])))})

    # Apply zero-wind RTS to all measured BALL waveforms. Compare old and
    # refitted mechanical coefficients at current and reviewed per-axis q.
    wind_rows = []
    wind_metrics = []
    q_specs = {"q_0.001": Q_CURRENT, "q_selected": None}
    for r in records:
        axis = r["axis"]
        coeff_pair = (("current", old[axis]), ("K_fixed_refit", fitted[axis]))
        for q_name, q_default in q_specs.items():
            q = Q_SELECTED[axis] if q_default is None else q_default
            for model_name, coeff in coeff_pair:
                wind = observer_run(r["angle"], coeff, q, r)
                score = r["score_mask"]
                bias = float(np.mean(wind[score]))
                metric_row = {
                    "segment_id": r["segment_id"], "axis": axis,
                    "direction": r["direction"], "q_case": q_name,
                    "q_N_per_sample": q, "model": model_name,
                    "true_wind_m_s": 0.0,
                    "wind_bias_m_s": bias,
                    "wind_zero_rmse_m_s": rms(wind[score]),
                    "wind_bias_removed_rms_m_s": rms(wind[score]-bias),
                    "wind_zero_mae_m_s": float(np.mean(np.abs(wind[score]))),
                    "wind_zero_p95_abs_m_s": float(np.quantile(np.abs(wind[score]), .95)),
                }
                wind_metrics.append(metric_row)
                for i, (t, obs_angle, value) in enumerate(zip(r["time"], r["angle_deg"], wind)):
                    wind_rows.append({**metric_row, "time_s": float(t),
                                      "measured_angle_deg": float(obs_angle),
                                      "estimated_wind_m_s": float(value),
                                      "score_window": bool(score[i])})

    pd.DataFrame(fit_rows).to_csv(OUT/"ow07_ball_coefficients_k_fixed.csv", index=False,
                                   encoding="utf-8-sig", float_format="%.10g")
    pd.DataFrame(angle_rows).to_csv(OUT/"ow07_ball_angle_fit_metrics.csv", index=False,
                                     encoding="utf-8-sig", float_format="%.8g")
    pd.DataFrame(loo_rows).to_csv(OUT/"ow07_ball_angle_leave_one_out.csv", index=False,
                                   encoding="utf-8-sig", float_format="%.8g")
    pd.DataFrame(wind_metrics).to_csv(OUT/"ow07_ball_zero_wind_metrics.csv", index=False,
                                       encoding="utf-8-sig", float_format="%.8g")
    pd.DataFrame(wind_rows).to_csv(OUT/"ow07_ball_zero_wind_waveforms.csv", index=False,
                                    encoding="utf-8-sig", float_format="%.8g")
    (OUT/"ow07_ball_coefficients_k_fixed.json").write_text(json.dumps({
        "method": "Batch robust nonlinear least squares on all approved BALL free decays, K fixed",
        "coefficient_status": "diagnostic refit; not copied into adopted registry",
        "records": len(records), "records_by_axis": {a: sum(r["axis"] == a for r in records) for a in ("IN", "OUT")},
        "fixed_K_by_axis_Nm_per_rad": fixed_k,
        "current_coefficients": old, "fitted_coefficients": fitted,
        "b_fixed_Nm_s_per_rad": 0.0,
        "fit_frequency_hz": FS/FIT_STRIDE,
        "fit_window_s": {"start_after_release": FIT_START_S, "end_before_stop": FIT_END_MARGIN_S},
        "observer": "3-state EKF/RTS offline smoother; true applied wind is zero",
        "q_cases_N_per_sample": {"q_0.001": Q_CURRENT, "q_selected_by_axis": Q_SELECTED},
        "sensor_noise_model": "measured sensor traces are used; q settings from OW-05 selected values and recent q=0.001 comparison",
    }, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    # One panel per waveform; all twelve approved BALL records are visible.
    frame = pd.DataFrame(wind_rows)
    for q_name, filename, title in (
        ("q_0.001", "ow07_ball_zero_wind_all_waveforms_q001.png", "RTS q=0.001 N/sample"),
        ("q_selected", "ow07_ball_zero_wind_all_waveforms_q_selected.png", "OW-05 selected per-axis RTS q"),
    ):
        fig, axes = plt.subplots(6, 2, figsize=(14, 15), sharex="col", layout="constrained")
        for col, axis in enumerate(("IN", "OUT")):
            group = sorted([r for r in records if r["axis"] == axis],
                           key=lambda r: (r["direction"], int(r["repetition"])))
            for row_i, rec in enumerate(group):
                ax = axes[row_i, col]
                subset = frame[(frame.segment_id == rec["segment_id"]) & (frame.q_case == q_name)]
                for model_name, color, label in (("current", "#555555", "Current"),
                                                 ("K_fixed_refit", "#D55E00", "K-fixed refit")):
                    s = subset[subset.model == model_name]
                    ax.plot(s.time_s, s.estimated_wind_m_s, color=color, lw=.9, label=label)
                ax.axhline(0.0, color="#2878a5", ls="--", lw=.75, label="True wind = 0")
                ax.axvspan(SCORE_START_S, max(SCORE_START_S, rec["time"][-1]-SCORE_END_MARGIN_S),
                           color="#cccccc", alpha=.15, lw=0)
                ax.set_title(f"{axis} {rec['segment_id']}  |  q={Q_SELECTED[axis] if q_name=='q_selected' else Q_CURRENT:g}",
                             loc="left", pad=2, fontsize=8)
                ax.grid(alpha=.22)
                if col == 0:
                    ax.set_ylabel("Estimated wind (m/s)")
                if row_i == 5:
                    ax.set_xlabel("Time after release (s)")
                if row_i == 0 and col == 0:
                    ax.legend(fontsize=7, ncol=3, loc="upper right", frameon=False)
        fig.suptitle(f"All 12 approved BALL free decays — zero-wind RTS estimate\n{title}; current vs K-fixed I/τ/c refit", fontsize=13)
        figure_path = OUT/filename
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        # These are dense 12-panel figures. An indexed palette keeps them
        # legible while making the report images practical to version.
        with Image.open(figure_path) as image:
            palette_image = image.convert("RGB").quantize(
                colors=256, method=Image.Quantize.FASTOCTREE,
                dither=Image.Dither.NONE,
            )
            palette_image.save(figure_path, optimize=True)

    print("OUTPUT", OUT)
    for axis in ("IN", "OUT"):
        print("COEFF", axis, "K fixed", fixed_k[axis])
        print("COEFF", axis, "I/tau/c initial", initial[axis].tolist(), "fit", [
            fitted[axis]["inertia_kg_m2"], fitted[axis]["tau_n_m"],
            fitted[axis]["total_quadratic_drag_n_m_s2_per_rad2"]])
        for q_name in ("q_0.001", "q_selected"):
            for model_name in ("current", "K_fixed_refit"):
                m = [x for x in wind_metrics if x["axis"] == axis and x["q_case"] == q_name and x["model"] == model_name]
                print("WIND", axis, q_name, model_name,
                      "total RMSE mean", float(np.mean([x["wind_zero_rmse_m_s"] for x in m])),
                      "mean abs bias", float(np.mean(np.abs([x["wind_bias_m_s"] for x in m]))),
                      "centered RMS mean", float(np.mean([x["wind_bias_removed_rms_m_s"] for x in m])))
        for label in ("current", "K_fixed_refit"):
            values = [x["angle_rmse_deg"] for x in angle_rows if x["axis"] == axis and x["model"] == label]
            print("ANGLE_RMSE", axis, label, "mean", float(np.mean(values)))
        values = [x["heldout_angle_rmse_deg"] for x in loo_rows if x["axis"] == axis]
        print("LOO_RMSE", axis, "mean", float(np.mean(values)))


if __name__ == "__main__":
    main()
