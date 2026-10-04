"""Refit the zero-wind BALL dynamics directly to approved free-decay waveforms.

This is a diagnostic fit kept separate from the adopted coefficient registry.
It fits normalized dynamics (K/I, b/I, c/I, tau/I) because free decay alone
cannot identify the absolute inertia scale. The fitted ratios are converted
to physical coefficients using the registry inertia solely to run the force
observer. Outputs include angle-model overlays and RTS wind estimates at the
original q=1e-3 and the reviewed smaller per-axis q values.

Run from the repository root:
    python 06_Analysis/simulation/src/fit_ow05_ball_free_decay_coefficients.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[3]
SIM = REPO / "06_Analysis/simulation"
OUT = SIM / "results/observer_wind/ow05_sensor_nonideality"
sys.path.insert(0, str(SIM / "src"))
from hbk_model_coefficients import select_coefficients
from hbk_nonlinear_estimators import nonlinear_ekf_rts_force
from run_ow03_hbk_observer_tuning import force_to_speed

DATA = REPO / "04_Data/05_Fitting/20260921"
SELECTION = REPO / "06_Analysis/fitting_pipeline/results/20260921/waveform_review/waveform_selection.csv"
CONFIG = json.loads((SIM / "config/ow03_hbk_observer_tuning.json").read_text())
SAMPLE_RATE = float(CONFIG["sample_rate_hz"])
DT = 1.0 / SAMPLE_RATE
EPSILON = math.radians(float(CONFIG["friction_epsilon_deg_s"]))
ANGLE_SIGMA = math.radians(float(CONFIG["assumed_angle_noise_deg"]))
Q_OLD = 1e-3
Q_REVIEWED = {"IN": 3e-5, "OUT": 3e-4}
SCORE_START_S, SCORE_END_MARGIN_S = 3.0, 0.5


def rhs(state: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Normalized zero-wind dynamics, p=[K/I,b/I,c/I,tau/I]."""
    angle, rate = state
    stiffness, viscous, quadratic, coulomb = p
    acceleration = (-stiffness * math.sin(angle)
                    - viscous * rate
                    - quadratic * abs(rate) * rate
                    - coulomb * math.tanh(rate / EPSILON))
    return np.array([rate, acceleration])


def integrate(angle0: float, rate0: float, n: int, dt: float,
              p: np.ndarray) -> np.ndarray:
    state = np.array([angle0, rate0], dtype=float)
    output = np.empty(n, dtype=float)
    for i in range(n):
        output[i] = state[0]
        if i + 1 == n:
            break
        k1 = rhs(state, p)
        k2 = rhs(state + 0.5 * dt * k1, p)
        k3 = rhs(state + 0.5 * dt * k2, p)
        k4 = rhs(state + dt * k3, p)
        state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return output


def initial_rate(angle_rad: np.ndarray) -> float:
    t = np.arange(min(len(angle_rad), int(round(.20 * SAMPLE_RATE)))) * DT
    return float(np.polyfit(t, angle_rad[:len(t)], 1)[0])


def read_records() -> list[dict]:
    selection = pd.read_csv(SELECTION, encoding="utf-8-sig")
    selection = selection[(pd.to_numeric(selection["use_for_fitting"], errors="coerce") == 1)
                          & (selection["review_status"].astype(str).str.upper() == "APPROVED")
                          & (selection["configuration"].astype(str).str.upper() == "BALL")]
    cache: dict[str, pd.DataFrame] = {}
    records = []
    for row in selection.to_dict("records"):
        rel = str(row["data_file"])
        if rel not in cache:
            cache[rel] = pd.read_csv(DATA / rel)
        frame = cache[rel]
        col = "angle0[deg]" if row["axis"] == "IN" else "angle1[deg]"
        angle_deg = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        angle_deg = (angle_deg + 180.0) % 360.0 - 180.0
        angle_deg = angle_deg[int(row["start_index"]):int(row["end_index"])]
        angle_rad = np.deg2rad(angle_deg)
        t = np.arange(len(angle_rad)) * DT
        score = (t >= SCORE_START_S) & (t <= t[-1] - SCORE_END_MARGIN_S)
        records.append({**row, "time": t, "angle_deg": angle_deg,
                        "angle_rad": angle_rad, "rate0": initial_rate(angle_rad),
                        "score": score})
    if len(records) != 12:
        raise RuntimeError(f"Expected 12 approved BALL free-decay records, found {len(records)}")
    return records


def fit_axis(axis: str, records: list[dict], initial: np.ndarray,
             exclude_segment: str | None = None) -> tuple[np.ndarray, object]:
    subset = [r for r in records if r["axis"] == axis
              and r["segment_id"] != exclude_segment]
    scale = np.array([initial[0], 0.1, max(initial[2], 0.01), initial[3]])
    # Optimize every other sample for speed; 50 Hz still resolves the ~1 Hz
    # free-decay waveform with ample margin. The final models use 100 Hz.
    def residual(z: np.ndarray) -> np.ndarray:
        parts = []
        for r in subset:
            ids = np.arange(0, len(r["time"]), 2)
            p = z * scale
            model = integrate(r["angle_rad"][0], r["rate0"], len(ids), DT*2, p)
            parts.append((np.rad2deg(model) - r["angle_deg"][ids]))
        return np.concatenate(parts)

    # Scaled variables keep the optimizer well conditioned. Positive bounds
    # constrain damping terms while allowing the data to decide whether b>0.
    x0 = np.array([1.0, 0.1, 1.0, 1.0])
    lower = np.array([0.65, 0.0, 1e-4, 0.05])
    upper = np.array([1.35, 10.0, 100.0, 10.0])
    result = least_squares(residual, x0, bounds=(lower, upper), x_scale="jac",
                           loss="soft_l1", f_scale=0.4, max_nfev=160,
                           ftol=2e-8, xtol=2e-8, gtol=2e-8, verbose=0)
    if not result.success and result.status <= 0:
        raise RuntimeError(f"{axis} fit failed: {result.message}")
    return result.x * scale, result


def metric(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = read_records()
    fits: dict[str, np.ndarray] = {}
    optim: dict[str, object] = {}
    old_ratios: dict[str, np.ndarray] = {}
    for axis in ("IN", "OUT"):
        base = select_coefficients(axis, "BALL")
        inertia = float(base["inertia_kg_m2"])
        old_ratios[axis] = np.array([
            float(base["restoring_n_m_per_rad"]) / inertia,
            float(base["viscous_damping_n_m_s_per_rad"]) / inertia,
            float(base["total_quadratic_drag_n_m_s2_per_rad2"]) / inertia,
            float(base["tau_n_m"]) / inertia,
        ])
        fits[axis], optim[axis] = fit_axis(axis, records, old_ratios[axis])

    # Leave one full waveform out at a time to check whether the coefficient
    # improvement transfers to a trace not used in that fit.
    cv_rows = []
    for axis in ("IN", "OUT"):
        for held in [r for r in records if r["axis"] == axis]:
            cv_coeff, _ = fit_axis(axis, records, old_ratios[axis],
                                   exclude_segment=held["segment_id"])
            cv_model = integrate(held["angle_rad"][0], held["rate0"],
                                 len(held["time"]), DT, cv_coeff)
            old_model = integrate(held["angle_rad"][0], held["rate0"],
                                  len(held["time"]), DT, old_ratios[axis])
            score = held["score"]
            old_rmse = metric(np.rad2deg(old_model[score]) - held["angle_deg"][score])
            cv_rmse = metric(np.rad2deg(cv_model[score]) - held["angle_deg"][score])
            cv_rows.append({"held_out_segment_id": held["segment_id"], "axis": axis,
                            "training_waveforms": 5, "old_model_angle_rmse_deg": old_rmse,
                            "refit_model_angle_rmse_deg": cv_rmse,
                            "refit_to_old_ratio": cv_rmse/old_rmse})

    parameters = []
    coefficients = {}
    for axis in ("IN", "OUT"):
        base = select_coefficients(axis, "BALL")
        inertia = float(base["inertia_kg_m2"])
        p = fits[axis]
        coefficients[axis] = {
            **base,
            "restoring_n_m_per_rad": float(p[0] * inertia),
            "viscous_damping_n_m_s_per_rad": float(p[1] * inertia),
            "rod_quadratic_drag_n_m_s2_per_rad2": None,
            "ball_quadratic_drag_n_m_s2_per_rad2": None,
            "total_quadratic_drag_n_m_s2_per_rad2": float(p[2] * inertia),
            "tau_n_m": float(p[3] * inertia),
            "sources": {"refit": "Diagnostic direct time-domain fit to 6 approved BALL free-decay waveforms for this axis; not adopted."},
        }
        for name, old, new, unit in zip(
            ("K_over_I", "b_over_I", "c_over_I", "tau_over_I"),
            old_ratios[axis], p, ("s^-2", "s^-1", "1", "s^-2"),
        ):
            parameters.append({"axis": axis, "parameter": name, "unit": unit,
                               "adopted_old": float(old), "waveform_refit": float(new),
                               "ratio_refit_to_old": float(new/old) if old else None})
        for name, key, unit in (
            ("inertia", "inertia_kg_m2", "kg m^2"),
            ("restoring_K", "restoring_n_m_per_rad", "N m/rad"),
            ("viscous_b", "viscous_damping_n_m_s_per_rad", "N m s/rad"),
            ("quadratic_c", "total_quadratic_drag_n_m_s2_per_rad2", "N m s^2/rad^2"),
            ("coulomb_tau", "tau_n_m", "N m"),
        ):
            parameters.append({"axis": axis, "parameter": name, "unit": unit,
                               "adopted_old": float(base[key]),
                               "waveform_refit": float(coefficients[axis][key]),
                               "ratio_refit_to_old": float(coefficients[axis][key]/base[key]) if base[key] else None})
    coeff_path = OUT / "ow05_ball_free_decay_refit_coefficients.csv"
    pd.DataFrame(parameters).to_csv(coeff_path, index=False, encoding="utf-8-sig", float_format="%.10g")
    coeff_json = OUT / "ow05_ball_free_decay_refit_coefficients.json"
    coeff_json.write_text(json.dumps({"purpose": "diagnostic waveform refit; not adopted registry",
       "fit_scope": "12 approved BALL free-decay waveforms, 6 per axis",
       "absolute_scale_note": "Free decay identifies normalized K/I, b/I, c/I, tau/I. Physical coefficients below anchor inertia to current registry values.",
       "coefficients_by_axis": coefficients}, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    summary_rows, wave_rows = [], []
    representative = {"IN": "IN_BALL_P_R02", "OUT": "OUT_BALL_P_R02"}
    for r in records:
        axis = r["axis"]
        model_old = integrate(r["angle_rad"][0], r["rate0"], len(r["time"]), DT, old_ratios[axis])
        model_fit = integrate(r["angle_rad"][0], r["rate0"], len(r["time"]), DT, fits[axis])
        score = r["score"]
        residual_old = np.rad2deg(model_old) - r["angle_deg"]
        residual_fit = np.rad2deg(model_fit) - r["angle_deg"]
        summary_rows.append({"segment_id": r["segment_id"], "axis": axis,
            "direction": r["direction"], "repetition": int(r["repetition"]),
            "samples_scored": int(score.sum()),
            "angle_rmse_old_model_deg": metric(residual_old[score]),
            "angle_rmse_refit_model_deg": metric(residual_fit[score]),
            "residual_rms_ratio_refit_to_old": metric(residual_fit[score])/metric(residual_old[score]),
            "old_model_residual_acf1": float(np.corrcoef(residual_old[score][:-1], residual_old[score][1:])[0,1]),
            "refit_model_residual_acf1": float(np.corrcoef(residual_fit[score][:-1], residual_fit[score][1:])[0,1]),
        })
        for coeff_name, coeff, ratios in (
            ("adopted", select_coefficients(axis, "BALL"), old_ratios[axis]),
            ("waveform_refit", coefficients[axis], fits[axis]),
        ):
            for q_type, q_value in (("initial_q_0.001", Q_OLD),
                                    ("reviewed_smaller_q", Q_REVIEWED[axis])):
                try:
                    _, force = nonlinear_ekf_rts_force(r["angle_rad"], coeff, DT,
                        float(CONFIG["force_lever_m"]), ANGLE_SIGMA, q_value, 0,
                        float(CONFIG["friction_epsilon_deg_s"]),
                        initial_state=np.array([r["angle_rad"][0], r["rate0"], 0.0]))
                    wind = force_to_speed(CONFIG, force)
                    bias = float(np.mean(wind[score]))
                    wind_centered_rms = metric(wind[score] - bias)
                    wind_total_rmse = metric(wind[score])
                except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                    wind = np.full(len(r["time"]), np.nan)
                    bias = wind_centered_rms = wind_total_rmse = float("nan")
                summary_rows[-1][f"{coeff_name}_{q_type}_wind_bias_removed_rms_m_s"] = wind_centered_rms
                summary_rows[-1][f"{coeff_name}_{q_type}_wind_total_rmse_m_s"] = wind_total_rmse
                if r["segment_id"] == representative[axis] and coeff_name == "waveform_refit":
                    key = "wind_initial_q_m_s" if q_type == "initial_q_0.001" else "wind_reviewed_q_m_s"
                    for i in range(0, len(r["time"]), 5):
                        wave_rows.append({"segment_id": r["segment_id"], "axis": axis,
                            "time_s": float(r["time"][i]), "measured_angle_deg": float(r["angle_deg"][i]),
                            "old_model_angle_deg": float(np.rad2deg(model_old[i])),
                            "refit_model_angle_deg": float(np.rad2deg(model_fit[i])),
                            "score_window": bool(score[i]), "q_type": q_type,
                            "q_N_per_sample": q_value, "wind_estimate_m_s": float(wind[i]),
                            "wind_bias_removed_m_s": float(wind[i]-bias)})
    summary_path = OUT / "ow05_ball_free_decay_refit_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    cv_path = OUT / "ow05_ball_free_decay_refit_leave_one_out.csv"
    pd.DataFrame(cv_rows).to_csv(cv_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    waves_path = OUT / "ow05_ball_free_decay_refit_waveforms.csv"
    pd.DataFrame(wave_rows).to_csv(waves_path, index=False, encoding="utf-8-sig", float_format="%.7g")

    # Figure: measured/model angle overlays and the two refit-coefficient q outputs.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    colors = {"IN": "#2878a5", "OUT": "#d17a22"}
    for row, axis in enumerate(("IN", "OUT")):
        r = next(x for x in records if x["segment_id"] == representative[axis])
        m0 = integrate(r["angle_rad"][0], r["rate0"], len(r["time"]), DT, old_ratios[axis])
        m1 = integrate(r["angle_rad"][0], r["rate0"], len(r["time"]), DT, fits[axis])
        stride = 2
        ax = axes[row,0]
        ax.plot(r["time"][::stride], r["angle_deg"][::stride], color="#222", lw=1.0, label="Measured")
        ax.plot(r["time"][::stride], np.rad2deg(m0[::stride]), color="#888", ls="--", lw=.9, label="Current coefficients")
        ax.plot(r["time"][::stride], np.rad2deg(m1[::stride]), color=colors[axis], lw=.9, label="BALL waveform refit")
        ax.set_title(f"{axis}: {r['segment_id']} / free-decay angle")
        ax.set_ylabel("Angle [deg]"); ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=3)
        ax = axes[row,1]
        wave = pd.DataFrame(wave_rows)
        selected = wave[(wave.axis == axis)]
        for q_type, label, color in (("initial_q_0.001", "Initial q=0.001", "#888"),
                                     ("reviewed_smaller_q", f"Reviewed q={Q_REVIEWED[axis]:g}", colors[axis])):
            s = selected[selected.q_type == q_type]
            ax.plot(s.time_s, s.wind_bias_removed_m_s, lw=.9, color=color, label=label)
        ax.axhline(0, color="#222", lw=.7, label="True wind = 0")
        ax.set_title(f"{axis}: refit coefficients / RTS free-decay response")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Bias-removed estimate [m/s]")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[0,0].set_xlabel("Time [s]"); axes[1,0].set_xlabel("Time [s]")
    fig.suptitle("BALL free-decay waveform refit: angle model and RTS q comparison")
    fig_path = OUT / "ow05_ball_free_decay_refit.png"
    fig.savefig(fig_path, dpi=160); plt.close(fig)

    summary_frame = pd.DataFrame(summary_rows)
    axis_rollup = summary_frame.groupby("axis").agg(
        old_angle_rmse=("angle_rmse_old_model_deg", "mean"),
        refit_angle_rmse=("angle_rmse_refit_model_deg", "mean"),
    )
    coefficient_frame = pd.DataFrame(parameters)
    lines = ["# OW-05球あり自由振動の波形係数再フィット", "",
      "承認済みBALL自由振動12波形（IN/OUT各6本）を使い、ゼロ外力の非線形運動方程式を時系列波形へ直接再フィットした診断結果。既存の係数レジストリは変更しない。", "",
      "## 再フィット方法", "",
      "自由振動データから絶対慣性Iとトルク係数を同時に一意同定することはできないため、K/I、b/I、c/I、tau/Iを軸ごとに再フィットした。角度・初速の初期値は各波形の先頭0.2秒から求め、球ありの12波形をまとめて最小化した。フィット区間は計算負荷を抑えて50 Hzへ間引き、評価と描画は100 Hzで行った。再フィット時は従来固定していたb=0も候補に含めた。結果の物理係数は既存レジストリのIを固定基準として換算しており、実機の絶対Iを独立再測定した値ではない。", "",
      "## 角度波形の誤差", "",
      "| 軸 | 現行モデル平均RMSE [deg] | 波形再フィット平均RMSE [deg] | 残差比 |", "|---|---:|---:|---:|"]
    for axis in ("IN", "OUT"):
        old, new = axis_rollup.loc[axis]
        lines.append(f"| {axis} | {old:.3f} | {new:.3f} | {new/old:.3f}× |")
    cv_frame = pd.DataFrame(cv_rows)
    lines += ["", "同じ波形への過剰適合を確認するため、各軸で1波形を順番に除外し、残る5波形で再フィットして除外波形を計算した。", "",
      "| 軸 | 除外波形数 | 現行係数の平均RMSE [deg] | 除外波形再計算の平均RMSE [deg] | 比 |", "|---|---:|---:|---:|---:|"]
    for axis, g in cv_frame.groupby("axis"):
        old_cv = g.old_model_angle_rmse_deg.mean()
        new_cv = g.refit_model_angle_rmse_deg.mean()
        lines.append(f"| {axis} | {len(g)} | {old_cv:.3f} | {new_cv:.3f} | {new_cv/old_cv:.3f}× |")
    lines += ["", "![自由振動波形と再フィットモデル、およびq別RTS推定](ow05_ball_free_decay_refit.png)", "",
      "## 係数とq比較", "",
      "角度の前向き計算にはqは入りません。同じ再フィット係数の角度波形を用いて、RTS外力状態のqを初期値0.001 N/sampleと見直し値（IN 3e-5、OUT 3e-4 N/sample）に変えて、ゼロ風速推定を再計算した。図の推定波形は各採点区間の平均Biasを除去している。全12波形の指標はCSVに、正規化係数・既存I基準の係数換算値は係数CSV/JSONに記録した。", "",
      "### 正規化係数", "",
      "| 軸 | K/I [s⁻²] | b/I [s⁻¹] | c/I [1] | τ/I [s⁻²] |", "|---|---:|---:|---:|---:|"]
    for axis in ("IN", "OUT"):
        values = {r["parameter"]: r["waveform_refit"] for r in parameters if r["axis"] == axis}
        lines.append(f"| {axis} | {values['K_over_I']:.5f} | {values['b_over_I']:.5f} | {values['c_over_I']:.5f} | {values['tau_over_I']:.5f} |")
    lines += ["", "### Bias除去後の自由振動推定風速RMS", "",
      "| 軸 | 係数 | 初期q=0.001 | 見直し後q |", "|---|---|---:|---:|"]
    for axis in ("IN", "OUT"):
        g = summary_frame[summary_frame.axis == axis]
        for coeff_name, label in (("adopted", "現行係数"), ("waveform_refit", "再フィット係数")):
            old_q = g[f"{coeff_name}_initial_q_0.001_wind_bias_removed_rms_m_s"].mean()
            small_q = g[f"{coeff_name}_reviewed_smaller_q_wind_bias_removed_rms_m_s"].mean()
            lines.append(f"| {axis} | {label} | {old_q:.3f} m/s | {small_q:.3f} m/s |")
    lines += ["", "Bias除去後RMSは採点区間内の波形ごとの平均Biasを差し引いた変動量。真値外力は0であり、これは無風モデルへ実測角度を入力したときのRTS出力である。", "",
      "自由度を抑えるためIは従来値のまま固定し、b・c・tauをすべて自由な有効係数として扱った。特に非ゼロのbは、粘性摩擦の実在を立証した値ではなく、別の減衰・未モデル化効果も吸収し得る。", "",
      "これは同じ12波形を使った波形単位の交差検証であり、別日・別個体への汎化を証明するものではない。係数が自由振動に適合しても、周期的な残差が外部トルク、二軸結合、センサの周期誤差による可能性は別途残る。", "",
      "波形除外評価は `ow05_ball_free_decay_refit_leave_one_out.csv` に保存した。再生成: `python 06_Analysis/simulation/src/fit_ow05_ball_free_decay_coefficients.py`", ""]
    report_path = OUT / "OW05_BALL_FREE_DECAY_REFIT_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("refit normalized coeffs", {a: fits[a].tolist() for a in fits})
    print("angle RMSE mean", axis_rollup.to_dict(orient="index"))
    print("outputs", coeff_path.name, summary_path.name, waves_path.name, fig_path.name, report_path.name)


if __name__ == "__main__":
    main()
