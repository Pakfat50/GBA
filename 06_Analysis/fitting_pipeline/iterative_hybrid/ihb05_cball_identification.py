"""IHB-05: 球ありの自由振動データから球抗力係数 c_ball を同定する。

プログラムに詳しくない方でも流れを追えるよう、入力・計算・出力を関数に
分けてコメントしています。IHB-04をスキップした場合は、IHB-02で得た球なし
基準I/KとIHB-03の単調振幅一回積分法のτを固定入力として使います。

実行例（リポジトリのルートから）:
    python 06_Analysis/fitting_pipeline/iterative_hybrid/ihb05_cball_identification.py

結果は results/<日付>/iterative_hybrid/ihb05_cball/ に作られます。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Make the shared pipeline modules importable when this script is launched by path.
PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIRECTORY))

from fitting_tools import component_increments
from peak_to_peak_solver import solve_next_turning_point
from run_hybrid_rod_damping_identification import (
    AIR_DENSITY_KG_M3,
    AIR_DYNAMIC_VISCOSITY_PA_S,
    FRICTION_EPSILON_DEG_S,
    MAX_STEP_PERIOD_FRACTION,
    MAXIMUM_SEARCH_PERIODS,
    MINIMUM_AMPLITUDE_DEG,
    THEORETICAL_ROD_C,
    explicit_energy_basis,
)


DATE = "20260921"
SPHERE_DIAMETER_M = 0.100
SPHERE_RADIUS_M = SPHERE_DIAMETER_M / 2.0
SPHERE_AREA_M2 = math.pi * SPHERE_DIAMETER_M**2 / 4.0
SPHERE_CENTER_ARM_M = 0.174
SPHERE_CENTROID_INERTIA_KG_M2 = 4.0e-6
EXPOSED_ROD_UPPER_M = 0.129
FULL_ROD_UPPER_M = 0.229
ROD_C_BALL = THEORETICAL_ROD_C * (EXPOSED_ROD_UPPER_M / FULL_ROD_UPPER_M) ** 4
TAU_BY_AXIS_N_M = {"IN": 7.945731419193821e-5, "OUT": 1.769496760658e-4}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261002
AMPLITUDE_FIT_MIN_R = 0.99
AMPLITUDE_FIT_MAX_RMSE_DEG = 0.5


def fit_monotone_log_envelope(time_s: np.ndarray, signed_angle_rad: np.ndarray) -> np.ndarray:
    """IHB-03と同じ正値・単調減少の対数包絡線を頂点列に当てる。

    時刻を0〜1に正規化し、A(u) = A_end + B log((1+C)/(u+C))
    を最小二乗フィットする。A_end、B、Cを正に制約するため、振幅は
    常に正で、時間とともに単調減少する。戻り値は元の頂点符号を保つ。
    """
    time_s = np.asarray(time_s, dtype=float)
    signed_angle_rad = np.asarray(signed_angle_rad, dtype=float)
    amplitude = np.abs(signed_angle_rad)
    if len(time_s) < 5 or np.any(np.diff(time_s) <= 0) or np.any(amplitude <= 0):
        raise ValueError("単調振幅フィットには時刻順の正振幅頂点が5点以上必要です")
    u = (time_s - time_s[0]) / (time_s[-1] - time_s[0])

    def curve(parameters, x):
        a_end, b, c = parameters
        return a_end + b * np.log((1.0 + c) / (x + c))

    initial = [max(1e-8, float(amplitude[-1])),
               max(1e-8, float(amplitude[0] - amplitude[-1]) / 2), 0.8]
    fit = least_squares(
        lambda p: curve(p, u) - amplitude, initial,
        bounds=([1e-10, 1e-10, 1e-5], [np.inf, np.inf, 100.0]),
        x_scale="jac", max_nfev=50000,
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"単調振幅フィットに失敗しました: {fit.message}")
    return np.sign(signed_angle_rad) * curve(fit.x, u)


def fit_ball_peak_amplitudes(turning: pd.DataFrame, selection: pd.DataFrame):
    """承認済みBALL波形の頂点をフィットし、品質指標とフィット角を返す。

    頂点番号が飛んだ場合は別々の列としてフィットし、欠測をまたいで
    包絡線をつなぎません。品質ゲートの結果は記録し、未達をレポートで明示します。
    """
    approved = selection[
        (pd.to_numeric(selection.use_for_fitting, errors="coerce") == 1)
        & (selection.configuration.astype(str).str.upper() == "BALL")
    ]
    fitted_by_peak = {}
    diagnostics = []
    for segment_id in approved.segment_id.astype(str):
        peaks = turning[turning.segment_id.astype(str) == segment_id].sort_values("peak_number")
        starts = np.flatnonzero(pd.to_numeric(peaks.is_initial_peak, errors="coerce").to_numpy() == 1)
        if len(starts) != 1:
            raise ValueError(f"{segment_id}: 初期頂点が一つに定まりません")
        peaks = peaks.iloc[int(starts[0]):].copy().reset_index(drop=True)
        number = pd.to_numeric(peaks.peak_number).to_numpy(dtype=int)
        runs = np.split(np.arange(len(peaks)), np.where(np.diff(number) != 1)[0] + 1)
        raw_amplitudes, fitted_amplitudes = [], []
        for run in runs:
            if len(run) < 5:
                raise ValueError(f"{segment_id}: 連続頂点列が5点未満のためフィットできません")
            subset = peaks.iloc[run]
            raw = np.deg2rad(pd.to_numeric(subset.centered_peak_angle_deg).to_numpy(float))
            fitted = fit_monotone_log_envelope(pd.to_numeric(subset.peak_time_s).to_numpy(float), raw)
            for peak_number, value in zip(number[run], fitted):
                fitted_by_peak[(segment_id, int(peak_number))] = float(value)
            raw_amplitudes.extend(np.abs(np.rad2deg(raw)))
            fitted_amplitudes.extend(np.abs(np.rad2deg(fitted)))
        raw_a = np.asarray(raw_amplitudes)
        fit_a = np.asarray(fitted_amplitudes)
        corr = float(np.corrcoef(raw_a, fit_a)[0, 1]) if np.std(raw_a) and np.std(fit_a) else float("nan")
        rmse = float(np.sqrt(np.mean((fit_a - raw_a) ** 2)))
        diagnostics.append({
            "segment_id": segment_id, "peaks": len(raw_a), "amplitude_pearson_r": corr,
            "amplitude_rmse_deg": rmse, "amplitude_bias_deg": float(np.mean(fit_a - raw_a)),
            "quality_gate_pass": bool(np.isfinite(corr) and corr >= AMPLITUDE_FIT_MIN_R
                                       and rmse <= AMPLITUDE_FIT_MAX_RMSE_DEG),
        })
    return fitted_by_peak, diagnostics


def parse_args() -> argparse.Namespace:
    """Read optional paths and date from the command line."""
    script_dir = Path(__file__).resolve().parent
    default_results = script_dir.parent / "results"
    parser = argparse.ArgumentParser(description="IHB-05球抗力係数 c_ball 同定")
    parser.add_argument("--date", default=DATE, help="解析データ日付")
    parser.add_argument("--result-root", type=Path, default=default_results,
                        help="解析結果ルート（既定値はこのスクリプトと同じpipeline内）")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="出力先（省略時はiterative_hybrid/ihb05_cball）")
    return parser.parse_args()


def load_inputs(result_root: Path, date: str):
    """前処理済み頂点、承認表、物理マニフェスト、IHB-02基準I/Kを読む。"""
    day = result_root / date
    turning = pd.read_csv(day / "hybrid_identification/01_preprocessing/turning_points.csv")
    selection = pd.read_csv(day / "waveform_review/waveform_selection.csv", encoding="utf-8-sig")
    manifest = pd.read_csv(day / "resolved_manifest.csv", encoding="utf-8-sig")
    base_parameters = pd.read_csv(day / "iterative_hybrid/ihb02_base_parameters.csv")
    return turning, selection, manifest, base_parameters


def ball_physics(manifest: pd.DataFrame, base_parameters: pd.DataFrame) -> dict:
    """IHB-02の基準I/Kへ、球の実測質量・位置・重心回り慣性を加える。"""
    ball_rows = manifest[
        (manifest.configuration.astype(str).str.upper() == "BALL")
        & (pd.to_numeric(manifest.valid, errors="coerce") == 1)
    ]
    if ball_rows.empty:
        raise ValueError("resolved_manifest.csvに有効なBALL物理入力がありません")
    # IN/OUTは同じ球・同じ作用位置を共有するため、マニフェストの1行を使う。
    delta_i, delta_k = component_increments(ball_rows.iloc[0].to_dict())
    bases = base_parameters.set_index("axis")
    result = {}
    for axis in ("IN", "OUT"):
        base = bases.loc[axis]
        result[axis] = {
            "inertia_kg_m2": float(base.base_inertia_kg_m2) + delta_i,
            "restoring_n_m": float(base.base_restoring_n_m) + delta_k,
            "base_inertia_kg_m2": float(base.base_inertia_kg_m2),
            "base_restoring_n_m": float(base.base_restoring_n_m),
            "delta_inertia_kg_m2": delta_i,
            "delta_restoring_n_m": delta_k,
        }
    return result


def make_intervals(turning: pd.DataFrame, selection: pd.DataFrame, physics: dict,
                   fitted_by_peak: dict) -> list[dict]:
    """承認済みBALL波形の隣り合う適格頂点を半周期データにする。"""
    approved = selection[
        (pd.to_numeric(selection.use_for_fitting, errors="coerce") == 1)
        & (selection.configuration.astype(str).str.upper() == "BALL")
    ]
    allowed = set(approved.segment_id.astype(str))
    selected = turning[turning.segment_id.astype(str).isin(allowed)]
    intervals: list[dict] = []
    for segment_id, original in selected.groupby("segment_id", sort=False):
        peaks = original.sort_values("peak_number").reset_index(drop=True)
        starts = np.flatnonzero(pd.to_numeric(peaks.is_initial_peak).to_numpy() == 1)
        if len(starts) != 1:
            raise ValueError(f"{segment_id}: 初期頂点が一つに定まりません")
        peaks = peaks.iloc[int(starts[0]):].reset_index(drop=True)
        axis = str(peaks.loc[0, "axis"]).upper()
        info = physics[axis]
        kept = 0
        for i in range(len(peaks) - 1):
            first, last = peaks.iloc[i], peaks.iloc[i + 1]
            # 初期頂点を取り除いた後も、品質選別済みの連続頂点だけを使う。
            if int(first.eligible_for_later_stages) != 1 or int(last.eligible_for_later_stages) != 1:
                continue
            if min(float(first.amplitude_deg), float(last.amplitude_deg)) < MINIMUM_AMPLITUDE_DEG:
                continue
            raw_a = float(first.centered_peak_angle_deg)
            raw_z = float(last.centered_peak_angle_deg)
            a = float(np.rad2deg(fitted_by_peak[(str(segment_id), int(first.peak_number))]))
            z = float(np.rad2deg(fitted_by_peak[(str(segment_id), int(last.peak_number))]))
            if a * z >= 0:
                raise ValueError(f"{segment_id}: 頂点の正負が交互になっていません")
            kept += 1
            intervals.append({
                "interval_id": f"{segment_id}_H{kept:03d}",
                "segment_id": str(segment_id), "axis": axis,
                "configuration": "BALL", "direction": str(peaks.loc[0, "direction"]),
                "interval_number": kept,
                "start_peak_number": int(first.peak_number),
                "end_peak_number": int(last.peak_number),
                "start_time_s": float(first.peak_time_s), "end_time_s": float(last.peak_time_s),
                "start_angle_rad": float(np.deg2rad(a)),
                "measured_next_angle_rad": float(np.deg2rad(z)),
                "start_amplitude_deg": abs(a), "end_amplitude_deg": abs(z),
                "raw_start_angle_deg": raw_a, "raw_next_angle_deg": raw_z,
                "fitted_start_angle_deg": a, "fitted_next_angle_deg": z,
                "inertia_kg_m2": info["inertia_kg_m2"],
                "restoring_n_m": info["restoring_n_m"],
            })
    if not intervals:
        raise ValueError("同定に使える承認済みBALL半周期がありません")
    counts = pd.Series([row["segment_id"] for row in intervals]).value_counts().to_dict()
    for row in intervals:
        row["waveform_interval_count"] = int(counts[row["segment_id"]])
    return intervals


def fit_one_shared_c(intervals: list[dict]) -> tuple[float, dict[str, tuple[float, float]]]:
    """波形ごとの総重みを等しくして、全軸共通の非負c_ballを一度だけ解く。"""
    grouped: dict[str, list[dict]] = {}
    for row in intervals:
        grouped.setdefault(row["segment_id"], []).append(row)
    contributions: dict[str, tuple[float, float]] = {}
    for name, rows in grouped.items():
        num = den = 0.0
        for row in rows:
            # energy basis: [粘性, 二乗抗力, クーロン摩擦] の各散逸積分。
            basis = explicit_energy_basis(row["start_angle_rad"], row["inertia_kg_m2"], row["restoring_n_m"])
            k = row["restoring_n_m"]
            measured_loss = k * (math.cos(abs(row["measured_next_angle_rad"])) - math.cos(abs(row["start_angle_rad"])))
            # クーロン摩擦の仕事は角度移動量そのものに等しい。
            # したがって実測された両端振幅の和を使い、2×始点振幅とは置かない。
            tau_travel = abs(row["start_angle_rad"]) + abs(row["measured_next_angle_rad"])
            target = measured_loss - TAU_BY_AXIS_N_M[row["axis"]] * tau_travel - ROD_C_BALL * basis[1]
            weight = 1.0 / len(rows)
            num += weight * basis[1] * target
            den += weight * basis[1] ** 2
            row["quadratic_drag_basis"] = float(basis[1])
            row["observed_loss_j"] = measured_loss
            row["tau_basis"] = tau_travel
            row["rod_basis"] = float(basis[1])
            row["sphere_target_j"] = target
            row["interval_weight"] = weight
        contributions[name] = (num, den)
    c_ball = max(0.0, sum(n for n, _ in contributions.values()) / sum(d for _, d in contributions.values()))
    return c_ball, contributions


def predict_next(row: dict, c_ball: float) -> float:
    """半周期ODEを解き、次の折返し角を予測する。"""
    eps = np.deg2rad(FRICTION_EPSILON_DEG_S)
    result = solve_next_turning_point(
        row["start_angle_rad"], row["inertia_kg_m2"], row["restoring_n_m"],
        0.0, ROD_C_BALL + c_ball, TAU_BY_AXIS_N_M[row["axis"]], eps,
        max_step_fraction=MAX_STEP_PERIOD_FRACTION,
        max_periods=MAXIMUM_SEARCH_PERIODS,
    )
    return float(result["next_angle_rad"])


def weighted_energy_metrics(rows: list[dict], c_ball: float) -> list[dict]:
    """エネルギー損失のR、RMSE、R²を波形等重みで計算する。"""
    table = pd.DataFrame(rows)
    table["predicted_loss_j"] = (
        ROD_C_BALL * table.rod_basis
        + table["sphere_coulomb_loss_j"]
        + c_ball * table.quadratic_drag_basis
    )
    outputs = []
    for label, data in [("ALL", table), ("IN", table[table.axis == "IN"]), ("OUT", table[table.axis == "OUT"])]:
        if data.empty:
            continue
        # 각 파형의 전체重みを1にそろえ、区間数の差をならす。
        w = 1.0 / data.groupby("segment_id").interval_id.transform("count").to_numpy(dtype=float)
        w = w / w.sum()
        obs = data.observed_loss_j.to_numpy(dtype=float)
        pred = data.predicted_loss_j.to_numpy(dtype=float)
        mean = float(np.sum(w * obs))
        ss_res = float(np.sum(w * (obs - pred) ** 2))
        ss_tot = float(np.sum(w * (obs - mean) ** 2))
        obs_mean = float(np.sum(w * obs))
        pred_mean = float(np.sum(w * pred))
        covariance = float(np.sum(w * (obs - obs_mean) * (pred - pred_mean)))
        obs_sd = math.sqrt(float(np.sum(w * (obs - obs_mean) ** 2)))
        pred_sd = math.sqrt(float(np.sum(w * (pred - pred_mean) ** 2)))
        corr = covariance / (obs_sd * pred_sd) if obs_sd > 0 and pred_sd > 0 else float("nan")
        outputs.append({
            "scope": label, "intervals": len(data), "waveforms": int(data.segment_id.nunique()),
            "pearson_r": corr, "rmse_mj": math.sqrt(ss_res) * 1000.0,
            "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        })
    return outputs


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    output = args.output_dir or root / args.date / "iterative_hybrid/ihb05_cball"
    output.mkdir(parents=True, exist_ok=True)

    turning, selection, manifest, bases = load_inputs(root, args.date)
    physics = ball_physics(manifest, bases)
    fitted_by_peak, amplitude_fit_metrics = fit_ball_peak_amplitudes(turning, selection)
    intervals = make_intervals(turning, selection, physics, fitted_by_peak)
    # 旧版（生の計測角）を感度比較用に再計算する。主結果は単調関数角による値。
    raw_intervals = []
    for row in intervals:
        raw = dict(row)
        raw_start = float(np.deg2rad(row["raw_start_angle_deg"]))
        raw_next = float(np.deg2rad(row["raw_next_angle_deg"]))
        raw.update(
            start_angle_rad=raw_start, measured_next_angle_rad=raw_next,
            start_amplitude_deg=abs(row["raw_start_angle_deg"]),
            end_amplitude_deg=abs(row["raw_next_angle_deg"]),
        )
        raw_intervals.append(raw)
    c_ball_raw, _ = fit_one_shared_c(raw_intervals)
    for row in raw_intervals:
        row["sphere_coulomb_loss_j"] = TAU_BY_AXIS_N_M[row["axis"]] * row["tau_basis"]
    raw_energy_metrics = weighted_energy_metrics(raw_intervals, c_ball_raw)
    c_ball, wave_contributions = fit_one_shared_c(intervals)

    # 全半周期を一つずつ数値積分し、実測次頂点との角度誤差を調べる。
    for row in intervals:
        row["sphere_coulomb_loss_j"] = TAU_BY_AXIS_N_M[row["axis"]] * row["tau_basis"]
        row["predicted_next_angle_rad"] = predict_next(row, c_ball)
        row["predicted_next_angle_deg"] = float(np.rad2deg(row["predicted_next_angle_rad"]))
        row["measured_next_angle_deg"] = float(np.rad2deg(row["measured_next_angle_rad"]))
        row["angle_residual_deg"] = row["predicted_next_angle_deg"] - row["measured_next_angle_deg"]
        row["raw_angle_residual_deg"] = row["predicted_next_angle_deg"] - row["raw_next_angle_deg"]
        row["predicted_loss_j"] = ROD_C_BALL * row["rod_basis"] + row["sphere_coulomb_loss_j"] + c_ball * row["quadratic_drag_basis"]

    # 波形ごとの寄与とRMSEを記録し、波形単位の再現性を確認する。
    prediction_table = pd.DataFrame(intervals)
    waveform_rows = []
    for name, group in prediction_table.groupby("segment_id", sort=False):
        subset = group
        num, den = wave_contributions[name]
        c_wave = max(0.0, num / den) if den > 0 else 0.0
        waveform_rows.append({
            "segment_id": name, "axis": str(subset.axis.iloc[0]),
            "intervals": len(subset), "independent_waveform_c_ball": c_wave,
            "endpoint_angle_rmse_deg": float(np.sqrt(np.mean(subset.angle_residual_deg**2))),
        })
    waveform_table = pd.DataFrame(waveform_rows)

    # 波形を単位にブートストラップし、同じ波形内の隣接半周期を独立扱いしない。
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ids = list(wave_contributions)
    bootstrap = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(ids), size=len(ids))
        numerator = sum(wave_contributions[ids[i]][0] for i in sample)
        denominator = sum(wave_contributions[ids[i]][1] for i in sample)
        bootstrap.append(max(0.0, numerator / denominator) if denominator > 0 else 0.0)
    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])

    reynolds_rows = []
    for row in intervals:
        omega = math.sqrt(2.0 * row["restoring_n_m"] * (1.0 - math.cos(row["start_angle_rad"])) / row["inertia_kg_m2"])
        speed = SPHERE_CENTER_ARM_M * omega
        reynolds_rows.append({
            "interval_id": row["interval_id"], "segment_id": row["segment_id"], "axis": row["axis"],
            "start_amplitude_deg": row["start_amplitude_deg"], "peak_speed_m_s": speed,
            "reynolds_number_at_peak": AIR_DENSITY_KG_M3 * speed * SPHERE_DIAMETER_M / AIR_DYNAMIC_VISCOSITY_PA_S,
        })
    reynolds = pd.DataFrame(reynolds_rows)
    energy_metrics = weighted_energy_metrics(intervals, c_ball)
    residual_amplitude_correlation = {
        axis: float(group.start_amplitude_deg.corr(group.angle_residual_deg))
        for axis, group in prediction_table.groupby("axis")
    }
    residual_amplitude_correlation["ALL"] = float(
        prediction_table.start_amplitude_deg.corr(prediction_table.angle_residual_deg)
    )
    pd.DataFrame(energy_metrics).to_csv(output / "energy_fit_metrics.csv", index=False)
    pd.DataFrame(raw_energy_metrics).to_csv(output / "raw_peak_sensitivity_energy_fit_metrics.csv", index=False)
    pd.DataFrame(amplitude_fit_metrics).to_csv(output / "amplitude_fit_metrics.csv", index=False)
    waveform_table.to_csv(output / "waveform_estimates.csv", index=False, float_format="%.10g")
    prediction_table.to_csv(output / "interval_predictions.csv", index=False, float_format="%.10g")
    reynolds.to_csv(output / "reynolds_assessment.csv", index=False, float_format="%.10g")

    cd = 2.0 * c_ball / (AIR_DENSITY_KG_M3 * SPHERE_AREA_M2 * SPHERE_CENTER_ARM_M**3)
    cd_ci = [2.0 * ci_low / (AIR_DENSITY_KG_M3 * SPHERE_AREA_M2 * SPHERE_CENTER_ARM_M**3),
             2.0 * ci_high / (AIR_DENSITY_KG_M3 * SPHERE_AREA_M2 * SPHERE_CENTER_ARM_M**3)]
    angle_rmse = float(np.sqrt(np.mean(prediction_table.angle_residual_deg**2)))
    raw_angle_rmse = float(np.sqrt(np.mean(prediction_table.raw_angle_residual_deg**2)))
    settings = {
        "stage": "IHB-05", "date": args.date,
        "interpretation": "IHB-04 skipped: uses IHB-02 base I/K plus measured sphere increments, IHB-03 monotone fitted A peaks, and IHB-03 one-pass tau",
        "c_ball_n_m_s2_per_rad2": c_ball, "c_ball_raw_peak_sensitivity_n_m_s2_per_rad2": c_ball_raw,
        "c_ball_waveform_bootstrap_95pct": [float(ci_low), float(ci_high)],
        "equivalent_Cd": cd, "equivalent_Cd_95pct": cd_ci,
        "waveforms": len(ids), "half_cycles": len(intervals),
        "half_cycle_ode_next_peak_angle_rmse_deg": angle_rmse,
        "half_cycle_ode_next_peak_angle_rmse_vs_raw_deg": raw_angle_rmse,
        "amplitude_fit_quality_gate": {"pearson_r_min": AMPLITUDE_FIT_MIN_R,
                                       "amplitude_rmse_deg_max": AMPLITUDE_FIT_MAX_RMSE_DEG},
        "amplitude_fit_gate_failures": [r["segment_id"] for r in amplitude_fit_metrics
                                        if not r["quality_gate_pass"]],
        "tau_by_axis_n_m": TAU_BY_AXIS_N_M, "b_by_axis_n_m_s_per_rad": {"IN": 0.0, "OUT": 0.0},
        "c_rod_no_ball": THEORETICAL_ROD_C, "c_rod_ball_exposed_length_adjusted": ROD_C_BALL,
        "sphere_diameter_m": SPHERE_DIAMETER_M, "sphere_center_arm_m": SPHERE_CENTER_ARM_M,
        "bootstrap_method": f"{BOOTSTRAP_REPLICATES} waveform-cluster resamples; seed {BOOTSTRAP_SEED}",
        "energy_fit_metrics": energy_metrics,
        "raw_peak_sensitivity_energy_fit_metrics": raw_energy_metrics,
        "endpoint_residual_vs_start_amplitude_pearson_r": residual_amplitude_correlation,
    }
    (output / "ihb05_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 図1は実測次頂点とODE予測の一致、図2は残差と開始振幅の関係を示す。
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    colors = {"IN": "#1368aa", "OUT": "#db5f02"}
    for axis, part in prediction_table.groupby("axis"):
        axes[0].scatter(part.measured_next_angle_deg, part.predicted_next_angle_deg,
                        s=14, alpha=0.58, label=axis, color=colors[axis])
        axes[1].scatter(part.start_amplitude_deg, part.angle_residual_deg,
                        s=14, alpha=0.58, label=axis, color=colors[axis])
    low = float(min(prediction_table.measured_next_angle_deg.min(), prediction_table.predicted_next_angle_deg.min()))
    high = float(max(prediction_table.measured_next_angle_deg.max(), prediction_table.predicted_next_angle_deg.max()))
    axes[0].plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    axes[0].set(title="Monotone-fit vs ODE next peak", xlabel="Monotone-fit next angle [deg]", ylabel="ODE predicted next angle [deg]")
    axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1].set(title="ODE endpoint residual", xlabel="Starting peak amplitude [deg]", ylabel="Predicted - measured [deg]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(f"IHB-05 sphere drag fit: c_ball={c_ball:.4g}; RMSE={angle_rmse:.3f} deg")
    fig.savefig(output / "ihb05_cball_validation.svg")
    fig.savefig(output / "ihb05_cball_validation.png", dpi=180)
    plt.close(fig)
    # Matplotlib's SVG path data may end lines with spaces; remove them for a clean diff.
    svg_path = output / "ihb05_cball_validation.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")

    r_all = next(m["pearson_r"] for m in energy_metrics if m["scope"] == "ALL")
    rmse_all = next(m["rmse_mj"] for m in energy_metrics if m["scope"] == "ALL")
    r2_all = next(m["r_squared"] for m in energy_metrics if m["scope"] == "ALL")
    r_raw = next(m["pearson_r"] for m in raw_energy_metrics if m["scope"] == "ALL")
    rmse_raw = next(m["rmse_mj"] for m in raw_energy_metrics if m["scope"] == "ALL")
    r2_raw = next(m["r_squared"] for m in raw_energy_metrics if m["scope"] == "ALL")
    failed_amplitude_fits = [r for r in amplitude_fit_metrics if not r["quality_gate_pass"]]
    failed_fit_summary = (
        "、".join(f"{r['segment_id']}（R={r['amplitude_pearson_r']:.5f}, RMSE={r['amplitude_rmse_deg']:.4f}°）"
                  for r in failed_amplitude_fits)
        if failed_amplitude_fits else "なし"
    )
    # Keep LaTeX outside the f-string parser so the backslashes stay intact.
    equation_i_k = r"I_{\mathrm{BALL}}=I_0+J_{G,\mathrm{ball}}+m_{\mathrm{ball}}\ell_{\mathrm{ball}}^2,\quad K_{\mathrm{BALL}}=K_0+m_{\mathrm{ball}}g \ell_{\mathrm{signed}}."
    report = rf"""# IHB-05: 球抗力係数 c_ball の同定

## 結論

IHB-04をスキップする指示に合わせ、IHB-02の球なし基準 I/K に実測球質量・形状から計算した増分を加え、IHB-03で採用した単調減少Aフィットと一回積分法の考え方をIHB-05の球あり頂点列にも適用した。単調フィット後の頂点角をエネルギー損失計算に用い、軸別τ等を固定して共通の非負 c_ball を一回の重み付き最小二乗で求めた。

単調関数Aを使った暫定推定値は **c_ball = {c_ball:.8g} N·m·s²/rad²**。生の計測Aを使った旧方式の感度比較値は **{c_ball_raw:.8g} N·m·s²/rad²** である。単調フィットを使った半周期ODEの次頂点角RMSEは、フィット頂点との比較で **{angle_rmse:.4f}°**、生の計測頂点との比較で **{raw_angle_rmse:.4f}°**（{len(intervals)}区間）だった。単調フィットAから作ったエネルギー損失はPearson R **{r_all:.5f}**、RMSE **{rmse_all:.5f} mJ**、R² **{r2_all:.5f}**（生A方式: R {r_raw:.5f}、RMSE {rmse_raw:.5f} mJ、R² {r2_raw:.5f}）となった。

ただし、IHB-03と同じ品質基準（AフィットR ≥ 0.99、振幅RMSE ≤ 0.5°）を満たさない波形が **{len(failed_amplitude_fits)}/{len(amplitude_fit_metrics)}本**あった（{failed_fit_summary}）。従って上記は全波形でゲートを通過した確定値ではなく、A平滑化の影響を調べる暫定感度解析値として扱う。

この値はIHB-04の更新結果を用いた値ではない。IHB-04をスキップしたため、IHB-02/03の係数に依存する暫定同定値として記録する。

## 固定入力と対象

- 対象: 承認済み球あり波形 {len(ids)}本、{len(intervals)}半周期（IN {sum(1 for w in waveform_rows if w['axis']=='IN')}本、OUT {sum(1 for w in waveform_rows if w['axis']=='OUT')}本）。
- 各軸の球あり I/K: IHB-02の I₀/K₀ に、`resolved_manifest.csv` の球質量・重心位置・重心回り慣性から求めたΔI/ΔKを加算。
- τ: IHB-03単調振幅一回積分法。IN {TAU_BY_AXIS_N_M['IN']:.8g} N·m、OUT {TAU_BY_AXIS_N_M['OUT']:.8g} N·m。
- b: 0 N·m·s/rad。
- 球で覆われない上側ロッド長129 mmを反映し、ロッド抗力は長さの4乗比 `(129/229)^4` で理論値を縮小。
- 初期リリースから最初の適格頂点までを除外し、隣接頂点の両方が4°以上の半周期を使用。波形ごとの総重みは等しい。

球あり係数の展開は、球なしIHB-02基準値と球の物理増分から次のように定めた。

$$
{equation_i_k}
$$

## 同定方法：実測エネルギー損失から球抗力を分ける

### なぜIHB-03と同じくAを単調関数で近似するか

IHB-03では、生の頂点振幅の小さな局所変動でも、隣接頂点のエネルギーを引き算して作る半周期損失に無視できない変化が出ることを確認した。エネルギー損失は各頂点のエネルギーそのものより小さいため、頂点Aのわずかな揺れが差分の相対誤差を大きくする。IHB-05も同じ隣接頂点差から観測損失を作り、さらにその損失を c_ball 同定に使うので、同じ理由で単調減少包絡線に置き換えるのが妥当である。

ここではIHB-03最終採用の関数をそのまま用いる。

~~~math
A(u)=A_{{\mathrm{{end}}}}+B\ln\!\left(\frac{{1+C}}{{u+C}}\right),
\qquad A_{{\mathrm{{end}}}}>0,\quad B>0,\quad C>0.
~~~

各波形の連続した頂点列に対して当てはめ、符号は計測頂点の交互符号を保つ。フィット頂点と生の頂点の振幅R/RMSEを波形ごとに記録する。IHB-03と同じ判定基準を当てた結果、未達波形は {len(failed_amplitude_fits)}本で、該当は {failed_fit_summary}。未達を除外したり閾値を緩めたりせず、今回の c_ball は全波形での厳密な品質ゲート合格値とは区別する。

IHB-03が示すように、平滑後の同じA列から観測損失とモデル基底の両方を作り直した適合指標は、平滑後の列への同一データ内適合である。測定誤差を独立に推定した結果や、別データへの予測精度ではない。そのため、ここでは c_ball の生A感度比較値と、ODE予測をフィット頂点・生頂点の双方に比較した角度RMSEも併記する。

ここでは、球抗力係数をいきなりODEの波形合わせで探すのではなく、まず頂点から頂点までに失われたエネルギーを使う。頂点では振り子が一瞬止まるため、角速度はほぼゼロである。その瞬間の力学的エネルギーは位置エネルギーだけになり、平衡点からの角度 A [rad] を使って E=K(1−cos|A|) と計算できる。隣り合う頂点のエネルギー差が、実測半周期損失になる。以下の主計算の A_n は、生角ではなくIHB-03と同じ単調関数でフィットした頂点角である。

~~~math
E_n=K_j\left(1-\cos|A_n|\right),\qquad
\Delta E_{{\mathrm{{obs}},n}}=E_n-E_{{n+1}}
=K_j\left[\cos|A_{{n+1}}|-\cos|A_n|\right].
~~~

ここで j は IN または OUT の軸、A_n は単調関数フィット後の半周期始点角、A_{{n+1}} は次のフィット頂点角である。角度はこの計算の前にラジアンへ変換する。頂点列は品質確認済みの前処理結果を使い、リリース直後の区間を除外する。さらにフィット後の始点・終点振幅が両方4°以上で、隣り合う頂点が連続している半周期だけを残した。

## 係数を固定する理由と、抗力仕事の式

球抗力以外の寄与が球係数に混ざらないよう、同定前に I、K、τ、b、ロッド抗力を決めて固定する。

- **I と K**: IHB-02の球なし基準値に、球の物理量による増分を加えた値を使う。球の質量を m、支点から球重心までの距離を ℓ、球重心まわりの慣性を J_G とすると、増分は ΔI=J_G+mℓ²、ΔK=mgℓ_signed である。I は回りにくさ、K は重力が元の位置へ戻そうとする強さを表す。球半径と支点からの距離を混同しないよう、ℓ は支点から球重心までの腕の長さを表す。
- **τ**: IHB-03で選んだ単調振幅一回積分法の軸別値をそのまま使う。
- **b**: 0 に固定する。
- **ロッド抗力**: 球に覆われない長さ129 mmのみが空気に露出するとして、球なし時の理論ロッド係数に (129/229)^4 を掛ける。

クーロン摩擦の仕事は速度の大きさではなく、摩擦トルクと移動角の積で決まる。半周期中は角度が一方向に進むので、実測された始点・終点振幅の和 R_n=|A_n|+|A_{{n+1}}| が移動角であり、摩擦が失わせるエネルギーは τ_j R_n となる。従って、半周期のエネルギー収支は次の形になる。

~~~math
R_n=|A_n|+|A_{{n+1}}|,\qquad
y_n=\Delta E_{{\mathrm{{obs}},n}}-\tau_jR_n-c_{{\mathrm{{rod,BALL}}}}C_n
\approx c_{{\mathrm{{ball}}}}C_n.
~~~

左辺 y_n は「実測された損失から、既知とみなしたクーロン摩擦とロッド抗力の損失を差し引いた残り」である。右辺は球の二乗抗力が説明する損失である。C_n は半周期の二乗抗力基底で、C_n=∫|θ̇|³dt に相当する。抗力トルクが速度の二乗に比例するため、抗力がする仕事を時間で積分すると速度の絶対値の3乗が現れる。

速度を計測してこの積分を直接行う代わりに、保存振り子（摩擦・抗力がないと仮定した振り子）の軌道から一度だけ計算する。始点振幅を a=|A_n|、ω₀=√(K_j/I_j) とおく。保存エネルギーから各角度での速度を求め、それを角度について積分すると、次の閉じた式になる。

保存軌道では、始点振幅 a における位置エネルギーと、途中の角度 θ における位置・運動エネルギーが等しい。従って角速度は次式で表される。

~~~math
\frac{{1}}{{2}}I_j\dot{{\theta}}^{{2}}+K_j(1-\cos\theta)
=K_j(1-\cos a),\qquad
\dot{{\theta}}^{{2}}=2\omega_0^2(\cos\theta-\cos a),\quad
\omega_0^2=\frac{{K_j}}{{I_j}}.
~~~

一半周期では角度が a から −a まで一方向に進む。dt=dθ/θ̇ を使うと、積分する量は θ̇² dθ になり、上式を −a から a まで積分して次を得る。

~~~math
C_n=\int|\dot{{\theta}}|^3dt
=2\omega_0^2\int_{{-a}}^{{a}}(\cos\theta-\cos a)\,d\theta
=4\omega_0^2\left[\sin(a)-a\cos(a)\right].
~~~

この式の C_n は球だけでなく、同じ回転運動に対する二乗抗力なら同じ形の基底として使える。したがって球抗力と露出ロッド抗力は、ともに係数×C_n の形で表せる。球とロッドの抗力係数を分けるため、固定したロッド寄与 c_{{rod,BALL}}C_n を先に引く。保存軌道による C_n は減衰した実軌道を厳密に再現する積分値ではないので、ここは近似である。最後に行うODE計算は、この近似を含む同定値が頂点予測でも妥当かを確かめる独立チェックである。

## 重み付き最小二乗で c_ball を一度求める

一つの半周期だけを見ると、測定誤差や頂点検出誤差の影響が大きい。そこで全12波形・379半周期のデータをまとめ、全波形に共通する c_ball を求める。残差とは、予測損失 c_ball C_n と差し引き後の損失 y_n の差である。c_ball は0以上とし、自由な切片（基底 C では説明されない一定損失）は加えない。

波形ごとの区間数は同じではない。全区間を同じ重みにすると、半周期数の多い長い記録が結果を強く左右する。そのため、まず各波形の中では区間数 N_w で割り、波形ごとの重み合計を1にする。そのうえで波形ごとの寄与を平均し、12波形を等しく扱う。重みを a_{{w,n}}=1/(W N_w) とすると、誤差二乗和を最小にする解は次の一回の割り算で計算できる。

~~~math
\widehat c_{{\mathrm{{ball}}}}
=\max\!\left(0,
\frac{{\displaystyle\sum_{{w=1}}^{{W}}\sum_{{n=1}}^{{N_w}}
a_{{w,n}}C_{{w,n}}y_{{w,n}}}}
{{\displaystyle\sum_{{w=1}}^{{W}}\sum_{{n=1}}^{{N_w}}
a_{{w,n}}C_{{w,n}}^{{2}}}}\right),
\qquad a_{{w,n}}=\frac{{1}}{{W N_w}}.
~~~

この式は、傾き c を持つ直線 y≈cC を原点を通るように当てはめたときの最小二乗解である。実際の計算では波形ごとに分子 ΣaCy と分母 ΣaC² を先に足し、最後に全波形分を合計して c_ball を得る。非負制約は、抗力係数が負にならないという物理条件を反映する。軸ごとに異なる τ を差し引くが、球が共通部品なので c_ball 自体は IN/OUT 共通の一つとする。ここでの係数決定は「各半周期の実測エネルギー損失を説明する」一回の計算であり、ODEを何度も積分して係数を探す反復最適化ではない。

## エネルギー損失の適合度の読み方

係数を決めた後、各区間について一回積分法の予測損失を計算する。τとロッド抗力は固定し、推定した球抗力の寄与を足し戻す。

~~~math
\Delta E_{{\mathrm{{model}},n}}
=\tau_jR_n+\left(c_{{\mathrm{{rod,BALL}}}}+\widehat c_{{\mathrm{{ball}}}}\right)C_n.
~~~

観測値と予測値がどれだけ合うかを、全波形の総重みが等しくなるようにして評価する。重みの合計を1に正規化しているので、重み付きRMSEは次式である。

~~~math
\mathrm{{RMSE}}_E
=\sqrt{{\sum_{{w,n}}a_{{w,n}}
\left(\Delta E_{{\mathrm{{obs}},w,n}}-\Delta E_{{\mathrm{{model}},w,n}}\right)^2}}.
~~~

Pearson R は、損失の大きい区間で予測損失も大きいという増減のそろい具合を測る相関である。R²は、予測誤差の二乗和を「全区間へ観測損失の加重平均を予測した場合」の二乗和と比べる。1なら予測誤差がゼロ、0なら平均値だけの予測と同程度、負なら平均値だけの予測より悪い。

~~~math
R^2=1-
\frac{{\sum_{{w,n}}a_{{w,n}}
\left(\Delta E_{{\mathrm{{obs}},w,n}}-\Delta E_{{\mathrm{{model}},w,n}}\right)^2}}
{{\sum_{{w,n}}a_{{w,n}}
\left(\Delta E_{{\mathrm{{obs}},w,n}}-\overline{{\Delta E}}_{{\mathrm{{obs}}}}\right)^2}}.
~~~

R、RMSE、R²は違う側面を見るため、一つだけで適合の良し悪しを決めない。特にRが高くても、予測値に一定の偏りがあればRMSEは残りうる。

得られた値は c_ball=CBALL_RESULT_TOKEN N·m·s²/rad² である。球抗力を慣例的な無次元抗力係数 Cd に換算する場合、球の速度を ℓ_ball θ̇ とし、抗力を 1/2 ρ Cd A v² と置くと、トルク係数との関係は次の通りである。ここで ℓ_ball=0.174 m は支点から球重心までの距離、球半径はD/2=0.050 mである。

~~~math
c_{{\mathrm{{ball}}}}=\frac{{1}}{{2}}\rho C_D A \ell_{{\mathrm{{ball}}}}^3,\qquad
C_D=\frac{{2c_{{\mathrm{{ball}}}}}}{{\rho A \ell_{{\mathrm{{ball}}}}^3}},\qquad
A=\frac{{\pi D^2}}{{4}}.
~~~

本解析では直径 D=0.100 m、支点から球重心まで ℓ_ball=0.174 m とし、設定ファイルに記録した空気密度・粘度を用いた換算 Cd は0.5686となった。これはこの振り子・速度域・固定した他係数のもとでの等価値である。

## 不確かさと独立検証

95%区間は、379個の半周期を独立と見なすのではなく、12本の波形を単位に復元抽出するクラスタ・ブートストラップで求めた。各反復で波形を12本、重複を許して引き直し、その波形に属する全半周期をまとめて同じ最小二乗計算を行う。これを2,000回行い、推定値の分布の2.5百分位と97.5百分位を区間とした。この区間が表すのは、観測された波形間のばらつきである。I/K/τ、質量・寸法、空気密度、ロッド長近似などの不確かさは含まない。

最後に、求めた c_ball を固定して検証済みの半周期ODEソルバーを全379区間に適用する。これは、保存軌道近似ではなく減衰を含む運動方程式を各区間で数値積分して、次の折返し角を予測する確認である。

~~~math
I_j\ddot{{\theta}}+K_j\sin\theta
+\left(c_{{\mathrm{{rod,BALL}}}}+\widehat c_{{\mathrm{{ball}}}}\right)
|\dot{{\theta}}|\dot{{\theta}}
+\tau_j\tanh\left(\frac{{\dot{{\theta}}}}{{\varepsilon}}\right)=0,
\qquad b=0.
~~~

ε は、速度ゼロ付近で符号関数が急に切り替わる計算を滑らかにする小さな速度幅で、実装では0.5°/sをラジアン毎秒へ変換している。各区間は実測始点角・角速度ゼロから独立に開始する。ここでは c_ball を再調整しないので、係数を合わせたデータと同じデータではあるものの、ODEという別の計算経路による検算となる。次頂点角RMSEは、この数値計算の折返し角と実測次頂点角との差から求める。エネルギー損失のRMSE/R²は上記の一回積分予測式に対する指標であり、ODE頂点角RMSEとは比較対象が異なる。

この方法は「τとロッド損失を正しいものとして固定したとき、残るエネルギー損失を説明する球係数」を推定する。固定入力に誤差があれば、その一部を c_ball が引き受ける可能性がある。またODE残差には振幅依存の傾向が残った。従って、本値を確定した普遍定数と見なさず、IHB-04を未実施の暫定係数として、オブザーバー評価で誤差伝播を確認する。

## 結果

| 評価 | 結果 |
|---|---:|
| c_ball | {c_ball:.8g} N·m·s²/rad² |
| 生Aによる従来方式の感度比較値 | {c_ball_raw:.8g} N·m·s²/rad² |
| 波形クラスタ・ブートストラップ95%区間 | {ci_low:.8g}〜{ci_high:.8g} N·m·s²/rad² |
| 等価 Cd | {cd:.4f} |
| エネルギー損失 Pearson R | {r_all:.5f} |
| エネルギー損失 RMSE | {rmse_all:.5f} mJ |
| エネルギー損失 R² | {r2_all:.5f} |
| エネルギー損失 RMSE（生A方式） | {rmse_raw:.5f} mJ |
| 半周期ODE次頂点角RMSE | {angle_rmse:.4f}° |
| 半周期ODE次頂点角RMSE（生計測頂点との比較） | {raw_angle_rmse:.4f}° |
| Aフィット品質ゲート未達波形 | {len(failed_amplitude_fits)}/{len(amplitude_fit_metrics)} |
| 波形数・半周期数 | {len(ids)}・{len(intervals)} |

![実測次頂点と半周期ODE予測の比較、開始振幅に対する残差](ihb05_cball_validation.svg)

## 解釈と次のオブザーバー評価

ブートストラップ区間は波形間のばらつきを表し、I/K/τ、球寸法・質量、ロッド抗力近似の不確かさを含まない。Cdは往復運動中のデータに対する等価値であり、孤立した球の普遍値としては扱わない。今後このc_ballを使うときは、本レポートに記載した固定I/K/τと組み合わせ、IHB-04を未実施であることを保ったままオブザーバー側の誤差伝播を評価する。

図のODE次頂点残差は開始振幅に対して全体のPearson R = {residual_amplitude_correlation['ALL']:.3f}（IN {residual_amplitude_correlation['IN']:.3f}、OUT {residual_amplitude_correlation['OUT']:.3f}）となり、低振幅側では予測が実測より小さく、高振幅側では大きくなる傾向が残った。従って、c_ball一つで全振幅域の半周期波形誤差を説明しきったとは言えない。この振幅依存の残差は、オブザーバー誤差評価でパラメータ不確かさとともに考慮する。

## 再現方法

```bash
python 06_Analysis/fitting_pipeline/iterative_hybrid/ihb05_cball_identification.py
```

出力: `ihb05_settings.json`、`amplitude_fit_metrics.csv`、`energy_fit_metrics.csv`、`raw_peak_sensitivity_energy_fit_metrics.csv`、`waveform_estimates.csv`、`interval_predictions.csv`、`reynolds_assessment.csv`、およびPNG/SVG比較図。
"""
    report = report.replace("CBALL_RESULT_TOKEN", f"{c_ball:.8g}")
    (output / "IHB-05_C_BALL_IDENTIFICATION.md").write_text(report, encoding="utf-8")
    print(json.dumps(settings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
