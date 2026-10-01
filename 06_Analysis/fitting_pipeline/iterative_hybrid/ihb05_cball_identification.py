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


def make_intervals(turning: pd.DataFrame, selection: pd.DataFrame, physics: dict) -> list[dict]:
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
            a = float(first.centered_peak_angle_deg)
            z = float(last.centered_peak_angle_deg)
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
    intervals = make_intervals(turning, selection, physics)
    c_ball, wave_contributions = fit_one_shared_c(intervals)

    # 全半周期を一つずつ数値積分し、実測次頂点との角度誤差を調べる。
    for row in intervals:
        row["sphere_coulomb_loss_j"] = TAU_BY_AXIS_N_M[row["axis"]] * row["tau_basis"]
        row["predicted_next_angle_rad"] = predict_next(row, c_ball)
        row["predicted_next_angle_deg"] = float(np.rad2deg(row["predicted_next_angle_rad"]))
        row["measured_next_angle_deg"] = float(np.rad2deg(row["measured_next_angle_rad"]))
        row["angle_residual_deg"] = row["predicted_next_angle_deg"] - row["measured_next_angle_deg"]
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
    waveform_table.to_csv(output / "waveform_estimates.csv", index=False, float_format="%.10g")
    prediction_table.to_csv(output / "interval_predictions.csv", index=False, float_format="%.10g")
    reynolds.to_csv(output / "reynolds_assessment.csv", index=False, float_format="%.10g")

    cd = 2.0 * c_ball / (AIR_DENSITY_KG_M3 * SPHERE_AREA_M2 * SPHERE_CENTER_ARM_M**3)
    cd_ci = [2.0 * ci_low / (AIR_DENSITY_KG_M3 * SPHERE_AREA_M2 * SPHERE_CENTER_ARM_M**3),
             2.0 * ci_high / (AIR_DENSITY_KG_M3 * SPHERE_AREA_M2 * SPHERE_CENTER_ARM_M**3)]
    angle_rmse = float(np.sqrt(np.mean(prediction_table.angle_residual_deg**2)))
    settings = {
        "stage": "IHB-05", "date": args.date,
        "interpretation": "IHB-04 skipped: uses IHB-02 base I/K plus measured sphere increments and IHB-03 monotone one-pass tau",
        "c_ball_n_m_s2_per_rad2": c_ball, "c_ball_waveform_bootstrap_95pct": [float(ci_low), float(ci_high)],
        "equivalent_Cd": cd, "equivalent_Cd_95pct": cd_ci,
        "waveforms": len(ids), "half_cycles": len(intervals),
        "half_cycle_ode_next_peak_angle_rmse_deg": angle_rmse,
        "tau_by_axis_n_m": TAU_BY_AXIS_N_M, "b_by_axis_n_m_s_per_rad": {"IN": 0.0, "OUT": 0.0},
        "c_rod_no_ball": THEORETICAL_ROD_C, "c_rod_ball_exposed_length_adjusted": ROD_C_BALL,
        "sphere_diameter_m": SPHERE_DIAMETER_M, "sphere_center_arm_m": SPHERE_CENTER_ARM_M,
        "bootstrap_method": f"{BOOTSTRAP_REPLICATES} waveform-cluster resamples; seed {BOOTSTRAP_SEED}",
        "energy_fit_metrics": energy_metrics,
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
    axes[0].set(title="Measured vs ODE next peak", xlabel="Measured next angle [deg]", ylabel="ODE predicted next angle [deg]")
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
    # Keep LaTeX outside the f-string parser so the backslashes stay intact.
    equation_i_k = r"I_{\mathrm{BALL}}=I_0+J_{G,\mathrm{ball}}+m_{\mathrm{ball}}r_{\mathrm{ball}}^2,\quad K_{\mathrm{BALL}}=K_0+m_{\mathrm{ball}}g r_{\mathrm{signed}}."
    equation_energy = r"\Delta E_{\mathrm{obs}}-\tau\left(|A_n|+|A_{n+1}|\right)-c_{\mathrm{rod,BALL}}C \approx c_{\mathrm{ball}}C,\quad c_{\mathrm{ball}}\ge 0."
    report = f"""# IHB-05: 球抗力係数 c_ball の同定

## 結論

IHB-04をスキップする指示に合わせ、IHB-02の球なし基準 I/K に実測球質量・形状から計算した増分を加え、IHB-03で採用した単調振幅一回積分法のτを固定して、承認済みの球あり自由振動データから共通の非負 c_ball を一回の重み付き最小二乗で推定した。

推定値は **c_ball = {c_ball:.8g} N·m·s²/rad²**。波形単位のクラスタ・ブートストラップ95%区間は **{ci_low:.8g}〜{ci_high:.8g} N·m·s²/rad²**、等価球抗力係数は **Cd = {cd:.4f}**（同95%区間 {cd_ci[0]:.4f}〜{cd_ci[1]:.4f}）だった。半周期ODEの次頂点角RMSEは **{angle_rmse:.4f}°**（{len(intervals)}区間）、エネルギー損失は波形等重みでPearson R **{r_all:.5f}**、RMSE **{rmse_all:.5f} mJ**、R² **{r2_all:.5f}** となった。

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

半周期の観測エネルギー損失から固定摩擦・ロッド損失を引いた分を、球の二乗抗力基底で説明する。

$$
{equation_energy}
$$

ここでCは保存軌道近似から一度計算する二乗抗力基底。推定後はc_ballを固定し、検証済みの半周期ODEソルバーで次頂点を予測した。

## 結果

| 評価 | 結果 |
|---|---:|
| c_ball | {c_ball:.8g} N·m·s²/rad² |
| 波形クラスタ・ブートストラップ95%区間 | {ci_low:.8g}〜{ci_high:.8g} N·m·s²/rad² |
| 等価 Cd | {cd:.4f} |
| エネルギー損失 Pearson R | {r_all:.5f} |
| エネルギー損失 RMSE | {rmse_all:.5f} mJ |
| エネルギー損失 R² | {r2_all:.5f} |
| 半周期ODE次頂点角RMSE | {angle_rmse:.4f}° |
| 波形数・半周期数 | {len(ids)}・{len(intervals)} |

![実測次頂点と半周期ODE予測の比較、開始振幅に対する残差](ihb05_cball_validation.svg)

## 解釈と次のオブザーバー評価

ブートストラップ区間は波形間のばらつきを表し、I/K/τ、球寸法・質量、ロッド抗力近似の不確かさを含まない。Cdは往復運動中のデータに対する等価値であり、孤立した球の普遍値としては扱わない。今後このc_ballを使うときは、本レポートに記載した固定I/K/τと組み合わせ、IHB-04を未実施であることを保ったままオブザーバー側の誤差伝播を評価する。

図のODE次頂点残差は開始振幅に対して全体のPearson R = {residual_amplitude_correlation['ALL']:.3f}（IN {residual_amplitude_correlation['IN']:.3f}、OUT {residual_amplitude_correlation['OUT']:.3f}）となり、低振幅側では予測が実測より小さく、高振幅側では大きくなる傾向が残った。従って、c_ball一つで全振幅域の半周期波形誤差を説明しきったとは言えない。この振幅依存の残差は、オブザーバー誤差評価でパラメータ不確かさとともに考慮する。

## 再現方法

```bash
python 06_Analysis/fitting_pipeline/iterative_hybrid/ihb05_cball_identification.py
```

出力: `ihb05_settings.json`、`energy_fit_metrics.csv`、`waveform_estimates.csv`、`interval_predictions.csv`、`reynolds_assessment.csv`、およびPNG/SVG比較図。
"""
    (output / "IHB-05_C_BALL_IDENTIFICATION.md").write_text(report, encoding="utf-8")
    print(json.dumps(settings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
