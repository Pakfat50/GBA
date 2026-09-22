"""ハイブリッド同定Stage 1の決定論的な波形前処理を実行する。

物理係数は同定せず、確認済み波形ごとに次を決める。

1. 平衡中心を上下頂点包絡線の中点の算術平均から算出する。
2. 最初の半周期を除外し、最初の折返し頂点を時間原点にする。
3. 初期角度を実測頂点、初期速度を0として固定する。
4. 全点平均、従来の終端中央値と平衡中心を比較する。
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fitting_tools import ensure_directory
from fitting_tools import preprocess_free_decay
from fitting_tools import read_angle_log
from fitting_tools import save_csv
from run_calibration import read_waveform_selection


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "04_Data" / "05_Fitting"
DEFAULT_RESULT_ROOT = SCRIPT_DIRECTORY / "results"
MINIMUM_AMPLITUDE_DEG = 4.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ハイブリッド同定Stage 1: 頂点・中心・初期状態の決定"
    )
    parser.add_argument("--date", required=True, help="試験日。例: 20260921")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def analyze_waveforms(date_directory, selection):
    """確認済み全波形へ同じ決定論的前処理を適用する。"""

    waveform_rows = []
    extrema_rows = []
    midpoint_rows = []
    cache = {}
    for unused_index, row in selection.iterrows():
        if int(row["use_for_fitting"]) != 1:
            continue
        key = (str(row["data_file"]), str(row["angle_column"]))
        if key not in cache:
            cache[key] = read_angle_log(date_directory / key[0], key[1])
        complete_time, complete_angle = cache[key]
        start = int(row["start_index"])
        end = int(row["end_index"]) + 1
        time_s = complete_time[start:end]
        angle_rad = complete_angle[start:end]
        existing_center = np.deg2rad(float(row["baseline_deg"]))
        result = preprocess_free_decay(
            time_s,
            angle_rad,
            existing_center_rad=existing_center,
            minimum_amplitude_deg=MINIMUM_AMPLITUDE_DEG,
        )
        extrema = result["extrema"]
        initial_peak = int(result["initial_peak_number"])
        center_deg = float(np.rad2deg(result["center_rad"]))
        mean_deg = float(np.rad2deg(result["all_point_mean_rad"]))
        tail_deg = float(np.rad2deg(result["tail_median_rad"]))
        segment_tail_deg = float(
            np.rad2deg(result["computed_segment_tail_median_rad"])
        )
        midpoint_deg = np.rad2deg(result["midpoint_rad"])
        quarter_count = max(1, int(np.ceil(len(midpoint_deg) / 4.0)))
        early_midpoint_mean_deg = float(np.mean(midpoint_deg[:quarter_count]))
        late_midpoint_mean_deg = float(np.mean(midpoint_deg[-quarter_count:]))
        waveform_rows.append(
            {
                "segment_id": str(row["segment_id"]),
                "axis": str(row["axis"]).strip().upper(),
                "configuration": str(row["configuration"]).strip().upper(),
                "direction": str(row["direction"]).strip().upper(),
                "repetition": int(row["repetition"]),
                "samples": len(time_s),
                "extrema_count": len(extrema["indices"]),
                "positive_extrema": int(np.count_nonzero(extrema["kind"] > 0)),
                "negative_extrema": int(np.count_nonzero(extrema["kind"] < 0)),
                "center_extrema_count": int(
                    np.count_nonzero(extrema["used_for_center"])
                ),
                "quadratic_interpolation_count": int(
                    np.count_nonzero(extrema["quadratic_interpolation_used"])
                ),
                "envelope_center_deg": center_deg,
                "all_point_mean_deg": mean_deg,
                "existing_tail_baseline_deg": tail_deg,
                "segment_tail_median_deg": segment_tail_deg,
                "mean_minus_envelope_deg": mean_deg - center_deg,
                "tail_minus_envelope_deg": tail_deg - center_deg,
                "segment_tail_minus_envelope_deg": segment_tail_deg - center_deg,
                "midpoint_mean_absolute_deviation_deg": float(
                    np.rad2deg(
                        result["midpoint_mean_absolute_deviation_rad"]
                    )
                ),
                "midpoint_range_deg": float(
                    np.rad2deg(result["midpoint_range_rad"])
                ),
                "early_midpoint_mean_deg": early_midpoint_mean_deg,
                "late_midpoint_mean_deg": late_midpoint_mean_deg,
                "late_minus_early_midpoint_deg": (
                    late_midpoint_mean_deg - early_midpoint_mean_deg
                ),
                "midpoint_slope_deg_s": float(
                    np.rad2deg(result["midpoint_slope_rad_s"])
                ),
                "segment_start_time_s": float(time_s[0]),
                "initial_peak_time_s": float(result["initial_time_s"]),
                "ignored_initial_half_cycle_s": float(
                    result["ignored_initial_half_cycle_s"]
                ),
                "initial_angle_deg": float(
                    np.rad2deg(result["initial_angle_rad"])
                ),
                "initial_speed_deg_s": 0.0,
                "initial_peak_kind": (
                    "MAX" if extrema["kind"][initial_peak] > 0 else "MIN"
                ),
            }
        )

        for peak_number in range(len(extrema["indices"])):
            amplitude_deg = float(np.rad2deg(extrema["amplitude_rad"][peak_number]))
            extrema_rows.append(
                {
                    "segment_id": str(row["segment_id"]),
                    "axis": str(row["axis"]).strip().upper(),
                    "configuration": str(row["configuration"]).strip().upper(),
                    "direction": str(row["direction"]).strip().upper(),
                    "peak_number": peak_number,
                    "sample_index_in_segment": int(extrema["indices"][peak_number]),
                    "peak_time_s": float(extrema["time_s"][peak_number]),
                    "time_from_initial_peak_s": float(
                        extrema["time_s"][peak_number] - result["initial_time_s"]
                    ),
                    "peak_angle_deg": float(
                        np.rad2deg(extrema["angle_rad"][peak_number])
                    ),
                    "centered_peak_angle_deg": float(
                        np.rad2deg(
                            extrema["angle_rad"][peak_number]
                            - result["center_rad"]
                        )
                    ),
                    "amplitude_deg": amplitude_deg,
                    "peak_kind": "MAX" if extrema["kind"][peak_number] > 0 else "MIN",
                    "quadratic_interpolation_used": int(
                        extrema["quadratic_interpolation_used"][peak_number]
                    ),
                    "used_for_center": int(extrema["used_for_center"][peak_number]),
                    "is_initial_peak": int(extrema["is_initial_peak"][peak_number]),
                    "eligible_for_later_stages": int(
                        peak_number >= initial_peak
                        and amplitude_deg >= MINIMUM_AMPLITUDE_DEG
                    ),
                }
            )

        for midpoint_time, midpoint in zip(
            result["midpoint_time_s"], result["midpoint_rad"]
        ):
            midpoint_rows.append(
                {
                    "segment_id": str(row["segment_id"]),
                    "axis": str(row["axis"]).strip().upper(),
                    "configuration": str(row["configuration"]).strip().upper(),
                    "direction": str(row["direction"]).strip().upper(),
                    "time_s": float(midpoint_time),
                    "midpoint_deg": float(np.rad2deg(midpoint)),
                    "deviation_from_center_deg": float(
                        np.rad2deg(midpoint - result["center_rad"])
                    ),
                }
            )

    return waveform_rows, extrema_rows, midpoint_rows, cache


def plot_center_method_comparison(waveform_rows, output_path):
    """各波形で求めた中心と比較用中心を明示的な凡例付きで示す。"""

    table = pd.DataFrame(waveform_rows).sort_values(
        ["axis", "configuration", "direction", "repetition"]
    ).reset_index(drop=True)
    x = np.arange(len(table))
    labels = table["segment_id"].tolist()
    figure, axes = plt.subplots(2, 1, figsize=(22, 11), sharex=True)

    axes[0].plot(
        x,
        table["envelope_center_deg"],
        "o-",
        linewidth=1.0,
        markersize=4,
        label="Adopted: mean of envelope midpoints",
    )
    axes[0].plot(
        x,
        table["all_point_mean_deg"],
        "x",
        markersize=5,
        label="Comparison: mean of all angle samples",
    )
    axes[0].plot(
        x,
        table["existing_tail_baseline_deg"],
        "+",
        markersize=6,
        label="Comparison: end-of-test reference",
    )
    axes[0].set_ylabel("Center [deg]")
    axes[0].set_title("Center estimate for each waveform")
    axes[0].legend(ncol=3, loc="upper center", fontsize=11)

    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].plot(
        x,
        table["mean_minus_envelope_deg"],
        "o",
        markersize=4,
        label="All-sample mean minus adopted center",
    )
    axes[1].plot(
        x,
        table["tail_minus_envelope_deg"],
        "s",
        markersize=3.5,
        label="End reference minus adopted center",
    )
    axes[1].set_ylabel("Difference from adopted center [deg]")
    axes[1].legend(ncol=2, loc="upper center", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].set_xlabel("Waveform (each waveform is processed independently)")

    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_center_stability_and_initial_state(waveform_rows, output_path):
    """試験中の中点変化と、固定した波形別初期角度を示す。"""

    table = pd.DataFrame(waveform_rows).sort_values(
        ["axis", "configuration", "direction", "repetition"]
    ).reset_index(drop=True)
    x = np.arange(len(table))
    labels = table["segment_id"].tolist()
    figure, axes = plt.subplots(2, 1, figsize=(22, 11), sharex=True)

    axes[0].axhspan(-0.5, 0.5, color="0.90", label="Measurement-error range (±0.5 deg)")
    axes[0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0].plot(
        x,
        table["late_minus_early_midpoint_deg"],
        "o",
        markersize=5,
        label="Last-quarter mean minus first-quarter mean",
    )
    axes[0].set_ylabel("Change in envelope midpoint [deg]")
    axes[0].set_title("Within-waveform center stability (diagnostic only; no correction applied)")
    axes[0].legend(ncol=2, loc="upper center", fontsize=11)

    maximum = table["initial_peak_kind"] == "MAX"
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].scatter(
        x[maximum], table.loc[maximum, "initial_angle_deg"],
        color="tab:red", s=28, label="Initial state is a positive maximum",
    )
    axes[1].scatter(
        x[~maximum], table.loc[~maximum, "initial_angle_deg"],
        color="tab:blue", s=28, label="Initial state is a negative minimum",
    )
    axes[1].set_ylabel("Initial angle about adopted center [deg]")
    axes[1].set_title("Fixed initial state after excluding the first half-cycle (initial speed = 0)")
    axes[1].legend(ncol=2, loc="upper center", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].set_xlabel("Waveform (initial state is not shared between waveforms)")

    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def group_summary(table):
    rows = []
    for (axis, configuration), group in table.groupby(["axis", "configuration"]):
        rows.append(
            {
                "axis": axis,
                "configuration": configuration,
                "waveforms": len(group),
                "envelope_center_mean_deg": float(group["envelope_center_deg"].mean()),
                "envelope_center_std_deg": float(group["envelope_center_deg"].std(ddof=0)),
                "mean_difference_abs_mean_deg": float(
                    group["mean_minus_envelope_deg"].abs().mean()
                ),
                "tail_difference_abs_mean_deg": float(
                    group["tail_minus_envelope_deg"].abs().mean()
                ),
                "midpoint_mean_absolute_deviation_mean_deg": float(
                    group["midpoint_mean_absolute_deviation_deg"].mean()
                ),
                "midpoint_range_max_deg": float(group["midpoint_range_deg"].max()),
                "late_minus_early_midpoint_mean_deg": float(
                    group["late_minus_early_midpoint_deg"].mean()
                ),
            }
        )
    return rows


def direction_comparison(table):
    """同一軸・形態の正負開始波形で中心推定値の差を比較する。"""

    rows = []
    for (axis, configuration), group in table.groupby(["axis", "configuration"]):
        positive = group[group["direction"] == "P"]
        negative = group[group["direction"] == "N"]
        if len(positive) == 0 or len(negative) == 0:
            continue
        envelope_difference = float(
            positive["envelope_center_deg"].mean()
            - negative["envelope_center_deg"].mean()
        )
        mean_difference = float(
            positive["all_point_mean_deg"].mean()
            - negative["all_point_mean_deg"].mean()
        )
        rows.append(
            {
                "axis": axis,
                "configuration": configuration,
                "positive_waveforms": len(positive),
                "negative_waveforms": len(negative),
                "envelope_positive_minus_negative_deg": envelope_difference,
                "all_point_mean_positive_minus_negative_deg": mean_difference,
                "envelope_abs_direction_difference_deg": abs(envelope_difference),
                "all_point_mean_abs_direction_difference_deg": abs(mean_difference),
            }
        )
    return rows


def write_report(output_path, waveform_rows, group_rows, direction_rows):
    table = pd.DataFrame(waveform_rows)
    groups = pd.DataFrame(group_rows)
    directions = pd.DataFrame(direction_rows)
    largest_mean = table.iloc[table["mean_minus_envelope_deg"].abs().argsort()[::-1][:5]]
    largest_tail = table.iloc[table["tail_minus_envelope_deg"].abs().argsort()[::-1][:5]]
    lines = [
        "# Stage 1: 頂点・平衡中心・初期状態の決定",
        "",
        "## 採用済みの判断",
        "",
        "| 項目 | 採用内容 | フィッティング変数か |",
        "|---|---|---|",
        "| 平衡中心 | 各波形の上下包絡線中点の算術平均 | いいえ |",
        "| 時間原点 | 解放後の最初の半周期を除外した、最初の折返し頂点 | いいえ |",
        "| 初期角度 | 上記頂点の実測角度から、その波形の平衡中心を引いた値 | いいえ |",
        "| 初期速度 | 頂点なので0 | いいえ |",
        "| センサーオフセット | Stage 1では考慮しない | いいえ |",
        "| 試験中の中心変化 | 約0.5 degは計測誤差範囲として補正・除外しない | いいえ |",
        "",
        "中心、時間原点、初期角度は46波形それぞれで決めている。軸・形態別または全波形の",
        "平均値を、個々の波形の初期条件として流用していない。物理係数の同定はまだ行っていない。",
        "",
        "## 算出方法",
        "",
        "1. 平衡中心を仮定せず極大・極小を抽出する。",
        "2. 各頂点の時刻と角度を、近傍3点の二次補間で補正する。",
        "3. 振幅4 deg以上の正負頂点から上下包絡線を作る。",
        "4. 同じ時刻の上下包絡線の中点を求め、その算術平均を当該波形の平衡中心とする。",
        "5. 解放後最初の折返し頂点を、時間原点・初期角度として固定する。",
        "",
        "## 全体集計",
        "",
        f"- 採用波形: {len(table)}",
        f"- 検出頂点: {int(table['extrema_count'].sum())}",
        f"- 二次補間適用率: {100.0 * table['quadratic_interpolation_count'].sum() / table['extrema_count'].sum():.1f}%",
        f"- |全点平均 - 包絡線中心| 中央値: {table['mean_minus_envelope_deg'].abs().median():.3f} deg",
        f"- |全点平均 - 包絡線中心| 最大: {table['mean_minus_envelope_deg'].abs().max():.3f} deg",
        f"- |既存終端中心 - 包絡線中心| 中央値: {table['tail_minus_envelope_deg'].abs().median():.3f} deg",
        f"- |既存終端中心 - 包絡線中心| 最大: {table['tail_minus_envelope_deg'].abs().max():.3f} deg",
        f"- 包絡線中点の平均絶対偏差の中央値: {table['midpoint_mean_absolute_deviation_deg'].median():.3f} deg",
        f"- 包絡線中点範囲 最大: {table['midpoint_range_deg'].max():.3f} deg",
        f"- 後半1/4中点平均 - 前半1/4中点平均の中央値: {table['late_minus_early_midpoint_deg'].median():.3f} deg",
        f"- 除外した最初の半周期 中央値: {table['ignored_initial_half_cycle_s'].median():.3f} s",
        f"- 初期振幅 中央値: {table['initial_angle_deg'].abs().median():.3f} deg",
        f"- 包絡線中心の|P開始 - N開始| 中央値: {directions['envelope_abs_direction_difference_deg'].median():.3f} deg",
        f"- 全点平均の|P開始 - N開始| 中央値: {directions['all_point_mean_abs_direction_difference_deg'].median():.3f} deg",
        "",
        "### 指標の定義",
        "",
        "- 包絡線中点の平均絶対偏差: 各中点と、その波形で採用した中点平均との差の絶対値を平均した値。",
        "- 包絡線中点範囲: 同じ波形における中点の最大値と最小値の差。外れ値にも反応する診断値。",
        "- 前半・後半1/4中点平均: 中点系列の先頭25%と末尾25%をそれぞれ算術平均した値。",
        "- 表中の「中央値」: 46波形の集計値を極端な波形に左右されにくく示すためだけに使用。波形中心の算出には使用しない。",
        "",
        "## 試験中の中心変化に関する誤差メモ",
        "",
        "包絡線中点には、前半から後半へ約0.5 deg変化する傾向が見られる。今回は計測誤差範囲と判断し、",
        "中心は時間変化させず、各波形の中点系列全体の算術平均で固定する。この変化だけを理由に波形を除外しない。",
        "",
        "後段の連続波形再現で誤差が大きい場合、または残差に時間・振幅・方向依存性が残る場合は、",
        "次を修正候補として再評価する。",
        "",
        "- 振幅または時間に依存する平衡中心",
        "- 角度センサーの非線形性またはゼロ点変化",
        "- 方向別摩擦またはヒステリシス",
        "",
        "## 軸・形態別集計",
        "",
        "ここでの群平均は診断用であり、個々の波形の中心・初期条件には使用しない。",
        "",
        "| 軸 | 形態 | 波形数 | 波形別中心の群平均 [deg] | 群内標準偏差 [deg] | |全点平均との差|平均 [deg] | |終端との差|平均 [deg] | 中点の平均絶対偏差の群平均 [deg] | 中点範囲最大 [deg] | 後半-前半の群平均 [deg] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for unused_index, row in groups.iterrows():
        lines.append(
            f"| {row['axis']} | {row['configuration']} | {int(row['waveforms'])} | "
            f"{row['envelope_center_mean_deg']:.3f} | {row['envelope_center_std_deg']:.3f} | "
            f"{row['mean_difference_abs_mean_deg']:.3f} | {row['tail_difference_abs_mean_deg']:.3f} | "
            f"{row['midpoint_mean_absolute_deviation_mean_deg']:.3f} | "
            f"{row['midpoint_range_max_deg']:.3f} | "
            f"{row['late_minus_early_midpoint_mean_deg']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 開始方向による中心差",
            "",
            "全点平均は有限長の減衰波形を開始側の山から平均するため、開始方向の影響を受ける。",
            "同一軸・形態でP開始平均からN開始平均を差し引き、包絡線中心と比較した。",
            "",
            "| 軸 | 形態 | 包絡線中心 P-N [deg] | 全点平均 P-N [deg] |",
            "|---|---|---:|---:|",
        ]
    )
    for unused_index, row in directions.iterrows():
        lines.append(
            f"| {row['axis']} | {row['configuration']} | "
            f"{row['envelope_positive_minus_negative_deg']:.3f} | "
            f"{row['all_point_mean_positive_minus_negative_deg']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 全点平均との差が大きい波形",
            "",
            "| 波形 | 軸 | 形態 | 全点平均 - 包絡線中心 [deg] |",
            "|---|---|---|---:|",
        ]
    )
    for unused_index, row in largest_mean.iterrows():
        lines.append(
            f"| {row['segment_id']} | {row['axis']} | {row['configuration']} | "
            f"{row['mean_minus_envelope_deg']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 既存終端中心との差が大きい波形",
            "",
            "| 波形 | 軸 | 形態 | 既存終端中心 - 包絡線中心 [deg] |",
            "|---|---|---|---:|",
        ]
    )
    for unused_index, row in largest_tail.iterrows():
        lines.append(
            f"| {row['segment_id']} | {row['axis']} | {row['configuration']} | "
            f"{row['tail_minus_envelope_deg']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Stage 1のレビュー結論",
            "",
            "1. 承認: 各波形の包絡線中点の算術平均を平衡中心とする。",
            "2. 承認: 最初の半周期を除外し、最初の折返し頂点で時間・角度・速度を固定する。",
            "3. 承認: 約0.5 degの中点変化は今回は計測誤差として補正せず、波形も除外しない。",
            "4. 記録: 後段の誤差が大きい場合に、中心変化を修正候補へ戻す。",
            "",
            "## 出力",
            "",
            "- [波形別の中心推定方法比較](preprocessing_overview.png)",
            "- [試験中の中心変化と波形別初期状態](center_stability_and_initial_state.png)",
            "- [波形別前処理結果](waveform_preprocessing.csv)",
            "- [二次補間後の全頂点](turning_points.csv)",
            "- [包絡線中点系列](center_midpoints.csv)",
            "- [軸・形態別集計](preprocessing_group_summary.csv)",
            "- [開始方向別の中心比較](center_direction_comparison.csv)",
            "- [実行条件](preprocessing_settings.json)",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    arguments = parse_arguments()
    date_directory = arguments.data_root / arguments.date
    if not date_directory.is_dir():
        raise FileNotFoundError("試験日フォルダがありません: " + str(date_directory))
    parent_result = arguments.result_root / arguments.date
    selection, selection_path = read_waveform_selection(parent_result)
    if selection is None:
        raise FileNotFoundError("確認済み波形表がありません: " + str(selection_path))

    output_directory = (
        parent_result / "hybrid_identification" / "01_preprocessing"
    )
    ensure_directory(output_directory)
    waveform_rows, extrema_rows, midpoint_rows, cache = analyze_waveforms(
        date_directory, selection
    )
    group_rows = group_summary(pd.DataFrame(waveform_rows))
    direction_rows = direction_comparison(pd.DataFrame(waveform_rows))
    save_csv(output_directory / "waveform_preprocessing.csv", waveform_rows)
    save_csv(output_directory / "turning_points.csv", extrema_rows)
    save_csv(output_directory / "center_midpoints.csv", midpoint_rows)
    save_csv(output_directory / "preprocessing_group_summary.csv", group_rows)
    save_csv(output_directory / "center_direction_comparison.csv", direction_rows)
    plot_center_method_comparison(
        waveform_rows, output_directory / "preprocessing_overview.png"
    )
    plot_center_stability_and_initial_state(
        waveform_rows,
        output_directory / "center_stability_and_initial_state.png",
    )
    write_report(
        output_directory / "PREPROCESSING_REPORT.md",
        waveform_rows,
        group_rows,
        direction_rows,
    )

    input_hashes = {}
    for data_file, unused_angle_column in cache:
        input_path = date_directory / data_file
        input_hashes[str(input_path.relative_to(REPOSITORY_ROOT))] = file_sha256(
            input_path
        )
    settings = {
        "stage": 1,
        "date": arguments.date,
        "minimum_amplitude_deg": MINIMUM_AMPLITUDE_DEG,
        "smooth_time_s": 0.07,
        "peak_prominence_deg": 0.25,
        "center_method": "arithmetic mean of interpolated positive/negative envelope midpoint for each waveform",
        "center_drift_reference_deg": 0.5,
        "center_drift_policy": "ignore as measurement error in Stage 1; reconsider if downstream residuals are large or systematic",
        "initial_state_method": "first detected turning point after release; speed fixed to zero",
        "selection_file": str(selection_path.relative_to(REPOSITORY_ROOT)),
        "selection_sha256": file_sha256(selection_path),
        "input_sha256": input_hashes,
    }
    (output_directory / "preprocessing_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("結果: " + str(output_directory))


if __name__ == "__main__":
    main()
