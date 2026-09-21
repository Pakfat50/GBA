"""連続取得した自由振動ログを、確認用の1波形へ分割する。

このスクリプトは係数フィッティングを行わない。最初に全候補を図へ出し、
人が異常波形を確認するための waveform_selection.csv を作る。

使い方:
    python 06_Analysis/fitting_pipeline/review_waveforms.py --date 20260921

確認後は waveform_selection.csv の use_for_fitting に 1（採用）または
0（除外）を入力し、review_status を APPROVED に変更する。
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fitting_tools import detect_free_decay_candidates
from fitting_tools import ensure_directory
from fitting_tools import read_angle_log


# =============================================================================
# 利用者が通常変更する設定
# =============================================================================

# 一覧図と個別図を作成するかを選ぶ。
PLOT_REVIEW_OVERVIEWS = True
PLOT_INDIVIDUAL_WAVEFORMS = True

# 連続ログ用の自動分割条件。
CANDIDATE_SETTINGS = {
    "smooth_time_s": 0.05,
    "baseline_time_s": 5.0,
    "candidate_peak_min_deg": 35.0,
    "candidate_peak_prominence_deg": 0.5,
    "candidate_gap_s": 8.0,
    "peak_distance_s": 0.15,
    "zero_crossing_search_s": 5.0,
    "release_drop_deg": 0.7,
    "minimum_decay_time_s": 5.0,
    "maximum_decay_time_s": 90.0,
    "settle_duration_s": 3.0,
    "settle_center_max_deg": 3.0,
    "settle_std_max_deg": 0.35,
    # 以下は除外を確定する条件ではなく、確認を促す品質フラグである。
    "quality_peak_prominence_deg": 0.2,
    "quality_growth_min_deg": 3.0,
    "quality_growth_ratio": 0.12,
    "quality_max_angle_deg": 75.0,
    "quality_max_jump_deg": 10.0,
}


# =============================================================================
# ここから下は解析処理。通常は変更しない。
# =============================================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "04_Data" / "05_Fitting"
DEFAULT_RESULT_ROOT = SCRIPT_DIRECTORY / "results"
MANIFEST_FILE_NAME = "test_manifest.csv"
SELECTION_FILE_NAME = "waveform_selection.csv"


def parse_arguments():
    """コマンドライン引数を読み取る。"""

    parser = argparse.ArgumentParser(description="自由振動候補波形の確認図を作成")
    parser.add_argument("--date", required=True, help="試験日フォルダ。例: 20260921")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args()


def read_review_manifest(date_directory):
    """連続ログと軸・試験形態の対応表を読み込む。"""

    manifest_path = date_directory / MANIFEST_FILE_NAME
    if not manifest_path.exists():
        raise FileNotFoundError("対応表がありません: " + str(manifest_path))

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required_columns = [
        "data_file",
        "axis",
        "angle_column",
        "configuration",
        "spacer_count",
        "valid",
        "notes",
    ]
    for column in required_columns:
        if column not in manifest.columns:
            raise ValueError("test_manifest.csv に " + column + " 列がありません")
    return manifest, manifest_path


def make_segment_id(axis, configuration, direction, repetition):
    """CSVと図で共通使用する波形IDを作る。"""

    return (
        str(axis).strip().upper()
        + "_"
        + str(configuration).strip().upper()
        + "_"
        + str(direction).strip().upper()
        + "_R"
        + str(repetition).zfill(2)
    )


def preserve_manual_review(new_table, old_path):
    """再実行時に、人が入力した採否とメモを引き継ぐ。"""

    if not old_path.exists():
        return new_table
    old_table = pd.read_csv(old_path, encoding="utf-8-sig")
    if "segment_id" not in old_table.columns:
        return new_table

    manual_columns = ["use_for_fitting", "review_status", "review_note"]
    old_values = {}
    for unused_index, row in old_table.iterrows():
        segment_id = str(row["segment_id"])
        old_values[segment_id] = {}
        for column in manual_columns:
            if column in old_table.columns:
                old_values[segment_id][column] = row[column]

    for row_index, row in new_table.iterrows():
        segment_id = str(row["segment_id"])
        if segment_id not in old_values:
            continue
        for column in manual_columns:
            if column in old_values[segment_id]:
                new_table.at[row_index, column] = old_values[segment_id][column]
    return new_table


def find_candidates(date_directory, manifest):
    """全ログ・全軸から候補を検出し、波形配列と確認表を返す。"""

    selection_rows = []
    waveform_records = []

    for unused_index, manifest_row in manifest.iterrows():
        if int(manifest_row["valid"]) != 1:
            continue

        data_file = str(manifest_row["data_file"])
        file_path = date_directory / data_file
        if not file_path.exists():
            raise FileNotFoundError("データファイルがありません: " + str(file_path))

        axis = str(manifest_row["axis"]).strip().upper()
        angle_column = str(manifest_row["angle_column"]).strip()
        configuration = str(manifest_row["configuration"]).strip().upper()
        spacer_count = int(manifest_row["spacer_count"])
        time_s, angle_rad = read_angle_log(file_path, angle_column)
        candidates = detect_free_decay_candidates(
            time_s, angle_rad, CANDIDATE_SETTINGS
        )

        print(
            data_file
            + " / "
            + axis
            + ": "
            + str(len(candidates))
            + " candidates"
        )
        for candidate in candidates:
            segment_id = make_segment_id(
                axis,
                configuration,
                candidate["direction"],
                candidate["repetition"],
            )
            row = {
                "segment_id": segment_id,
                "data_file": data_file,
                "axis": axis,
                "angle_column": angle_column,
                "configuration": configuration,
                "spacer_count": spacer_count,
                "direction": candidate["direction"],
                "repetition": candidate["repetition"],
                "start_index": candidate["start_index"],
                "end_index": candidate["end_index"],
                "start_time_s": candidate["start_time_s"],
                "end_time_s": candidate["end_time_s"],
                "duration_s": candidate["duration_s"],
                "initial_angle_deg": candidate["initial_angle_deg"],
                "baseline_deg": candidate["baseline_deg"],
                "maximum_abs_angle_deg": candidate["maximum_abs_angle_deg"],
                "maximum_sample_jump_deg": candidate["maximum_sample_jump_deg"],
                "amplitude_growth_count": candidate["amplitude_growth_count"],
                "settled_automatically": candidate["settled_automatically"],
                "suggested_use": candidate["suggested_use"],
                "use_for_fitting": "",
                "review_status": "PENDING",
                "auto_quality_note": candidate["quality_note"],
                "review_note": "",
            }
            selection_rows.append(row)

            start = candidate["start_index"]
            end = candidate["end_index"] + 1
            waveform_records.append(
                {
                    "segment_id": segment_id,
                    "axis": axis,
                    "configuration": configuration,
                    "direction": candidate["direction"],
                    "time_s": time_s[start:end] - time_s[start],
                    "angle_deg": np.rad2deg(angle_rad[start:end]),
                    "baseline_deg": candidate["baseline_deg"],
                    "suggested_use": candidate["suggested_use"],
                    "quality_note": candidate["quality_note"],
                }
            )

    return selection_rows, waveform_records


def draw_one_waveform(plot_axis, record, show_labels):
    """一つの候補波形を指定されたグラフ領域へ描く。"""

    color = "#1f77b4"
    if int(record["suggested_use"]) != 1:
        color = "#d62728"
    plot_axis.plot(record["time_s"], record["angle_deg"], color=color, lw=0.8)
    plot_axis.axhline(record["baseline_deg"], color="black", lw=0.7, linestyle="--")
    title = record["segment_id"]
    if int(record["suggested_use"]) != 1:
        title += "  CHECK"
    plot_axis.set_title(title, fontsize=9)
    plot_axis.set_ylim(-75.0, 75.0)
    plot_axis.grid(alpha=0.25)
    if show_labels:
        plot_axis.set_xlabel("Time from release [s]")
        plot_axis.set_ylabel("Angle [deg]")


def plot_overviews(waveform_records, result_directory):
    """全候補一覧と、軸・形態ごとの拡大一覧を作る。"""

    if len(waveform_records) == 0:
        return

    # 全候補を一枚にまとめる。赤い波形は自動品質チェック対象であり、
    # 人の確認前に除外を確定したものではない。
    column_count = 4
    row_count = int(math.ceil(len(waveform_records) / float(column_count)))
    figure, plot_axes = plt.subplots(
        row_count, column_count, figsize=(18, 3.0 * row_count), squeeze=False
    )
    plot_number = 0
    while plot_number < row_count * column_count:
        row_number = plot_number // column_count
        column_number = plot_number % column_count
        plot_axis = plot_axes[row_number, column_number]
        if plot_number < len(waveform_records):
            draw_one_waveform(plot_axis, waveform_records[plot_number], True)
        else:
            plot_axis.axis("off")
        plot_number += 1
    figure.suptitle("All detected free-decay candidates (red = manual check requested)")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    figure.savefig(result_directory / "waveform_review_all.png", dpi=160)
    plt.close(figure)

    group_keys = []
    for record in waveform_records:
        key = record["axis"] + "_" + record["configuration"]
        if key not in group_keys:
            group_keys.append(key)
    group_keys.sort()

    for key in group_keys:
        records = []
        for record in waveform_records:
            record_key = record["axis"] + "_" + record["configuration"]
            if record_key == key:
                records.append(record)
        column_count = 3
        row_count = int(math.ceil(len(records) / float(column_count)))
        figure, plot_axes = plt.subplots(
            row_count, column_count, figsize=(15, 3.4 * row_count), squeeze=False
        )
        plot_number = 0
        while plot_number < row_count * column_count:
            row_number = plot_number // column_count
            column_number = plot_number % column_count
            plot_axis = plot_axes[row_number, column_number]
            if plot_number < len(records):
                draw_one_waveform(plot_axis, records[plot_number], True)
            else:
                plot_axis.axis("off")
            plot_number += 1
        figure.suptitle(key + " free-decay candidates")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        figure.savefig(result_directory / ("waveform_review_" + key + ".png"), dpi=170)
        plt.close(figure)


def plot_individual_waveforms(waveform_records, result_directory):
    """拡大確認できるよう、候補ごとに一枚の図を保存する。"""

    waveform_directory = result_directory / "waveforms"
    ensure_directory(waveform_directory)
    for record in waveform_records:
        figure, plot_axis = plt.subplots(1, 1, figsize=(11, 4.5))
        draw_one_waveform(plot_axis, record, True)
        note = "No automatic quality warning"
        if int(record["suggested_use"]) != 1:
            # 図は日本語フォントがないPCでも文字化けしない表記にする。
            # 詳細な日本語理由は waveform_selection.csv に保存される。
            note = "Automatic quality warning: manual review required"
        figure.text(0.01, 0.01, note, fontsize=9)
        figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
        figure.savefig(
            waveform_directory / (record["segment_id"] + ".png"), dpi=180
        )
        plt.close(figure)


def main():
    """候補抽出、確認表保存、確認図作成を順に行う。"""

    arguments = parse_arguments()
    date_directory = arguments.data_root / arguments.date
    if not date_directory.is_dir():
        raise FileNotFoundError("試験日フォルダがありません: " + str(date_directory))

    result_directory = arguments.result_root / arguments.date / "waveform_review"
    ensure_directory(result_directory)
    manifest, manifest_path = read_review_manifest(date_directory)
    selection_rows, waveform_records = find_candidates(date_directory, manifest)

    selection_path = result_directory / SELECTION_FILE_NAME
    selection_table = pd.DataFrame(selection_rows)
    selection_table = preserve_manual_review(selection_table, selection_path)
    selection_table.to_csv(selection_path, index=False, encoding="utf-8-sig")

    if PLOT_REVIEW_OVERVIEWS:
        plot_overviews(waveform_records, result_directory)
    if PLOT_INDIVIDUAL_WAVEFORMS:
        plot_individual_waveforms(waveform_records, result_directory)

    print("\nManifest: " + str(manifest_path))
    print("Selection table: " + str(selection_path))
    print("Candidates: " + str(len(selection_rows)))
    print("係数同定は、全行の採否確認が終わるまで実行しません。")


if __name__ == "__main__":
    main()
