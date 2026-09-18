"""TWELITE角度ログからセンサーノイズを自動解析する。

使い方:
    python 06_Analysis/sensor_noise_pipeline/run_sensor_noise_analysis.py
    python 06_Analysis/sensor_noise_pipeline/run_sensor_noise_analysis.py --date 20260918

Pythonに不慣れな方が変更しやすいよう、通常変更する値はこの直後にまとめる。
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensor_noise_tools import calculate_segment_metrics
from sensor_noise_tools import read_angle_log
from sensor_noise_tools import split_stationary_segments


# =============================================================================
# 利用者が通常変更する設定
# =============================================================================

# 作成するグラフをTrue / Falseで選ぶ。
# CSV結果は、グラフを作成しない場合でも必ず保存する。
PLOT_TIME_SERIES = True
PLOT_HISTOGRAM = True
PLOT_PSD = True
PLOT_AUTOCORRELATION = True
PLOT_SAMPLE_INTERVALS = True

# MT6701の14 bit角度出力から求めた量子化幅。
EXPECTED_QUANTIZATION_DEG = 360.0 / 16384.0

# 1ファイル中の静止区間を自動分割する条件。
# 3回の測定間では、目標角度から一度1 deg以上動かす。
SEGMENT_SETTINGS = {
    "stationary_window_s": 1.0,
    "stationary_std_max_deg": 0.08,
    "target_tolerance_deg": 0.8,
    "minimum_segment_s": 12.0,
    "edge_trim_s": 1.0,
}

# 有色ノイズの時定数を推定するときに使う自己相関の最大時間。
MAXIMUM_CORRELATION_TIME_S = 3.0


# =============================================================================
# ここから下は解析処理。通常は変更しない。
# =============================================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "04_Data" / "05_Fitting"
DEFAULT_RESULT_ROOT = SCRIPT_DIRECTORY / "results"
MANIFEST_FILE_NAME = "sensor_noise_manifest.csv"


def parse_arguments():
    """コマンドライン引数を読み取る。"""

    parser = argparse.ArgumentParser(description="角度センサーノイズの自動解析")
    parser.add_argument("--date", help="使用する試験日フォルダ。例: 20260918")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="入力日付フォルダの親ディレクトリ",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="結果を保存する親ディレクトリ",
    )
    return parser.parse_args()


def list_date_directories(data_root):
    """8桁日付名の入力フォルダを一覧にする。"""

    directories = []
    if not data_root.exists():
        return directories
    for path in data_root.iterdir():
        if path.is_dir() and len(path.name) == 8 and path.name.isdigit():
            directories.append(path)
    directories.sort(key=lambda item: item.name)
    return directories


def select_date_directory(data_root, requested_date):
    """日付指定または番号入力で解析対象フォルダを決める。"""

    if requested_date:
        selected = data_root / requested_date
        if not selected.is_dir():
            raise FileNotFoundError("試験日フォルダがありません: " + str(selected))
        return selected

    directories = list_date_directories(data_root)
    if len(directories) == 0:
        raise FileNotFoundError(
            str(data_root) + " にYYYYMMDD形式のフォルダがありません"
        )

    print("\n解析する試験日を選択してください。")
    index = 0
    while index < len(directories):
        print("  " + str(index + 1) + ": " + directories[index].name)
        index += 1
    selected_number = int(input("番号を入力: "))
    if selected_number < 1 or selected_number > len(directories):
        raise ValueError("選択番号が範囲外です")
    return directories[selected_number - 1]


def read_manifest(date_directory):
    """データファイルと試験条件を対応付けるCSVを読み込む。"""

    manifest_path = date_directory / MANIFEST_FILE_NAME
    if not manifest_path.exists():
        template = SCRIPT_DIRECTORY / "templates" / "sensor_noise_manifest_template.csv"
        message = str(manifest_path) + " がありません。\n"
        message += str(template) + " をコピーして実際のファイル名へ直してください。"
        raise FileNotFoundError(message)

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required_columns = [
        "data_file",
        "axis",
        "angle_column",
        "condition",
        "target_angle_deg",
        "expected_repetitions",
        "use_start_s",
        "use_end_s",
        "valid",
        "notes",
    ]
    for column in required_columns:
        if column not in manifest.columns:
            raise ValueError(MANIFEST_FILE_NAME + " に " + column + " 列がありません")

    numeric_columns = [
        "target_angle_deg",
        "expected_repetitions",
        "use_start_s",
        "use_end_s",
        "valid",
    ]
    for column in numeric_columns:
        manifest[column] = pd.to_numeric(manifest[column], errors="coerce")

    valid_rows = manifest[manifest["valid"] == 1]
    required_numbers = ["target_angle_deg", "expected_repetitions", "valid"]
    for column in required_numbers:
        if valid_rows[column].isna().any():
            bad_rows = valid_rows[valid_rows[column].isna()]["data_file"].tolist()
            raise ValueError(column + " が未記入です: " + ", ".join(bad_rows))
    return manifest, manifest_path


def safe_name(text):
    """条件名をファイル名に安全な文字へ置き換える。"""

    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text).strip())
    if cleaned == "":
        cleaned = "condition"
    return cleaned


def select_time_range(time_s, angle_deg, start_s, end_s):
    """対応表で指定した時間範囲だけを取り出す。空欄なら全範囲を使う。"""

    mask = np.ones(len(time_s), dtype=bool)
    if np.isfinite(start_s):
        mask &= time_s >= start_s
    if np.isfinite(end_s):
        mask &= time_s <= end_s
    selected_time = time_s[mask]
    selected_angle = angle_deg[mask]
    if len(selected_time) < 20:
        raise ValueError("指定時間内のデータが少なすぎます")
    return selected_time, selected_angle


def make_segment_id(axis, condition, repetition):
    """結果CSVで使用する静止区間の識別子を作る。"""

    return safe_name(axis) + "_" + safe_name(condition) + "_R" + str(repetition).zfill(2)


def analyze_files(date_directory, manifest):
    """対応表の全行を読み、静止区間分割とノイズ解析を実行する。"""

    segment_rows = []
    detail_records = []
    source_records = []

    for unused_index, row in manifest.iterrows():
        if int(row["valid"]) != 1:
            continue

        data_file = str(row["data_file"]).strip()
        file_path = date_directory / data_file
        if not file_path.exists():
            raise FileNotFoundError("データファイルがありません: " + str(file_path))

        axis = str(row["axis"]).strip().upper()
        angle_column = str(row["angle_column"]).strip()
        condition = str(row["condition"]).strip()
        target_angle_deg = float(row["target_angle_deg"])
        expected_repetitions = int(row["expected_repetitions"])

        print("読込・静止区間分割: " + data_file + " / " + axis + " / " + condition)
        time_s, angle_deg = read_angle_log(file_path, angle_column)
        time_s, angle_deg = select_time_range(
            time_s,
            angle_deg,
            float(row["use_start_s"]),
            float(row["use_end_s"]),
        )
        segments = split_stationary_segments(
            time_s,
            angle_deg,
            target_angle_deg,
            SEGMENT_SETTINGS,
        )

        source_record = {
            "data_file": data_file,
            "axis": axis,
            "angle_column": angle_column,
            "condition": condition,
            "target_angle_deg": target_angle_deg,
            "time_s": time_s,
            "angle_deg": angle_deg,
            "segments": segments,
        }
        source_records.append(source_record)

        if len(segments) < expected_repetitions:
            message = "警告: " + data_file + " / " + axis
            message += " の静止区間は " + str(len(segments)) + " 個です。"
            message += " 期待値=" + str(expected_repetitions)
            print(message)

        segment_index = 0
        while segment_index < len(segments):
            if segment_index >= expected_repetitions:
                break
            segment = segments[segment_index]
            start = segment["start_index"]
            end = segment["end_index"] + 1
            segment_time = time_s[start:end]
            segment_angle = angle_deg[start:end]
            segment_id = make_segment_id(axis, condition, segment["repetition"])

            metrics, details = calculate_segment_metrics(
                segment_time,
                segment_angle,
                EXPECTED_QUANTIZATION_DEG,
                MAXIMUM_CORRELATION_TIME_S,
            )
            result_row = {
                "segment_id": segment_id,
                "data_file": data_file,
                "axis": axis,
                "angle_column": angle_column,
                "condition": condition,
                "target_angle_deg": target_angle_deg,
                "repetition": segment["repetition"],
                "start_time_s": segment["start_time_s"],
                "end_time_s": segment["end_time_s"],
            }
            for key in metrics:
                result_row[key] = metrics[key]
            segment_rows.append(result_row)

            detail_record = {
                "segment_id": segment_id,
                "data_file": data_file,
                "axis": axis,
                "condition": condition,
                "time_s": segment_time,
                "angle_deg": segment_angle,
                "residual_deg": details["residual_deg"],
                "frequency_hz": details["frequency_hz"],
                "psd_deg2_per_hz": details["psd_deg2_per_hz"],
                "autocovariance_deg2": details["autocovariance_deg2"],
                "sample_period_s": metrics["sample_period_ms"] * 0.001,
            }
            detail_records.append(detail_record)
            print("  解析: " + segment_id)
            segment_index += 1

    if len(segment_rows) == 0:
        raise ValueError("解析できる静止区間がありません")
    return segment_rows, detail_records, source_records


def summarize_conditions(segment_table):
    """同じ軸・条件の3反復をまとめ、再現性を計算する。"""

    rows = []
    combinations = segment_table[["axis", "condition"]].drop_duplicates()
    for unused_index, combination in combinations.iterrows():
        axis = combination["axis"]
        condition = combination["condition"]
        selected = segment_table[
            (segment_table["axis"] == axis)
            & (segment_table["condition"] == condition)
        ]

        mean_angles = selected["mean_angle_deg"].to_numpy(dtype=float)
        repeatability = 0.0
        if len(mean_angles) >= 2:
            repeatability = float(np.std(mean_angles, ddof=1))

        summary = {
            "axis": axis,
            "condition": condition,
            "repetition_count": int(len(selected)),
            "mean_angle_deg": float(np.mean(mean_angles)),
            "repeatability_std_deg": repeatability,
        }
        mean_columns = [
            "noise_std_deg",
            "robust_noise_std_deg",
            "peak_to_peak_deg",
            "drift_slope_deg_s",
            "sample_period_ms",
            "sample_rate_hz",
            "jitter_std_ms",
            "quantization_grid_match_ratio",
            "low_frequency_power_fraction",
            "white_noise_std_deg",
            "colored_noise_std_deg",
            "colored_time_constant_s",
            "ar1_coefficient",
            "quantization_noise_std_deg",
        ]
        for column in mean_columns:
            summary[column + "_mean"] = float(np.mean(selected[column]))
        summary["missing_interval_count_sum"] = int(
            np.sum(selected["missing_interval_count"])
        )
        rows.append(summary)
    return rows


def make_sensor_recommendations(segment_table):
    """オブザーバーのセンサーモデルへ転記する代表値を軸ごとにまとめる。"""

    rows = []
    axes = sorted(segment_table["axis"].unique().tolist())
    for axis in axes:
        selected = segment_table[segment_table["axis"] == axis]
        white_std = float(np.median(selected["white_noise_std_deg"]))
        colored_std = float(np.median(selected["colored_noise_std_deg"]))
        positive_tau = selected[selected["colored_time_constant_s"] > 0.0]
        colored_tau = 0.0
        if len(positive_tau) > 0:
            colored_tau = float(np.median(positive_tau["colored_time_constant_s"]))
        quantization_std = EXPECTED_QUANTIZATION_DEG / np.sqrt(12.0)
        total_std = float(
            np.sqrt(white_std**2 + colored_std**2 + quantization_std**2)
        )
        row = {
            "axis": axis,
            "sample_rate_hz": float(np.median(selected["sample_rate_hz"])),
            "quantization_bits": 14,
            "quantization_deg": EXPECTED_QUANTIZATION_DEG,
            "white_noise_std_deg": white_std,
            "colored_noise_std_deg": colored_std,
            "colored_time_constant_s": colored_tau,
            "angle_noise_std_deg": total_std,
            "jitter_std_ms": float(np.max(selected["jitter_std_ms"])),
            "fixed_delay_ms": np.nan,
            "delay_note": "静止試験では同定不可。既知の角度変化との比較試験が必要",
        }
        rows.append(row)
    return rows


def save_detail_csvs(result_directory, detail_records):
    """PSDと自己相関を再確認できる長形式CSVとして保存する。"""

    psd_rows = []
    autocorrelation_rows = []
    for record in detail_records:
        frequency = record["frequency_hz"]
        psd = record["psd_deg2_per_hz"]
        index = 0
        while index < len(frequency):
            psd_rows.append(
                {
                    "segment_id": record["segment_id"],
                    "frequency_hz": frequency[index],
                    "psd_deg2_per_hz": psd[index],
                }
            )
            index += 1

        covariance = record["autocovariance_deg2"]
        lag = 0
        while lag < len(covariance):
            normalized = 0.0
            if covariance[0] > 0.0:
                normalized = covariance[lag] / covariance[0]
            autocorrelation_rows.append(
                {
                    "segment_id": record["segment_id"],
                    "lag_samples": lag,
                    "lag_s": lag * record["sample_period_s"],
                    "autocorrelation": normalized,
                    "autocovariance_deg2": covariance[lag],
                }
            )
            lag += 1

    pd.DataFrame(psd_rows).to_csv(result_directory / "psd.csv", index=False)
    pd.DataFrame(autocorrelation_rows).to_csv(
        result_directory / "autocorrelation.csv", index=False
    )


def plot_time_series(result_directory, source_records):
    """元波形と自動検出した静止区間を描く。"""

    for record in source_records:
        figure, axis = plt.subplots(figsize=(12, 4.8))
        axis.plot(record["time_s"], record["angle_deg"], color="tab:blue", linewidth=0.8)
        axis.axhline(record["target_angle_deg"], color="black", linestyle="--", linewidth=1.0)
        for segment in record["segments"]:
            axis.axvspan(
                segment["start_time_s"],
                segment["end_time_s"],
                color="tab:green",
                alpha=0.18,
            )
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Angle [deg]")
        axis.set_title(record["axis"] + " / " + record["condition"] + " stationary detection")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        name = "timeseries_" + safe_name(record["axis"] + "_" + record["condition"]) + ".png"
        figure.savefig(result_directory / name, dpi=160)
        plt.close(figure)


def plot_histograms(result_directory, detail_records):
    """一次ドリフト除去後の角度誤差分布を描く。"""

    groups = {}
    for record in detail_records:
        key = record["axis"] + "_" + record["condition"]
        if key not in groups:
            groups[key] = []
        groups[key].append(record)

    for key in groups:
        figure, axis = plt.subplots(figsize=(8, 5))
        for record in groups[key]:
            axis.hist(
                record["residual_deg"],
                bins=40,
                density=True,
                histtype="step",
                linewidth=1.2,
                label=record["segment_id"],
            )
        axis.set_xlabel("Detrended angle error [deg]")
        axis.set_ylabel("Probability density")
        axis.set_title(key + " noise distribution")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(result_directory / ("histogram_" + safe_name(key) + ".png"), dpi=160)
        plt.close(figure)


def plot_psd(result_directory, detail_records):
    """静止ノイズのパワースペクトル密度を描く。"""

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for record in detail_records:
        frequency = record["frequency_hz"]
        positive = frequency > 0.0
        axis.loglog(
            frequency[positive],
            record["psd_deg2_per_hz"][positive],
            linewidth=1.0,
            alpha=0.8,
            label=record["segment_id"],
        )
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("PSD [deg^2/Hz]")
    axis.set_title("Angle sensor noise spectrum")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(result_directory / "noise_psd.png", dpi=160)
    plt.close(figure)


def plot_autocorrelation(result_directory, detail_records):
    """有色ノイズの時間スケールを確認する自己相関を描く。"""

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for record in detail_records:
        covariance = record["autocovariance_deg2"]
        if covariance[0] <= 0.0:
            continue
        lag_s = np.arange(len(covariance)) * record["sample_period_s"]
        axis.plot(
            lag_s,
            covariance / covariance[0],
            linewidth=1.0,
            alpha=0.8,
            label=record["segment_id"],
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Lag [s]")
    axis.set_ylabel("Autocorrelation")
    axis.set_title("Angle sensor noise autocorrelation")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(result_directory / "noise_autocorrelation.png", dpi=160)
    plt.close(figure)


def plot_sample_intervals(result_directory, source_records):
    """サンプル周期と欠落の有無を確認するヒストグラムを描く。"""

    figure, axis = plt.subplots(figsize=(8, 5))
    for record in source_records:
        intervals_ms = np.diff(record["time_s"]) * 1000.0
        label = record["axis"] + " / " + record["condition"]
        axis.hist(intervals_ms, bins=30, histtype="step", linewidth=1.2, label=label)
    axis.set_xlabel("Sample interval [ms]")
    axis.set_ylabel("Count")
    axis.set_title("Logger sample intervals")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(result_directory / "sample_intervals.png", dpi=160)
    plt.close(figure)


def sha256_file(path):
    """入力ファイルの再現性確認に使うSHA-256を計算する。"""

    digest = hashlib.sha256()
    with open(path, "rb") as file_object:
        while True:
            block = file_object.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def save_provenance(result_directory, date_directory, manifest, manifest_path):
    """使用した入力、設定、ハッシュ値をJSONへ保存する。"""

    input_files = []
    used_names = []
    for unused_index, row in manifest.iterrows():
        if int(row["valid"]) == 1:
            name = str(row["data_file"]).strip()
            if name not in used_names:
                used_names.append(name)
    for name in used_names:
        path = date_directory / name
        input_files.append({"path": name, "sha256": sha256_file(path)})

    provenance = {
        "input_date_directory": str(date_directory.relative_to(REPOSITORY_ROOT)),
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "input_files": input_files,
        "expected_quantization_deg": EXPECTED_QUANTIZATION_DEG,
        "segment_settings": SEGMENT_SETTINGS,
        "maximum_correlation_time_s": MAXIMUM_CORRELATION_TIME_S,
        "plot_settings": {
            "time_series": PLOT_TIME_SERIES,
            "histogram": PLOT_HISTOGRAM,
            "psd": PLOT_PSD,
            "autocorrelation": PLOT_AUTOCORRELATION,
            "sample_intervals": PLOT_SAMPLE_INTERVALS,
        },
    }
    with open(result_directory / "provenance.json", "w", encoding="utf-8") as file_object:
        json.dump(provenance, file_object, ensure_ascii=False, indent=2)


def write_report(result_directory, date_name, segment_table, summary_table, recommendation_table):
    """主な結果と解釈上の注意をMarkdownへまとめる。"""

    lines = []
    lines.append("# センサーノイズ解析レポート")
    lines.append("")
    lines.append("対象試験日: `" + date_name + "`")
    lines.append("")
    lines.append("## センサーモデル推奨値")
    lines.append("")
    lines.append("| 軸 | 周波数 [Hz] | 白色σ [deg] | 有色σ [deg] | 時定数 [s] | 合成σ [deg] | ジッタσ [ms] |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for unused_index, row in recommendation_table.iterrows():
        line = "| " + str(row["axis"])
        line += " | " + format(row["sample_rate_hz"], ".4f")
        line += " | " + format(row["white_noise_std_deg"], ".6f")
        line += " | " + format(row["colored_noise_std_deg"], ".6f")
        line += " | " + format(row["colored_time_constant_s"], ".4f")
        line += " | " + format(row["angle_noise_std_deg"], ".6f")
        line += " | " + format(row["jitter_std_ms"], ".6f") + " |"
        lines.append(line)

    lines.append("")
    lines.append("## 条件別再現性")
    lines.append("")
    lines.append("| 軸 | 条件 | 反復数 | ノイズσ [deg] | 平均値再現性σ [deg] | 格子一致率 | 欠落周期数 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for unused_index, row in summary_table.iterrows():
        line = "| " + str(row["axis"])
        line += " | " + str(row["condition"])
        line += " | " + str(int(row["repetition_count"]))
        line += " | " + format(row["noise_std_deg_mean"], ".6f")
        line += " | " + format(row["repeatability_std_deg"], ".6f")
        line += " | " + format(row["quantization_grid_match_ratio_mean"], ".3f")
        line += " | " + str(int(row["missing_interval_count_sum"])) + " |"
        lines.append(line)

    lines.append("")
    lines.append("## 解釈上の注意")
    lines.append("")
    lines.append("- 白色・有色ノイズは、各静止区間から一次ドリフトを除いた後に分離した。")
    lines.append("- 量子化幅はMT6701の14 bit出力から `360 / 2^14` degとした。")
    lines.append("- 固定遅延は静止データだけでは同定できない。既知角度を同時記録する動的試験が必要である。")
    lines.append("- 機構の微小振動や固定治具の動きもセンサーノイズに含まれるため、装置を剛に固定する。")
    lines.append("")
    lines.append("解析した静止区間数: " + str(len(segment_table)))
    lines.append("")
    with open(result_directory / "SENSOR_NOISE_REPORT.md", "w", encoding="utf-8") as file_object:
        file_object.write("\n".join(lines))


def main():
    """フォルダ選択から結果保存までを順に実行する。"""

    arguments = parse_arguments()
    date_directory = select_date_directory(arguments.data_root, arguments.date)
    result_directory = arguments.result_root / date_directory.name
    result_directory.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path = read_manifest(date_directory)
    segment_rows, detail_records, source_records = analyze_files(date_directory, manifest)
    segment_table = pd.DataFrame(segment_rows)
    summary_table = pd.DataFrame(summarize_conditions(segment_table))
    recommendation_table = pd.DataFrame(make_sensor_recommendations(segment_table))

    segment_table.to_csv(result_directory / "segment_noise_metrics.csv", index=False)
    summary_table.to_csv(result_directory / "condition_summary.csv", index=False)
    recommendation_table.to_csv(result_directory / "recommended_sensor_model.csv", index=False)
    save_detail_csvs(result_directory, detail_records)
    save_provenance(result_directory, date_directory, manifest, manifest_path)
    write_report(
        result_directory,
        date_directory.name,
        segment_table,
        summary_table,
        recommendation_table,
    )

    if PLOT_TIME_SERIES:
        plot_time_series(result_directory, source_records)
    if PLOT_HISTOGRAM:
        plot_histograms(result_directory, detail_records)
    if PLOT_PSD:
        plot_psd(result_directory, detail_records)
    if PLOT_AUTOCORRELATION:
        plot_autocorrelation(result_directory, detail_records)
    if PLOT_SAMPLE_INTERVALS:
        plot_sample_intervals(result_directory, source_records)

    print("\n解析完了: " + str(result_directory))
    print("推奨センサーモデル: " + str(result_directory / "recommended_sensor_model.csv"))


if __name__ == "__main__":
    main()
