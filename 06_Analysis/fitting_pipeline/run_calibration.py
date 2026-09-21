"""真鍮スペーサ試験を自動解析する実行スクリプト。

使い方:
    python 06_Analysis/fitting_pipeline/run_calibration.py
    python 06_Analysis/fitting_pipeline/run_calibration.py --date 20260916

Pythonに不慣れな方が変更しやすいよう、通常変更する値はこの直後にまとめる。
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fitting_tools import component_increments
from fitting_tools import create_kaimal_wind
from fitting_tools import ensure_directory
from fitting_tools import estimate_force_eso
from fitting_tools import estimate_force_rts
from fitting_tools import fit_one_decay
from fitting_tools import fit_quality_statistics
from fitting_tools import identify_base_inertia_and_restoring
from fitting_tools import read_angle_log
from fitting_tools import resample_uniform
from fitting_tools import save_csv
from fitting_tools import simulate_forced_motion
from fitting_tools import split_free_decay


# =============================================================================
# 利用者が通常変更する設定
# =============================================================================

# 作成するグラフを True / False で選ぶ。
PLOT_WAVEFORM_FITS = True
PLOT_CALIBRATION_FIT = True
PLOT_FREE_DECAY_FORCE = True
PLOT_SIMULATED_WIND_FORCE = True

# 外力グラフに表示する推定方式を選ぶ。
PLOT_ESO_ESTIMATE = True
PLOT_RTS_ESTIMATE = True

# オブザーバー初期化直後は過渡誤差が出るため、自由減衰グラフではこの時間を除く。
# CSVには初期化直後を含む全サンプルを保存する。
FREE_DECAY_FORCE_PLOT_WARMUP_S = 1.0

# 自由減衰モデルの符号反転を滑らかにする幅。
FRICTION_EPSILON_DEG_S = 0.5

# 波形フィットの計算量に関する設定。
FIT_TARGET_RATE_HZ = 25.0
FIT_MAX_FUNCTION_EVALUATIONS = 180

# 自動分割条件。実データで分割できない場合はここを調整する。
SEGMENT_SETTINGS = {
    "smooth_time_s": 0.15,
    "start_angle_min_deg": 35.0,
    "hold_duration_s": 1.5,
    "hold_std_max_deg": 0.35,
    "hold_speed_max_deg_s": 3.0,
    "release_drop_deg": 0.7,
    "minimum_decay_time_s": 5.0,
    "maximum_decay_time_s": 90.0,
    "settle_duration_s": 5.0,
    "settle_center_max_deg": 3.0,
    "settle_std_max_deg": 0.25,
}

# 自由減衰は「真の外力は0 N」を確認する試験である。
# RTSの外力変化幅を小さくし、角度ノイズを外力と誤認しにくくする。
FREE_DECAY_ESTIMATOR_SETTINGS = {
    # 外力0 Nの確認では、実測角度ノイズを外力と誤認しないよう
    # 変動風用より強く平滑化する。
    "eso_pole_hz": 1.0,
    "rts_force_random_walk_n_per_sample": 1e-7,
    "angle_noise_rad": np.deg2rad(0.02),
    "epsilon_rad_s": np.deg2rad(FRICTION_EPSILON_DEG_S),
}

# 変動風の外力変化に追従させる設定。RTSの値は、既存の摩擦オブザーバー
# 解析で風入力に対して選択した値を使う。
WIND_ESTIMATOR_SETTINGS = {
    "eso_pole_hz": 45.0,
    "rts_force_random_walk_n_per_sample": 30.0,
    "angle_noise_rad": np.deg2rad(0.02),
    "epsilon_rad_s": np.deg2rad(FRICTION_EPSILON_DEG_S),
}

# 既存のKaimal想定風条件。
WIND_SETTINGS = {
    "sample_rate_hz": 100.0,
    "duration_s": 180.0,
    "mean_speed_m_s": 3.75,
    "maximum_speed_m_s": 6.0,
    "turbulence_intensity": 0.2,
    "integral_scale_m": 11.34,
    "air_density_kg_m3": 1.225,
    "drag_coefficient": 0.55,
    "projected_area_m2": 0.007853981633974483,
    "seed": 20260915,
    "evaluation_start_s": 30.0,
    "evaluation_end_s": 150.0,
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

# 錘計測結果.xlsx と試験治具寸法から補う物理量。
# 元の test_manifest.csv は変更せず、補完後の値を resolved_manifest.csv に保存する。
SPACER_MASS_KG = 0.00118
SPACER_OUTER_RADIUS_M = 0.004
SPACER_INNER_RADIUS_M = 0.00265
SPACER_LENGTH_M = 0.005
BALL_MASS_KG = 0.0039
BALL_RADIUS_M = 0.05
ROD_END_RADIUS_M = 0.229
BALL_NEAR_EDGE_INWARD_M = 0.005
BALL_COM_RADIUS_M = -(
    ROD_END_RADIUS_M - BALL_NEAR_EDGE_INWARD_M - BALL_RADIUS_M
)
FORCE_LEVER_M = abs(BALL_COM_RADIUS_M)


def parse_arguments():
    """コマンドライン引数を読み取る。"""

    parser = argparse.ArgumentParser(description="真鍮スペーサ自由減衰の自動解析")
    parser.add_argument("--date", help="使用する試験日フォルダ。例: 20260916")
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
        template = SCRIPT_DIRECTORY / "templates" / "test_manifest_template.csv"
        message = str(manifest_path) + " がありません。\n"
        message += str(template) + " をコピーして測定値を記入してください。"
        raise FileNotFoundError(message)

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required_columns = [
        "data_file",
        "axis",
        "angle_column",
        "configuration",
        "spacer_count",
        "component_mass_kg",
        "signed_com_radius_m",
        "component_centroid_inertia_kg_m2",
        "force_lever_m",
        "use_for_calibration",
        "valid",
        "notes",
    ]
    for column in required_columns:
        if column not in manifest.columns:
            raise ValueError("test_manifest.csv に " + column + " 列がありません")

    # 数値列を明示的に数値へ変換する。物理量の空欄は、この後で計測表から補う。
    numeric_columns = [
        "spacer_count",
        "component_mass_kg",
        "signed_com_radius_m",
        "component_centroid_inertia_kg_m2",
        "force_lever_m",
        "use_for_calibration",
        "valid",
    ]
    for column in numeric_columns:
        manifest[column] = pd.to_numeric(manifest[column], errors="coerce")

    return manifest, manifest_path


def one_spacer_centroid_inertia():
    """円筒スペーサ1個の重心を通る横軸まわり慣性を返す。"""

    radius_term = SPACER_OUTER_RADIUS_M * SPACER_OUTER_RADIUS_M
    radius_term += SPACER_INNER_RADIUS_M * SPACER_INNER_RADIUS_M
    length_term = SPACER_LENGTH_M * SPACER_LENGTH_M
    return SPACER_MASS_KG / 12.0 * (3.0 * radius_term + length_term)


def spacer_group_properties(spacer_count):
    """複数スペーサの合計質量、合成重心位置、重心慣性を返す。"""

    if spacer_count == 0:
        return 0.0, 0.0, 0.0

    # 指定位置は各スペーサのロッド端面側の端点である。重心はそこから
    # さらに長さの半分だけ支柱側にある。
    radii = []
    spacer_number = 1
    while spacer_number <= spacer_count:
        rod_side_edge = ROD_END_RADIUS_M - spacer_number * SPACER_LENGTH_M
        distance_from_pivot = rod_side_edge - 0.5 * SPACER_LENGTH_M
        radii.append(-distance_from_pivot)
        spacer_number += 1

    total_mass = spacer_count * SPACER_MASS_KG
    group_radius = float(np.mean(radii))
    centroid_inertia = spacer_count * one_spacer_centroid_inertia()
    for radius in radii:
        centroid_inertia += SPACER_MASS_KG * (radius - group_radius) ** 2
    return total_mass, group_radius, centroid_inertia


def resolve_physical_inputs(manifest, result_directory):
    """計測表と既知寸法から、空欄の質量・取付半径・慣性を補う。"""

    resolved = manifest.copy()
    source_rows = []
    for row_index, row in resolved.iterrows():
        configuration = str(row["configuration"]).strip().upper()
        spacer_count = int(row["spacer_count"])

        if configuration.startswith("SP"):
            mass, radius, centroid_inertia = spacer_group_properties(spacer_count)
            source = "錘計測結果.xlsx（1個1.18 g）とロッド端面側端点5 mm刻み"
        elif configuration == "BALL":
            mass = BALL_MASS_KG
            radius = BALL_COM_RADIUS_M
            centroid_inertia = 2.0 / 5.0 * mass * BALL_RADIUS_M * BALL_RADIUS_M
            source = "錘計測結果.xlsx（3.9 g）と球表面224 mm・直径100 mm"
        else:
            raise ValueError("未対応の装置形態です: " + configuration)

        resolved.at[row_index, "component_mass_kg"] = mass
        resolved.at[row_index, "signed_com_radius_m"] = radius
        resolved.at[row_index, "component_centroid_inertia_kg_m2"] = centroid_inertia
        resolved.at[row_index, "force_lever_m"] = FORCE_LEVER_M
        source_rows.append(
            {
                "data_file": row["data_file"],
                "axis": row["axis"],
                "configuration": configuration,
                "component_mass_kg": mass,
                "signed_com_radius_m": radius,
                "component_centroid_inertia_kg_m2": centroid_inertia,
                "force_lever_m": FORCE_LEVER_M,
                "source_or_assumption": source,
            }
        )

    valid_rows = resolved[resolved["valid"] == 1]
    physical_columns = [
        "component_mass_kg",
        "signed_com_radius_m",
        "component_centroid_inertia_kg_m2",
        "force_lever_m",
    ]
    for column in physical_columns:
        if valid_rows[column].isna().any():
            raise ValueError(column + " を補完できませんでした")

    resolved.to_csv(
        result_directory / "resolved_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_csv(result_directory / "physical_input_sources.csv", source_rows)
    return resolved


def read_waveform_selection(result_directory):
    """確認済みの波形採否表を読み込む。"""

    selection_path = result_directory / "waveform_review" / SELECTION_FILE_NAME
    if not selection_path.exists():
        return None, selection_path
    selection = pd.read_csv(selection_path, encoding="utf-8-sig")
    required_columns = [
        "segment_id",
        "data_file",
        "axis",
        "angle_column",
        "configuration",
        "direction",
        "repetition",
        "start_index",
        "end_index",
        "use_for_fitting",
        "review_status",
    ]
    for column in required_columns:
        if column not in selection.columns:
            raise ValueError("waveform_selection.csv に " + column + " 列がありません")
    if not (selection["review_status"].astype(str).str.upper() == "APPROVED").all():
        raise ValueError("waveform_selection.csv に未確認の波形があります")
    selection["use_for_fitting"] = pd.to_numeric(
        selection["use_for_fitting"], errors="raise"
    ).astype(int)
    return selection, selection_path


def make_segment_identifier(axis, configuration, direction, repetition):
    """解析結果で使用する短い波形識別子を作る。"""

    return axis + "_" + configuration + "_" + direction + "_R" + str(repetition).zfill(2)


def analyze_all_files(date_directory, manifest, result_directory):
    """全ログを分割し、各自由減衰波形へ正規化係数をフィットする。"""

    segment_rows = []
    fit_rows = []
    waveform_records = []
    epsilon = np.deg2rad(FRICTION_EPSILON_DEG_S)

    for unused_index, manifest_row in manifest.iterrows():
        if int(manifest_row["valid"]) != 1:
            continue
        data_file = str(manifest_row["data_file"])
        file_path = date_directory / data_file
        if not file_path.exists():
            raise FileNotFoundError("データファイルがありません: " + str(file_path))

        axis = str(manifest_row["axis"]).strip().upper()
        configuration = str(manifest_row["configuration"]).strip().upper()
        angle_column = str(manifest_row["angle_column"]).strip()
        print("読込・分割: " + data_file)
        time_s, angle_rad = read_angle_log(file_path, angle_column)
        segments = split_free_decay(time_s, angle_rad, SEGMENT_SETTINGS)

        positive_count = 0
        negative_count = 0
        valid_positive_count = 0
        valid_negative_count = 0
        for segment in segments:
            if segment["direction"] == "P45":
                positive_count += 1
            else:
                negative_count += 1

            # 正負それぞれ最初の3回を解析する。追加試験は元ログに残るが使用しない。
            if segment["repetition"] > 3:
                continue
            segment_id = make_segment_identifier(
                axis,
                configuration,
                segment["direction"],
                segment["repetition"],
            )
            start = segment["start_index"]
            end = segment["end_index"] + 1
            segment_time = time_s[start:end]
            segment_angle = angle_rad[start:end]

            segment_row = {
                "segment_id": segment_id,
                "data_file": data_file,
                "axis": axis,
                "configuration": configuration,
                "direction": segment["direction"],
                "repetition": segment["repetition"],
                "start_index": start,
                "end_index": end - 1,
                "start_time_s": float(segment_time[0]),
                "end_time_s": float(segment_time[-1]),
                "plateau_angle_deg": segment["plateau_angle_deg"],
                "settled_automatically": segment["settled_automatically"],
                "valid": int(segment["settled_automatically"]),
                "quality_note": "" if segment["settled_automatically"] else "静止終了を自動検出できない",
            }
            segment_rows.append(segment_row)

            # 次の試験準備動作が混ざった波形は係数を大きく誤らせる。
            # そのため元データと分割記録は残すが、自動フィットには使用しない。
            if not segment["settled_automatically"]:
                print("  除外: " + segment_id + "（静止終了を検出できません）")
                continue
            if segment["direction"] == "P45":
                valid_positive_count += 1
            else:
                valid_negative_count += 1

            print("  フィット: " + segment_id)
            fit = fit_one_decay(
                segment_time,
                segment_angle,
                epsilon,
                FIT_TARGET_RATE_HZ,
                FIT_MAX_FUNCTION_EVALUATIONS,
            )
            fit_row = dict(segment_row)
            for key in [
                "k_over_i_per_s2",
                "b_over_i_per_s",
                "c_over_i_per_rad",
                "tau_over_i_rad_s2",
                "offset_deg",
                "initial_angle_deg",
                "initial_speed_rad_s",
                "rmse_deg",
                "r_value",
                "r_squared",
                "success",
                "message",
            ]:
                fit_row[key] = fit[key]
            fit_rows.append(fit_row)

            record = {
                "segment_id": segment_id,
                "axis": axis,
                "configuration": configuration,
                "direction": segment["direction"],
                "repetition": segment["repetition"],
                "time_s": segment_time - segment_time[0],
                "angle_rad": segment_angle,
                "fit_rad": fit["prediction_rad"],
                "rmse_deg": fit["rmse_deg"],
                "r_value": fit["r_value"],
                "r_squared": fit["r_squared"],
            }
            waveform_records.append(record)

        if valid_positive_count < 3 or valid_negative_count < 3:
            message = "警告: " + data_file + " で必要な自由減衰を検出できませんでした。"
            message += " 有効P45=" + str(valid_positive_count)
            message += ", 有効N45=" + str(valid_negative_count)
            message += "（検出総数 P45=" + str(positive_count)
            message += ", N45=" + str(negative_count) + "）"
            print(message)

    save_csv(result_directory / "segments.csv", segment_rows)
    save_csv(result_directory / "segment_fits.csv", fit_rows)
    return segment_rows, fit_rows, waveform_records


def analyze_selected_waveforms(date_directory, selection, result_directory):
    """確認済みCSVの一行を一波形として読み込み、個別にフィットする。"""

    segment_rows = []
    fit_rows = []
    waveform_records = []
    log_cache = {}
    epsilon = np.deg2rad(FRICTION_EPSILON_DEG_S)

    for unused_index, selection_row in selection.iterrows():
        data_file = str(selection_row["data_file"])
        angle_column = str(selection_row["angle_column"])
        cache_key = data_file + "|" + angle_column
        if cache_key not in log_cache:
            file_path = date_directory / data_file
            if not file_path.exists():
                raise FileNotFoundError("データファイルがありません: " + str(file_path))
            log_cache[cache_key] = read_angle_log(file_path, angle_column)

        time_s, angle_rad = log_cache[cache_key]
        start = int(selection_row["start_index"])
        end = int(selection_row["end_index"]) + 1
        if start < 0 or end > len(time_s) or end <= start:
            raise ValueError("波形範囲が不正です: " + str(selection_row["segment_id"]))

        segment_row = {
            "segment_id": str(selection_row["segment_id"]),
            "data_file": data_file,
            "axis": str(selection_row["axis"]).strip().upper(),
            "configuration": str(selection_row["configuration"]).strip().upper(),
            "direction": str(selection_row["direction"]).strip().upper(),
            "repetition": int(selection_row["repetition"]),
            "start_index": start,
            "end_index": end - 1,
            "start_time_s": float(time_s[start]),
            "end_time_s": float(time_s[end - 1]),
            "valid": int(selection_row["use_for_fitting"]),
            "quality_note": str(selection_row.get("auto_quality_note", "")),
        }
        segment_rows.append(segment_row)

        if int(selection_row["use_for_fitting"]) != 1:
            print("  除外: " + segment_row["segment_id"])
            continue

        segment_time = time_s[start:end]
        segment_angle = angle_rad[start:end]
        print("  フィット: " + segment_row["segment_id"])
        fit = fit_one_decay(
            segment_time,
            segment_angle,
            epsilon,
            FIT_TARGET_RATE_HZ,
            FIT_MAX_FUNCTION_EVALUATIONS,
        )

        fit_row = dict(segment_row)
        for key in [
            "k_over_i_per_s2",
            "b_over_i_per_s",
            "c_over_i_per_rad",
            "tau_over_i_rad_s2",
            "offset_deg",
            "initial_angle_deg",
            "initial_speed_rad_s",
            "rmse_deg",
            "r_value",
            "r_squared",
            "success",
            "message",
        ]:
            fit_row[key] = fit[key]
        fit_rows.append(fit_row)

        waveform_records.append(
            {
                "segment_id": segment_row["segment_id"],
                "axis": segment_row["axis"],
                "configuration": segment_row["configuration"],
                "direction": segment_row["direction"],
                "repetition": segment_row["repetition"],
                "time_s": segment_time - segment_time[0],
                "angle_rad": segment_angle,
                "fit_rad": fit["prediction_rad"],
                "rmse_deg": fit["rmse_deg"],
                "r_value": fit["r_value"],
                "r_squared": fit["r_squared"],
            }
        )

    save_csv(result_directory / "segments.csv", segment_rows)
    save_csv(result_directory / "segment_fits.csv", fit_rows)
    return segment_rows, fit_rows, waveform_records


def make_level_rows(axis, manifest, fit_table):
    """軸ごと・装置構成ごとのK/I平均値と追加量をまとめる。"""

    rows = []
    axis_manifest = manifest[(manifest["axis"].str.upper() == axis) & (manifest["valid"] == 1)]
    for unused_index, manifest_row in axis_manifest.iterrows():
        configuration = str(manifest_row["configuration"]).strip().upper()
        matching = fit_table[
            (fit_table["axis"] == axis)
            & (fit_table["configuration"] == configuration)
            & (fit_table["success"] == 1)
        ]
        if len(matching) == 0:
            continue
        values = matching["k_over_i_per_s2"].to_numpy(dtype=float)
        uncertainty = 0.001 * float(np.mean(values))
        if len(values) >= 2:
            uncertainty = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            uncertainty = max(uncertainty, 0.001 * float(np.mean(values)))
        delta_inertia, delta_restoring = component_increments(manifest_row)
        row = {
            "axis": axis,
            "configuration": configuration,
            "spacer_count": int(manifest_row["spacer_count"]),
            "use_for_calibration": int(manifest_row["use_for_calibration"]),
            "number_of_segments": int(len(values)),
            "k_over_i_mean": float(np.mean(values)),
            "k_over_i_std": float(np.std(values, ddof=1)) if len(values) >= 2 else 0.0,
            "k_over_i_uncertainty": uncertainty,
            "delta_inertia": float(delta_inertia),
            "delta_restoring": float(delta_restoring),
            "force_lever_m": float(manifest_row["force_lever_m"]),
        }
        rows.append(row)
    return rows


def identify_physical_parameters(manifest, fit_rows, result_directory):
    """複数水準のK/IからIとKを求め、b,c,tau_fを絶対値へ変換する。"""

    fit_table = pd.DataFrame(fit_rows)
    if len(fit_table) == 0:
        raise RuntimeError("係数フィット結果がありません")
    axes = sorted(fit_table["axis"].unique().tolist())
    all_level_rows = []
    base_rows = []
    parameter_rows = []
    product_parameters = {}

    for axis in axes:
        level_rows = make_level_rows(axis, manifest, fit_table)
        calibration_rows = []
        for row in level_rows:
            if row["use_for_calibration"] == 1:
                calibration_rows.append(row)
        base_inertia, base_restoring, checked_rows = identify_base_inertia_and_restoring(
            calibration_rows
        )
        measured_q = np.asarray(
            [row["k_over_i_mean"] for row in checked_rows], dtype=float
        )
        predicted_q = np.asarray(
            [row["k_over_i_predicted"] for row in checked_rows], dtype=float
        )
        calibration_rmse, calibration_r, calibration_r_squared = (
            fit_quality_statistics(measured_q, predicted_q)
        )
        base_rows.append(
            {
                "axis": axis,
                "base_inertia_kg_m2": base_inertia,
                "base_restoring_n_m_per_rad": base_restoring,
                "calibration_levels": len(calibration_rows),
                "calibration_rmse_per_s2": calibration_rmse,
                "calibration_r": calibration_r,
                "calibration_r_squared": calibration_r_squared,
            }
        )

        for row in level_rows:
            checked = dict(row)
            checked["k_over_i_predicted"] = ""
            checked["k_over_i_residual"] = ""
            for calibration_row in checked_rows:
                if calibration_row["configuration"] == row["configuration"]:
                    checked = calibration_row
            all_level_rows.append(checked)

            configuration = row["configuration"]
            matching = fit_table[
                (fit_table["axis"] == axis)
                & (fit_table["configuration"] == configuration)
                & (fit_table["success"] == 1)
            ]
            restoring = base_restoring + row["delta_restoring"]
            geometric_inertia = base_inertia + row["delta_inertia"]
            inertia = geometric_inertia
            # 球ありは空気付加慣性を含む実効IをK/(K/I)から求める。
            if configuration == "BALL":
                inertia = restoring / row["k_over_i_mean"]

            damping_values = matching["b_over_i_per_s"].to_numpy(dtype=float) * inertia
            quadratic_values = matching["c_over_i_per_rad"].to_numpy(dtype=float) * inertia
            friction_values = matching["tau_over_i_rad_s2"].to_numpy(dtype=float) * inertia
            parameter_row = {
                "axis": axis,
                "configuration": configuration,
                "inertia_kg_m2": inertia,
                "geometric_inertia_kg_m2": geometric_inertia,
                "restoring_n_m_per_rad": restoring,
                "damping_n_m_s_per_rad": float(np.mean(damping_values)),
                "quadratic_n_m_s2_per_rad2": float(np.mean(quadratic_values)),
                "friction_n_m": float(np.mean(friction_values)),
                "force_lever_m": row["force_lever_m"],
                "k_over_i_per_s2": row["k_over_i_mean"],
                "b_over_i_per_s": float(np.mean(matching["b_over_i_per_s"])),
                "c_over_i_per_rad": float(np.mean(matching["c_over_i_per_rad"])),
                "tau_over_i_rad_s2": float(np.mean(matching["tau_over_i_rad_s2"])),
                "segments": len(matching),
            }
            parameter_rows.append(parameter_row)
            if configuration == "BALL":
                product = dict(parameter_row)
                product["epsilon_rad_s"] = np.deg2rad(FRICTION_EPSILON_DEG_S)
                product_parameters[axis] = product

    save_csv(result_directory / "calibration_levels.csv", all_level_rows)
    save_csv(result_directory / "base_parameters.csv", base_rows)
    save_csv(result_directory / "identified_parameters.csv", parameter_rows)
    return base_rows, parameter_rows, product_parameters


def plot_calibration_fit(level_rows, base_rows, output_path):
    """スペーサ較正の実測値、最小二乗予測、残差と適合度を描く。"""

    level_table = pd.DataFrame(level_rows)
    base_table = pd.DataFrame(base_rows)
    axes = sorted(base_table["axis"].tolist())
    figure, plot_axes = plt.subplots(
        2, len(axes), figsize=(7.0 * len(axes), 8.0), squeeze=False
    )

    for axis_number, axis in enumerate(axes):
        levels = level_table[
            (level_table["axis"] == axis)
            & (level_table["use_for_calibration"] == 1)
        ].copy()
        levels = levels.sort_values("spacer_count")
        base = base_table[base_table["axis"] == axis].iloc[0]
        spacer_count = levels["spacer_count"].to_numpy(dtype=float)
        measured = levels["k_over_i_mean"].to_numpy(dtype=float)
        predicted = levels["k_over_i_predicted"].to_numpy(dtype=float)
        uncertainty = levels["k_over_i_uncertainty"].to_numpy(dtype=float)
        residual = predicted - measured

        fit_axis = plot_axes[0, axis_number]
        fit_axis.errorbar(
            spacer_count,
            measured,
            yerr=uncertainty,
            fmt="o",
            capsize=4,
            color="#1f77b4",
            label="Measured mean ± uncertainty",
        )
        fit_axis.plot(
            spacer_count,
            predicted,
            "--o",
            color="#d62728",
            label="Least-squares prediction",
        )
        annotation = "I0=" + format(base["base_inertia_kg_m2"], ".6e") + " kg m²"
        annotation += "\nK0=" + format(base["base_restoring_n_m_per_rad"], ".6e") + " N m/rad"
        annotation += "\nR=" + format(base["calibration_r"], ".6f")
        annotation += "   R²=" + format(base["calibration_r_squared"], ".6f")
        annotation += "\nRMSE=" + format(base["calibration_rmse_per_s2"], ".4f") + " s⁻²"
        fit_axis.text(
            0.98,
            0.97,
            annotation,
            transform=fit_axis.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
        )
        fit_axis.set_title(axis + " axis calibration")
        fit_axis.set_ylabel("K/I [s⁻²]")
        fit_axis.grid(alpha=0.25)
        fit_axis.legend()

        residual_axis = plot_axes[1, axis_number]
        residual_axis.axhline(0.0, color="black", lw=0.8)
        residual_axis.bar(spacer_count, residual, color="#7f7f7f", width=0.55)
        residual_axis.set_xlabel("Spacer count")
        residual_axis.set_ylabel("Predicted - measured [s⁻²]")
        residual_axis.set_xticks(spacer_count)
        residual_axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Base inertia and restoring coefficient calibration")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_waveform_fit(plot_axis, record, show_legend):
    """一つの実測波形とフィット波形を同じ領域へ描く。"""

    plot_axis.plot(
        record["time_s"],
        np.rad2deg(record["angle_rad"]),
        color="#1f77b4",
        lw=0.9,
        label="Measured",
    )
    plot_axis.plot(
        record["time_s"],
        np.rad2deg(record["fit_rad"]),
        color="#d62728",
        lw=1.0,
        linestyle="--",
        label="Fitted model",
    )
    title = record["segment_id"] + "  RMSE=" + format(record["rmse_deg"], ".3f") + " deg"
    title += "  R=" + format(record["r_value"], ".5f")
    title += "  R²=" + format(record["r_squared"], ".5f")
    plot_axis.set_title(title, fontsize=9)
    plot_axis.set_xlabel("Time from release [s]")
    plot_axis.set_ylabel("Angle [deg]")
    plot_axis.set_ylim(-75.0, 75.0)
    plot_axis.grid(alpha=0.25)
    if show_legend:
        plot_axis.legend(fontsize=8)


def save_waveform_fit_overview(records, output_path, title, column_count):
    """指定した波形群を一覧図として保存する。"""

    if len(records) == 0:
        return
    row_count = int(np.ceil(len(records) / float(column_count)))
    figure, plot_axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.6 * column_count, 3.0 * row_count),
        squeeze=False,
    )
    plot_number = 0
    while plot_number < row_count * column_count:
        row_number = plot_number // column_count
        column_number = plot_number % column_count
        plot_axis = plot_axes[row_number, column_number]
        if plot_number < len(records):
            draw_waveform_fit(plot_axis, records[plot_number], plot_number == 0)
        else:
            plot_axis.axis("off")
        plot_number += 1
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def plot_waveform_fits(waveform_records, result_directory):
    """全採用波形について実測値とフィットを重ねた図を保存する。"""

    records = sorted(waveform_records, key=lambda item: item["segment_id"])
    save_waveform_fit_overview(
        records,
        result_directory / "waveform_fits_all.png",
        "All approved free-decay waveforms and fitted models",
        4,
    )

    group_keys = []
    for record in records:
        key = record["axis"] + "_" + record["configuration"]
        if key not in group_keys:
            group_keys.append(key)
    group_keys.sort()
    for key in group_keys:
        group_records = []
        for record in records:
            if record["axis"] + "_" + record["configuration"] == key:
                group_records.append(record)
        save_waveform_fit_overview(
            group_records,
            result_directory / ("waveform_fits_" + key + ".png"),
            key + " measured waveforms and fitted models",
            3,
        )

    individual_directory = result_directory / "waveform_fits"
    ensure_directory(individual_directory)
    individual_paths = []
    for record in records:
        figure, plot_axis = plt.subplots(1, 1, figsize=(11, 4.5))
        draw_waveform_fit(plot_axis, record, True)
        figure.tight_layout()
        output_path = individual_directory / (record["segment_id"] + ".png")
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        individual_paths.append(output_path)

    zip_path = result_directory / "waveform_fits_individual.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for output_path in individual_paths:
            archive.write(output_path, arcname=output_path.name)


def calculate_free_decay_forces(waveform_records, product_parameters, result_directory):
    """球あり自由減衰では真の外力0 NとしてESO・RTS推定を比較する。"""

    output_rows = []
    metric_rows = []
    for axis in sorted(product_parameters):
        parameters = product_parameters[axis]
        selected_records = []
        for record in waveform_records:
            if record["axis"] != axis or record["configuration"] != "BALL":
                continue
            if record["repetition"] == 1:
                selected_records.append(record)
        selected_records.sort(key=lambda item: item["direction"], reverse=True)

        if len(selected_records) == 0:
            print("警告: " + axis + " のBALL波形がないため自由減衰外力を省略します")
            continue
        figure, plot_axes = plt.subplots(len(selected_records), 1, figsize=(12, 4 * len(selected_records)), squeeze=False)
        record_index = 0
        while record_index < len(selected_records):
            record = selected_records[record_index]
            time_s, angle_rad, dt = resample_uniform(record["time_s"], record["angle_rad"])
            eso_force = estimate_force_eso(
                angle_rad, dt, parameters, FREE_DECAY_ESTIMATOR_SETTINGS
            )
            rts_force = estimate_force_rts(
                angle_rad, dt, parameters, FREE_DECAY_ESTIMATOR_SETTINGS
            )
            sample_index = 0
            while sample_index < len(time_s):
                output_rows.append(
                    {
                        "axis": axis,
                        "segment_id": record["segment_id"],
                        "time_s": float(time_s[sample_index] - time_s[0]),
                        "true_force_n": 0.0,
                        "eso_force_n": float(eso_force[sample_index]),
                        "rts_force_n": float(rts_force[sample_index]),
                    }
                )
                sample_index += 1

            plot_axis = plot_axes[record_index, 0]
            relative_time = time_s - time_s[0]
            evaluation_mask = relative_time >= FREE_DECAY_FORCE_PLOT_WARMUP_S
            for method_name, estimate in [("ESO", eso_force), ("RTS", rts_force)]:
                selected = estimate[evaluation_mask]
                metric_rows.append(
                    {
                        "axis": axis,
                        "segment_id": record["segment_id"],
                        "method": method_name,
                        "warmup_s": FREE_DECAY_FORCE_PLOT_WARMUP_S,
                        "rmse_n": float(np.sqrt(np.mean(selected * selected))),
                        "bias_n": float(np.mean(selected)),
                        "max_abs_error_n": float(np.max(np.abs(selected))),
                    }
                )
            plot_axis.axhline(0.0, color="black", lw=1.2, label="True external force")
            if PLOT_ESO_ESTIMATE:
                plot_axis.plot(relative_time[evaluation_mask], 1000.0 * eso_force[evaluation_mask], label="ESO estimate", lw=1.0)
            if PLOT_RTS_ESTIMATE:
                plot_axis.plot(relative_time[evaluation_mask], 1000.0 * rts_force[evaluation_mask], label="RTS estimate", lw=1.0)
            plot_axis.set_title(axis + " " + record["direction"] + " free-decay external force")
            plot_axis.set_xlabel("Time from release [s]")
            plot_axis.set_ylabel("Force [mN]")
            plot_axis.grid(alpha=0.25)
            plot_axis.legend()
            record_index += 1
        figure.tight_layout()
        if PLOT_FREE_DECAY_FORCE:
            figure.savefig(result_directory / ("free_decay_force_" + axis + ".png"), dpi=170)
        plt.close(figure)

    save_csv(result_directory / "free_decay_force.csv", output_rows)
    save_csv(result_directory / "free_decay_force_metrics.csv", metric_rows)


def calculate_simulated_wind_forces(product_parameters, result_directory):
    """同定済み係数で想定風をシミュレーションし、外力推定値と比較する。"""

    time_s, wind_speed, true_force = create_kaimal_wind(WIND_SETTINGS)
    dt = 1.0 / WIND_SETTINGS["sample_rate_hz"]
    output_rows = []
    metric_rows = []
    for axis in sorted(product_parameters):
        parameters = product_parameters[axis]
        states = simulate_forced_motion(true_force, dt, parameters)
        angle_rad = states[:, 0]
        eso_force = estimate_force_eso(
            angle_rad, dt, parameters, WIND_ESTIMATOR_SETTINGS
        )
        rts_force = estimate_force_rts(
            angle_rad, dt, parameters, WIND_ESTIMATOR_SETTINGS
        )

        evaluation_mask = (time_s >= WIND_SETTINGS["evaluation_start_s"]) & (
            time_s <= WIND_SETTINGS["evaluation_end_s"]
        )
        for method_name, estimate in [("ESO", eso_force), ("RTS", rts_force)]:
            error = estimate[evaluation_mask] - true_force[evaluation_mask]
            metric_rows.append(
                {
                    "axis": axis,
                    "method": method_name,
                    "evaluation_start_s": WIND_SETTINGS["evaluation_start_s"],
                    "evaluation_end_s": WIND_SETTINGS["evaluation_end_s"],
                    "rmse_n": float(np.sqrt(np.mean(error * error))),
                    "bias_n": float(np.mean(error)),
                    "max_abs_error_n": float(np.max(np.abs(error))),
                }
            )

        index = 0
        while index < len(time_s):
            output_rows.append(
                {
                    "axis": axis,
                    "time_s": float(time_s[index]),
                    "wind_speed_m_s": float(wind_speed[index]),
                    "angle_deg": float(np.rad2deg(angle_rad[index])),
                    "true_force_n": float(true_force[index]),
                    "eso_force_n": float(eso_force[index]),
                    "rts_force_n": float(rts_force[index]),
                }
            )
            index += 1

        if PLOT_SIMULATED_WIND_FORCE:
            mask = evaluation_mask
            figure, plot_axis = plt.subplots(figsize=(13, 5))
            plot_axis.plot(time_s[mask], 1000.0 * true_force[mask], color="black", lw=1.3, label="True force")
            if PLOT_ESO_ESTIMATE:
                plot_axis.plot(time_s[mask], 1000.0 * eso_force[mask], lw=0.9, label="ESO estimate")
            if PLOT_RTS_ESTIMATE:
                plot_axis.plot(time_s[mask], 1000.0 * rts_force[mask], lw=0.9, label="RTS estimate")
            plot_axis.set_title(axis + " force under assumed Kaimal wind")
            plot_axis.set_xlabel("Time [s]")
            plot_axis.set_ylabel("Force [mN]")
            plot_axis.grid(alpha=0.25)
            plot_axis.legend()
            figure.tight_layout()
            figure.savefig(result_directory / ("simulated_wind_force_" + axis + ".png"), dpi=170)
            plt.close(figure)

    save_csv(result_directory / "simulated_wind_force.csv", output_rows)
    save_csv(result_directory / "simulated_wind_force_metrics.csv", metric_rows)


def file_sha256(path):
    """再現性記録用にファイルのSHA-256を計算する。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def path_for_provenance(path):
    """リポジトリ内のパスは環境に依存しない相対パスで返す。"""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        # --data-rootでリポジトリ外を指定した場合は絶対パスを残す。
        return str(resolved)


def save_provenance(date_directory, manifest, manifest_path, result_directory, product_parameters):
    """入力・設定・同定結果の来歴をJSONへ保存する。"""

    input_files = []
    for unused_index, row in manifest.iterrows():
        if int(row["valid"]) != 1:
            continue
        path = date_directory / str(row["data_file"])
        input_files.append(
            {"path": path_for_provenance(path), "sha256": file_sha256(path)}
        )
    provenance = {
        "input_directory": path_for_provenance(date_directory),
        "manifest": path_for_provenance(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "input_files": input_files,
        "plot_switches": {
            "waveform_fits": PLOT_WAVEFORM_FITS,
            "calibration_fit": PLOT_CALIBRATION_FIT,
            "free_decay_force": PLOT_FREE_DECAY_FORCE,
            "simulated_wind_force": PLOT_SIMULATED_WIND_FORCE,
            "eso": PLOT_ESO_ESTIMATE,
            "rts": PLOT_RTS_ESTIMATE,
            "free_decay_plot_warmup_s": FREE_DECAY_FORCE_PLOT_WARMUP_S,
        },
        "segment_settings": SEGMENT_SETTINGS,
        "estimator_settings": {
            "eso_pole_hz": WIND_ESTIMATOR_SETTINGS["eso_pole_hz"],
            "free_decay_rts_force_random_walk_n_per_sample": (
                FREE_DECAY_ESTIMATOR_SETTINGS[
                    "rts_force_random_walk_n_per_sample"
                ]
            ),
            "wind_rts_force_random_walk_n_per_sample": WIND_ESTIMATOR_SETTINGS[
                "rts_force_random_walk_n_per_sample"
            ],
            "angle_noise_deg": float(
                np.rad2deg(WIND_ESTIMATOR_SETTINGS["angle_noise_rad"])
            ),
            "epsilon_deg_s": FRICTION_EPSILON_DEG_S,
        },
        "wind_settings": WIND_SETTINGS,
        "identified_product_parameters": product_parameters,
    }
    output_path = result_directory / "provenance.json"
    output_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    """全処理を順番に実行する。"""

    arguments = parse_arguments()
    date_directory = select_date_directory(arguments.data_root, arguments.date)
    result_directory = arguments.result_root / date_directory.name
    ensure_directory(result_directory)

    print("入力フォルダ: " + str(date_directory))
    print("出力フォルダ: " + str(result_directory))
    manifest, manifest_path = read_manifest(date_directory)
    manifest = resolve_physical_inputs(manifest, result_directory)
    selection, selection_path = read_waveform_selection(result_directory)

    if selection is None:
        print("確認済み波形表がないため、従来の自動分割を使用します")
        segment_rows, fit_rows, waveform_records = analyze_all_files(
            date_directory, manifest, result_directory
        )
    else:
        print("確認済み波形表: " + str(selection_path))
        segment_rows, fit_rows, waveform_records = analyze_selected_waveforms(
            date_directory, selection, result_directory
        )
    base_rows, unused_parameters, product_parameters = identify_physical_parameters(
        manifest, fit_rows, result_directory
    )

    if PLOT_CALIBRATION_FIT:
        calibration_levels = pd.read_csv(result_directory / "calibration_levels.csv")
        plot_calibration_fit(
            calibration_levels.to_dict(orient="records"),
            base_rows,
            result_directory / "calibration_fit.png",
        )

    if PLOT_WAVEFORM_FITS:
        plot_waveform_fits(waveform_records, result_directory)
    calculate_free_decay_forces(waveform_records, product_parameters, result_directory)
    calculate_simulated_wind_forces(product_parameters, result_directory)
    save_provenance(
        date_directory, manifest, manifest_path, result_directory, product_parameters
    )

    print("\n解析が完了しました。")
    print("分割波形数: " + str(len(segment_rows)))
    print("フィット成功数: " + str(sum(int(row["success"]) for row in fit_rows)))
    print("結果: " + str(result_directory))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nエラー: " + str(error), file=sys.stderr)
        raise
