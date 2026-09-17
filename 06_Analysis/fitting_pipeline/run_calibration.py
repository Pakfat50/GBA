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
PLOT_FREE_DECAY_FORCE = True
PLOT_SIMULATED_WIND_FORCE = True

# 外力グラフに表示する推定方式を選ぶ。
PLOT_ESO_ESTIMATE = True
PLOT_RTS_ESTIMATE = True

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

# 既存の摩擦オブザーバー解析で選択した設定。
ESTIMATOR_SETTINGS = {
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

    # 数値列を明示的に数値へ変換する。空欄や文字列はエラーとして検出する。
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

    valid_rows = manifest[manifest["valid"] == 1].copy()
    for column in numeric_columns:
        if valid_rows[column].isna().any():
            bad = valid_rows[valid_rows[column].isna()]["data_file"].tolist()
            raise ValueError(column + " が未記入です: " + ", ".join(bad))
    return manifest, manifest_path


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
            }
            segment_rows.append(segment_row)

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
            }
            waveform_records.append(record)

        if positive_count < 3 or negative_count < 3:
            message = "警告: " + data_file + " で必要な自由減衰を検出できませんでした。"
            message += " P45=" + str(positive_count) + ", N45=" + str(negative_count)
            print(message)

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
        base_rows.append(
            {
                "axis": axis,
                "base_inertia_kg_m2": base_inertia,
                "base_restoring_n_m_per_rad": base_restoring,
                "calibration_levels": len(calibration_rows),
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


def plot_waveform_fits(waveform_records, result_directory):
    """構成×方向ごとに3反復の測定波形とフィットを重ねて表示する。"""

    axes = sorted(set(record["axis"] for record in waveform_records))
    for axis in axes:
        axis_records = [record for record in waveform_records if record["axis"] == axis]
        configurations = sorted(set(record["configuration"] for record in axis_records))
        figure, plot_axes = plt.subplots(
            len(configurations), 2, figsize=(13, 3.0 * len(configurations)), squeeze=False
        )
        row_index = 0
        while row_index < len(configurations):
            configuration = configurations[row_index]
            directions = ["P45", "N45"]
            column_index = 0
            while column_index < 2:
                direction = directions[column_index]
                plot_axis = plot_axes[row_index, column_index]
                for record in axis_records:
                    if record["configuration"] != configuration:
                        continue
                    if record["direction"] != direction:
                        continue
                    label = "Measured R" + str(record["repetition"])
                    plot_axis.plot(
                        record["time_s"], np.rad2deg(record["angle_rad"]), lw=0.8, alpha=0.55, label=label
                    )
                    plot_axis.plot(
                        record["time_s"], np.rad2deg(record["fit_rad"]), lw=1.0, color="black", alpha=0.55
                    )
                plot_axis.set_title(configuration + " " + direction)
                plot_axis.set_xlabel("Time from release [s]")
                plot_axis.set_ylabel("Angle [deg]")
                plot_axis.grid(alpha=0.25)
                if row_index == 0 and column_index == 0:
                    plot_axis.legend(fontsize=7, ncol=2)
                column_index += 1
            row_index += 1
        figure.suptitle(axis + " measured free decay and fitted model")
        figure.tight_layout()
        figure.savefig(result_directory / ("waveform_fits_" + axis + ".png"), dpi=160)
        plt.close(figure)


def calculate_free_decay_forces(waveform_records, product_parameters, result_directory):
    """球あり自由減衰では真の外力0 NとしてESO・RTS推定を比較する。"""

    output_rows = []
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
            eso_force = estimate_force_eso(angle_rad, dt, parameters, ESTIMATOR_SETTINGS)
            rts_force = estimate_force_rts(angle_rad, dt, parameters, ESTIMATOR_SETTINGS)
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
            plot_axis.axhline(0.0, color="black", lw=1.2, label="True external force")
            if PLOT_ESO_ESTIMATE:
                plot_axis.plot(relative_time, 1000.0 * eso_force, label="ESO estimate", lw=1.0)
            if PLOT_RTS_ESTIMATE:
                plot_axis.plot(relative_time, 1000.0 * rts_force, label="RTS estimate", lw=1.0)
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


def calculate_simulated_wind_forces(product_parameters, result_directory):
    """同定済み係数で想定風をシミュレーションし、外力推定値と比較する。"""

    time_s, wind_speed, true_force = create_kaimal_wind(WIND_SETTINGS)
    dt = 1.0 / WIND_SETTINGS["sample_rate_hz"]
    output_rows = []
    for axis in sorted(product_parameters):
        parameters = product_parameters[axis]
        states = simulate_forced_motion(true_force, dt, parameters)
        angle_rad = states[:, 0]
        eso_force = estimate_force_eso(angle_rad, dt, parameters, ESTIMATOR_SETTINGS)
        rts_force = estimate_force_rts(angle_rad, dt, parameters, ESTIMATOR_SETTINGS)

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
            mask = (time_s >= WIND_SETTINGS["evaluation_start_s"]) & (
                time_s <= WIND_SETTINGS["evaluation_end_s"]
            )
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


def save_provenance(date_directory, manifest, manifest_path, result_directory, product_parameters):
    """入力・設定・同定結果の来歴をJSONへ保存する。"""

    input_files = []
    for unused_index, row in manifest.iterrows():
        if int(row["valid"]) != 1:
            continue
        path = date_directory / str(row["data_file"])
        input_files.append({"path": str(path), "sha256": file_sha256(path)})
    provenance = {
        "input_directory": str(date_directory),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "input_files": input_files,
        "plot_switches": {
            "waveform_fits": PLOT_WAVEFORM_FITS,
            "free_decay_force": PLOT_FREE_DECAY_FORCE,
            "simulated_wind_force": PLOT_SIMULATED_WIND_FORCE,
            "eso": PLOT_ESO_ESTIMATE,
            "rts": PLOT_RTS_ESTIMATE,
        },
        "segment_settings": SEGMENT_SETTINGS,
        "estimator_settings": {
            "eso_pole_hz": ESTIMATOR_SETTINGS["eso_pole_hz"],
            "rts_force_random_walk_n_per_sample": ESTIMATOR_SETTINGS[
                "rts_force_random_walk_n_per_sample"
            ],
            "angle_noise_deg": float(np.rad2deg(ESTIMATOR_SETTINGS["angle_noise_rad"])),
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

    segment_rows, fit_rows, waveform_records = analyze_all_files(
        date_directory, manifest, result_directory
    )
    unused_base, unused_parameters, product_parameters = identify_physical_parameters(
        manifest, fit_rows, result_directory
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
