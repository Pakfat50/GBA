"""既存の実測自由振動を元に、較正ツール用デモデータを作る。

実行例:
    python 06_Analysis/fitting_pipeline/demo/build_demo_data.py

出力先:
    04_Data/05_Fitting/20990101/

このデータは実測値そのものではない。旧ハードの実測ログから得た
減衰係数と角度ノイズを使い、真鍮スペーサ試験の形式へ再構成したデモである。
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PIPELINE_DIRECTORY = SCRIPT_DIRECTORY.parent
REPOSITORY_ROOT = PIPELINE_DIRECTORY.parents[1]
sys.path.insert(0, str(PIPELINE_DIRECTORY))

from fitting_tools import simulate_normalized_decay


# -----------------------------------------------------------------------------
# デモ生成に使う固定値
# -----------------------------------------------------------------------------

DEMO_DATE = "20990101"
SOURCE_LOG = REPOSITORY_ROOT / "04_Data" / "00_Calibration" / "swing" / "LOG00014.TXT"
SOURCE_WINDOWS_S = [(20.4, 32.0), (47.05, 59.05), (70.2, 82.0), (93.25, 105.25)]

SAMPLE_RATE_HZ = 25.0
HOLD_TIME_S = 2.5
DECAY_TIME_S = 30.0
REST_TIME_S = 6.0
FRICTION_EPSILON_RAD_S = np.deg2rad(0.5)
RANDOM_SEED = 20990101

# 既存の実測4区間から得た代表的な正規化係数。
# bとcは自由振動だけでは配分が強く相関するため、VCとVQCの両結果を半分ずつ使う。
SOURCE_K_OVER_I = 5.614039
SOURCE_B_OVER_I_VC = 0.082885
SOURCE_C_OVER_I_VQC = 0.063541
SOURCE_TAU_OVER_I = 0.116662

# 絶対係数を復元できることを確認するためのデモ真値。
BASE_INERTIA_KG_M2 = 0.0045
FORCE_LEVER_M = 0.17

# 真鍮円筒スペーサ: 外径8 mm、内径5.3 mm、長さ5 mm。
BRASS_DENSITY_KG_M3 = 8500.0
SPACER_OUTER_RADIUS_M = 0.004
SPACER_INNER_RADIUS_M = 0.00265
SPACER_LENGTH_M = 0.005
SPACER_COM_RADIUS_M = -0.18

# BALLは既存自由振動の動特性を対応付けるためのデモ仮定。
BALL_MASS_KG = 0.0043
BALL_COM_RADIUS_M = -0.17
BALL_CENTROID_INERTIA_KG_M2 = 4.3e-6


def parse_arguments():
    """出力先を必要に応じて変更できるようにする。"""

    parser = argparse.ArgumentParser(description="既存自由振動からデモ較正データを作成")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPOSITORY_ROOT / "04_Data" / "05_Fitting" / DEMO_DATE,
        help="デモCSVの出力先",
    )
    return parser.parse_args()


def calculate_component_increments(mass_kg, radius_m, centroid_inertia_kg_m2):
    """既知部品を追加したときの慣性と復元係数の変化を返す。"""

    delta_inertia = centroid_inertia_kg_m2 + mass_kg * radius_m * radius_m
    delta_restoring = mass_kg * 9.81 * radius_m
    return delta_inertia, delta_restoring


def read_source_noise(source_path):
    """既存ログから角度ノイズの時系列を4本取り出す。"""

    rows = []
    text_lines = source_path.read_text(encoding="utf-8").splitlines()
    for line in text_lines:
        fields = line.split("\t")
        if len(fields) != 4:
            continue
        try:
            time_s = float(fields[0]) * 1e-6
            adc_value = float(fields[2])
        except ValueError:
            continue
        angle_deg = -0.0803317 * adc_value + 120.248
        rows.append([time_s, angle_deg])

    raw = np.asarray(rows, dtype=float)
    if len(raw) == 0:
        raise ValueError("実測元ログから数値を読み込めません")

    noise_records = []
    target_time = np.arange(0.0, DECAY_TIME_S, 1.0 / SAMPLE_RATE_HZ)
    for start_s, end_s in SOURCE_WINDOWS_S:
        selected = raw[(raw[:, 0] >= start_s) & (raw[:, 0] <= end_s)]
        relative_time = selected[:, 0] - selected[0, 0]
        source_duration = min(DECAY_TIME_S, float(relative_time[-1]))
        source_target_time = np.arange(
            0.0, source_duration, 1.0 / SAMPLE_RATE_HZ
        )
        resampled_angle = np.interp(
            source_target_time, relative_time, selected[:, 1]
        )

        # 0.21秒程度の滑らかな曲線を引き、実機の微小な角度ばらつきを残す。
        # 元区間より長い30秒のデモでは、残差を繰り返し使用する。
        # 25 Hzでは5点が約0.2秒に相当する。実波形の低周波な形を残しつつ、
        # センサ由来の細かな残差だけを取り出すための短い平滑化窓とする。
        smooth_angle = savgol_filter(resampled_angle, 5, 3)
        residual = resampled_angle - smooth_angle
        residual -= float(np.mean(residual))
        residual = np.clip(residual, -0.20, 0.20)
        repeated_residual = np.resize(residual, len(target_time))
        noise_records.append(repeated_residual)
    return target_time, noise_records


def spacer_properties(spacer_count):
    """スペーサ個数から合計質量と重心まわり慣性を返す。"""

    volume_m3 = np.pi * (
        SPACER_OUTER_RADIUS_M * SPACER_OUTER_RADIUS_M
        - SPACER_INNER_RADIUS_M * SPACER_INNER_RADIUS_M
    ) * SPACER_LENGTH_M
    one_mass = BRASS_DENSITY_KG_M3 * volume_m3
    one_centroid_inertia = one_mass / 12.0 * (
        3.0
        * (
            SPACER_OUTER_RADIUS_M * SPACER_OUTER_RADIUS_M
            + SPACER_INNER_RADIUS_M * SPACER_INNER_RADIUS_M
        )
        + SPACER_LENGTH_M * SPACER_LENGTH_M
    )
    return spacer_count * one_mass, spacer_count * one_centroid_inertia


def build_configurations():
    """デモに使うSP00、SP01、SP02、BALLの条件を作る。"""

    ball_delta_i, ball_delta_k = calculate_component_increments(
        BALL_MASS_KG,
        BALL_COM_RADIUS_M,
        BALL_CENTROID_INERTIA_KG_M2,
    )
    ball_inertia = BASE_INERTIA_KG_M2 + ball_delta_i
    ball_restoring = SOURCE_K_OVER_I * ball_inertia
    base_restoring = ball_restoring - ball_delta_k

    damping = 0.5 * SOURCE_B_OVER_I_VC * ball_inertia
    quadratic = 0.5 * SOURCE_C_OVER_I_VQC * ball_inertia
    friction = SOURCE_TAU_OVER_I * ball_inertia

    configurations = []
    for spacer_count in [0, 1, 2]:
        mass_kg, centroid_inertia = spacer_properties(spacer_count)
        delta_i, delta_k = calculate_component_increments(
            mass_kg, SPACER_COM_RADIUS_M, centroid_inertia
        )
        configurations.append(
            {
                "configuration": "SP" + str(spacer_count).zfill(2),
                "spacer_count": spacer_count,
                "component_mass_kg": mass_kg,
                "signed_com_radius_m": 0.0 if spacer_count == 0 else SPACER_COM_RADIUS_M,
                "component_centroid_inertia_kg_m2": centroid_inertia,
                "inertia_kg_m2": BASE_INERTIA_KG_M2 + delta_i,
                "restoring_n_m_per_rad": base_restoring + delta_k,
                "damping_n_m_s_per_rad": damping,
                "quadratic_n_m_s2_per_rad2": quadratic,
                "friction_n_m": friction,
                "use_for_calibration": 1,
            }
        )

    configurations.append(
        {
            "configuration": "BALL",
            "spacer_count": 0,
            "component_mass_kg": BALL_MASS_KG,
            "signed_com_radius_m": BALL_COM_RADIUS_M,
            "component_centroid_inertia_kg_m2": BALL_CENTROID_INERTIA_KG_M2,
            "inertia_kg_m2": ball_inertia,
            "restoring_n_m_per_rad": ball_restoring,
            "damping_n_m_s_per_rad": damping,
            "quadratic_n_m_s2_per_rad2": quadratic,
            "friction_n_m": friction,
            "use_for_calibration": 0,
        }
    )
    return configurations


def append_constant_part(time_values, angle_values, duration_s, angle_deg):
    """保持または静止の一定角度区間を追加する。"""

    count = int(round(duration_s * SAMPLE_RATE_HZ))
    index = 0
    while index < count:
        if len(time_values) == 0:
            next_time = 0.0
        else:
            next_time = time_values[-1] + 1.0 / SAMPLE_RATE_HZ
        time_values.append(next_time)
        angle_values.append(angle_deg)
        index += 1


def create_one_log(configuration, noise_records, random_generator):
    """1条件に対する±45 deg・各3回の連続ログを作る。"""

    time_values = []
    angle_values = []
    directions = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    variation = [-0.02, 0.0, 0.02]
    decay_time = np.arange(0.0, DECAY_TIME_S, 1.0 / SAMPLE_RATE_HZ)

    trial_index = 0
    while trial_index < len(directions):
        direction = directions[trial_index]
        repetition_index = trial_index // 2
        append_constant_part(
            time_values, angle_values, HOLD_TIME_S, direction * 45.0
        )

        inertia = configuration["inertia_kg_m2"]
        normalized_parameters = np.array(
            [
                configuration["restoring_n_m_per_rad"] / inertia,
                configuration["damping_n_m_s_per_rad"]
                / inertia
                * (1.0 + variation[repetition_index]),
                configuration["quadratic_n_m_s2_per_rad2"]
                / inertia
                * (1.0 - variation[repetition_index]),
                configuration["friction_n_m"]
                / inertia
                * (1.0 + 0.5 * variation[repetition_index]),
                0.0,
                direction * np.deg2rad(45.0),
                0.0,
            ]
        )
        angle_rad = simulate_normalized_decay(
            decay_time, normalized_parameters, FRICTION_EPSILON_RAD_S
        )
        angle_deg = np.rad2deg(angle_rad)

        # 実測ログの残差と極小さい再現性ノイズを加える。
        source_noise = noise_records[trial_index % len(noise_records)]
        generated_noise = random_generator.normal(0.0, 0.006, len(angle_deg))
        angle_deg = angle_deg + 0.35 * source_noise + generated_noise

        index = 0
        while index < len(decay_time):
            next_time = time_values[-1] + 1.0 / SAMPLE_RATE_HZ
            time_values.append(next_time)
            angle_values.append(float(angle_deg[index]))
            index += 1

        append_constant_part(time_values, angle_values, REST_TIME_S, 0.0)
        trial_index += 1

    return np.asarray(time_values), np.asarray(angle_values)


def write_angle_csv(output_path, time_s, angle1_deg):
    """LoggerDecoderと同じ列構成で角度CSVを保存する。"""

    header = [
        "systime[ms]",
        "angle0[deg]",
        "angle1[deg]",
        "angle0_raw[deg]",
        "angle1_raw[deg]",
        "angle0_average[deg]",
        "angle1_average[deg]",
        "err_angle0[-]",
        "err_angle1[-]",
        "",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        index = 0
        while index < len(time_s):
            angle_text = round(float(angle1_deg[index]), 6)
            writer.writerow(
                [
                    int(round(time_s[index] * 1000.0)),
                    0.0,
                    angle_text,
                    0.0,
                    angle_text,
                    0.0,
                    angle_text,
                    0,
                    0,
                    "",
                ]
            )
            index += 1


def write_manifest(output_directory, configurations):
    """データファイルと試験条件の対応表を保存する。"""

    header = [
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
    with (output_directory / "test_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=header)
        writer.writeheader()
        index = 0
        while index < len(configurations):
            configuration = configurations[index]
            writer.writerow(
                {
                    "data_file": "LOG" + str(index + 1).zfill(5) + "_ANGLE.csv",
                    "axis": "IN",
                    "angle_column": "angle1[deg]",
                    "configuration": configuration["configuration"],
                    "spacer_count": configuration["spacer_count"],
                    "component_mass_kg": configuration["component_mass_kg"],
                    "signed_com_radius_m": configuration["signed_com_radius_m"],
                    "component_centroid_inertia_kg_m2": configuration[
                        "component_centroid_inertia_kg_m2"
                    ],
                    "force_lever_m": FORCE_LEVER_M,
                    "use_for_calibration": configuration["use_for_calibration"],
                    "valid": 1,
                    "notes": "DEMO generated from existing LOG00014 free-decay characteristics",
                }
            )
            index += 1


def write_truth(output_directory, configurations):
    """デモの同定値を照合するための真値表を保存する。"""

    header = [
        "configuration",
        "inertia_kg_m2",
        "restoring_n_m_per_rad",
        "damping_n_m_s_per_rad",
        "quadratic_n_m_s2_per_rad2",
        "friction_n_m",
        "force_lever_m",
        "k_over_i_per_s2",
        "b_over_i_per_s",
        "c_over_i_per_rad",
        "tau_over_i_rad_s2",
    ]
    with (output_directory / "demo_truth.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=header)
        writer.writeheader()
        for configuration in configurations:
            inertia = configuration["inertia_kg_m2"]
            writer.writerow(
                {
                    "configuration": configuration["configuration"],
                    "inertia_kg_m2": inertia,
                    "restoring_n_m_per_rad": configuration[
                        "restoring_n_m_per_rad"
                    ],
                    "damping_n_m_s_per_rad": configuration[
                        "damping_n_m_s_per_rad"
                    ],
                    "quadratic_n_m_s2_per_rad2": configuration[
                        "quadratic_n_m_s2_per_rad2"
                    ],
                    "friction_n_m": configuration["friction_n_m"],
                    "force_lever_m": FORCE_LEVER_M,
                    "k_over_i_per_s2": configuration[
                        "restoring_n_m_per_rad"
                    ]
                    / inertia,
                    "b_over_i_per_s": configuration[
                        "damping_n_m_s_per_rad"
                    ]
                    / inertia,
                    "c_over_i_per_rad": configuration[
                        "quadratic_n_m_s2_per_rad2"
                    ]
                    / inertia,
                    "tau_over_i_rad_s2": configuration["friction_n_m"]
                    / inertia,
                }
            )


def write_provenance(output_directory):
    """デモ生成元と、実測とデモの区別をJSONで残す。"""

    source_hash = hashlib.sha256(SOURCE_LOG.read_bytes()).hexdigest()
    provenance = {
        "data_kind": "demo_generated_from_existing_free_decay",
        "not_raw_measurement": True,
        "source_log": str(SOURCE_LOG.relative_to(REPOSITORY_ROOT)),
        "source_sha256": source_hash,
        "source_windows_s": SOURCE_WINDOWS_S,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "random_seed": RANDOM_SEED,
        "source_normalized_coefficients": {
            "k_over_i_per_s2": SOURCE_K_OVER_I,
            "b_over_i_vc_per_s": SOURCE_B_OVER_I_VC,
            "c_over_i_vqc_per_rad": SOURCE_C_OVER_I_VQC,
            "tau_over_i_rad_s2": SOURCE_TAU_OVER_I,
        },
        "note": (
            "The demo preserves measured noise characteristics and representative "
            "decay coefficients, but spacer levels and absolute coefficients are assumed."
        ),
    }
    path = output_directory / "demo_provenance.json"
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    """デモ入力ファイルを一式生成する。"""

    arguments = parse_arguments()
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    unused_time, noise_records = read_source_noise(SOURCE_LOG)
    configurations = build_configurations()
    random_generator = np.random.default_rng(RANDOM_SEED)

    index = 0
    while index < len(configurations):
        configuration = configurations[index]
        time_s, angle_deg = create_one_log(
            configuration, noise_records, random_generator
        )
        file_name = "LOG" + str(index + 1).zfill(5) + "_ANGLE.csv"
        write_angle_csv(output_directory / file_name, time_s, angle_deg)
        print("Created: " + file_name)
        index += 1

    write_manifest(output_directory, configurations)
    write_truth(output_directory, configurations)
    write_provenance(output_directory)
    print("Demo directory: " + str(output_directory))


if __name__ == "__main__":
    main()
