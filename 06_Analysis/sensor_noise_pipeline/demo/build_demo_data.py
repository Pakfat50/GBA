"""センサーノイズ解析パイプライン用の再現可能なデモログを作る。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[2]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "04_Data" / "05_Fitting" / "20990102"


def create_colored_noise(sample_count, sample_period_s, sigma_deg, tau_s, seed):
    """標準偏差を保つ一次Gauss-Markovノイズを作る。"""

    random_generator = np.random.default_rng(seed)
    coefficient = np.exp(-sample_period_s / tau_s)
    innovation_sigma = sigma_deg * np.sqrt(1.0 - coefficient * coefficient)
    values = np.zeros(sample_count, dtype=float)
    values[0] = random_generator.normal(0.0, sigma_deg)
    index = 1
    while index < sample_count:
        values[index] = coefficient * values[index - 1]
        values[index] += random_generator.normal(0.0, innovation_sigma)
        index += 1
    return values


def create_base_angle(time_s):
    """20秒の静止を3回含み、反復間だけ目標角度から動く波形を作る。"""

    angle = np.zeros(len(time_s), dtype=float)
    first_move = (time_s >= 20.0) & (time_s < 25.0)
    second_move = (time_s >= 45.0) & (time_s < 50.0)
    angle[first_move] = 5.0 * np.sin(np.pi * (time_s[first_move] - 20.0) / 5.0)
    angle[second_move] = -5.0 * np.sin(np.pi * (time_s[second_move] - 45.0) / 5.0)
    return angle


def make_axis(time_s, offset_deg, white_sigma_deg, colored_sigma_deg, tau_s, seed):
    """白色・有色ノイズを加え、14 bit角度格子へ量子化する。"""

    random_generator = np.random.default_rng(seed)
    base = create_base_angle(time_s) + offset_deg
    white = random_generator.normal(0.0, white_sigma_deg, len(time_s))
    colored = create_colored_noise(
        len(time_s),
        time_s[1] - time_s[0],
        colored_sigma_deg,
        tau_s,
        seed + 100,
    )
    quantum = 360.0 / 16384.0
    return np.round((base + white + colored) / quantum) * quantum


def main():
    """デモ角度CSV、対応表、真値JSONを保存する。"""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sample_period_s = 0.01
    time_s = np.arange(0.0, 70.0, sample_period_s)
    table = pd.DataFrame(
        {
            "systime[ms]": np.round(time_s * 1000.0).astype(int),
            "angle0[deg]": make_axis(time_s, 0.04, 0.015, 0.010, 0.475, 20260918),
            "angle1[deg]": make_axis(time_s, -0.03, 0.018, 0.008, 0.700, 20260919),
        }
    )
    table.to_csv(OUTPUT_DIRECTORY / "LOG00001_ANGLE.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "data_file": "LOG00001_ANGLE.csv",
                "axis": "OUT",
                "angle_column": "angle0[deg]",
                "condition": "FIXED_0",
                "target_angle_deg": 0,
                "expected_repetitions": 3,
                "use_start_s": "",
                "use_end_s": "",
                "valid": 1,
                "notes": "合成デモ外軸",
            },
            {
                "data_file": "LOG00001_ANGLE.csv",
                "axis": "IN",
                "angle_column": "angle1[deg]",
                "condition": "FIXED_0",
                "target_angle_deg": 0,
                "expected_repetitions": 3,
                "use_start_s": "",
                "use_end_s": "",
                "valid": 1,
                "notes": "合成デモ内軸",
            },
        ]
    )
    manifest.to_csv(OUTPUT_DIRECTORY / "sensor_noise_manifest.csv", index=False)

    truth = {
        "sample_rate_hz": 100.0,
        "quantization_bits": 14,
        "OUT": {
            "white_noise_std_deg": 0.015,
            "colored_noise_std_deg": 0.010,
            "colored_time_constant_s": 0.475,
        },
        "IN": {
            "white_noise_std_deg": 0.018,
            "colored_noise_std_deg": 0.008,
            "colored_time_constant_s": 0.700,
        },
    }
    with open(OUTPUT_DIRECTORY / "sensor_noise_demo_truth.json", "w", encoding="utf-8") as file_object:
        json.dump(truth, file_object, ensure_ascii=False, indent=2)
    print("デモデータを作成しました: " + str(OUTPUT_DIRECTORY))


if __name__ == "__main__":
    main()

