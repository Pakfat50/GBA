"""自由減衰の分割、係数フィット、外力推定に使う基本関数。

このファイルは、処理を追いやすくするために機能ごとに短い関数へ分けている。
高度なPython固有表現を避け、通常のforループと辞書を中心に記述する。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter
from scipy.special import ellipe, ellipk


GRAVITY_M_S2 = 9.81


def wrap_angle_deg(angle_deg):
    """0～360 degを含む角度を-180～180 degへ折り返す。"""

    angle = np.asarray(angle_deg, dtype=float)
    return (angle + 180.0) % 360.0 - 180.0


def read_angle_log(file_path, angle_column):
    """LoggerDecoderのCSVから時刻と指定軸の角度を読み込む。"""

    table = pd.read_csv(file_path)
    time_column = "systime[ms]"
    if time_column not in table.columns:
        raise ValueError(str(file_path) + " に systime[ms] 列がありません")
    if angle_column not in table.columns:
        raise ValueError(str(file_path) + " に " + angle_column + " 列がありません")

    # 数値へ変換できない行は欠損値にし、その後で除外する。
    time_ms = pd.to_numeric(table[time_column], errors="coerce")
    angle_deg = pd.to_numeric(table[angle_column], errors="coerce")
    valid = np.isfinite(time_ms.to_numpy()) & np.isfinite(angle_deg.to_numpy())
    time_s = time_ms.to_numpy(dtype=float)[valid] * 0.001
    angle_deg = angle_deg.to_numpy(dtype=float)[valid]

    if len(time_s) < 10:
        raise ValueError(str(file_path) + " の有効データが少なすぎます")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError(str(file_path) + " の時刻が単調増加ではありません")

    # 垂直0 degを中心とした角度として扱う。
    angle_deg = wrap_angle_deg(angle_deg)
    return time_s, np.deg2rad(angle_deg)


def moving_average(values, count):
    """単純移動平均。分割判定のノイズ低減だけに使用する。"""

    if count <= 1:
        return np.asarray(values, dtype=float).copy()
    kernel = np.ones(count, dtype=float) / float(count)
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def find_true_groups(mask):
    """Trueが連続している範囲を(start, end)で返す。endは範囲に含む。"""

    groups = []
    start = None
    index = 0
    while index < len(mask):
        if mask[index] and start is None:
            start = index
        if start is not None and (not mask[index] or index == len(mask) - 1):
            end = index - 1
            if mask[index] and index == len(mask) - 1:
                end = index
            groups.append((start, end))
            start = None
        index += 1
    return groups


def split_free_decay(time_s, angle_rad, settings):
    """1ファイル中の複数の±45 deg自由減衰を自動分割する。

    解放前に可動端で静止するという試験手順を利用する。分割結果には元配列の
    開始・終了インデックスを残すため、元CSVを分割して保存する必要はない。
    """

    time_s = np.asarray(time_s, dtype=float)
    angle_rad = np.asarray(angle_rad, dtype=float)
    angle_deg = np.rad2deg(angle_rad)
    dt = float(np.median(np.diff(time_s)))

    smooth_count = max(3, int(round(settings["smooth_time_s"] / dt)))
    smooth_deg = moving_average(angle_deg, smooth_count)
    velocity_deg_s = np.gradient(smooth_deg, time_s)

    # 移動標準偏差を E[x^2]-E[x]^2 から計算する。
    hold_count = max(3, int(round(settings["hold_duration_s"] / dt)))
    mean_angle = moving_average(smooth_deg, hold_count)
    mean_square = moving_average(smooth_deg * smooth_deg, hold_count)
    variance = np.maximum(mean_square - mean_angle * mean_angle, 0.0)
    moving_std = np.sqrt(variance)

    hold_mask = np.abs(mean_angle) >= settings["start_angle_min_deg"]
    hold_mask &= moving_std <= settings["hold_std_max_deg"]
    hold_mask &= np.abs(velocity_deg_s) <= settings["hold_speed_max_deg_s"]

    raw_groups = find_true_groups(hold_mask)
    hold_groups = []
    for start, end in raw_groups:
        duration = time_s[end] - time_s[start]
        if duration >= settings["hold_duration_s"] * 0.5:
            hold_groups.append((start, end))

    segments = []
    positive_count = 0
    negative_count = 0

    group_index = 0
    while group_index < len(hold_groups):
        hold_start, hold_end = hold_groups[group_index]
        plateau_angle = float(np.median(smooth_deg[hold_start : hold_end + 1]))
        direction = "P45"
        if plateau_angle < 0.0:
            direction = "N45"

        # 保持区間を出て、可動端から中心側へ一定量移動した点を解放とする。
        release_index = hold_end + 1
        while release_index < len(time_s):
            inward_move = abs(plateau_angle) - abs(smooth_deg[release_index])
            if inward_move >= settings["release_drop_deg"]:
                break
            release_index += 1
        if release_index >= len(time_s):
            group_index += 1
            continue

        search_start_time = time_s[release_index] + settings["minimum_decay_time_s"]
        search_end_time = time_s[release_index] + settings["maximum_decay_time_s"]
        next_hold_start = len(time_s)
        if group_index + 1 < len(hold_groups):
            next_hold_start = hold_groups[group_index + 1][0]

        settle_count = max(3, int(round(settings["settle_duration_s"] / dt)))
        end_index = min(next_hold_start - 1, len(time_s) - 1)
        settled = False
        index = release_index
        while index < end_index:
            if time_s[index] < search_start_time:
                index += 1
                continue
            if time_s[index] > search_end_time:
                break
            window_end = index + settle_count
            if window_end >= end_index:
                break
            window = smooth_deg[index:window_end]
            window_center = float(np.mean(window))
            window_std = float(np.std(window))
            if abs(window_center) <= settings["settle_center_max_deg"]:
                if window_std <= settings["settle_std_max_deg"]:
                    # 静止確認に使った5秒間は、フィット対象へ含めない。
                    # 手で次の試験準備を始める直前の静止データを大量に含めると、
                    # 低速摩擦の推定が不自然に大きくなるためである。
                    end_index = index
                    settled = True
                    break
            index += 1

        if direction == "P45":
            positive_count += 1
            repetition = positive_count
        else:
            negative_count += 1
            repetition = negative_count

        segment = {
            "start_index": int(release_index),
            "end_index": int(end_index),
            "hold_start_index": int(hold_start),
            "hold_end_index": int(hold_end),
            "direction": direction,
            "repetition": int(repetition),
            "plateau_angle_deg": plateau_angle,
            "settled_automatically": int(settled),
        }
        segments.append(segment)
        group_index += 1

    return segments


def detect_free_decay_candidates(time_s, angle_rad, settings):
    """連続ログから自由減衰の候補波形をすべて取り出す。

    新ハードの試験では、正負の試験、再試験、IN軸とOUT軸を一つのCSVへ
    続けて記録する。そのため「解放前に一定時間静止した」という条件だけでは
    波形を取りこぼす。ここでは大振幅の山を検出し、8秒以内に続く山を一回の
    自由減衰としてまとめる。

    この関数は候補を自動除外しない。品質指標と推奨値だけを返し、最終的な
    採否は waveform_selection.csv の一行ごとに人が確認する。
    """

    time_s = np.asarray(time_s, dtype=float)
    angle_rad = np.asarray(angle_rad, dtype=float)
    angle_deg = np.rad2deg(angle_rad)
    dt = float(np.median(np.diff(time_s)))

    smooth_count = max(3, int(round(settings["smooth_time_s"] / dt)))
    smooth_deg = moving_average(angle_deg, smooth_count)
    velocity_deg_s = np.gradient(smooth_deg, time_s)

    # 試験終了時の静止角を基準にする。ゼロ点が1～2 degずれていても、
    # 大振幅ピークと静止終了を同じ基準で判定できる。
    tail_count = max(10, int(round(settings["baseline_time_s"] / dt)))
    tail_count = min(tail_count, len(smooth_deg))
    baseline_deg = float(np.median(smooth_deg[-tail_count:]))
    centered_deg = smooth_deg - baseline_deg

    peak_distance = max(1, int(round(settings["peak_distance_s"] / dt)))
    peak_indices, unused_properties = find_peaks(
        np.abs(centered_deg),
        height=settings["candidate_peak_min_deg"],
        distance=peak_distance,
        prominence=settings["candidate_peak_prominence_deg"],
    )

    # 一回の減衰中には正負のピークが交互に並ぶ。それらの間隔は短いので、
    # 一定時間以上離れたピークだけを別試験として扱う。
    peak_groups = []
    for peak_index in peak_indices:
        if len(peak_groups) == 0:
            peak_groups.append([int(peak_index)])
            continue
        previous_peak = peak_groups[-1][-1]
        gap_s = time_s[peak_index] - time_s[previous_peak]
        if gap_s > settings["candidate_gap_s"]:
            peak_groups.append([int(peak_index)])
        else:
            peak_groups[-1].append(int(peak_index))

    # 手やリミットへの接触後、静止を待たずに端部へ戻して再試験した場合は、
    # 二つの試験の間隔が短く、上の時間条件だけでは一群になる。減衰途中で
    # 振幅が明確に再増加したピークを、新しい試験の先頭として分割する。
    refined_groups = []
    for group in peak_groups:
        current_group = [group[0]]
        peak_number = 1
        while peak_number < len(group):
            previous_amplitude = abs(centered_deg[group[peak_number - 1]])
            current_amplitude = abs(centered_deg[group[peak_number]])
            growth_limit = max(
                settings["quality_growth_min_deg"],
                settings["quality_growth_ratio"] * previous_amplitude,
            )
            if current_amplitude - previous_amplitude > growth_limit:
                refined_groups.append(current_group)
                current_group = [group[peak_number]]
            else:
                current_group.append(group[peak_number])
            peak_number += 1
        refined_groups.append(current_group)
    peak_groups = refined_groups

    settle_count = max(3, int(round(settings["settle_duration_s"] / dt)))
    candidates = []
    positive_count = 0
    negative_count = 0

    group_number = 0
    while group_number < len(peak_groups):
        group = peak_groups[group_number]
        first_peak = group[0]

        # 最初のピークから最初のゼロ交差までが、端部保持から解放へ移る区間。
        # その区間の上位振幅を端部角度とし、そこから中心側へ動いた点を開始にする。
        first_sign = 1.0
        if centered_deg[first_peak] < 0.0:
            first_sign = -1.0
        zero_crossing = first_peak + 1
        while zero_crossing < len(centered_deg):
            if centered_deg[zero_crossing] * first_sign <= 0.0:
                break
            if time_s[zero_crossing] - time_s[first_peak] > settings["zero_crossing_search_s"]:
                break
            zero_crossing += 1
        if zero_crossing >= len(centered_deg):
            group_number += 1
            continue

        search_values = np.abs(centered_deg[first_peak : zero_crossing + 1])
        if len(search_values) < 2:
            group_number += 1
            continue
        plateau_amplitude = float(np.percentile(search_values, 90.0))
        release_threshold = plateau_amplitude - settings["release_drop_deg"]

        # 端部角度は、手で保持している間にも少し変わることがある。角度の閾値
        # だけで開始点を決めると、この保持区間がフィット対象へ残る。そのため、
        # 中心向きの速度が一定時間続いた最初の点を、実際の解放点として使う。
        sustain_count = max(
            2,
            int(round(settings.get("release_speed_sustain_s", 0.05) / dt)),
        )
        release_speed_min = settings.get("release_speed_min_deg_s", 5.0)
        release_index = None
        signed_speed = first_sign * velocity_deg_s[first_peak : zero_crossing + 1]
        inward_mask = signed_speed <= -release_speed_min
        inward_groups = find_true_groups(inward_mask)

        # 手で端部へ動かす途中にも中心向きの動きが生じることがある。
        # 実際の自由振動は、解放後から最初のゼロ交差まで中心向き運動が
        # 続くため、ゼロ交差に最も近い連続区間の先頭を採用する。
        group_position = len(inward_groups) - 1
        while group_position >= 0:
            inward_start, inward_end = inward_groups[group_position]
            inward_length = inward_end - inward_start + 1
            if inward_length >= sustain_count:
                release_index = first_peak + inward_start
                break
            group_position -= 1

        # 速度による解放点が見つからない古いデータでは、従来の角度閾値へ
        # 戻す。これにより、サンプリング周期が粗いログも解析可能にする。
        if release_index is None:
            last_plateau_index = first_peak
            index = first_peak
            while index <= zero_crossing:
                if abs(centered_deg[index]) >= release_threshold:
                    last_plateau_index = index
                index += 1
            release_index = min(last_plateau_index + 1, zero_crossing)

        next_candidate_start = len(time_s) - 1
        if group_number + 1 < len(peak_groups):
            next_candidate_start = peak_groups[group_number + 1][0]

        minimum_end_time = time_s[release_index] + settings["minimum_decay_time_s"]
        maximum_end_time = time_s[release_index] + settings["maximum_decay_time_s"]
        end_index = min(next_candidate_start - 1, len(time_s) - 1)
        settled = False
        index = release_index
        while index < end_index:
            if time_s[index] < minimum_end_time:
                index += 1
                continue
            if time_s[index] > maximum_end_time:
                end_index = index
                break
            window_end = index + settle_count
            if window_end >= end_index:
                break
            window = smooth_deg[index:window_end]
            window_center = float(np.mean(window))
            window_std = float(np.std(window))
            if abs(window_center - baseline_deg) <= settings["settle_center_max_deg"]:
                if window_std <= settings["settle_std_max_deg"]:
                    end_index = index
                    settled = True
                    break
            index += 1

        if end_index <= release_index:
            group_number += 1
            continue

        direction = "P"
        positive_count += 1
        repetition = positive_count
        if centered_deg[release_index] < 0.0:
            direction = "N"
            positive_count -= 1
            negative_count += 1
            repetition = negative_count

        segment_values = centered_deg[release_index : end_index + 1]
        segment_time = time_s[release_index : end_index + 1]
        segment_abs = np.abs(segment_values)
        local_peaks, unused_local_properties = find_peaks(
            segment_abs,
            distance=peak_distance,
            prominence=settings["quality_peak_prominence_deg"],
        )

        growth_count = 0
        local_number = 1
        while local_number < len(local_peaks):
            previous_value = segment_abs[local_peaks[local_number - 1]]
            current_value = segment_abs[local_peaks[local_number]]
            growth_limit = max(
                settings["quality_growth_min_deg"],
                settings["quality_growth_ratio"] * previous_value,
            )
            if current_value - previous_value > growth_limit:
                growth_count += 1
            local_number += 1

        maximum_jump = 0.0
        if len(segment_values) >= 2:
            maximum_jump = float(np.max(np.abs(np.diff(segment_values))))

        quality_reasons = []
        if not settled:
            quality_reasons.append("静止終了を自動検出できない")
        if growth_count > 0:
            quality_reasons.append("減衰途中で振幅が再増加")
        if float(np.max(segment_abs)) > settings["quality_max_angle_deg"]:
            quality_reasons.append("角度が品質上限を超過")
        if maximum_jump > settings["quality_max_jump_deg"]:
            quality_reasons.append("隣接サンプルの角度跳躍が大きい")

        suggested_use = 1
        if len(quality_reasons) > 0:
            suggested_use = 0

        candidates.append(
            {
                "start_index": int(release_index),
                "end_index": int(end_index),
                "direction": direction,
                "repetition": int(repetition),
                "start_time_s": float(time_s[release_index]),
                "end_time_s": float(time_s[end_index]),
                "duration_s": float(segment_time[-1] - segment_time[0]),
                "initial_angle_deg": float(centered_deg[release_index]),
                "baseline_deg": baseline_deg,
                "maximum_abs_angle_deg": float(np.max(segment_abs)),
                "maximum_sample_jump_deg": maximum_jump,
                "amplitude_growth_count": int(growth_count),
                "settled_automatically": int(settled),
                "suggested_use": suggested_use,
                "quality_note": " / ".join(quality_reasons),
            }
        )
        group_number += 1

    return candidates


def simulate_normalized_decay(time_s, parameters, epsilon_rad_s):
    """正規化係数を使って自由減衰角度を数値積分する。"""

    k_over_i = parameters[0]
    b_over_i = parameters[1]
    c_over_i = parameters[2]
    tau_over_i = parameters[3]
    offset = parameters[4]
    initial_angle = parameters[5]
    initial_speed = parameters[6]

    def differential_equation(unused_time, state):
        angle = state[0]
        speed = state[1]
        acceleration = -k_over_i * np.sin(angle)
        acceleration -= b_over_i * speed
        acceleration -= c_over_i * abs(speed) * speed
        acceleration -= tau_over_i * np.tanh(speed / epsilon_rad_s)
        return [speed, acceleration]

    solution = solve_ivp(
        differential_equation,
        (0.0, float(time_s[-1])),
        [initial_angle, initial_speed],
        t_eval=time_s,
        rtol=2e-7,
        atol=2e-9,
        max_step=0.04,
    )
    if not solution.success:
        raise RuntimeError("自由減衰モデルの数値積分に失敗しました")
    return solution.y[0] + offset


def fit_quality_statistics(measured, predicted):
    """実測値と予測値のRMSE、相関係数R、決定係数R^2を返す。"""

    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if measured.shape != predicted.shape:
        raise ValueError("実測値と予測値の配列形状が一致しません")
    if measured.size == 0:
        raise ValueError("適合度を計算するデータがありません")

    residual = predicted - measured
    rmse = float(np.sqrt(np.mean(residual * residual)))
    measured_centered = measured - float(np.mean(measured))
    predicted_centered = predicted - float(np.mean(predicted))
    measured_sum_square = float(np.sum(measured_centered * measured_centered))
    predicted_sum_square = float(np.sum(predicted_centered * predicted_centered))

    r_value = float("nan")
    if measured_sum_square > 0.0 and predicted_sum_square > 0.0:
        numerator = float(np.sum(measured_centered * predicted_centered))
        r_value = numerator / np.sqrt(measured_sum_square * predicted_sum_square)

    r_squared = float("nan")
    if measured_sum_square > 0.0:
        r_squared = 1.0 - float(np.sum(residual * residual)) / measured_sum_square
    return rmse, r_value, r_squared


def estimate_initial_frequency(time_s, angle_rad):
    """ゼロ交差から初期固有角周波数を概算する。"""

    tail_count = max(5, int(len(angle_rad) * 0.1))
    center = float(np.median(angle_rad[-tail_count:]))
    centered = angle_rad - center
    crossing_times = []
    index = 1
    while index < len(centered):
        if centered[index - 1] <= 0.0 and centered[index] > 0.0:
            crossing_times.append(time_s[index])
        index += 1
    if len(crossing_times) >= 3:
        periods = np.diff(np.asarray(crossing_times))
        period = float(np.median(periods))
        if period > 0.0:
            return 2.0 * np.pi / period
    return np.sqrt(5.6)


def extract_decay_turning_points(
    time_s,
    angle_rad,
    center_rad,
    smooth_time_s=0.07,
    prominence_deg=0.25,
):
    """自由減衰波形から正負が交互に並ぶ頂点を抽出する。

    center_radは波形フィットから求めず、試験前後の静止区間など独立に求めた
    平衡角を渡す。頂点角度は平滑化波形から読み、時刻は元のサンプル時刻を使う。
    """

    time_s = np.asarray(time_s, dtype=float)
    angle_rad = np.asarray(angle_rad, dtype=float)
    if len(time_s) != len(angle_rad) or len(time_s) < 9:
        raise ValueError("頂点抽出には同じ長さの9点以上の時刻・角度が必要です")
    sample_period = float(np.median(np.diff(time_s)))
    if sample_period <= 0.0:
        raise ValueError("時刻は単調増加でなければなりません")

    window = max(5, int(round(smooth_time_s / sample_period)))
    if window % 2 == 0:
        window += 1
    maximum_window = len(angle_rad) if len(angle_rad) % 2 == 1 else len(angle_rad) - 1
    window = min(window, maximum_window)
    if window < 5:
        smoothed = angle_rad.copy()
    else:
        smoothed = savgol_filter(angle_rad, window, 3)

    omega_guess = estimate_initial_frequency(time_s - time_s[0], smoothed)
    period_guess = 2.0 * np.pi / max(omega_guess, 1e-6)
    minimum_distance = max(3, int(0.30 * period_guess / sample_period))
    prominence = np.deg2rad(prominence_deg)
    positive, unused_positive_properties = find_peaks(
        smoothed, prominence=prominence, distance=minimum_distance
    )
    negative, unused_negative_properties = find_peaks(
        -smoothed, prominence=prominence, distance=minimum_distance
    )
    candidates = np.sort(np.concatenate([positive, negative]))

    # 同符号の候補が連続した場合は、平衡位置から遠い方だけを残す。
    selected = []
    for index in candidates:
        sign = np.sign(smoothed[index] - center_rad)
        if sign == 0.0:
            continue
        if len(selected) == 0:
            selected.append(int(index))
            continue
        previous_sign = np.sign(smoothed[selected[-1]] - center_rad)
        if sign != previous_sign:
            selected.append(int(index))
        elif abs(smoothed[index] - center_rad) > abs(
            smoothed[selected[-1]] - center_rad
        ):
            selected[-1] = int(index)

    indices = np.asarray(selected, dtype=int)
    return {
        "indices": indices,
        "time_s": time_s[indices],
        "angle_rad": smoothed[indices],
        "amplitude_rad": np.abs(smoothed[indices] - center_rad),
        "sign": np.sign(smoothed[indices] - center_rad).astype(int),
        "smoothed_angle_rad": smoothed,
    }


def refine_turning_point_quadratic(time_s, values, index):
    """近傍3点の二次補間で頂点時刻と値をサンプル間へ補正する。"""

    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    index = int(index)
    if index <= 0 or index >= len(values) - 1:
        return float(time_s[index]), float(values[index]), 0

    local_time = time_s[index - 1 : index + 2] - time_s[index]
    local_values = values[index - 1 : index + 2]
    quadratic, linear, constant = np.polyfit(local_time, local_values, 2)
    scale = max(float(np.max(np.abs(local_values))), 1.0)
    if abs(quadratic) <= np.finfo(float).eps * scale:
        return float(time_s[index]), float(values[index]), 0

    vertex_time = -linear / (2.0 * quadratic)
    if vertex_time < local_time[0] or vertex_time > local_time[-1]:
        return float(time_s[index]), float(values[index]), 0

    vertex_value = quadratic * vertex_time * vertex_time
    vertex_value += linear * vertex_time + constant
    return float(time_s[index] + vertex_time), float(vertex_value), 1


def extract_decay_extrema(
    time_s,
    angle_rad,
    smooth_time_s=0.07,
    prominence_deg=0.25,
):
    """平衡中心を仮定せず、交互に並ぶ極大・極小を二次補間して返す。"""

    time_s = np.asarray(time_s, dtype=float)
    angle_rad = np.asarray(angle_rad, dtype=float)
    if len(time_s) != len(angle_rad) or len(time_s) < 9:
        raise ValueError("極値抽出には同じ長さの9点以上の時刻・角度が必要です")
    sample_period = float(np.median(np.diff(time_s)))
    if sample_period <= 0.0:
        raise ValueError("時刻は単調増加でなければなりません")

    window = max(5, int(round(smooth_time_s / sample_period)))
    if window % 2 == 0:
        window += 1
    maximum_window = len(angle_rad) if len(angle_rad) % 2 == 1 else len(angle_rad) - 1
    window = min(window, maximum_window)
    if window < 5:
        smoothed = angle_rad.copy()
    else:
        smoothed = savgol_filter(angle_rad, window, 3)

    omega_guess = estimate_initial_frequency(time_s - time_s[0], smoothed)
    period_guess = 2.0 * np.pi / max(omega_guess, 1e-6)
    minimum_distance = max(3, int(0.30 * period_guess / sample_period))
    prominence = np.deg2rad(prominence_deg)
    positive, unused_positive_properties = find_peaks(
        smoothed, prominence=prominence, distance=minimum_distance
    )
    negative, unused_negative_properties = find_peaks(
        -smoothed, prominence=prominence, distance=minimum_distance
    )

    candidates = []
    for index in positive:
        candidates.append((int(index), 1))
    for index in negative:
        candidates.append((int(index), -1))
    candidates.sort(key=lambda item: item[0])

    # 中心値を使わず、同じ種類の極値が連続した場合は外側の候補を残す。
    selected = []
    for index, kind in candidates:
        if len(selected) == 0 or kind != selected[-1][1]:
            selected.append((index, kind))
            continue
        previous_index = selected[-1][0]
        replace = kind > 0 and smoothed[index] > smoothed[previous_index]
        replace = replace or (
            kind < 0 and smoothed[index] < smoothed[previous_index]
        )
        if replace:
            selected[-1] = (index, kind)

    if len(selected) < 4:
        raise ValueError("上下包絡線の算出に必要な極値が4点以上ありません")

    indices = []
    kinds = []
    refined_time = []
    refined_angle = []
    interpolated = []
    for index, kind in selected:
        peak_time, peak_angle, used_quadratic = refine_turning_point_quadratic(
            time_s, smoothed, index
        )
        indices.append(index)
        kinds.append(kind)
        refined_time.append(peak_time)
        refined_angle.append(peak_angle)
        interpolated.append(used_quadratic)

    return {
        "indices": np.asarray(indices, dtype=int),
        "time_s": np.asarray(refined_time, dtype=float),
        "angle_rad": np.asarray(refined_angle, dtype=float),
        "kind": np.asarray(kinds, dtype=int),
        "quadratic_interpolation_used": np.asarray(interpolated, dtype=int),
        "smoothed_angle_rad": smoothed,
    }


def estimate_envelope_center(
    extrema_time_s,
    extrema_angle_rad,
    extrema_kind,
    minimum_amplitude_deg=4.0,
):
    """上下の極値包絡線の中点の算術平均から平衡中心を決定する。"""

    time_s = np.asarray(extrema_time_s, dtype=float)
    angle = np.asarray(extrema_angle_rad, dtype=float)
    kind = np.asarray(extrema_kind, dtype=int)
    if len(time_s) != len(angle) or len(time_s) != len(kind):
        raise ValueError("極値の時刻、角度、種類は同じ長さでなければなりません")

    def calculate_midline(use_mask):
        positive = use_mask & (kind > 0)
        negative = use_mask & (kind < 0)
        if np.count_nonzero(positive) < 2 or np.count_nonzero(negative) < 2:
            raise ValueError("平衡中心の算出には正負各2点以上の極値が必要です")
        positive_time = time_s[positive]
        positive_angle = angle[positive]
        negative_time = time_s[negative]
        negative_angle = angle[negative]
        common_start = max(positive_time[0], negative_time[0])
        common_end = min(positive_time[-1], negative_time[-1])
        evaluation_time = time_s[
            use_mask & (time_s >= common_start) & (time_s <= common_end)
        ]
        evaluation_time = np.unique(evaluation_time)
        if len(evaluation_time) < 2:
            raise ValueError("上下包絡線に共通する時間範囲が不足しています")
        upper = np.interp(evaluation_time, positive_time, positive_angle)
        lower = np.interp(evaluation_time, negative_time, negative_angle)
        midpoint = 0.5 * (upper + lower)
        return evaluation_time, upper, lower, midpoint

    all_extrema = np.ones(len(time_s), dtype=bool)
    initial_time, initial_upper, initial_lower, initial_midpoint = calculate_midline(
        all_extrema
    )
    initial_center = float(np.mean(initial_midpoint))
    minimum_amplitude = np.deg2rad(minimum_amplitude_deg)
    amplitude_mask = np.abs(angle - initial_center) >= minimum_amplitude
    if np.count_nonzero(amplitude_mask & (kind > 0)) >= 2:
        if np.count_nonzero(amplitude_mask & (kind < 0)) >= 2:
            used_mask = amplitude_mask
        else:
            used_mask = all_extrema
    else:
        used_mask = all_extrema

    midpoint_time, upper_envelope, lower_envelope, midpoint = calculate_midline(
        used_mask
    )
    center = float(np.mean(midpoint))
    midpoint_deviation = midpoint - center
    midpoint_mean_absolute_deviation = float(np.mean(np.abs(midpoint_deviation)))
    midpoint_range = float(np.max(midpoint) - np.min(midpoint))
    if len(midpoint_time) >= 2:
        relative_time = midpoint_time - midpoint_time[0]
        midpoint_slope = float(np.polyfit(relative_time, midpoint, 1)[0])
    else:
        midpoint_slope = 0.0

    return {
        "center_rad": center,
        "used_extrema_mask": used_mask,
        "midpoint_time_s": midpoint_time,
        "upper_envelope_rad": upper_envelope,
        "lower_envelope_rad": lower_envelope,
        "midpoint_rad": midpoint,
        "midpoint_mean_absolute_deviation_rad": midpoint_mean_absolute_deviation,
        "midpoint_range_rad": midpoint_range,
        "midpoint_slope_rad_s": midpoint_slope,
    }


def preprocess_free_decay(
    time_s,
    angle_rad,
    existing_center_rad=None,
    minimum_amplitude_deg=4.0,
):
    """中心、最初の静止頂点、初期状態を自由変数なしで決定する。"""

    time_s = np.asarray(time_s, dtype=float)
    angle_rad = np.asarray(angle_rad, dtype=float)
    extrema = extract_decay_extrema(time_s, angle_rad)
    center = estimate_envelope_center(
        extrema["time_s"],
        extrema["angle_rad"],
        extrema["kind"],
        minimum_amplitude_deg,
    )
    center_rad = center["center_rad"]
    amplitude = np.abs(extrema["angle_rad"] - center_rad)
    valid = np.flatnonzero(amplitude >= np.deg2rad(minimum_amplitude_deg))
    if len(valid) == 0:
        raise ValueError("初期状態に使える振幅の頂点がありません")
    initial_peak = int(valid[0])

    tail_count = max(5, int(round(0.1 * len(angle_rad))))
    computed_tail_center = float(np.median(angle_rad[-tail_count:]))
    if existing_center_rad is None:
        existing_center_rad = computed_tail_center

    extrema["amplitude_rad"] = amplitude
    extrema["used_for_center"] = center["used_extrema_mask"].astype(int)
    extrema["is_initial_peak"] = np.zeros(len(amplitude), dtype=int)
    extrema["is_initial_peak"][initial_peak] = 1
    return {
        "center_rad": center_rad,
        "all_point_mean_rad": float(np.mean(angle_rad)),
        "tail_median_rad": float(existing_center_rad),
        "computed_segment_tail_median_rad": computed_tail_center,
        "initial_peak_number": initial_peak,
        "initial_time_s": float(extrema["time_s"][initial_peak]),
        "initial_angle_rad": float(
            extrema["angle_rad"][initial_peak] - center_rad
        ),
        "initial_speed_rad_s": 0.0,
        "ignored_initial_half_cycle_s": float(
            extrema["time_s"][initial_peak] - time_s[0]
        ),
        "midpoint_mean_absolute_deviation_rad": center[
            "midpoint_mean_absolute_deviation_rad"
        ],
        "midpoint_range_rad": center["midpoint_range_rad"],
        "midpoint_slope_rad_s": center["midpoint_slope_rad_s"],
        "midpoint_time_s": center["midpoint_time_s"],
        "upper_envelope_rad": center["upper_envelope_rad"],
        "lower_envelope_rad": center["lower_envelope_rad"],
        "midpoint_rad": center["midpoint_rad"],
        "extrema": extrema,
    }


def nonlinear_period_samples(
    turning_time_s,
    turning_amplitude_rad,
    minimum_amplitude_deg=5.0,
):
    """同符号頂点間の周期から有限振幅補正済みK/Iサンプルを返す。

    二つ隣の頂点までを1周期として時刻量子化の影響を抑える。周期中に振幅が
    減るため、両端のポテンシャルエネルギー平均に対応する代表振幅を使う。
    """

    time_s = np.asarray(turning_time_s, dtype=float)
    amplitude = np.asarray(turning_amplitude_rad, dtype=float)
    if len(time_s) != len(amplitude) or len(time_s) < 3:
        raise ValueError("K/I同定には3点以上の頂点が必要です")
    minimum_amplitude = np.deg2rad(minimum_amplitude_deg)
    samples = []
    for index in range(len(time_s) - 2):
        first = amplitude[index]
        last = amplitude[index + 2]
        if min(first, last) < minimum_amplitude:
            continue
        period = time_s[index + 2] - time_s[index]
        if period <= 0.0:
            continue
        normalized_energy = 0.5 * (
            (1.0 - np.cos(first)) + (1.0 - np.cos(last))
        )
        representative_amplitude = np.arccos(
            np.clip(1.0 - normalized_energy, -1.0, 1.0)
        )
        elliptic_parameter = np.sin(0.5 * representative_amplitude) ** 2
        elliptic_integral = float(ellipk(elliptic_parameter))
        q_value = (4.0 * elliptic_integral / period) ** 2
        small_angle_q = (2.0 * np.pi / period) ** 2
        samples.append(
            {
                "start_peak_number": int(index),
                "end_peak_number": int(index + 2),
                "start_time_s": float(time_s[index]),
                "end_time_s": float(time_s[index + 2]),
                "period_s": float(period),
                "start_amplitude_rad": float(first),
                "end_amplitude_rad": float(last),
                "representative_amplitude_rad": float(representative_amplitude),
                "elliptic_parameter": float(elliptic_parameter),
                "elliptic_integral_first_kind": elliptic_integral,
                "k_over_i_per_s2": float(q_value),
                "small_angle_k_over_i_per_s2": float(small_angle_q),
                "finite_amplitude_correction_ratio": float(
                    q_value / small_angle_q
                ),
            }
        )

    return samples


def identify_k_over_i_from_turning_points(
    turning_time_s,
    turning_amplitude_rad,
    minimum_amplitude_deg=5.0,
):
    """同符号頂点間の周期から、有限振幅補正済みのK/Iを求める。"""

    sample_rows = nonlinear_period_samples(
        turning_time_s,
        turning_amplitude_rad,
        minimum_amplitude_deg,
    )

    if len(sample_rows) < 2:
        raise ValueError("振幅条件を満たす周期が2個以上ありません")
    samples = np.asarray(
        [row["k_over_i_per_s2"] for row in sample_rows], dtype=float
    )
    median = float(np.median(samples))
    median_absolute_deviation = float(np.median(np.abs(samples - median)))
    robust_sigma = 1.4826 * median_absolute_deviation
    return {
        "k_over_i_per_s2": median,
        "sample_std_per_s2": float(np.std(samples, ddof=1)),
        "robust_sigma_per_s2": robust_sigma,
        "number_of_periods": int(len(samples)),
        "samples_per_s2": samples,
        "sample_rows": sample_rows,
    }


def explicit_half_cycle_energy_basis(amplitude_rad, inertia, restoring):
    """前の頂点振幅から半周期のb、c、tauエネルギー基底を返す。

    戻り値B、C、Rは、半周期の損失を
    ``delta_E = b*B + c*C + tau*R`` と書くための係数である。有限振幅の
    非線形振り子を使い、sin(theta)をthetaへ近似しない。
    """

    amplitude = float(amplitude_rad)
    inertia = float(inertia)
    restoring = float(restoring)
    if amplitude <= 0.0 or amplitude >= np.pi:
        raise ValueError("頂点振幅は0より大きくpiより小さくしてください")
    if inertia <= 0.0 or restoring <= 0.0:
        raise ValueError("IとKは正でなければなりません")

    omega_zero = np.sqrt(restoring / inertia)
    elliptic_parameter = np.sin(0.5 * amplitude) ** 2
    first_kind = ellipk(elliptic_parameter)
    second_kind = ellipe(elliptic_parameter)
    viscous_basis = 8.0 * omega_zero * (
        second_kind - (1.0 - elliptic_parameter) * first_kind
    )
    quadratic_basis = 4.0 * omega_zero * omega_zero * (
        np.sin(amplitude) - amplitude * np.cos(amplitude)
    )
    friction_basis = 2.0 * amplitude
    half_period = 2.0 * first_kind / omega_zero
    return {
        "viscous_basis": float(viscous_basis),
        "quadratic_basis": float(quadratic_basis),
        "friction_basis": float(friction_basis),
        "half_period_s": float(half_period),
    }


def choose_fit_samples(time_s, target_rate_hz):
    """計算時間を抑えるため、ほぼ一定間隔でフィット点を選ぶ。"""

    if len(time_s) < 2:
        return np.arange(len(time_s))
    original_rate = 1.0 / float(np.median(np.diff(time_s)))
    step = max(1, int(round(original_rate / target_rate_hz)))
    indices = np.arange(0, len(time_s), step, dtype=int)
    if indices[-1] != len(time_s) - 1:
        indices = np.append(indices, len(time_s) - 1)
    return indices


def fit_one_decay(time_s, angle_rad, epsilon_rad_s, target_rate_hz, max_nfev):
    """1本の自由減衰へ粘性・二乗・クーロン摩擦モデルをフィットする。"""

    time_s = np.asarray(time_s, dtype=float)
    angle_rad = np.asarray(angle_rad, dtype=float)
    relative_time = time_s - time_s[0]
    sample_indices = choose_fit_samples(relative_time, target_rate_hz)
    fit_time = relative_time[sample_indices]
    fit_angle = angle_rad[sample_indices]

    omega_guess = estimate_initial_frequency(fit_time, fit_angle)
    k_guess = max(0.2, omega_guess * omega_guess)
    offset_guess = float(np.median(fit_angle[-max(5, len(fit_angle) // 10) :]))
    initial = np.array(
        [k_guess, 0.15, 0.02, 0.05, offset_guess, fit_angle[0], 0.0],
        dtype=float,
    )
    lower = np.array([0.05, 0.0, 0.0, 0.0, -0.20, -1.7, -8.0])
    upper = np.array([100.0, 8.0, 20.0, 10.0, 0.20, 1.7, 8.0])

    # まず単純な粘性モデルで初期値を改善する。
    simple_indices = [0, 1, 4, 5, 6]

    def simple_residual(selected):
        trial = initial.copy()
        position = 0
        for parameter_index in simple_indices:
            trial[parameter_index] = selected[position]
            position += 1
        prediction = simulate_normalized_decay(fit_time, trial, epsilon_rad_s)
        return prediction - fit_angle

    simple_fit = least_squares(
        simple_residual,
        initial[simple_indices],
        bounds=(lower[simple_indices], upper[simple_indices]),
        max_nfev=max(40, max_nfev // 3),
        diff_step=1e-3,
    )
    position = 0
    for parameter_index in simple_indices:
        initial[parameter_index] = simple_fit.x[position]
        position += 1

    # 本フィットでは低振幅領域が埋もれないよう、振幅の大きい点の重みを少し下げる。
    amplitude_scale = max(float(np.max(np.abs(fit_angle))), np.deg2rad(1.0))
    weight = 1.0 / np.sqrt(0.25 + np.abs(fit_angle) / amplitude_scale)
    weight = weight / float(np.mean(weight))

    def full_residual(parameters):
        prediction = simulate_normalized_decay(fit_time, parameters, epsilon_rad_s)
        return (prediction - fit_angle) * weight

    result = least_squares(
        full_residual,
        initial,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        diff_step=1e-3,
        xtol=1e-7,
        ftol=1e-7,
        gtol=1e-7,
    )

    def evaluate_result(fit_result):
        prediction_rad = simulate_normalized_decay(
            relative_time, fit_result.x, epsilon_rad_s
        )
        error_deg_value = np.rad2deg(prediction_rad - angle_rad)
        rmse_value = float(np.sqrt(np.mean(error_deg_value * error_deg_value)))
        return prediction_rad, error_deg_value, rmse_value

    prediction, error_deg, rmse_deg = evaluate_result(result)

    # 長時間波形では、最初の局所解で減衰項が動かず終了する場合がある。
    # RMSEが大きい波形だけ、異なる減衰係数から再探索して最良解を採用する。
    if rmse_deg > 1.2:
        retry_starts = []
        retry_starts.append(result.x.copy())

        retry = result.x.copy()
        retry[1] = 0.02
        retry[2] = 0.02
        retry[3] = 0.20
        retry_starts.append(retry)

        retry = result.x.copy()
        retry[1] = 0.08
        retry[2] = 0.0
        retry[3] = 0.20
        retry_starts.append(retry)

        for retry_start in retry_starts:
            retry_result = least_squares(
                full_residual,
                retry_start,
                bounds=(lower, upper),
                max_nfev=max_nfev * 2,
                diff_step=1e-3,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )
            retry_prediction, retry_error, retry_rmse = evaluate_result(
                retry_result
            )
            if retry_rmse < rmse_deg:
                result = retry_result
                prediction = retry_prediction
                error_deg = retry_error
                rmse_deg = retry_rmse

    unused_rmse_rad, r_value, r_squared = fit_quality_statistics(
        angle_rad, prediction
    )
    output = {
        "k_over_i_per_s2": float(result.x[0]),
        "b_over_i_per_s": float(result.x[1]),
        "c_over_i_per_rad": float(result.x[2]),
        "tau_over_i_rad_s2": float(result.x[3]),
        "offset_deg": float(np.rad2deg(result.x[4])),
        "initial_angle_deg": float(np.rad2deg(result.x[5])),
        "initial_speed_rad_s": float(result.x[6]),
        "rmse_deg": rmse_deg,
        "r_value": r_value,
        "r_squared": r_squared,
        "success": int(result.success),
        "message": str(result.message),
        "parameters": result.x,
        "prediction_rad": prediction,
    }
    return output


def component_increments(manifest_row):
    """マニフェスト1行から追加慣性と追加復元係数を計算する。"""

    mass = float(manifest_row["component_mass_kg"])
    radius = float(manifest_row["signed_com_radius_m"])
    centroid_inertia = float(manifest_row["component_centroid_inertia_kg_m2"])
    delta_inertia = centroid_inertia + mass * radius * radius
    delta_restoring = mass * GRAVITY_M_S2 * radius
    return delta_inertia, delta_restoring


def identify_base_inertia_and_restoring(level_rows):
    """複数スペーサ水準から球なし基準のI0とK0を最小二乗同定する。"""

    if len(level_rows) < 3:
        raise ValueError("I0とK0の同定には3水準以上の有効な較正条件が必要です")

    # q*I0-K0 = deltaK-q*deltaI を線形最小二乗で解き、初期値とする。
    matrix = []
    right_side = []
    for row in level_rows:
        q_value = row["k_over_i_mean"]
        matrix.append([q_value, -1.0])
        right_side.append(row["delta_restoring"] - q_value * row["delta_inertia"])
    initial_solution, unused_residual, unused_rank, unused_singular = np.linalg.lstsq(
        np.asarray(matrix), np.asarray(right_side), rcond=None
    )
    initial_inertia = max(float(initial_solution[0]), 1e-7)
    initial_restoring = max(float(initial_solution[1]), 1e-6)

    minimum_delta_restoring = min(row["delta_restoring"] for row in level_rows)
    lower_restoring = max(1e-8, -minimum_delta_restoring + 1e-8)
    initial_restoring = max(initial_restoring, lower_restoring * 1.05)

    def residual(parameters):
        base_inertia = parameters[0]
        base_restoring = parameters[1]
        values = []
        for row in level_rows:
            predicted = (base_restoring + row["delta_restoring"]) / (
                base_inertia + row["delta_inertia"]
            )
            uncertainty = max(row["k_over_i_uncertainty"], 0.001 * row["k_over_i_mean"])
            values.append((predicted - row["k_over_i_mean"]) / uncertainty)
        return np.asarray(values)

    result = least_squares(
        residual,
        [initial_inertia, initial_restoring],
        bounds=([1e-9, lower_restoring], [0.1, 10.0]),
        max_nfev=500,
    )
    if not result.success:
        raise RuntimeError("I0とK0の最小二乗同定に失敗しました: " + result.message)

    prediction_rows = []
    for row in level_rows:
        predicted = (result.x[1] + row["delta_restoring"]) / (
            result.x[0] + row["delta_inertia"]
        )
        prediction_row = dict(row)
        prediction_row["k_over_i_predicted"] = float(predicted)
        prediction_row["k_over_i_residual"] = float(predicted - row["k_over_i_mean"])
        prediction_rows.append(prediction_row)

    return float(result.x[0]), float(result.x[1]), prediction_rows


def build_augmented_matrices(inertia, damping, restoring, sample_period_s):
    """外力トルクを状態に含む3状態モデルを離散化する。"""

    continuous = np.zeros((3, 3), dtype=float)
    continuous[0, 1] = 1.0
    continuous[1, 0] = -restoring / inertia
    continuous[1, 1] = -damping / inertia
    continuous[1, 2] = 1.0 / inertia
    output = np.array([[1.0, 0.0, 0.0]])

    torque_input = np.zeros((3, 1), dtype=float)
    torque_input[1, 0] = 1.0 / inertia
    block = np.zeros((4, 4), dtype=float)
    block[:3, :3] = continuous
    block[:3, 3:] = torque_input
    discrete = expm(block * sample_period_s)
    return discrete[:3, :3], discrete[:3, 3:], output


def ackermann_observer_gain(discrete_a, output, desired_poles):
    """反復極を許すAckermann公式でオブザーバーゲインを求める。"""

    n_state = discrete_a.shape[0]
    dual_a = discrete_a.T
    dual_b = output.T
    columns = []
    power = 0
    while power < n_state:
        columns.append(np.linalg.matrix_power(dual_a, power) @ dual_b)
        power += 1
    controllability = np.column_stack(columns)
    if np.linalg.matrix_rank(controllability) != n_state:
        raise ValueError("外力推定モデルが可観測ではありません")

    coefficients = np.poly(desired_poles)
    polynomial = np.linalg.matrix_power(dual_a, n_state).astype(complex)
    index = 0
    while index < len(coefficients) - 1:
        coefficient = coefficients[index + 1]
        matrix_power = n_state - 1 - index
        polynomial += coefficient * np.linalg.matrix_power(dual_a, matrix_power)
        index += 1
    selector = np.zeros((1, n_state))
    selector[0, -1] = 1.0
    gain = (selector @ np.linalg.inv(controllability) @ polynomial).T
    return np.real_if_close(gain, tol=1000).astype(float)


def known_nonlinear_torque(angle, speed, restoring, quadratic, friction, epsilon):
    """線形オブザーバーへ既知入力として加える非線形補正トルク。"""

    torque = restoring * (angle - np.sin(angle))
    torque -= quadratic * abs(speed) * speed
    torque -= friction * np.tanh(speed / epsilon)
    return torque


def nonlinear_observer_state_step(state, sample_period_s, parameters, epsilon):
    """外力トルクを一定とした3状態非線形モデルをRK4で1ステップ進める。"""

    inertia = parameters["inertia_kg_m2"]
    damping = parameters["damping_n_m_s_per_rad"]
    restoring = parameters["restoring_n_m_per_rad"]
    quadratic = parameters["quadratic_n_m_s2_per_rad2"]
    friction = parameters["friction_n_m"]

    def derivative(current):
        angle = current[0]
        speed = current[1]
        external_torque = current[2]
        acceleration = external_torque
        acceleration -= damping * speed
        acceleration -= restoring * np.sin(angle)
        acceleration -= quadratic * abs(speed) * speed
        acceleration -= friction * np.tanh(speed / epsilon)
        acceleration /= inertia
        return np.array([speed, acceleration, 0.0])

    k1 = derivative(state)
    k2 = derivative(state + 0.5 * sample_period_s * k1)
    k3 = derivative(state + 0.5 * sample_period_s * k2)
    k4 = derivative(state + sample_period_s * k3)
    return state + sample_period_s * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def nonlinear_observer_jacobian(state, sample_period_s, parameters, epsilon):
    """非線形3状態モデルの局所線形化を離散化して返す。"""

    angle = state[0]
    speed = state[1]
    inertia = parameters["inertia_kg_m2"]
    damping = parameters["damping_n_m_s_per_rad"]
    restoring = parameters["restoring_n_m_per_rad"]
    quadratic = parameters["quadratic_n_m_s2_per_rad2"]
    friction = parameters["friction_n_m"]
    normalized_speed = speed / epsilon
    tanh_speed = np.tanh(normalized_speed)

    continuous = np.zeros((3, 3), dtype=float)
    continuous[0, 1] = 1.0
    continuous[1, 0] = -restoring * np.cos(angle) / inertia
    continuous[1, 1] = -damping / inertia
    continuous[1, 1] -= 2.0 * quadratic * abs(speed) / inertia
    continuous[1, 1] -= friction / (inertia * epsilon) * (
        1.0 - tanh_speed * tanh_speed
    )
    continuous[1, 2] = 1.0 / inertia
    return expm(continuous * sample_period_s)


def torque_to_horizontal_force(torque, angle, force_lever_m):
    """一般化トルクを球へ加わる水平力へ変換する。"""

    cosine = np.cos(angle)
    safe_cosine = cosine.copy()
    small = np.abs(safe_cosine) < 0.2
    safe_cosine[small] = 0.2 * np.sign(safe_cosine[small] + 1e-12)
    return torque / (force_lever_m * safe_cosine)


def estimate_force_eso(angle_rad, sample_period_s, parameters, settings):
    """摩擦・二乗抵抗・sin(theta)を補償した因果ESO外力推定。"""

    ad, bd, output = build_augmented_matrices(
        parameters["inertia_kg_m2"],
        parameters["damping_n_m_s_per_rad"],
        parameters["restoring_n_m_per_rad"],
        sample_period_s,
    )
    target = np.exp(-2.0 * np.pi * settings["eso_pole_hz"] * sample_period_s)
    gain = ackermann_observer_gain(ad, output, np.full(3, target))
    transition = ad - gain @ output
    state = np.zeros(3, dtype=float)
    # 自由減衰は約45 degから始まる。角度状態を0から開始すると、最初の1点を
    # 巨大な外力と誤認するため、観測された初期角度で初期化する。
    if len(angle_rad) > 0:
        state[0] = float(angle_rad[0])
    estimated_torque = np.zeros(len(angle_rad), dtype=float)

    index = 0
    while index < len(angle_rad):
        estimated_torque[index] = state[2]
        correction = known_nonlinear_torque(
            state[0],
            state[1],
            parameters["restoring_n_m_per_rad"],
            parameters["quadratic_n_m_s2_per_rad2"],
            parameters["friction_n_m"],
            settings["epsilon_rad_s"],
        )
        state = transition @ state + gain[:, 0] * angle_rad[index]
        state += bd[:, 0] * correction
        index += 1

    return torque_to_horizontal_force(
        estimated_torque, angle_rad, parameters["force_lever_m"]
    )


def estimate_force_rts(angle_rad, sample_period_s, parameters, settings):
    """摩擦・二乗抵抗・sin(theta)を補償したEKF/RTS外力推定。"""

    output = np.array([[1.0, 0.0, 0.0]])
    count = len(angle_rad)
    process_noise = settings["rts_force_random_walk_n_per_sample"]
    torque_noise = parameters["force_lever_m"] * process_noise
    process_covariance = np.diag([1e-24, 1e-24, torque_noise * torque_noise])
    measurement_variance = settings["angle_noise_rad"] ** 2

    covariance = np.diag(
        [np.deg2rad(5.0) ** 2, 1.0, (parameters["force_lever_m"] * 0.5) ** 2]
    )
    filtered = np.zeros((count, 3), dtype=float)
    predicted = np.zeros((count, 3), dtype=float)
    filtered_covariance = np.zeros((count, 3, 3), dtype=float)
    predicted_covariance = np.zeros((count, 3, 3), dtype=float)
    transition_jacobian = np.repeat(np.eye(3)[None, :, :], count, axis=0)
    state = np.zeros(3, dtype=float)
    if count > 0:
        state[0] = float(angle_rad[0])
    identity = np.eye(3)

    index = 0
    while index < count:
        if index > 0:
            previous = filtered[index - 1]
            jacobian = nonlinear_observer_jacobian(
                previous, sample_period_s, parameters, settings["epsilon_rad_s"]
            )
            transition_jacobian[index] = jacobian
            state = nonlinear_observer_state_step(
                previous, sample_period_s, parameters, settings["epsilon_rad_s"]
            )
            covariance = (
                jacobian @ filtered_covariance[index - 1] @ jacobian.T
                + process_covariance
            )

        predicted[index] = state
        predicted_covariance[index] = covariance
        innovation_variance = float((output @ covariance @ output.T).item())
        innovation_variance += measurement_variance
        gain = covariance @ output.T / innovation_variance
        innovation = angle_rad[index] - float((output @ state).item())
        state = state + gain[:, 0] * innovation
        correction_matrix = identity - gain @ output
        covariance = correction_matrix @ covariance @ correction_matrix.T
        covariance += gain * measurement_variance @ gain.T
        covariance = 0.5 * (covariance + covariance.T)
        filtered[index] = state
        filtered_covariance[index] = covariance
        index += 1

    smoothed = filtered.copy()
    index = count - 2
    while index >= 0:
        jacobian = transition_jacobian[index + 1]
        cross_covariance = filtered_covariance[index] @ jacobian.T
        smoother_gain = np.linalg.solve(
            predicted_covariance[index + 1].T, cross_covariance.T
        ).T
        smoothed[index] += smoother_gain @ (
            smoothed[index + 1] - predicted[index + 1]
        )
        index -= 1

    return torque_to_horizontal_force(
        smoothed[:, 2], angle_rad, parameters["force_lever_m"]
    )


def simulate_forced_motion(force_n, sample_period_s, parameters, initial_state=None):
    """同定済みモデルへ水平外力を加え、非線形運動をRK4で計算する。"""

    force_n = np.asarray(force_n, dtype=float)
    states = np.zeros((len(force_n), 2), dtype=float)
    if initial_state is not None:
        states[0] = np.asarray(initial_state, dtype=float)

    def derivative(state, force_value):
        angle = state[0]
        speed = state[1]
        applied_torque = parameters["force_lever_m"] * force_value * np.cos(angle)
        resisting = parameters["damping_n_m_s_per_rad"] * speed
        resisting += parameters["restoring_n_m_per_rad"] * np.sin(angle)
        resisting += parameters["quadratic_n_m_s2_per_rad2"] * abs(speed) * speed
        resisting += parameters["friction_n_m"] * np.tanh(
            speed / parameters["epsilon_rad_s"]
        )
        acceleration = (applied_torque - resisting) / parameters["inertia_kg_m2"]
        return np.array([speed, acceleration])

    index = 0
    while index < len(force_n) - 1:
        state = states[index]
        force_value = force_n[index]
        k1 = derivative(state, force_value)
        k2 = derivative(state + 0.5 * sample_period_s * k1, force_value)
        k3 = derivative(state + 0.5 * sample_period_s * k2, force_value)
        k4 = derivative(state + sample_period_s * k3, force_value)
        states[index + 1] = state + sample_period_s * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        index += 1
    return states


def create_kaimal_wind(settings):
    """既存シミュレーションと同じ式でKaimal風速と抗力を生成する。"""

    sample_rate = settings["sample_rate_hz"]
    duration = settings["duration_s"]
    mean_speed = settings["mean_speed_m_s"]
    turbulence = settings["turbulence_intensity"]
    integral_scale = settings["integral_scale_m"]
    maximum_speed = settings["maximum_speed_m_s"]
    sample_count = int(round(sample_rate * duration))
    dt = 1.0 / sample_rate
    time_s = np.arange(sample_count, dtype=float) * dt
    frequency = np.fft.rfftfreq(sample_count, dt)
    df = frequency[1] - frequency[0]
    sigma = turbulence * mean_speed
    spectrum = (
        4.0
        * sigma**2
        * integral_scale
        / mean_speed
        / (1.0 + 6.0 * frequency * integral_scale / mean_speed) ** (5.0 / 3.0)
    )
    random_generator = np.random.default_rng(settings["seed"])
    phase = random_generator.uniform(0.0, 2.0 * np.pi, len(frequency))
    coefficient = sample_count * np.sqrt(spectrum * df / 2.0) * np.exp(1j * phase)
    coefficient[0] = 0.0
    if sample_count % 2 == 0:
        coefficient[-1] = 0.0
    fluctuation = np.fft.irfft(coefficient, n=sample_count)
    fluctuation *= sigma / float(np.std(fluctuation))

    scale = 1.0
    minimum_fluctuation = float(np.min(fluctuation))
    maximum_fluctuation = float(np.max(fluctuation))
    if minimum_fluctuation < 0.0:
        scale = min(scale, mean_speed / -minimum_fluctuation)
    if maximum_fluctuation > 0.0:
        scale = min(scale, (maximum_speed - mean_speed) / maximum_fluctuation)
    speed = mean_speed + scale * fluctuation

    force = 0.5 * settings["air_density_kg_m3"]
    force *= settings["drag_coefficient"] * settings["projected_area_m2"]
    force = force * speed * np.abs(speed)
    return time_s, speed, force


def resample_uniform(time_s, angle_rad):
    """オブザーバー用に不等間隔データを中央値周期の等間隔へ補間する。"""

    dt = float(np.median(np.diff(time_s)))
    uniform_time = np.arange(time_s[0], time_s[-1] + 0.5 * dt, dt)
    uniform_angle = np.interp(uniform_time, time_s, angle_rad)
    return uniform_time, uniform_angle, dt


def save_csv(path, rows):
    """辞書行の一覧をCSVへ保存する。"""

    if len(rows) == 0:
        pd.DataFrame().to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def ensure_directory(path):
    """出力ディレクトリがなければ作成する。"""

    Path(path).mkdir(parents=True, exist_ok=True)
