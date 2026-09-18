"""角度センサーの静止ノイズ解析に使う基本関数。

処理を追いやすくするため、短い関数と通常のforループを中心に記述する。
角度の単位は、このファイルでは特記しない限りdegとする。
"""

import math

import numpy as np
import pandas as pd


def wrap_angle_deg(angle_deg):
    """0～360 degを含む角度を-180～180 degへ折り返す。"""

    angle = np.asarray(angle_deg, dtype=float)
    return (angle + 180.0) % 360.0 - 180.0


def read_angle_log(file_path, angle_column):
    """LoggerDecoder形式CSVから時刻と指定した角度列を読み込む。"""

    table = pd.read_csv(file_path)
    time_column = "systime[ms]"
    if time_column not in table.columns:
        raise ValueError(str(file_path) + " に systime[ms] 列がありません")
    if angle_column not in table.columns:
        raise ValueError(str(file_path) + " に " + angle_column + " 列がありません")

    time_ms = pd.to_numeric(table[time_column], errors="coerce")
    angle_deg = pd.to_numeric(table[angle_column], errors="coerce")
    valid = np.isfinite(time_ms.to_numpy()) & np.isfinite(angle_deg.to_numpy())
    time_s = time_ms.to_numpy(dtype=float)[valid] * 0.001
    angle_deg = angle_deg.to_numpy(dtype=float)[valid]

    if len(time_s) < 20:
        raise ValueError(str(file_path) + " の有効データが少なすぎます")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError(str(file_path) + " の時刻が単調増加ではありません")

    return time_s, wrap_angle_deg(angle_deg)


def moving_average(values, count):
    """端部を延長した単純移動平均を返す。"""

    values = np.asarray(values, dtype=float)
    if count <= 1:
        return values.copy()
    if count % 2 == 0:
        count += 1
    half = count // 2
    padded = np.pad(values, (half, half), mode="edge")
    kernel = np.ones(count, dtype=float) / float(count)
    return np.convolve(padded, kernel, mode="valid")


def find_true_groups(mask):
    """Trueが連続する範囲を(start, end)で返す。endも範囲に含む。"""

    groups = []
    start = None
    index = 0
    while index < len(mask):
        if mask[index] and start is None:
            start = index
        at_last = index == len(mask) - 1
        if start is not None and (not mask[index] or at_last):
            end = index - 1
            if at_last and mask[index]:
                end = index
            groups.append((start, end))
            start = None
        index += 1
    return groups


def split_stationary_segments(time_s, angle_deg, target_angle_deg, settings):
    """目標角度で静止している複数の区間を自動分割する。

    反復間でセンサーを目標角度から一度動かす試験手順を利用する。
    元CSVは変更せず、開始・終了インデックスだけを返す。
    """

    time_s = np.asarray(time_s, dtype=float)
    angle_deg = np.asarray(angle_deg, dtype=float)
    dt = float(np.median(np.diff(time_s)))
    window_count = max(3, int(round(settings["stationary_window_s"] / dt)))

    mean_angle = moving_average(angle_deg, window_count)
    mean_square = moving_average(angle_deg * angle_deg, window_count)
    variance = np.maximum(mean_square - mean_angle * mean_angle, 0.0)
    moving_std = np.sqrt(variance)

    stationary = moving_std <= settings["stationary_std_max_deg"]
    stationary &= np.abs(mean_angle - target_angle_deg) <= settings["target_tolerance_deg"]

    edge_count = int(round(settings["edge_trim_s"] / dt))
    minimum_count = int(round(settings["minimum_segment_s"] / dt))
    groups = find_true_groups(stationary)
    segments = []
    repetition = 1
    for raw_start, raw_end in groups:
        start = raw_start + edge_count
        end = raw_end - edge_count
        if end - start + 1 < minimum_count:
            continue
        segment = {
            "start_index": int(start),
            "end_index": int(end),
            "repetition": int(repetition),
            "start_time_s": float(time_s[start]),
            "end_time_s": float(time_s[end]),
            "duration_s": float(time_s[end] - time_s[start]),
        }
        segments.append(segment)
        repetition += 1
    return segments


def remove_linear_trend(time_s, angle_deg):
    """平均値と一次ドリフトを除き、残差と傾きを返す。"""

    time_s = np.asarray(time_s, dtype=float)
    angle_deg = np.asarray(angle_deg, dtype=float)
    centered_time = time_s - float(np.mean(time_s))
    design = np.column_stack([centered_time, np.ones(len(centered_time))])
    coefficient, unused_residual, unused_rank, unused_singular = np.linalg.lstsq(
        design, angle_deg, rcond=None
    )
    fitted = design @ coefficient
    residual = angle_deg - fitted
    slope_deg_s = float(coefficient[0])
    center_deg = float(coefficient[1])
    return residual, slope_deg_s, center_deg


def quantization_grid_match(angle_deg, expected_quantization_deg):
    """値が既知の量子化格子へ一致する割合を返す。

    オフセットは未知なので、最初の値を格子の基準として使う。
    許容幅は量子化幅の2 %とする。
    """

    angle_deg = np.asarray(angle_deg, dtype=float)
    if expected_quantization_deg <= 0.0 or len(angle_deg) == 0:
        return float("nan")
    normalized = (angle_deg - angle_deg[0]) / expected_quantization_deg
    distance = np.abs(normalized - np.round(normalized))
    return float(np.mean(distance <= 0.02))


def autocovariance(values, maximum_lag):
    """平均を除いた信号の自己共分散を0～maximum_lagまで求める。"""

    values = np.asarray(values, dtype=float)
    values = values - float(np.mean(values))
    maximum_lag = min(maximum_lag, len(values) - 2)
    covariance = np.zeros(maximum_lag + 1, dtype=float)
    lag = 0
    while lag <= maximum_lag:
        covariance[lag] = float(np.mean(values[: len(values) - lag] * values[lag:]))
        lag += 1
    return covariance


def estimate_white_and_colored_noise(
    residual_deg,
    sample_period_s,
    quantization_deg,
    grid_match_ratio,
    maximum_correlation_time_s,
):
    """白色ノイズと一次Gauss-Markov型の有色ノイズを分離して推定する。

    遅れ1以降の自己共分散には白色ノイズが現れない性質を使い、
    正の自己共分散へ指数関数を当てはめる。分離できない場合は、全残差を
    白色ノイズとして安全側に扱う。
    """

    residual_deg = np.asarray(residual_deg, dtype=float)
    maximum_lag = int(round(maximum_correlation_time_s / sample_period_s))
    maximum_lag = max(3, min(maximum_lag, len(residual_deg) // 4))
    covariance = autocovariance(residual_deg, maximum_lag)
    total_variance = max(float(covariance[0]), 0.0)

    quantization_variance = 0.0
    if np.isfinite(grid_match_ratio) and grid_match_ratio >= 0.8:
        quantization_variance = quantization_deg * quantization_deg / 12.0

    usable_lag = []
    usable_covariance = []
    threshold = max(total_variance * 0.01, 1e-18)
    lag = 1
    while lag < len(covariance):
        if covariance[lag] <= threshold:
            break
        usable_lag.append(lag)
        usable_covariance.append(covariance[lag])
        lag += 1

    colored_variance = 0.0
    time_constant_s = 0.0
    ar1_coefficient = 0.0
    if len(usable_lag) >= 3:
        lag_time = np.asarray(usable_lag, dtype=float) * sample_period_s
        log_covariance = np.log(np.asarray(usable_covariance, dtype=float))
        line = np.polyfit(lag_time, log_covariance, 1)
        if line[0] < 0.0:
            time_constant_s = float(-1.0 / line[0])
            ar1_coefficient = float(math.exp(-sample_period_s / time_constant_s))
            colored_variance = float(math.exp(line[1]))

    available_variance = max(total_variance - quantization_variance, 0.0)
    colored_variance = min(max(colored_variance, 0.0), available_variance)
    white_variance = max(available_variance - colored_variance, 0.0)
    return {
        "white_noise_std_deg": math.sqrt(white_variance),
        "colored_noise_std_deg": math.sqrt(colored_variance),
        "colored_time_constant_s": time_constant_s,
        "ar1_coefficient": ar1_coefficient,
        "quantization_noise_std_deg": math.sqrt(quantization_variance),
        "autocovariance": covariance,
    }


def calculate_welch_psd(values, sample_rate_hz):
    """追加ライブラリを使わずWelch法の片側PSDを計算する。"""

    values = np.asarray(values, dtype=float)
    values = values - float(np.mean(values))
    maximum_length = min(1024, len(values))
    segment_length = 1
    while segment_length * 2 <= maximum_length:
        segment_length *= 2
    segment_length = max(32, segment_length)
    if segment_length > len(values):
        segment_length = len(values)
    step = max(1, segment_length // 2)
    window = np.hanning(segment_length)
    window_power = float(np.sum(window * window))

    spectra = []
    start = 0
    while start + segment_length <= len(values):
        part = values[start : start + segment_length]
        transformed = np.fft.rfft(part * window)
        spectrum = np.abs(transformed) ** 2 / (sample_rate_hz * window_power)
        if len(spectrum) > 2:
            spectrum[1:-1] *= 2.0
        spectra.append(spectrum)
        start += step

    if len(spectra) == 0:
        spectra.append(np.zeros(segment_length // 2 + 1, dtype=float))
    psd = np.mean(np.asarray(spectra), axis=0)
    frequency_hz = np.fft.rfftfreq(segment_length, 1.0 / sample_rate_hz)
    return frequency_hz, psd


def calculate_segment_metrics(
    time_s,
    angle_deg,
    expected_quantization_deg,
    maximum_correlation_time_s,
):
    """1静止区間の時刻品質、ばらつき、ノイズモデル係数を計算する。"""

    time_s = np.asarray(time_s, dtype=float)
    angle_deg = np.asarray(angle_deg, dtype=float)
    intervals = np.diff(time_s)
    sample_period_s = float(np.median(intervals))
    sample_rate_hz = 1.0 / sample_period_s
    jitter_ms = (intervals - sample_period_s) * 1000.0
    missing_intervals = int(np.sum(intervals > 1.5 * sample_period_s))

    residual, drift_slope_deg_s, center_deg = remove_linear_trend(time_s, angle_deg)
    grid_match = quantization_grid_match(angle_deg, expected_quantization_deg)
    noise = estimate_white_and_colored_noise(
        residual,
        sample_period_s,
        expected_quantization_deg,
        grid_match,
        maximum_correlation_time_s,
    )
    frequency_hz, psd = calculate_welch_psd(residual, sample_rate_hz)
    low_mask = frequency_hz <= 0.5
    total_power = float(np.sum(psd))
    low_frequency_fraction = 0.0
    if total_power > 0.0:
        low_frequency_fraction = float(np.sum(psd[low_mask]) / total_power)

    median_value = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median_value)))
    metrics = {
        "sample_count": int(len(angle_deg)),
        "duration_s": float(time_s[-1] - time_s[0]),
        "mean_angle_deg": float(np.mean(angle_deg)),
        "detrended_center_deg": center_deg,
        "noise_std_deg": float(np.std(residual, ddof=1)),
        "robust_noise_std_deg": 1.4826 * mad,
        "peak_to_peak_deg": float(np.max(residual) - np.min(residual)),
        "drift_slope_deg_s": drift_slope_deg_s,
        "sample_period_ms": sample_period_s * 1000.0,
        "sample_rate_hz": sample_rate_hz,
        "jitter_std_ms": float(np.std(jitter_ms, ddof=1)),
        "interval_min_ms": float(np.min(intervals) * 1000.0),
        "interval_max_ms": float(np.max(intervals) * 1000.0),
        "missing_interval_count": missing_intervals,
        "expected_quantization_deg": expected_quantization_deg,
        "quantization_grid_match_ratio": grid_match,
        "low_frequency_power_fraction": low_frequency_fraction,
        "white_noise_std_deg": noise["white_noise_std_deg"],
        "colored_noise_std_deg": noise["colored_noise_std_deg"],
        "colored_time_constant_s": noise["colored_time_constant_s"],
        "ar1_coefficient": noise["ar1_coefficient"],
        "quantization_noise_std_deg": noise["quantization_noise_std_deg"],
    }
    details = {
        "residual_deg": residual,
        "frequency_hz": frequency_hz,
        "psd_deg2_per_hz": psd,
        "autocovariance_deg2": noise["autocovariance"],
    }
    return metrics, details

