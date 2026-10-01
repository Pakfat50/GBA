#!/usr/bin/env python3
"""IHB-05方式Aの係数を固定し、球あり自由振動の全波形を連続積分で評価する。

使い方 (リポジトリのルートから):
    python 06_Analysis/fitting_pipeline/iterative_hybrid/ihb05_ball_continuous_waveform_validation.py

承認済みBALL計測12波形について、最初の適格な折返し点から最後の適格点まで
運動方程式を一度だけ連続積分します。各頂点で計算状態を計測値へ戻さないため、
振幅の減り方と周期のずれが時間とともにどう蓄積するかを確認できます。

I/KはIHB-02基準値に球の質量・位置・慣性を加えた値、τはIHB-03の軸別値、
c_ballはIHB-05で方式Aから得た値を使います。係数の再調整は行いません。
出力は波形別・集計指標CSV、全12波形のPNG/SVG、再現用設定JSONです。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIRECTORY))
from iterative_hybrid.ihb05_cball_identification import ball_physics, load_inputs

DATE = '20260921'
EPSILON_DEG_S = 0.5
RTOL = 2e-9
ATOL = 2e-11
MAX_STEP_PERIOD_FRACTION = 1.0 / 100.0


def crossings(time_s: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """隣り合う計測標本の間を直線とみなし、角度ゼロの時刻を補間する。"""
    found = []
    for i in range(len(time_s) - 1):
        y0, y1 = float(angle_deg[i]), float(angle_deg[i + 1])
        if y0 == 0.0:
            found.append(float(time_s[i]))
        elif y0 * y1 < 0.0:
            found.append(float(time_s[i] - y0 * (time_s[i + 1] - time_s[i]) / (y1 - y0)))
    return np.asarray(found, dtype=float)


def read_waveform(data_dir: Path, selection_row, center_deg: float,
                  start_s: float, end_s: float):
    """選択表の行から時刻と中心補正済み角度を読み、適格区間だけを返す。"""
    source = data_dir / str(selection_row.data_file)
    raw = pd.read_csv(source, encoding='utf-8-sig')
    start, stop = int(selection_row.start_index), int(selection_row.end_index) + 1
    part = raw.iloc[start:stop]
    time_s = pd.to_numeric(part['systime[ms]'], errors='coerce').to_numpy(float) * 1e-3
    angle = pd.to_numeric(part[str(selection_row.angle_column)], errors='coerce').to_numpy(float)
    # センサー角度の±180°折返しを解いてから、前処理で求めた中心角を引く。
    angle = (angle + 180.0) % 360.0 - 180.0 - float(center_deg)
    valid = np.isfinite(time_s) & np.isfinite(angle)
    time_s, angle = time_s[valid], angle[valid]
    keep = (time_s >= start_s) & (time_s <= end_s)
    time_s, angle = time_s[keep], angle[keep]
    if len(time_s) < 2 or np.any(np.diff(time_s) <= 0):
        raise ValueError(f'{selection_row.segment_id}: 計測時間が不足、または単調増加ではありません')
    return time_s, angle


def integrate(time_s: np.ndarray, initial_angle_deg: float, inertia: float,
              restoring: float, tau: float, c_total: float) -> np.ndarray:
    """係数固定の非線形運動方程式を初期折返し点から連続積分する。"""
    relative_time = time_s - time_s[0]
    epsilon = math.radians(EPSILON_DEG_S)
    period = 2.0 * math.pi * math.sqrt(inertia / restoring)

    def derivative(_time, state):
        theta, omega = state
        # IHB-05と同じモデル: 球ありではロッド抗力と球抗力を足し合わせる。
        acceleration = (-restoring * math.sin(theta)
                        - c_total * abs(omega) * omega
                        - tau * math.tanh(omega / epsilon)) / inertia
        return omega, acceleration

    solution = solve_ivp(
        derivative, (0.0, float(relative_time[-1])),
        (math.radians(float(initial_angle_deg)), 0.0),
        method='DOP853', t_eval=relative_time, rtol=RTOL, atol=ATOL,
        max_step=period * MAX_STEP_PERIOD_FRACTION,
    )
    if not solution.success or solution.y.shape[1] != len(time_s):
        raise RuntimeError(f'ODE積分に失敗しました: {solution.message}')
    return np.rad2deg(solution.y[0])


def make_figure(path: Path, records: list[dict]):
    """計測値と連続積分値を一枚の図に12個の小グラフで示す。"""
    # PDF/SVG経路データを簡潔にし、図を開きやすく保つ（評価計算には影響しない）。
    matplotlib.rcParams['path.simplify'] = True
    matplotlib.rcParams['path.simplify_threshold'] = 0.5
    columns = 4
    rows = math.ceil(len(records) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, rows * 2.45), squeeze=False)
    for ax, record in zip(axes.flat, records):
        t = record['time_s'] - record['time_s'][0]
        # 描画点のみ5点に1点表示し、数値指標は元の全標本から計算する。
        index = np.unique(np.r_[np.arange(0, len(t), 5), len(t) - 1])
        ax.plot(t[index], record['measured_deg'][index], color='#222222', lw=.65, label='Measured')
        ax.plot(t[index], record['predicted_deg'][index], color='#d62728', lw=.8, label='ODE integration')
        ax.axhline(0, color='#888888', lw=.4)
        ax.set_title(f"{record['segment_id']}  RMSE={record['sample_rmse_deg']:.2f}°", fontsize=8)
        ax.set_xlabel('Elapsed time [s]', fontsize=7)
        ax.set_ylabel('Angle [deg]', fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        ax.grid(alpha=.2, lw=.35)
    for ax in axes.flat[len(records):]:
        ax.axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .974),
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle('IHB-05 Method A: BALL free-decay measured and continuous ODE waveforms',
                 y=1.025, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, format='svg', bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repository-root', type=Path, default=Path(__file__).resolve().parents[3],
                        help='GBAリポジトリのルート (通常は変更不要)')
    parser.add_argument('--date', default=DATE, help='解析データの日付')
    parser.add_argument('--output-dir', type=Path, default=None, help='出力先 (省略時はIHB-05結果フォルダ)')
    args = parser.parse_args()
    root = args.repository_root.resolve()
    results_root = root / '06_Analysis/fitting_pipeline/results'
    results = results_root / args.date
    data_dir = root / '04_Data/05_Fitting' / args.date
    output = (args.output_dir or results / 'iterative_hybrid/ihb05_cball').resolve()
    output.mkdir(parents=True, exist_ok=True)

    turning, selection, manifest, bases = load_inputs(results_root, args.date)
    physics = ball_physics(manifest, bases)
    settings_path = output / 'ihb05_settings.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    c_ball = float(settings['c_ball_n_m_s2_per_rad2'])  # 方式Aの最終採用値
    c_rod = float(settings['c_rod_ball_exposed_length_adjusted'])
    tau_by_axis = {k: float(v) for k, v in settings['tau_by_axis_n_m'].items()}

    summaries = pd.read_csv(results / 'hybrid_identification/01_preprocessing/waveform_preprocessing.csv',
                            encoding='utf-8-sig').set_index('segment_id')
    selected = selection[(pd.to_numeric(selection.use_for_fitting, errors='coerce') == 1)
                         & (selection.review_status.astype(str).str.upper() == 'APPROVED')
                         & (selection.configuration.astype(str).str.upper() == 'BALL')]
    if len(selected) != 12:
        raise ValueError(f'承認済みBALL波形は12本を想定していますが、{len(selected)}本です')

    records, metrics = [], []
    for row in selected.itertuples(index=False):
        sid = str(row.segment_id)
        peaks = turning[turning.segment_id.astype(str) == sid].sort_values('peak_number')
        peaks = peaks[pd.to_numeric(peaks.eligible_for_later_stages, errors='coerce') == 1]
        peaks = peaks[pd.to_numeric(peaks.amplitude_deg, errors='coerce') >= 4.0]
        if len(peaks) < 2:
            raise ValueError(f'{sid}: 4°以上の適格折返し点が2点未満です')
        first, last = peaks.iloc[0], peaks.iloc[-1]
        center = float(summaries.loc[sid, 'envelope_center_deg'])
        time_s, measured = read_waveform(data_dir, row, center,
                                         float(first.peak_time_s), float(last.peak_time_s))
        axis = str(row.axis).upper()
        props = physics[axis]
        c_total = c_rod + c_ball
        predicted = integrate(time_s, float(first.centered_peak_angle_deg),
                              props['inertia_kg_m2'], props['restoring_n_m'],
                              tau_by_axis[axis], c_total)
        error = predicted - measured
        measured_cross = crossings(time_s - time_s[0], measured)
        predicted_cross = crossings(time_s - time_s[0], predicted)
        matched = min(len(measured_cross), len(predicted_cross))
        shift = predicted_cross[:matched] - measured_cross[:matched]
        row_metrics = {
            'segment_id': sid, 'axis': axis, 'direction': str(row.direction),
            'repetition': int(row.repetition), 'samples': len(time_s),
            'duration_s': float(time_s[-1] - time_s[0]),
            'inertia_kg_m2': props['inertia_kg_m2'], 'restoring_n_m': props['restoring_n_m'],
            'tau_n_m': tau_by_axis[axis], 'c_ball_n_m_s2_per_rad2': c_ball,
            'sample_rmse_deg': float(np.sqrt(np.mean(error**2))),
            'sample_mae_deg': float(np.mean(np.abs(error))),
            'sample_max_abs_error_deg': float(np.max(np.abs(error))),
            'measured_zero_crossings': len(measured_cross),
            'predicted_zero_crossings': len(predicted_cross), 'matched_zero_crossings': matched,
            'zero_crossing_shift_bias_ms': float(np.mean(shift) * 1e3) if matched else float('nan'),
            'zero_crossing_shift_rmse_ms': float(np.sqrt(np.mean(shift**2)) * 1e3) if matched else float('nan'),
            'zero_crossing_shift_max_abs_ms': float(np.max(np.abs(shift)) * 1e3) if matched else float('nan'),
        }
        metrics.append(row_metrics)
        records.append({**row_metrics, 'time_s': time_s, 'measured_deg': measured, 'predicted_deg': predicted})

    metric_table = pd.DataFrame(metrics)
    metric_table.to_csv(output / 'ihb05_ball_continuous_waveform_metrics.csv', index=False,
                        float_format='%.10g', encoding='utf-8-sig')
    summary = metric_table.groupby('axis').agg(
        waveforms=('segment_id', 'count'), sample_rmse_mean_deg=('sample_rmse_deg', 'mean'),
        sample_mae_mean_deg=('sample_mae_deg', 'mean'),
        zero_crossing_shift_rmse_mean_ms=('zero_crossing_shift_rmse_ms', 'mean'),
        zero_crossing_shift_bias_mean_ms=('zero_crossing_shift_bias_ms', 'mean'),
        matched_zero_crossings=('matched_zero_crossings', 'sum')).reset_index()
    summary.to_csv(output / 'ihb05_ball_continuous_waveform_summary.csv', index=False,
                   float_format='%.10g', encoding='utf-8-sig')
    make_figure(output / 'ihb05_ball_continuous_waveform_overlay.svg', records)
    run_settings = {
        'method': 'continuous full-waveform ODE integration, no state reset or phase alignment',
        'coefficient_source': 'IHB-05 method A c_ball; IHB-02 I/K plus measured ball increments; IHB-03 tau',
        'waveforms': len(records), 'c_ball_n_m_s2_per_rad2': c_ball,
        'c_rod_ball_n_m_s2_per_rad2': c_rod, 'tau_by_axis_n_m': tau_by_axis,
        'I_K_by_axis': physics, 'friction_epsilon_deg_s': EPSILON_DEG_S,
        'rtol': RTOL, 'atol': ATOL, 'max_step_natural_period_fraction': MAX_STEP_PERIOD_FRACTION,
        'initial_state': 'first accepted measured turning point; measured angle and zero angular speed',
        'comparison_interval': 'first through last eligible turning point with amplitude >= 4 deg',
        'zero_crossing_method': 'linear interpolation between adjacent sampled points',
    }
    (output / 'ihb05_ball_continuous_waveform_settings.json').write_text(
        json.dumps(run_settings, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'完了: {len(records)}波形。結果: {output}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
