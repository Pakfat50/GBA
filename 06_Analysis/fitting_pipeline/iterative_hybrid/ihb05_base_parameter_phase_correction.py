#!/usr/bin/env python3
"""I,Kの軸別基準値だけを動かし、連続自由振動の交点ずれを診断する。

目的:
  IHB-02で同定した軸別の基準慣性 I0 と基準復元係数 K0 を少し変えたとき、
  全スペーサ条件に共通する補正だけで自由振動の位相ずれが改善するかを調べる。

重要な前提:
  * τ、ロッド抗力係数、b=0、初期角、初期角速度、解析区間は固定する。
  * スペーサによる既知の I と K の増分はそのままにし、軸ごとの I0,K0 のみを
    共通倍率で調整する。条件ごとに別の係数をフィットしない。
  * 5%の探索幅は正式な不確かさではなく、改善可能性を見るための診断範囲。
    探索結果をそのまま採用値とはしない。
  * 各SP条件から1波形を検証用に取り分け、残りの波形で係数を選ぶ。
    このため、最適化に使っていない波形への改善も確認できる。

使い方 (リポジトリルートで実行):
  python 06_Analysis/fitting_pipeline/iterative_hybrid/ihb05_base_parameter_phase_correction.py

出力:
  波形別の補正前後指標、訓練/検証の集計、軸別の候補係数、比較図、設定JSON。
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from ihb04_continuous_waveform_validation import (
    ATOL, C_ROD, DATE, EPSILON_DEG_S, MAX_STEP_PERIOD_FRACTION, RTOL,
    crossings, read_csv,
)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / '06_Analysis/fitting_pipeline/results' / DATE / 'iterative_hybrid'
EXPLORATORY_FRACTION = 0.05
# 最適化の候補比較では、計測誤差より十分小さい数値誤差を保ちつつ計算を軽くする。
FIT_RTOL = 2e-7
FIT_ATOL = 2e-9


def build_records(root):
    """既存の承認済み波形と前処理頂点から、比較に必要なデータを一度読む。"""
    results = root / '06_Analysis/fitting_pipeline/results' / DATE
    turning = read_csv(results / 'hybrid_identification/01_preprocessing/turning_points.csv')
    summaries = read_csv(results / 'hybrid_identification/01_preprocessing/waveform_preprocessing.csv').set_index('segment_id')
    selection = read_csv(results / 'waveform_review/waveform_selection.csv')
    physics = read_csv(OUTPUT / 'ihb02_condition_predictions.csv')
    base = read_csv(OUTPUT / 'ihb02_base_parameters.csv').set_index('axis')
    tau_table = read_csv(OUTPUT / 'ihb03_monotone_peak_summary.csv')
    baseline_metrics = read_csv(OUTPUT / 'ihb04_continuous_waveform_metrics.csv').set_index('segment_id')
    peaks = {key: group.sort_values('peak_number').reset_index(drop=True)
             for key, group in turning.groupby('segment_id', sort=False)}
    selected = selection[(selection.use_for_fitting.astype(int) == 1) &
                         (selection.review_status.str.upper() == 'APPROVED') &
                         selection.configuration.str.startswith('SP')].copy()

    base_values = {
        axis: (float(row.base_inertia_kg_m2), float(row.base_restoring_n_m))
        for axis, row in base.iterrows()
    }
    current = {
        (str(row.axis), str(row.configuration)):
        (float(row.inertia_kg_m2), float(row.restoring_n_m))
        for row in physics.itertuples() if str(row.configuration).startswith('SP')
    }
    tau = {row.axis: float(row.tau_n_m) for row in tau_table.itertuples()
           if row.method == 'monotone_log_endpoint'}

    records = []
    data_cache = {}
    data_dir = root / '04_Data/05_Fitting' / DATE
    for row in selected.itertuples(index=False):
        sid = str(row.segment_id)
        points = peaks[sid]
        eligible = points[(points.eligible_for_later_stages.astype(int) == 1) &
                          (points.amplitude_deg.astype(float) >= 4.0)]
        if len(eligible) < 2:
            raise ValueError(f'{sid}: 採用頂点が2点未満です')
        first, last = eligible.iloc[0], eligible.iloc[-1]
        center = float(summaries.loc[sid, 'envelope_center_deg'])

        # 複数波形が同じログファイルにあるため、ファイルは最初の一度だけ読む。
        path = data_dir / str(row.data_file)
        cache_key = str(path)
        if cache_key not in data_cache:
            data_cache[cache_key] = pd.read_csv(path, encoding='utf-8-sig')
        # read_segmentが期待する形式で、キャッシュしたログの対象行だけを抽出。
        data = data_cache[cache_key]
        part = data.iloc[int(row.start_index):int(row.end_index) + 1]
        t_all = pd.to_numeric(part['systime[ms]'], errors='coerce').to_numpy(float) * 1e-3
        raw = pd.to_numeric(part[str(row.angle_column)], errors='coerce').to_numpy(float)
        raw = (raw + 180.0) % 360.0 - 180.0
        y_all = raw - center
        valid = np.isfinite(t_all) & np.isfinite(y_all)
        t_all, y_all = t_all[valid], y_all[valid]
        keep = (t_all >= float(first.peak_time_s)) & (t_all <= float(last.peak_time_s))
        time = t_all[keep]
        measured = y_all[keep]
        if len(time) < 2:
            raise ValueError(f'{sid}: 計算区間の計測点が不足しています')
        time = time - time[0]
        target_crossings = crossings(time, measured)
        axis, configuration = str(row.axis), str(row.configuration)
        inertia, restoring = current[(axis, configuration)]
        records.append({
            'segment_id': sid, 'axis': axis, 'configuration': configuration,
            'direction': str(row.direction), 'repetition': int(row.repetition),
            'time_s': time, 'measured_deg': measured,
            'initial_deg': float(first.centered_peak_angle_deg),
            'duration_s': float(time[-1]), 'tau_n_m': tau[axis],
            'current_inertia': inertia, 'current_restoring': restoring,
            'target_crossings_s': target_crossings,
        })

    # 各軸・各SP条件で再現番号の大きい1波形を検証用に固定する。
    # フィット後に都合よく検証波形を選ばないよう、規則を事前に決めておく。
    frame = pd.DataFrame([{k: r[k] for k in ('segment_id', 'axis', 'configuration', 'repetition')}
                          for r in records])
    held_out = set()
    for _, group in frame.groupby(['axis', 'configuration'], sort=True):
        chosen = group.sort_values(['repetition', 'segment_id']).iloc[-1]
        held_out.add(chosen.segment_id)

    # IHB-04で得た現在値の交点数を使い、最適化残差の長さを固定する。
    # これにより候補評価のためだけの基準ODE再積分を省き、IHB-04との定義も揃える。
    for record in records:
        record['split'] = 'validation' if record['segment_id'] in held_out else 'train'
        baseline = baseline_metrics.loc[record['segment_id']]
        record['fit_crossing_count'] = min(int(baseline.measured_zero_crossings),
                                           int(baseline.predicted_zero_crossings))
        if record['fit_crossing_count'] < 3:
            raise ValueError(f"{record['segment_id']}: 評価に使える交点が3個未満です")
    return records, base_values, current


def simulate(record, inertia, restoring, *, fit_mode=False):
    """固定τで運動方程式を積分し、計測時刻の予測角とゼロ交点時刻を返す。"""
    theta0 = math.radians(record['initial_deg'])
    epsilon = math.radians(EPSILON_DEG_S)
    period = 2.0 * math.pi * math.sqrt(inertia / restoring)
    duration = record['duration_s']

    def derivative(_t, state):
        theta, omega = state
        acceleration = (-restoring * math.sin(theta)
                        - C_ROD * abs(omega) * omega
                        - record['tau_n_m'] * math.tanh(omega / epsilon)) / inertia
        return omega, acceleration

    def zero(_t, state):
        return state[0]
    zero.terminal = False
    zero.direction = 0

    # 終端を一周期延ばし、区間末尾近くの交点が係数変更で少し遅れても検出する。
    # 探索中は交点だけが必要なので、密な補間出力を作らず許容誤差も少し緩める。
    # 最終指標の計算ではIHB-04と同じ高精度設定で全計測時刻を評価する。
    sol = solve_ivp(derivative, (0.0, duration + period), (theta0, 0.0),
                    method='DOP853', dense_output=not fit_mode, events=zero,
                    rtol=FIT_RTOL if fit_mode else RTOL,
                    atol=FIT_ATOL if fit_mode else ATOL,
                    max_step=period * (1.0/30.0 if fit_mode else MAX_STEP_PERIOD_FRACTION))
    if not sol.success:
        raise RuntimeError(f"{record['segment_id']}: ODE計算失敗: {sol.message}")
    predicted = None if fit_mode else np.rad2deg(sol.sol(record['time_s'])[0])
    event_times = sol.t_events[0]
    event_times = event_times[(event_times >= 0) & (event_times <= duration)]
    return predicted, event_times


def adjusted_condition(base, current, axis, configuration, factor_i, factor_k):
    """基準I0,K0だけを倍率変更し、既知の条件差分を維持して条件値を作る。"""
    i0, k0 = base[axis]
    ci, ck = current[(axis, configuration)]
    delta_i, delta_k = ci - i0, ck - k0
    return i0 * factor_i + delta_i, k0 * factor_k + delta_k


def optimization_residual(factors, records, axis_records, base, current):
    """交点時刻残差を連結する。各波形の総重みが等しくなるよう交点数で割る。"""
    factor_i, factor_k = map(float, factors)
    residuals = []
    for record in axis_records:
        inertia, restoring = adjusted_condition(base, current, record['axis'],
                                                record['configuration'], factor_i, factor_k)
        _, predicted_crossings = simulate(record, inertia, restoring, fit_mode=True)
        count = record['fit_crossing_count']
        target = record['target_crossings_s'][:count]
        if len(predicted_crossings) < count:
            # 探索端で振動周期が大きく変わってイベント不足となる候補には大きな罰則。
            predicted_crossings = np.r_[predicted_crossings,
                                        np.full(count-len(predicted_crossings), record['duration_s'] + 2.0)]
        error_ms = (predicted_crossings[:count] - target) * 1000.0
        residuals.extend(error_ms / math.sqrt(count))
    return np.asarray(residuals)


def measure(record, inertia, restoring):
    """既存IHB-04と同じ標本比較・交点補間で指標を計算する。"""
    predicted, _ = simulate(record, inertia, restoring)
    error = predicted - record['measured_deg']
    mt = record['target_crossings_s']
    pt = crossings(record['time_s'], predicted)
    count = min(len(mt), len(pt))
    shift = (pt[:count] - mt[:count]) * 1000.0
    return {
        'sample_rmse_deg': float(np.sqrt(np.mean(error ** 2))),
        'cross_rmse_ms': float(np.sqrt(np.mean(shift ** 2))) if count else float('nan'),
        'cross_bias_ms': float(np.mean(shift)) if count else float('nan'),
        'cross_max_abs_ms': float(np.max(np.abs(shift))) if count else float('nan'),
        'measured_crossings': int(len(mt)), 'predicted_crossings': int(len(pt)),
        'matched_crossings': int(count),
    }


def summarize(frame):
    """波形単位で等重みにした、軸・データ分割別の集計を作る。"""
    rows = []
    for (axis, split), group in frame.groupby(['axis', 'split'], sort=True):
        for label in ('baseline', 'adjusted'):
            rows.append({
                'axis': axis, 'split': split, 'fit': label, 'waveforms': len(group),
                'sample_rmse_mean_deg': group[f'{label}_sample_rmse_deg'].mean(),
                'cross_rmse_mean_ms': group[f'{label}_cross_rmse_ms'].mean(),
                'cross_bias_mean_ms': group[f'{label}_cross_bias_ms'].mean(),
                'max_abs_cross_shift_ms': group[f'{label}_cross_max_abs_ms'].max(),
            })
    return pd.DataFrame(rows)


def make_plot(path, wave_frame):
    """検証用波形のずれ改善と、全波形の角度誤差を1枚に並べる。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    validation = wave_frame[wave_frame.split == 'validation']
    x = np.arange(len(validation))
    width = .36
    axes[0].bar(x-width/2, validation.baseline_cross_rmse_ms, width, label='Current I,K', color='#777777')
    axes[0].bar(x+width/2, validation.adjusted_cross_rmse_ms, width, label='Adjusted I,K', color='#2878b5')
    axes[0].set_xticks(x, [f"{r.axis}-{r.configuration}" for r in validation.itertuples()], rotation=55, ha='right')
    axes[0].set_ylabel('Zero-crossing time RMSE [ms]')
    axes[0].set_title('Held-out waveform validation')
    axes[0].grid(axis='y', alpha=.25)
    axes[0].legend(frameon=False)

    all_rows = []
    for axis in ('IN', 'OUT'):
        group = wave_frame[wave_frame.axis == axis]
        for fit, col in [('baseline', 'baseline_sample_rmse_deg'), ('adjusted', 'adjusted_sample_rmse_deg')]:
            all_rows.append((axis, fit, group[col].mean()))
    positions = np.arange(2)
    for j, fit in enumerate(('baseline', 'adjusted')):
        vals = [next(v for a,f,v in all_rows if a == axis and f == fit) for axis in ('IN','OUT')]
        axes[1].bar(positions + (j-.5)*width, vals, width,
                    label='Current I,K' if fit == 'baseline' else 'Adjusted I,K',
                    color='#777777' if fit == 'baseline' else '#2878b5')
    axes[1].set_xticks(positions, ['IN','OUT'])
    axes[1].set_ylabel('Waveform angle RMSE [deg]')
    axes[1].set_title('All waveform angle error (diagnostic fit)')
    axes[1].grid(axis='y', alpha=.25)
    axes[1].legend(frameon=False)
    fig.suptitle('Common I0,K0 adjustment with tau and damping fixed', fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def main():
    records, base, current = build_records(ROOT)
    metrics = []
    parameter_rows = []
    adjusted_factors = {}

    for axis in ('IN', 'OUT'):
        axis_records = [r for r in records if r['axis'] == axis]
        train = [r for r in axis_records if r['split'] == 'train']
        # 係数は検証波形を使わずに決定する。
        fit = least_squares(optimization_residual, x0=np.array([1.0, 1.0]),
                            bounds=([1-EXPLORATORY_FRACTION]*2,
                                    [1+EXPLORATORY_FRACTION]*2),
                            args=(records, train, base, current),
                            x_scale='jac', max_nfev=18, diff_step=0.002,
                            ftol=2e-4, xtol=2e-4, gtol=2e-4)
        factor_i, factor_k = map(float, fit.x)
        adjusted_factors[axis] = (factor_i, factor_k)
        parameter_rows.append({
            'axis': axis, 'baseline_I0_kg_m2': base[axis][0],
            'baseline_K0_n_m': base[axis][1],
            'I0_multiplier': factor_i, 'K0_multiplier': factor_k,
            'adjusted_I0_kg_m2': base[axis][0]*factor_i,
            'adjusted_K0_n_m': base[axis][1]*factor_k,
            'I0_change_percent': (factor_i-1)*100,
            'K0_change_percent': (factor_k-1)*100,
            'train_waveforms': len(train), 'validation_waveforms': len(axis_records)-len(train),
            # 残差は波形ごとに交点数の平方根で割ってあるため、二乗和を
            # 波形数で割ると「各波形を同じ重みにしたRMSE」になる。
            'train_wave_equal_crossing_rmse_before_ms': float(
                np.linalg.norm(optimization_residual([1,1], records, train, base, current)) /
                math.sqrt(len(train))),
            'train_wave_equal_crossing_rmse_after_ms': float(
                np.linalg.norm(fit.fun) / math.sqrt(len(train))),
            'optimizer_success': bool(fit.success), 'optimizer_message': fit.message,
            'function_evaluations': int(fit.nfev),
            'bound_contact': bool(np.any(np.isclose(fit.x, 1-EXPLORATORY_FRACTION, atol=2e-4)) or
                                  np.any(np.isclose(fit.x, 1+EXPLORATORY_FRACTION, atol=2e-4))),
        })

    for record in records:
        before = measure(record, record['current_inertia'], record['current_restoring'])
        factor_i, factor_k = adjusted_factors[record['axis']]
        inertia, restoring = adjusted_condition(base, current, record['axis'],
                                                record['configuration'], factor_i, factor_k)
        after = measure(record, inertia, restoring)
        row = {k: record[k] for k in ('segment_id','axis','configuration','direction','repetition','split')}
        row.update({'baseline_inertia_kg_m2': record['current_inertia'],
                    'baseline_restoring_n_m': record['current_restoring'],
                    'adjusted_inertia_kg_m2': inertia,
                    'adjusted_restoring_n_m': restoring,
                    'I0_multiplier': factor_i, 'K0_multiplier': factor_k})
        for key, value in before.items(): row['baseline_'+key] = value
        for key, value in after.items(): row['adjusted_'+key] = value
        metrics.append(row)

    frame = pd.DataFrame(metrics)
    summary = summarize(frame)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT/'ihb05_ik_phase_correction_waveforms.csv', index=False,
                 float_format='%.10g', encoding='utf-8-sig')
    summary.to_csv(OUTPUT/'ihb05_ik_phase_correction_summary.csv', index=False,
                  float_format='%.10g', encoding='utf-8-sig')
    pd.DataFrame(parameter_rows).to_csv(OUTPUT/'ihb05_ik_phase_correction_parameters.csv', index=False,
                                        float_format='%.10g', encoding='utf-8-sig')
    make_plot(OUTPUT/'ihb05_ik_phase_correction_comparison.png', frame)
    settings = {
        'method': 'Fit axis-specific base I0,K0 multipliers to continuous zero-crossing times; known spacer increments preserved',
        'exploratory_multiplier_bounds': [1-EXPLORATORY_FRACTION,1+EXPLORATORY_FRACTION],
        'validation_split': 'one highest-repetition waveform held out per axis and spacer configuration',
        'objective': 'mean squared zero-crossing time error per training waveform, giving equal total weight to each waveform',
        'fixed': {'tau_from': 'IHB-03 monotone one-pass energy method', 'rod_drag': C_ROD,
                  'viscous_b': 0.0, 'epsilon_deg_s': EPSILON_DEG_S,
                  'initial_state': 'first accepted measured peak, zero angular velocity',
                  'phase_alignment': False},
        'note': 'The 5% range is an exploratory sensitivity range, not a measured parameter uncertainty or an accepted physical bound.',
    }
    (OUTPUT/'ihb05_ik_phase_correction_settings.json').write_text(
        json.dumps(settings, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('軸別候補:')
    print(pd.DataFrame(parameter_rows).to_string(index=False))
    print('\n集計:')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
