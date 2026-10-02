#!/usr/bin/env python3
"""IHB-05: 球あり大振幅周期から取付位置を一変数補正し、係数と全波形を再評価する。

実行例 (リポジトリのルートから):
    python 06_Analysis/fitting_pipeline/iterative_hybrid/ihb05_ball_position_period_adjustment.py

この処理は、質量をマニフェストの実測値 3.9 g に固定し、支点から球重心までの
距離だけを現在値 174 mm の±3 mmで調整します。周期フィットはIHB-01と同じ
非線形振り子の式、代表振幅、25°以上・大振幅基準比±1%の採用条件を使います。
位置候補は0.01 mm刻みで走査します。走査部分は楕円積分と四則演算だけで、ODEは
最終候補の全波形確認にのみ使うため、反復数値最適化より計算負荷が小さくなります。

位置確定後、IHB-02の球なし基準I/Kへ球の寄与を加え直し、IHB-03のτを固定して
IHB-05方式Aのc_ballを再同定します。最後に計測12波形を係数固定で連続積分し、
補正前・補正後の周期フィット図、波形重ね描き、誤差表を出力します。
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

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIRECTORY))
from iterative_hybrid.ihb01_period_ratio import elliptic_k
from iterative_hybrid.ihb05_cball_identification import (
    AIR_DENSITY_KG_M3, SPHERE_AREA_M2, TAU_BY_AXIS_N_M,
    ball_physics, fit_ball_peak_amplitudes, fit_one_shared_c,
    load_inputs, make_intervals, weighted_energy_metrics,
)
from iterative_hybrid.ihb05_ball_continuous_waveform_validation import (
    crossings, integrate, read_waveform,
)

DATE = '20260921'
BASE_MASS_KG = 0.0039
BASE_ARM_M = 0.174
ARM_MIN_M = 0.171
ARM_MAX_M = 0.177
ARM_STEP_M = 0.00001  # 0.01 mm
PERIOD_RMSE_REFERENCE_PERCENT = 1.0  # Selection only, same as IHB-01.
ODR_TESTPOINT = 25.0


def measured_cycles(turning: pd.DataFrame, selection: pd.DataFrame):
    """Extract same-sign peak-to-peak periods, amplitudes and IHB-01 accept flags."""
    approved = selection[(pd.to_numeric(selection.use_for_fitting, errors='coerce') == 1)
                         & (selection.review_status.astype(str).str.upper() == 'APPROVED')
                         & (selection.configuration.astype(str).str.upper() == 'BALL')]
    selected_ids = set(approved.segment_id.astype(str))
    cycles = []
    for sid in approved.segment_id.astype(str):
        peaks = turning[(turning.segment_id.astype(str) == sid)
                        & (pd.to_numeric(turning.eligible_for_later_stages, errors='coerce') == 1)]
        peaks = peaks.sort_values('peak_number').reset_index(drop=True)
        rows = []
        for i in range(max(0, len(peaks) - 2)):
            start, end = peaks.iloc[i], peaks.iloc[i + 2]
            if str(start.peak_kind) != str(end.peak_kind):
                continue
            period = float(end.peak_time_s) - float(start.peak_time_s)
            if period <= 0:
                continue
            a0 = math.radians(abs(float(start.centered_peak_angle_deg)))
            a1 = math.radians(abs(float(end.centered_peak_angle_deg)))
            # Mean endpoint potential energy, converted back to equivalent amplitude.
            representative_rad = math.acos(max(-1.0, min(1.0, (math.cos(a0) + math.cos(a1)) / 2.0)))
            representative_deg = math.degrees(representative_rad)
            elliptic_parameter = math.sin(representative_rad / 2.0) ** 2
            h = 4.0 * elliptic_k(elliptic_parameter)
            observed_kappa = (h / period) ** 2
            rows.append({
                'segment_id': sid, 'axis': str(start.axis).upper(),
                'period_start_time_s': float(start.peak_time_s),
                'period_end_time_s': float(end.peak_time_s),
                'period_s': period, 'representative_amplitude_deg': representative_deg,
                'period_factor_h': h, 'observed_k_over_i_s2': observed_kappa,
            })
        if len(rows) < 5:
            raise ValueError(f'{sid}: IHB-01と同じ周期選別には5周期以上必要です (候補={len(rows)})')
        # IHB-01: high-amplitude reference from the largest-amplitude 10% (at least five).
        n_reference = max(5, math.ceil(0.10 * len(rows)))
        reference = float(np.median([x['observed_k_over_i_s2'] for x in
                                     sorted(rows, key=lambda x: x['representative_amplitude_deg'], reverse=True)[:n_reference]]))
        for row in rows:
            deviation = row['observed_k_over_i_s2'] / reference - 1.0
            row['high_amplitude_reference_k_over_i_s2'] = reference
            row['relative_deviation_from_reference'] = deviation
            row['accepted'] = int(row['representative_amplitude_deg'] >= ODR_TESTPOINT
                                  and abs(deviation) <= PERIOD_RMSE_REFERENCE_PERCENT / 100.0)
            cycles.append(row)
        if not any(x['accepted'] for x in rows):
            raise ValueError(f'{sid}: IHB-01条件 (振幅25°以上・基準比±1%) の採用周期がありません')
    if len(selected_ids) != approved.segment_id.astype(str).nunique():
        raise AssertionError('波形選択IDの重複があります')
    return pd.DataFrame(cycles), approved


def fit_observed_alpha(cycle_table: pd.DataFrame) -> dict:
    """IHB-01と同じゼロ切片・波形等重みの周期 fit から alpha=sqrt(I/K) を求める。"""
    accepted = cycle_table[cycle_table.accepted == 1].copy()
    by_axis = {}
    for axis, group in accepted.groupby('axis'):
        waveform_counts = group.groupby('segment_id').size()
        weights = group.segment_id.map(lambda sid: 1.0 / (len(waveform_counts) * waveform_counts[sid])).to_numpy(float)
        h = group.period_factor_h.to_numpy(float)
        t = group.period_s.to_numpy(float)
        alpha = float(np.sum(weights * h * t) / np.sum(weights * h * h))
        residual = t - alpha * h
        mean_t = float(np.sum(weights * t) / np.sum(weights))
        ss_res = float(np.sum(weights * residual**2))
        ss_total = float(np.sum(weights * (t - mean_t)**2))
        by_axis[axis] = {
            'axis': axis, 'waveforms': int(group.segment_id.nunique()),
            'accepted_periods': len(group), 'fitted_alpha_s': alpha,
            'observed_k_over_i_s2': 1.0 / alpha**2,
            'period_rmse_ms': float(np.sqrt(ss_res / np.sum(weights)) * 1000.0),
            'period_r_squared': 1.0 - ss_res / ss_total if ss_total > 0 else float('nan'),
            'period_bias_ms': float(np.sum(weights * residual) / np.sum(weights) * 1000.0),
        }
    return by_axis


def physics_at_arm(manifest: pd.DataFrame, bases: pd.DataFrame, arm_m: float):
    """Keep measured ball mass fixed and update only its lever arm in the manifest."""
    changed = manifest.copy()
    ball = changed.configuration.astype(str).str.upper() == 'BALL'
    changed.loc[ball, 'component_mass_kg'] = BASE_MASS_KG
    changed.loc[ball, 'signed_com_radius_m'] = -arm_m
    changed.loc[ball, 'force_lever_m'] = arm_m
    return ball_physics(changed, bases), changed


def period_scan(cycles: pd.DataFrame, manifest: pd.DataFrame, bases: pd.DataFrame):
    """Cheap scalar grid scan: calculate T predictions from I/K and elliptic integral only."""
    accepted = cycles[cycles.accepted == 1].copy()
    waveforms_by_axis = {axis: int(group.segment_id.nunique()) for axis, group in accepted.groupby('axis')}
    n_axes = len(waveforms_by_axis)
    # Each axis has equal weight; within an axis every waveform has equal total weight.
    cycle_count = accepted.groupby('segment_id').segment_id.transform('count').to_numpy(float)
    axis_count = accepted.axis.map(lambda a: waveforms_by_axis[a]).to_numpy(float)
    weights = 1.0 / (n_axes * axis_count * cycle_count)
    obs_period = accepted.period_s.to_numpy(float)
    h = accepted.period_factor_h.to_numpy(float)
    axes = accepted.axis.to_numpy(str)
    grid = np.round(np.arange(ARM_MIN_M, ARM_MAX_M + ARM_STEP_M / 2.0, ARM_STEP_M), 8)
    scan = []
    best = None
    for arm in grid:
        physics, _ = physics_at_arm(manifest, bases, float(arm))
        alpha = {axis: math.sqrt(physics[axis]['inertia_kg_m2'] /
                                 physics[axis]['restoring_n_m']) for axis in ('IN', 'OUT')}
        predicted = np.asarray([alpha[a] for a in axes]) * h
        residual = obs_period - predicted
        rmse = math.sqrt(float(np.sum(weights * residual**2) / np.sum(weights)))
        scan.append({'arm_m': float(arm), 'arm_mm': float(arm * 1000),
                     'weighted_period_rmse_ms': rmse * 1000.0,
                     'IN_predicted_alpha_s': alpha['IN'], 'OUT_predicted_alpha_s': alpha['OUT'],
                     'IN_predicted_k_over_i_s2': 1.0 / alpha['IN']**2,
                     'OUT_predicted_k_over_i_s2': 1.0 / alpha['OUT']**2})
        if best is None or rmse < best[0]:
            best = (rmse, float(arm), physics)
    return pd.DataFrame(scan), best[1], best[2], accepted, weights


def plot_period_fit(path: Path, accepted: pd.DataFrame, physics_nominal: dict,
                    physics_adjusted: dict, arm_nominal: float, arm_adjusted: float):
    """Plot measured selected periods and predicted amplitude-dependent curves by axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=False)
    colors = {'IN': '#1f77b4', 'OUT': '#2ca02c'}
    for ax, axis in zip(axes, ('IN', 'OUT')):
        group = accepted[accepted.axis == axis]
        ax.scatter(group.representative_amplitude_deg, group.period_s, s=22,
                   color=colors[axis], alpha=.75, label='Measured accepted cycles')
        amp = np.linspace(25.0, max(26.0, float(group.representative_amplitude_deg.max())), 250)
        h = np.asarray([4.0 * elliptic_k(math.sin(math.radians(a) / 2.0) ** 2) for a in amp])
        for physics, arm, style, label in (
            (physics_nominal, arm_nominal, '--', f'Nominal arm {arm_nominal*1000:.1f} mm'),
            (physics_adjusted, arm_adjusted, '-', f'Adjusted arm {arm_adjusted*1000:.2f} mm'),
        ):
            alpha = math.sqrt(physics[axis]['inertia_kg_m2'] / physics[axis]['restoring_n_m'])
            ax.plot(amp, alpha * h, linestyle=style, color='#d62728' if style == '--' else '#111111',
                    lw=1.6, label=label)
        ax.set_title(f'{axis}: measured period vs amplitude')
        ax.set_xlabel('Representative amplitude [deg]')
        ax.set_ylabel('Full-cycle period [s]')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('IHB-01 nonlinear-period fit applied to BALL waveforms (mass fixed)')
    fig.tight_layout()
    fig.savefig(path, format='svg', bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_waveforms(path: Path, records: list[dict]):
    """Plot all 12 measured and adjusted-ODE trajectories on one page."""
    matplotlib.rcParams['path.simplify'] = True
    matplotlib.rcParams['path.simplify_threshold'] = .5
    columns = 4
    rows = math.ceil(len(records) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, rows * 2.45), squeeze=False)
    for ax, rec in zip(axes.flat, records):
        t = rec['time_s'] - rec['time_s'][0]
        idx = np.unique(np.r_[np.arange(0, len(t), 5), len(t) - 1])
        ax.plot(t[idx], rec['measured_deg'][idx], color='#222222', lw=.65, label='Measured')
        ax.plot(t[idx], rec['predicted_deg'][idx], color='#d62728', lw=.8, label='Adjusted I/K ODE')
        ax.axhline(0, color='#888888', lw=.4)
        ax.set_title(f"{rec['segment_id']}  RMSE={rec['sample_rmse_deg']:.2f}°", fontsize=8)
        ax.set_xlabel('Elapsed time [s]', fontsize=7)
        ax.set_ylabel('Angle [deg]', fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        ax.grid(alpha=.2, lw=.35)
    for ax in axes.flat[len(records):]:
        ax.axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .974), ncol=2, frameon=False)
    fig.suptitle('BALL free decay after period-based mounting-position adjustment', y=1.025)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, format='svg', bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repository-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--date', default=DATE)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    results_root = root / '06_Analysis/fitting_pipeline/results'
    day = results_root / args.date
    out = (args.output_dir or day / 'iterative_hybrid/ihb05_cball').resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_dir = root / '04_Data/05_Fitting' / args.date

    turning, selection, manifest, bases = load_inputs(results_root, args.date)
    cycles, approved = measured_cycles(turning, selection)
    observed_fit = fit_observed_alpha(cycles)
    scan, arm_fit, adjusted_physics, accepted, weights = period_scan(cycles, manifest, bases)
    nominal_physics, _ = physics_at_arm(manifest, bases, BASE_ARM_M)
    scan.to_csv(out / 'ihb05_ball_position_period_scan.csv', index=False,
                float_format='%.12g', encoding='utf-8-sig')
    cycles.to_csv(out / 'ihb05_ball_position_period_cycles.csv', index=False,
                  float_format='%.12g', encoding='utf-8-sig')
    fit_rows = []
    for axis in ('IN', 'OUT'):
        obs = observed_fit[axis]
        for label, physics, arm in [('Nominal', nominal_physics, BASE_ARM_M),
                                    ('Period_adjusted', adjusted_physics, arm_fit)]:
            p = physics[axis]
            alpha = math.sqrt(p['inertia_kg_m2'] / p['restoring_n_m'])
            fit_rows.append({**obs, 'case': label, 'axis': axis, 'arm_mm': arm * 1000,
                             'inertia_kg_m2': p['inertia_kg_m2'],
                             'restoring_n_m': p['restoring_n_m'],
                             'model_alpha_s': alpha, 'model_k_over_i_s2': 1.0 / alpha**2,
                             'period_scale_residual_percent': (alpha / obs['fitted_alpha_s'] - 1.0) * 100})
    pd.DataFrame(fit_rows).to_csv(out / 'ihb05_ball_position_period_fit_by_axis.csv', index=False,
                                  float_format='%.12g', encoding='utf-8-sig')

    # At the selected geometry, refit c_ball using the approved IHB-05 Method A amplitudes.
    _, adjusted_manifest = physics_at_arm(manifest, bases, arm_fit)
    fitted_peaks, amplitude_diagnostics = fit_ball_peak_amplitudes(turning, selection)
    intervals = make_intervals(turning, selection, adjusted_physics, fitted_peaks)
    c_ball, _ = fit_one_shared_c(intervals)
    for row in intervals:
        row['sphere_coulomb_loss_j'] = TAU_BY_AXIS_N_M[row['axis']] * row['tau_basis']
    energy_metrics = weighted_energy_metrics(intervals, c_ball)
    pd.DataFrame(energy_metrics).to_csv(out / 'ihb05_position_adjusted_energy_metrics.csv', index=False,
                                        float_format='%.12g', encoding='utf-8-sig')

    settings_old = json.loads((out / 'ihb05_settings.json').read_text(encoding='utf-8'))
    c_rod = float(settings_old['c_rod_ball_exposed_length_adjusted'])
    tau = {axis: float(settings_old['tau_by_axis_n_m'][axis]) for axis in ('IN', 'OUT')}
    preprocessing = pd.read_csv(day / 'hybrid_identification/01_preprocessing/waveform_preprocessing.csv',
                                encoding='utf-8-sig').set_index('segment_id')
    selected = approved.reset_index(drop=True)
    records, metric_rows = [], []
    for row in selected.itertuples(index=False):
        sid = str(row.segment_id)
        peaks = turning[(turning.segment_id.astype(str) == sid)
                        & (pd.to_numeric(turning.eligible_for_later_stages, errors='coerce') == 1)
                        & (pd.to_numeric(turning.amplitude_deg, errors='coerce') >= 4.0)].sort_values('peak_number')
        if len(peaks) < 2:
            raise ValueError(f'{sid}: 全波形評価用の適格折返し点が不足しています')
        first, last = peaks.iloc[0], peaks.iloc[-1]
        time_s, measured = read_waveform(data_dir, row,
                                         float(preprocessing.loc[sid, 'envelope_center_deg']),
                                         float(first.peak_time_s), float(last.peak_time_s))
        axis = str(row.axis).upper()
        props = adjusted_physics[axis]
        predicted = integrate(time_s, float(first.centered_peak_angle_deg),
                              props['inertia_kg_m2'], props['restoring_n_m'], tau[axis], c_rod + c_ball)
        error = predicted - measured
        mx = crossings(time_s - time_s[0], measured)
        px = crossings(time_s - time_s[0], predicted)
        n = min(len(mx), len(px))
        shift = px[:n] - mx[:n]
        metric = {
            'segment_id': sid, 'axis': axis, 'samples': len(time_s), 'duration_s': time_s[-1] - time_s[0],
            'sample_rmse_deg': float(np.sqrt(np.mean(error**2))),
            'sample_mae_deg': float(np.mean(np.abs(error))),
            'sample_max_abs_error_deg': float(np.max(np.abs(error))),
            'measured_zero_crossings': len(mx), 'predicted_zero_crossings': len(px),
            'matched_zero_crossings': n,
            'zero_crossing_shift_bias_ms': float(np.mean(shift) * 1e3) if n else float('nan'),
            'zero_crossing_shift_rmse_ms': float(np.sqrt(np.mean(shift**2)) * 1e3) if n else float('nan'),
            'zero_crossing_shift_max_abs_ms': float(np.max(np.abs(shift)) * 1e3) if n else float('nan'),
            'inertia_kg_m2': props['inertia_kg_m2'], 'restoring_n_m': props['restoring_n_m'],
            'c_ball_n_m_s2_per_rad2': c_ball,
        }
        metric_rows.append(metric)
        records.append({**metric, 'time_s': time_s, 'measured_deg': measured, 'predicted_deg': predicted})
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(out / 'ihb05_position_adjusted_waveform_metrics.csv', index=False,
                        float_format='%.12g', encoding='utf-8-sig')
    summaries = []
    for axis, group in metric_table.groupby('axis'):
        summaries.append({'axis': axis, 'waveforms': len(group),
                          'sample_rmse_mean_deg': group.sample_rmse_deg.mean(),
                          'sample_mae_mean_deg': group.sample_mae_deg.mean(),
                          'zero_crossing_shift_rmse_mean_ms': group.zero_crossing_shift_rmse_ms.mean(),
                          'zero_crossing_shift_bias_mean_ms': group.zero_crossing_shift_bias_ms.mean(),
                          'matched_zero_crossings': int(group.matched_zero_crossings.sum())})
    summaries.append({'axis': 'ALL', 'waveforms': len(metric_table),
                      'sample_rmse_mean_deg': metric_table.sample_rmse_deg.mean(),
                      'sample_mae_mean_deg': metric_table.sample_mae_deg.mean(),
                      'zero_crossing_shift_rmse_mean_ms': metric_table.zero_crossing_shift_rmse_ms.mean(),
                      'zero_crossing_shift_bias_mean_ms': metric_table.zero_crossing_shift_bias_ms.mean(),
                      'matched_zero_crossings': int(metric_table.matched_zero_crossings.sum())})
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(out / 'ihb05_position_adjusted_waveform_summary.csv', index=False,
                         float_format='%.12g', encoding='utf-8-sig')

    plot_period_fit(out / 'ihb05_ball_position_period_fit.svg', accepted,
                    nominal_physics, adjusted_physics, BASE_ARM_M, arm_fit)
    plot_waveforms(out / 'ihb05_position_adjusted_waveform_overlay.svg', records)

    nominal_i = nominal_physics
    parameter_rows = []
    for axis in ('IN', 'OUT'):
        old, new = nominal_i[axis], adjusted_physics[axis]
        parameter_rows.append({
            'axis': axis, 'ball_mass_g_fixed': BASE_MASS_KG * 1000.0,
            'nominal_arm_mm': BASE_ARM_M * 1000.0, 'adjusted_arm_mm': arm_fit * 1000.0,
            'inertia_before_kg_m2': old['inertia_kg_m2'], 'inertia_after_kg_m2': new['inertia_kg_m2'],
            'inertia_change_percent': (new['inertia_kg_m2'] / old['inertia_kg_m2'] - 1) * 100,
            'restoring_before_n_m': old['restoring_n_m'], 'restoring_after_n_m': new['restoring_n_m'],
            'restoring_change_percent': (new['restoring_n_m'] / old['restoring_n_m'] - 1) * 100,
        })
    pd.DataFrame(parameter_rows).to_csv(out / 'ihb05_position_adjusted_coefficients.csv', index=False,
                                        float_format='%.12g', encoding='utf-8-sig')
    derived_settings = {
        'method': 'IHB-01 nonlinear period-amplitude fit on approved BALL cycles; one-dimensional bounded grid scan',
        'mass_fixed_kg': BASE_MASS_KG, 'arm_nominal_m': BASE_ARM_M,
        'arm_bounds_m': [ARM_MIN_M, ARM_MAX_M], 'arm_scan_step_m': ARM_STEP_M,
        'arm_fitted_m': arm_fit, 'c_ball_refitted_n_m_s2_per_rad2': c_ball,
        'equivalent_Cd_at_adjusted_arm': 2.0 * c_ball / (AIR_DENSITY_KG_M3 * SPHERE_AREA_M2 * arm_fit**3),
        'tau_by_axis_n_m': tau, 'period_selection': 'representative amplitude >=25 deg; ratio within 1% of per-waveform high-amplitude reference',
        'fit_waveforms': int(accepted.segment_id.nunique()), 'fit_cycles': int(len(accepted)),
        'energy_fit_metrics': energy_metrics, 'waveform_summary': summaries,
        'amplitude_fit_diagnostics': amplitude_diagnostics,
    }
    (out / 'ihb05_position_adjusted_settings.json').write_text(
        json.dumps(derived_settings, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(derived_settings, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
