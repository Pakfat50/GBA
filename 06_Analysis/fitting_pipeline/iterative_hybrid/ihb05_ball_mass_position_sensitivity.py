#!/usr/bin/env python3
"""IHB-05の球質量・球位置の局所誤差感度を評価する。

実行例:
    python 06_Analysis/fitting_pipeline/iterative_hybrid/ihb05_ball_mass_position_sensitivity.py

基準入力の球質量と支点から球重心までの距離を中心に、質量±0.1 g、距離±1 mm
の3×3組合せを評価する。これらは測定器の精度が資料にないため設定した感度確認用の
シナリオ幅であり、実測不確かさの主張ではない。球形状・直径は同じままとし、球の
重心まわり慣性は質量に比例すると仮定する。

各シナリオで球ありI/Kを再計算し、方式Aのc_ballを改めて同定してから、12本の
自由振動を初期点から最後の適格点まで連続積分する。基準ケースとの差をCSVに出す。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIRECTORY))
from iterative_hybrid.ihb05_cball_identification import (
    ball_physics, fit_ball_peak_amplitudes, fit_one_shared_c,
    load_inputs, make_intervals, weighted_energy_metrics,
)
from iterative_hybrid.ihb05_ball_continuous_waveform_validation import (
    crossings, integrate, read_waveform,
)

DATE = '20260921'
BASE_MASS_KG = 0.0039
BASE_ARM_M = 0.174
MASS_OFFSETS_KG = (-0.0001, 0.0, 0.0001)
ARM_OFFSETS_M = (-0.001, 0.0, 0.001)
GRAVITY = 9.80665


def scenario_metrics(turning, selection, manifest, bases, fitted, summaries, data_dir,
                     base_c_ball, mass, arm):
    """一つの質量・位置条件について係数同定と全波形評価を実行する。"""
    changed = manifest.copy()
    mask = changed.configuration.astype(str).str.upper() == 'BALL'
    changed.loc[mask, 'component_mass_kg'] = mass
    # 同じ直径・一様な球の仮定では重心慣性は質量に比例する。
    changed.loc[mask, 'component_centroid_inertia_kg_m2'] *= mass / BASE_MASS_KG
    # Ball COM lies on the opposite arm in the signed restoring-torque convention.
    changed.loc[mask, 'signed_com_radius_m'] = -arm
    changed.loc[mask, 'force_lever_m'] = arm
    physics = ball_physics(changed, bases)
    intervals = make_intervals(turning, selection, physics, fitted)
    c_ball, _ = fit_one_shared_c(intervals)
    for row in intervals:
        row['sphere_coulomb_loss_j'] = row['tau_basis'] * {
            'IN': 7.945731419193821e-5, 'OUT': 1.769496760658e-4,
        }[row['axis']]
    energy = weighted_energy_metrics(intervals, c_ball)
    all_energy = next(x for x in energy if x['scope'] == 'ALL')

    selected = selection[(pd.to_numeric(selection.use_for_fitting, errors='coerce') == 1)
                         & (selection.review_status.astype(str).str.upper() == 'APPROVED')
                         & (selection.configuration.astype(str).str.upper() == 'BALL')]
    waveform_rmse, crossing_rmse, crossing_bias = [], [], []
    fixed_c_waveform_rmse, fixed_c_crossing_rmse = [], []
    waveform_axis = []
    for row in selected.itertuples(index=False):
        sid = str(row.segment_id)
        peaks = turning[(turning.segment_id.astype(str) == sid)
                        & (pd.to_numeric(turning.eligible_for_later_stages, errors='coerce') == 1)
                        & (pd.to_numeric(turning.amplitude_deg, errors='coerce') >= 4.0)]
        peaks = peaks.sort_values('peak_number')
        first, last = peaks.iloc[0], peaks.iloc[-1]
        time_s, measured = read_waveform(
            data_dir, row, float(summaries.loc[sid, 'envelope_center_deg']),
            float(first.peak_time_s), float(last.peak_time_s))
        axis = str(row.axis).upper()
        p = physics[axis]
        def predict(c_value):
            return integrate(time_s, float(first.centered_peak_angle_deg),
                             p['inertia_kg_m2'], p['restoring_n_m'],
                             {'IN': 7.945731419193821e-5, 'OUT': 1.769496760658e-4}[axis],
                             2.5664419286471763e-7 + c_value)
        # Full-pipeline output: use the c_ball reidentified for this scenario.
        predicted = predict(c_ball)
        waveform_rmse.append(float(np.sqrt(np.mean((predicted - measured) ** 2))))
        measured_x = crossings(time_s - time_s[0], measured)
        predicted_x = crossings(time_s - time_s[0], predicted)
        n = min(len(measured_x), len(predicted_x))
        shift = predicted_x[:n] - measured_x[:n]
        crossing_rmse.append(float(np.sqrt(np.mean(shift**2)) * 1e3))
        crossing_bias.append(float(np.mean(shift) * 1e3))
        # Isolate the I/K change: repeat the ODE comparison while holding c_ball at baseline.
        fixed_predicted = predict(base_c_ball)
        fixed_c_waveform_rmse.append(float(np.sqrt(np.mean((fixed_predicted - measured) ** 2))))
        fixed_cross = crossings(time_s - time_s[0], fixed_predicted)
        fixed_n = min(len(measured_x), len(fixed_cross))
        fixed_shift = fixed_cross[:fixed_n] - measured_x[:fixed_n]
        fixed_c_crossing_rmse.append(float(np.sqrt(np.mean(fixed_shift**2)) * 1e3))
        waveform_axis.append(axis)
    return {
        'mass_kg': mass, 'arm_m': arm,
        'IN_inertia_kg_m2': physics['IN']['inertia_kg_m2'],
        'OUT_inertia_kg_m2': physics['OUT']['inertia_kg_m2'],
        'IN_restoring_n_m': physics['IN']['restoring_n_m'],
        'OUT_restoring_n_m': physics['OUT']['restoring_n_m'],
        'c_ball_n_m_s2_per_rad2': c_ball,
        'energy_loss_r': all_energy['pearson_r'],
        'energy_loss_rmse_mj': all_energy['rmse_mj'],
        'full_waveform_rmse_mean_deg': float(np.mean(waveform_rmse)),
        'zero_crossing_shift_rmse_mean_ms': float(np.mean(crossing_rmse)),
        'zero_crossing_shift_bias_mean_ms': float(np.mean(crossing_bias)),
        'fixed_c_ball_waveform_rmse_mean_deg': float(np.mean(fixed_c_waveform_rmse)),
        'fixed_c_ball_zero_crossing_shift_rmse_mean_ms': float(np.mean(fixed_c_crossing_rmse)),
        'waveform_rmse_in_mean_deg': float(np.mean([v for v,a in zip(waveform_rmse,waveform_axis) if a=='IN'])),
        'waveform_rmse_out_mean_deg': float(np.mean([v for v,a in zip(waveform_rmse,waveform_axis) if a=='OUT'])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repository-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--date', default=DATE)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    results_root = root / '06_Analysis/fitting_pipeline/results'
    result_day = results_root / args.date
    output = (args.output_dir or result_day / 'iterative_hybrid/ihb05_cball').resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_dir = root / '04_Data/05_Fitting' / args.date

    turning, selection, manifest, bases = load_inputs(results_root, args.date)
    fitted, _ = fit_ball_peak_amplitudes(turning, selection)
    summaries = pd.read_csv(result_day / 'hybrid_identification/01_preprocessing/waveform_preprocessing.csv',
                            encoding='utf-8-sig').set_index('segment_id')
    settings = json.loads((output / 'ihb05_settings.json').read_text(encoding='utf-8'))
    base_c_ball = float(settings['c_ball_n_m_s2_per_rad2'])

    results = []
    for dm in MASS_OFFSETS_KG:
        for dr in ARM_OFFSETS_M:
            results.append(scenario_metrics(
                turning, selection, manifest, bases, fitted, summaries, data_dir,
                base_c_ball, BASE_MASS_KG + dm, BASE_ARM_M + dr))
    table = pd.DataFrame(results)
    nominal = table[(table.mass_kg == BASE_MASS_KG) & (table.arm_m == BASE_ARM_M)].iloc[0]
    table['mass_error_g'] = (table.mass_kg - BASE_MASS_KG) * 1e3
    table['arm_error_mm'] = (table.arm_m - BASE_ARM_M) * 1e3
    table['IN_inertia_change_percent'] = (table.IN_inertia_kg_m2 / nominal.IN_inertia_kg_m2 - 1) * 100
    table['OUT_inertia_change_percent'] = (table.OUT_inertia_kg_m2 / nominal.OUT_inertia_kg_m2 - 1) * 100
    table['IN_restoring_change_percent'] = (table.IN_restoring_n_m / nominal.IN_restoring_n_m - 1) * 100
    table['OUT_restoring_change_percent'] = (table.OUT_restoring_n_m / nominal.OUT_restoring_n_m - 1) * 100
    table['c_ball_change_percent'] = (table.c_ball_n_m_s2_per_rad2 / nominal.c_ball_n_m_s2_per_rad2 - 1) * 100
    table['waveform_rmse_change_deg'] = table.full_waveform_rmse_mean_deg - nominal.full_waveform_rmse_mean_deg
    table['crossing_rmse_change_ms'] = table.zero_crossing_shift_rmse_mean_ms - nominal.zero_crossing_shift_rmse_mean_ms
    table['crossing_bias_change_ms'] = table.zero_crossing_shift_bias_mean_ms - nominal.zero_crossing_shift_bias_mean_ms
    table.to_csv(output / 'ihb05_ball_mass_position_sensitivity.csv', index=False,
                 float_format='%.10g', encoding='utf-8-sig')
    compact = table[['mass_error_g','arm_error_mm','IN_inertia_change_percent','IN_restoring_change_percent',
                     'c_ball_change_percent','energy_loss_r','energy_loss_rmse_mj',
                     'full_waveform_rmse_mean_deg','waveform_rmse_change_deg',
                     'zero_crossing_shift_rmse_mean_ms','crossing_rmse_change_ms',
                     'zero_crossing_shift_bias_mean_ms','crossing_bias_change_ms',
                     'fixed_c_ball_waveform_rmse_mean_deg','fixed_c_ball_zero_crossing_shift_rmse_mean_ms']]
    print(compact.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
