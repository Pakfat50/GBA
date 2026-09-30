#!/usr/bin/env python3
"""Compare IHB-03 half-cycle and waveform-total cumulative energy fits."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import ihb03_friction_identification as base


def metrics(obs, pred, weights):
    obs, pred, weights = map(lambda x: np.asarray(x, float), (obs, pred, weights))
    total = weights.sum()
    mean_obs = np.sum(weights * obs) / total
    mean_pred = np.sum(weights * pred) / total
    sse = np.sum(weights * (obs - pred) ** 2)
    sst = np.sum(weights * (obs - mean_obs) ** 2)
    cov = np.sum(weights * (obs - mean_obs) * (pred - mean_pred))
    var_obs = np.sum(weights * (obs - mean_obs) ** 2)
    var_pred = np.sum(weights * (pred - mean_pred) ** 2)
    return dict(rmse_mj=float(math.sqrt(sse / total) * 1000),
                pearson_r=float(cov / math.sqrt(var_obs * var_pred)) if var_obs and var_pred else float('nan'),
                r2=float(1 - sse / sst) if sst else float('nan'))


def fit_tau_from_wave_totals(waves):
    # One independent endpoint-to-endpoint total per waveform, equal waveform weight.
    x = np.asarray([sum(r['tau_basis'] for r in w['intervals']) for w in waves])
    y = np.asarray([sum(r['delta_energy_j'] - base.C_ROD * r['c_basis']
                        for r in w['intervals']) for w in waves])
    return max(0.0, float(np.sum(x * y) / np.sum(x * x)))


def fit_tau_half_cycles(waves):
    # Equal total weight per waveform; same one-pass estimator as IHB-03.
    n_wave = len(waves)
    x, y, wt = [], [], []
    for w in waves:
        for r in w['intervals']:
            x.append(r['tau_basis'])
            y.append(r['delta_energy_j'] - base.C_ROD * r['c_basis'])
            wt.append(1 / (n_wave * len(w['intervals'])))
    x, y, wt = map(lambda v: np.asarray(v, float), (x, y, wt))
    return max(0.0, float(np.sum(wt * x * y) / np.sum(wt * x * x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--turning-points-csv', type=Path, required=True)
    p.add_argument('--selection-csv', type=Path, required=True)
    p.add_argument('--condition-physics-csv', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    waves = base.assemble_intervals(base.read_csv(args.turning_points_csv),
                                    base.read_csv(args.selection_csv),
                                    base.read_csv(args.condition_physics_csv), 3.0)
    summaries, trajectory_rows, waveform_rows = [], [], []
    for axis in ('IN', 'OUT'):
        selected = sorted([w for w in waves if w['axis'] == axis], key=lambda w: w['segment_id'])
        tau_half = fit_tau_half_cycles(selected)
        tau_total = fit_tau_from_wave_totals(selected)
        taus = {'half_cycle_fit_tau': tau_half, 'waveform_total_fit_tau': tau_total}
        model_energy = {key: {'cum_obs': [], 'cum_pred': [], 'cum_weight': [],
                              'local_obs': [], 'local_pred': [], 'local_weight': [],
                              'end_obs': [], 'end_pred': [], 'end_weight': []}
                        for key in taus}
        for w in selected:
            cumulative_obs = cumulative_c = cumulative_r = 0.0
            n = len(w['intervals'])
            local_errors = {key: [] for key in taus}
            for idx, r in enumerate(w['intervals'], start=1):
                cumulative_obs += r['delta_energy_j']
                cumulative_c += r['c_basis']
                cumulative_r += r['tau_basis']
                out = dict(segment_id=w['segment_id'], axis=axis, configuration=w['configuration'],
                    direction=w['direction'], cumulative_interval=idx, end_peak_number=r['end_peak_number'],
                    endpoint_amplitude_deg=r['next_amplitude_deg'],
                    observed_half_cycle_loss_j=r['delta_energy_j'],
                    observed_cumulative_loss_j=cumulative_obs)
                for key, tau in taus.items():
                    local_pred = base.C_ROD * r['c_basis'] + tau * r['tau_basis']
                    cum_pred = base.C_ROD * cumulative_c + tau * cumulative_r
                    out[f'{key}_half_cycle_loss_j'] = local_pred
                    out[f'{key}_cumulative_loss_j'] = cum_pred
                    local_errors[key].append(local_pred - r['delta_energy_j'])
                    m = model_energy[key]
                    m['local_obs'].append(r['delta_energy_j']); m['local_pred'].append(local_pred)
                    m['local_weight'].append(1 / (len(selected) * n))
                    m['cum_obs'].append(cumulative_obs); m['cum_pred'].append(cum_pred)
                    m['cum_weight'].append(1 / (len(selected) * n))
                trajectory_rows.append(out)
            end_obs = cumulative_obs
            for key, tau in taus.items():
                end_pred = base.C_ROD * cumulative_c + tau * cumulative_r
                m = model_energy[key]
                m['end_obs'].append(end_obs); m['end_pred'].append(end_pred); m['end_weight'].append(1 / len(selected))
                waveform_rows.append(dict(segment_id=w['segment_id'], axis=axis,
                    configuration=w['configuration'], direction=w['direction'], intervals=n,
                    observed_total_loss_j=end_obs, tau_method=key, tau_n_m=tau,
                    predicted_total_loss_j=end_pred, total_loss_residual_j=end_pred-end_obs,
                    half_cycle_rmse_mj=1000*math.sqrt(np.mean(np.asarray(local_errors[key])**2))))
        # Leave one complete waveform out; estimate tau on the remaining waveforms.
        cv_obs, cv_pred_total, cv_weight = [], [], []
        fold_taus = {key: [] for key in taus}
        for held in selected:
            train = [w for w in selected if w['segment_id'] != held['segment_id']]
            for key, fitter in (('half_cycle_fit_tau', fit_tau_half_cycles),
                                ('waveform_total_fit_tau', fit_tau_from_wave_totals)):
                t = fitter(train); fold_taus[key].append(t)
                end_obs = sum(r['delta_energy_j'] for r in held['intervals'])
                end_pred = sum(base.C_ROD*r['c_basis'] + t*r['tau_basis'] for r in held['intervals'])
                cv_obs.append((key, held['segment_id'], end_obs))
                cv_pred_total.append((key, held['segment_id'], end_pred))
                cv_weight.append((key, held['segment_id'], 1/len(selected)))
        for key, tau in taus.items():
            m = model_energy[key]
            final = metrics(m['end_obs'], m['end_pred'], m['end_weight'])
            cumulative = metrics(m['cum_obs'], m['cum_pred'], m['cum_weight'])
            local = metrics(m['local_obs'], m['local_pred'], m['local_weight'])
            obs_cv = [v for k,s,v in cv_obs if k == key]
            pred_cv = [v for k,s,v in cv_pred_total if k == key]
            weight_cv = [v for k,s,v in cv_weight if k == key]
            cv = metrics(obs_cv, pred_cv, weight_cv)
            summaries.append(dict(axis=axis, tau_method=key, tau_n_m=tau,
                waveforms=len(selected), half_cycles=sum(len(w['intervals']) for w in selected),
                final_total_rmse_mj=final['rmse_mj'], final_total_pearson_r=final['pearson_r'], final_total_r2=final['r2'],
                cumulative_curve_rmse_mj=cumulative['rmse_mj'], cumulative_curve_pearson_r=cumulative['pearson_r'],
                cumulative_curve_r2=cumulative['r2'], half_cycle_rmse_mj=local['rmse_mj'],
                half_cycle_pearson_r=local['pearson_r'], half_cycle_r2=local['r2'],
                loo_final_total_rmse_mj=cv['rmse_mj'], loo_final_total_pearson_r=cv['pearson_r'],
                loo_final_total_r2=cv['r2'], loo_tau_min_n_m=min(fold_taus[key]),
                loo_tau_max_n_m=max(fold_taus[key])))
    for path, records in ((args.output_dir/'ihb03_cumulative_energy_parameters.csv', summaries),
                          (args.output_dir/'ihb03_cumulative_energy_predictions.csv', trajectory_rows),
                          (args.output_dir/'ihb03_cumulative_energy_waveform_totals.csv', waveform_rows)):
        with path.open('w', newline='', encoding='utf-8') as f:
            writer=csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(summaries)

if __name__ == '__main__':
    main()
