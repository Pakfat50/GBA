#!/usr/bin/env python3
"""Compare fixed-c and simultaneous c/tau fits using IHB-03 energy bases."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
import ihb03_friction_identification as base


def weighted_metrics(obs, pred, weights):
    weights = np.asarray(weights, float)
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    total = weights.sum()
    mean = np.sum(weights * obs) / total
    sse = np.sum(weights * (obs - pred) ** 2)
    sst = np.sum(weights * (obs - mean) ** 2)
    return dict(rmse_mj=math.sqrt(sse / total) * 1000,
                pearson_r=float(np.corrcoef(obs, pred)[0, 1]),
                r2=1 - sse / sst)


def solve_nonnegative(x, y, weights):
    sw = np.sqrt(weights)
    return lsq_linear(x * sw[:, None], y * sw, bounds=(0, np.inf),
                      tol=1e-13, lsmr_tol=1e-13, max_iter=1000).x


def axis_rows(waves, axis):
    selected = [w for w in waves if w['axis'] == axis]
    return selected


def fit_axiswise(selected, fixed_c=None):
    n = len(selected)
    pairs = [(w, r) for w in selected for r in w['intervals']]
    weights = np.asarray([1 / (n * len(w['intervals'])) for w, r in pairs])
    obs = np.asarray([r['delta_energy_j'] for w, r in pairs])
    c_basis = np.asarray([r['c_basis'] for w, r in pairs])
    tau_basis = np.asarray([r['tau_basis'] for w, r in pairs])
    scale = base.C_ROD
    if fixed_c is None:
        x = np.column_stack((scale * c_basis, tau_basis))
        alpha, tau = solve_nonnegative(x, obs, weights)
        c = scale * alpha
        pred = x @ np.asarray([alpha, tau])
    else:
        c = fixed_c
        tau = max(0.0, np.sum(weights * tau_basis * (obs - c * c_basis)) /
                  np.sum(weights * tau_basis ** 2))
        pred = c * c_basis + tau * tau_basis
        alpha = c / scale
    basis_corr = float(np.sum(weights * (c_basis - np.average(c_basis, weights=weights)) *
                               (tau_basis - np.average(tau_basis, weights=weights))) /
                       math.sqrt(np.sum(weights * (c_basis - np.average(c_basis, weights=weights)) ** 2) *
                                 np.sum(weights * (tau_basis - np.average(tau_basis, weights=weights)) ** 2)))
    xw = np.column_stack((scale * c_basis, tau_basis)) * np.sqrt(weights)[:, None]
    xn = xw / np.linalg.norm(xw, axis=0)
    cond = float(np.linalg.cond(xn))
    return dict(c=c, tau=tau, alpha=alpha, pairs=pairs, weights=weights,
                obs=obs, pred=pred, c_basis=c_basis, tau_basis=tau_basis,
                basis_corr=basis_corr, normalized_condition_number=cond)


def cross_validate(selected, fit_kind):
    obs_all, pred_all, weights_all, params = [], [], [], []
    for held in selected:
        train = [w for w in selected if w['segment_id'] != held['segment_id']]
        fit = fit_axiswise(train, base.C_ROD if fit_kind == 'fixed_c_tau' else None)
        params.append((fit['c'], fit['tau']))
        for r in held['intervals']:
            obs_all.append(r['delta_energy_j'])
            if fit_kind == 'fixed_c_tau':
                c, tau = base.C_ROD, fit['tau']
            else:
                c, tau = fit['c'], fit['tau']
            pred_all.append(c * r['c_basis'] + tau * r['tau_basis'])
            weights_all.append(1 / (len(selected) * len(held['intervals'])))
    return weighted_metrics(obs_all, pred_all, weights_all), params


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--turning-points-csv', type=Path, required=True)
    p.add_argument('--selection-csv', type=Path, required=True)
    p.add_argument('--condition-physics-csv', type=Path, required=True)
    p.add_argument('--output-csv', type=Path, required=True)
    args = p.parse_args()
    waves = base.assemble_intervals(base.read_csv(args.turning_points_csv),
                                    base.read_csv(args.selection_csv),
                                    base.read_csv(args.condition_physics_csv), 3.0)
    results = []
    fits = {}
    for axis in ('IN', 'OUT'):
        selected = axis_rows(waves, axis)
        joint = fit_axiswise(selected)
        fixed = fit_axiswise(selected, base.C_ROD)
        fits[axis] = (joint, fixed, selected)
        cv_joint, pars_joint = cross_validate(selected, 'joint')
        cv_fixed, pars_fixed = cross_validate(selected, 'fixed_c_tau')
        for name, fit, cv, pars in (
            ('fixed_c_tau', fixed, cv_fixed, pars_fixed),
            ('axiswise_joint_c_tau', joint, cv_joint, pars_joint)):
            full = weighted_metrics(fit['obs'], fit['pred'], fit['weights'])
            results.append(dict(model=name, axis=axis, waveforms=len(selected), intervals=len(fit['obs']),
                c_rod_n_m_s2_per_rad2=fit['c'], tau_n_m=fit['tau'],
                fit_energy_rmse_mj=full['rmse_mj'], fit_energy_pearson_r=full['pearson_r'],
                fit_energy_r2=full['r2'], loo_energy_rmse_mj=cv['rmse_mj'],
                loo_energy_pearson_r=cv['pearson_r'], loo_energy_r2=cv['r2'],
                weighted_basis_correlation=fit['basis_corr'],
                normalized_design_condition_number=fit['normalized_condition_number'],
                loo_c_min=min(x[0] for x in pars), loo_c_max=max(x[0] for x in pars),
                loo_tau_min=min(x[1] for x in pars), loo_tau_max=max(x[1] for x in pars)))
    # Physically more restrictive check: one shared c, with separate IN/OUT tau.
    allpairs = [(axis, w, r) for axis in ('IN', 'OUT') for w in fits[axis][2] for r in w['intervals']]
    weights = np.asarray([1 / (2 * len(fits[a][2]) * len(w['intervals'])) for a, w, r in allpairs])
    obs = np.asarray([r['delta_energy_j'] for a, w, r in allpairs])
    x = np.asarray([[base.C_ROD * r['c_basis'], r['tau_basis'] if a == 'IN' else 0,
                     r['tau_basis'] if a == 'OUT' else 0] for a, w, r in allpairs])
    params = solve_nonnegative(x, obs, weights)
    shared_cv = {axis: ([], [], []) for axis in ('IN', 'OUT')}
    for held_axis in ('IN', 'OUT'):
        for held in fits[held_axis][2]:
            train_waves = {axis: [w for w in fits[axis][2]
                                  if not (axis == held_axis and w['segment_id'] == held['segment_id'])]
                           for axis in ('IN', 'OUT')}
            train_pairs = [(axis, w, r) for axis in ('IN', 'OUT')
                           for w in train_waves[axis] for r in w['intervals']]
            train_weights = np.asarray([1 / (2 * len(train_waves[axis]) * len(w['intervals']))
                                        for axis, w, r in train_pairs])
            train_obs = np.asarray([r['delta_energy_j'] for axis, w, r in train_pairs])
            train_x = np.asarray([[base.C_ROD * r['c_basis'],
                                   r['tau_basis'] if axis == 'IN' else 0,
                                   r['tau_basis'] if axis == 'OUT' else 0]
                                  for axis, w, r in train_pairs])
            fold_params = solve_nonnegative(train_x, train_obs, train_weights)
            sink = shared_cv[held_axis]
            for r in held['intervals']:
                sink[0].append(r['delta_energy_j'])
                sink[1].append(base.C_ROD * fold_params[0] * r['c_basis'] +
                               fold_params[1 if held_axis == 'IN' else 2] * r['tau_basis'])
                sink[2].append(1 / (len(fits[held_axis][2]) * len(held['intervals'])))
    for axis, tau_ix in (('IN', 1), ('OUT', 2)):
        selected = fits[axis][2]
        pairs = [(w, r) for w in selected for r in w['intervals']]
        ww = np.asarray([1 / (len(selected) * len(w['intervals'])) for w, r in pairs])
        yy = np.asarray([r['delta_energy_j'] for w, r in pairs])
        pp = np.asarray([base.C_ROD * params[0] * r['c_basis'] + params[tau_ix] * r['tau_basis'] for w, r in pairs])
        full = weighted_metrics(yy, pp, ww)
        cv = weighted_metrics(*shared_cv[axis])
        results.append(dict(model='shared_c_axiswise_tau', axis=axis, waveforms=len(selected), intervals=len(pairs),
            c_rod_n_m_s2_per_rad2=base.C_ROD * params[0], tau_n_m=params[tau_ix],
            fit_energy_rmse_mj=full['rmse_mj'], fit_energy_pearson_r=full['pearson_r'], fit_energy_r2=full['r2'],
            loo_energy_rmse_mj=cv['rmse_mj'], loo_energy_pearson_r=cv['pearson_r'], loo_energy_r2=cv['r2']))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    print(args.output_csv)

if __name__ == '__main__':
    main()
