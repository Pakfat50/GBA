#!/usr/bin/env python3
"""Fit IHB-03 Coulomb friction by half-cycle ODE energy loss."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.interpolate import PchipInterpolator
import ihb03_friction_identification as base


def weighted_metrics(obs, pred, weights):
    obs, pred, weights = map(lambda x: np.asarray(x, float), (obs, pred, weights))
    mean = np.sum(weights * obs) / np.sum(weights)
    sse = np.sum(weights * (obs - pred) ** 2)
    return dict(rmse_mj=float(np.sqrt(sse / weights.sum()) * 1000),
                pearson_r=float(np.corrcoef(obs, pred)[0, 1]),
                r2=float(1 - sse / np.sum(weights * (obs - mean) ** 2)))


def solve(w, row, tau):
    return base.solve_next_turning_point(
        row['start_angle_rad'], w['inertia_kg_m2'], w['restoring_n_m'],
        0.0, base.C_ROD, tau, base.EPSILON_DEG_S * math.pi / 180,
        rtol=1e-4, angle_speed_atol=1e-7, energy_atol=1e-11,
        max_step_fraction=.2, max_periods=2.0)


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
    fits = {}
    for axis in ('IN', 'OUT'):
        selected = [w for w in waves if w['axis'] == axis]
        pairs = [(w, r) for w in selected for r in w['intervals']]
        counts = {w['segment_id']: len(w['intervals']) for w in selected}
        weights = np.asarray([1 / (len(selected) * counts[w['segment_id']]) for w, r in pairs])
        observed_loss = np.asarray([r['delta_energy_j'] for w, r in pairs])
        observed_angle = np.asarray([r['observed_next_angle_rad'] for w, r in pairs])
        seed = base.fit_tau_energy(waves, axis)['tau']

        def predict_all(tau):
            return [solve(w, r, float(tau)) for w, r in pairs]

        def residual(x):
            results = predict_all(x[0])
            model_loss = np.asarray([s['integrated_loss_j'] for s in results])
            return 1000 * np.sqrt(weights) * (model_loss - observed_loss)  # scale residuals to mJ for optimizer tolerances

        opt = least_squares(residual, np.asarray([seed]), bounds=(0.0, 0.001),
                            x_scale=max(seed, 1e-5), ftol=1e-10, xtol=1e-10,
                            gtol=1e-10, max_nfev=30)
        tau = float(opt.x[0])
        results = predict_all(tau)
        model_loss = np.asarray([s['integrated_loss_j'] for s in results])
        model_angle = np.asarray([s['next_angle_rad'] for s in results])
        energy_rmse = float(np.sqrt(np.sum(weights * (model_loss-observed_loss)**2) / weights.sum()))
        energy_r = float(np.corrcoef(observed_loss, model_loss)[0, 1])
        angle_res_deg = (model_angle - observed_angle) * 180 / math.pi
        angle_rmse = float(np.sqrt(np.sum(weights * angle_res_deg**2) / weights.sum()))
        angle_r2 = weighted_metrics(observed_angle, model_angle, weights)['r2']
        rows, grouped = [], {}
        for (w, r), sol, mloss, mangle, weight in zip(pairs, results, model_loss, model_angle, weights):
            out = dict(r, tau_ode_energy_n_m=tau,
                       model_next_angle_rad=mangle, model_next_angle_deg=mangle*180/math.pi,
                       angle_residual_deg=(mangle-r['observed_next_angle_rad'])*180/math.pi,
                       model_integrated_energy_loss_j=mloss,
                       model_mechanical_energy_loss_j=sol['mechanical_loss_j'],
                       energy_closure_error_j=sol['energy_closure_error_j'],
                       energy_residual_j=mloss-r['delta_energy_j'],
                       model_half_period_s=sol['half_period_s'], fit_weight=weight)
            rows.append(out); grouped.setdefault(w['segment_id'], []).append(out)
        per_wave=[]
        for w in selected:
            rr=grouped[w['segment_id']]
            per_wave.append(dict(segment_id=w['segment_id'],axis=axis,configuration=w['configuration'],
                direction=w['direction'],intervals=len(rr),tau_ode_energy_n_m=tau,
                energy_rmse_mj=1000*math.sqrt(np.mean([r['energy_residual_j']**2 for r in rr])),
                angle_rmse_deg=math.sqrt(np.mean([r['angle_residual_deg']**2 for r in rr]))))
        fits[axis]=dict(axis=axis,tau=tau,waveforms=selected,rows=rows,per_wave=per_wave,
            intervals=len(rows),energy_rmse_j=energy_rmse,energy_r=energy_r,
            energy_r2=weighted_metrics(observed_loss,model_loss,weights)['r2'],angle_rmse_deg=angle_rmse,
            angle_r2=angle_r2,energy_closure_max_j=max(abs(s['energy_closure_error_j']) for s in results),
            optimizer_success=bool(opt.success),nfev=int(opt.nfev),optimizer_optimality=float(opt.optimality))
    # Five-fold CV grouped by complete waveforms. A 9-point ODE response grid
    # plus shape-preserving interpolation makes fold refits tractable. Held-out
    # predictions are then recomputed by exact ODE integration at each fold tau.
    for axis in ('IN', 'OUT'):
        selected = sorted(fits[axis]['waveforms'], key=lambda w: w['segment_id'])
        pairs = [(w, r) for w in selected for r in w['intervals']]
        seed = base.fit_tau_energy(waves, axis)['tau']
        grid = np.linspace(0.0, 2.0 * seed, 9)
        response = np.asarray([[solve(w, r, tau)['integrated_loss_j'] for w, r in pairs]
                               for tau in grid]).T
        interpolator = PchipInterpolator(grid, response, axis=1)
        fold_ids = [set(w['segment_id'] for i, w in enumerate(selected) if i % 5 == fold)
                    for fold in range(5)]
        cv_obs, cv_ode, cv_basis, cv_weight = [], [], [], []
        fold_tau = []
        for held_ids in fold_ids:
            train_indices = np.asarray([w['segment_id'] not in held_ids for w, r in pairs])
            train_pairs = [pairs[i] for i in np.where(train_indices)[0]]
            train_waves = {w['segment_id'] for w, r in train_pairs}
            counts = {sid: sum(w['segment_id'] == sid for w, r in train_pairs)
                      for sid in train_waves}
            train_weights = np.asarray([1 / (len(train_waves) * counts[w['segment_id']])
                                        for w, r in train_pairs])
            train_obs = np.asarray([r['delta_energy_j'] for w, r in train_pairs])
            c_basis = np.asarray([r['c_basis'] for w, r in train_pairs])
            tau_basis = np.asarray([r['tau_basis'] for w, r in train_pairs])
            basis_tau = max(0.0, np.sum(train_weights * tau_basis *
                                        (train_obs - base.C_ROD * c_basis)) /
                            np.sum(train_weights * tau_basis ** 2))
            train_index = np.where(train_indices)[0]
            def fold_objective(tau):
                residuals = interpolator(tau)[train_index] - train_obs
                return float(np.sum(train_weights * residuals ** 2))
            fold_fit = minimize_scalar(fold_objective, bounds=(grid[1], grid[-2]),
                                       method='bounded', options={'xatol': 1e-10})
            ode_tau = float(fold_fit.x)
            fold_tau.append(ode_tau)
            held_pairs = [(w, r) for w, r in pairs if w['segment_id'] in held_ids]
            held_wave_count = len(held_ids)
            held_counts = {sid: sum(w['segment_id'] == sid for w, r in held_pairs)
                           for sid in held_ids}
            for w, r in held_pairs:
                cv_obs.append(r['delta_energy_j'])
                cv_ode.append(solve(w, r, ode_tau)['integrated_loss_j'])
                cv_basis.append(solve(w, r, basis_tau)['integrated_loss_j'])
                cv_weight.append(1 / (held_wave_count * held_counts[w['segment_id']]))
        fits[axis]['cv_ode'] = weighted_metrics(cv_obs, cv_ode, cv_weight)
        fits[axis]['cv_basis'] = weighted_metrics(cv_obs, cv_basis, cv_weight)
        fits[axis]['cv_tau_min'] = min(fold_tau)
        fits[axis]['cv_tau_max'] = max(fold_tau)
    summary=[]
    for axis,fit in fits.items():
        summary.append(dict(axis=axis,tau_ode_energy_n_m=fit['tau'],waveforms=len(fit['waveforms']),
            half_cycles=fit['intervals'],energy_rmse_mj=fit['energy_rmse_j']*1000,
            energy_pearson_r=fit['energy_r'],energy_r2=fit['energy_r2'],
            angle_rmse_deg=fit['angle_rmse_deg'],angle_r2=fit['angle_r2'],
            max_energy_closure_error_j=fit['energy_closure_max_j'],
            optimizer_success=fit['optimizer_success'],function_evaluations=fit['nfev'],
            cv_energy_fit_rmse_mj=fit['cv_ode']['rmse_mj'],cv_energy_fit_r2=fit['cv_ode']['r2'],
            cv_onepass_tau_rmse_mj=fit['cv_basis']['rmse_mj'],cv_onepass_tau_r2=fit['cv_basis']['r2'],
            cv_energy_tau_min_n_m=fit['cv_tau_min'],cv_energy_tau_max_n_m=fit['cv_tau_max']))
    for path, records in ((args.output_dir/'ihb03_half_cycle_ode_energy_parameters.csv',summary),
                          (args.output_dir/'ihb03_half_cycle_ode_energy_predictions.csv',[r for a in ('IN','OUT') for r in fits[a]['rows']]),
                          (args.output_dir/'ihb03_half_cycle_ode_energy_waveform_metrics.csv',[r for a in ('IN','OUT') for r in fits[a]['per_wave']])):
        with path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(summary)

if __name__=='__main__':
    main()
