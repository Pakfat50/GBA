#!/usr/bin/env python3
"""Refit IHB-03 tau by resetting and integrating each measured half-cycle."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import ihb03_friction_identification as base

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def write_residual_svg(path, fits):
    """Write a compact SVG residual figure, sampling every 12th interval."""
    width,height=1080,690
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#444;stroke-width:1.2}</style>',
           '<text x="540" y="28" text-anchor="middle" font-size="18">IHB-03 direct ODE fit: state reset at each measured peak</text>']
    panels=[(70,75,'IN','angle_residual_deg',-1.5,1.5,'Next-peak angle residual [deg]'),
            (575,75,'IN','half_period_residual_ms',-30,30,'Half-period residual [ms]'),
            (70,380,'OUT','angle_residual_deg',-1.5,1.5,'Next-peak angle residual [deg]'),
            (575,380,'OUT','half_period_residual_ms',-30,30,'Half-period residual [ms]')]
    plotw,ploth=420,220
    for x,y,axis,key,ymin,ymax,ylabel in panels:
        parts.append(f'<text x="{x+plotw/2}" y="{y-12}" text-anchor="middle" font-size="14">{axis} — {ylabel}</text>')
        for j in range(5):
            gx=x+j*plotw/4; gy=y+j*ploth/4
            parts.append(f'<line class="grid" x1="{gx:.1f}" y1="{y}" x2="{gx:.1f}" y2="{y+ploth}"/>')
            parts.append(f'<line class="grid" x1="{x}" y1="{gy:.1f}" x2="{x+plotw}" y2="{gy:.1f}"/>')
            tick=4+j*10
            parts.append(f'<text x="{gx:.1f}" y="{y+ploth+17}" text-anchor="middle" font-size="10">{tick}</text>')
            val=ymax-j*(ymax-ymin)/4
            parts.append(f'<text x="{x-8}" y="{gy+3:.1f}" text-anchor="end" font-size="10">{val:.1f}</text>')
        zero=y+(ymax/(ymax-ymin))*ploth
        parts.append(f'<line x1="{x}" y1="{zero:.1f}" x2="{x+plotw}" y2="{zero:.1f}" stroke="#222" stroke-width="1.3"/>')
        parts.append(f'<line class="axis" x1="{x}" y1="{y}" x2="{x}" y2="{y+ploth}"/><line class="axis" x1="{x}" y1="{y+ploth}" x2="{x+plotw}" y2="{y+ploth}"/>')
        parts.append(f'<text x="{x+plotw/2}" y="{y+ploth+34}" text-anchor="middle" font-size="11">Measured start amplitude [deg]</text>')
        for ci,conf in enumerate((f'SP{i:02d}' for i in range(5))):
            rows=[r for i,r in enumerate(fits[axis]['rows']) if r['configuration']==conf and i%12==0]
            for r in rows:
                xv=max(4,min(54,float(r['start_amplitude_deg'])))
                v=float(r[key]); v=max(ymin,min(ymax,v))
                px=x+(xv-4)/50*plotw; py=y+(ymax-v)/(ymax-ymin)*ploth
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{COLORS[ci]}" fill-opacity=".68"/>')
    for i in range(5):
        xx=360+i*80
        parts.append(f'<circle cx="{xx}" cy="665" r="4" fill="{COLORS[i]}"/><text x="{xx+9}" y="669" font-size="11">SP0{i}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts),encoding='utf-8')


def direct_fit(waves, axis, tau_seed):
    selected = [w for w in waves if w["axis"] == axis]
    n_wave = len(selected)
    data = [(w, r) for w in selected for r in w["intervals"]
            if r["start_amplitude_deg"] >= 4.0 and r["next_amplitude_deg"] >= 4.0]
    weights = np.asarray([1.0 / (n_wave * sum(
        1 for rr in w["intervals"] if rr["start_amplitude_deg"] >= 4.0 and rr["next_amplitude_deg"] >= 4.0
    )) for w, r in data])

    def predict(w, r, tau):
        sol = base.solve_next_turning_point(
            r["start_angle_rad"], w["inertia_kg_m2"], w["restoring_n_m"],
            0.0, base.C_ROD, tau, base.EPSILON_DEG_S*math.pi/180,
            rtol=1e-4, angle_speed_atol=1e-7, energy_atol=1e-11,
            max_step_fraction=.2, max_periods=2.0)
        return sol

    observed = np.asarray([r["observed_next_angle_rad"] for w, r in data])
    times = np.asarray([r["observed_half_period_s"] for w, r in data])

    def residual(x):
        tau = float(x[0])
        pred = np.asarray([predict(w, r, tau)["next_angle_rad"] for w, r in data])
        return np.sqrt(weights) * (pred-observed)

    opt = least_squares(residual, np.asarray([tau_seed]), bounds=(0, 0.001),
                        x_scale=max(tau_seed, 1e-5), ftol=2e-7, xtol=2e-7, gtol=2e-7,
                        max_nfev=30)
    tau = float(opt.x[0])
    rows, grouped = [], {}
    for w, r in data:
        sol = predict(w, r, tau)
        err = sol["next_angle_rad"]-r["observed_next_angle_rad"]
        out = dict(r, tau_direct_ode_n_m=tau,
                   model_next_angle_rad=sol["next_angle_rad"],
                   model_next_angle_deg=sol["next_angle_rad"]*180/math.pi,
                   angle_residual_deg=err*180/math.pi,
                   model_half_period_s=sol["half_period_s"],
                   half_period_residual_ms=1000*(sol["half_period_s"]-r["observed_half_period_s"]))
        rows.append(out)
        grouped.setdefault(w["segment_id"], []).append(out)
    errors = np.asarray([r["angle_residual_deg"] for r in rows])
    t_errors = np.asarray([r["half_period_residual_ms"] for r in rows])
    rmse = float(np.sqrt(np.sum(weights*errors**2)/weights.sum()))
    trmse = float(np.sqrt(np.sum(weights*t_errors**2)/weights.sum()))
    per_wave = []
    for w in selected:
        rr = grouped[w["segment_id"]]
        e = np.asarray([r["angle_residual_deg"] for r in rr])
        te = np.asarray([r["half_period_residual_ms"] for r in rr])
        per_wave.append(dict(segment_id=w["segment_id"], axis=axis,
                             configuration=w["configuration"], direction=w["direction"],
                             intervals=len(rr), tau_direct_ode_n_m=tau,
                             angle_rmse_deg=float(np.sqrt(np.mean(e**2))),
                             angle_bias_deg=float(np.mean(e)),
                             half_period_rmse_ms=float(np.sqrt(np.mean(te**2)))))
    return dict(axis=axis,tau=tau,rows=rows,waveforms=selected,per_wave=per_wave,
                intervals=len(rows),angle_rmse_deg=rmse,half_period_rmse_ms=trmse,
                optimality=float(opt.optimality),nfev=int(opt.nfev),success=bool(opt.success))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--turning-points-csv',type=Path,required=True)
    p.add_argument('--selection-csv',type=Path,required=True)
    p.add_argument('--condition-physics-csv',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    waves=base.assemble_intervals(base.read_csv(a.turning_points_csv),base.read_csv(a.selection_csv),
                                 base.read_csv(a.condition_physics_csv),3.0)
    # Energy fit supplies only the starting value; final tau minimizes direct ODE peak errors.
    fits={ax:direct_fit(waves,ax,base.fit_tau_energy(waves,ax)['tau']) for ax in ('IN','OUT')}
    with open(a.output_dir/'ihb03_half_cycle_ode_parameters.csv','w',newline='',encoding='utf-8') as f:
        fields=['axis','tau_direct_ode_n_m','waveforms','half_cycles','angle_rmse_deg','half_period_rmse_ms','optimizer_success','function_evaluations']
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for ax,x in fits.items():wr.writerow(dict(axis=ax,tau_direct_ode_n_m=x['tau'],waveforms=len(x['waveforms']),half_cycles=x['intervals'],angle_rmse_deg=x['angle_rmse_deg'],half_period_rmse_ms=x['half_period_rmse_ms'],optimizer_success=x['success'],function_evaluations=x['nfev']))
    for filename,key in [('ihb03_half_cycle_ode_predictions.csv','rows'),('ihb03_half_cycle_ode_waveform_metrics.csv','per_wave')]:
        rows=[r for ax in ('IN','OUT') for r in fits[ax][key]]
        with open(a.output_dir/filename,'w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    write_residual_svg(a.output_dir/'ihb03_half_cycle_ode_fit.svg',fits)
    print(json.dumps([{k:x[k] for k in ('axis','tau','intervals','angle_rmse_deg','half_period_rmse_ms','success','nfev')} for x in fits.values()],indent=2))

if __name__=='__main__': main()
