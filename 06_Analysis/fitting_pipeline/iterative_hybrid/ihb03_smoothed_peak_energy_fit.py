#!/usr/bin/env python3
"""Exploratory IHB-03 fit using smoothed peak-amplitude envelopes."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
import ihb03_friction_identification as base


def summarize(fit):
    return dict(tau_n_m=fit['tau'],energy_rmse_mj=1000*fit['energy_rmse_j'],
                energy_pearson_r=fit['energy_r'],energy_r2=fit['energy_r2'],
                waveforms=fit['waveform_count'],half_cycles=fit['interval_count'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--turning-points-csv',type=Path,required=True)
    p.add_argument('--selection-csv',type=Path,required=True)
    p.add_argument('--condition-physics-csv',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    waves=base.assemble_intervals(base.read_csv(args.turning_points_csv),
        base.read_csv(args.selection_csv),base.read_csv(args.condition_physics_csv),3.0)
    raw_fits={axis:base.fit_tau_energy(waves,axis) for axis in ('IN','OUT')}
    smoothed_waves=[];peak_rows=[]
    for wave in waves:
        peaks={}
        for row in wave['intervals']:
            peaks[row['start_peak_number']]=(row['start_time_s'],row['start_angle_rad'])
            peaks[row['end_peak_number']]=(row['start_time_s']+row['observed_half_period_s'],row['observed_next_angle_rad'])
        ordered=sorted((k,*v) for k,v in peaks.items())
        runs=[]
        for item in ordered:
            if not runs or item[0] != runs[-1][-1][0]+1:runs.append([item])
            else:runs[-1].append(item)
        smooth_angles={}
        for run in runs:
            t=np.asarray([x[1] for x in run]); signed=np.asarray([x[2] for x in run])
            if len(run)>=5:
                # Smooth log-amplitude with the cubic smoothing-spline lambda
                # selected by generalized cross-validation; retain measured signs.
                sp=make_smoothing_spline(t,np.log(np.abs(signed)),lam=None)
                a=np.exp(sp(t))
            else:a=np.abs(signed)
            for (peak_no,tm,raw),amp in zip(run,a):
                smooth_angles[peak_no]=math.copysign(float(amp),raw)
                peak_rows.append(dict(segment_id=wave['segment_id'],axis=wave['axis'],
                    configuration=wave['configuration'],direction=wave['direction'],peak_number=peak_no,
                    peak_time_s=tm,raw_angle_deg=raw*180/math.pi,
                    smoothed_angle_deg=smooth_angles[peak_no]*180/math.pi,
                    raw_amplitude_deg=abs(raw)*180/math.pi,
                    smoothed_amplitude_deg=amp*180/math.pi,
                    smoothing_delta_deg=(smooth_angles[peak_no]-raw)*180/math.pi))
        new=dict(wave);new_intervals=[]
        for row in wave['intervals']:
            a0=smooth_angles[row['start_peak_number']];a1=smooth_angles[row['end_peak_number']]
            delta,c_basis,tau_basis=base.energy_bases(a0,wave['inertia_kg_m2'],wave['restoring_n_m'],a1)
            new_intervals.append(dict(row,start_angle_rad=a0,observed_next_angle_rad=a1,
                start_amplitude_deg=abs(a0)*180/math.pi,next_amplitude_deg=abs(a1)*180/math.pi,
                delta_energy_j=delta,c_basis=c_basis,tau_basis=tau_basis,
                rod_drag_loss_j=base.C_ROD*c_basis,tau_adjusted_energy_j=delta-base.C_ROD*c_basis))
        new['intervals']=new_intervals;smoothed_waves.append(new)
    smooth_fits={axis:base.fit_tau_energy(smoothed_waves,axis) for axis in ('IN','OUT')}
    summary=[]
    for axis in ('IN','OUT'):
        for method,fit in (('raw_observed_peaks',raw_fits[axis]),('GCV_smoothed_log_amplitude',smooth_fits[axis])):
            row=dict(axis=axis,method=method,**summarize(fit))
            summary.append(row)
    peak_delta=np.asarray([float(r['smoothing_delta_deg']) for r in peak_rows])
    for row in summary:
        axis_rows=[r for r in peak_rows if r['axis']==row['axis']]
        d=np.asarray([float(r['smoothing_delta_deg']) for r in axis_rows])
        row['peak_amplitude_smoothing_rmse_deg']=float(np.sqrt(np.mean(d*d)))
        row['peak_amplitude_smoothing_abs_p95_deg']=float(np.percentile(np.abs(d),95))
        row['peak_amplitude_smoothing_abs_max_deg']=float(np.max(np.abs(d)))
        row['spline_method']='cubic smoothing spline on log(abs(A)); lambda selected by GCV; sign retained'
    for path,records in ((args.output_dir/'ihb03_smoothed_peak_fit_summary.csv',summary),
                         (args.output_dir/'ihb03_smoothed_peak_angles.csv',peak_rows)):
        with path.open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    fig,axs=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True,constrained_layout=True)
    for axis,ax in zip(('IN','OUT'),axs):
        rr=[r for r in peak_rows if r['axis']==axis]
        x=[float(r['raw_amplitude_deg']) for r in rr];y=[float(r['smoothed_amplitude_deg']) for r in rr]
        ax.scatter(x,y,s=9,alpha=.35)
        lo=min(x+y);hi=max(x+y);ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set_title(axis);ax.set_xlabel('Observed peak amplitude |A| [deg]');ax.grid(alpha=.2)
    axs[0].set_ylabel('Smoothed peak amplitude [deg]')
    fig.suptitle('Observed vs GCV-smoothed peak amplitudes')
    fig.savefig(args.output_dir/'ihb03_smoothed_peak_angles.svg',bbox_inches='tight')
    fig.savefig(args.output_dir/'ihb03_smoothed_peak_angles.png',dpi=160,bbox_inches='tight')
    print(summary)

if __name__=='__main__':main()
