#!/usr/bin/env python3
"""Exploratory IHB-03 fit using smoothed peak-amplitude envelopes."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
import ihb03_friction_identification as base

SP_COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
MIN_PEAKS_PER_SMOOTHING_RUN = 5
MIN_AMPLITUDE_PEARSON_R = 0.99
MAX_AMPLITUDE_RMSE_DEG = 0.5


def summarize(fit):
    return dict(tau_n_m=fit['tau'],energy_rmse_mj=1000*fit['energy_rmse_j'],
                energy_pearson_r=fit['energy_r'],energy_r2=fit['energy_r2'],
                waveforms=fit['waveform_count'],half_cycles=fit['interval_count'])


def pearson(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else float('nan')


def validate_smoothing_input(segment_id, peak_numbers, t, signed_angles):
    """Fail early on invalid or insufficient peak sequences for the spline."""
    if len(t) < MIN_PEAKS_PER_SMOOTHING_RUN:
        raise ValueError(f'{segment_id}: only {len(t)} consecutive peaks; need at least {MIN_PEAKS_PER_SMOOTHING_RUN}')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(signed_angles)):
        raise ValueError(f'{segment_id}: non-finite peak time or angle')
    if np.any(np.diff(t) <= 0):
        raise ValueError(f'{segment_id}: peak times must be strictly increasing')
    if np.any(np.abs(signed_angles) <= 0):
        raise ValueError(f'{segment_id}: zero-amplitude peak cannot be log-smoothed')
    if np.any(np.diff(peak_numbers) != 1):
        raise ValueError(f'{segment_id}: smoothing run contains non-consecutive peaks')


def energy_metrics_for_condition(waves, axis, conf, tau):
    """Evaluate condition intervals with the already fitted axis-wide tau."""
    selected=[w for w in waves if w['axis']==axis and w['configuration']==conf and w['intervals']]
    obs=[];pred=[];weights=[]
    for w in selected:
        intervals=w['intervals'];weight=1/(len(selected)*len(intervals))
        for r in intervals:
            obs.append(float(r['delta_energy_j']))
            pred.append(base.C_ROD*float(r['c_basis'])+tau*float(r['tau_basis']))
            weights.append(weight)
    obs=np.asarray(obs);pred=np.asarray(pred);weights=np.asarray(weights)
    mean=np.sum(weights*obs)/np.sum(weights)
    sst=np.sum(weights*(obs-mean)**2);sse=np.sum(weights*(obs-pred)**2)
    return dict(r=pearson(obs,pred),r2=float(1-sse/sst) if sst else float('nan'))


def read_wave_segment(path,start,end,column):
    """Read a reviewed waveform slice; manifest indices are zero-based data rows."""
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    rows=rows[int(start):int(end)+1]
    t=np.asarray([float(r['systime[ms]'])*1e-3 for r in rows])
    angle=np.asarray([float(r[column]) for r in rows])
    return t,angle


def write_condition_waveforms(path, peak_rows, selection, points, repository_root):
    """Plot every accepted raw waveform with its fitted +/- amplitude envelope."""
    pby={}
    for r in points:pby.setdefault(r['segment_id'],[]).append(r)
    fig,axs=plt.subplots(5,2,figsize=(15,17),constrained_layout=True)
    for si,conf in enumerate([f'SP{i:02d}' for i in range(5)]):
        for ai,axis in enumerate(('IN','OUT')):
            ax=axs[si,ai]
            waves=[r for r in selection if r['axis']==axis and r['configuration']==conf
                   and r['use_for_fitting']=='1' and r['review_status']=='APPROVED']
            for wi,sel in enumerate(sorted(waves,key=lambda x:x['segment_id'])):
                sid=sel['segment_id'];prs=sorted(pby.get(sid,[]),key=lambda x:int(x['peak_number']))
                if not prs:continue
                peak_times=np.asarray([float(x['peak_time_s']) for x in prs])
                t0=float(peak_times[0]);t1=float(peak_times[-1])
                center=float(np.median([float(x['peak_angle_deg'])-float(x['centered_peak_angle_deg']) for x in prs]))
                fpath=repository_root/'04_Data/05_Fitting/20260921'/sel['data_file']
                t,raw=read_wave_segment(fpath,sel['start_index'],sel['end_index'],sel['angle_column'])
                # Sensor angle is modulo 360 degrees; unwrap to the physical range then
                # shift by the same waveform-specific center used for peak amplitudes.
                angle=(raw+180)%360-180-center
                use=(t>=t0)&(t<=t1)
                color=plt.get_cmap('tab10')(wi%10)
                label=f"{sel['direction']}-R{int(sel['repetition']):02d}"
                ax.plot(t[use]-t0,angle[use],color=color,lw=.65,alpha=.46,label=label,rasterized=True)
                fit_peaks=[x for x in peak_rows if x['segment_id']==sid]
                fit_peaks=sorted(fit_peaks,key=lambda x:int(x['peak_number']))
                ft=np.asarray([float(x['peak_time_s']) for x in fit_peaks])
                fa=np.asarray([float(x['smoothed_amplitude_deg']) for x in fit_peaks])
                # Recreate each GCV spline on contiguous peak-number runs and
                # evaluate the actual fitted function along the raw trace.
                amp=np.full(np.count_nonzero(use),np.nan)
                for run in np.split(np.arange(len(ft)),np.where(np.diff([int(x['peak_number']) for x in fit_peaks])!=1)[0]+1):
                    if not len(run): continue
                    if len(run)>=5:
                        sp=make_smoothing_spline(ft[run],np.log(fa[run]),lam=None)
                        valid=(t[use]>=ft[run][0])&(t[use]<=ft[run][-1])
                        amp[valid]=np.exp(sp(t[use][valid]))
                    else:
                        valid=(t[use]>=ft[run][0])&(t[use]<=ft[run][-1])
                        amp[valid]=np.interp(t[use][valid],ft[run],fa[run])
                ax.plot(t[use]-t0,amp,color=color,lw=1.05,ls='--',alpha=.95,label='_nolegend_')
                ax.plot(t[use]-t0,-amp,color=color,lw=1.05,ls='--',alpha=.95,label='_nolegend_')
            ax.axhline(0,color='#555',lw=.55)
            ax.set_title(f'{axis} / {conf}  (n={len(waves)})')
            ax.set_xlabel('Time from first fitted peak [s]');ax.set_ylabel('Centered angle [deg]')
            ax.grid(alpha=.2)
            if waves:ax.legend(loc='upper right',fontsize=6,ncol=2,framealpha=.8)
    fig.suptitle('Free-decay waveforms with GCV-smoothed amplitude envelopes')
    fig.savefig(path,bbox_inches='tight',dpi=45)
    fig.savefig(path.with_suffix('.png'),bbox_inches='tight',dpi=90)
    plt.close(fig)


def write_smoothed_ode_check(path, fits):
    """Same four-panel residual layout as the independent ODE check figure."""
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex='col',constrained_layout=True)
    for row,axis in enumerate(('IN','OUT')):
        for col,(key,ylab,ylim) in enumerate((('angle_residual_deg','Next-peak angle residual [deg]',(-1.5,1.5)),
                                               ('half_period_residual_ms','Half-period residual [ms]',(-30,30)))):
            ax=axs[row,col]
            for si,conf in enumerate([f'SP{i:02d}' for i in range(5)]):
                rr=[r for r in fits[axis]['rows'] if r['configuration']==conf]
                rr=rr[::12] if len(rr)>12 else rr
                x=np.asarray([float(r['start_amplitude_deg']) for r in rr]);y=np.asarray([float(r[key]) for r in rr])
                ax.scatter(x,y,s=14,color=SP_COLORS[si],alpha=.72,label=conf)
            ax.axhline(0,color='#222',lw=1)
            ax.set_ylim(*ylim);ax.set_xlim(4,54);ax.grid(alpha=.22)
            ax.set_title(f'{axis}: {ylab}');ax.set_ylabel(ylab)
            ax.set_xlabel('Fitted start amplitude [deg]')
            if row==0 and col==0:ax.legend(fontsize=8,ncol=2)
    fig.suptitle('ODE check using GCV-smoothed peak amplitudes')
    fig.savefig(path,bbox_inches='tight');plt.close(fig)



def write_smoothed_fit_overview(path, waves, ode_fits):
    """Plot smoothed one-pass energy predictions and fixed-tau ODE angle residuals."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row, axis in enumerate(('IN', 'OUT')):
        tau = ode_fits[axis]['tau']
        selected = [w for w in waves if w['axis'] == axis]
        for col, kind in enumerate(('energy', 'ode')):
            ax = axs[row, col]
            for si, conf in enumerate([f'SP{i:02d}' for i in range(5)]):
                if kind == 'energy':
                    rr = [r for w in selected if w['configuration'] == conf for r in w['intervals']]
                    x = np.asarray([1000*float(r['delta_energy_j']) for r in rr])
                    y = np.asarray([1000*(base.C_ROD*float(r['c_basis']) + tau*float(r['tau_basis'])) for r in rr])
                    title = f'{axis}: one-pass energy fit'
                    ax.set_xlabel('Observed ΔE [mJ]'); ax.set_ylabel('Model cC + τR [mJ]')
                    if len(x):
                        lo=min(0,float(x.min()),float(y.min())); hi=max(float(x.max()),float(y.max()))
                        ax.plot([lo,hi],[lo,hi],'k--',lw=1)
                else:
                    rr = [r for r in ode_fits[axis]['rows'] if r['configuration'] == conf]
                    x = np.asarray([float(r['start_amplitude_deg']) for r in rr])
                    y = np.asarray([float(r['angle_residual_deg']) for r in rr])
                    title = f'{axis}: fixed-τ ODE validation'
                    ax.set_xlabel('Measured start amplitude [deg]'); ax.set_ylabel('ODE next-peak residual [deg]')
                    ax.axhline(0,color='#222',lw=1)
                ax.scatter(x,y,s=7,color=SP_COLORS[si],alpha=.55,label=conf,rasterized=True)
                ax.set_title(title); ax.grid(alpha=.22)
                if row == 0 and col == 0: ax.legend(fontsize=8,ncol=3)
    fig.suptitle('Smoothed peak amplitudes: one-pass energy fit and fixed-τ ODE check')
    fig.savefig(path,bbox_inches='tight'); plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--turning-points-csv',type=Path,required=True)
    p.add_argument('--selection-csv',type=Path,required=True)
    p.add_argument('--condition-physics-csv',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--repository-root',type=Path,default=Path.cwd(),
                   help='Repository root containing 04_Data/05_Fitting/20260921/Raw')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    waves=base.assemble_intervals(base.read_csv(args.turning_points_csv),
        base.read_csv(args.selection_csv),base.read_csv(args.condition_physics_csv),3.0)
    raw_fits={axis:base.fit_tau_energy(waves,axis) for axis in ('IN','OUT')}
    smoothed_waves=[];peak_rows=[];run_functions={}
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
            validate_smoothing_input(wave['segment_id'],np.asarray([x[0] for x in run]),t,signed)
            if len(run)>=MIN_PEAKS_PER_SMOOTHING_RUN:
                # Smooth log-amplitude with the cubic smoothing-spline lambda
                # selected by generalized cross-validation; retain measured signs.
                sp=make_smoothing_spline(t,np.log(np.abs(signed)),lam=None)
                a=np.exp(sp(t))
                if not np.all(np.isfinite(a)) or np.any(a <= 0):
                    raise ValueError(f'{wave[\'segment_id\']}: spline produced invalid amplitudes')
                run_functions[(wave['segment_id'],int(run[0][0]))]=(float(t[0]),float(t[-1]),sp)
            else: raise AssertionError('minimum smoothing-run length gate was bypassed')
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
    # Gate every waveform before any smoothed-amplitude tau fit is attempted.
    per_wave=[];gate_failures=[]
    for wave in waves:
        rr=[r for r in peak_rows if r['segment_id']==wave['segment_id']]
        obs=np.asarray([float(r['raw_amplitude_deg']) for r in rr]);fit=np.asarray([float(r['smoothed_amplitude_deg']) for r in rr])
        e=fit-obs;fit_r=pearson(obs,fit);fit_rmse=float(np.sqrt(np.mean(e*e))) if len(e) else float('nan')
        per_wave.append(dict(segment_id=wave['segment_id'],axis=wave['axis'],configuration=wave['configuration'],
            direction=wave['direction'],peaks=len(rr),pearson_r=fit_r,
            amplitude_rmse_deg=fit_rmse,amplitude_bias_deg=float(np.mean(e)) if len(e) else float('nan')))
        reasons=[]
        if len(rr)<MIN_PEAKS_PER_SMOOTHING_RUN: reasons.append(f'peaks={len(rr)} < {MIN_PEAKS_PER_SMOOTHING_RUN}')
        if not np.isfinite(fit_r) or fit_r<MIN_AMPLITUDE_PEARSON_R: reasons.append(f'R={fit_r:.6g} < {MIN_AMPLITUDE_PEARSON_R}')
        if not np.isfinite(fit_rmse) or fit_rmse>MAX_AMPLITUDE_RMSE_DEG: reasons.append(f'RMSE={fit_rmse:.6g} deg > {MAX_AMPLITUDE_RMSE_DEG} deg')
        if reasons: gate_failures.append(f"{wave['segment_id']}: " + '; '.join(reasons))
    if gate_failures:
        raise RuntimeError('Amplitude-fit quality gate failed; tau was not fitted:\\n'+'\\n'.join(gate_failures))

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

    # Per-waveform fit quality, then condition summaries that give each waveform
    # one vote so longer records cannot dominate the reported R values.
    condition_rows=[]
    for axis in ('IN','OUT'):
        for conf in [f'SP{i:02d}' for i in range(5)]:
            ww=[r for r in per_wave if r['axis']==axis and r['configuration']==conf]
            if not ww:continue
            peak_group=[r for r in peak_rows if r['axis']==axis and r['configuration']==conf]
            obs=np.asarray([float(r['raw_amplitude_deg']) for r in peak_group]);fit=np.asarray([float(r['smoothed_amplitude_deg']) for r in peak_group])
            de=[]
            for w0,w1 in zip([w for w in waves if w['axis']==axis and w['configuration']==conf],
                             [w for w in smoothed_waves if w['axis']==axis and w['configuration']==conf]):
                for a,b in zip(w0['intervals'],w1['intervals']):de.append(float(b['delta_energy_j'])-float(a['delta_energy_j']))
            d=np.asarray(de);raw_fit=base.fit_tau_energy(waves,axis,[conf]);smooth_fit=base.fit_tau_energy(smoothed_waves,axis,[conf])
            raw_global_metrics=energy_metrics_for_condition(waves,axis,conf,raw_fits[axis]['tau'])
            smooth_global_metrics=energy_metrics_for_condition(smoothed_waves,axis,conf,smooth_fits[axis]['tau'])
            condition_rows.append(dict(axis=axis,configuration=conf,waveforms=len(ww),peaks=len(peak_group),
                pearson_r_mean_waveform=float(np.nanmean([r['pearson_r'] for r in ww])),
                pearson_r_min_waveform=float(np.nanmin([r['pearson_r'] for r in ww])),
                pearson_r_max_waveform=float(np.nanmax([r['pearson_r'] for r in ww])),
                amplitude_rmse_mean_waveform_deg=float(np.mean([r['amplitude_rmse_deg'] for r in ww])),
                amplitude_fit_pearson_r_pooled=pearson(obs,fit),
                amplitude_smoothing_rms_deg=float(np.sqrt(np.mean([(float(r['smoothed_amplitude_deg'])-float(r['raw_amplitude_deg']))**2 for r in peak_group]))),
                delta_energy_change_rms_mj=float(np.sqrt(np.mean(d*d))*1000),
                energy_r_raw_pooled_tau=raw_global_metrics['r'],
                energy_r_smoothed_pooled_tau=smooth_global_metrics['r'],
                energy_r_change_pooled_tau=smooth_global_metrics['r']-raw_global_metrics['r'],
                energy_r2_raw_pooled_tau=raw_global_metrics['r2'],
                energy_r2_smoothed_pooled_tau=smooth_global_metrics['r2'],
                energy_r2_change_pooled_tau=smooth_global_metrics['r2']-raw_global_metrics['r2'],
                tau_raw_n_m=raw_fit['tau'],tau_smoothed_n_m=smooth_fit['tau'],
                tau_change_percent=100*(smooth_fit['tau']/raw_fit['tau']-1),
                energy_rmse_raw_mj=1000*raw_fit['energy_rmse_j'],energy_rmse_smoothed_mj=1000*smooth_fit['energy_rmse_j']))

    # ODE model check on the fitted peak-angle sequence. Use the tau obtained by
    # the same energy-basis fit on the smoothed amplitudes and reset at each peak.
    selection_by={r['segment_id']:r for r in base.read_csv(args.selection_csv)}
    ode_fits={}
    for axis in ('IN','OUT'):
        tau=smooth_fits[axis]['tau'];rows=[];bywave={}
        for wave in [w for w in smoothed_waves if w['axis']==axis]:
            for rr in wave['intervals']:
                sol=base.solve_next_turning_point(rr['start_angle_rad'],wave['inertia_kg_m2'],wave['restoring_n_m'],
                    0.0,base.C_ROD,tau,base.EPSILON_DEG_S*math.pi/180,
                    rtol=1e-4,angle_speed_atol=1e-7,energy_atol=1e-11,max_step_fraction=.2,max_periods=2.0)
                out=dict(rr,model_next_angle_rad=sol['next_angle_rad'],
                    model_next_angle_deg=sol['next_angle_rad']*180/math.pi,
                    angle_residual_deg=(sol['next_angle_rad']-rr['observed_next_angle_rad'])*180/math.pi,
                    model_half_period_s=sol['half_period_s'],
                    half_period_residual_ms=1000*(sol['half_period_s']-rr['observed_half_period_s']))
                rows.append(out);bywave.setdefault(wave['segment_id'],[]).append(out)
        n_wave=sum(w['axis']==axis for w in smoothed_waves)
        weight=np.asarray([1/(n_wave*len(bywave[r['segment_id']])) for r in rows])
        ae=np.asarray([r['angle_residual_deg'] for r in rows]);te=np.asarray([r['half_period_residual_ms'] for r in rows])
        obs=np.asarray([r['observed_next_angle_rad']*180/math.pi for r in rows]);pred=obs+ae
        mean=np.sum(weight*obs)/weight.sum();r2=1-np.sum(weight*ae**2)/np.sum(weight*(obs-mean)**2)
        ode_fits[axis]=dict(tau=tau,rows=rows,angle_rmse_deg=float(np.sqrt(np.sum(weight*ae**2)/weight.sum())),
            angle_pearson_r=pearson(obs,pred),angle_r2=float(r2),
            half_period_rmse_ms=float(np.sqrt(np.sum(weight*te**2)/weight.sum())))
    for path,records in ((args.output_dir/'ihb03_smoothed_peak_fit_summary.csv',summary),
                         (args.output_dir/'ihb03_smoothed_peak_angles.csv',peak_rows),
                         (args.output_dir/'ihb03_smoothed_peak_waveform_metrics.csv',per_wave),
                         (args.output_dir/'ihb03_smoothed_peak_condition_metrics.csv',condition_rows),
                         (args.output_dir/'ihb03_smoothed_peak_ode_metrics.csv',[
                             dict(axis=ax,tau_smoothed_n_m=v['tau'],half_cycles=len(v['rows']),
                                  angle_rmse_deg=v['angle_rmse_deg'],angle_pearson_r=v['angle_pearson_r'],
                                  angle_r2=v['angle_r2'],half_period_rmse_ms=v['half_period_rmse_ms'])
                             for ax,v in ode_fits.items()]),
                         (args.output_dir/'ihb03_smoothed_peak_ode_predictions.csv',
                          [r for ax in ('IN','OUT') for r in ode_fits[ax]['rows']])):
        with path.open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    fig,axs=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True,constrained_layout=True)
    for axis,ax in zip(('IN','OUT'),axs):
        rr=[r for r in peak_rows if r['axis']==axis]
        x=[float(r['raw_amplitude_deg']) for r in rr];y=[float(r['smoothed_amplitude_deg']) for r in rr]
        ax.scatter(x,y,s=9,alpha=.35,rasterized=True)
        lo=min(x+y);hi=max(x+y);ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set_title(axis);ax.set_xlabel('Observed peak amplitude |A| [deg]');ax.grid(alpha=.2)
    axs[0].set_ylabel('Smoothed peak amplitude [deg]')
    fig.suptitle('Observed vs GCV-smoothed peak amplitudes')
    fig.savefig(args.output_dir/'ihb03_smoothed_peak_angles.svg',bbox_inches='tight')
    fig.savefig(args.output_dir/'ihb03_smoothed_peak_angles.png',dpi=160,bbox_inches='tight')
    write_condition_waveforms(args.output_dir/'ihb03_smoothed_peak_waveforms.svg',peak_rows,
        base.read_csv(args.selection_csv),base.read_csv(args.turning_points_csv),args.repository_root)
    write_smoothed_ode_check(args.output_dir/'ihb03_smoothed_peak_ode_fit.svg',ode_fits)
    write_smoothed_fit_overview(args.output_dir/'ihb03_smoothed_peak_fit_overview.svg',smoothed_waves,ode_fits)
    print(summary)

if __name__=='__main__':main()
