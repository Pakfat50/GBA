#!/usr/bin/env python3
"""Fit monotone logarithmic envelopes to IHB-03 peak amplitudes and plot traces."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

R_MIN=0.99
RMSE_MAX_DEG=0.5


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def fit_envelope(t_sec,amplitude_deg,segment_id):
    t=np.asarray(t_sec,float);y=np.asarray(amplitude_deg,float)
    if len(t)<5:raise ValueError(f'{segment_id}: fewer than five peaks')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):raise ValueError(f'{segment_id}: non-finite input')
    if np.any(np.diff(t)<=0):raise ValueError(f'{segment_id}: peak times must strictly increase')
    if np.any(y<=0):raise ValueError(f'{segment_id}: amplitude must be positive')
    u=(t-t[0])/(t[-1]-t[0])
    def model(p,x):return p[0]+p[1]*np.log((1+p[2])/(x+p[2]))
    p0=[max(1e-6,y[-1]),max(1e-6,(y[0]-y[-1])/2),.8]
    res=least_squares(lambda p:model(p,u)-y,p0,bounds=([1e-8,1e-8,1e-5],[np.inf,np.inf,100]),
        x_scale='jac',max_nfev=50000,ftol=1e-12,xtol=1e-12,gtol=1e-12)
    if not res.success:raise RuntimeError(f'{segment_id}: {res.message}')
    fitted=model(res.x,u)
    if not np.all(np.isfinite(fitted)) or np.any(fitted<=0) or res.x[1]<=0:
        raise RuntimeError(f'{segment_id}: fit is not positive and strictly decreasing')
    return fitted,res.x


def pearson(x,y):
    return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else float('nan')


def read_trace(path,start,end,column):
    rows=read_csv(path)[int(start):int(end)+1]
    t=np.asarray([float(r['systime[ms]'])*1e-3 for r in rows])
    angle=np.asarray([float(r[column]) for r in rows])
    return t,angle


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--peak-csv',type=Path,required=True)
    ap.add_argument('--selection-csv',type=Path,required=True)
    ap.add_argument('--turning-points-csv',type=Path,required=True)
    ap.add_argument('--repository-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    peaks=read_csv(args.peak_csv);selection=read_csv(args.selection_csv);turning=read_csv(args.turning_points_csv)
    by_wave={}
    for r in peaks:by_wave.setdefault(r['segment_id'],[]).append(r)
    peak_out=[];wave_metrics=[];fail=[]
    for sid,rr in sorted(by_wave.items()):
        rr.sort(key=lambda x:int(x['peak_number']))
        t=np.asarray([float(x['peak_time_s']) for x in rr]);obs=np.asarray([float(x['raw_amplitude_deg']) for x in rr])
        fitted,p=fit_envelope(t,obs,sid)
        R=pearson(obs,fitted);rmse=float(np.sqrt(np.mean((fitted-obs)**2)));bias=float(np.mean(fitted-obs))
        wave_metrics.append(dict(segment_id=sid,axis=rr[0]['axis'],configuration=rr[0]['configuration'],
            direction=rr[0]['direction'],peaks=len(rr),pearson_r=R,amplitude_rmse_deg=rmse,
            amplitude_bias_deg=bias,A_end_deg=float(p[0]),B_deg=float(p[1]),C=float(p[2])))
        if not np.isfinite(R) or R<R_MIN or rmse>RMSE_MAX_DEG:fail.append(f'{sid}: R={R:.6f}, RMSE={rmse:.6f} deg')
        for row,yhat in zip(rr,fitted):
            signed=math.copysign(float(yhat),float(row['raw_angle_deg']))
            peak_out.append(dict(segment_id=sid,axis=row['axis'],configuration=row['configuration'],
                direction=row['direction'],peak_number=row['peak_number'],peak_time_s=row['peak_time_s'],
                observed_amplitude_deg=row['raw_amplitude_deg'],fitted_amplitude_deg=float(yhat),
                amplitude_residual_deg=float(yhat)-float(row['raw_amplitude_deg']),
                observed_angle_deg=row['raw_angle_deg'],fitted_signed_angle_deg=signed,
                A_end_deg=float(p[0]),B_deg=float(p[1]),C=float(p[2])))
    if fail:raise RuntimeError('Amplitude fit quality gate failed:\n'+'\n'.join(fail))
    condition_metrics=[]
    for axis in ('IN','OUT'):
        for conf in [f'SP{i:02d}' for i in range(5)]:
            waves=[w for w in wave_metrics if w['axis']==axis and w['configuration']==conf]
            pts=[r for r in peak_out if r['axis']==axis and r['configuration']==conf]
            if not waves:continue
            condition_metrics.append(dict(axis=axis,configuration=conf,waveforms=len(waves),peaks=len(pts),
                waveform_R_mean=float(np.mean([w['pearson_r'] for w in waves])),
                waveform_R_min=float(np.min([w['pearson_r'] for w in waves])),
                waveform_R_max=float(np.max([w['pearson_r'] for w in waves])),
                pooled_peak_R=pearson(np.asarray([float(r['observed_amplitude_deg']) for r in pts]),
                                      np.asarray([float(r['fitted_amplitude_deg']) for r in pts])),
                waveform_RMSE_mean_deg=float(np.mean([w['amplitude_rmse_deg'] for w in waves])),
                waveform_RMSE_max_deg=float(np.max([w['amplitude_rmse_deg'] for w in waves]))))
    for filename,records in (('ihb03_monotone_amplitude_waveform_metrics.csv',wave_metrics),
        ('ihb03_monotone_amplitude_condition_metrics.csv',condition_metrics),
        ('ihb03_monotone_amplitude_peak_fits.csv',peak_out)):
        with (args.output_dir/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

    selection_by={r['segment_id']:r for r in selection}
    turning_by={}
    for r in turning:turning_by.setdefault(r['segment_id'],[]).append(r)
    peak_by={}
    for r in peak_out:peak_by.setdefault(r['segment_id'],[]).append(r)
    fig,axs=plt.subplots(5,2,figsize=(15,17),constrained_layout=True)
    for si,conf in enumerate([f'SP{i:02d}' for i in range(5)]):
        for ai,axis in enumerate(('IN','OUT')):
            ax=axs[si,ai]
            waves=[r for r in selection if r['axis']==axis and r['configuration']==conf and
                   r['use_for_fitting']=='1' and r['review_status']=='APPROVED' and r['segment_id'] in peak_by]
            for wi,sel in enumerate(sorted(waves,key=lambda x:x['segment_id'])):
                sid=sel['segment_id'];pp=sorted(peak_by[sid],key=lambda x:int(x['peak_number']))
                pr=sorted(turning_by[sid],key=lambda x:int(x['peak_number']))
                if not pp or not pr:continue
                t0=float(pp[0]['peak_time_s']);t1=float(pp[-1]['peak_time_s'])
                center=float(np.median([float(x['peak_angle_deg'])-float(x['centered_peak_angle_deg']) for x in pr]))
                trace=args.repository_root/'04_Data/05_Fitting/20260921'/sel['data_file']
                t,raw=read_trace(trace,sel['start_index'],sel['end_index'],sel['angle_column'])
                centered=(raw+180)%360-180-center;use=(t>=t0)&(t<=t1);color=plt.get_cmap('tab10')(wi%10)
                label=f"{sel['direction']}-R{int(sel['repetition']):02d}"
                ax.plot(t[use]-t0,centered[use],color=color,lw=.65,alpha=.46,label=label,rasterized=True)
                u=(t[use]-t0)/(t1-t0);p=[float(pp[0]['A_end_deg']),float(pp[0]['B_deg']),float(pp[0]['C'])]
                envelope=p[0]+p[1]*np.log((1+p[2])/(u+p[2]))
                ax.plot(t[use]-t0,envelope,color=color,lw=1.05,ls='--',alpha=.95,label='_nolegend_')
                ax.plot(t[use]-t0,-envelope,color=color,lw=1.05,ls='--',alpha=.95,label='_nolegend_')
            ax.axhline(0,color='#555',lw=.55);ax.set_title(f'{axis} / {conf} (n={len(waves)})')
            ax.set_xlabel('Time from first fitted peak [s]');ax.set_ylabel('Centered angle [deg]');ax.grid(alpha=.2)
            # Keep the displayed angle range identical to the GCV-spline panels.
            ax.set_ylim(-65,65)
            if waves:ax.legend(loc='upper right',fontsize=6,ncol=2,framealpha=.8)
    fig.suptitle('Free-decay waveforms with monotone logarithmic amplitude envelopes')
    fig.savefig(args.output_dir/'ihb03_monotone_amplitude_waveforms.svg',bbox_inches='tight',dpi=45)
    fig.savefig(args.output_dir/'ihb03_monotone_amplitude_waveforms.png',bbox_inches='tight',dpi=90)
    plt.close(fig)
    print('waveforms',len(wave_metrics),'all passed R >=',R_MIN,'and RMSE <=',RMSE_MAX_DEG,'deg')
    print('condition metrics:')
    for r in condition_metrics:print(r)


if __name__=='__main__':main()
