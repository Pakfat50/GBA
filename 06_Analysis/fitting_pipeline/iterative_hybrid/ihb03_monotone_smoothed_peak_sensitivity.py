#!/usr/bin/env python3
"""Run the complete monotone-amplitude tau-identification workflow.

This command reads measured turning-point amplitudes, fits one positive and
monotonically decreasing envelope to each continuous peak sequence, computes
one-pass energy-basis tau estimates, evaluates energy-loss Pearson R/RMSE/R2,
and writes CSV tables, plots, and a Markdown report. It is intended to be run
from the repository root; see the generated report for the exact inputs.
"""
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
import ihb03_friction_identification as base

SP_COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
R_MIN=0.99
RMSE_MAX_DEG=0.5


def pearson(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else float('nan')


def fit_monotone_log_envelope(t, signed_angle_rad, segment_id):
    """Fit A_end+B*log((1+C)/(u+C)); positive and monotonically decreasing."""
    t=np.asarray(t,float); signed_angle_rad=np.asarray(signed_angle_rad,float)
    amp=np.abs(signed_angle_rad)
    if len(t)<5: raise ValueError(f'{segment_id}: fewer than 5 consecutive peaks')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(signed_angle_rad)):
        raise ValueError(f'{segment_id}: non-finite time or angle')
    if np.any(np.diff(t)<=0): raise ValueError(f'{segment_id}: times are not strictly increasing')
    if np.any(amp<=0): raise ValueError(f'{segment_id}: zero peak amplitude')
    u=(t-t[0])/(t[-1]-t[0])
    def model(p,x):
        a_end,b,c=p
        return a_end+b*np.log((1+c)/(x+c))
    p0=[max(1e-8,float(amp[-1])),max(1e-8,float(amp[0]-amp[-1])/2),0.8]
    fit=least_squares(lambda p:model(p,u)-amp,p0,
        bounds=([1e-10,1e-10,1e-5],[np.inf,np.inf,100.0]),
        x_scale='jac',max_nfev=50000,ftol=1e-12,xtol=1e-12,gtol=1e-12)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f'{segment_id}: monotone-envelope fit failed: {fit.message}')
    a_fit=model(fit.x,u)
    if np.any(~np.isfinite(a_fit)) or np.any(a_fit<=0) or fit.x[1]<=0:
        raise ValueError(f'{segment_id}: fitted envelope is not positive and strictly decreasing')
    return np.sign(signed_angle_rad)*a_fit, fit.x, (float(t[0]),float(t[-1]),fit.x.copy())


def envelope_at(t, params):
    t=np.asarray(t,float); t0,t1,p=params
    u=(t-t0)/(t1-t0)
    a_end,b,c=p
    return a_end+b*np.log((1+c)/(u+c))


def energy_metrics_for_condition(waves,axis,conf,tau):
    selected=[w for w in waves if w['axis']==axis and w['configuration']==conf and w['intervals']]
    obs=[];pred=[]
    for w in selected:
        for r in w['intervals']:
            obs.append(float(r['delta_energy_j']))
            pred.append(base.C_ROD*float(r['c_basis'])+tau*float(r['tau_basis']))
    obs=np.asarray(obs);pred=np.asarray(pred)
    weights=np.concatenate([np.full(len(w['intervals']),1/(len(selected)*len(w['intervals']))) for w in selected])
    mean=np.sum(weights*obs)/np.sum(weights)
    sst=np.sum(weights*(obs-mean)**2);sse=np.sum(weights*(obs-pred)**2)
    return dict(r=pearson(obs,pred),r2=float(1-sse/sst) if sst else float('nan'))


def read_wave_segment(path,start,end,column):
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    rows=rows[int(start):int(end)+1]
    return (np.asarray([float(r['systime[ms]'])*1e-3 for r in rows]),
            np.asarray([float(r[column]) for r in rows]))


def write_waveform_overlay(path,peak_rows,selection,points,repository_root):
    pby={}
    for r in points:pby.setdefault(r['segment_id'],[]).append(r)
    fits={}
    for r in peak_rows:fits.setdefault(r['segment_id'],[]).append(r)
    fig,axs=plt.subplots(5,2,figsize=(15,17),constrained_layout=True)
    for si,conf in enumerate([f'SP{i:02d}' for i in range(5)]):
        for ai,axis in enumerate(('IN','OUT')):
            ax=axs[si,ai]
            waves=[r for r in selection if r['axis']==axis and r['configuration']==conf and
                   r['use_for_fitting']=='1' and r['review_status']=='APPROVED']
            for wi,sel in enumerate(sorted(waves,key=lambda x:x['segment_id'])):
                sid=sel['segment_id'];prs=sorted(pby.get(sid,[]),key=lambda x:int(x['peak_number']))
                if not prs: continue
                times=np.asarray([float(x['peak_time_s']) for x in prs]); t0=float(times[0]);t1=float(times[-1])
                center=float(np.median([float(x['peak_angle_deg'])-float(x['centered_peak_angle_deg']) for x in prs]))
                fpath=repository_root/'04_Data/05_Fitting/20260921'/sel['data_file']
                t,raw=read_wave_segment(fpath,sel['start_index'],sel['end_index'],sel['angle_column'])
                angle=(raw+180)%360-180-center; use=(t>=t0)&(t<=t1); color=plt.get_cmap('tab10')(wi%10)
                label=f"{sel['direction']}-R{int(sel['repetition']):02d}"
                ax.plot(t[use]-t0,angle[use],color=color,lw=.65,alpha=.46,label=label,rasterized=True)
                pp=sorted(fits.get(sid,[]),key=lambda x:int(x['peak_number']))
                peak_numbers=np.asarray([int(x['peak_number']) for x in pp])
                runs=np.split(np.arange(len(pp)),np.where(np.diff(peak_numbers)!=1)[0]+1)
                for run in runs:
                    if len(run)<2: continue
                    first,last=int(run[0]),int(run[-1])
                    pars=np.array([float(pp[first]['A_end_rad']),float(pp[first]['B_rad']),float(pp[first]['C'])])
                    rt0=float(pp[first]['peak_time_s']);rt1=float(pp[last]['peak_time_s'])
                    trace_t=t[use];valid=(trace_t>=rt0)&(trace_t<=rt1)
                    if not np.any(valid): continue
                    env=envelope_at(trace_t[valid],(rt0,rt1,pars))
                    ax.plot(trace_t[valid]-t0,env*180/math.pi,color=color,lw=1.05,ls='--',alpha=.95,label='_nolegend_')
                    ax.plot(trace_t[valid]-t0,-env*180/math.pi,color=color,lw=1.05,ls='--',alpha=.95,label='_nolegend_')
            ax.axhline(0,color='#555',lw=.55);ax.set_title(f'{axis} / {conf} (n={len(waves)})')
            ax.set_xlabel('Time from first fitted peak [s]');ax.set_ylabel('Centered angle [deg]');ax.grid(alpha=.2)
            # Match the corresponding GCV-spline panel's vertical scale.
            ax.set_ylim(-65,65)
            if waves:ax.legend(loc='upper right',fontsize=6,ncol=2,framealpha=.8)
    fig.suptitle('Free-decay waveforms with monotone logarithmic amplitude envelopes')
    fig.savefig(path,bbox_inches='tight',dpi=45);fig.savefig(path.with_suffix('.png'),bbox_inches='tight',dpi=90);plt.close(fig)


def write_ode_check(path,fits):
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex='col',constrained_layout=True)
    for row,axis in enumerate(('IN','OUT')):
        for col,(key,ylab,ylim) in enumerate((('angle_residual_deg','Next-peak angle residual [deg]',(-1.5,1.5)),
                                               ('half_period_residual_ms','Half-period residual [ms]',(-30,30)))):
            ax=axs[row,col]
            for si,conf in enumerate([f'SP{i:02d}' for i in range(5)]):
                rr=[r for r in fits[axis]['rows'] if r['configuration']==conf]
                rr=rr[::12] if len(rr)>12 else rr
                ax.scatter([float(r['start_amplitude_deg']) for r in rr],[float(r[key]) for r in rr],
                           s=14,color=SP_COLORS[si],alpha=.72,label=conf)
            ax.axhline(0,color='#222',lw=1);ax.set_ylim(*ylim);ax.set_xlim(4,64);ax.grid(alpha=.22)
            ax.set_title(f'{axis}: {ylab}');ax.set_ylabel(ylab);ax.set_xlabel('Fitted start amplitude [deg]')
            if row==0 and col==0:ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Half-cycle ODE validation using monotone-smoothed peak amplitudes')
    fig.savefig(path,bbox_inches='tight');fig.savefig(path.with_suffix('.png'),bbox_inches='tight',dpi=140);plt.close(fig)


def write_overview(path,waves,ode_fits):
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for row,axis in enumerate(('IN','OUT')):
        tau=ode_fits[axis]['tau']
        selected=[w for w in waves if w['axis']==axis]
        for col,kind in enumerate(('energy','ode')):
            ax=axs[row,col]
            for si,conf in enumerate([f'SP{i:02d}' for i in range(5)]):
                if kind=='energy':
                    rr=[r for w in selected if w['configuration']==conf for r in w['intervals']]
                    x=np.asarray([1000*float(r['delta_energy_j']) for r in rr])
                    y=np.asarray([1000*(base.C_ROD*float(r['c_basis'])+tau*float(r['tau_basis'])) for r in rr])
                    ax.set_xlabel('Observed ΔE [mJ]');ax.set_ylabel('Model cC + τR [mJ]')
                    # Match the spline report's condition-wise energy scales.
                    if axis=='IN':
                        # The spline figure uses microjoules; this panel uses mJ.
                        ax.set_xlim(-0.25,1.05);ax.set_ylim(-0.25,1.05)
                    else:
                        ax.set_xlim(0,1.5);ax.set_ylim(0,1.5)
                    if len(x):
                        lo=min(0,float(x.min()),float(y.min()));hi=max(float(x.max()),float(y.max()));ax.plot([lo,hi],[lo,hi],'k--',lw=1)
                    title=f'{axis}: one-pass energy fit'
                else:
                    rr=[r for r in ode_fits[axis]['rows'] if r['configuration']==conf]
                    x=np.asarray([float(r['start_amplitude_deg']) for r in rr]);y=np.asarray([float(r['angle_residual_deg']) for r in rr])
                    ax.set_xlabel('Measured start amplitude [deg]');ax.set_ylabel('ODE next-peak residual [deg]');ax.axhline(0,color='#222',lw=1)
                    ax.set_xlim(4,64);ax.set_ylim(-1.5,1.5)
                    title=f'{axis}: fixed-τ ODE validation'
                ax.scatter(x,y,s=7,color=SP_COLORS[si],alpha=.55,label=conf,rasterized=True)
                ax.set_title(title);ax.grid(alpha=.22)
                if row==0 and col==0:ax.legend(fontsize=8,ncol=3)
    fig.suptitle('Monotone-smoothed peak amplitudes: energy fit and fixed-τ ODE check')
    fig.savefig(path,bbox_inches='tight');fig.savefig(path.with_suffix('.png'),bbox_inches='tight',dpi=140);plt.close(fig)


def write_automatic_report(report_path, summary, condition_rows, per_wave, peak_rows, ode_metrics, input_paths):
    """Write a human-readable report from the exact outputs made in this run.

    The report deliberately keeps amplitude-fit quality separate from the
    energy-loss fit metrics. Pearson R measures correlation; RMSE measures the
    typical residual size; R2 compares residual error with the observed spread.
    """
    by_key={(row['axis'],row['method']):row for row in summary}
    raw_name='raw_observed_peaks'
    smooth_name='monotone_log_endpoint'
    report=[]
    report += ['# 単調振幅平滑化による一回積分τ同定レポート','',
        '## 解析の流れ','',
        r'このレポートは解析スクリプトが実行ごとに自動生成します。入力の計測頂点列から、波形ごとに振幅の単調関数フィットを行い、フィット後の頂点角で半周期エネルギー収支を作り、一回の重み付き最小二乗で軸別の $\tau$ を同定します。最後にエネルギー損失のPearson $R$、RMSE、$R^2$ を評価し、図表を出力します。', '',
        '処理順は次の通りです。', '',
        '1. 計測頂点列を波形・軸・形態ごとに読み込み、連続した頂点列に分割する。',
        '2. 各列に正値かつ単調減少を保証する対数関数をフィットする。',
        '3. 各波形の振幅フィット品質ゲートを確認する。失敗時は理由を表示して停止し、τを計算しない。',
        '4. フィット振幅から隣接頂点対ごとのエネルギー損失と散逸基底を計算する。',
        r'5. 全採用波形を軸ごとにまとめて $\tau$ を一度だけ最小二乗同定する。',
        r'6. エネルギー損失の $R$、RMSE、$R^2$ を計算し、同定した $\tau$ を変更せずODEで補助確認する。',
        '7. CSV、図、実行条件入りの本レポートを出力する。', '',
        '## 方法','',
        r'振幅包絡線には $A(u)=A_{\rm end}+B\ln\{(1+C)/(u+C)\}$ を使います。フィット区間ごとに $A_{\rm end}>0$、$B>0$、$C>0$ とするため、包絡線は正値で厳密に単調減少します。頂点の正負符号は計測値から引き継ぎます。', '',
        r'隣接する頂点 $A_n,A_{n+1}$ から、観測エネルギー損失は $\Delta E_n=K_j[\cos(|A_{n+1}|)-\cos(|A_n|)]$ とします。固定した $I,K,c_{\rm rod}$ と $b=0$ のもとで、ロッド抗力基底を $C_n$、クーロン摩擦基底を $R_n=|A_n|+|A_{n+1}|$ とします。各波形の総重みを等しくし、$\Delta E_n-c_{\rm rod}C_n\approx\tau R_n$ を切片なし、$\tau\ge0$ で一回の重み付き線形最小二乗により解きます。', '',
        '## 振幅フィット品質','',
        rf'対象波形数: {len(per_wave)}。対象頂点数: {len(peak_rows)}。全波形が $R\ge{R_MIN}$ かつ振幅RMSE $\le{RMSE_MAX_DEG}^\circ$ のゲートを通過した場合だけ、τ同定へ進みます。', '',
        '| 軸 | 形態 | 波形数 | 頂点数 | 振幅Pearson R平均 | 振幅RMSE平均 [deg] |','|---|---|---:|---:|---:|---:|']
    for row in condition_rows:
        report.append(f"| {row['axis']} | {row['configuration']} | {row['waveforms']} | {row['peaks']} | {row['amplitude_R_mean_waveform']:.6f} | {row['amplitude_RMSE_mean_waveform_deg']:.4f} |")
    report += ['', '### 振幅フィット図','', '![計測振幅と単調関数フィット](ihb03_monotone_peak_angles.svg)', '',
        '![自由振動波形上に重ねた単調包絡線](ihb03_monotone_peak_waveforms.svg)', '',
        '## τ同定とエネルギー損失評価','',
        '| 軸 | 振幅列 | 波形数 | 半周期数 | τ [N m] | Pearson R | RMSE [mJ] | R² |','|---|---|---:|---:|---:|---:|---:|---:|']
    for axis in ('IN','OUT'):
        for method,label in ((raw_name,'計測頂点'),(smooth_name,'単調関数フィット')):
            row=by_key[(axis,method)]
            report.append(f"| {axis} | {label} | {row['waveforms']} | {row['half_cycles']} | {row['tau_n_m']:.7g} | {row['energy_pearson_r']:.6f} | {row['energy_rmse_mj']:.5f} | {row['energy_r2']:.6f} |")
    report += ['', 'Pearson Rは観測損失とモデル損失の相関、RMSEは損失残差の大きさ、R²は観測損失のばらつきに対する残差の比を表します。平滑後の指標はフィットに使った同一データ上の評価で、独立データでの予測精度を意味しません。計測頂点列との比較は平滑化による感度を確認するために併記しています。', '',
        '### エネルギー損失図','',
        '![単調関数振幅によるエネルギーフィットと固定τ ODE確認](ihb03_monotone_peak_fit_overview.svg)', '',
        '## 固定τによるODE確認','',
        '| 軸 | 半周期数 | 頂点角RMSE [deg] | 頂点角R² | 半周期時間RMSE [ms] |','|---|---:|---:|---:|---:|']
    for row in ode_metrics:
        report.append(f"| {row['axis']} | {row['half_cycles']} | {row['angle_rmse_deg']:.5f} | {row['angle_r2']:.6f} | {row['half_period_rmse_ms']:.3f} |")
    report += ['', '![固定τで行った半周期ODE残差](ihb03_monotone_peak_ode_fit.svg)', '',
        'この確認ではτを再フィットしません。エネルギー基底法のτを固定したODE積分による補助評価です。', '',
        '## 入力ファイル','']
    for input_path in input_paths:
        report.append(f'- `{input_path}`')
    report += ['', '## 生成ファイル','',
        '- `ihb03_monotone_peak_summary.csv`: 計測頂点と単調フィットの軸別τ・エネルギー指標',
        '- `ihb03_monotone_peak_angles.csv`: 頂点ごとの生角度・フィット角度・包絡線係数',
        '- `ihb03_monotone_peak_waveform_metrics.csv`、`ihb03_monotone_peak_condition_metrics.csv`: 振幅品質と条件別指標',
        '- `ihb03_monotone_peak_ode_metrics.csv`、`ihb03_monotone_peak_ode_predictions.csv`: τ固定ODE確認',
        '- SVGおよびPNG形式の図: 振幅フィット、自由振動への包絡線重ね描き、エネルギー適合、ODE残差', '']
    report_path.write_text('\n'.join(report), encoding='utf-8')


def main():
    """Read user-supplied data files and create a complete analysis bundle."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--turning-points-csv',type=Path,required=True,
                   help='Measured peak-angle point sequence (turning_points.csv).')
    p.add_argument('--selection-csv',type=Path,required=True,
                   help='Waveform review table defining approved segments and raw-log locations.')
    p.add_argument('--condition-physics-csv',type=Path,required=True,
                   help='Fixed I and K values for each axis/configuration.')
    p.add_argument('--output-dir',type=Path,required=True,
                   help='Directory for CSV results, graphs, and the generated Markdown report.')
    p.add_argument('--repository-root',type=Path,default=Path.cwd(),
                   help='Repository root used to locate the measured angle logs for overlay plots.')
    p.add_argument('--report-name',default='IHB03_MONOTONE_TAU_REPORT.md',
                   help='Markdown report filename written inside output-dir.')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # Build the accepted half-cycle list once. Both raw and smoothed results
    # reuse these same waveform/interval identities for a fair comparison.
    waves=base.assemble_intervals(base.read_csv(args.turning_points_csv),base.read_csv(args.selection_csv),
                                  base.read_csv(args.condition_physics_csv),3.0)
    # assemble_intervals has already chosen the common raw-data intervals.
    # A zero extra cutoff keeps those exact interval identities after replacing A.
    raw_fits={axis:base.fit_tau_energy(waves,axis,minimum_amplitude_deg=0.0) for axis in ('IN','OUT')}
    smoothed_waves=[];peak_rows=[];per_wave=[]
    # Fit A(t) independently on each contiguous run of measured extrema.
    # A missing peak breaks the run so the fit never bridges a data gap.
    for wave in waves:
        peaks={}
        for r in wave['intervals']:
            peaks[r['start_peak_number']]=(r['start_time_s'],r['start_angle_rad'])
            peaks[r['end_peak_number']]=(r['start_time_s']+r['observed_half_period_s'],r['observed_next_angle_rad'])
        ordered=sorted((k,*v) for k,v in peaks.items());runs=[]
        for item in ordered:
            if not runs or item[0]!=runs[-1][-1][0]+1:runs.append([item])
            else:runs[-1].append(item)
        smooth_angles={};fit_info={}
        for run in runs:
            nums=np.asarray([x[0] for x in run]);t=np.asarray([x[1] for x in run]);signed=np.asarray([x[2] for x in run])
            fitted,params,curve=fit_monotone_log_envelope(t,signed,wave['segment_id'])
            fit_info[(int(nums[0]),int(nums[-1]))]=params
            for (num,tm,raw),amp_signed in zip(run,fitted):
                amp=abs(float(amp_signed));smooth_angles[int(num)]=math.copysign(amp,float(raw))
                peak_rows.append(dict(segment_id=wave['segment_id'],axis=wave['axis'],configuration=wave['configuration'],
                    direction=wave['direction'],peak_number=int(num),peak_time_s=float(tm),
                    raw_angle_deg=float(raw)*180/math.pi,smoothed_angle_deg=smooth_angles[int(num)]*180/math.pi,
                    raw_amplitude_deg=abs(float(raw))*180/math.pi,smoothed_amplitude_deg=amp*180/math.pi,
                    smoothing_delta_deg=(smooth_angles[int(num)]-float(raw))*180/math.pi,
                    A_end_rad=float(params[0]),B_rad=float(params[1]),C=float(params[2])))
        new=dict(wave);new_intervals=[]
        for r in wave['intervals']:
            a0=smooth_angles[r['start_peak_number']];a1=smooth_angles[r['end_peak_number']]
            de,cb,tb=base.energy_bases(a0,wave['inertia_kg_m2'],wave['restoring_n_m'],a1)
            new_intervals.append(dict(r,start_angle_rad=a0,observed_next_angle_rad=a1,
                start_amplitude_deg=abs(a0)*180/math.pi,next_amplitude_deg=abs(a1)*180/math.pi,
                delta_energy_j=de,c_basis=cb,tau_basis=tb,rod_drag_loss_j=base.C_ROD*cb,
                tau_adjusted_energy_j=de-base.C_ROD*cb))
        new['intervals']=new_intervals;smoothed_waves.append(new)
    # Quality gates run before tau estimation. A failed amplitude fit stops
    # the pipeline; bad waves are never silently omitted from the result.
    # Amplitude gates are checked before any smoothed tau fit.
    failures=[]
    for wave in waves:
        rr=[r for r in peak_rows if r['segment_id']==wave['segment_id']]
        obs=np.asarray([float(r['raw_amplitude_deg']) for r in rr]);fit=np.asarray([float(r['smoothed_amplitude_deg']) for r in rr])
        R=pearson(obs,fit);rmse=float(np.sqrt(np.mean((fit-obs)**2)));bias=float(np.mean(fit-obs))
        per_wave.append(dict(segment_id=wave['segment_id'],axis=wave['axis'],configuration=wave['configuration'],
            direction=wave['direction'],peaks=len(rr),pearson_r=R,amplitude_rmse_deg=rmse,amplitude_bias_deg=bias))
        if not np.isfinite(R) or R<R_MIN or rmse>RMSE_MAX_DEG:
            failures.append(f"{wave['segment_id']}: R={R:.6f}, amplitude RMSE={rmse:.6f} deg")
    if failures: raise RuntimeError('Monotone A-fit gate failed; tau not fitted:\n'+'\n'.join(failures))
    # This is the requested one-pass energy-basis tau fit, using the fitted
    # monotone A values and the fixed physical coefficients.
    smooth_fits={axis:base.fit_tau_energy(smoothed_waves,axis,minimum_amplitude_deg=0.0) for axis in ('IN','OUT')}
    summary=[]
    for axis in ('IN','OUT'):
        for method,fit in (('raw_observed_peaks',raw_fits[axis]),('monotone_log_endpoint',smooth_fits[axis])):
            summary.append(dict(axis=axis,method=method,tau_n_m=fit['tau'],energy_rmse_mj=1000*fit['energy_rmse_j'],
                energy_pearson_r=fit['energy_r'],energy_r2=fit['energy_r2'],waveforms=fit['waveform_count'],
                half_cycles=fit['interval_count']))
    condition_rows=[]
    for axis in ('IN','OUT'):
        for conf in [f'SP{i:02d}' for i in range(5)]:
            ww=[r for r in per_wave if r['axis']==axis and r['configuration']==conf]
            if not ww:continue
            peak_group=[r for r in peak_rows if r['axis']==axis and r['configuration']==conf]
            d=[]
            for w0,w1 in zip([w for w in waves if w['axis']==axis and w['configuration']==conf],
                             [w for w in smoothed_waves if w['axis']==axis and w['configuration']==conf]):
                for x,y in zip(w0['intervals'],w1['intervals']):d.append(float(y['delta_energy_j'])-float(x['delta_energy_j']))
            raw_sp=base.fit_tau_energy(waves,axis,[conf],minimum_amplitude_deg=0.0);smooth_sp=base.fit_tau_energy(smoothed_waves,axis,[conf],minimum_amplitude_deg=0.0)
            r_raw=energy_metrics_for_condition(waves,axis,conf,raw_fits[axis]['tau'])
            r_smooth=energy_metrics_for_condition(smoothed_waves,axis,conf,smooth_fits[axis]['tau'])
            condition_rows.append(dict(axis=axis,configuration=conf,waveforms=len(ww),peaks=len(peak_group),
                amplitude_R_mean_waveform=float(np.mean([r['pearson_r'] for r in ww])),
                amplitude_R_min_waveform=float(np.min([r['pearson_r'] for r in ww])),
                amplitude_R_max_waveform=float(np.max([r['pearson_r'] for r in ww])),
                amplitude_RMSE_mean_waveform_deg=float(np.mean([r['amplitude_rmse_deg'] for r in ww])),
                amplitude_smoothing_rms_deg=float(np.sqrt(np.mean([(float(r['smoothed_amplitude_deg'])-float(r['raw_amplitude_deg']))**2 for r in peak_group]))),
                delta_energy_change_rms_mj=float(np.sqrt(np.mean(np.asarray(d)**2))*1000),
                energy_R_raw_pooled_tau=r_raw['r'],energy_R_monotone_pooled_tau=r_smooth['r'],energy_R_change=r_smooth['r']-r_raw['r'],
                energy_R2_raw_pooled_tau=r_raw['r2'],energy_R2_monotone_pooled_tau=r_smooth['r2'],energy_R2_change=r_smooth['r2']-r_raw['r2'],
                tau_raw_sp_fit_n_m=raw_sp['tau'],tau_monotone_sp_fit_n_m=smooth_sp['tau'],
                tau_change_percent=100*(smooth_sp['tau']/raw_sp['tau']-1),
                energy_rmse_raw_sp_fit_mj=1000*raw_sp['energy_rmse_j'],energy_rmse_monotone_sp_fit_mj=1000*smooth_sp['energy_rmse_j']))
    ode_fits={}
    for axis in ('IN','OUT'):
        tau=smooth_fits[axis]['tau'];rows=[];bywave={}
        for wave in [w for w in smoothed_waves if w['axis']==axis]:
            for rr in wave['intervals']:
                sol=base.solve_next_turning_point(rr['start_angle_rad'],wave['inertia_kg_m2'],wave['restoring_n_m'],
                    0.0,base.C_ROD,tau,base.EPSILON_DEG_S*math.pi/180,rtol=1e-4,angle_speed_atol=1e-7,
                    energy_atol=1e-11,max_step_fraction=.2,max_periods=2.0)
                out=dict(rr,model_next_angle_rad=sol['next_angle_rad'],model_next_angle_deg=sol['next_angle_rad']*180/math.pi,
                    angle_residual_deg=(sol['next_angle_rad']-rr['observed_next_angle_rad'])*180/math.pi,
                    model_half_period_s=sol['half_period_s'],half_period_residual_ms=1000*(sol['half_period_s']-rr['observed_half_period_s']))
                rows.append(out);bywave.setdefault(wave['segment_id'],[]).append(out)
        nw=sum(w['axis']==axis for w in smoothed_waves)
        weights=np.asarray([1/(nw*len(bywave[r['segment_id']])) for r in rows])
        ae=np.asarray([r['angle_residual_deg'] for r in rows]);te=np.asarray([r['half_period_residual_ms'] for r in rows])
        obs=np.asarray([r['observed_next_angle_rad']*180/math.pi for r in rows]);pred=obs+ae
        mean=np.sum(weights*obs)/weights.sum();r2=1-np.sum(weights*ae**2)/np.sum(weights*(obs-mean)**2)
        ode_fits[axis]=dict(tau=tau,rows=rows,angle_rmse_deg=float(np.sqrt(np.sum(weights*ae**2)/weights.sum())),
            angle_pearson_r=pearson(obs,pred),angle_r2=float(r2),half_period_rmse_ms=float(np.sqrt(np.sum(weights*te**2)/weights.sum())))
    ode_metrics=[dict(axis=ax,tau_n_m=v['tau'],half_cycles=len(v['rows']),angle_rmse_deg=v['angle_rmse_deg'],
        angle_pearson_r=v['angle_pearson_r'],angle_r2=v['angle_r2'],half_period_rmse_ms=v['half_period_rmse_ms']) for ax,v in ode_fits.items()]
    for name,records in (
        ('ihb03_monotone_peak_summary.csv',summary),('ihb03_monotone_peak_angles.csv',peak_rows),
        ('ihb03_monotone_peak_waveform_metrics.csv',per_wave),('ihb03_monotone_peak_condition_metrics.csv',condition_rows),
        ('ihb03_monotone_peak_ode_metrics.csv',ode_metrics),
        ('ihb03_monotone_peak_ode_predictions.csv',[r for ax in ('IN','OUT') for r in ode_fits[ax]['rows']])):
        with (args.output_dir/name).open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    # Plot raw vs fitted peak amplitudes.
    fig,axs=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True,constrained_layout=True)
    for axis,ax in zip(('IN','OUT'),axs):
        rr=[r for r in peak_rows if r['axis']==axis];xx=[float(r['raw_amplitude_deg']) for r in rr];yy=[float(r['smoothed_amplitude_deg']) for r in rr]
        ax.scatter(xx,yy,s=9,alpha=.35,rasterized=True);lo=min(xx+yy);hi=max(xx+yy);ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        # Match the GCV-spline peak-amplitude comparison panels.
        ax.set_xlim(5,65);ax.set_ylim(5,65)
        ax.set_title(axis);ax.set_xlabel('Observed peak amplitude |A| [deg]');ax.grid(alpha=.2)
    axs[0].set_ylabel('Monotone-fitted amplitude [deg]');fig.suptitle('Observed vs monotone log-endpoint peak amplitudes')
    fig.savefig(args.output_dir/'ihb03_monotone_peak_angles.svg',bbox_inches='tight');fig.savefig(args.output_dir/'ihb03_monotone_peak_angles.png',bbox_inches='tight',dpi=160);plt.close(fig)
    write_waveform_overlay(args.output_dir/'ihb03_monotone_peak_waveforms.svg',peak_rows,base.read_csv(args.selection_csv),
                           base.read_csv(args.turning_points_csv),args.repository_root)
    write_ode_check(args.output_dir/'ihb03_monotone_peak_ode_fit.svg',ode_fits)
    write_overview(args.output_dir/'ihb03_monotone_peak_fit_overview.svg',smoothed_waves,ode_fits)
    # Write the report last, after all data tables and figures exist. Relative
    # image links therefore render when the output directory is viewed on GitHub.
    report_path=args.output_dir/args.report_name
    write_automatic_report(report_path,summary,condition_rows,per_wave,peak_rows,ode_metrics,
                           [args.turning_points_csv,args.selection_csv,args.condition_physics_csv])
    print('report:',report_path)
    print('tau raw:',{a:raw_fits[a]['tau'] for a in raw_fits})
    print('tau monotone:',{a:smooth_fits[a]['tau'] for a in smooth_fits})
    print('ODE metrics:',ode_metrics)
    print('A gates passed for',len(per_wave),'waveforms')


if __name__=='__main__':main()

