"""Generate the WBS 2.2 figures and transparent oracle selection tables."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
import run_ow06_wbs22_oracle as w

plt.rcParams.update({'svg.fonttype':'none','font.size':10})
COLORS={'EKF_q0.03':'#ae7b24','EKF_q0.1':'#80a63e','EKF_q0.3':'#15977d',
        'EKF_q1':'#247abd','RTS_q1':'#202633','Fixed_p5':'#d95645',
        'Fixed_p2':'#ad67b9','EKF_q300':'#a64e9c'}


def passing(d):
    return (d.valid & d.gain.between(w.GATE['gain_min'],w.GATE['gain_max'])
        & (abs(d.phase_deg)<=w.GATE['phase_abs_max_deg'])
        & (d.wind_rmse_m_s<=w.GATE['wind_rmse_max_m_s']))


def selections(d):
    rows=[]
    for (axis,f,amp),g in d.groupby(['axis','frequency_hz','amplitude_ratio']):
        for family,base,candidates in [('EKF','EKF_q1',['EKF_q0.03','EKF_q0.1','EKF_q0.3']),
                ('Fixed','Fixed_p5',['Fixed_p1','Fixed_p2']),
                ('Any_online','EKF_q1',[x for x in g.method.unique() if not x.startswith('RTS')])]:
            tr=g[(g.plant_case=='matched') & g.sensor_case.isin(['ideal','sensor_seed0']) & g.method.isin(candidates)]
            feasible=[m for m,h in tr.groupby('method') if len(h)==2 and passing(h).all()]
            scores=tr[tr.sensor_case=='sensor_seed0'].copy()
            scores=scores[scores.valid]
            if feasible:scores=scores[scores.method.isin(feasible)]
            if scores.empty:
                rows.append(dict(axis=axis,frequency_hz=f,amplitude_ratio=amp,family=family,
                    selected_method='none',training_feasible=False,validation_response_pass=False,
                    relative_benefit_pass=False,gate_pass=False))
                continue
            # Among feasible reduced settings prefer the least sensor increment.
            # If none is feasible, report lowest RMSE only as a diagnostic.
            rank='noise_increment_rms_m_s' if feasible and family!='Any_online' else 'wind_rmse_m_s'
            sel=scores.sort_values(rank).iloc[0].method
            va=g[(g.method==sel)&g.sensor_case.isin(['ideal','delay_quant','sensor_seed1','sensor_seed2'])]
            stochastic=va[va.sensor_case.isin(['sensor_seed1','sensor_seed2'])]
            ba=g[(g.method==base)&g.sensor_case.isin(['sensor_seed1','sensor_seed2'])]
            comp=stochastic.merge(ba,on=['plant_case','sensor_case'],suffixes=('_sel','_base'))
            er=(comp.wind_rmse_m_s_sel/comp.wind_rmse_m_s_base).max()
            nr=(comp.noise_increment_rms_m_s_sel/comp.noise_increment_rms_m_s_base).max()
            gr=(comp.gain_force_rms_sel/comp.gain_force_rms_base).max()
            response=len(va)==8 and passing(va).all()
            relative=(comp.valid_sel.all() and comp.valid_base.all() and len(comp)==4
                      and er<=1.05 and nr<=.9 and gr<1.)
            rows.append(dict(axis=axis,frequency_hz=f,amplitude_ratio=amp,family=family,
                selected_method=sel,baseline_method=base,training_feasible=bool(feasible),
                training_feasible_count=len(feasible),validation_response_pass=bool(response),
                validation_worst_wind_rmse_m_s=va.wind_rmse_m_s.max(),
                validation_gain_min=va.gain.min(),validation_gain_max=va.gain.max(),
                validation_phase_abs_max_deg=va.phase_deg.abs().max(),
                worst_rmse_ratio_to_baseline=er,worst_noise_ratio_to_baseline=nr,
                worst_force_gain_ratio_to_baseline=gr,relative_benefit_pass=bool(relative),
                gate_pass=bool(feasible and response and relative and family!='Any_online')))
    result=pd.DataFrame(rows);result.to_csv(w.OUT/'oracle_selection.csv',index=False)
    return result


def save(fig,name):
    fig.savefig(w.OUT/(name+'.png'),dpi=180)
    fig.savefig(w.OUT/(name+'.svg'))
    plt.close(fig)


def frequency_plot(d,case,zoom):
    methods=['EKF_q0.1','EKF_q0.3','EKF_q1','RTS_q1','Fixed_p5']
    fig,axes=plt.subplots(4,2,figsize=(12,12),layout='constrained',sharex=True,sharey='row')
    metrics=[('gain','Force amplitude ratio'),('phase_deg','Phase [deg]'),
             ('wind_rmse_m_s','Wind RMSE [m/s]'),('noise_increment_rms_m_s','Sensor-noise increment RMS [m/s]')]
    for col,axis in enumerate(w.Q0):
        sub=d[(d.axis==axis)&(d.plant_case==case)&(d.amplitude_ratio==.35)&(d.sensor_case=='sensor_seed1')]
        if zoom:sub=sub[sub.frequency_hz.between(.8,1.2)]
        else:sub=sub[sub.frequency_hz<=10]
        for name in methods:
            s=sub[sub.method==name].sort_values('frequency_hz')
            for r,(metric,label) in enumerate(metrics):
                z=s[metric].where(s.valid).copy()
                if metric=='phase_deg':z=z.where(s.gain>=.05)
                axes[r,col].plot(s.frequency_hz,z,'o-',ms=4,lw=1.5,color=COLORS[name],label=name)
                axes[r,col].grid(alpha=.25);axes[r,col].set_ylabel(label)
        axes[0,col].set_title(f'{axis}: {case}')
        axes[0,col].axhspan(.9,1.1,color='#cccccc',alpha=.35)
        axes[1,col].axhspan(-20,20,color='#cccccc',alpha=.35)
        axes[2,col].axhline(.1,color='gray',ls=':')
        axes[3,col].set_yscale('log')
        for row,limits in enumerate([(0,1.85),(-180,180),(0,.75),(.001,.2)]):
            axes[row,col].set_ylim(limits)
        axes[3,col].set_xlabel('Force frequency [Hz]')
        if not zoom:
            axes[3,col].set_xscale('log')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Known-frequency screening, 2 m/s equivalent mean, A/F0=0.35\n'
        'OW-05 sensor assumptions, held-out seed 1; gray bands are provisional; phase hidden for gain < 0.05')
    save(fig,f'frequency_{case}'+('_zoom' if zoom else ''))


def wave_plots():
    for case in ('matched','OW04'):
        fig,axes=plt.subplots(2,2,figsize=(13,7.5),layout='constrained')
        for r,axis in enumerate(w.Q0):
            d=pd.read_csv(w.OUT/f'trace_{axis}_1_{case}.csv')
            for c,(lo,hi) in enumerate([(0,24),(10,12)]):
                s=d[d.time_s.between(lo,hi)]
                axes[r,c].plot(s.time_s,s.wind_true_m_s,color='black',lw=2,label='True wind')
                for m in ['EKF_q0.3','EKF_q1','RTS_q1','Fixed_p5']:
                    axes[r,c].plot(s.time_s,s[m+'_wind_m_s'],lw=1.1,color=COLORS[m],label=m)
                axes[r,c].set(xlabel='Time [s]',ylabel='Wind [m/s]',title=f'{axis}: {case}, 1 Hz')
                axes[r,c].grid(alpha=.25)
        axes[0,0].legend(fontsize=8,ncol=3)
        fig.suptitle('Full record and time-axis enlargement; sensor seed 1, A/F0=0.35')
        save(fig,f'timeseries_1hz_{case}')
    fig,axes=plt.subplots(2,2,figsize=(13,7.5),layout='constrained')
    for r,axis in enumerate(w.Q0):
        d=pd.read_csv(w.OUT/f'trace_{axis}_10_matched.csv')
        s=d[d.time_s.between(10,10.5)]
        for c,ms in enumerate([['EKF_q0.3','EKF_q1','RTS_q1'],['EKF_q300','Fixed_p5']]):
            axes[r,c].plot(s.time_s,s.wind_true_m_s,color='black',lw=2,label='True wind')
            for m in ms:axes[r,c].plot(s.time_s,s[m+'_wind_m_s'],'.-',ms=3,color=COLORS[m],label=m)
            axes[r,c].set(xlabel='Time [s]',ylabel='Wind [m/s]',title=f'{axis}: 10 Hz, '+('baseline / reduced q' if c==0 else 'high q diagnostic'))
            axes[r,c].grid(alpha=.25);axes[r,c].legend(fontsize=8)
    fig.suptitle('10 Hz: only 10 samples/cycle; OW-05 assumed sensor, seed 1; matched plant')
    save(fig,'timeseries_10hz')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for r,axis in enumerate(w.Q0):
        for c,case in enumerate(('matched','OW04')):
            d=pd.read_csv(w.OUT/f'trace_{axis}_1_{case}.csv');d=d[d.time_s.between(8,22)]
            for m in ['EKF_q0.3','EKF_q1','RTS_q1','Fixed_p5']:
                error=d[m+'_wind_m_s'].to_numpy()-d.wind_true_m_s.to_numpy()
                f,p=welch(error,fs=100,nperseg=512)
                axes[r,c].loglog(f[1:],p[1:],label=m,color=COLORS[m])
            axes[r,c].set(xlabel='Frequency [Hz]',ylabel='Wind-error PSD [(m/s)^2/Hz]',title=f'{axis}: {case}, 1 Hz forcing')
            axes[r,c].grid(alpha=.25,which='both')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Total estimation-error spectrum (tracking + coefficient error + sensor effects)')
    save(fig,'error_spectra_1hz')


def tradeoff(d):
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,axis in zip(axes,w.Q0):
        for case,marker in [('matched','o'),('OW04','s')]:
            s=d[(d.axis==axis)&(d.frequency_hz==1)&(d.amplitude_ratio==.35)&(d.plant_case==case)
                &(d.sensor_case=='sensor_seed1')&d.method.str.startswith('EKF')].sort_values('q_multiplier')
            ax.plot(s.noise_increment_rms_m_s,s.wind_rmse_m_s,marker+'-',label=case)
            for _,r in s.iterrows():ax.annotate(f'q x{r.q_multiplier:g}',(r.noise_increment_rms_m_s,r.wind_rmse_m_s),fontsize=8,xytext=(4,4),textcoords='offset points')
        ax.set(xscale='log',yscale='log',xlabel='Paired sensor-noise increment RMS [m/s]',ylabel='Raw wind RMSE [m/s]',title=axis)
        ax.grid(which='both',alpha=.25);ax.legend()
    fig.suptitle('1 Hz online EKF: lower q reduces noise but does not guarantee better tracking')
    save(fig,'noise_tracking_tradeoff')


def main():
    d=pd.read_csv(w.OUT/'metrics.csv');sel=selections(d)
    for case in ('matched','OW04'):
        frequency_plot(d,case,False);frequency_plot(d,case,True)
    wave_plots();tradeoff(d)
    d[~d.valid].to_csv(w.OUT/'invalid_cases.csv',index=False)
    gains=[]
    for axis in w.Q0:
        co=w.select_coefficients(axis,'BALL')
        for name,_,L,_,pole,v in w.methods(co,axis):
            if L is None:continue
            _,eig,err=w.fixed_gain(co,pole,v)
            gains.append(dict(axis=axis,method=name,design_wind_m_s=v,pole_hz=pole,
                L_theta=L[0],L_rate=L[1],L_force=L[2],max_abs_linear_error_pole=max(abs(eig)),
                polynomial_error_max=max(abs(err))))
    pd.DataFrame(gains).to_csv(w.OUT/'fixed_gains.csv',index=False)
    summary=dict(metric_rows=len(d),plant_cases=len(d[['axis','frequency_hz','amplitude_ratio','plant_case']].drop_duplicates()),
        invalid_rows=int((~d.valid).sum()),invalid_by_method=d[~d.valid].groupby('method').size().to_dict(),
        max_physical_angle_deg=d.max_angle_deg.max(),contact_samples_max=int(d.contact_samples.max()),
        reduced_setting_gate_pass_count=int(sel[sel.family!='Any_online'].gate_pass.sum()),
        validation_response_pass_count_by_family=sel.groupby('family').validation_response_pass.sum().astype(int).to_dict(),
        provisional_criteria=w.GATE,stage_23_authorized=False)
    (w.OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
