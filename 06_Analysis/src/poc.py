"""GBA proof of concept: measured-angle disturbance observer and common-mask scoring.

Only the morning record selects clock offset and observer/filter pole frequency.
The afternoon record is held out. Source logs are never edited.
The CSV wind-speed conversion intentionally preserves analysis.py's componentwise
square-root convention to isolate the effect of the dynamical estimator.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy import signal, ndimage
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parent
ANALYSIS_ROOT=ROOT.parent
SETTINGS=json.loads((ANALYSIS_ROOT/'config/settings.json').read_text())
CACHE=ANALYSIS_ROOT/'data/processed'
OUT=ANALYSIS_ROOT/'results'; OUT.mkdir(parents=True,exist_ok=True)
FS=100.; DT=1/FS
K0=9.81*(.0288*.180-.0146*.245-.0058*.070)
LENGTH=.245
OLD_K=.0488*9.81*.0293
CD=.55*K0/OLD_K  # preserves K/Cd from the attached Cd=0.55 plot
RHO=1.225; AREA=np.pi*.15**2/4
POLES=SETTINGS['pole_hz_grid']
COLORS={'Static':'#2166c2','Low-pass':'#9063ad','Observer':'#008870','Reference':'#dc3b3b'}

def load_records():
    return {label:dict(np.load(CACHE/f'{label}.npz')) for label in ('am','pm')}

def identify():
    fits=json.loads((CACHE/'free_decay.json').read_text())
    wn2=np.median([r['omega_d']**2+r['decay_per_s']**2 for r in fits])
    decay=np.median([r['decay_per_s'] for r in fits])
    I=K0/wn2; b=2*decay*I
    return dict(K=K0,I=I,b=b,wn2=wn2,fn_hz=np.sqrt(wn2)/(2*np.pi),decay_per_s=decay,period_s=2*np.pi/np.sqrt(wn2),zeta=decay/np.sqrt(wn2))

def analog_filter(y,num,den):
    # Tustin discretization of the proper continuous-time transfer function.
    bz,az=signal.bilinear(num,den,fs=FS)
    zi=signal.lfilter_zi(bz,az)[:,None]*y[0][None,:]
    return signal.lfilter(bz,az,y,axis=0,zi=zi)[0]

def basis(angle,pole_hz,delay_compensate=True):
    """Q(s)={p^3}/{(s+p)^3}; returns Q[sin(theta)], Q[s theta], Q[s^2 theta].

    This is the disturbance-torque output of an extended-state Luenberger
    observer with measured nonlinear gravity compensation. For A21=0,
    L1=3p-b/I, L2=3p^2-(b/I)*L1, L3=I*p^3.
    The known gravity input is -K*sin(theta_measured)/I.
    d(tau)/dt=0 is the predictor assumption, not a statement about actual wind.
    """
    theta=np.deg2rad(angle); p=2*np.pi*pole_hz
    den=[1,3*p,3*p*p,p**3]
    q0=analog_filter(np.sin(theta),[p**3],den)
    q1=analog_filter(theta,[p**3,0],den)
    q2=analog_filter(theta,[p**3,0,0],den)
    if delay_compensate:
        # Offline advance by the known low-frequency group delay of Q.
        # This is an approximation; it is not a reference-fitted time shift.
        grid=np.arange(len(theta)); advance=3/p*FS
        def advance_array(x):return np.column_stack([np.interp(grid+advance,grid,x[:,j]) for j in range(2)])
        q0,q1,q2=map(advance_array,(q0,q1,q2))
    return q0,q1,q2

def to_speed(torque):
    force=torque/LENGTH
    # Exactly the magnitude produced by the legacy independent-axis conversion.
    return np.sqrt(2*np.sum(np.abs(force),axis=1)/(RHO*CD*AREA))

def mask_data(d,label,shift,threshold=None,padding=None):
    if threshold is None:threshold=SETTINGS['threshold_deg']
    if padding is None:padding=SETTINGS['saturation_padding_s']
    q=np.deg2rad(d['angle'])
    tilt=np.rad2deg(np.arccos(np.clip(np.cos(q[:,0])*np.cos(q[:,1]),-1,1)))
    saturation=(tilt>=threshold)|(np.max(np.abs(d['angle']),axis=1)>=SETTINGS['axis_limit_deg'])
    near=ndimage.maximum_filter1d(saturation.astype(np.uint8),size=2*int(padding*FS)+1,mode='constant')>0
    tj=d['tj']+shift
    touch=np.zeros(len(tj),bool)
    for t in SETTINGS['tap_times_jst_s'][label]:
        touch|=(tj>=t-SETTINGS['tap_before_s'])&(tj<=t+SETTINGS['tap_after_s'])
    edge=(d['t']<d['t'][0]+SETTINGS['edge_seconds'])|(d['t']>d['t'][-1]-SETTINGS['edge_seconds'])
    manual=np.zeros(len(tj),bool)
    for start,end in SETTINGS['manual_exclusions_jst_s'][label]:
        if start>end:raise ValueError('Exclusion interval start must not exceed end')
        manual|=(tj>=start)&(tj<=end)
    reasons=saturation.astype(np.uint8)+2*near.astype(np.uint8)+4*touch.astype(np.uint8)+8*edge.astype(np.uint8)+16*manual.astype(np.uint8)
    return reasons==0,reasons,tilt

class Scorer:
    """Scores one GBA and reference average per common fixed-duration time bin.

    Bins with any masked GBA samples, inadequate coverage, no reference sample,
    or any nonzero reference flag are excluded. All methods share these bins.
    SSE, MSE, RMSE, MAE, bias and correlation are reported. Reference is never
    upsampled to 100 Hz for scoring.
    """
    def __init__(self,d,label,shift,threshold=None,padding=None,width=None):
        if width is None:width=SETTINGS['evaluation_bin_s']
        self.mask,self.reasons,self.tilt=mask_data(d,label,shift,threshold,padding)
        self.tj=d['tj']+shift; self.width=width
        self.lo=np.floor(self.tj[0]/width)*width
        self.ids=np.floor((self.tj-self.lo)/width).astype(int)
        self.n=int(self.ids[-1])+1
        self.count=np.bincount(self.ids,minlength=self.n)
        bad=np.bincount(self.ids,weights=~self.mask,minlength=self.n)
        ref=d['ref'];rid=np.floor((ref[:,0]-self.lo)/width).astype(int)
        keep=(rid>=0)&(rid<self.n);rid=rid[keep];ref=ref[keep]
        rc=np.bincount(rid,minlength=self.n)
        rb=np.bincount(rid,weights=(ref[:,1]!=0),minlength=self.n)
        total=np.bincount(rid,weights=ref[:,2],minlength=self.n)
        self.reference=np.divide(total,rc,out=np.full(self.n,np.nan),where=rc>0)
        self.valid=(bad==0)&(self.count>=FS*width*.8)&(rc>0)&(rb==0)
        self.time=self.lo+(np.arange(self.n)+.5)*width
        self.refcount=rc
    def average(self,y):
        return np.divide(np.bincount(self.ids,weights=y,minlength=self.n),self.count,out=np.full(self.n,np.nan),where=self.count>0)
    def metric(self,y):
        yb=self.average(y);ok=self.valid
        if not np.isfinite(yb[ok]).all():raise ValueError('Nonfinite prediction on scored bins')
        e=yb[ok]-self.reference[ok]
        return dict(n=int(ok.sum()),sse=float(e@e),mse=float(np.mean(e*e)),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(np.abs(e))),bias=float(np.mean(e)),corr=float(np.corrcoef(yb[ok],self.reference[ok])[0,1]))

def clock_alignment(d):
    # A single shared inter-instrument offset estimated from morning low-frequency
    # data. No afternoon reference enters this decision.
    speed=to_speed(K0*np.sin(np.deg2rad(d['angle'])))
    smooth=signal.sosfiltfilt(signal.butter(3,.1,fs=FS,output='sos'),speed)
    ref=d['ref'];grid=np.arange(np.ceil(ref[0,0]*4)/4,ref[-1,0],.25)
    r=np.interp(grid,ref[:,0],ref[:,2])
    r=signal.sosfiltfilt(signal.butter(3,.1,fs=4,output='sos'),r)
    valid,_,_=mask_data(d,'am',0)
    # A fixed mask valid for the whole +/-5 s offset search.
    valid=ndimage.minimum_filter1d(valid.astype(np.uint8),size=1001,mode='constant')>0
    ok=(grid>d['tj'][0]+30)&(grid<d['tj'][-1]-30)&(np.interp(grid,d['tj'],valid.astype(float))>.999)
    records=[]
    for lag in np.arange(-5,5.00001,.05):
        p=np.interp(grid,d['tj']+lag,smooth)
        # Remove gain as a nuisance only for synchronization; no output gain is fit.
        gain=np.dot(p[ok],r[ok])/np.dot(p[ok],p[ok])
        records.append((lag,np.mean((gain*p[ok]-r[ok])**2),gain))
    tab=np.array(records);best=tab[np.argmin(tab[:,1])]
    pd.DataFrame(tab,columns=['shared_clock_shift_s','alignment_mse','nuisance_gain_not_applied']).to_csv(OUT/'clock_alignment.csv',index=False)
    return float(best[0])

def main():
    ds=load_records();m=identify();lag=clock_alignment(ds['am'])
    print('MODEL',m,'LAG',lag,'CD_EQ',CD,flush=True)
    scorers={s:Scorer(d,s,lag) for s,d in ds.items()}
    # One common scale correction from AM only, using the fixed slow LPF below.
    # It is frozen for all methods, parameter sweeps and PM evaluation.
    calibration_q0,_,_=basis(ds['am']['angle'],.2)
    calibration_x=scorers['am'].average(to_speed(K0*calibration_q0))[scorers['am'].valid]
    calibration_r=scorers['am'].reference[scorers['am'].valid]
    gain=float(calibration_x@calibration_r/(calibration_x@calibration_x))
    def speed(tau):return gain*to_speed(tau)
    print('COMMON AM GAIN',gain,'CALIBRATED EQUIVALENT CD',CD/gain**2,flush=True)
    scan=[]; arrays={s:{} for s in ds}; basics={s:{} for s in ds}
    for label,d in ds.items():
        arrays[label]['Static']=speed(K0*np.sin(np.deg2rad(d['angle'])))
        scan.append(dict(dataset=label,method='Static',pole_hz=0,**scorers[label].metric(arrays[label]['Static'])))
        for f in POLES:
            q0,q1,q2=basis(d['angle'],f)
            # Cache only nominal outputs; sensitivity basis is recomputed once later.
            low=speed(K0*q0);obs=speed(m['I']*q2+m['b']*q1+K0*q0)
            arrays[label][('Low-pass',f)]=low;arrays[label][('Observer',f)]=obs
            scan.extend([dict(dataset=label,method=method,pole_hz=f,**scorers[label].metric(y)) for method,y in [('Low-pass',low),('Observer',obs)]])
    grid=pd.DataFrame(scan);grid.to_csv(OUT/'pole_scan.csv',index=False)
    choice={}
    for method in ['Low-pass','Observer']:
        train=grid[(grid.dataset=='am')&(grid.method==method)]
        row=train.loc[train.rmse.idxmin()];choice[method]=float(row.pole_hz)
    print('CHOICE',choice,flush=True)
    selected=[];preds={}
    for label,d in ds.items():
        preds[label]={'Static':arrays[label]['Static'],**{method:arrays[label][(method,f)] for method,f in choice.items()}}
        # Same smoothing strength as observer: isolates dynamics from filter tuning.
        preds[label]['Low-pass matched']=arrays[label][('Low-pass',choice['Observer'])]
        q0,q1,q2=basis(d['angle'],choice['Observer'],False)
        preds[label]['Observer causal']=speed(m['I']*q2+m['b']*q1+K0*q0)
        for method,y in preds[label].items():
            selected.append(dict(dataset=label,method=method,**scorers[label].metric(y)))
    metrics=pd.DataFrame(selected);metrics.to_csv(OUT/'metrics.csv',index=False)
    fixed_rows=[]
    for label in ds:
        for method,y in preds[label].items():fixed_rows.append(dict(dataset=label,method=method,**scorers[label].metric(y/gain)))
    pd.DataFrame(fixed_rows).to_csv(OUT/'metrics_fixed_scale.csv',index=False)
    print(metrics.to_string(index=False),flush=True)
    sensitivities=[]
    for label,d in ds.items():
        q0,q1,q2=basis(d['angle'],choice['Observer'])
        for name in ['I','K','b','all']:
            factors=[0,.25,.5,.75,.9,1.,1.1,1.25,1.5,2.] if name=='b' else [.5,.75,.9,1.,1.1,1.25,1.5,2.]
            for factor in factors:
                kk=K0*(factor if name in ['K','all'] else 1)
                ii=m['I']*(factor if name in ['I','all'] else 1)
                bb=m['b']*(factor if name in ['b','all'] else 1)
                y=speed(ii*q2+bb*q1+kk*q0)
                sensitivities.append(dict(dataset=label,pole_hz=choice['Observer'],parameter=name,factor=factor,**scorers[label].metric(y)))
        for pole in [1.,2.]:
            q0,q1,q2=basis(d['angle'],pole)
            for name in ['I','K','b']:
                for factor in [.5,.75,.9,1.,1.1,1.25,1.5,2.]:
                    kk=K0*(factor if name=='K' else 1);ii=m['I']*(factor if name=='I' else 1);bb=m['b']*(factor if name=='b' else 1)
                    sensitivities.append(dict(dataset=label,pole_hz=pole,parameter=name,factor=factor,**scorers[label].metric(speed(ii*q2+bb*q1+kk*q0))))
    sensitivity=pd.DataFrame(sensitivities);sensitivity.to_csv(OUT/'model_sensitivity.csv',index=False)
    mask_rows=[]
    for label,d in ds.items():
        for th in [34.,36.,37.]:
            for pad in [5.,10.,20.]:
                sc=Scorer(d,label,lag,th,pad)
                for method in ['Static','Low-pass','Observer']:
                    mask_rows.append(dict(dataset=label,threshold_deg=th,padding_s=pad,method=method,**sc.metric(preds[label][method])))
    pd.DataFrame(mask_rows).to_csv(OUT/'mask_sensitivity.csv',index=False)
    # Sensitivity to bin width and to retaining the original GPS clock alignment.
    eval_rows=[]
    for label,d in ds.items():
        for shift in [0.,lag]:
            for width in [.25,1.]:
                sc=Scorer(d,label,shift,width=width)
                for method in ['Static','Low-pass','Observer']:
                    eval_rows.append(dict(dataset=label,clock_shift_s=shift,bin_width_s=width,method=method,**sc.metric(preds[label][method])))
    pd.DataFrame(eval_rows).to_csv(OUT/'evaluation_sensitivity.csv',index=False)
    config=dict(source_commit='dbc153d89fa335aca14014e0173c587e8adc93a8',model=m,cd_effective=CD,legacy_cd=.55,legacy_K=OLD_K,rho=RHO,area=AREA,lever_m=LENGTH,fs_hz=FS,choices=choice,shared_clock_shift_s=lag,threshold_deg=36,padding_s=10,bin_width_s=.25,observer_delay_compensation_s=3/(2*np.pi*choice['Observer']),force_filter_3db_hz=choice['Observer']*np.sqrt(2**(1/3)-1),scope='PoC, independent axes, inner-axis free-decay characteristics provisionally used on both axes; legacy wind conversion retained; not validated absolute aerodynamic force')
    config.update(common_am_gain=gain,calibrated_cd_equivalent=CD/gain**2,calibration='AM low-pass with fixed pole parameter 0.2 Hz; one shared multiplier, no additive offset',settings=SETTINGS,threshold_deg=SETTINGS['threshold_deg'],padding_s=SETTINGS['saturation_padding_s'],bin_width_s=SETTINGS['evaluation_bin_s'])
    (OUT/'config_used.json').write_text(json.dumps(config,indent=2))
    for label,d in ds.items():
        sc=scorers[label]; q0,q1,q2=basis(d['angle'],choice['Observer']);tau=m['I']*q2+m['b']*q1+K0*q0
        frame=pd.DataFrame({'device_time_s':d['t'],'jst_time_s_gps':d['tj'],'jst_time_s_aligned':d['tj']+lag,'outer_angle_deg':d['angle'][:,0],'inner_angle_deg':d['angle'][:,1],'combined_tilt_proxy_deg':sc.tilt,'valid_sample':sc.mask.astype(int),'exclusion_bits':sc.reasons,'tau_outer_equiv_Nm':tau[:,0],'tau_inner_equiv_Nm':tau[:,1],'force_outer_equiv_N':tau[:,0]/LENGTH,'force_inner_equiv_N':tau[:,1]/LENGTH,'speed_static_mps':preds[label]['Static'],'speed_lowpass_mps':preds[label]['Low-pass'],'speed_observer_mps':preds[label]['Observer'],'speed_observer_causal_mps':preds[label]['Observer causal']})
        frame['speed_static_original_cd055_mps']=preds[label]['Static']/gain
        frame['speed_observer_fixed_scale_mps']=preds[label]['Observer']/gain
        # Numeric estimates in exclusion windows remain available but are invalid.
        frame.to_csv(OUT/f'estimates_100hz_{label}.csv.gz',index=False,float_format='%.7g',compression={'method':'gzip','mtime':0})
        scored=pd.DataFrame({'jst_time_s':sc.time,'valid_comparison':sc.valid.astype(int),'reference_mps':sc.reference,'reference_count':sc.refcount})
        for method,y in preds[label].items():scored[method]=sc.average(y)
        scored.to_csv(OUT/f'comparison_binned_{label}.csv',index=False,float_format='%.7g')
    plots(ds,scorers,preds,metrics,grid,sensitivity,choice,m,lag)
    export_wider_band(ds,scorers,gain,m,lag)
    validate_and_theory(m)
    print('DONE',flush=True)

def shade(ax,time,bad):
    edges=np.diff(np.r_[False,bad,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    for a,b in zip(starts,ends):ax.axvspan(time[a],time[min(b,len(time)-1)],color='#d9dde1',alpha=.7,lw=0)

def plots(ds,scorers,preds,metrics,grid,sensitivity,choice,m,lag):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    fig,axes=plt.subplots(2,1,figsize=(15,8),layout='constrained')
    for ax,label in zip(axes,['am','pm']):
        sc=scorers[label];t=sc.time/3600
        shade(ax,t,~sc.valid)
        for method in ['Static','Observer']:
            y=sc.average(preds[label][method]);y[~sc.valid]=np.nan
            ax.plot(t,y,color=COLORS[method],lw=.65,alpha=.7 if method=='Static' else 1,label=method)
        r=sc.reference.copy();r[~sc.valid]=np.nan;ax.plot(t,r,color=COLORS['Reference'],lw=.8,alpha=.85,label='Reference')
        ax.set_ylabel('Wind-speed proxy [m/s]');ax.set_xlabel('JST [hours]');ax.set_ylim(0,3.6);ax.set_title(f'{label.upper()} — '+('parameter selection' if label=='am' else 'held-out evaluation'));ax.legend(loc='upper right',ncol=3)
    fig.suptitle(f"GBA observer PoC | {SETTINGS['evaluation_bin_s']:g} s averages | common AM gain | gray = excluded",fontsize=15)
    fig.savefig(OUT/'overview.png',dpi=170);plt.close(fig)
    # Deterministic zoom: earliest fully valid 60 s window after 1/3 of PM.
    label='pm';sc=scorers[label];eligible=np.flatnonzero(sc.valid & (sc.time>sc.time[0]+(sc.time[-1]-sc.time[0])/3))
    start=None
    bins_in_minute=int(np.ceil(60/sc.width))
    for i in eligible:
        if i+bins_in_minute<len(sc.valid) and sc.valid[i:i+bins_in_minute].all():start=sc.time[i];break
    if start is None:start=sc.time[eligible[0]] if len(eligible) else sc.time[0]
    end=start+60;d=ds[label];idx=(d['tj']+lag>=start)&(d['tj']+lag<=end)
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained',sharex=True)
    for method in ['Static','Low-pass','Observer']:
        axes[0].plot(d['tj'][idx]+lag-start,preds[label][method][idx],color=COLORS[method],lw=.8 if method=='Static' else 1.4,alpha=.5 if method=='Static' else .95,label=method)
    ref=d['ref'];ri=(ref[:,0]>=start)&(ref[:,0]<=end);axes[0].plot(ref[ri,0]-start,ref[ri,2],color=COLORS['Reference'],lw=1.2,marker='.',ms=3,label='Reference (~4 Hz)')
    axes[0].set_ylabel('Wind-speed proxy [m/s]');axes[0].legend(ncol=4,loc='upper right');axes[0].set_title(f'PM evaluation | common AM gain | start JST {int(start//3600):02}:{int(start%3600//60):02}:{start%60:04.1f}')
    bi=(sc.time>=start)&(sc.time<=end)&sc.valid
    for method in ['Static','Low-pass','Observer']:axes[1].plot(sc.time[bi]-start,sc.average(preds[label][method])[bi]-sc.reference[bi],color=COLORS[method],lw=1,label=method)
    axes[1].axhline(0,color='black',lw=.6);axes[1].set_ylabel(f'Error of {sc.width:g} s means [m/s]');axes[1].set_xlabel('Elapsed time [s]')
    fig.savefig(OUT/'detail.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,label in zip(axes,['am','pm']):
        tab=metrics[(metrics.dataset==label)&metrics.method.isin(['Static','Low-pass','Observer'])].set_index('method').loc[['Static','Low-pass','Observer']]
        bars=ax.bar(tab.index,tab.rmse,color=[COLORS[x] for x in tab.index],width=.6)
        ax.bar_label(bars,fmt='%.3f',padding=4);ax.set_ylim(0,tab.rmse.max()*1.25);ax.set_ylabel('RMSE [m/s]');ax.set_title(label.upper()+(' (selection)' if label=='am' else ' (held out)'))
    fig.suptitle('Common AM gain, same timestamps, same exclusion mask',fontsize=13);fig.savefig(OUT/'rmse.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    for ax,param in zip(axes,['I','K','b']):
        for label,color in [('am','#3573b5'),('pm','#dd7a25')]:
            tab=sensitivity[(sensitivity.dataset==label)&(sensitivity.parameter==param)&(sensitivity.pole_hz==choice['Observer'])];ax.plot(tab.factor,tab.rmse,'o-',color=color,label=label.upper())
        ax.axvline(1,color='gray',ls='--');ax.set_xlabel(f'{param} / nominal {param}');ax.set_ylabel('RMSE [m/s]');ax.set_title({'I':'Inertia mismatch','K':'Restoring coefficient mismatch','b':'Damping mismatch'}[param]);ax.legend()
    fig.suptitle('One parameter at a time; observer pole, clock shift and Cd fixed',fontsize=13);fig.savefig(OUT/'sensitivity.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for label,style in [('am','-'),('pm','--')]:
        for method in ['Low-pass','Observer']:
            tab=grid[(grid.dataset==label)&(grid.method==method)];ax.semilogx(tab.pole_hz,tab.rmse,style,marker='o',color=COLORS[method],label=f'{method} {label.upper()}')
    ax.set_xlabel('Repeated observer/filter pole parameter p/(2pi) [Hz]');ax.set_ylabel('RMSE [m/s]');ax.legend();ax.set_title('Bandwidth trade-off (selection uses AM only)');fig.savefig(OUT/'bandwidth_scan.png',dpi=180);plt.close(fig)

def export_wider_band(ds,scorers,gain,m,lag):
    """A fixed 1 Hz pole example, retaining substantially more pendulum-band input.

    This is a comparison setting, not selected using the afternoon reference.
    Q's -3 dB torque bandwidth is approximately 0.51 Hz.
    """
    values={}
    for label,d in ds.items():
        q0,q1,q2=basis(d['angle'],1.)
        tau=m['I']*q2+m['b']*q1+K0*q0
        low=gain*to_speed(K0*q0);obs=gain*to_speed(tau)
        values[label]=(low,obs)
        pd.DataFrame({'device_time_s':d['t'],'jst_time_s_aligned':d['tj']+lag,
          'valid_sample':scorers[label].mask.astype(int),'exclusion_bits':scorers[label].reasons,
          'tau_outer_equiv_Nm':tau[:,0],'tau_inner_equiv_Nm':tau[:,1],
          'speed_lowpass_mps':low,'speed_observer_mps':obs
        }).to_csv(OUT/f'estimates_100hz_pole1hz_{label}.csv.gz',index=False,float_format='%.7g',compression={'method':'gzip','mtime':0})
    # Same deterministic PM interval as the main detail plot.
    d=ds['pm'];sc=scorers['pm'];nb=int(np.ceil(60/sc.width))
    eligible=np.flatnonzero(sc.valid&(sc.time>sc.time[0]+(sc.time[-1]-sc.time[0])/3))
    start=next((sc.time[i] for i in eligible if i+nb<len(sc.valid) and sc.valid[i:i+nb].all()),sc.time[0])
    idx=(d['tj']+lag>=start)&(d['tj']+lag<=start+60)
    low,obs=values['pm'];static=gain*to_speed(K0*np.sin(np.deg2rad(d['angle'])))
    fig,ax=plt.subplots(figsize=(13,4.5),layout='constrained')
    for name,y in [('Static',static),('Low-pass',low),('Observer',obs)]:
        ax.plot(d['tj'][idx]+lag-start,y[idx],color=COLORS[name],lw=.8 if name=='Static' else 1.3,alpha=.5 if name=='Static' else 1,label=name)
    ref=d['ref'];ri=(ref[:,0]>=start)&(ref[:,0]<=start+60)
    ax.plot(ref[ri,0]-start,ref[ri,2],'.-',color=COLORS['Reference'],lw=1,label='Reference (~4 Hz)')
    ax.set_title('PM evaluation | pole parameter 1 Hz | torque-filter bandwidth 0.51 Hz | common AM gain')
    ax.set_xlabel('Elapsed time [s]');ax.set_ylabel('Wind-speed proxy [m/s]');ax.legend(ncol=4)
    fig.savefig(OUT/'detail_pole1hz.png',dpi=180);plt.close(fig)

def validate_and_theory(m):
    checks={}
    angle=np.full((10000,2),20.)
    q0,q1,q2=basis(angle,1.,False)
    expected=K0*np.sin(np.deg2rad(angle))
    actual=m['I']*q2+m['b']*q1+K0*q0
    checks['constant_angle_max_abs_torque_error_Nm']=float(np.max(np.abs(actual-expected)))
    assert checks['constant_angle_max_abs_torque_error_Nm']<1e-8
    t=np.arange(0,120,.01);sl=(t>30)&(t<100);test=[]
    for f in [.1,m['fn_hz'],1.]:
        s=2j*np.pi*f;plant=m['I']*s*s+m['b']*s+K0
        amp=1e-5;theta=np.real(amp/plant*np.exp(s*t))
        angle=np.rad2deg(np.column_stack([theta,theta]))
        q0,q1,q2=basis(angle,1.,False);estimated=(m['I']*q2+m['b']*q1+K0*q0)[:,0]
        design=np.column_stack([np.cos(2*np.pi*f*t[sl]),np.sin(2*np.pi*f*t[sl]),np.ones(sl.sum())])
        coeff=np.linalg.lstsq(design,estimated[sl],rcond=None)[0];observed_gain=float(np.hypot(*coeff[:2])/amp)
        p=2*np.pi;target_gain=float(abs(p**3/(s+p)**3))
        test.append(dict(f_hz=f,estimated_gain=observed_gain,continuous_target_gain=target_gain))
        assert abs(observed_gain/target_gain-1)<.01
    checks['noise_free_sinusoid_checks']=test
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2))
    f=np.geomspace(.02,10,800);s=2j*np.pi*f
    plant=m['I']*s*s+m['b']*s+K0
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    p=2*np.pi;Q=p**3/(s+p)**3
    axes[0].loglog(f,abs(K0/plant),color=COLORS['Static'],label='Static K theta')
    axes[0].loglog(f,abs(Q*K0/plant),color=COLORS['Low-pass'],label='Low-pass static')
    axes[0].loglog(f,abs(Q),color=COLORS['Observer'],label='Matched observer')
    for factor in [.75,1.,1.25]:
        transfer=Q*(factor*m['I']*s*s+m['b']*s+K0)/plant
        axes[1].loglog(f,abs(transfer),label=f'I model / I true = {factor:g}')
    for ax in axes:
        ax.axvline(m['fn_hz'],color='gray',ls='--');ax.axhline(1,color='gray',lw=.7);ax.set_xlabel('Input torque frequency [Hz]');ax.set_ylabel('|Estimated torque / input torque|');ax.legend();ax.set_ylim(.01,20)
    axes[0].set_title('Nominal linear model: resonance compensation');axes[1].set_title('Inertia mismatch near resonance')
    fig.suptitle('Theory only | pole parameter 1 Hz | not a measured wind bandwidth',fontsize=13);fig.savefig(OUT/'theory_frequency_response.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path,default=ANALYSIS_ROOT/'config/settings.json')
    parser.add_argument('--cache',type=Path,default=CACHE)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    SETTINGS=json.loads(args.settings.read_text())
    POLES=SETTINGS['pole_hz_grid'];CACHE=args.cache;OUT=args.output
    if not POLES or min(POLES)<=0:raise ValueError('Pole frequencies must be positive')
    if SETTINGS['saturation_padding_s']<0 or SETTINGS['evaluation_bin_s']<=0:raise ValueError('Invalid exclusion or scoring settings')
    OUT.mkdir(parents=True,exist_ok=True)
    main()
