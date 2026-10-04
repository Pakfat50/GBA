"""OW-05 sensor non-ideality evaluation for adopted HBK observers."""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, normaltest, skew

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hbk_model_coefficients import select_coefficients, simulate_hbk_plant
from hbk_nonlinear_estimators import nonlinear_ekf_rts_force, nonlinear_luenberger_force
from run_ow03_hbk_observer_tuning import applied_force, force_to_speed, make_wind, metrics
from run_ow04_hbk_coefficient_sensitivity import best_parameter_values
from sensor_model import AngleSensorParameters, apply_angle_sensor_model
from ow05_free_decay_validation import evaluate_free_decay
from run_ow03_hbk_observer_tuning import static_force

SIM=Path(__file__).resolve().parents[1]
OUT=SIM/'results/observer_wind/ow05_sensor_nonideality'
REPO=SIM.parents[1]

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def analyze_measured_sensor_noise():
    """Measure each static-window angle perturbation from its own mean."""
    OUT.mkdir(parents=True,exist_ok=True)
    data_root=REPO/'04_Data/05_Fitting/20260921'
    selection_path=REPO/'06_Analysis/fitting_pipeline/results/20260921/waveform_review/waveform_selection.csv'
    selection=pd.read_csv(selection_path,encoding='utf-8-sig')
    selection=selection[(selection['use_for_fitting']==1)&(selection['review_status'].astype(str).str.upper()=='APPROVED')]
    tables={}
    estimates={}
    window_n=50
    for axis in ('IN','OUT'):
      angle_col='angle0[deg]' if axis=='IN' else 'angle1[deg]'
      windows=[]
      for _,record in selection[selection['axis']==axis].iterrows():
        relative_path=str(record['data_file'])
        if relative_path not in tables:
          tables[relative_path]=pd.read_csv(data_root/relative_path)
        table=tables[relative_path]
        y=pd.to_numeric(table[angle_col],errors='coerce').to_numpy(float)
        y=(y+180.0)%360.0-180.0
        event=int(record['start_index'])
        expected_sign=1.0 if str(record['direction']).upper().startswith('P') else -1.0
        selected=None
        selected_start=None
        for end in range(event-10,max(window_n,event-1000),-10):
          start=end-window_n
          x=y[start:end]
          if len(x)!=window_n or not np.all(np.isfinite(x)):
            continue
          slope=float(np.polyfit(np.arange(window_n)*0.01,x,1)[0])
          if expected_sign*np.mean(x)>35.0 and np.std(x)<0.20 and abs(slope)<1.0:
            selected=x
            selected_start=start
            break
        if selected is None:
          continue
        perturbation=selected-float(np.mean(selected))
        windows.append({'segment_id':str(record['segment_id']),'data_file':relative_path,
                        'direction':str(record['direction']),'start_index':selected_start,
                        'samples':window_n,'perturbation':perturbation})
      if not windows:
        raise RuntimeError('No stationary pre-release windows found for '+axis)
      perturbations=np.concatenate([w['perturbation'] for w in windows])
      sigma=float(np.std(perturbations,ddof=1))
      acf=[]
      denominator=sum(float(np.dot(w['perturbation'],w['perturbation'])) for w in windows)
      for lag in range(1,21):
        numerator=sum(float(np.dot(w['perturbation'][:-lag],w['perturbation'][lag:])) for w in windows)
        acf.append(numerator/denominator if denominator>0 else 0.0)
      normal=normaltest(perturbations)
      estimates[axis]={
        'n_windows':len(windows),'n_samples':int(len(perturbations)),
        'static_perturbation_sigma_deg':sigma,'three_sigma_deg':3*sigma,
        'assumed_white_sigma_deg':0.015,'assumed_white_three_sigma_deg':0.045,
        'skewness':float(skew(perturbations)),
        'excess_kurtosis':float(kurtosis(perturbations)),
        'normality_k2':float(normal.statistic),'normality_p_value':float(normal.pvalue),
        'fraction_within_three_sigma':float(np.mean(np.abs(perturbations)<=3*sigma)),
        'residual_acf_lag1':float(acf[0]),
        'acf_95_bound':float(1.96/np.sqrt(len(perturbations))),
        'acf':acf,'windows':windows,'perturbations':perturbations}
    summary_rows=[]
    for axis,v in estimates.items():
      summary_rows.append({k:value for k,value in v.items() if k not in ('acf','windows','perturbations')})
    write_csv(OUT/'ow05_measured_noise_summary.csv',summary_rows)
    window_rows=[]
    for axis,v in estimates.items():
      for w in v['windows']:
        window_rows.append({'axis':axis,'segment_id':w['segment_id'],'data_file':w['data_file'],
                            'direction':w['direction'],'start_index':w['start_index'],
                            'samples':w['samples'],'duration_s':w['samples']/100.0,
                            'perturbation_sigma_deg':float(np.std(w['perturbation'],ddof=1))})
    write_csv(OUT/'ow05_measured_noise_windows.csv',window_rows)
    for axis,v in estimates.items():
      fig,axes=plt.subplots(3,1,figsize=(9,8),layout='constrained')
      perturbations=v['perturbations']; sigma=v['static_perturbation_sigma_deg']
      axes[0].plot(np.arange(len(perturbations))*0.01,perturbations,lw=.8,color='#2673a8',
                   label='Measured static-window perturbation')
      axes[0].axhspan(-3*sigma,3*sigma,color='#4c9f70',alpha=.16,
                      label=f'Measured perturbation ±3σ = ±{3*sigma:.4f}°')
      axes[0].axhline(0,color='#444',lw=.7)
      axes[0].set_ylabel('Window-mean-subtracted angle [deg]')
      axes[0].set_xlabel('Concatenated stationary-window time [s]')
      axes[0].set_title(f'{axis} axis: measured readings minus each window mean')
      axes[0].legend(loc='upper right',fontsize=8); axes[0].grid(alpha=.2)
      limit=max(float(np.max(np.abs(perturbations))*1.05),3*sigma*1.15)
      bins=np.linspace(-limit,limit,81)
      axes[1].hist(perturbations,bins=bins,density=True,color='#79a9c9',alpha=.65,
                   label='Measured static-window perturbations')
      xx=np.linspace(-limit,limit,800)
      gaussian=np.exp(-0.5*(xx/sigma)**2)/(sigma*np.sqrt(2*np.pi))
      model_sigma=0.015
      model_gaussian=np.exp(-0.5*(xx/model_sigma)**2)/(model_sigma*np.sqrt(2*np.pi))
      axes[1].plot(xx,gaussian,color='#193d5a',lw=2,
                   label=f'Gaussian reference, σ={sigma:.4f}°')
      axes[1].plot(xx,model_gaussian,color='#c44e52',ls='--',lw=1.4,
                   label='OW-05 assumed white σ=0.015°')
      axes[1].axvline(-3*sigma,color='#238b45',ls=':',lw=1.4)
      axes[1].axvline(3*sigma,color='#238b45',ls=':',lw=1.4)
      axes[1].axvline(-3*model_sigma,color='#c44e52',ls='--',lw=1)
      axes[1].axvline(3*model_sigma,color='#c44e52',ls='--',lw=1)
      axes[1].set_xlim(-limit,limit); axes[1].set_ylabel('Probability density')
      axes[1].set_xlabel('Window-mean-subtracted angle [deg]')
      axes[1].set_title(f"Static-window perturbations; {100*v['fraction_within_three_sigma']:.1f}% within ±3σ (Gaussian: 99.73%)")
      axes[1].legend(loc='upper right',fontsize=8); axes[1].grid(alpha=.2)
      lags=np.arange(1,len(v['acf'])+1); bound=v['acf_95_bound']
      axes[2].bar(lags,v['acf'],color='#8172b2',width=.8)
      axes[2].axhline(0,color='#444',lw=.7)
      axes[2].axhspan(-bound,bound,color='#777',alpha=.16,
                      label=f'Approx. 95% white-noise bound ±{bound:.3f}')
      axes[2].set_xlabel('Lag [samples]'); axes[2].set_ylabel('ACF')
      axes[2].set_title(f"Static-window perturbation autocorrelation; lag 1 = {v['residual_acf_lag1']:.3f}")
      axes[2].legend(loc='upper right',fontsize=8); axes[2].grid(alpha=.2)
      fig.savefig(OUT/f'ow05_measured_noise_{axis.lower()}.png',dpi=170); plt.close(fig)
    return estimates
def run():
    cfg=json.loads((SIM/'config/ow03_hbk_observer_tuning.json').read_text())
    scfg=json.loads((SIM/'config/sensor_model_stage3.json').read_text())
    prev=json.loads((SIM/'results/observer_wind/ow03_tuning/ow03_summary.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True); dt=1/cfg['sample_rate_hz']; rate=cfg['sample_rate_hz']
    measured_noise=analyze_measured_sensor_noise()
    cases=[x for x in cfg['validation_winds'] if x['name'] in ('独立Kaimal乱流 平均2 m/s TI20%','ガスト 2→6 m/s')]
    plants={}
    for case in cases:
      t,v,_=make_wind(cfg,case); F=applied_force(cfg,v)
      for axis in cfg['axis_names']:
        co=select_coefficients(axis,'BALL'); st=simulate_hbk_plant(F,co,dt,force_lever_m=cfg['force_lever_m'],friction_epsilon_deg_s=cfg['friction_epsilon_deg_s'])
        ang=st[:,0]; mask=(t>=cfg['evaluation_start_s']); maxang=float(np.max(np.abs(np.rad2deg(ang[mask]))))
        plants[(case['name'],axis)]={'t':t,'v':v,'F':F,'ang':ang,'mask':mask,'maxang':maxang}
    nominal=scfg['sensor']; pars=AngleSensorParameters(sample_rate_hz=rate,**{k:v for k,v in nominal.items() if k!='resolution_bits'},resolution_bits=nominal['resolution_bits'])
    # Train on a separate deterministic Kaimal record; choose only causal ESO and offline RTS 3-state.
    train_desc={'name':'OW05 tuning Kaimal','kind':'kaimal','mean_wind_speed_m_s':3.0,'target_turbulence_intensity':.2,'maximum_wind_speed_m_s':5.0,'seed':20261031}
    tc,tv,_=make_wind(cfg,train_desc); tf=applied_force(cfg,tv); tunings={}; tuning_rows=[]
    for axis in cfg['axis_names']:
      co=select_coefficients(axis,'BALL'); st=simulate_hbk_plant(tf,co,dt,force_lever_m=cfg['force_lever_m'],friction_epsilon_deg_s=cfg['friction_epsilon_deg_s'])
      ma,_=apply_angle_sensor_model(tc,st[:,0],pars,20261032+(axis=='OUT'))
      for method in ['ESO 3状態','RTS 3状態（オフライン）']:
        best=(float('inf'),None)
        grid=cfg['eso_pole_grid_hz'] if method.startswith('ESO') else [1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1,3,10,30]
        for p in grid:
          try:
            if method.startswith('ESO'): force=nonlinear_luenberger_force(ma,co,dt,cfg['force_lever_m'],p,0,cfg['friction_epsilon_deg_s'])
            else: _,force=nonlinear_ekf_rts_force(ma,co,dt,cfg['force_lever_m'],np.deg2rad(.02),p,0,cfg['friction_epsilon_deg_s'])
            est=force_to_speed(cfg,force); met=metrics(tv,est,tc>=15)
            if np.isfinite(met['rmse_m_s']) and met['rmse_m_s']<best[0]: best=(met['rmse_m_s'],p)
            tuning_rows.append({'axis':axis,'method':method,'parameter':p,'training_rmse_m_s':met['rmse_m_s']})
          except (ValueError,FloatingPointError,np.linalg.LinAlgError): pass
        tunings[(axis,method)]=best[1]
    rows=[]; noise_rows=[]; wave={}; static_sigma_wave={}
    measured_sigma={axis:measured_noise[axis]['static_perturbation_sigma_deg'] for axis in ('IN','OUT')}
    modes={'ideal':None,'nominal':pars,'noise_2x':AngleSensorParameters(sample_rate_hz=rate,resolution_bits=14,white_noise_std_deg=.03,coloured_noise_std_deg=.02,coloured_noise_time_constant_s=pars.coloured_noise_time_constant_s,fixed_delay_s=.01)}
    static_sigma_modes={axis:AngleSensorParameters(sample_rate_hz=rate,resolution_bits=14,white_noise_std_deg=measured_sigma[axis],coloured_noise_std_deg=0.0,coloured_noise_time_constant_s=pars.coloured_noise_time_constant_s,fixed_delay_s=pars.fixed_delay_s) for axis in cfg['axis_names']}
    for cname,axis in plants:
      plant=plants[(cname,axis)]
      for sens,sp in list(modes.items())+[('static_window_sigma',static_sigma_modes[axis])]:
        sensor_seed=20261100+len(rows)
        angle=plant['ang'] if sp is None else apply_angle_sensor_model(plant['t'],plant['ang'],sp,sensor_seed)[0]
        if sp is not None:
          # Match all deterministic sensor effects and remove only random angle noise.
          angle0_params=AngleSensorParameters(sample_rate_hz=sp.sample_rate_hz,resolution_bits=sp.resolution_bits,full_scale_deg=sp.full_scale_deg,white_noise_std_deg=0.0,coloured_noise_std_deg=0.0,coloured_noise_time_constant_s=sp.coloured_noise_time_constant_s,fixed_delay_s=sp.fixed_delay_s,sampling_jitter_std_s=sp.sampling_jitter_std_s,gain_error_fraction=sp.gain_error_fraction,offset_deg=sp.offset_deg)
          angle0=apply_angle_sensor_model(plant['t'],plant['ang'],angle0_params,sensor_seed)[0]
          dtheta=angle-angle0
        for method in ['ESO 3状態','RTS 3状態（オフライン）']:
          p=tunings[(axis,method)]; co=select_coefficients(axis,'BALL')
          try:
            if method.startswith('ESO'): force=nonlinear_luenberger_force(angle,co,dt,cfg['force_lever_m'],p,0,cfg['friction_epsilon_deg_s'])
            else: _,force=nonlinear_ekf_rts_force(angle,co,dt,cfg['force_lever_m'],np.deg2rad(.02),p,0,cfg['friction_epsilon_deg_s'])
            est=force_to_speed(cfg,force); met=metrics(plant['v'],est,plant['mask'])
            if sp is not None:
              if method.startswith('ESO'): force0=nonlinear_luenberger_force(angle0,co,dt,cfg['force_lever_m'],p,0,cfg['friction_epsilon_deg_s'])
              else: _,force0=nonlinear_ekf_rts_force(angle0,co,dt,cfg['force_lever_m'],np.deg2rad(.02),p,0,cfg['friction_epsilon_deg_s'])
              est0=force_to_speed(cfg,force0)
              h=1e-5
              vplus=force_to_speed(cfg,static_force(angle0+h,co,cfg['force_lever_m']))
              vminus=force_to_speed(cfg,static_force(angle0-h,co,cfg['force_lever_m']))
              slope=(vplus-vminus)/(2*h); linear_ref=slope*dtheta; observer_delta=est-est0; m=plant['mask']
              rmse_ref=float(np.sqrt(np.mean(linear_ref[m]**2))); rmse_out=float(np.sqrt(np.mean(observer_delta[m]**2)))
              max_ref=float(np.max(np.abs(linear_ref[m]))); max_out=float(np.max(np.abs(observer_delta[m])))
              p95_ref=float(np.quantile(np.abs(linear_ref[m]),0.95)); p95_out=float(np.quantile(np.abs(observer_delta[m]),0.95))
              linear_valid=float(np.mean(np.abs(dtheta[m])<=0.1*np.abs(angle0[m])))
              noise_rows.append({'case':cname,'axis':axis,'sensor_case':sens,'method':method,'angle_noise_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(dtheta[m]**2)))),'angle_noise_max_abs_deg':float(np.max(np.abs(np.rad2deg(dtheta[m])))),'local_linear_valid_fraction':linear_valid,'linear_reference_rmse_m_s':rmse_ref,'linear_reference_max_abs_m_s':max_ref,'observer_noise_rmse_m_s':rmse_out,'observer_noise_max_abs_m_s':max_out,'linear_reference_p95_abs_m_s':p95_ref,'observer_noise_p95_abs_m_s':p95_out,'p95_amplification_ratio':p95_out/p95_ref if p95_ref>0 else None,'rmse_amplification_ratio':rmse_out/rmse_ref if rmse_ref>0 else None,'max_amplification_ratio':max_out/max_ref if max_ref>0 else None})
          except (ValueError,FloatingPointError,np.linalg.LinAlgError): est=np.full_like(plant['v'],np.nan); met={'rmse_m_s':None,'bias_m_s':None,'mae_m_s':None,'p95_abs_error_m_s':None,'max_abs_error_m_s':None}
          row={'case':cname,'axis':axis,'sensor_case':sens,'method':method,'tuning_parameter':p,**met,'maximum_abs_angle_deg':plant['maxang'],'within_plus_minus_60_deg':plant['maxang']<=60,'offline':method.startswith('RTS')}
          rows.append(row)
          if sens=='nominal': wave[(cname,axis,method)]=(plant,est,row)
          if sens=='static_window_sigma': static_sigma_wave[(cname,axis,method)]=(plant,est,row)
    write_csv(OUT/'ow05_metrics.csv',rows); write_csv(OUT/'ow05_tuning_scan.csv',tuning_rows); write_csv(OUT/'ow05_noise_amplification.csv',noise_rows)
    free_decay_validation=evaluate_free_decay(cfg,tunings,OUT,REPO)
    # Plot sensor-impact comparison for both valid wind models, zoom around global max error for each.
    figs=[]
    # Give ESO and RTS identical wind and error scales for each case and axis.
    # Each method is centered on its own peak-error time, so shared limits are
    # needed to make visual comparisons fair.
    shared_limits={}
    for cname in dict.fromkeys(key[0] for key in wave):
      for axis in cfg['axis_names']:
        pair=[wave[(cname,axis,m)] for m in ('ESO 3状態','RTS 3状態（オフライン）')]
        speed_max=max(float(np.nanmax(np.abs(item[0]['v']))) for item in pair)
        error_max=max(float(np.nanmax(np.abs(item[1][item[0]['mask']]-item[0]['v'][item[0]['mask']]))) for item in pair)
        shared_limits[(cname,axis)]=(max(0.0,speed_max*1.05),max(0.05,error_max*1.10))
    for (cname,axis,method),(p,e,r) in wave.items():
      err=e-p['v']; ids=np.flatnonzero(p['mask']&np.isfinite(err)); i=int(ids[np.argmax(np.abs(err[ids]))]); center=p['t'][i]; view=(p['t']>=center-2)&(p['t']<=center+2)
      fig,(a,b)=plt.subplots(2,1,figsize=(9,5.6),sharex=True,layout='constrained')
      a.plot(p['t'][view],p['v'][view],label='True wind speed',color='#222'); a.plot(p['t'][view],e[view],label='Estimated wind speed (nominal sensor)'); a.set_ylabel('Wind speed [m/s]'); a.set_ylim(0,shared_limits[(cname,axis)][0]); a.set_title(f"{'Independent Kaimal, mean 2 m/s' if 'Kaimal' in cname else 'Gust, 2 to 6 m/s'} / {axis} / {'RTS 3-state (offline)' if method.startswith('RTS') else 'ESO 3-state'}"); a.legend(); a.grid(alpha=.25)
      b.plot(p['t'][view],err[view]); b.axhline(0,color='#555'); b.axvline(center,color='#b34b35',ls='--'); b.set_xlabel('Time [s]'); b.set_ylabel('Estimate - true\n[m/s]'); b.set_ylim(-shared_limits[(cname,axis)][1],shared_limits[(cname,axis)][1]); b.grid(alpha=.25)
      name=f"ow05_{'kaimal' if 'Kaimal' in cname else 'gust'}_{axis}_{'rts' if method.startswith('RTS') else 'eso'}.png"; fig.savefig(OUT/name,dpi=160); plt.close(fig); figs.append(name)
    # Static-window sigma evaluation: both observers on each case/axis, with a shared error scale.
    static_figs=[]
    for cname in dict.fromkeys(key[0] for key in static_sigma_wave):
      for axis in cfg['axis_names']:
        p=static_sigma_wave[(cname,axis,'ESO 3状態')][0]
        eso=static_sigma_wave[(cname,axis,'ESO 3状態')][1]
        rts=static_sigma_wave[(cname,axis,'RTS 3状態（オフライン）')][1]
        m=p['mask']; errors=[eso[m]-p['v'][m],rts[m]-p['v'][m]]
        peak_idx=int(np.argmax(np.maximum(np.abs(errors[0]),np.abs(errors[1]))))
        ids=np.flatnonzero(m); center=p['t'][ids[peak_idx]]
        view=(p['t']>=center-2)&(p['t']<=center+2)
        lim=max(0.05,float(max(np.max(np.abs(x)) for x in errors))*1.1)
        fig,(a,b)=plt.subplots(2,1,figsize=(9,5.8),sharex=True,layout='constrained')
        a.plot(p['t'][view],p['v'][view],color='#222',label='True wind speed')
        a.plot(p['t'][view],eso[view],label='ESO 3-state')
        a.plot(p['t'][view],rts[view],label='RTS 3-state (offline)')
        a.set_ylabel('Wind speed [m/s]'); a.set_title(f"Static-window σ noise / {'Independent Kaimal, mean 2 m/s' if 'Kaimal' in cname else 'Gust, 2 to 6 m/s'} / {axis}")
        a.legend(); a.grid(alpha=.25)
        b.plot(p['t'][view],eso[view]-p['v'][view],label='ESO error')
        b.plot(p['t'][view],rts[view]-p['v'][view],label='RTS error')
        b.axhline(0,color='#555',lw=.8); b.set_ylim(-lim,lim)
        b.set_xlabel('Time [s]'); b.set_ylabel('Estimate - true [m/s]'); b.legend(); b.grid(alpha=.25)
        name=f"ow05_static_sigma_{'kaimal' if 'Kaimal' in cname else 'gust'}_{axis.lower()}.png"
        fig.savefig(OUT/name,dpi=160); plt.close(fig); static_figs.append(name)
    report=['# OW-05 センサー非理想性とオブザーバー選定','','## 目的・結論','',
      'OW-03で角度±60°内だった独立Kaimal乱流（平均2 m/s、TI 20%）と2→6 m/sガストを使い、採用BALLプラントにセンサーの量子化、ノイズ、遅延を含めて比較した。比較対象はオンラインの3状態ESOと未来データを使うオフライン3状態RTS。4状態ESOはOW-03で除外済みのため含めない。', '',
      f"公称センサー条件では、風条件・軸ごとのRMSE最小方式は下表の通り。帯域/プロセス雑音は別seedの平均3 m/s Kaimal訓練波形で方式ごと・軸ごとに一度だけ選定し、検証波形には再調整せず適用した。最大角度は全ケースで±60°内である。","",
      '| 風条件 | 軸 | ESO RMSE | RTS RMSE |','|---|---|---:|---:|']
    for cname,axis in plants:
      vals={r['method']:r for r in rows if r['case']==cname and r['axis']==axis and r['sensor_case']=='nominal'}
      report.append(f"| {cname} | {axis} | {vals['ESO 3状態']['rmse_m_s']:.5f} m/s | {vals['RTS 3状態（オフライン）']['rmse_m_s']:.5f} m/s |")
    report += ['', '## センサー条件', '',f"角度サンプルは{rate:.0f} Hz、MT6701の14 bit（量子化幅{360/2**14:.8f}°）、白色ノイズσ={pars.white_noise_std_deg:.3f}°、有色ノイズσ={pars.coloured_noise_std_deg:.3f}°・時定数{pars.coloured_noise_time_constant_s:.3f} s、固定遅延{pars.fixed_delay_s*1000:.0f} msを公称条件とした。センサー段階評価に従った設定であり、量子化/サンプリング仕様以外のノイズと遅延は実測前の暫定仮定である。感度としてノイズσを2倍にした条件も計算した。", '',
     ]
    report += ['', '## MT6701の計測角出力誤差と静止窓内の摂動','',
      '今回の目的は、センサー単体の誤差を同定することではなく、静止と判定した窓で記録角が窓平均からどれだけ変動したかを把握することである。各窓の平均だけを差し引き、摂動 r_i=y_i-mean(y_window) をそのまま評価した。二階差分は使わない。静止判定条件は、符号付き角度35°超、0.5秒窓の標準偏差0.20°未満、線形ドリフト1°/s未満である。', '',
      '静止窓内の摂動標準偏差は、全選定窓の r_i を結合して算出した。これは記録角に現れた窓内変動であり、真角度基準がないため、実際の微小運動とセンサー出力変動を分離できない。センサー出力誤差 e_s=θ_meas-θ_true やセンサー単体ノイズとは区別する。', '',
      '| 軸 | 静止窓数 | 摂動点数 | 摂動σ [°] | ±3σ [°] | ±3σ内 [%] | 正規性検定 p値 | ラグ1自己相関 |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for axis,v in measured_noise.items():
      report.append(f"| {axis} | {v['n_windows']} | {v['n_samples']} | {v['static_perturbation_sigma_deg']:.5f} | ±{v['three_sigma_deg']:.5f} | {100*v['fraction_within_three_sigma']:.1f} | {v['normality_p_value']:.2e} | {v['residual_acf_lag1']:.3f} |")
    report += ['', '上段の青線は各静止窓の平均値からの実測摂動、緑帯は同じ摂動列から計算した±3σである。中央のヒストグラムと緑の破線も同じ摂動列に対する表示である。図の時系列・分布・包含率の対象は一致している。', '',
      '±3σは絶対的な上下限ではない。正規分布なら約99.73%が範囲内に入るが、今回の摂動は正規性検定で棄却された（両軸 p<0.001）。観測包含率はIN {:.1f}%、OUT {:.1f}%である。ラグ1自己相関も高く、独立白色ノイズとはみなせない。'.format(100*measured_noise['IN']['fraction_within_three_sigma'],100*measured_noise['OUT']['fraction_within_three_sigma']), '',
      '参考として、従来の二階差分方式で得た高周波変動相当σはIN 0.00484°、OUT 0.00554°だった。今回の静止窓摂動σはそれぞれ約{:.1f}倍、{:.1f}倍であり、二階差分が窓内変動全体を表していなかったことが分かる。この倍率はセンサー誤差の過小評価率ではない。今回の摂動σにも治具などの微小運動が含まれ得る。'.format(measured_noise['IN']['static_perturbation_sigma_deg']/0.00484014296,measured_noise['OUT']['static_perturbation_sigma_deg']/0.00554022955), '',
      'MT6701の14 bit角度量子化幅は{:.5f}°である。今回の摂動σとの比較は記録変動の大きさを示すだけで、量子化誤差やセンサー出力誤差を分離するものではない。センサー誤差の幅やリニアリティを求めるには、十分な精度の独立した角度基準との同時測定が必要である。'.format(360/2**14), '']
    for axis in ('IN','OUT'):
      report += [f"![{axis} axis: static-window perturbation, histogram and autocorrelation](ow05_measured_noise_{axis.lower()}.png)",'']
    report += ['詳細値は `ow05_measured_noise_summary.csv`、使用した静止区間は `ow05_measured_noise_windows.csv` に保存した。', '',
      '## 今回特定できた誤差範囲と今後の課題','',
      'センサーを真角度 $\\\\theta_{\\\\mathrm{true}}$ から計測角 $\\\\theta_{\\\\mathrm{meas}}$ への変換系と定義すると、評価対象は $e_s(\\\\theta,t)=\\\\theta_{\\\\mathrm{meas}}(\\\\theta,t)-\\\\theta_{\\\\mathrm{true}}(t)$ である。誤差要因は次のように整理できる。','',
      '| 誤差要因 | 内容 | 今回の特定状況 |','|---|---|---|',
      '| 分解能・量子化 | 有限の角度コード幅とコード化誤差 | OW-05モデルで14 bit、1 LSB=0.02197266°として設定。これは刻み幅であり、真角度との誤差や全体精度の上限ではない |',
      '| オフセット・ゼロ点 | 計測角の基準ずれ | 真角度基準がなく未特定 |',
      '| ゲイン・リニアリティ | 角度に応じた傾き誤差、非直線性 | 未特定。今回のσや±3σからは評価できない |',
      f"| 繰返し性・時間変動 | 同じ真角度での出力ばらつき、電子的な揺れ | 真角度を固定した基準測定がなく、センサー分は分離できていない。今回算出したIN σ={measured_noise['IN']['static_perturbation_sigma_deg']:.5f}°、OUT σ={measured_noise['OUT']['static_perturbation_sigma_deg']:.5f}°は、静止窓の平均からの摂動標準偏差 |",
      '| 動的応答・タイミング | 内部フィルタ、遅れ、サンプリング時刻の揺れ | OW-05では100 Hz、固定遅延10 ms等をモデル仮定として設定。実機の遅延・帯域・ジッタは未測定 |',
      '| 取付け・磁気条件 | 芯ずれ、傾き、磁石配置や磁場状態による角度依存誤差 | 実機条件での角度基準比較がなく未特定 |','',
      f'今回数値としてまとめたのは、モデルに設定した14 bitのコード刻み幅と、静止窓で計測角が各窓平均から変動した標準偏差（IN {measured_noise["IN"]["static_perturbation_sigma_deg"]:.5f}°、OUT {measured_noise["OUT"]["static_perturbation_sigma_deg"]:.5f}°）です。センサー出力誤差 $e_s$ の最大幅、リニアリティ誤差、センサー単体ノイズは特定できていません。図の±3σは静止窓摂動の範囲を要約し、センサー出力誤差の範囲を示すものではありません。','',
      '### 今後の試験メニュー','','1. **独立角度基準との同時測定**  ','   MT6701の使用時と同じ磁石・取付け・配線・読み出し処理を用い、精度が既知の角度基準と同期して記録する。基準器の誤差は、評価したいセンサー誤差より十分小さくする。生の読み値、変換後の角度、時刻も保存する。','','2. **使用角度範囲の静的な往復掃引**  ','   角度を複数点で止めて安定後に記録し、正方向・逆方向に複数回掃引する。基準との差から、ゼロ点オフセット、ゲイン誤差、リニアリティを求める。リニアリティは採用した基準直線を明示し、直線からの最大偏差と偏差幅で示す。往復差も角度ごとに評価する。','','3. **固定角度での繰返し測定**  ','   使用範囲内の複数角度で真角度を固定し、同じ条件で繰り返し記録する。センサー出力の繰返し性、時間変動、高周波ノイズを評価する。量子化の影響を見落とさないよう、一つの角度だけでなく複数角度で測る。','','4. **既知の角度運動による動的応答試験**  ','   基準角を同時記録しながら、複数の速度・周波数で角度を動かす。固定遅延、帯域、振幅誤差、サンプリング時刻の揺れを、静的な角度誤差と分けて確認する。','','5. **使用環境の影響確認**  ','   必要に応じて温度、電源、磁石の位置・取付け状態を変えて、角度誤差と再現性への影響を測る。まず実使用条件を優先する。','','6. **誤差幅の集計**  ','   真角度との差の角度依存曲線、最大絶対誤差、リニアリティ、往復差、繰返し性、時間変動、動的遅れを分けて報告する。ヒストグラム・自己相関も確認し、正規性と白色性が妥当な場合に限って±3σをノイズ幅の目安とする。','','まずは **1の基準器との同時測定**と **2の静的往復掃引**を優先する。今回の記録角変動相当σおよび±3σは、これらの試験で真角度基準との差を測定するまで、センサー出力誤差の幅として扱わない。','',
      '## 角度ノイズから風速への増幅倍率','', r'線形基準は、ノイズなしセンサー角度θ₀の近傍における静的な角度→風速写像の傾きで求めた。各時点の角度摂動Δθを、局所傾き g(θ₀)=dV/dθ（θ=θ₀で評価）に掛け、線形基準風速摂動 $v_{\mathrm{lin}}(t)=g(\theta_0(t))\Delta\theta(t)$ とした。角度摂動は、遅延・ゲイン・オフセット・量子化・ジッタを保って乱数ノイズだけを0にしたセンサー出力との差である。局所線形近似の妥当性確認として $\lvert\Delta\theta\rvert \leq 0.1\lvert\theta_0\rvert$ を満たす評価点の割合も記載した。これを満たさない時点では、線形基準倍率を慎重に解釈する。観測器側は、同一条件におけるノイズあり/なし推定出力差 $\Delta\hat{V}$ を全評価時点（15秒以降）で比較した。RMSE倍率は $\mathrm{RMS}(\Delta\hat{V})/\mathrm{RMS}(v_{\mathrm{lin}})$、最大値倍率は $\max_t \lvert\Delta\hat{V}(t)\rvert/\max_t \lvert v_{\mathrm{lin}}(t)\rvert$ である。', '']
    for cname,axis in plants:
      subset=[nr for nr in noise_rows if nr['case']==cname and nr['axis']==axis]
      report += ['', f'### {cname}・{axis}軸', '', '| ノイズ条件 | 推定器 | 角度ノイズRMSE [deg] | 線形条件成立 [%] | 基準RMSE [m/s] | 出力RMSE [m/s] | RMSE倍率 | 95%値倍率 |','|---|---|---:|---:|---:|---:|---:|---:|']
      for nr in subset:
        report.append(f"| {'公称' if nr['sensor_case']=='nominal' else '2倍'} | {nr['method']} | {nr['angle_noise_rmse_deg']:.5f} | {100*nr['local_linear_valid_fraction']:.1f} | {nr['linear_reference_rmse_m_s']:.6f} | {nr['observer_noise_rmse_m_s']:.6f} | {nr['rmse_amplification_ratio']:.3f} | {nr['p95_amplification_ratio']:.3f} |")
      report += ['', '| ノイズ条件 | 推定器 | 基準最大値 [m/s] | 出力最大値 [m/s] | 最大値倍率 |','|---|---|---:|---:|---:|']
      for nr in subset:
        report.append(f"| {'公称' if nr['sensor_case']=='nominal' else '2倍'} | {nr['method']} | {nr['linear_reference_max_abs_m_s']:.6f} | {nr['observer_noise_max_abs_m_s']:.6f} | {nr['max_amplification_ratio']:.3f} |")
    report += ['', '増幅倍率は全点RMSE比を主指標、絶対値95パーセンタイル比を外れ値に頑健な補助指標、最大絶対値比をピーク影響の補助指標として併記した。最大値倍率は単一点に強く左右されるため慎重に読む。線形基準は準静的な写像であり、動的な真値風速誤差の代用ではない。観測器の履歴依存性はノイズあり/なし出力差側に反映される。全値は `ow05_noise_amplification.csv` に保存した。', '', '## 静止窓摂動σを使った追加オブザーバー評価','',
      '静止窓の平均からの摂動標準偏差を、OW-05センサーモデルの白色ノイズσとして置いた追加条件である（IN σ={:.5f}°、OUT σ={:.5f}°、±3σ幅はそれぞれ±{:.5f}°、±{:.5f}°）。モデルのσ欄には標準偏差を入力し、±3σを標準偏差として入力していない。有色ノイズは0にし、静止窓σを総ノイズ標準偏差の白色近似として評価した。'.format(measured_sigma['IN'],measured_sigma['OUT'],3*measured_sigma['IN'],3*measured_sigma['OUT']),
      '静止窓で観測した変動にはセンサー以外の微小運動も含まれ得る。実測摂動は時間相関（ラグ1自己相関約0.92）を持つ一方、今回の追加評価では同じ標準偏差の白色ノイズに近似しており、相関構造までは再現しない。ESO/RTSの調整値は公称条件で決めたものを固定し、追加条件に再調整せず適用した。','',
      '| 風条件 | 軸 | ノイズσ [°] | 推定器 | RMSE [m/s] | 公称比 | MAE [m/s] | 95%絶対誤差 [m/s] | 最大絶対誤差 [m/s] |','|---|---|---:|---|---:|---:|---:|---:|---:|']
    for cname,axis in plants:
      for method in ('ESO 3状態','RTS 3状態（オフライン）'):
        rr=next(r for r in rows if r['case']==cname and r['axis']==axis and r['sensor_case']=='static_window_sigma' and r['method']==method)
        nominal_rmse=next(r['rmse_m_s'] for r in rows if r['case']==cname and r['axis']==axis and r['method']==method and r['sensor_case']=='nominal')
        rmse_ratio=rr['rmse_m_s']/nominal_rmse
        report.append(f"| {cname} | {axis} | {measured_sigma[axis]:.5f} | {method} | {rr['rmse_m_s']:.5f} | {rmse_ratio:.2f}× | {rr['mae_m_s']:.5f} | {rr['p95_abs_error_m_s']:.5f} | {rr['max_abs_error_m_s']:.5f} |")
    eso_ratios=[r['rmse_m_s']/next(n['rmse_m_s'] for n in rows if n['case']==r['case'] and n['axis']==r['axis'] and n['method']==r['method'] and n['sensor_case']=='nominal') for r in rows if r['sensor_case']=='static_window_sigma' and r['method']=='ESO 3状態']
    rts_ratios=[r['rmse_m_s']/next(n['rmse_m_s'] for n in rows if n['case']==r['case'] and n['axis']==r['axis'] and n['method']==r['method'] and n['sensor_case']=='nominal') for r in rows if r['sensor_case']=='static_window_sigma' and r['method']=='RTS 3状態（オフライン）']
    report += ['', f"公称条件とのRMSE比は、ESOで{min(eso_ratios):.1f}〜{max(eso_ratios):.1f}倍、RTSで{min(rts_ratios):.1f}〜{max(rts_ratios):.1f}倍だった。今回設定した白色近似では両方式とも誤差が増え、RTSのRMSEはESOより小さい。ただしRTSは未来データを使うオフライン評価であり、実測摂動の時間相関も再現していないため、この順位や倍率を実機の相関ノイズ条件へ直接一般化できない。", '', '追加条件の風速推定と誤差。各図は両推定器のピーク誤差を含む時刻を中心に±2秒表示し、誤差軸は共通。', '']
    for name in static_figs: report += [f'![静止窓摂動σによるオブザーバー比較]({name})','']
    report += ['', '## 自由振動の実測波形によるゼロ風速妥当性確認','','自由振動時は外部風力を0と置き、承認済みBALL条件の実測角度をそのまま観測器へ入力した。IN/OUT各6波形、計12波形を評価した。初期角は実測先頭値、初期角速度は先頭0.2秒の直線近似で設定し、初期外力を0とした。先頭3秒と末尾0.5秒を採点から除外した。測定角と同じ初期条件でOW-05の自由減衰モデルを外力0で積分した基準波形との差を角度残差とし、実測入力と基準入力で得た推定風速の差を、実測残差に対する観測器応答とした。','',
      '実測角度とゼロ風速モデルの残差にはセンサーノイズだけでなく、減衰係数・摩擦・バックラッシュ等のモデル差や微小外乱も含まれる。したがって、残差RMSをセンサー単体ノイズ、実測応答倍率をセンサー単体の倍率とは断定しない。残差の自己相関も表に併記した。モデル予測波形を観測器に入力した場合の推定風速は0 m/sとなり、表のゼロ風速RMSEは実測波形でのみ発生する。白色モデル出力RMSEは、静止窓σを白色ノイズとしてOW-05シミュレーションに加えたときの推定器出力差を2風条件で合算した値である。','',
      '| 軸 | 観測器 | n波形 | ゼロ風速RMSE [m/s] | ゼロ風速Bias [m/s] | 白色応答RMSE [m/s] | 実測/白色 | 角度残差RMS [°] | 残差ACF lag1 | 実測倍率 [m/s/°] | 白色倍率 [m/s/°] | 倍率比 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for fr in free_decay_validation['summary']:
      report.append(f"| {fr['axis']} | {fr['method']} | {fr['segments']} | {fr['wind_zero_rmse_m_s']:.5f} | {fr['wind_zero_bias_m_s']:.5f} | {fr['white_sigma_observer_response_rmse_m_s']:.5f} | {fr['zero_wind_to_white_response_ratio']:.2f}× | {fr['measured_angle_residual_rmse_deg']:.5f} | {fr['measured_angle_residual_acf_lag1']:.3f} | {fr['measured_amplification_m_s_per_deg']:.5f} | {fr['white_sigma_amplification_m_s_per_deg']:.5f} | {fr['measured_to_white_gain_ratio']:.3f} |")
    zero_ratios=[fr['zero_wind_to_white_response_ratio'] for fr in free_decay_validation['summary']]
    gain_ratios=[fr['measured_to_white_gain_ratio'] for fr in free_decay_validation['summary']]
    residual_acfs=[fr['measured_angle_residual_acf_lag1'] for fr in free_decay_validation['summary']]
    residual_rmse=[fr['measured_angle_residual_rmse_deg'] for fr in free_decay_validation['summary']]
    report += ['', f"妥当性確認の結果、実測自由振動を0風速として処理した推定RMSEは{min(fr['wind_zero_rmse_m_s'] for fr in free_decay_validation['summary']):.2f}〜{max(fr['wind_zero_rmse_m_s'] for fr in free_decay_validation['summary']):.2f} m/sだった。白色ノイズモデルの応答RMSEとの比は{min(zero_ratios):.2f}〜{max(zero_ratios):.2f}倍、残差で正規化した応答倍率比は{min(gain_ratios):.3f}〜{max(gain_ratios):.3f}だった。値は大きく異なり、静止窓σを独立白色ノイズとする仮定は実測自由振動に対する応答を再現していない。", f"ただし角度残差RMSは{min(residual_rmse):.2f}〜{max(residual_rmse):.2f}°、ラグ1自己相関は{min(residual_acfs):.3f}〜{max(residual_acfs):.3f}で、静止窓σ（IN {measured_sigma['IN']:.3f}°、OUT {measured_sigma['OUT']:.3f}°）より大きく、強く時間相関したモデル残差である。これをセンサーノイズと同一視できないため、差の全てをノイズモデルの誤りとは断定できない。一方、ゼロ風速でも推定風速が残るので、OW-05のノイズ倍率だけでは実機誤差を説明できず、b=0を含む機械モデル差と実測ノイズの時間構造を分けて再評価する必要がある。",'']
    report += ['', '上段は実測角度と同初期条件の無風モデル、下段は観測器の推定風速と真値0 m/s。角度残差にはモデル誤差も含まれる。倍率図は実測残差に対する応答を白色近似の応答ゲインと、追加の棒グラフはゼロ風速RMSEを白色ノイズ時の応答RMSEと比較する。実測/白色の差はノイズ分布・時間相関とモデル残差の差を含むため、単独でモデル誤りの証明とはしない。','',
      '### ESOゲイン感度（診断）','',
      '選定済み12 Hzを含む0.5〜12 HzでESO極周波数を振り、同じ実測自由振動・初期化・採点区間でゼロ風速RMSEを比較した。これはゲイン過大の診断であり、自由振動検証波形を最終ゲイン選定に流用しない。低ゲインでゼロ風速誤差が改善すれば、現行ゲインが実測摂動に過敏な可能性が高まる。ただし、自由振動のモデル残差も入力に含むため、それだけで原因をセンサーノイズと断定はできない。','',
      '| 軸 | ESO極 [Hz] | ゼロ風速RMSE [m/s] | 収束波形/対象波形 |','|---|---:|---:|---:|']
    for sr in free_decay_validation['gain_sensitivity']:
      rmse_text = f"{sr['wind_zero_rmse_m_s']:.5f}" if sr['wind_zero_rmse_m_s'] is not None else "発散（採点不可）"
      count_text = f"{sr['segments']}/{sr['total_segments']}"
      report.append(f"| {sr['axis']} | {sr['pole_hz']:.1f} | {rmse_text} | {count_text} |")
    report += ['', f"![ESOゲイン感度の自由振動診断](ow05_free_decay_gain_sensitivity.png)",'',
      '波形別感度値は `ow05_free_decay_gain_sensitivity.csv`、軸別集計は `ow05_free_decay_gain_sensitivity_summary.csv` に保存した。','',
      f"![実測自由振動の角度・ゼロ風速推定](ow05_free_decay_zero_wind.png)",'',
      f"![自由振動実測応答と白色ノイズ倍率](ow05_free_decay_gain_comparison.png)",'',
      f"![ゼロ風速RMSEと白色ノイズ応答RMSE](ow05_free_decay_zero_vs_white.png)",'',
      '波形別値は ow05_free_decay_metrics.csv、軸・観測器別集計は ow05_free_decay_summary.csv に保存した。','','## 公称センサー条件の拡大時系列','', '各図は最大誤差時刻を中心に±2秒を表示する。上段は真値と推定風速、下段は誤差。同じ風条件・軸のESO図とRTS図では、上段と下段それぞれの縦軸範囲を共通にして比較できるようにした。描画環境に日本語フォントがないため、図中ラベルは英語表記とし、本文と図題は日本語で記載する。', '']
    for name in figs: report += [f'![OW-05 拡大時系列]({name})','']
    report += ['## 解釈と選定','', 'RTS（Rauch–Tung–Striebel）スムーザーは、まず時系列を前向きに推定し、その後、将来の観測も使って過去の状態推定を後向きに修正する。時間的に独立なホワイトノイズによる一時的な観測の揺れが、運動モデルや前後の観測と整合しない場合、その揺れを実際の状態変化ではなく観測ノイズとして扱いやすくなり、推定への影響を弱められる。これは未来の観測が過去の観測ノイズを物理的に打ち消すという意味ではなく、全時系列に最も整合する状態系列を再推定する効果である。', '', 'この平滑化は、運動モデルが十分妥当で、観測ノイズとモデル誤差の大きさ（観測・プロセス雑音の共分散）が適切に設定されていることを前提とする。ノイズが時間相関を持つ場合、その影響は独立な白色ノイズほど平均化されない。また、実際の急な風速変化をモデルが説明できないときに平滑化が強すぎると、真の変化まで抑えたり、遅らせたりする可能性がある。したがって、オフラインRTSは必ずノイズに強いわけではなく、今回の低い誤差をホワイトノイズ単独の効果と断定することもできない。今回のセンサー条件には白色ノイズに加えて有色ノイズも含まれるため、成分ごとの寄与を分けるには追加の比較が必要である。', '', 'RTS 3状態は将来データを使うオフライン評価であるため、RMSEが小さくてもオンライン実装候補とは分ける。オンライン用途はESO 3状態を候補とし、公称ノイズ条件に対する帯域選定を反映した値を使用する。RTSはログ解析や遅延許容用途の基準として残す。ノイズ2倍感度を含む推定誤差は `ow05_metrics.csv`、増幅倍率は `ow05_noise_amplification.csv`、調整スキャンは `ow05_tuning_scan.csv` に保存した。', '', '## 再実行','', '```bash','python 06_Analysis/simulation/src/run_ow05_sensor_nonideality.py','```','']
    (OUT/'OW-05_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    summary={'task_id':'OW-05','sensor_nominal':nominal,'free_decay_validation':free_decay_validation,'static_window_sigma_condition':{'white_noise_std_deg_by_axis':measured_sigma,'coloured_noise_std_deg':0.0,'interpretation':'White-noise equivalent of static-window mean-centered perturbation sigma; temporal correlation and possible mechanical micro-motion are not represented.','observer_tuning':'Fixed values selected under nominal sensor noise.'},'measured_sensor_noise':{a:{k:v for k,v in d.items() if k not in ('acf','windows','perturbations')} for a,d in measured_noise.items()},'noise_amplification':noise_rows,'tuning_by_axis_method':{f'{a}|{m}':v for (a,m),v in tunings.items()},'results':rows}
    (OUT/'ow05_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('rows',len(rows),'tunings',tunings)

if __name__=='__main__': run()
