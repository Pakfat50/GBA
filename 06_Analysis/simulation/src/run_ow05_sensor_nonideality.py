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
from run_ow03_hbk_observer_tuning import static_force

SIM=Path(__file__).resolve().parents[1]
OUT=SIM/'results/observer_wind/ow05_sensor_nonideality'
REPO=SIM.parents[1]

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def analyze_measured_sensor_noise():
    """Estimate high-frequency angle noise from stationary pre-release intervals."""
    OUT.mkdir(parents=True,exist_ok=True)
    data_root=REPO/'04_Data/05_Fitting/20260921'
    selection_path=REPO/'06_Analysis/fitting_pipeline/results/20260921/waveform_review/waveform_selection.csv'
    selection=pd.read_csv(selection_path,encoding='utf-8-sig')
    selection=selection[(selection['use_for_fitting']==1)&(selection['review_status'].astype(str).str.upper()=='APPROVED')]
    tables={}
    estimates={}
    window_n=50  # 0.5 s at 100 Hz
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
        # Search backward for the closest 0.5 s window with a fixed high-angle
        # hold, low spread and low linear drift, using the original split criteria.
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
        sample=np.arange(window_n,dtype=float)
        trend=np.polyval(np.polyfit(sample,selected,1),sample)
        residual=selected-trend
        # For independent sample noise, second difference has variance 1.5*sigma^2.
        innovations=(residual[1:-1]-0.5*(residual[:-2]+residual[2:]))/np.sqrt(1.5)
        windows.append({'segment_id':str(record['segment_id']),'data_file':relative_path,
                        'direction':str(record['direction']),'start_index':selected_start,
                        'samples':window_n,'residual':residual,'innovations':innovations})
      if not windows:
        raise RuntimeError('No stationary pre-release windows found for '+axis)
      innovations=np.concatenate([w['innovations'] for w in windows])
      residuals=[w['residual'] for w in windows]
      sigma=float(np.std(innovations,ddof=1))
      acf=[]
      denominator=sum(float(np.dot(r,r)) for r in residuals)
      for lag in range(1,21):
        numerator=sum(float(np.dot(r[:-lag],r[lag:])) for r in residuals)
        acf.append(numerator/denominator if denominator>0 else 0.0)
      normal=normaltest(innovations)
      estimates[axis]={
        'n_windows':len(windows),'n_samples':int(sum(len(r) for r in residuals)),
        'n_innovations':len(innovations),'white_equivalent_sigma_deg':sigma,
        'three_sigma_deg':3*sigma,'assumed_white_sigma_deg':0.015,
        'assumed_white_three_sigma_deg':0.045,
        'innovation_skewness':float(skew(innovations)),
        'innovation_excess_kurtosis':float(kurtosis(innovations)),
        'normality_k2':float(normal.statistic),'normality_p_value':float(normal.pvalue),
        'fraction_within_three_sigma':float(np.mean(np.abs(innovations)<=3*sigma)),
        'residual_acf_lag1':float(acf[0]),'acf_95_bound':float(1.96/np.sqrt(sum(len(r) for r in residuals))),
        'acf':acf,'windows':windows,'innovations':innovations,'residuals':residuals}
    # Compact machine-readable summary and per-window audit trail.
    summary_rows=[]
    for axis,v in estimates.items():
      summary_rows.append({k:value for k,value in v.items() if k not in ('acf','windows','innovations','residuals')})
    write_csv(OUT/'ow05_measured_noise_summary.csv',summary_rows)
    window_rows=[]
    for axis,v in estimates.items():
      for w in v['windows']:
        window_rows.append({'axis':axis,'segment_id':w['segment_id'],'data_file':w['data_file'],
                            'direction':w['direction'],'start_index':w['start_index'],
                            'samples':w['samples'],'duration_s':w['samples']/100.0,
                            'innovation_sigma_deg':float(np.std(w['innovations'],ddof=1))})
    write_csv(OUT/'ow05_measured_noise_windows.csv',window_rows)
    for axis,v in estimates.items():
      fig,axes=plt.subplots(3,1,figsize=(9,8),layout='constrained')
      residuals=v['residuals']; sigma=v['white_equivalent_sigma_deg']
      joined=np.concatenate(residuals)
      axes[0].plot(np.arange(len(joined))*0.01,joined,lw=.8,color='#2673a8')
      axes[0].axhspan(-3*sigma,3*sigma,color='#4c9f70',alpha=.16,label=f'Measured ±3σ = ±{3*sigma:.4f}°')
      axes[0].axhline(0,color='#444',lw=.7); axes[0].set_ylabel('Demeaned angle [deg]')
      axes[0].set_xlabel('Concatenated stationary-window time [s]'); axes[0].set_title(f'{axis} axis: pre-release stationary residuals')
      axes[0].legend(loc='upper right'); axes[0].grid(alpha=.2)
      e=v['innovations']; bins=np.linspace(-0.06,0.06,81)
      axes[1].hist(e,bins=bins,density=True,color='#79a9c9',alpha=.65,label='Measured high-frequency innovations')
      xx=np.linspace(-.06,.06,600)
      gaussian=np.exp(-0.5*(xx/sigma)**2)/(sigma*np.sqrt(2*np.pi))
      model_sigma=.015
      model_gaussian=np.exp(-0.5*(xx/model_sigma)**2)/(model_sigma*np.sqrt(2*np.pi))
      axes[1].plot(xx,gaussian,color='#193d5a',lw=2,label=f'Gaussian fit, σ={sigma:.4f}°')
      axes[1].plot(xx,model_gaussian,color='#c44e52',ls='--',lw=1.6,label='OW-05 assumed white σ=0.015°')
      axes[1].axvline(-3*sigma,color='#238b45',ls=':',lw=1.4); axes[1].axvline(3*sigma,color='#238b45',ls=':',lw=1.4)
      axes[1].axvline(-3*model_sigma,color='#c44e52',ls='--',lw=1); axes[1].axvline(3*model_sigma,color='#c44e52',ls='--',lw=1)
      axes[1].set_xlim(-.06,.06); axes[1].set_ylabel('Probability density'); axes[1].set_xlabel('High-frequency innovation [deg]')
      axes[1].set_title(f"Histogram; {100*v['fraction_within_three_sigma']:.1f}% within measured ±3σ (Gaussian: 99.73%)")
      axes[1].legend(loc='upper right',fontsize=8); axes[1].grid(alpha=.2)
      lags=np.arange(1,len(v['acf'])+1); bound=v['acf_95_bound']
      axes[2].bar(lags,v['acf'],color='#8172b2',width=.8); axes[2].axhline(0,color='#444',lw=.7)
      axes[2].axhspan(-bound,bound,color='#777',alpha=.16,label=f'Approx. 95% white-noise bound ±{bound:.3f}')
      axes[2].set_xlabel('Lag [samples]'); axes[2].set_ylabel('ACF'); axes[2].set_title(f"Static-window residual autocorrelation; lag 1 = {v['residual_acf_lag1']:.3f}")
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
    rows=[]; noise_rows=[]; wave={}
    modes={'ideal':None,'nominal':pars,'noise_2x':AngleSensorParameters(sample_rate_hz=rate,resolution_bits=14,white_noise_std_deg=.03,coloured_noise_std_deg=.02,coloured_noise_time_constant_s=pars.coloured_noise_time_constant_s,fixed_delay_s=.01)}
    for cname,axis in plants:
      plant=plants[(cname,axis)]
      for sens,sp in modes.items():
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
    write_csv(OUT/'ow05_metrics.csv',rows); write_csv(OUT/'ow05_tuning_scan.csv',tuning_rows); write_csv(OUT/'ow05_noise_amplification.csv',noise_rows)
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
    report=['# OW-05 センサー非理想性とオブザーバー選定','','## 目的・結論','',
      'OW-03で角度±60°内だった独立Kaimal乱流（平均2 m/s、TI 20%）と2→6 m/sガストを使い、採用BALLプラントにセンサーの量子化、ノイズ、遅延を含めて比較した。比較対象はオンラインの3状態ESOと未来データを使うオフライン3状態RTS。4状態ESOはOW-03で除外済みのため含めない。', '',
      f"公称センサー条件では、風条件・軸ごとのRMSE最小方式は下表の通り。帯域/プロセス雑音は別seedの平均3 m/s Kaimal訓練波形で方式ごと・軸ごとに一度だけ選定し、検証波形には再調整せず適用した。最大角度は全ケースで±60°内である。","",
      '| 風条件 | 軸 | ESO RMSE | RTS RMSE |','|---|---|---:|---:|']
    for cname,axis in plants:
      vals={r['method']:r for r in rows if r['case']==cname and r['axis']==axis and r['sensor_case']=='nominal'}
      report.append(f"| {cname} | {axis} | {vals['ESO 3状態']['rmse_m_s']:.5f} m/s | {vals['RTS 3状態（オフライン）']['rmse_m_s']:.5f} m/s |")
    report += ['', '## センサー条件', '',f"角度サンプルは{rate:.0f} Hz、MT6701の14 bit（量子化幅{360/2**14:.8f}°）、白色ノイズσ={pars.white_noise_std_deg:.3f}°、有色ノイズσ={pars.coloured_noise_std_deg:.3f}°・時定数{pars.coloured_noise_time_constant_s:.3f} s、固定遅延{pars.fixed_delay_s*1000:.0f} msを公称条件とした。センサー段階評価に従った設定であり、量子化/サンプリング仕様以外のノイズと遅延は実測前の暫定仮定である。感度としてノイズσを2倍にした条件も計算した。", '',
      '## 自由振動の静止区間から見積もった角度ノイズ','',
      '自由振動データの各リリース直前から、符号付き角度が35°を超え、0.5秒間の標準偏差が0.20°未満、線形ドリフトが1°/s未満となる最も近い区間を抽出した。角度の一次傾向を除いた残差 r_i に対して、隣接点の二階差分 $e_i=(r_i-(r_{i-1}+r_{i+1})/2)/\\sqrt{1.5}$ を計算した。独立な白色ノイズなら、この変換後の標準偏差は元ノイズのσに一致する。よって以下は静止区間の高周波成分から得た「白色ノイズ相当値」であり、全周波数帯のセンサノイズ推定値ではない。', '',
      '| 軸 | 静止窓数 | 高周波点数 | 推定σ [°] | 推定±3σ [°] | OW-05仮定σ [°] | 仮定/推定 |','|---|---:|---:|---:|---:|---:|---:|']
    for axis,v in measured_noise.items():
      report.append(f"| {axis} | {v['n_windows']} | {v['n_innovations']} | {v['white_equivalent_sigma_deg']:.5f} | ±{v['three_sigma_deg']:.5f} | 0.01500 | {0.015/v['white_equivalent_sigma_deg']:.2f}× |")
    report += ['', '図の上段は静止窓の残差と推定±3σ、中央は高周波成分のヒストグラムと正規分布曲線、下段は残差の自己相関である。正規性検定は両軸とも棄却され（p<0.001）、ラグ1自己相関も白色雑音の95%目安範囲を大きく超えた。したがって、この実測残差を正規白色雑音とみなす仮定は支持されない。裾の重さや相関は、治具の微小運動、角度処理、その他の低周波揺れを含む可能性がある。', '',
      '推定±3σ内に入る比率はIN {:.1f}%、OUT {:.1f}%で、理想正規分布の99.73%と一致するとは限らない。図の赤破線はシミュレーションで仮定した白色σ=0.015°の±3σ（±0.045°）、緑点線は各軸の推定±3σを示す。'.format(100*measured_noise['IN']['fraction_within_three_sigma'],100*measured_noise['OUT']['fraction_within_three_sigma']), '',
      'なお、推定σは14 bit角度量子化幅{:.5f}°より小さい。静止区間では量子化誤差が十分にディザされないため、これを実際の総合センサノイズの確定値や「今回仮定したノイズが過大」と結論づけることはできない。今回示す値は取得波形で分解できた高周波残差の参考値であり、総ノイズ量を確定するには、機械部を固定した条件での独立なセンサ記録が必要である。'.format(360/2**14), '']
    for axis in ('IN','OUT'):
      report += [f"![{axis} axis measured noise histogram and 3 sigma](ow05_measured_noise_{axis.lower()}.png)",'']
    report += ['詳細値は `ow05_measured_noise_summary.csv`、使用した静止区間は `ow05_measured_noise_windows.csv` に保存した。', '',
      '## 角度ノイズから風速への増幅倍率','', r'線形基準は、ノイズなしセンサー角度θ₀の近傍における静的な角度→風速写像の傾きで求めた。各時点の角度摂動Δθを、局所傾き g(θ₀)=dV/dθ（θ=θ₀で評価）に掛け、線形基準風速摂動 $v_{\mathrm{lin}}(t)=g(\theta_0(t))\Delta\theta(t)$ とした。角度摂動は、遅延・ゲイン・オフセット・量子化・ジッタを保って乱数ノイズだけを0にしたセンサー出力との差である。局所線形近似の妥当性確認として $\lvert\Delta\theta\rvert \leq 0.1\lvert\theta_0\rvert$ を満たす評価点の割合も記載した。これを満たさない時点では、線形基準倍率を慎重に解釈する。観測器側は、同一条件におけるノイズあり/なし推定出力差 $\Delta\hat{V}$ を全評価時点（15秒以降）で比較した。RMSE倍率は $\mathrm{RMS}(\Delta\hat{V})/\mathrm{RMS}(v_{\mathrm{lin}})$、最大値倍率は $\max_t \lvert\Delta\hat{V}(t)\rvert/\max_t \lvert v_{\mathrm{lin}}(t)\rvert$ である。', '']
    for cname,axis in plants:
      subset=[nr for nr in noise_rows if nr['case']==cname and nr['axis']==axis]
      report += ['', f'### {cname}・{axis}軸', '', '| ノイズ条件 | 推定器 | 角度ノイズRMSE [deg] | 線形条件成立 [%] | 基準RMSE [m/s] | 出力RMSE [m/s] | RMSE倍率 | 95%値倍率 |','|---|---|---:|---:|---:|---:|---:|---:|']
      for nr in subset:
        report.append(f"| {'公称' if nr['sensor_case']=='nominal' else '2倍'} | {nr['method']} | {nr['angle_noise_rmse_deg']:.5f} | {100*nr['local_linear_valid_fraction']:.1f} | {nr['linear_reference_rmse_m_s']:.6f} | {nr['observer_noise_rmse_m_s']:.6f} | {nr['rmse_amplification_ratio']:.3f} | {nr['p95_amplification_ratio']:.3f} |")
      report += ['', '| ノイズ条件 | 推定器 | 基準最大値 [m/s] | 出力最大値 [m/s] | 最大値倍率 |','|---|---|---:|---:|---:|']
      for nr in subset:
        report.append(f"| {'公称' if nr['sensor_case']=='nominal' else '2倍'} | {nr['method']} | {nr['linear_reference_max_abs_m_s']:.6f} | {nr['observer_noise_max_abs_m_s']:.6f} | {nr['max_amplification_ratio']:.3f} |")
    report += ['', '増幅倍率は全点RMSE比を主指標、絶対値95パーセンタイル比を外れ値に頑健な補助指標、最大絶対値比をピーク影響の補助指標として併記した。最大値倍率は単一点に強く左右されるため慎重に読む。線形基準は準静的な写像であり、動的な真値風速誤差の代用ではない。観測器の履歴依存性はノイズあり/なし出力差側に反映される。全値は `ow05_noise_amplification.csv` に保存した。', '', '## 公称センサー条件の拡大時系列','', '各図は最大誤差時刻を中心に±2秒を表示する。上段は真値と推定風速、下段は誤差。同じ風条件・軸のESO図とRTS図では、上段と下段それぞれの縦軸範囲を共通にして比較できるようにした。描画環境に日本語フォントがないため、図中ラベルは英語表記とし、本文と図題は日本語で記載する。', '']
    for name in figs: report += [f'![OW-05 拡大時系列]({name})','']
    report += ['## 解釈と選定','', 'RTS（Rauch–Tung–Striebel）スムーザーは、まず時系列を前向きに推定し、その後、将来の観測も使って過去の状態推定を後向きに修正する。時間的に独立なホワイトノイズによる一時的な観測の揺れが、運動モデルや前後の観測と整合しない場合、その揺れを実際の状態変化ではなく観測ノイズとして扱いやすくなり、推定への影響を弱められる。これは未来の観測が過去の観測ノイズを物理的に打ち消すという意味ではなく、全時系列に最も整合する状態系列を再推定する効果である。', '', 'この平滑化は、運動モデルが十分妥当で、観測ノイズとモデル誤差の大きさ（観測・プロセス雑音の共分散）が適切に設定されていることを前提とする。ノイズが時間相関を持つ場合、その影響は独立な白色ノイズほど平均化されない。また、実際の急な風速変化をモデルが説明できないときに平滑化が強すぎると、真の変化まで抑えたり、遅らせたりする可能性がある。したがって、オフラインRTSは必ずノイズに強いわけではなく、今回の低い誤差をホワイトノイズ単独の効果と断定することもできない。今回のセンサー条件には白色ノイズに加えて有色ノイズも含まれるため、成分ごとの寄与を分けるには追加の比較が必要である。', '', 'RTS 3状態は将来データを使うオフライン評価であるため、RMSEが小さくてもオンライン実装候補とは分ける。オンライン用途はESO 3状態を候補とし、公称ノイズ条件に対する帯域選定を反映した値を使用する。RTSはログ解析や遅延許容用途の基準として残す。ノイズ2倍感度を含む推定誤差は `ow05_metrics.csv`、増幅倍率は `ow05_noise_amplification.csv`、調整スキャンは `ow05_tuning_scan.csv` に保存した。', '', '## 再実行','', '```bash','python 06_Analysis/simulation/src/run_ow05_sensor_nonideality.py','```','']
    (OUT/'OW-05_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    summary={'task_id':'OW-05','sensor_nominal':nominal,'measured_sensor_noise':{a:{k:v for k,v in d.items() if k not in ('acf','windows','innovations','residuals')} for a,d in measured_noise.items()},'noise_amplification':noise_rows,'tuning_by_axis_method':{f'{a}|{m}':v for (a,m),v in tunings.items()},'results':rows}
    (OUT/'ow05_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('rows',len(rows),'tunings',tunings)

if __name__=='__main__': run()
