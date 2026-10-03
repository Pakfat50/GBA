"""OW-05 sensor non-ideality evaluation for adopted HBK observers."""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def run():
    cfg=json.loads((SIM/'config/ow03_hbk_observer_tuning.json').read_text())
    scfg=json.loads((SIM/'config/sensor_model_stage3.json').read_text())
    prev=json.loads((SIM/'results/observer_wind/ow03_tuning/ow03_summary.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True); dt=1/cfg['sample_rate_hz']; rate=cfg['sample_rate_hz']
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
    report += ['## 解釈と選定','', 'RTS 3状態はオフライン評価で未来サンプルを使用するため、RMSEが小さくてもオンライン実装候補とは分ける。オンライン用途はESO 3状態を候補とし、公称ノイズ条件に対する帯域選定を反映した値を使用する。RTSはログ解析や遅延許容用途の基準として残す。ノイズ2倍感度を含む推定誤差は `ow05_metrics.csv`、増幅倍率は `ow05_noise_amplification.csv`、調整スキャンは `ow05_tuning_scan.csv` に保存した。', '', '## 再実行','', '```bash','python 06_Analysis/simulation/src/run_ow05_sensor_nonideality.py','```','']
    (OUT/'OW-05_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    summary={'task_id':'OW-05','sensor_nominal':nominal,'noise_amplification':noise_rows,'tuning_by_axis_method':{f'{a}|{m}':v for (a,m),v in tunings.items()},'results':rows}
    (OUT/'ow05_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('rows',len(rows),'tunings',tunings)

if __name__=='__main__': run()
