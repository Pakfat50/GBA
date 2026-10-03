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
    rows=[]; wave={}
    modes={'ideal':None,'nominal':pars,'noise_2x':AngleSensorParameters(sample_rate_hz=rate,resolution_bits=14,white_noise_std_deg=.03,coloured_noise_std_deg=.02,coloured_noise_time_constant_s=pars.coloured_noise_time_constant_s,fixed_delay_s=.01)}
    for cname,axis in plants:
      plant=plants[(cname,axis)]
      for sens,sp in modes.items():
        angle=plant['ang'] if sp is None else apply_angle_sensor_model(plant['t'],plant['ang'],sp,20261100+len(rows))[0]
        for method in ['ESO 3状態','RTS 3状態（オフライン）']:
          p=tunings[(axis,method)]; co=select_coefficients(axis,'BALL')
          try:
            if method.startswith('ESO'): force=nonlinear_luenberger_force(angle,co,dt,cfg['force_lever_m'],p,0,cfg['friction_epsilon_deg_s'])
            else: _,force=nonlinear_ekf_rts_force(angle,co,dt,cfg['force_lever_m'],np.deg2rad(.02),p,0,cfg['friction_epsilon_deg_s'])
            est=force_to_speed(cfg,force); met=metrics(plant['v'],est,plant['mask'])
          except (ValueError,FloatingPointError,np.linalg.LinAlgError): est=np.full_like(plant['v'],np.nan); met={'rmse_m_s':None,'bias_m_s':None,'mae_m_s':None,'p95_abs_error_m_s':None,'max_abs_error_m_s':None}
          row={'case':cname,'axis':axis,'sensor_case':sens,'method':method,'tuning_parameter':p,**met,'maximum_abs_angle_deg':plant['maxang'],'within_plus_minus_60_deg':plant['maxang']<=60,'offline':method.startswith('RTS')}
          rows.append(row)
          if sens=='nominal': wave[(cname,axis,method)]=(plant,est,row)
    write_csv(OUT/'ow05_metrics.csv',rows); write_csv(OUT/'ow05_tuning_scan.csv',tuning_rows)
    # Plot sensor-impact comparison for both valid wind models, zoom around global max error for each.
    figs=[]
    for (cname,axis,method),(p,e,r) in wave.items():
      err=e-p['v']; ids=np.flatnonzero(p['mask']&np.isfinite(err)); i=int(ids[np.argmax(np.abs(err[ids]))]); center=p['t'][i]; view=(p['t']>=center-2)&(p['t']<=center+2)
      fig,(a,b)=plt.subplots(2,1,figsize=(9,5.6),sharex=True,layout='constrained')
      a.plot(p['t'][view],p['v'][view],label='真値風速',color='#222'); a.plot(p['t'][view],e[view],label='推定風速（公称センサー）'); a.set_ylabel('Wind speed [m/s]'); a.set_title(f"{'Independent Kaimal, mean 2 m/s' if 'Kaimal' in cname else 'Gust, 2 to 6 m/s'} / {axis} / {'RTS 3-state (offline)' if method.startswith('RTS') else 'ESO 3-state'}"); a.legend(); a.grid(alpha=.25)
      b.plot(p['t'][view],err[view]); b.axhline(0,color='#555'); b.axvline(center,color='#b34b35',ls='--'); b.set_xlabel('Time [s]'); b.set_ylabel('Estimate - true\n[m/s]'); b.grid(alpha=.25)
      name=f"ow05_{'kaimal' if 'Kaimal' in cname else 'gust'}_{axis}_{'rts' if method.startswith('RTS') else 'eso'}.png"; fig.savefig(OUT/name,dpi=160); plt.close(fig); figs.append(name)
    report=['# OW-05 センサー非理想性とオブザーバー選定','','## 目的・結論','',
      'OW-03で角度±60°内だった独立Kaimal乱流（平均2 m/s、TI 20%）と2→6 m/sガストを使い、採用BALLプラントにセンサーの量子化、ノイズ、遅延を含めて比較した。比較対象はオンラインの3状態ESOと未来データを使うオフライン3状態RTS。4状態ESOはOW-03で除外済みのため含めない。', '',
      f"公称センサー条件では、風条件・軸ごとのRMSE最小方式は下表の通り。帯域/プロセス雑音は別seedの平均3 m/s Kaimal訓練波形で方式ごと・軸ごとに一度だけ選定し、検証波形には再調整せず適用した。最大角度は全ケースで±60°内である。","",
      '| 風条件 | 軸 | ESO RMSE | RTS RMSE |','|---|---|---:|---:|']
    for cname,axis in plants:
      vals={r['method']:r for r in rows if r['case']==cname and r['axis']==axis and r['sensor_case']=='nominal'}
      report.append(f"| {cname} | {axis} | {vals['ESO 3状態']['rmse_m_s']:.5f} m/s | {vals['RTS 3状態（オフライン）']['rmse_m_s']:.5f} m/s |")
    report += ['', '## センサー条件', '',f"角度サンプルは{rate:.0f} Hz、MT6701の14 bit（量子化幅{360/2**14:.8f}°）、白色ノイズσ={pars.white_noise_std_deg:.3f}°、有色ノイズσ={pars.coloured_noise_std_deg:.3f}°・時定数{pars.coloured_noise_time_constant_s:.3f} s、固定遅延{pars.fixed_delay_s*1000:.0f} msを公称条件とした。センサー段階評価に従った設定であり、量子化/サンプリング仕様以外のノイズと遅延は実測前の暫定仮定である。感度としてノイズσを2倍にした条件も計算した。", '',
      '## 公称センサー条件の拡大時系列','', '各図は最大誤差時刻を中心に±2秒を表示する。上段は真値と推定風速、下段は誤差。', '']
    for name in figs: report += [f'![OW-05 拡大時系列]({name})','']
    report += ['## 解釈と選定','', 'RTS 3状態はオフライン評価で未来サンプルを使用するため、RMSEが小さくてもオンライン実装候補とは分ける。オンライン用途はESO 3状態を候補とし、公称ノイズ条件に対する帯域選定を反映した値を使用する。RTSはログ解析や遅延許容用途の基準として残す。ノイズ2倍感度を含む全数値は `ow05_metrics.csv`、調整スキャンは `ow05_tuning_scan.csv` に保存した。', '', '## 再実行','', '```bash','python 06_Analysis/simulation/src/run_ow05_sensor_nonideality.py','```','']
    (OUT/'OW-05_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    summary={'task_id':'OW-05','sensor_nominal':nominal,'tuning_by_axis_method':{f'{a}|{m}':v for (a,m),v in tunings.items()},'results':rows}
    (OUT/'ow05_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('rows',len(rows),'tunings',tunings)

if __name__=='__main__': run()
