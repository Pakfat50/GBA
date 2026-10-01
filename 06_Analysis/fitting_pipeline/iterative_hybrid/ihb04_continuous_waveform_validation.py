#!/usr/bin/env python3
"""連続ODE積分で球なし自由振動波形を一括検証するスクリプト。

使い方の例 (リポジトリのルートで実行):
  python 06_Analysis/fitting_pipeline/iterative_hybrid/ihb04_continuous_waveform_validation.py

このスクリプトは承認済みの球なし計測波形を読み、IHB-02で求めた条件別 I,K と、
IHB-03の「単調関数で振幅を平滑化した一回積分法」で求めた軸別 τ を固定して、
運動方程式を最初の採用折返し点から最後の採用折返し点まで一度だけ連続積分します。
実測点へ何度も状態を戻す半周期検証とは異なり、位相誤差が時間とともに蓄積します。

出力は波形別の指標CSV、軸別集計CSV、全波形を一枚に並べたSVG/PNG、および
入力値・対象数を記録したJSONです。物理モデルや評価区間を変える場合は、まず
下の定数とコメントを読み、変更理由を記録してください。
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# 空気中のロッド抗力係数。IHB-03のエネルギー法と同じ固定値を使う。
C_ROD = 2.5486754169187504e-6  # N m s^2 / rad^2
# IHB-03のODE検証と合わせたクーロン摩擦の滑らかな近似幅。
EPSILON_DEG_S = 0.5
# ODE計算の相対・絶対許容誤差。十分細かい刻みにして数値誤差を小さくする。
RTOL = 2e-9
ATOL = 2e-11
MAX_STEP_PERIOD_FRACTION = 1.0 / 100.0
DATE = '20260921'


def read_csv(path):
    """日本語を含むCSVも読み込めるよう、UTF-8 BOMを許容する。"""
    return pd.read_csv(path, encoding='utf-8-sig')


def crossings(time_s, angle_deg):
    """計測標本間を線形補間し、角度ゼロを横切る時刻を返す。"""
    out=[]
    for i in range(len(time_s)-1):
        y0=float(angle_deg[i]); y1=float(angle_deg[i+1])
        if y0 == 0.0:
            out.append(float(time_s[i]))
        elif y0*y1 < 0.0:
            out.append(float(time_s[i] - y0*(time_s[i+1]-time_s[i])/(y1-y0)))
    return np.asarray(out, dtype=float)


def read_segment(date_dir, row, center_deg, first_peak_s, last_peak_s):
    """指定されたCSVの選択区間を読み、中心角を引いた角度を返す。"""
    path=date_dir/str(row.data_file)
    data=pd.read_csv(path, encoding='utf-8-sig')
    start=int(row.start_index); end=int(row.end_index)+1
    part=data.iloc[start:end]
    time=pd.to_numeric(part['systime[ms]'],errors='coerce').to_numpy(float)*1e-3
    raw=pd.to_numeric(part[str(row.angle_column)],errors='coerce').to_numpy(float)
    # 角度の±180度折り返しを解いてから、前処理と同じ包絡線中心を引く。
    raw=(raw+180.0)%360.0-180.0
    angle=raw-float(center_deg)
    valid=np.isfinite(time)&np.isfinite(angle)
    time=time[valid]; angle=angle[valid]
    keep=(time>=first_peak_s)&(time<=last_peak_s)
    return time[keep],angle[keep]


def integrate_wave(time_s, measured_deg, initial_deg, inertia, restoring, tau):
    """固定係数の非線形運動方程式を連続時間で数値積分する。"""
    t0=float(time_s[0]); t=time_s-t0
    theta0=math.radians(float(initial_deg))
    epsilon=math.radians(EPSILON_DEG_S)
    natural_period=2*math.pi*math.sqrt(inertia/restoring)
    def derivative(_time, state):
        theta, omega=state
        # I θ¨ + K sin(θ) + c|θ̇|θ̇ + τ tanh(θ̇/ε) = 0
        acceleration=(-restoring*math.sin(theta)
                      -C_ROD*abs(omega)*omega
                      -tau*math.tanh(omega/epsilon))/inertia
        return (omega, acceleration)
    sol=solve_ivp(derivative,(0.0,float(t[-1])),(theta0,0.0),method='DOP853',
                  t_eval=t,rtol=RTOL,atol=ATOL,
                  max_step=natural_period*MAX_STEP_PERIOD_FRACTION)
    if not sol.success or sol.y.shape[1]!=len(t):
        raise RuntimeError(f'ODE integration failed: {sol.message}')
    return np.rad2deg(sol.y[0])


def waveform_figure(path, records):
    """全波形の計測値と連続ODE値を、1枚の小分割グラフにまとめる。"""
    cols=6; rows=math.ceil(len(records)/cols)
    fig,axes=plt.subplots(rows,cols,figsize=(18,rows*1.72),squeeze=False,sharex=False)
    for ax,rec in zip(axes.flat,records):
        t=rec['time_s']-rec['time_s'][0]
        # 図を軽く保つため描画だけ約5 ms間隔に間引く。評価は全標本で行う。
        display=np.unique(np.r_[np.arange(0,len(t),5),len(t)-1])
        ax.plot(t[display],rec['measured_deg'][display],color='#222222',lw=.55,label='Measured')
        ax.plot(t[display],rec['predicted_deg'][display],color='#d62728',lw=.7,label='ODE integration')
        ax.axhline(0,color='#888888',lw=.35)
        ax.set_title(f"{rec['segment_id']}  (RMSE {rec['sample_rmse_deg']:.2f}°)",fontsize=6.5)
        ax.tick_params(labelsize=5,length=2);ax.grid(alpha=.18,lw=.3)
        ax.set_xlabel('Elapsed time [s]',fontsize=6);ax.set_ylabel('Angle [deg]',fontsize=6)
    for ax in axes.flat[len(records):]: ax.axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.975),ncol=2,fontsize=9,frameon=False)
    fig.suptitle('Ball-free free decay: measured and ODE-predicted waveforms',y=.999,fontsize=14)
    fig.tight_layout(rect=(0,.01,1,.955))
    fig.savefig(path,format='svg',bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repository-root',type=Path,default=Path(__file__).resolve().parents[3],
                        help='GBAリポジトリのルート (通常は変更不要)')
    parser.add_argument('--output-dir',type=Path,default=None,
                        help='出力先 (省略時はIHB-03結果フォルダ)')
    args=parser.parse_args(); root=args.repository_root.resolve()
    results=root/'06_Analysis/fitting_pipeline/results'/DATE
    data_dir=root/'04_Data/05_Fitting'/DATE
    output=(args.output_dir or results/'iterative_hybrid').resolve();output.mkdir(parents=True,exist_ok=True)
    turning=read_csv(results/'hybrid_identification/01_preprocessing/turning_points.csv')
    summaries=read_csv(results/'hybrid_identification/01_preprocessing/waveform_preprocessing.csv').set_index('segment_id')
    selection=read_csv(results/'waveform_review/waveform_selection.csv')
    physics=read_csv(output/'ihb02_condition_predictions.csv')
    tau_table=read_csv(output/'ihb03_monotone_peak_summary.csv')
    # 球なし、レビュー承認済み、解析採用フラグ=1の全波形を使う。
    selected=selection[(selection.use_for_fitting.astype(int)==1)&
                       (selection.review_status.str.upper()=='APPROVED')&
                       (selection.configuration.str.startswith('SP'))].copy()
    # 波形選択CSVに球ありが混入しても、明示条件によりここで除かれる。
    fixed={(r.axis,r.configuration):(float(r.inertia_kg_m2),float(r.restoring_n_m))
           for r in physics.itertuples() if str(r.configuration).startswith('SP')}
    taus={r.axis:float(r.tau_n_m) for r in tau_table.itertuples()
          if r.method=='monotone_log_endpoint'}
    peaks={key:g.sort_values('peak_number').reset_index(drop=True)
           for key,g in turning.groupby('segment_id',sort=False)}
    records=[];metrics=[]
    for row in selected.itertuples(index=False):
        sid=str(row.segment_id)
        if sid not in peaks or sid not in summaries.index:
            raise ValueError(f'{sid}: 前処理済み頂点または中心角がありません')
        pts=peaks[sid]
        # 最初の初期頂点から最後の隣接ペアまで。4度以上などIHB-03の
        # 適格半周期条件を満たす範囲に限り、静止ノイズ部分を含めない。
        eligible=pts[pts.eligible_for_later_stages.astype(int)==1]
        eligible=eligible[eligible.amplitude_deg.astype(float)>=4.0]
        if len(eligible)<2: raise ValueError(f'{sid}: 採用頂点が2点未満です')
        first=eligible.iloc[0];last=eligible.iloc[-1]
        first_time=float(first.peak_time_s);last_time=float(last.peak_time_s)
        center=float(summaries.loc[sid,'envelope_center_deg'])
        time,measured=read_segment(data_dir,row,center,first_time,last_time)
        if len(time)<2: raise ValueError(f'{sid}: 計算区間の計測点が不足しています')
        inertia,restoring=fixed[(str(row.axis),str(row.configuration))]
        tau=taus[str(row.axis)]
        predicted=integrate_wave(time,measured,float(first.centered_peak_angle_deg),inertia,restoring,tau)
        rmse=float(np.sqrt(np.mean((predicted-measured)**2)))
        mae=float(np.mean(np.abs(predicted-measured)))
        maxerr=float(np.max(np.abs(predicted-measured)))
        # ゼロ交点は計測・計算の両系列に同じ線形補間ルールを適用する。
        tm=crossings(time-time[0],measured);tp=crossings(time-time[0],predicted)
        n=min(len(tm),len(tp));shift=tp[:n]-tm[:n]
        cross_rmse=float(np.sqrt(np.mean(shift**2))) if n else float('nan')
        cross_bias=float(np.mean(shift)) if n else float('nan')
        cross_max=float(np.max(np.abs(shift))) if n else float('nan')
        rec=dict(segment_id=sid,axis=str(row.axis),configuration=str(row.configuration),
                 direction=str(row.direction),repetition=int(row.repetition),
                 samples=len(time),duration_s=float(time[-1]-time[0]),
                 inertia_kg_m2=inertia,restoring_n_m=restoring,tau_n_m=tau,
                 sample_rmse_deg=rmse,sample_mae_deg=mae,sample_max_abs_error_deg=maxerr,
                 measured_zero_crossings=len(tm),predicted_zero_crossings=len(tp),
                 matched_zero_crossings=n,zero_crossing_shift_bias_ms=cross_bias*1e3,
                 zero_crossing_shift_rmse_ms=cross_rmse*1e3,
                 zero_crossing_shift_max_abs_ms=cross_max*1e3)
        metrics.append(rec);records.append(dict(rec,time_s=time,measured_deg=measured,predicted_deg=predicted))
    if len(records)!=len(selected):raise AssertionError('選択波形の一部を処理できませんでした')
    pd.DataFrame(metrics).to_csv(output/'ihb04_continuous_waveform_metrics.csv',index=False,float_format='%.10g',encoding='utf-8-sig')
    # 軸ごとに全サンプルを連結した値とは別に、波形ごとのRMSEを平均し、
    # 長い波形が集計を支配しない全波形等重み指標も示す。
    aggregate=[]
    for axis,g in pd.DataFrame(metrics).groupby('axis'):
        for label,subset in [('全波形等重み',g),('SP条件別',g)]:
            groups=[('全条件',subset)] if label=='全波形等重み' else list(subset.groupby('configuration'))
            for conf,h in groups:
                aggregate.append(dict(axis=axis,集計=label,configuration=conf,waveforms=len(h),
                    sample_rmse_mean_deg=float(h.sample_rmse_deg.mean()),
                    zero_crossing_shift_rmse_mean_ms=float(h.zero_crossing_shift_rmse_ms.mean()),
                    zero_crossing_shift_bias_mean_ms=float(h.zero_crossing_shift_bias_ms.mean()),
                    zero_crossing_shift_max_abs_ms=float(h.zero_crossing_shift_max_abs_ms.max()),
                    matched_zero_crossings=int(h.matched_zero_crossings.sum())))
    pd.DataFrame(aggregate).to_csv(output/'ihb04_continuous_waveform_summary.csv',index=False,float_format='%.10g',encoding='utf-8-sig')
    waveform_figure(output/'ihb04_continuous_waveform_overlay.svg',records)
    settings={
      'method':'Continuous full-waveform ODE integration, no state reset after initial accepted peak',
      'waveforms':len(records),'axes':{'IN':sum(r['axis']=='IN' for r in records),'OUT':sum(r['axis']=='OUT' for r in records)},
      'tau_method':'monotone_log_endpoint one-pass energy method',
      'tau_n_m':taus,'rod_drag_c_n_m_s2_per_rad2':C_ROD,'viscous_damping_b_n_m_s_per_rad':0.0,
      'friction_regularization_epsilon_deg_s':EPSILON_DEG_S,'rtol':RTOL,'atol':ATOL,
      'zero_crossing_definition':'linear interpolation between consecutive measured time samples',
      'comparison':'no phase alignment; numerical integration starts at first accepted measured turning point with zero speed',
      'excluded':'ball-installed waveforms, not-approved waveforms, and selection rows with use_for_fitting != 1',
    }
    (output/'ihb04_continuous_waveform_settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'完了: {len(records)}波形。結果: {output}')
    print(pd.DataFrame(aggregate).to_string(index=False))

if __name__=='__main__':main()
