"""Internal consistency check of the existing observer on measured free decay.

Run beside poc.py. The nominal model, 1 Hz pole and offline advance are unchanged.
These free-decay intervals also contributed to identification: not held-out data.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import poc

ROOT=Path(__file__).resolve().parent
ANALYSIS_ROOT=ROOT.parent
COLORS={'Static':'#2563eb','Low-pass':'#8b45ba','Observer':'#00876c'}
LABELS={'Static':'Raw angle -> static force','Low-pass':'Low-pass filter','Observer':'Observer'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=ANALYSIS_ROOT.parent)
    parser.add_argument('--output',type=Path,default=ANALYSIS_ROOT/'results/free_decay_validation')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    path=args.repo_root/'04_Data/00_Calibration/swing/LOG00014.TXT'
    rows=[];skipped=0
    for line in path.read_text(encoding='utf-8').splitlines():
        f=line.split('\t')
        if len(f)!=4:skipped+=1;continue
        try:rows.append([float(f[0])/1e6,float(f[2])*(-.0803317)+120.248])
        except ValueError:skipped+=1
    raw=np.asarray(rows)
    if not np.all(np.diff(raw[:,0])>0):raise ValueError('Original times must be increasing')
    # Preserve irregular original samples in the input-angle graph.
    # The dynamical estimator uses the existing 100 Hz implementation.
    t=np.arange(raw[0,0],raw[-1,0],.01)
    angle=np.interp(t,raw[:,0],raw[:,1])
    cfg=json.loads((ANALYSIS_ROOT/'results/config_used.json').read_text())
    model=cfg['model'];pole=1.;lever=cfg['lever_m']
    q0,q1,q2=poc.basis(np.column_stack([angle,np.zeros(len(t))]),pole)
    forces={'Static':model['K']*np.sin(np.deg2rad(angle))/lever*1000,
      'Low-pass':model['K']*q0[:,0]/lever*1000,
      'Observer':(model['I']*q2[:,0]+model['b']*q1[:,0]+model['K']*q0[:,0])/lever*1000}
    fits=json.loads((ANALYSIS_ROOT/'data/processed/free_decay.json').read_text())
    # Fixed common scoring margins, not selected to optimize observer performance.
    margin_start=3.;margin_end=.5
    all_valid=np.zeros(len(t),bool);segment=np.zeros(len(t),int);metrics=[]
    for j,fit in enumerate(fits,1):
        start,end=fit['start'],fit['end']
        segment[(t>=start)&(t<=end)]=j
        valid=(t>=start+margin_start)&(t<=end-margin_end);all_valid|=valid
        for name,y in forces.items():
            metrics.append(dict(segment=j,start_s=start,end_s=end,score_start_s=start+margin_start,
              score_end_s=end-margin_end,method=name,n=int(valid.sum()),
              rms_to_zero_mN=float(np.sqrt(np.mean(y[valid]**2))),bias_mN=float(np.mean(y[valid]))))
    for name,y in forces.items():
        metrics.append(dict(segment='all',start_s=fits[0]['start'],end_s=fits[-1]['end'],
          score_start_s=None,score_end_s=None,method=name,n=int(all_valid.sum()),
          rms_to_zero_mN=float(np.sqrt(np.mean(y[all_valid]**2))),bias_mN=float(np.mean(y[all_valid]))))
    table=pd.DataFrame(metrics);table.to_csv(out/'free_decay_metrics.csv',index=False)
    pd.DataFrame({'device_time_s':t,'inner_angle_deg':angle,'segment':segment,
      'valid_for_zero_rms':all_valid.astype(int),'static_force_mN':forces['Static'],
      'lowpass_force_mN':forces['Low-pass'],'observer_force_mN':forces['Observer']
    }).to_csv(out/'free_decay_estimates_100hz.csv',index=False,float_format='%.9g')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
      'axes.spines.right':False,'axes.grid':True,'grid.alpha':.17})
    def shade(ax,start,end):
        ax.axvspan(start,start+margin_start,color='#d9dde2',alpha=.6,zorder=0)
        ax.axvspan(end-margin_end,end,color='#d9dde2',alpha=.6,zorder=0)
    def plot_force(ax,fit,j):
        start,end=fit['start'],fit['end'];idx=(t>=start)&(t<=end)
        shade(ax,start,end)
        for name,y in forces.items():ax.plot(t[idx],y[idx],color=COLORS[name],lw=1.45,label=LABELS[name])
        ax.axhline(0,color='#34383e',ls='--',lw=.9,label='Ideal model residual: zero')
        ax.set_xlim(start,end);ax.set_ylabel('Signed equivalent force [mN]')
        ax.set_xlabel('Device time [s]')
        row=table[table.segment.eq(j)].set_index('method').rms_to_zero_mN
        ax.set_title(f"Segment {j}: {start:g}-{end:g} s | RMS: {row['Static']:.2f} / {row['Low-pass']:.2f} / {row['Observer']:.2f} mN",loc='left',fontsize=11)
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,gridspec_kw={'height_ratios':[1,1.7]})
    fig.subplots_adjust(left=.09,right=.98,top=.81,bottom=.14,hspace=.34)
    first=fits[0];start,end=first['start'],first['end'];ix=(raw[:,0]>=start)&(raw[:,0]<=end)
    axes[0].plot(raw[ix,0],raw[ix,1],color=COLORS['Static'],lw=1.3)
    axes[0].set_ylabel('Measured angle [deg]');axes[0].set_title('Original inner-axis angle (no smoothing)',loc='left',fontsize=12)
    shade(axes[0],start,end);plot_force(axes[1],first,1)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.53,.93),frameon=False,fontsize=10)
    fig.suptitle('Measured free decay | Does the observer remove pendulum oscillation?',x=.09,ha='left',fontsize=15,y=.99)
    fig.text(.09,.065,'Same nominal model and pole parameter 1 Hz as the preceding wind-data plot. No Cd or wind-speed multiplier.',fontsize=9,color='#505866')
    fig.text(.09,.035,'Gray: first 3 s and last 0.5 s excluded from RMS. Model identification used these records: internal consistency, not independent validation.',fontsize=9,color='#505866')
    fig.savefig(out/'GBA_FreeDecay_Comparison.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    fig.subplots_adjust(left=.075,right=.98,top=.83,bottom=.13,hspace=.38,wspace=.23)
    for j,(ax,fit) in enumerate(zip(axes.flat,fits),1):plot_force(ax,fit,j)
    # Common vertical range enables comparison of all four trials.
    bound=max(np.max(np.abs(y[segment>0])) for y in forces.values())*1.08
    for ax in axes.flat:ax.set_ylim(-bound,bound)
    fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.53,.925),frameon=False,fontsize=10)
    fig.suptitle('Four measured free-decay intervals | same model, same filters',x=.075,ha='left',fontsize=16,y=.99)
    fig.text(.075,.055,'RMS order: static / low-pass / observer. Gray margins are excluded for every method; no per-segment zero-offset correction.',fontsize=10,color='#505866')
    fig.text(.075,.025,'Equivalent force = residual torque / 0.245 m. Zero is the model expectation, not a separately measured force reference.',fontsize=10,color='#505866')
    fig.savefig(out/'GBA_FreeDecay_AllFour.png',dpi=200);plt.close(fig)
    meta={'source':'04_Data/00_Calibration/swing/LOG00014.TXT','source_commit':cfg['source_commit'],
      'original_rows':len(raw),'skipped_lines':skipped,'model':model,'pole_parameter_hz':pole,
      'torque_filter_3db_hz':.5098245285,'offline_advance_s':3/(2*np.pi*pole),
      'margin_start_s':margin_start,'margin_end_s':margin_end,'zero_offset_correction':False,
      'identification_data_reused':True,'interpretation':'zero residual under the free-decay model; not independently measured total aerodynamic force'}
    (out/'conditions.json').write_text(json.dumps(meta,indent=2))
    print(table.to_string(index=False),flush=True)

if __name__=='__main__':main()
