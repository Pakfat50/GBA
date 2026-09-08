"""Plot all four PoC series over full records and ten-minute intervals.

Run beside poc.py, using its existing results and cache. --repo-root points to
the GBA checkout if original raw samples are to be read from another location.
No observer tuning or scoring is repeated, and no plot averaging is applied.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = ROOT.parent
COLORS = {'Raw':'#2563eb', 'Low-pass':'#8b45ba', 'Observer':'#00876c', 'Reference':'#df333b'}
LABELS = {'Raw':'Raw angle -> static speed (no smoothing)',
          'Low-pass':'Low-pass filter', 'Observer':'Observer', 'Reference':'Reference'}

def clock_label(t, _=None, seconds=False):
    t = int(round(t))
    h, rest = divmod(t,3600)
    m, s = divmod(rest,60)
    return f'{h:02}:{m:02}:{s:02}' if seconds else f'{h:02}:{m:02}'

def original_speed(repo, label, cache, config):
    name = {'am':'LOG00016.TXT', 'pm':'LOG00017.TXT'}[label]
    path = repo/'05_Script/01_Analysis/20250114'/name
    if path.is_file():
        rows=[]
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                cols=line.split('\t')
                if len(cols)!=6: continue
                try: rows.append([float(v) for v in cols[:5]])
                except ValueError: continue
        x=np.asarray(rows)
        wraps=np.r_[0,np.cumsum(np.diff(x[:,0])<0)]
        device_time=(x[:,0]+wraps*2**32)/1e6
        a,b=cache['clock']
        tj=a*device_time+b+config['shared_clock_shift_s']
        angle=x[:,3:5]
        origin='original logged angle samples; no time resampling'
    else:
        tj=cache['tj']+config['shared_clock_shift_s']
        angle=cache['angle']
        origin='existing 100 Hz resampled angle cache (original file unavailable)'
    tau=config['model']['K']*np.sin(np.deg2rad(angle))
    speed=config['common_am_gain']*np.sqrt(
        2*np.abs(tau).sum(axis=1)/(
          config['lever_m']*config['rho']*config['cd_effective']*config['area']))
    return tj,speed,origin

def load(label,repo,config):
    d=dict(np.load(ANALYSIS_ROOT/'data/processed'/f'{label}.npz'))
    estimates=pd.read_csv(ANALYSIS_ROOT/'results'/f'estimates_100hz_pole1hz_{label}.csv.gz')
    tr,raw,origin=original_speed(repo,label,d,config)
    tj=estimates.jst_time_s_aligned.to_numpy()
    # Read the prior exclusion flags without redefining their thresholds.
    bad=estimates.valid_sample.to_numpy()==0
    result={'Raw':(tr,raw), 'Low-pass':(tj,estimates.speed_lowpass_mps.to_numpy()),
            'Observer':(tj,estimates.speed_observer_mps.to_numpy()),
            'Reference':(d['ref'][:,0],d['ref'][:,2]),
            'mask_time':tj,'bad':bad,'origin':origin,'start':max(tr[0],tj[0]),
            'end':min(tr[-1],tj[-1])}
    print(label,origin,'samples',len(raw),'JST',clock_label(result['start'],seconds=True),
          clock_label(result['end'],seconds=True),flush=True)
    return result

def intervals(t,bad):
    changes=np.diff(np.r_[False,bad,False].astype(int))
    starts=np.flatnonzero(changes==1); ends=np.flatnonzero(changes==-1)
    for a,b in zip(starts,ends):
        yield t[a],t[min(b,len(t)-1)]

def draw(ax,d,start,end,title,full=False,mark_previous=False):
    # Keep excluded samples visible. Gray marks where force estimates are invalid.
    for a,b in intervals(d['mask_time'],d['bad']):
        if b>=start and a<=end:
            ax.axvspan(max(a,start),min(b,end),facecolor='#c8cdd3',alpha=.42,zorder=0)
    handles={}; maxima=[]
    for name in ['Raw','Low-pass','Observer','Reference']:
        t,y=d[name];ok=(t>=start)&(t<=end)
        maxima.append(np.nanmax(y[ok]))
        width={'Raw':.43 if full else .65,'Low-pass':.7 if full else 1.25,
               'Observer':.8 if full else 1.25,'Reference':.68 if full else 1.0}[name]
        alpha={'Raw':.55,'Low-pass':.92,'Observer':.94,'Reference':.85}[name]
        handles[name],=ax.plot(t[ok],y[ok],color=COLORS[name],lw=width,
                             alpha=alpha,label=LABELS[name],zorder=2)
    ymax=max(maxima)*1.1
    ax.set_xlim(start,end);ax.set_ylim(0,ymax)
    ax.xaxis.set_major_locator(MultipleLocator(600 if full else 60))
    ax.xaxis.set_major_formatter(FuncFormatter(clock_label))
    ax.set_xlabel('JST clock time on 2025-01-14 [HH:MM]')
    ax.set_ylabel('Wind-speed proxy [m/s]')
    ax.set_title(title,loc='left',fontsize=13,pad=11)
    ax.grid(axis='both',alpha=.18,zorder=-1)
    if mark_previous:
        previous_start=16*3600+21*60+3.375
        ax.axvline(previous_start,color='#50535a',ls='--',lw=.8)
        ax.axvline(previous_start+60,color='#50535a',ls='--',lw=.8)
        ax.annotate('Previous 60 s view',xy=(previous_start+30,ymax*.89),
                    xytext=(previous_start-58,ymax*.965),fontsize=9,
                    arrowprops={'arrowstyle':'->','color':'#50535a'},color='#343941')
    return handles

def save_one(out,name,d,start,end,title,full=False,mark_previous=False):
    fig,ax=plt.subplots(figsize=(15,5.7))
    fig.subplots_adjust(left=.065,right=.985,bottom=.18,top=.78)
    handles=draw(ax,d,start,end,title,full,mark_previous)
    fig.legend(handles.values(),[LABELS[k] for k in handles],loc='upper center',
               bbox_to_anchor=(.52,.925),ncol=4,frameon=False,fontsize=10)
    fig.suptitle('GBA | Raw, low-pass, observer and reference',fontsize=16,x=.065,ha='left',y=.99)
    fig.text(.065,.061,'All samples plotted; no display averaging. GBA curves share the prior AM multiplier (0.7849); reference unchanged.',fontsize=9,color='#4d5663')
    fig.text(.065,.025,'LPF / observer: pole parameter 1 Hz; torque-filter bandwidth 0.51 Hz. Gray: excluded transients / saturation; estimates there are invalid.',fontsize=9,color='#4d5663')
    fig.savefig(out/name,dpi=220,facecolor='white');plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=ANALYSIS_ROOT.parent)
    parser.add_argument('--output',type=Path,default=ANALYSIS_ROOT/'results/wide_plots')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((ANALYSIS_ROOT/'results/config_used.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
      'axes.spines.top':False,'axes.spines.right':False,
      'path.simplify':True,'path.simplify_threshold':0.0,'agg.path.chunksize':12000})
    ds={label:load(label,args.repo_root,config) for label in ['am','pm']}
    for label,d in ds.items():
        title=f"{label.upper()} full record: {clock_label(d['start'],seconds=True)} - {clock_label(d['end'],seconds=True)} JST ({(d['end']-d['start'])/60:.1f} min)"
        save_one(args.output,f'GBA_FourSeries_{label.upper()}_Full.png',d,d['start'],d['end'],title,full=True)
    save_one(args.output,'GBA_FourSeries_PM_10min.png',ds['pm'],16*3600+15*60,16*3600+25*60,
      'PM expanded view: 16:15 - 16:25 JST (10 min)',mark_previous=True)
    save_one(args.output,'GBA_FourSeries_AM_10min.png',ds['am'],9*3600+50*60,10*3600,
      'AM expanded view: 09:50 - 10:00 JST (10 min)')
    metadata={'source_commit':config['source_commit'],'raw_source':{k:d['origin'] for k,d in ds.items()},
      'clock':'JST seconds from midnight, with the prior +1.60 s shared device shift',
      'display_averaging':False,'pole_parameter_hz':1,'torque_filter_3db_hz':.5098245285,
      'common_am_speed_multiplier':config['common_am_gain'],
      'full_ranges_jst_s':{k:[d['start'],d['end']] for k,d in ds.items()},
      'previous_plot_range_jst_s':[58863.375,58923.375],
      'note':'Full records include excluded windows. Gray spans show the previously fixed sensor exclusion mask, not missing reference bins. No metrics or fitted parameters are changed.'}
    (args.output/'plot_conditions.json').write_text(json.dumps(metadata,indent=2))
    print('All four-series plots complete',flush=True)

if __name__=='__main__':main()
