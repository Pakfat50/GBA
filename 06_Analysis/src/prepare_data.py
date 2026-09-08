"""Extract the known GBA 20250114 logs and fit the inner-axis free-decay record.

Run with --repo-root /path/to/GBA to read a local checkout. The source is read-only.
This parser intentionally targets this experiment's schema, not arbitrary logs.
"""
from pathlib import Path
import argparse,json,re
import numpy as np
from scipy import signal,optimize
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parent
ANALYSIS_ROOT=ROOT.parent
REPO=ANALYSIS_ROOT.parent
BASE=REPO/'05_Script/01_Analysis/20250114'
CACHE=ANALYSIS_ROOT/'data/processed'; CACHE.mkdir(parents=True,exist_ok=True)

def sensor(name):
    rows=[]; gps=[]; wraps=0;prev=None
    for line in (BASE/name).read_text().splitlines():
        f=line.split('\t')
        if len(f)!=6: continue
        try: a=list(map(float,f[:5]))
        except ValueError:continue
        if prev is not None and a[0]<prev:wraps+=1
        prev=a[0]; t=(a[0]+wraps*2**32)/1e6
        rows.append([t,*a[1:]])
        if 'Location:' in f[5] and 'Date:' in f[5] and 'Time:' in f[5]:
            m=re.search(r'Time:(\d+):(\d+):(\d+):(\d+)',f[5])
            if m:
                h,mi,s,cs=map(int,m.groups());gps.append([t,(h+9)*3600+mi*60+s+cs/100])
    x=np.array(rows); gp=np.array(gps)
    # Preserve the original linear clock mapping, fixing centiseconds.
    a,b=np.polyfit(gp[:,0],gp[:,1],1)
    grid=np.arange(x[0,0],x[-1,0],.01)
    angle=np.column_stack([np.interp(grid,x[:,0],x[:,j]) for j in (3,4)])
    return dict(t=grid,tj=a*grid+b,angle=angle,clock=np.array([a,b]),gps_resid=gp[:,1]-(a*gp[:,0]+b))

def reference(name):
    rows=[]
    for line in (BASE/name).read_text().splitlines()[1:]:
        if not line.strip(): continue
        f=[s.strip() for s in line.split(',')]
        h,m,s=map(float,f[0].split(' ')[1].split(':'))
        rows.append([h*3600+m*60+s,*map(float,f[1:])])
    return np.array(rows)

def prepare():
    for label,ln,rn in [('am','LOG00016.TXT','WS400am.txt'),('pm','LOG00017.TXT','WS400pm.txt')]:
        d=sensor(ln); d['ref']=reference(rn)
        np.savez_compressed(CACHE/f'{label}.npz',**d)
        print(label,len(d['t']),d['tj'][[0,-1]],'gps residual rms',np.std(d['gps_resid']),flush=True)
    p=REPO/'04_Data/00_Calibration/swing/LOG00014.TXT'
    rows=[]
    for line in p.read_text().splitlines():
        try:
            f=line.split('\t');assert len(f)==4
            rows.append([float(f[0])/1e6,float(f[2])*(-.0803317)+120.248])
        except (ValueError,AssertionError):continue
    x=np.array(rows)
    intervals=[(20.4,32),(47.05,59.05),(70.2,82),(93.25,105.25)]
    fig,axes=plt.subplots(4,1,figsize=(12,9),layout='constrained')
    results=[]
    def damp(t,c,a,b,lam,w):return c+np.exp(-lam*t)*(a*np.cos(w*t)+b*np.sin(w*t))
    for ax,(start,end) in zip(axes,intervals):
        t=np.arange(start,end,.01);y=np.interp(t,x[:,0],x[:,1]);tr=t-t[0]
        yy=signal.detrend(y); freqs=np.fft.rfftfreq(len(y),.01);spec=np.abs(np.fft.rfft(yy))
        select=(freqs>.1)&(freqs<10); w0=2*np.pi*freqs[select][np.argmax(spec[select])]
        params,_=optimize.curve_fit(damp,tr,y,p0=[np.mean(y),y[0]-np.mean(y),0,.1,w0],bounds=([-10,-100,-100,0,.2],[10,100,100,3,30]),maxfev=20000)
        c,a,b,lam,w=params;fit=damp(tr,*params);rms=np.sqrt(np.mean((fit-y)**2))
        results.append(dict(start=start,end=end,offset_deg=c,decay_per_s=lam,omega_d=w,fn_hz=np.sqrt(w*w+lam*lam)/(2*np.pi),fit_rmse_deg=rms))
        ax.plot(t,y,lw=.7,label='Measured');ax.plot(t,fit,lw=1.1,label='Damped sine fit');ax.set_ylabel('Angle [deg]');ax.set_title(f'{start:g}–{end:g} s: fd={w/(2*np.pi):.3f} Hz, decay={lam:.3f}/s, RMSE={rms:.2f} deg');ax.legend(loc='upper right')
    axes[-1].set_xlabel('Device time [s]');fig.savefig(CACHE/'free_decay.png',dpi=150);plt.close(fig)
    (CACHE/'free_decay.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=REPO,help='GBA repository root')
    parser.add_argument('--cache',type=Path,default=CACHE)
    args=parser.parse_args()
    REPO=args.repo_root;BASE=REPO/'05_Script/01_Analysis/20250114'
    CACHE=args.cache;CACHE.mkdir(parents=True,exist_ok=True)
    prepare()
