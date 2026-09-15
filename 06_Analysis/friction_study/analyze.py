"""Free decay friction comparison. Normalized coefficients; original irregular times.
Run: python analyze.py --input PATH/LOG00014.TXT
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=argparse.ArgumentParser();P.add_argument('--input',type=Path,required=True);args=P.parse_args()
out=Path(__file__).resolve().parent
windows=[(20.4,32),(47.05,59.05),(70.2,82),(93.25,105.25)]
rows=[]
for line in args.input.read_text().splitlines():
 f=line.split('\t')
 if len(f)==4:
  try: rows.append([float(f[0])*1e-6,np.deg2rad(-.0803317*float(f[2])+120.248)])
  except ValueError: pass
raw=np.array(rows);assert np.all(np.diff(raw[:,0])>0)
segments=[];starts=[]
for a,b in windows:
 z=raw[(raw[:,0]>=a)&(raw[:,0]<=b)];starts.append(z[0,0]);segments.append((z[:,0]-z[0,0],z[:,1]))
# p=[K/I,b/I,c/I,tau_f/I,offset,theta0,omega0]. smoothed dynamic friction only.
def sim(t,p,eps=np.deg2rad(.5),rtol=1e-8):
 k,b,c,f,o,x,v=p
 def rhs(t,y):
  x,v=y;return [v,-k*np.sin(x)-b*v-c*abs(v)*v-f*np.tanh(v/eps)]
 sol=solve_ivp(rhs,(0,float(t[-1])),[x,v],t_eval=t,rtol=rtol,atol=rtol*.01,max_step=.04)
 return sol.y[0]+o
models={'V':(False,False),'VQ':(True,False),'VC':(False,True),'VQC':(True,True)}
fits={};metrics=[]
for name,(quad,coul) in models.items():
 for j,(t,y) in enumerate(segments):
  idx=[0,1]+([2] if quad else [])+([3] if coul else [])+[4,5,6]
  base=np.array([5.65,.28,.02,.02, .01,y[0]-.01,0.])
  if coul:
   base=fits['V',j].copy();base[3]=.05
   if quad:base[2]=.01
  if not quad:base[2]=0
  if not coul:base[3]=0
  lo=np.array([2,0,0,0,-.08,-1.2,-3]);hi=np.array([10,3,5,1,.08,1.2,3])
  def unpack(z):
   p=base.copy();p[idx]=z;return p
  tf=t[::4];yf=y[::4]
  opt=least_squares(lambda z:sim(tf,unpack(z))-yf,base[idx],bounds=(lo[idx],hi[idx]),diff_step=1e-3,max_nfev=120,xtol=1e-7,ftol=1e-7,gtol=1e-7)
  p=unpack(opt.x);pred=sim(t,p);err=np.rad2deg(pred-y)
  fits[name,j]=p
  metrics.append(dict(model=name,segment=j+1,rmse_deg=np.sqrt(np.mean(err**2)),k_over_I=p[0],b_over_I=p[1],c_over_I=p[2],friction_over_I=p[3],offset_deg=np.rad2deg(p[4]),success=opt.success))
  print(name,j+1,metrics[-1],flush=True)
pd.DataFrame(metrics).to_csv(out/'fits.csv',index=False)
# Hold out whole trial: median dynamics from other trials, initial state/offset from first 1s only.
cv=[];predictions=[]
for name in models:
 for j,(t,y) in enumerate(segments):
  p=np.median([fits[name,i] for i in range(4) if i!=j],axis=0)
  ix=np.where(t<=1)[0][::4]
  def unpack(z):
   pp=p.copy();pp[4:]=z;return pp
  opt=least_squares(lambda z:sim(t[ix],unpack(z))-y[ix],p[4:],bounds=([-.08,-1.2,-3],[.08,1.2,3]),diff_step=1e-3,max_nfev=70)
  pp=unpack(opt.x);pred=sim(t,pp);valid=t>1;err=np.rad2deg(pred-y)
  cv.append(dict(model=name,segment=j+1,rmse_deg=np.sqrt(np.mean(err[valid]**2)),mae_deg=np.mean(abs(err[valid]))))
  predictions.append(pd.DataFrame(dict(model=name,segment=j+1,time_s=t+starts[j],angle_deg=np.rad2deg(y),fit_deg=np.rad2deg(sim(t,fits[name,j])),validation_deg=np.rad2deg(pred))))
  print('CV',cv[-1],flush=True)
pd.DataFrame(cv).to_csv(out/'validation.csv',index=False)
data=pd.concat(predictions);data.to_csv(out/'waveforms.csv',index=False)
colors={'V':'#ea8800','VQ':'#377eb8','VC':'#d62728','VQC':'#009e73'}
model_names=list(models)
# Separate each model so nearly identical curves cannot hide one another.
# Rows are trials 1--4 and columns are V / VQ / VC / VQC.
fig,axs=plt.subplots(4,4,figsize=(18,12),sharex='row',sharey='row')
for j,(t,y) in enumerate(segments):
 for k,name in enumerate(model_names):
  ax=axs[j,k]
  d=data[(data.model==name)&(data.segment==j+1)]
  ax.plot(t+starts[j],np.rad2deg(y),color='black',alpha=.55,lw=1,label='Measured')
  ax.plot(d.time_s,d.fit_deg,color=colors[name],lw=1.25,label='Fit')
  if j==0:ax.set_title(name)
  if k==0:ax.set_ylabel(f'Trial {j+1}\nAngle (deg)')
  ax.set_xlabel('Device elapsed time (s)')
  ax.grid(alpha=.25)
  ax.legend(loc='upper right',fontsize=8)
fig.suptitle('Measured free-decay waveform and in-trial fit',y=.995)
fig.tight_layout();fig.savefig(out/'fits.png',dpi=160);plt.close(fig)

fig,axs=plt.subplots(4,4,figsize=(18,12),sharex='row',sharey='row')
for j in range(4):
 for k,name in enumerate(model_names):
  ax=axs[j,k]
  d=data[(data.model==name)&(data.segment==j+1)]
  ax.plot(d.time_s,d.fit_deg-d.angle_deg,color=colors[name],lw=1)
  ax.axhline(0,color='black',lw=.6,alpha=.5)
  if j==0:ax.set_title(name)
  if k==0:ax.set_ylabel(f'Trial {j+1}\nFit - measured (deg)')
  ax.set_xlabel('Device elapsed time (s)')
  ax.grid(alpha=.25)
fig.suptitle('In-trial fit residuals',y=.995)
fig.tight_layout();fig.savefig(out/'residuals.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4))
for ax,df,title in [(axs[0],pd.DataFrame(metrics),'In-trial RMSE'),(axs[1],pd.DataFrame(cv),'Held-out dynamics: RMSE after 1 s')]:
 for n,name in enumerate(models):ax.bar(np.arange(4)+n*.2,df[df.model==name].rmse_deg,width=.19,label=name,color=colors[name])
 ax.set_xticks(np.arange(4)+.3,['Trial 1','Trial 2','Trial 3','Trial 4']);ax.set_ylabel('RMSE (deg)');ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.25)
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180)
# Integration / friction regularization sensitivity at the fitted parameters.
checks=[]
for name in ['VC','VQC']:
 for j,(t,y) in enumerate(segments):
  p=fits[name,j];ref=sim(t,p)
  for ep in [.1,.5,1.]:
   z=sim(t,p,eps=np.deg2rad(ep),rtol=1e-8)
   checks.append(dict(model=name,segment=j+1,epsilon_deg_s=ep,rmse_deg=np.sqrt(np.mean(np.rad2deg(z-y)**2)),difference_deg=np.sqrt(np.mean(np.rad2deg(z-ref)**2))))
pd.DataFrame(checks).to_csv(out/'numerics.csv',index=False)
(out/'provenance.json').write_text(json.dumps(dict(input=str(args.input),sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),windows=windows,models=models,notes='Old hardware; inner-axis ADC; normalized coefficients; fit every fourth original sample, score all original samples; no static friction identification'),indent=2))
