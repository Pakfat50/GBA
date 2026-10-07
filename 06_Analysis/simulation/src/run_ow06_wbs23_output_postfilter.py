"""Offline output-filter fallback after the WBS 2.3 observer comparison."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, freqz, iirnotch, sosfiltfilt
from scipy.ndimage import uniform_filter1d

HERE=Path(__file__).resolve().parent; SIM=HERE.parent
OUT=SIM/"results/observer_wind/ow06_frequency_aware_observer_design/wbs24_output_postprocess"
sys.path.insert(0,str(HERE))
from hbk_model_coefficients import select_coefficients,simulate_hbk_plant
from run_ow05_force_gain_frequency_response import load_config,mismatch_coefficients
from run_ow05_theoretical_error_separation import build_inputs
from run_ow06_wbs22_oracle import DT,F0,LEVER,DRAG,params,step,fixed_gain,wind,plant as sine_plant
from run_ow06_wbs23_offline_observer import metrics
from run_ow05_robust_bandwidth_damper_sweep import _fast_transition
from hbk_nonlinear_estimators import _continuous_jacobian
from run_ow06_wbs22_oracle import EPS
from sensor_model import AngleSensorParameters,apply_angle_sensor_model

CONFIG=load_config()
SENSOR=AngleSensorParameters(100.,white_noise_std_deg=.015,coloured_noise_std_deg=.010,
    coloured_noise_time_constant_s=.475,fixed_delay_s=.01)


def fixed_observer(angle,c,axis,initial_force):
    gain=fixed_gain(c,5.)[0]; x=np.array([angle[0],0.,initial_force]); p=params(c)
    out=np.empty(len(angle))
    for k,obs in enumerate(angle):
        if k: x=step(x,DT,p)
        x=x+gain*(obs-x[0]); out[k]=x[2]
    return out


def filter_output(force,fn):
    n=len(force); fs=1/DT
    out={"raw":force}
    for seconds in (.25,.5):
        width=max(3,int(round(seconds*fs))|1)
        out[f"boxcar_{seconds:g}s_zero_phase"]=uniform_filter1d(force,size=width,mode="nearest")
    for q in (3.,5.,10.):
        b,a=iirnotch(fn,q,fs=fs)
        out[f"notch_Q{q:g}"]=filtfilt(b,a,force,padtype="odd")
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    axes=("IN","OUT"); nominal={a:select_coefficients(a,"BALL") for a in axes}
    mismatch={a:mismatch_coefficients(a,nominal[a]) for a in axes}
    inputs={a:build_inputs(CONFIG,a) for a in axes}
    records=[]; responses=[]; series_meta=[]; series_arrays={}; series_cases={}
    for axis in axes:
        tdec=np.arange(3001)*DT
        inputs[axis].pop("自由振動ステップ",None)
        inputs[axis]["自由振動_30deg_release"]=(tdec,np.zeros_like(tdec),np.zeros_like(tdec),np.deg2rad(30.),0.)
        t,F,y,_,_=sine_plant(mismatch[axis],1.,.35,duration_s=30.)
        inputs[axis]["1Hz_sine"]=(t,wind(F),F,float(y[0]),F0)
        fn=np.sqrt(nominal[axis]["restoring_n_m_per_rad"]/nominal[axis]["inertia_kg_m2"])/(2*np.pi)
        for plant_case,pc in (("matched",nominal[axis]),("OW04_plant_mismatch",mismatch[axis])):
            for input_name,data in inputs[axis].items():
                t,speed,F,angle0,initial_force=data
                if t[-1]>45:
                    keep=t<=45.;t=t[keep];speed=speed[keep];F=F[keep]
                start=.5 if input_name=="自由振動_30deg_release" else 15.
                if angle0 is None: angle0=float(np.arctan(LEVER*F[0]/pc["restoring_n_m_per_rad"]))
                angle=simulate_hbk_plant(F,pc,DT,angle0,0.,LEVER,CONFIG["friction_epsilon_deg_s"])[:,0]
                for sensor_case,measurement in (("ideal",angle),
                    ("OW05_noise_seed0",apply_angle_sensor_model(t,angle,SENSOR,20261007)[0])):
                    raw=fixed_observer(measurement,nominal[axis],axis,float(initial_force))
                    for method,fhat in filter_output(raw,fn).items():
                        vhat=wind(fhat); mask=t>=start
                        row={"axis":axis,"input":input_name,"plant_case":plant_case,"sensor_case":sensor_case,
                             "method":method,"natural_frequency_hz":fn,"evaluation_start_s":start,
                             **metrics(t,speed,vhat,start,fn,max_lag_s=.5 if input_name=="1Hz_sine" else None)}
                        if input_name=="1Hz_sine":
                            X=np.column_stack((np.sin(2*np.pi*t[mask]),np.cos(2*np.pi*t[mask]),np.ones(mask.sum())))
                            aa,bb,_=np.linalg.lstsq(X,vhat[mask],rcond=None)[0]
                            at,bt,_=np.linalg.lstsq(X,speed[mask],rcond=None)[0]
                            row["sine_gain"]=float(np.hypot(aa,bb)/np.hypot(at,bt))
                            row["sine_phase_deg"]=float(np.rad2deg(np.arctan2(bb,aa)-np.arctan2(bt,at)))
                        case=(axis,input_name,plant_case,sensor_case)
                        if case not in series_cases:
                            cid=f"c{len(series_cases):03d}"; series_cases[case]=cid
                            series_arrays[f"{cid}_t"]=t.copy(); series_arrays[f"{cid}_truth"]=speed.copy()
                        cid=series_cases[case]; eid=f"e{len(series_meta):03d}"
                        series_arrays[eid]=vhat.copy()
                        series_meta.append({"axis":axis,"input":input_name,"plant_case":plant_case,
                            "sensor_case":sensor_case,"method":method,"case_id":cid,"estimate_id":eid})
                        records.append(row)
        # Analytic zero-phase filter attenuation at selected frequencies.
        freqs=np.array([.5,fn,1.5,5.,10.])
        for method in ("raw","boxcar_0.25s_zero_phase","boxcar_0.5s_zero_phase","notch_Q3","notch_Q5","notch_Q10"):
            if method.startswith("notch"):
                q=float(method.split("Q")[1]); b,a=iirnotch(fn,q,fs=100.); _,h=freqz(b,a,worN=freqs,fs=100.)
                gains=np.abs(h)**2 # filtfilt applies the magnitude response twice.
            elif method.startswith("boxcar"):
                secs=float(method.split("_")[1][:-1]); width=max(3,int(round(secs*100))|1)
                # uniform_filter1d is one symmetric moving average, so its
                # zero-phase gain is |sinc| (not the squared filtfilt gain).
                gains=np.abs(np.sinc(freqs*width/100))
            else:gains=np.ones_like(freqs)
            for f,g in zip(freqs,gains): responses.append({"axis":axis,"method":method,"frequency_hz":f,"zero_phase_gain":g})
    pd.DataFrame(records).to_csv(OUT/"postfilter_metrics.csv",index=False,float_format="%.9g")
    pd.DataFrame(responses).to_csv(OUT/"postfilter_frequency_response.csv",index=False,float_format="%.9g")
    np.savez_compressed(OUT/"timeseries.npz",__metadata__=np.array(json.dumps(series_meta,ensure_ascii=False)),**series_arrays)
    data=pd.DataFrame(records)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for ax,axis in zip(axs,axes):
        s=data[(data.axis==axis)&(data.sensor_case=="ideal")]
        tone=s[(s.input=="1Hz_sine")&(s.plant_case=="OW04_plant_mismatch")]
        free=s[(s.input=="自由振動_30deg_release")&(s.plant_case=="OW04_plant_mismatch")]
        for _,r in tone.iterrows():
            ax.scatter(r.sine_gain,r.rmse_m_s,s=45,label=r.method)
            ax.annotate(r.method.replace("_zero_phase","").replace("_","\n"),(r.sine_gain,r.rmse_m_s),fontsize=7,xytext=(4,3),textcoords="offset points")
        ax.set(title=f"{axis}: tone preservation vs RMSE",xlabel="1 Hz wind amplitude gain",ylabel="Wind-speed RMSE [m/s]")
        ax.grid(alpha=.25)
    fig.savefig(OUT/"postfilter_tone_tradeoff.png",dpi=170);plt.close(fig)
    print(f"Wrote {len(records)} output-filter rows to {OUT}")


if __name__=="__main__":main()
