"""WBS 2.3: offline frequency-aware RTS and calibration-based coefficient fit.

The study compares fixed-q RTS, an offline q selected from a baseline spectral
estimate, a frequency-augmented force smoother, and coefficient calibration
from independent zero-wind release records. No online requirement is assumed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.signal import butter, correlate, correlation_lags, periodogram, sosfiltfilt

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
OUT = SIM / "results/observer_wind/ow06_frequency_aware_observer_design/wbs23_offline"
sys.path.insert(0, str(HERE))

from hbk_model_coefficients import select_coefficients, simulate_hbk_plant
from run_ow05_force_gain_frequency_response import load_config, mismatch_coefficients
from run_ow05_theoretical_error_separation import build_inputs
from run_ow06_wbs22_oracle import (DT, F0, LEVER, R, Q0, DRAG, EPS, params,
                                    plant as sine_plant, step, wind)
from run_ow05_robust_bandwidth_damper_sweep import _fast_transition
from hbk_nonlinear_estimators import _continuous_jacobian
from sensor_model import AngleSensorParameters, apply_angle_sensor_model

CONFIG = json.loads((SIM / "config/ow03_hbk_observer_tuning.json").read_text(encoding="utf-8"))
SENSOR = AngleSensorParameters(100., white_noise_std_deg=.015,
    coloured_noise_std_deg=.010, coloured_noise_time_constant_s=.475,
    fixed_delay_s=.01)
FREQUENCY_GRID = np.asarray([.2, .8, .95, .975, .98, 1., 1.05, 1.2, 3., 10.])


def estimate_rts(angle, c, q, initial_force):
    """Vectorized 3-state EKF forward pass with a full-record RTS backward pass."""
    n = len(angle); x = np.array([angle[0], 0., initial_force]); p = params(c)
    P = np.diag([math.radians(5.)**2, 1., .5**2]); eye = np.eye(3)
    Q = np.eye(3)*1e-24; Q[2, 2] = q*q
    xf = np.empty((n, 3)); xp = np.empty((n, 3)); Pf = np.empty((n, 3, 3))
    Pp = np.empty((n, 3, 3)); As = np.empty((n, 3, 3))
    for k, obs in enumerate(angle):
        if k:
            A = _fast_transition(_continuous_jacobian(x, c, LEVER, EPS), DT)
            P = A@P@A.T+Q
            x = step(x, DT, p)
        else: A = eye
        xp[k] = x; Pp[k] = P; As[k] = A
        L = P[:, 0]/(P[0, 0]+R)
        x = x+L*(obs-x[0])
        C = eye.copy(); C[:, 0] -= L
        P = C@P@C.T+np.outer(L, L)*R; P=(P+P.T)/2
        xf[k] = x; Pf[k] = P
    xs = xf.copy()
    for k in range(n-2, -1, -1):
        G = np.linalg.solve(Pp[k+1].T, (Pf[k]@As[k+1].T).T).T
        xs[k] += G@(xs[k+1]-xp[k+1])
    return xs[:, 2]


def dominant_frequency(t, force, low=.1, high=10.):
    z = np.asarray(force, float)-np.mean(force)
    if np.std(z) < 1e-7:
        return float("nan"), 0.
    f, p = periodogram(z, fs=1/DT, window="hann", nfft=max(16384, 2**int(np.ceil(np.log2(len(z)*4)))))
    mask = (f >= low) & (f <= high)
    band_f=f[mask]; band_p=p[mask]; peak=int(np.argmax(band_p)); peak_f=float(band_f[peak])
    df=float(f[1]-f[0]); half_width=max(.1,2*df)
    total=float(np.sum(band_p)*df)
    concentrated=float(np.sum(band_p[np.abs(band_f-peak_f)<=half_width])*df/max(total,1e-30))
    return peak_f, concentrated


def selected_q(axis, f_est, oracle):
    table = oracle[(oracle.axis == axis) & (oracle.family == "EKF") & (oracle.amplitude_ratio == .35)]
    row = table.iloc[int(np.argmin(np.abs(table.frequency_hz.to_numpy(float)-f_est)))]
    name = str(row.selected_method)
    try: mult = float(name.split("q", 1)[1])
    except (IndexError, ValueError): mult = 1.
    return float(Q0[axis]*mult), float(row.frequency_hz), mult


def harmonic_rts(angle, c, frequency, initial_force, q_force):
    """RTS with force = random-walk mean + a known-frequency oscillator pair."""
    w = 2*np.pi*frequency; n = len(angle); I,K,b,cq,tau = params(c)
    x = np.array([angle[0], 0., initial_force, 0., 0.])
    P = np.diag([np.deg2rad(5.)**2, 1., .5**2, .5**2, .5**2]); eye=np.eye(5)
    Q = np.diag([1e-24, 1e-24, q_force**2, (2*q_force)**2, (2*q_force)**2])
    H = np.array([1.,0.,0.,0.,0.]); r=R
    xf=np.empty((n,5)); xp=np.empty_like(xf); Pf=np.empty((n,5,5)); Pp=np.empty_like(Pf); As=np.empty_like(Pf)
    def deriv(z):
        th,rate,fmean,fs,fc=z
        loss=b*rate+cq*abs(rate)*rate+tau*math.tanh(rate/EPS)
        acc=(LEVER*(fmean+fs)*math.cos(th)-K*math.sin(th)-loss)/I
        return np.array([rate,acc,0.,-w*fc,w*fs])
    def advance(z):
        k1=deriv(z); k2=deriv(z+DT*k1/2); k3=deriv(z+DT*k2/2); k4=deriv(z+DT*k3)
        return z+DT*(k1+2*k2+2*k3+k4)/6
    for k,obs in enumerate(angle):
        if k:
            th,rate,fmean,fs,fc=x
            J=np.zeros((5,5)); J[0,1]=1.
            J[1,0]=(-LEVER*(fmean+fs)*math.sin(th)-K*math.cos(th))/I
            J[1,1]=(-b-2*cq*abs(rate)-tau/EPS*(1-math.tanh(rate/EPS)**2))/I
            J[1,2]=J[1,3]=LEVER*math.cos(th)/I
            J[3,4]=-w; J[4,3]=w
            A=expm(J*DT); P=A@P@A.T+Q; x=advance(x)
        else: A=eye
        xp[k]=x; Pp[k]=P; As[k]=A
        L=P@H/(H@P@H+r); x=x+L*(obs-H@x)
        C=eye-np.outer(L,H); P=C@P@C.T+np.outer(L,L)*r
        xf[k]=x; Pf[k]=(P+P.T)/2
    xs=xf.copy()
    for k in range(n-2,-1,-1):
        G=np.linalg.solve(Pp[k+1].T,(Pf[k]@As[k+1].T).T).T
        xs[k]+=G@(xs[k+1]-xp[k+1])
    return xs[:,2]+xs[:,3]


def metrics(t, truth, estimate, start, fn, max_lag_s=None):
    mask=t>=start; e=estimate[mask]-truth[mask]
    # The project permits time shifts when judging waveform shape. Select the
    # lag with the highest demeaned correlation, allowing at most 20% of the
    # scored window (so at least 80% of samples remain in the comparison).
    yt=np.asarray(truth[mask],float); yh=np.asarray(estimate[mask],float)
    if np.std(yt)<1e-12:
        # With zero true input, there is no waveform to align against; the
        # false-output RMSE must remain an absolute magnitude check.
        lag=0; ya,ha=yt,yh; shape_corr=float("nan")
    else:
        yc=yt-yt.mean(); hc=yh-yh.mean()
        cross=correlate(yh,yt,mode="full",method="fft")
        lags=correlation_lags(len(hc),len(yc),mode="full")
        max_lag=int(.2*len(yc))
        if max_lag_s is not None: max_lag=min(max_lag,int(round(max_lag_s/DT)))
        allowed=np.abs(lags)<=max_lag
        # Normalize each overlap independently; otherwise the raw covariance
        # can favor a longer overlap when its waveform correlation is lower.
        ey=np.r_[0.,np.cumsum(yh)]; ty=np.r_[0.,np.cumsum(yt)]
        ey2=np.r_[0.,np.cumsum(yh*yh)]; ty2=np.r_[0.,np.cumsum(yt*yt)]
        candidates=[]
        for idx in np.flatnonzero(allowed):
            lag_i=int(lags[idx]); n=len(yh)-abs(lag_i)
            if lag_i>=0: e0,t0=lag_i,0
            else: e0,t0=0,-lag_i
            se=ey[e0+n]-ey[e0]; st=ty[t0+n]-ty[t0]
            ve=max(0.,ey2[e0+n]-ey2[e0]-se*se/n)
            vt=max(0.,ty2[t0+n]-ty2[t0]-st*st/n)
            covariance=cross[idx]-se*st/n
            candidates.append(covariance/np.sqrt(ve*vt) if ve>0 and vt>0 else -np.inf)
        lag=int(lags[np.flatnonzero(allowed)[int(np.argmax(candidates))]])
        if lag>0: ya,ha=yt[:-lag],yh[lag:]
        elif lag<0: ya,ha=yt[-lag:],yh[:lag]
        else: ya,ha=yt,yh
        denom=np.sqrt(np.sum((ya-ya.mean())**2)*np.sum((ha-ha.mean())**2))
        shape_corr=float(np.sum((ya-ya.mean())*(ha-ha.mean()))/denom) if denom>0 else float("nan")
    lag_rmse=float(np.sqrt(np.mean((ha-ya)**2)))
    lo=max(.05,.75*fn); hi=min(20.,1.25*fn)
    sos=butter(4,(lo,hi),btype="bandpass",fs=1/DT,output="sos")
    band=float(np.sqrt(np.mean(sosfiltfilt(sos,e)**2)))
    return {"rmse_m_s":float(np.sqrt(np.mean(e*e))),"bias_m_s":float(np.mean(e)),
            "centered_rms_m_s":float(np.std(e)),"near_fn_error_rms_m_s":band,
            "best_lag_s":float(lag*DT),"lag_aligned_rmse_m_s":lag_rmse,
            "lag_aligned_shape_corr":shape_corr}


def fit_normalized_free_decay(axis, true_coeff, noise_seed=991):
    """Fit K/I,b/I,c/I,tau/I on varied zero-wind releases; check held-out angles."""
    I=float(true_coeff["inertia_kg_m2"]); keys=("restoring_n_m_per_rad","viscous_damping_n_m_s_per_rad",
        "total_quadratic_drag_n_m_s2_per_rad2","tau_n_m")
    true=np.array([true_coeff[k]/I for k in keys]); nominal=select_coefficients(axis,"BALL")
    old=np.array([nominal[k]/nominal["inertia_kg_m2"] for k in keys])
    rng=np.random.default_rng(noise_seed+(0 if axis=="IN" else 1)); angles=np.deg2rad([15,20,25,30,35,40])
    records=[]; dt=DT; n=1501
    def sim_batch(a0, p, count):
        z=np.column_stack([np.asarray(a0,float),np.zeros(len(a0))]); out=np.empty((len(a0),count))
        def rhs(s):
            th,rate=s[:,0],s[:,1]; Kb,bb,cb,tb=p
            return np.column_stack([rate,-Kb*np.sin(th)-bb*rate-cb*np.abs(rate)*rate-tb*np.tanh(rate/EPS)])
        for i in range(count):
            out[:,i]=z[:,0]
            if i<count-1:
                k1=rhs(z); k2=rhs(z+dt*k1/2); k3=rhs(z+dt*k2/2); k4=rhs(z+dt*k3)
                z+=dt*(k1+2*k2+2*k3+k4)/6
        return out
    for a0 in angles:
        y=sim_batch([a0],true,n)[0]+np.deg2rad(rng.normal(0,.015,n))
        records.append((a0,y))
    scale=np.maximum(old,np.array([1e-4,1e-4,1e-4,1e-4]))
    lower=np.array([.55,.0,.05,.05]); upper=np.array([1.5,8.,8.,4.])
    def optimize(excluded=None):
        train=[r for i,r in enumerate(records) if i!=excluded]
        starts=np.array([r[0] for r in train]); observed=np.vstack([r[1] for r in train])
        def residual(z):
            p=z*scale; return ((sim_batch(starts,p,n)[:,::2]-observed[:,::2])/np.deg2rad(.015)).ravel()
        fit=least_squares(residual,np.ones(4),bounds=(lower,upper),loss="soft_l1",f_scale=1.,
                          x_scale="jac",max_nfev=100)
        return fit.x*scale
    cv=[]
    for i,(a0,y) in enumerate(records):
        pf=optimize(i)
        rmse_old=np.sqrt(np.mean(np.rad2deg(sim_batch([a0],old,n)[0,::2]-y[::2])**2))
        rmse_fit=np.sqrt(np.mean(np.rad2deg(sim_batch([a0],pf,n)[0,::2]-y[::2])**2))
        cv.append({"axis":axis,"held_angle_deg":float(np.rad2deg(a0)),"angle_rmse_old_deg":float(rmse_old),
                   "angle_rmse_calibrated_deg":float(rmse_fit),"fit_ratio":float(rmse_fit/rmse_old)})
    fitted=optimize(None)
    fitted_coeff=true_coeff.copy()
    for key,value in zip(keys,fitted): fitted_coeff[key]=float(value*I)
    return fitted_coeff, cv, {"axis":axis,"true_over_I":true.tolist(),"nominal_over_I":old.tolist(),
        "calibrated_over_I":fitted.tolist(),"inertia_assumed_known_kg_m2":I,
        "calibration_waveforms":6,"release_angles_deg":[15,20,25,30,35,40],"angle_noise_std_deg":.015}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    axes=("IN","OUT"); nominal={a:select_coefficients(a,"BALL") for a in axes}
    mismatch={a:mismatch_coefficients(a,nominal[a]) for a in axes}
    inputs={a:build_inputs(CONFIG,a) for a in axes}
    oracle=pd.read_csv(SIM/"results/observer_wind/ow06_frequency_aware_observer_design/wbs22_oracle/oracle_selection.csv")
    rows=[]; calibration=[]; fitmeta=[]; series_meta=[]; series_arrays={}; series_cases={}
    for axis in axes:
        # Replace the old 60-degree hold/release with a free release from rest.
        tdec=np.arange(3001)*DT
        inputs[axis].pop("自由振動ステップ",None)
        inputs[axis]["自由振動_30deg_release"]=(tdec,np.zeros_like(tdec),np.zeros_like(tdec),np.deg2rad(30.),0.)
        sine=sine_plant(mismatch[axis],1.,.35,duration_s=30.)
        ts,Fs,ys,_,_=sine
        inputs[axis]["1Hz_sine"]=(ts,wind(Fs),Fs,float(ys[0]),F0)
        for plant_case,pc in (("matched",nominal[axis]),("OW04_plant_mismatch",mismatch[axis])):
            # Fit a model from independent zero-wind calibration records on the same plant.
            fitc,cv,meta=fit_normalized_free_decay(axis,pc,noise_seed=990+len(rows))
            calibration.extend([{**r,"plant_case":plant_case} for r in cv]); fitmeta.append({**meta,"plant_case":plant_case})
            for input_name,data in inputs[axis].items():
                t,speed,F,angle0,initial_force=data
                if t[-1]>45.:
                    keep=t<=45.; t=t[keep]; speed=speed[keep]; F=F[keep]
                if input_name=="自由振動_30deg_release": start=.5
                else: start=15.
                if angle0 is None: angle0=float(np.arctan(LEVER*F[0]/pc["restoring_n_m_per_rad"]))
                angle=simulate_hbk_plant(F,pc,DT,angle0,0.,LEVER,CONFIG["friction_epsilon_deg_s"])[:,0]
                measured_clean=angle.copy()
                measured_noise=apply_angle_sensor_model(t,angle,SENSOR,20261007)[0]
                for sensor_case,measured in (("ideal",measured_clean),("OW05_noise_seed0",measured_noise)):
                    base_force=estimate_rts(measured,nominal[axis],Q0[axis],float(initial_force))
                    freq_est,freq_concentration=dominant_frequency(t[t>=start],base_force[t>=start])
                    if freq_concentration>=.5 and np.isfinite(freq_est) and .7<=freq_est<=1.3:
                        q_adapt,grid_f,qmult=selected_q(axis,freq_est,oracle)
                    else:
                        # Broadband/no-tone records must not be forced into a
                        # single-frequency q lookup; retain the fixed-q RTS.
                        q_adapt,grid_f,qmult=Q0[axis],np.nan,1.
                    adapt_force=estimate_rts(measured,nominal[axis],q_adapt,float(initial_force))
                    cal_force=estimate_rts(measured,fitc,q_adapt,float(initial_force))
                    cal_force_q1=estimate_rts(measured,fitc,Q0[axis],float(initial_force))
                    candidates={"RTS_fixed_q1":base_force,"RTS_frequency_from_record":adapt_force,
                                "RTS_calibrated_coeff_q1":cal_force_q1,
                                "RTS_calibrated_coeff_plus_frequency_q":cal_force}
                    if sensor_case=="ideal" and input_name=="1Hz_sine":
                        truth_f=1.
                        candidates["RTS_known_frequency_harmonic_state"]=harmonic_rts(measured,nominal[axis],truth_f,float(initial_force),Q0[axis])
                        candidates["RTS_estimated_frequency_harmonic_state"]=harmonic_rts(measured,nominal[axis],freq_est,float(initial_force),Q0[axis])
                    elif sensor_case=="ideal" and input_name in ("独立Kaimal乱流 平均2 m/s TI20%","ガスト 2→6 m/s"):
                        # A harmonic disturbance basis is a one-tone model; test it
                        # without oracle truth by using the frequency from the record.
                        candidates["RTS_estimated_frequency_harmonic_state"]=harmonic_rts(measured,nominal[axis],freq_est,float(initial_force),Q0[axis])
                    for method,force_hat in candidates.items():
                        estimate_speed=wind(force_hat)
                        m=metrics(t,speed,estimate_speed,start,math.sqrt(nominal[axis]["restoring_n_m_per_rad"]/nominal[axis]["inertia_kg_m2"])/(2*np.pi),
                                  max_lag_s=.5 if input_name=="1Hz_sine" else None)
                        rec={"axis":axis,"input":input_name,"plant_case":plant_case,"sensor_case":sensor_case,
                             "method":method,"estimated_frequency_hz":freq_est,"frequency_line_fraction":freq_concentration,
                             "nearest_oracle_frequency_hz":grid_f,
                             "selected_q_multiplier":qmult,"true_frequency_hz":1. if input_name=="1Hz_sine" else np.nan,
                             "evaluation_start_s":start,"max_angle_deg":float(np.max(np.abs(np.rad2deg(angle)))),**m}
                        if input_name=="1Hz_sine":
                            mask=t>=start; X=np.column_stack([np.sin(2*np.pi*t[mask]),np.cos(2*np.pi*t[mask]),np.ones(mask.sum())])
                            a,bias,_=np.linalg.lstsq(X,estimate_speed[mask],rcond=None)[0]
                            aa,bb,_=np.linalg.lstsq(X,speed[mask],rcond=None)[0]
                            rec["sine_gain"]=float(np.hypot(a,bias)/np.hypot(aa,bb))
                            rec["sine_phase_deg"]=float(np.rad2deg(np.arctan2(bias,a)-np.arctan2(bb,aa)))
                        case=(axis,input_name,plant_case,sensor_case)
                        if case not in series_cases:
                            cid=f"c{len(series_cases):03d}"; series_cases[case]=cid
                            series_arrays[f"{cid}_t"]=t.copy(); series_arrays[f"{cid}_truth"]=speed.copy()
                        cid=series_cases[case]; eid=f"e{len(series_meta):03d}"
                        series_arrays[eid]=estimate_speed.copy()
                        series_meta.append({"axis":axis,"input":input_name,"plant_case":plant_case,
                            "sensor_case":sensor_case,"method":method,"case_id":cid,"estimate_id":eid})
                        rows.append(rec)
                print(f"{axis} {plant_case} {input_name}: {len(rows)} rows",flush=True)
    pd.DataFrame(rows).to_csv(OUT/"metrics.csv",index=False,float_format="%.9g")
    pd.DataFrame(calibration).to_csv(OUT/"coefficient_calibration_loco.csv",index=False,float_format="%.9g")
    np.savez_compressed(OUT/"timeseries.npz",__metadata__=np.array(json.dumps(series_meta,ensure_ascii=False)),**series_arrays)
    (OUT/"coefficient_fit_settings.json").write_text(json.dumps(fitmeta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    settings={"sample_rate_hz":100,"noise":"OW-05 white 0.015 deg + coloured 0.010 deg, tau 0.475s, delay 0.01s; seed 20261007",
      "plant_mismatch":"OW-04 correlated high I/arm/tau/c corner applied to plant; nominal coefficients retained by estimators except calibration-fit arm",
      "frequency_adaptation":"dominant frequency estimated from full analyzed-record baseline RTS force periodogram; q lookup is used only for a concentrated tone estimate between 0.7 and 1.3 Hz, otherwise fixed q is retained",
      "harmonic_state":"force represented as random-walk mean plus known/estimated-frequency oscillator quadrature pair; compared on tone and broadband/gust data",
      "coefficient_calibration":"six synthetic zero-wind releases at 15..40 deg with angle noise; fit normalized K/I,b/I,c/I,tau/I; inertia assumed independently known; leave-one-release-angle-out validation",
      "limits":"synthetic truth available for score; field wind cannot supervise estimator; unknown-wind coefficient co-estimation is not claimed identifiable"}
    (OUT/"settings.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Wrote {len(rows)} observer rows, {len(calibration)} calibration rows to {OUT}")


if __name__=="__main__": main()
