"""WBS 2.2: known-frequency oracle screening, not online adaptation.

Run from the repository root. Each gain is constant for an entire experiment.
Only plant coefficients are perturbed. Settings are selected on nominal-plant
seed 0, then frozen for seeds 1/2 and the OW-04 correlated plant corner.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import expm

from hbk_model_coefficients import select_coefficients
from hbk_nonlinear_estimators import _continuous_jacobian, _rk4_step, nonlinear_ekf_rts_force
from observer import ackermann_observer_gain
from run_ow05_force_gain_frequency_response import load_config, mismatch_coefficients, SCENARIO
from run_ow05_robust_bandwidth_damper_sweep import _fast_transition
from sensor_model import AngleSensorParameters, apply_angle_sensor_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/observer_wind/ow06_frequency_aware_observer_design/wbs22_oracle'
DT = .01
CFG = load_config()
LEVER = CFG['force_lever_m']
EPS = math.radians(CFG['friction_epsilon_deg_s'])
R = math.radians(CFG['assumed_angle_noise_deg'])**2
DRAG = .5*CFG['air_density_kg_m3']*CFG['drag_coefficient']*CFG['projected_area_m2']
F0 = DRAG*2**2
Q0 = {'IN':3e-5, 'OUT':3e-4}
FREQ = [.2, .8, .95, .975, .98, 1., 1.05, 1.2, 3., 10., 30.]
QMULT = [.03, .1, .3, 1., 30., 300.]
POLES = [1., 2., 5., 10., 20.]
SENSOR = AngleSensorParameters(100., white_noise_std_deg=.015,
    coloured_noise_std_deg=.010, coloured_noise_time_constant_s=.475, fixed_delay_s=.01)
# Working screening criteria, NOT user-approved accuracy requirements.
GATE = dict(gain_min=.9, gain_max=1.1, phase_abs_max_deg=20.,
            wind_rmse_max_m_s=.1, noise_reduction_min_fraction=.1,
            rmse_increase_max_fraction=.05)


def params(c):
    return np.array([c['inertia_kg_m2'],c['restoring_n_m_per_rad'],
        c['viscous_damping_n_m_s_per_rad'], c['total_quadratic_drag_n_m_s2_per_rad2'],c['tau_n_m']])


def step(x, dt, p):
    """Scalar RK4, same constant-force predictor as the existing EKF."""
    I,K,b,c,tau=p
    th,v,F=x
    def acc(a,w):
        return (LEVER*F*math.cos(a)-K*math.sin(a)-b*w-c*abs(w)*w-tau*math.tanh(w/EPS))/I
    a1=acc(th,v)
    v2=v+dt*a1/2; a2=acc(th+dt*v/2,v2)
    v3=v+dt*a2/2; a3=acc(th+dt*v2/2,v3)
    v4=v+dt*a3; a4=acc(th+dt*v3,v4)
    return np.array([th+dt*(v+2*v2+2*v3+v4)/6,
                     v+dt*(a1+2*a2+2*a3+a4)/6,F])


def plant(c, f, ratio, substeps=5, friction='tanh', duration_s=None):
    """Continuous sine forcing, nonlinear dynamics, inelastic hard stops.

    Every sample from the first contact onward is invalid for estimation
    scoring. No post-contact recovery model is claimed.
    """
    p=params(c); I,K,b,cq,tau=p
    n=int(round((max(24.,10/f) if duration_s is None else duration_s)/DT))+1
    t=np.arange(n)*DT
    F=F0*(1+ratio*np.sin(2*np.pi*f*t))
    theta=math.atan(LEVER*F0/K); rate=0.
    y=np.empty(n); y[0]=theta
    contact=np.zeros(n,dtype=bool); first=n
    h=DT/substeps; lim=math.radians(60)
    def rhs(tm,th,v):
        fr=tau*(math.tanh(v/EPS) if friction=='tanh' else np.sign(v))
        force=F0*(1+ratio*math.sin(2*math.pi*f*tm))
        return v,(LEVER*force*math.cos(th)-K*math.sin(th)-b*v-cq*abs(v)*v-fr)/I
    for k in range(1,n):
        for j in range(substeps):
            tm=t[k-1]+j*h
            u1,a1=rhs(tm,theta,rate)
            u2,a2=rhs(tm+h/2,theta+h*u1/2,rate+h*a1/2)
            u3,a3=rhs(tm+h/2,theta+h*u2/2,rate+h*a2/2)
            u4,a4=rhs(tm+h,theta+h*u3,rate+h*a3)
            theta+=h*(u1+2*u2+2*u3+u4)/6
            rate+=h*(a1+2*a2+2*a3+a4)/6
            if abs(theta)>=lim:
                theta=math.copysign(lim,theta)
                if theta*rate>0: rate=0.
                contact[k]=True; first=min(first,k)
        y[k]=theta
    return t,F,y,contact,first


def fixed_gain(c,pole,design_wind=2.):
    fd=DRAG*design_wind**2
    state=np.array([math.atan(LEVER*fd/c['restoring_n_m_per_rad']),0.,fd])
    A=expm(_continuous_jacobian(state,c,LEVER,EPS)*DT)
    H=np.array([[1.,0.,0.]])
    # Post-measurement error transition is (I-LH)A = A-L(HA).
    # Using H rather than HA here would implement a different timing scheme.
    target=np.exp(-2*np.pi*pole*DT)
    L=ackermann_observer_gain(A,H@A,np.full(3,target)).ravel()
    E=(np.eye(3)-L[:,None]@H)@A
    return L, np.linalg.eigvals(E), np.poly(E)-np.poly(np.full(3,target))


def estimate(y,c,q=None,gain=None,smooth=False):
    """Posterior EKF output at k uses y[k], matching WBS 2.1."""
    n=len(y); x=np.array([y[0],0.,F0]); p=params(c)
    P=np.diag([math.radians(5)**2,1.,.5**2]); eye=np.eye(3)
    states=np.empty((n,3)); gains=np.empty((n,3))
    Q=np.eye(3)*1e-24
    if q is not None: Q[2,2]=q*q
    if smooth:
        cov=np.empty((n,3,3)); pred=np.empty((n,3)); pcov=np.empty((n,3,3)); trans=np.empty((n,3,3))
    for k,obs in enumerate(y):
        if k:
            if gain is None:
                A=_fast_transition(_continuous_jacobian(x,c,LEVER,EPS),DT)
                P=A@P@A.T+Q
            x=step(x,DT,p)
        else: A=eye
        if smooth: pred[k]=x; pcov[k]=P; trans[k]=A
        L=P[:,0]/(P[0,0]+R) if gain is None else gain
        x=x+L*(obs-x[0]); gains[k]=L
        if not np.isfinite(x).all() or abs(x[0])>20*np.pi or abs(x[2])>DRAG*20**2:
            # Do not plot a diverged curve as if it were a valid large signal.
            states[k:]=np.nan; gains[k:]=np.nan
            return states[:,2], None, gains, k
        if gain is None:
            C=eye.copy(); C[:,0]-=L
            P=C@P@C.T+np.outer(L,L)*R; P=(P+P.T)/2
        states[k]=x
        if smooth: cov[k]=P
    smoothed=None
    if smooth:
        sx=states.copy()
        for k in range(n-2,-1,-1):
            G=np.linalg.solve(pcov[k+1].T,(cov[k]@trans[k+1].T).T).T
            sx[k]+=G@(sx[k+1]-pred[k+1])
        smoothed=sx[:,2]
    return states[:,2],smoothed,gains,-1


def wind(F):
    return np.sign(F)*np.sqrt(np.abs(F)/DRAG)


def phasor(t,z,f):
    X=np.column_stack([np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t),np.ones(len(t))])
    a,b,_=np.linalg.lstsq(X,z,rcond=None)[0]
    return a+1j*b


def metrics(t,F,Fhat,f,first):
    mask=(t>=max(8.,4/f))&(t<=t[-1]-max(2.,2/f))&(np.arange(len(t))<first)
    base={'valid':False,'score_samples':int(mask.sum())}
    if first<len(t) or mask.sum()<max(100,200/f) or not np.isfinite(Fhat).all():
        return base
    ratio=phasor(t[mask],Fhat[mask],f)/phasor(t[mask],F[mask],f)
    phase=np.angle(ratio); error=wind(Fhat[mask])-wind(F[mask]); bias=error.mean()
    # Phase adjustment is a diagnostic only; raw online errors are unshifted.
    shifted=np.interp(t[mask]-phase/(2*np.pi*f),t,Fhat)
    ferr=shifted-F[mask]; ferr-=ferr.mean()
    return {**base,'valid':True,'gain':abs(ratio),'phase_deg':np.degrees(phase),
        'lag_ms':-1000*phase/(2*np.pi*f),'wind_rmse_m_s':np.sqrt(np.mean(error**2)),
        'wind_bias_m_s':bias,'wind_centered_rms_m_s':np.std(error),
        'wind_p95_abs_m_s':np.quantile(abs(error),.95),'wind_max_abs_m_s':max(abs(error)),
        'force_shape_nrmse':np.sqrt(np.mean(ferr**2))/np.std(F[mask]),
        'force_rmse_n':np.sqrt(np.mean((Fhat[mask]-F[mask])**2))}


def methods(c,axis):
    out=[]
    for q in QMULT: out.append((f'EKF_q{q:g}',Q0[axis]*q,None,q,np.nan,2.))
    for pole in POLES:
        out.append((f'Fixed_p{pole:g}',None,fixed_gain(c,pole)[0],np.nan,pole,2.))
    out.append(('Fixed_p5_design4',None,fixed_gain(c,5.,4.)[0],np.nan,5.,4.))
    return out


def run_case(job):
    axis,f,ratio,corner=job
    c=select_coefficients(axis,'BALL')
    cp=mismatch_coefficients(axis,c) if corner else c.copy()
    t,F,y,contact,first=plant(cp,f,ratio)
    delayed=np.interp(t-.01,t,y,left=y[0])
    measurements={'ideal':y,'delay_quant':np.deg2rad(np.round(np.rad2deg(delayed)/SENSOR.quantisation_step_deg)*SENSOR.quantisation_step_deg)}
    for seed in range(3):
        measurements[f'sensor_seed{seed}']=apply_angle_sensor_model(t,y,SENSOR,20261007+seed)[0]
    rows=[]; traces={'time_s':t,'force_true_n':F,'wind_true_m_s':wind(F),'angle_true_deg':np.rad2deg(y)}
    for name,q,L,qm,pole,dwind in methods(c,axis):
        outputs={}; gain_outputs={}; failures={}
        for mode,measurement in measurements.items():
            ef,sf,kg,fail=estimate(measurement,c,q,L,smooth=(name=='EKF_q1'))
            outputs[mode]=ef; gain_outputs[mode]=kg; failures[mode]=fail
            if sf is not None:
                outputs[mode+'_RTS']=sf
        for method,is_rts in [(name,False)]+([('RTS_q1',True)] if name=='EKF_q1' else []):
            for mode in measurements:
                key=mode+('_RTS' if is_rts else '')
                z=outputs[key] if key in outputs else np.full(len(t),np.nan)
                m=metrics(t,F,z,f,first)
                row=dict(axis=axis,frequency_hz=f,amplitude_ratio=ratio,plant_case='OW04' if corner else 'matched',
                    sensor_case=mode,method=method,q_multiplier=qm,pole_hz=pole,design_wind_m_s=dwind,
                    max_angle_deg=max(abs(np.rad2deg(y))),contact_samples=int(contact.sum()),
                    contact_fraction=contact.mean(),first_contact_s=t[first] if first<len(t) else np.nan,
                    invalid_after_contact_fraction=(len(t)-first)/len(t),divergence_sample=failures[mode],**m)
                if m['valid']:
                    mask=(t>=max(8.,4/f))&(t<=t[-1]-max(2.,2/f))
                    kg=gain_outputs[mode][mask]
                    row.update(gain_theta_rms=np.sqrt(np.mean(kg[:,0]**2)),gain_rate_rms=np.sqrt(np.mean(kg[:,1]**2)),gain_force_rms=np.sqrt(np.mean(kg[:,2]**2)))
                    if mode.startswith('sensor'):
                        reference=outputs['delay_quant'+('_RTS' if is_rts else '')]
                        dz=wind(z[mask])-wind(reference[mask])
                        row['noise_increment_rms_m_s']=np.sqrt(np.mean(dz**2))
                        dy=measurements[mode][mask]-measurements['delay_quant'][mask]
                        row['noise_amplification_m_s_per_deg']=row['noise_increment_rms_m_s']/np.sqrt(np.mean(np.rad2deg(dy)**2))
                        # Includes fixed delay/quantisation as well as stochastic noise.
                        row['total_sensor_increment_rms_m_s']=np.sqrt(np.mean((wind(z[mask])-wind(outputs['ideal'+('_RTS' if is_rts else '')][mask]))**2))
                rows.append(row)
                if mode=='sensor_seed1' and ratio==.35 and f in (.98,1.,10.) and method in ('EKF_q0.03','EKF_q0.1','EKF_q0.3','EKF_q1','EKF_q300','Fixed_p1','Fixed_p2','Fixed_p5','Fixed_p10','Fixed_p20','RTS_q1'):
                    traces[method+'_wind_m_s']=wind(z)
        if name=='EKF_q1': traces['angle_sensor_deg']=np.rad2deg(measurements['sensor_seed1'])
    if len(traces)>6:
        pd.DataFrame(traces).to_csv(OUT/f'trace_{axis}_{f:g}_{"OW04" if corner else "matched"}.csv',index=False)
    return rows


def self_check():
    c=select_coefficients('IN','BALL'); rng=np.random.default_rng(731)
    rkerr=0.; trans_err=0.; poleerr=0.
    for _ in range(20):
        x=np.array([rng.uniform(-1,1),rng.uniform(-3,3),rng.uniform(0,.08)])
        rkerr=max(rkerr,float(np.max(abs(step(x,DT,params(c))-_rk4_step(x,DT,c,LEVER,EPS)))))
        J=_continuous_jacobian(x,c,LEVER,EPS)
        trans_err=max(trans_err,float(np.max(abs(_fast_transition(J,DT)-expm(J*DT)))))
    for axis in ('IN','OUT'):
        for pole in POLES:
            _,eig,err=fixed_gain(select_coefficients(axis,'BALL'),pole)
            poleerr=max(poleerr,float(max(abs(err))))
            assert max(abs(eig))<1
    t,F,y,_,_=plant(c,1.,.1)
    n=350
    ef,sf,_,_=estimate(y[:n],c,Q0['IN'],smooth=True)
    e0,s0=nonlinear_ekf_rts_force(y[:n],c,DT,LEVER,math.sqrt(R),Q0['IN'],initial_state=np.array([y[0],0.,F0]))
    ekferr=float(max(abs(ef-e0))); rtserr=float(max(abs(sf-s0)))
    # Causality check: changing future measurements cannot affect past EKF.
    y2=y[:n].copy(); y2[200:]+=.01
    e2,_,_,_=estimate(y2,c,Q0['IN'])
    causal=float(max(abs(e2[:200]-ef[:200])))
    assert rkerr<1e-12 and trans_err<1e-10 and poleerr<1e-9
    assert ekferr<1e-10 and rtserr<1e-10 and causal==0
    return dict(rk4_max_abs_difference=rkerr,transition_max_abs_difference=trans_err,
        fixed_error_polynomial_max_difference=poleerr,ekf_force_vs_existing_max_n=ekferr,
        rts_force_vs_existing_max_n=rtserr,causal_prefix_max_difference_n=causal)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    checks=self_check();(OUT/'implementation_checks.json').write_text(json.dumps(checks,indent=2))
    print(checks,flush=True)
    if args.check_only:return
    settings=dict(task='WBS 2.2',base_commit='4cc3b63260c8800807bca8769938e85e3bdd7270',
        frequencies_hz=FREQ,main_amplitude_ratio=.35,low_amplitude_cases={'ratio':.1,'frequencies_hz':[1.,10.]},
        mean_force_n=F0,mean_force_wind_equivalent_m_s=2.,sample_period_s=DT,plant_internal_dt_s=.002,
        continuous_sine_force=True,friction='tanh(rate/epsilon)',epsilon_rad_s=EPS,
        q_multiplier=QMULT,base_q=Q0,fixed_poles_hz=POLES,fixed_design_wind_m_s=[2.,4.],
        sensor=asdict(SENSOR),noise_seeds=[20261007,20261008,20261009],
        mismatch_scenario=SCENARIO,perturbation_side='plant only; force lever and wind conversion fixed as in OW04',
        nominal={a:select_coefficients(a,'BALL') for a in Q0},
        perturbed={a:mismatch_coefficients(a,select_coefficients(a,'BALL')) for a in Q0},
        provisional_screening_criteria=GATE,
        warning='Thresholds are analysis working criteria, not approved product requirements. No online frequency detection or switching.')
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2,ensure_ascii=False)+'\n')
    jobs=[(a,f,.35,c) for a in Q0 for f in FREQ for c in (False,True)]
    jobs += [(a,f,.1,c) for a in Q0 for f in (1.,10.) for c in (False,True)]
    start=time.time();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k,rs in enumerate(pool.map(run_case,jobs),1):
            rows.extend(rs)
            print(f'{k}/{len(jobs)} {rs[0]["axis"]} f={rs[0]["frequency_hz"]} {rs[0]["plant_case"]}; {time.time()-start:.1f}s',flush=True)
            pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    print(f'Completed {len(rows)} rows in {time.time()-start:.1f}s',flush=True)


if __name__=='__main__':main()
