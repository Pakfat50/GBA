"""Targeted integration and friction-model checks for the WBS 2.2 decision."""
import json
import numpy as np
import pandas as pd
import run_ow06_wbs22_oracle as w


def run():
    rows=[]
    for axis in w.Q0:
        nominal=w.select_coefficients(axis,'BALL')
        for corner in (False,True):
            c=w.mismatch_coefficients(axis,nominal) if corner else nominal.copy()
            for f in (.98,10.):
                traces={}
                for friction,substeps in [('tanh',5),('tanh',10),('sgn',10),('sgn',20)]:
                    t,F,y,contact,first=w.plant(c,f,.35,substeps,friction)
                    measured=w.apply_angle_sensor_model(t,y,w.SENSOR,20261008)[0]
                    ef,sf,_,_=w.estimate(measured,nominal,w.Q0[axis],smooth=True)
                    traces[(friction,substeps)]=y
                    for method,z in [('EKF_q1',ef),('RTS_q1',sf)]:
                        rows.append(dict(axis=axis,plant_case='OW04' if corner else 'matched',frequency_hz=f,
                            friction=friction,internal_dt_s=.01/substeps,method=method,
                            max_angle_deg=np.degrees(abs(y)).max(),contact_samples=int(contact.sum()),
                            **w.metrics(t,F,z,f,first)))
                for friction,coarse,fine in [('tanh',5,10),('sgn',10,20)]:
                    diff=np.degrees(traces[(friction,coarse)]-traces[(friction,fine)])
                    for r in rows:
                        if (r['axis']==axis and r['plant_case']==('OW04' if corner else 'matched')
                            and r['frequency_hz']==f and r['friction']==friction):
                            r['coarse_fine_angle_max_difference_deg']=max(abs(diff))
                            r['coarse_fine_angle_rms_difference_deg']=np.sqrt(np.mean(diff**2))
                print(axis,corner,f,flush=True)
    pd.DataFrame(rows).to_csv(w.OUT/'numerical_friction_checks.csv',index=False)
    # Exercise the hard-stop invalidation path independently of real cases.
    c=w.select_coefficients('IN','BALL');c['restoring_n_m_per_rad']*=.1
    t,F,y,contact,first=w.plant(c,.2,.9)
    assert first<len(t) and max(abs(y))<=np.deg2rad(60)+1e-12
    assert not w.metrics(t,F,F,.2,first)['valid']
    (w.OUT/'hard_stop_check.json').write_text(json.dumps(dict(contact_detected=True,
        max_angle_deg=float(np.degrees(abs(y)).max()),score_invalid_after_contact=True),indent=2)+'\n')
    late=[]
    for axis in w.Q0:
        nominal=w.select_coefficients(axis,'BALL')
        for corner in (False,True):
            c=w.mismatch_coefficients(axis,nominal) if corner else nominal.copy()
            t,F,y,_,first=w.plant(c,.98,.35,duration_s=60.)
            obs=w.apply_angle_sensor_model(t,y,w.SENSOR,20261008)[0]
            for qm in (.3,1.):
                ef,sf,_,_=w.estimate(obs,nominal,w.Q0[axis]*qm,smooth=(qm==1.))
                for method,z in [(f'EKF_q{qm:g}',ef)]+([('RTS_q1',sf)] if sf is not None else []):
                    for start,end in [(8.,22.),(40.,58.)]:
                        mask=(t>=start)&(t<=end+2.)
                        # metrics trims the final two seconds, leaving [start,end].
                        late.append(dict(axis=axis,plant_case='OW04' if corner else 'matched',
                            method=method,window_start_s=start,window_end_s=end,
                            **w.metrics(t[mask],F[mask],z[mask],.98,len(t[mask]) if first==len(t) else 0)))
    pd.DataFrame(late).to_csv(w.OUT/'late_window_checks.csv',index=False)


if __name__=='__main__':run()
