"""OW-06 supplemental comparison: free decay and wind input under plant mismatch.

The plant receives OW-04 coefficient offsets while each observer retains the
nominal BALL coefficients. Synthetic truth is available here for scoring.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, periodogram

from hbk_model_coefficients import select_coefficients, simulate_hbk_plant
from run_ow05_force_gain_frequency_response import load_config, mismatch_coefficients
from run_ow05_theoretical_error_separation import build_inputs
from run_ow06_wbs22_oracle import DT, F0, LEVER, R, Q0, params, step, fixed_gain, wind

SIM = Path(__file__).resolve().parents[1]
OUT = SIM / "results/observer_wind/ow06_frequency_aware_observer_design/wbs22_free_decay_wind"
CONFIG = SIM / "config/ow03_hbk_observer_tuning.json"


def estimate_custom(y, c, q, gain=None, smooth=False, initial_force=F0):
    """WBS-2.2 estimator with explicit initial force for free-decay setup."""
    n = len(y); x = np.array([y[0], 0., initial_force]); p = params(c)
    P = np.diag([np.deg2rad(5.)**2, 1., .5**2]); eye = np.eye(3)
    Q = np.eye(3)*1e-24; Q[2, 2] = q*q
    states = np.empty((n, 3)); gains = np.empty((n, 3))
    cov = np.empty((n, 3, 3)) if smooth else None
    pred = np.empty((n, 3)) if smooth else None
    pcov = np.empty((n, 3, 3)) if smooth else None
    trans = np.empty((n, 3, 3)) if smooth else None
    from hbk_nonlinear_estimators import _continuous_jacobian
    from run_ow05_robust_bandwidth_damper_sweep import _fast_transition
    from run_ow06_wbs22_oracle import EPS
    for k, obs in enumerate(y):
        if k:
            A = _fast_transition(_continuous_jacobian(x, c, LEVER, EPS), DT)
            P = A @ P @ A.T + Q
            x = step(x, DT, p)
        else: A = eye
        if smooth: pred[k] = x; pcov[k] = P; trans[k] = A
        L = P[:, 0]/(P[0, 0]+R) if gain is None else gain
        x = x + L*(obs-x[0]); gains[k] = L
        if gain is None:
            C = eye.copy(); C[:, 0] -= L
            P = C@P@C.T + np.outer(L, L)*R; P = (P+P.T)/2
        states[k] = x
        if smooth: cov[k] = P
    smoothed = None
    if smooth:
        sx = states.copy()
        for k in range(n-2, -1, -1):
            G = np.linalg.solve(pcov[k+1].T, (cov[k]@trans[k+1].T).T).T
            sx[k] += G@(sx[k+1]-pred[k+1])
        smoothed = sx[:, 2]
    return states[:, 2], smoothed, gains


def score(truth, estimate_, mask):
    e = estimate_[mask]-truth[mask]
    return {"rmse_m_s": float(np.sqrt(np.mean(e*e))),
            "bias_m_s": float(np.mean(e)),
            "centered_rms_m_s": float(np.std(e)),
            "p95_abs_m_s": float(np.quantile(np.abs(e), .95)),
            "max_abs_m_s": float(np.max(np.abs(e)))}


def band_rms(err, t, band):
    fs = 1/DT
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    return float(np.sqrt(np.mean(sosfiltfilt(sos, err)**2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    axes = ("IN", "OUT")
    nominal = {a: select_coefficients(a, "BALL") for a in axes}
    mismatched = {a: mismatch_coefficients(a, nominal[a]) for a in axes}
    inputs = {a: build_inputs(config, a) for a in axes}
    # A genuine release-from-rest free decay: no applied wind at any time.
    # Start at 30 deg to keep the OW-04 plant corner within the modeled range.
    t_decay = np.arange(3001)*DT
    free_decay = (t_decay, np.zeros_like(t_decay), np.zeros_like(t_decay),
                  np.deg2rad(30.), 0.)
    for axis in axes:
        inputs[axis].pop("自由振動ステップ", None)
        inputs[axis]["自由振動_30deg_release"] = free_decay
    records = []
    methods = ("EKF_q1", "RTS_q1", "Fixed_p5")
    for axis in axes:
        for input_name, data in inputs[axis].items():
            t, speed, force, angle0, initial_force = data
            eval_start = 0.5 if input_name == "自由振動_30deg_release" else config["evaluation_start_s"]
            mask = t >= eval_start
            if angle0 is None:
                angle0 = float(np.arctan(LEVER*force[0]/nominal[axis]["restoring_n_m_per_rad"]))
            for plant_case, plant_c in (("matched", nominal[axis]), ("OW04_plant_mismatch", mismatched[axis])):
                states = simulate_hbk_plant(force, plant_c, DT, angle0, 0., LEVER,
                                            config["friction_epsilon_deg_s"])
                angle = states[:, 0]
                estimates = {}
                L5 = fixed_gain(nominal[axis], 5.)[0]
                ekf, rts, _ = estimate_custom(angle, nominal[axis], Q0[axis], smooth=True,
                                              initial_force=float(initial_force))
                fixed, _, _ = estimate_custom(angle, nominal[axis], Q0[axis], gain=L5,
                                               initial_force=float(initial_force))
                estimates.update(EKF_q1=ekf, RTS_q1=rts, Fixed_p5=fixed)
                for method, fhat in estimates.items():
                    what = wind(fhat)
                    what = np.maximum(what, 0.)
                    base = dict(axis=axis, input=input_name, plant_case=plant_case,
                                method=method, evaluation_start_s=eval_start,
                                natural_frequency_hz=float(np.sqrt(nominal[axis]["restoring_n_m_per_rad"]/
                                                                  nominal[axis]["inertia_kg_m2"])/(2*np.pi)),
                                max_angle_deg=float(np.max(np.abs(np.rad2deg(angle)))))
                    row = {**base, **score(speed, what, mask)}
                    if input_name == "自由振動_30deg_release":
                        fn = base["natural_frequency_hz"]
                        err = what[mask]-speed[mask]
                        row["near_fn_error_rms_m_s"] = band_rms(err, t[mask], (max(.05, .75*fn), min(20., 1.25*fn)))
                    else:
                        fn = base["natural_frequency_hz"]
                        err = what[mask]-speed[mask]
                        row["near_fn_error_rms_m_s"] = band_rms(err, t[mask], (max(.05, .75*fn), min(20., 1.25*fn)))
                    records.append(row)
    pd.DataFrame(records).to_csv(OUT/"metrics.csv", index=False)
    summary = {"coefficients": "OW-04 high I, arm, tau, c corner on plant only; nominal BALL in observers",
               "signals": "free-decay release, independent Kaimal mean 2 m/s TI20%, gust 2 to 6 m/s",
               "sensor": "ideal angle, no added sensor noise; isolates coefficient effect",
               "methods": "EKF q1, offline RTS q1 using same EKF forward pass, fixed p5",
               "frequency_band": "0.75 to 1.25 times nominal natural frequency",
               "limitation": "synthetic known-wind benchmark, not real field truth"}
    (OUT/"settings.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__": main()
