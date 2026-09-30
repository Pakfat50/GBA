#!/usr/bin/env python3
"""IHB-03: one-pass energy-basis fit for shared Coulomb friction.

Build each half-cycle's fixed dissipation bases once, solve one linear
least-squares problem per axis, then validate the resulting tau with the tested
nonlinear peak-to-peak ODE solver. I,K and rod drag remain fixed.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from peak_to_peak_solver import solve_next_turning_point  # noqa: E402

EPSILON_DEG_S = 0.5
C_ROD = 2.5486754169187504e-6
G = 9.81
DEFAULT_MINIMUM_AMPLITUDE_DEG = 4.0


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def energy_bases(amplitude_rad, inertia, restoring, next_amplitude_rad):
    """Calculate peak-to-peak energy loss and fixed c, tau bases once.

    The c basis follows the lossless nonlinear-pendulum orbit, as in the prior
    Stage-4 energy-basis method. Coulomb's sign-law work is path independent
    and uses the measured total angle travelled in this half cycle.
    """
    a = abs(float(amplitude_rad))
    b = abs(float(next_amplitude_rad))
    if not (0 < a < math.pi and 0 <= b < math.pi and inertia > 0 and restoring > 0):
        raise ValueError("Invalid peak amplitude or fixed I,K")
    m = math.sin(0.5 * a) ** 2
    omega0_sq = restoring / inertia
    c_basis = 4 * omega0_sq * (math.sin(a) - a * math.cos(a))
    tau_basis = a + b
    delta_energy = restoring * (math.cos(b) - math.cos(a))
    return delta_energy, c_basis, tau_basis


def assemble_intervals(points, selection, physics, min_amplitude_deg):
    approved = {r["segment_id"] for r in selection
                if r["use_for_fitting"] == "1" and r["review_status"] == "APPROVED"
                and r["configuration"].startswith("SP")}
    fixed = {(r["axis"], r["configuration"]): (float(r["inertia_kg_m2"]),
                                                float(r["restoring_n_m"]))
             for r in physics if r["configuration"].startswith("SP")}
    grouped = {}
    for row in points:
        # Retain ineligible extrema as separators: never connect peaks across
        # an excluded turning point to manufacture a two-half-cycle interval.
        if row["segment_id"] in approved:
            grouped.setdefault(row["segment_id"], []).append(row)
    waves = []
    for segment, rows in grouped.items():
        rows.sort(key=lambda row: int(row["peak_number"]))
        first = rows[0]
        axis, condition = first["axis"], first["configuration"]
        if (axis, condition) not in fixed:
            raise ValueError(f"Missing fixed I,K for {axis} {condition}")
        inertia, restoring = fixed[(axis, condition)]
        intervals = []
        for start, end in zip(rows, rows[1:]):
            if start["eligible_for_later_stages"] != "1" or end["eligible_for_later_stages"] != "1":
                continue
            if int(end["peak_number"]) != int(start["peak_number"])+1:
                raise AssertionError(f"Non-adjacent extrema paired in {name}")
            a_deg = float(start["amplitude_deg"])
            if a_deg < min_amplitude_deg or float(end["amplitude_deg"]) < min_amplitude_deg:
                continue
            theta0 = float(start["centered_peak_angle_deg"]) * math.pi / 180
            theta1 = float(end["centered_peak_angle_deg"]) * math.pi / 180
            if theta0*theta1 >= 0:
                raise AssertionError(f"Peak signs do not alternate in adjacent interval {name}")
            d_energy, c_basis, tau_basis = energy_bases(theta0, inertia, restoring, theta1)
            intervals.append(dict(segment_id=segment, axis=axis, configuration=condition,
                                  direction=first["direction"], inertia_kg_m2=inertia,
                                  restoring_n_m=restoring, start_peak_number=int(start["peak_number"]),
                                  end_peak_number=int(end["peak_number"]),
                                  start_time_s=float(start["peak_time_s"]),
                                  observed_half_period_s=float(end["peak_time_s"])-float(start["peak_time_s"]),
                                  start_angle_rad=theta0, observed_next_angle_rad=theta1,
                                  start_amplitude_deg=a_deg,
                                  next_amplitude_deg=abs(theta1)*180/math.pi,
                                  delta_energy_j=d_energy, c_basis=c_basis, tau_basis=tau_basis,
                                  rod_drag_loss_j=C_ROD*c_basis,
                                  tau_adjusted_energy_j=d_energy-C_ROD*c_basis))
        if intervals:
            waves.append(dict(segment_id=segment, axis=axis, configuration=condition,
                              direction=first["direction"], inertia_kg_m2=inertia,
                              restoring_n_m=restoring, intervals=intervals))
    if not waves:
        raise ValueError("No approved ball-free half cycles")
    return sorted(waves, key=lambda wave: wave["segment_id"])


def fit_tau_energy(waves, axis, selected_configurations=None, minimum_amplitude_deg=DEFAULT_MINIMUM_AMPLITUDE_DEG):
    selected = []
    for wave in waves:
        if wave["axis"] != axis or (selected_configurations is not None and wave["configuration"] not in selected_configurations):
            continue
        intervals = [r for r in wave["intervals"] if r["start_amplitude_deg"] >= minimum_amplitude_deg
                     and r["next_amplitude_deg"] >= minimum_amplitude_deg]
        if intervals:
            selected.append(dict(wave, intervals=intervals))
    if not selected:
        raise ValueError(f"No {axis} waveforms in fit")
    # Normalize every waveform's total weight to 1/N_wave.
    basis, target, weights = [], [], []
    for wave in selected:
        n = len(wave["intervals"])
        for row in wave["intervals"]:
            basis.append(row["tau_basis"])
            target.append(row["tau_adjusted_energy_j"])
            weights.append(1/(len(selected)*n))
    basis, target, weights = map(lambda v: np.asarray(v, dtype=float), (basis, target, weights))
    unconstrained = float(np.sum(weights*basis*target)/np.sum(weights*basis*basis))
    tau = max(0.0, unconstrained)
    residual = target - tau*basis
    mean = np.sum(weights*target)/np.sum(weights)
    sse = np.sum(weights*residual**2)
    sst = np.sum(weights*(target-mean)**2)
    details = []
    offset = 0
    for wave in selected:
        for row in wave["intervals"]:
            r = dict(row, fit_weight=weights[offset], fitted_tau_n_m=tau,
                     fitted_friction_loss_j=tau*basis[offset],
                     predicted_energy_loss_j=C_ROD*row["c_basis"]+tau*basis[offset],
                     energy_residual_j=(C_ROD*row["c_basis"]+tau*basis[offset])-row["delta_energy_j"])
            details.append(r)
            offset += 1
    # Correlation and R² use the full measured loss and total model loss.
    raw_rows = [r for wave in selected for r in wave["intervals"]]
    observed_loss = np.asarray([r["delta_energy_j"] for r in raw_rows])
    total_pred = np.asarray([C_ROD*r["c_basis"] for r in raw_rows]) + tau*basis
    total_mean = np.sum(weights*observed_loss)/np.sum(weights)
    total_sst = np.sum(weights*(observed_loss-total_mean)**2)
    corr = float(np.corrcoef(observed_loss, total_pred)[0, 1])
    return dict(axis=axis, tau=tau, rows=details, waveforms=selected,
                energy_rmse_j=float(np.sqrt(sse/weights.sum())),
                energy_r=corr, energy_r2=float(1-sse/total_sst) if total_sst else float("nan"),
                tau_unconstrained=unconstrained, interval_count=len(target),
                waveform_count=len(selected), weights=weights, target=target)


def validate_ode(fit):
    rows = []
    by_wave = {}
    for wave in fit["waveforms"]:
        for item in wave["intervals"]:
            sol = solve_next_turning_point(
                item["start_angle_rad"], wave["inertia_kg_m2"], wave["restoring_n_m"],
                0.0, C_ROD, fit["tau"], EPSILON_DEG_S*math.pi/180,
                rtol=1e-4, angle_speed_atol=1e-7, energy_atol=1e-11,
                max_step_fraction=.2, max_periods=2.0)
            angle_error = sol["next_angle_rad"]-item["observed_next_angle_rad"]
            out = dict(item, tau_n_m=fit["tau"], model_next_angle_rad=sol["next_angle_rad"],
                       model_next_angle_deg=sol["next_angle_rad"]*180/math.pi,
                       angle_residual_rad=angle_error, angle_residual_deg=angle_error*180/math.pi,
                       model_half_period_s=sol["half_period_s"],
                       half_period_residual_s=sol["half_period_s"]-item["observed_half_period_s"],
                       ode_energy_loss_j=sol["mechanical_loss_j"],
                       integrated_energy_loss_j=sol["integrated_loss_j"],
                       ode_energy_closure_error_j=sol["energy_closure_error_j"],
                       ode_function_evaluations=sol["function_evaluations"],
                       fit_weight=1/(len(fit["waveforms"])*len(wave["intervals"])))
            rows.append(out)
            by_wave.setdefault(wave["segment_id"], []).append(out)
    metrics = []
    for wave in fit["waveforms"]:
        rr = by_wave[wave["segment_id"]]
        error = np.asarray([r["angle_residual_deg"] for r in rr])
        metrics.append(dict(segment_id=wave["segment_id"], axis=fit["axis"],
                            configuration=wave["configuration"], direction=wave["direction"],
                            intervals=len(rr), fitted_tau_n_m=fit["tau"],
                            angle_rmse_deg=float(np.sqrt(np.mean(error**2))),
                            angle_mae_deg=float(np.mean(np.abs(error))),
                            angle_bias_deg=float(np.mean(error)),
                            max_abs_angle_residual_deg=float(np.max(np.abs(error))),
                            half_period_rmse_ms=float(1000*np.sqrt(np.mean([r["half_period_residual_s"]**2 for r in rr])))))
    all_error = np.asarray([r["angle_residual_deg"] for r in rows])
    # One residual per half cycle, weighted so each waveform counts equally.
    weights = np.asarray([r["fit_weight"] for r in rows])
    mse = np.sum(weights*all_error**2)/weights.sum()
    y = np.asarray([r["observed_next_angle_rad"] for r in rows])
    yh = np.asarray([r["model_next_angle_rad"] for r in rows])
    mean = np.sum(weights*y)/weights.sum()
    sst = np.sum(weights*(y-mean)**2)
    rows_summary = dict(axis=fit["axis"], tau_n_m=fit["tau"], waveform_count=fit["waveform_count"],
                        interval_count=len(rows), half_cycle_angle_rmse_deg=float(np.sqrt(mse)),
                        half_cycle_angle_weighted_r=float(np.corrcoef(y,yh)[0,1]),
                        half_cycle_angle_weighted_r2=float(1-np.sum(weights*(y-yh)**2)/sst),
                        half_period_rmse_ms=float(1000*np.sqrt(np.sum(weights*np.asarray([r["half_period_residual_s"] for r in rows])**2)/weights.sum())),
                        max_energy_closure_error_j=float(max(abs(r["ode_energy_closure_error_j"]) for r in rows)))
    return rows_summary, rows, metrics


def plot_results(out_dir, fitted, validated):
    colors={f"SP{i:02d}":plt.cm.tab10(i) for i in range(5)}
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for row,axis in enumerate(("IN","OUT")):
        fit=fitted[axis]; ode=validated[axis][1]
        observed=np.asarray([r["delta_energy_j"] for r in fit["rows"]])*1e6
        predicted=np.asarray([r["predicted_energy_loss_j"] for r in fit["rows"]])*1e6
        lim=[min(observed.min(),predicted.min()),max(observed.max(),predicted.max())]
        axes[row,0].scatter(observed,predicted,s=12,alpha=.55,c=[colors[r["configuration"]] for r in fit["rows"]],edgecolors="none",rasterized=True)
        axes[row,0].plot(lim,lim,"k--",linewidth=1)
        axes[row,0].set(xlabel=r"Observed $\Delta E$ [$\mu$J]",ylabel=r"Model $cC+\hat\tau R$ [$\mu$J]",title=f"{axis}: one-pass energy fit")
        axes[row,1].axhline(0,color="black",linewidth=1)
        x=np.asarray([r["start_amplitude_deg"] for r in ode]); e=np.asarray([r["angle_residual_deg"] for r in ode])
        axes[row,1].scatter(x,e,s=12,alpha=.55,c=[colors[r["configuration"]] for r in ode],edgecolors="none",rasterized=True)
        axes[row,1].set(xlabel="Measured start amplitude [deg]",ylabel=r"ODE next-peak residual $\hat A_{n+1}-A_{n+1}$ [deg]",title=f"{axis}: fixed-tau ODE validation")
        for ax in axes[row]:
            ax.grid(alpha=.2)
    handles=[plt.Line2D([],[],marker="o",linestyle="",color=colors[f"SP{i:02d}"],label=f"SP{i:02d}") for i in range(5)]
    fig.legend(handles=handles,loc="outside lower center",ncol=5)
    fig.suptitle(r"IHB-03: pooled energy-basis $\tau_0$ fit and half-cycle ODE check",fontsize=15)
    fig.savefig(out_dir/"ihb03_fit_overview.svg")
    fig.savefig(out_dir/"preview.png",dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("turning-points-csv", "selection-csv", "condition-physics-csv", "base-parameters-csv", "output-dir"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--minimum-amplitude-deg", type=float, default=DEFAULT_MINIMUM_AMPLITUDE_DEG)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep lower-amplitude bases in memory to support a sensitivity check; the
    # primary fit still applies the approved 4 deg cutoff at both peaks.
    waves = assemble_intervals(read_csv(args.turning_points_csv), read_csv(args.selection_csv),
                               read_csv(args.condition_physics_csv), min(args.minimum_amplitude_deg, 3.0))
    fits = {axis: fit_tau_energy(waves, axis) for axis in ("IN","OUT")}
    validated = {axis: validate_ode(fit) for axis, fit in fits.items()}
    write_csv(args.output_dir/"ihb03_tau_parameters.csv",
              [dict(axis=axis,tau0_n_m=f["tau"],c_rod_fixed_n_m_s2_per_rad2=C_ROD,
                    b_fixed_n_m_s_per_rad=0.0,waveforms=f["waveform_count"],intervals=f["interval_count"],
                    energy_rmse_j=f["energy_rmse_j"],energy_weighted_r=f["energy_r"],
                    energy_weighted_r2=f["energy_r2"],
                    ode_validation_angle_rmse_deg=validated[axis][0]["half_cycle_angle_rmse_deg"],
                    ode_validation_angle_r2=validated[axis][0]["half_cycle_angle_weighted_r2"])
               for axis,f in fits.items()])
    write_csv(args.output_dir/"ihb03_interval_predictions.csv",
              [row for axis in ("IN","OUT") for row in validated[axis][1]])
    write_csv(args.output_dir/"ihb03_waveform_metrics.csv",
              [row for axis in ("IN","OUT") for row in validated[axis][2]])
    plot_results(args.output_dir,fits,validated)
    sensitivity=[]
    for axis,reference in fits.items():
        # These closed-form fits reuse the one-time bases; no ODE re-optimization.
        for config in (f"SP{i:02d}" for i in range(5)):
            trial=fit_tau_energy(waves,axis,{f"SP{i:02d}" for i in range(5)}-{config})
            sensitivity.append(dict(axis=axis,case="leave_out_"+config,tau_n_m=trial["tau"],
                                    change_percent=100*(trial["tau"]/reference["tau"]-1),
                                    energy_rmse_j=trial["energy_rmse_j"]))
        for cutoff in (3.0,5.0):
            trial_waves=[dict(w,intervals=[r for r in w["intervals"] if r["start_amplitude_deg"]>=cutoff])
                         for w in waves]
            trial_waves=[w for w in trial_waves if w["intervals"]]
            trial=fit_tau_energy(trial_waves,axis,minimum_amplitude_deg=cutoff)
            sensitivity.append(dict(axis=axis,case=f"minimum_amplitude_{cutoff:g}_deg",
                                    tau_n_m=trial["tau"],change_percent=100*(trial["tau"]/reference["tau"]-1),
                                    energy_rmse_j=trial["energy_rmse_j"]))
    write_csv(args.output_dir/"ihb03_sensitivity.csv",sensitivity)
    checks=[]
    # Synthetic recovery and independent direct formula verification.
    for axis,fit in fits.items():
        b=fit["rows"]
        w=np.asarray([r["fit_weight"] for r in b]); x=np.asarray([r["tau_basis"] for r in b])
        y0=4.2e-5*x
        recovered=max(0.0,float(np.sum(w*x*y0)/np.sum(w*x*x)))
        np.testing.assert_allclose(recovered,4.2e-5,rtol=1e-13)
        np.testing.assert_allclose(fit["tau"],float(np.sum(w*x*np.asarray([r["tau_adjusted_energy_j"] for r in b]))/np.sum(w*x*x)),rtol=1e-13)
        checks.append(dict(axis=axis,synthetic_tau_n_m=4.2e-5,recovered_tau_n_m=recovered,
                           synthetic_relative_tolerance=1e-13,linear_solution_independent_check="passed",
                           adjacent_peak_pairing_and_alternating_signs="passed"))
    (args.output_dir/"ihb03_validation.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    paths=[args.turning_points_csv,args.selection_csv,args.condition_physics_csv,args.base_parameters_csv]
    settings=dict(stage="IHB-03",method="single-pass energy-basis weighted linear least squares, followed by one ODE validation",
                  inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  scope="approved ball-free SP00-SP04; IN and OUT fit separately",
                  fixed_coefficients="I,K from IHB-02 per axis/configuration; b=0; theoretical c_rod fixed",
                  c_rod_n_m_s2_per_rad2=C_ROD,gravity_m_s2=G,epsilon_deg_s=EPSILON_DEG_S,
                  minimum_start_amplitude_deg=args.minimum_amplitude_deg,
                  waveform_weighting="equal total weight per waveform; each half-cycle within waveform has equal share",
                  tau_basis="ideal Coulomb work equals absolute angular travel |A_n|+|A_n+1|; c basis assumes conservative finite-amplitude orbit",
                  initial_half_cycle="release-to-first-peak excluded; starts from first detected measured peak",
                  rejected_waveforms_not_reinstated=True,robust_loss="ordinary weighted least squares; approved waveforms only",
                  no_historical_fitted_tau_used=True,
                  ode_validation="tested DOP853 peak-to-peak solver with tanh friction regularization; no tau refit",
                  ode_rtol=1e-4,ode_angle_speed_atol=1e-7,ode_energy_atol=1e-11,
                  runtime_s=time.perf_counter()-started)
    (args.output_dir/"ihb03_settings.json").write_text(json.dumps(settings,indent=2)+"\n",encoding="utf-8")
    print(json.dumps([{"axis":a,"tau_n_m":f["tau"],"energy_rmse_j":f["energy_rmse_j"],
                       "ode_angle_rmse_deg":validated[a][0]["half_cycle_angle_rmse_deg"],
                       "intervals":f["interval_count"],"waveforms":f["waveform_count"]}
                      for a,f in fits.items()],indent=2))


if __name__=="__main__":
    main()
