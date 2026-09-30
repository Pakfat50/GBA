#!/usr/bin/env python3
"""IHB-02: identify fresh ball-free I0,K0 from IHB-01 and known increments.

Requires numpy, scipy and matplotlib. No historical fitted coefficients are read.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

GRAVITY = 9.81
SCALE = np.array([1e-4, 1e-2])


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit(q, di, dk, counts):
    """Weighted ratio residual fit, initialized only by current linear equations."""
    q, di, dk, counts = map(lambda x: np.asarray(x, dtype=float), (q, di, dk, counts))
    if len(q) < 3 or np.any(counts <= 0):
        raise ValueError("At least three conditions with positive waveform counts required")
    weights = counts / counts.sum()
    # q*I0 - K0 = deltaK - q*deltaI; this provides a fresh seed.
    matrix = np.column_stack((q, -np.ones(len(q)))) * SCALE
    seed, _, rank, _ = np.linalg.lstsq(matrix * np.sqrt(weights[:, None]),
                                    (dk - q * di) * np.sqrt(weights), rcond=None)
    if rank != 2:
        raise ValueError("I0,K0 cannot be separated with these conditions")
    # Keep every configuration's inertia and restoring coefficient positive.
    lower = np.array([max(0.0, -di.min()), max(0.0, -dk.min())]) / SCALE
    seed = np.maximum(seed, lower + 1e-5)

    def residual(p):
        inertia, restoring = p * SCALE
        return np.sqrt(weights) * ((restoring + dk) / (inertia + di) - q)

    def jacobian(p):
        inertia, restoring = p * SCALE
        denominator = inertia + di
        return np.sqrt(weights[:, None]) * np.column_stack(
            (-(restoring + dk) / denominator ** 2 * SCALE[0],
             np.ones(len(q)) / denominator * SCALE[1]))

    result = least_squares(residual, seed, jac=jacobian,
                           bounds=(lower + 1e-10, np.full(2, np.inf)),
                           ftol=1e-13, xtol=1e-13, gtol=1e-11, max_nfev=1000)
    if not result.success:
        raise RuntimeError(result.message)
    inertia, restoring = result.x * SCALE
    prediction = (restoring + dk) / (inertia + di)
    mean = np.sum(weights * q)
    sse = np.sum(weights * (prediction - q) ** 2)
    sst = np.sum(weights * (q - mean) ** 2)
    prediction_mean = np.sum(weights * prediction)
    covariance = np.sum(weights * (q - mean) * (prediction - prediction_mean))
    correlation = covariance / np.sqrt(sst * np.sum(weights * (prediction - prediction_mean) ** 2))
    return dict(inertia=float(inertia), restoring=float(restoring), prediction=prediction,
                weights=weights, rmse=float(np.sqrt(sse)), r2=float(1 - sse / sst),
                r=float(correlation), nfev=result.nfev,
                scaled_jacobian_condition=float(np.linalg.cond(result.jac)),
                linear_seed_inertia=float(seed[0] * SCALE[0]),
                linear_seed_restoring=float(seed[1] * SCALE[1]))


def build_inputs(conditions, manifest):
    rows = []
    for axis in ("IN", "OUT"):
        for configuration in (f"SP{i:02d}" for i in range(5)):
            cs = [r for r in conditions if r["axis"] == axis and r["configuration"] == configuration]
            ms = [r for r in manifest if r["axis"] == axis and r["configuration"] == configuration
                  and r["valid"] == "1" and r["use_for_calibration"] == "1"]
            if len(cs) != 1 or len(ms) != 1:
                raise ValueError(f"Expected one condition and manifest row: {axis} {configuration}")
            c, m = cs[0], ms[0]
            mass, radius, centroid = (float(m[k]) for k in
                                     ("component_mass_kg", "signed_com_radius_m", "component_centroid_inertia_kg_m2"))
            rows.append(dict(axis=axis, configuration=configuration,
                             number_of_waveforms=int(c["waveforms"]),
                             number_of_accepted_periods=int(c["accepted_periods"]),
                             kappa_obs_s2=float(c["fitted_k_over_i_s2"]),
                             component_mass_kg=mass, signed_com_radius_m=radius,
                             component_centroid_inertia_kg_m2=centroid,
                             delta_inertia_kg_m2=centroid + mass * radius ** 2,
                             delta_restoring_n_m=mass * GRAVITY * radius))
    return rows


def arrays(rows):
    return tuple(np.array([r[key] for r in rows], dtype=float) for key in
                 ("kappa_obs_s2", "delta_inertia_kg_m2", "delta_restoring_n_m", "number_of_waveforms"))


def plot(output, checked, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "path"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for row_index, axis in enumerate(("IN", "OUT")):
        rows = [r for r in checked if r["axis"] == axis]
        metrics = next(r for r in base if r["axis"] == axis)
        x = np.arange(5)
        obs = np.array([r["kappa_obs_s2"] for r in rows])
        model = np.array([r["kappa_model_s2"] for r in rows])
        ax = axes[row_index, 0]
        ax.scatter(x, obs, color="tab:blue", label=r"$\kappa_{\mathrm{obs}}$", zorder=3)
        ax.plot(x, model, "k.-", label=r"$\kappa_{\mathrm{model}}$")
        ax.set_title(axis + ": known-increment fit")
        ax.set_ylabel(r"$\kappa=K/I$ [$\mathrm{s}^{-2}$]")
        ax.set_xticks(x, [r["configuration"] for r in rows])
        ax.legend()
        ax.text(.04, .05, rf"$I_0={metrics['base_inertia_kg_m2']:.7f}$ kg m$^2$" + "\n" +
                rf"$K_0={metrics['base_restoring_n_m']:.7f}$ N m" + "\n" +
                rf"$\mathrm{{RMSE}}_w={metrics['weighted_rmse_s2']:.4f}$ s$^{{-2}}$" + "\n" +
                rf"$R_w={metrics['weighted_r']:.6f}$; $R_w^2={metrics['weighted_r2']:.6f}$",
                transform=ax.transAxes, fontsize=10)
        ax = axes[row_index, 1]
        ax.axhline(0, color="black", linewidth=1)
        ax.bar(x, 100 * (model / obs - 1), color="tab:orange")
        ax.set_xticks(x, [r["configuration"] for r in rows])
        ax.set_title(axis + ": condition residuals")
        ax.set_ylabel(r"$(\kappa_{\mathrm{model}}/\kappa_{\mathrm{obs}}-1)$ [%]")
        ax = axes[row_index, 2]
        ax.scatter(obs, model, color="tab:blue")
        limits = [min(obs.min(), model.min()) * .95, max(obs.max(), model.max()) * 1.05]
        ax.plot(limits, limits, "k--", label="Agreement")
        for configuration, observed, predicted in zip([r["configuration"] for r in rows], obs, model):
            offset = (-34, 6) if configuration == "SP00" else (4, 6)
            ax.annotate(configuration, (observed, predicted), xytext=offset, textcoords="offset points", fontsize=8)
        ax.set_xlim(limits); ax.set_ylim(limits)
        ax.set_xlabel(r"$\kappa_{\mathrm{obs}}$ [$\mathrm{s}^{-2}$]")
        ax.set_ylabel(r"$\kappa_{\mathrm{model}}$ [$\mathrm{s}^{-2}$]")
        ax.set_title(axis + ": observed vs model")
        for ax in axes[row_index]:
            ax.grid(alpha=.2)
    fig.suptitle(r"IHB-02: $\kappa_{q,\mathrm{model}}=(K_0+\Delta K_q)/(I_0+\Delta I_q)$" +
                 "\nWeights proportional to waveform counts; ball-free SP00-SP04", fontsize=15)
    fig.savefig(output / "ihb02_base_fit_overview.svg")
    fig.savefig(output / "preview.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-csv", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(read_csv(args.condition_csv), read_csv(args.manifest_csv))
    base, checked, sensitivity = [], [], []
    for axis in ("IN", "OUT"):
        rows = [r for r in inputs if r["axis"] == axis]
        data = arrays(rows)
        result = fit(*data)
        base.append(dict(axis=axis, base_inertia_kg_m2=result["inertia"],
                         base_restoring_n_m=result["restoring"], calibration_levels=len(rows),
                         number_of_waveforms=int(data[3].sum()), weighted_rmse_s2=result["rmse"],
                         weighted_r=result["r"], weighted_r2=result["r2"],
                         max_abs_relative_residual_percent=float(np.max(np.abs(result["prediction"] / data[0] - 1)) * 100),
                         scaled_jacobian_condition=result["scaled_jacobian_condition"],
                         number_of_function_evaluations=result["nfev"],
                         fresh_linear_seed_inertia_kg_m2=result["linear_seed_inertia"],
                         fresh_linear_seed_restoring_n_m=result["linear_seed_restoring"]))
        for row, prediction, weight in zip(rows, result["prediction"], result["weights"]):
            checked.append(dict(**row, fit_weight=float(weight),
                                inertia_kg_m2=result["inertia"] + row["delta_inertia_kg_m2"],
                                restoring_n_m=result["restoring"] + row["delta_restoring_n_m"],
                                kappa_model_s2=float(prediction), residual_model_minus_obs_s2=float(prediction - row["kappa_obs_s2"]),
                                relative_residual_percent=float(100 * (prediction / row["kappa_obs_s2"] - 1))))
        for excluded in range(len(rows)):
            subset = [row for i, row in enumerate(rows) if i != excluded]
            test = fit(*arrays(subset))
            held_prediction = (test["restoring"] + data[2][excluded]) / (test["inertia"] + data[1][excluded])
            sensitivity.append(dict(axis=axis, case="leave_out_" + rows[excluded]["configuration"],
                                    base_inertia_kg_m2=test["inertia"], base_restoring_n_m=test["restoring"],
                                    inertia_change_percent=100 * (test["inertia"] / result["inertia"] - 1),
                                    restoring_change_percent=100 * (test["restoring"] / result["restoring"] - 1),
                                    held_out_relative_residual_percent=float(100 * (held_prediction / data[0][excluded] - 1))))
        equal = fit(data[0], data[1], data[2], np.ones(5))
        sensitivity.append(dict(axis=axis, case="equal_condition_weights", base_inertia_kg_m2=equal["inertia"],
                                base_restoring_n_m=equal["restoring"],
                                inertia_change_percent=100 * (equal["inertia"] / result["inertia"] - 1),
                                restoring_change_percent=100 * (equal["restoring"] / result["restoring"] - 1),
                                held_out_relative_residual_percent=""))
    write_csv(args.output_dir / "ihb02_base_parameters.csv", base)
    write_csv(args.output_dir / "ihb02_condition_predictions.csv", checked)
    write_csv(args.output_dir / "ihb02_sensitivity.csv", sensitivity)
    settings = dict(stage="IHB-02", gravity_m_s2=GRAVITY, method="bounded weighted ratio least squares",
                    initialization="weighted linear equations using current observations and increments only",
                    weighting="waveform counts per condition, normalized per axis",
                    constraints="I0>0, K0>0 and all condition inertia/restoring coefficients positive",
                    historical_fitted_coefficients_used=False,
                    uncertainty_note="Sensitivity checks are not confidence intervals; geometry and mass treated as fixed",
                    input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (args.condition_csv, args.manifest_csv)})
    (args.output_dir / "ihb02_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    plot(args.output_dir, checked, base)
    print(json.dumps(base, indent=2))


if __name__ == "__main__":
    main()
