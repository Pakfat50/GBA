#!/usr/bin/env python3
"""Fresh IHB-01 K/I period-ratio calculation from approved selections and peak table."""
import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def median(values):
    xs = sorted(values)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0

def elliptic_k(m):
    # Complete elliptic integral of the first kind via the arithmetic-geometric mean.
    a, b = 1.0, math.sqrt(max(0.0, 1.0 - m))
    for _ in range(80):
        an, bn = (a + b) / 2.0, math.sqrt(a * b)
        if abs(an - bn) < 1e-15:
            return math.pi / (2.0 * an)
        a, b = an, bn
    return math.pi / (2.0 * ((a + b) / 2.0))

def fit_conditions(cycle_rows, wave_rows):
    """One zero-intercept linear fit per axis/configuration, equal waveform weight.

    T_model = alpha * H(A_obs); alpha = 1/sqrt(K/I).
    Cached features contain only observations/selection flags. Old coefficients
    are retained for diagnostics and never used as inputs to this fit.
    """
    started = time.perf_counter()
    condition_rows = []
    for axis in ("IN", "OUT"):
        for config in (f"SP{i:02d}" for i in range(5)):
            points = [p for p in cycle_rows if p["axis"] == axis
                      and p["configuration"] == config and str(p["accepted"]) == "1"]
            if not points:
                raise ValueError(f"No accepted periods for {axis} {config}")
            counts = defaultdict(int)
            for p in points:
                counts[p["segment_id"]] += 1
            ws = [w for w in wave_rows if w["axis"] == axis and w["configuration"] == config]
            if {w["segment_id"] for w in ws} != set(counts):
                raise ValueError(f"Waveform metadata mismatch for {axis} {config}")
            observed = [float(p["period_s"]) for p in points]
            h = [4 * elliptic_k(math.sin(math.radians(float(p["representative_amplitude_deg"])) / 2) ** 2)
                 for p in points]
            if not all(y > 0 and math.isfinite(y) for y in observed + h):
                raise ValueError("Periods and theoretical period factors must be finite and positive")
            weights = [1 / (len(counts) * counts[p["segment_id"]]) for p in points]
            alpha = math.fsum(a * x * y for a, x, y in zip(weights, h, observed)) / math.fsum(
                a * x * x for a, x in zip(weights, h))
            ratio = alpha ** -2
            predicted = [alpha * x for x in h]
            residuals = [y - z for y, z in zip(observed, predicted)]
            mean_y = math.fsum(a * y for a, y in zip(weights, observed)) / math.fsum(weights)
            sse = math.fsum(a * e * e for a, e in zip(weights, residuals))
            sst = math.fsum(a * (y - mean_y) ** 2 for a, y in zip(weights, observed))
            unweighted_mean = math.fsum(observed) / len(observed)
            unweighted_sse = math.fsum(e * e for e in residuals)
            unweighted_sst = math.fsum((y - unweighted_mean) ** 2 for y in observed)
            for p, factor, prediction, residual, weight in zip(points, h, predicted, residuals, weights):
                p.update(period_factor_model=factor, condition_alpha_s=alpha,
                         condition_k_over_i_s2=ratio, period_model_s=prediction,
                         period_residual_s=residual, fit_weight=weight)
            # Predictions for rejected periods are diagnostic only, with zero fit weight.
            for p in cycle_rows:
                if p["axis"] == axis and p["configuration"] == config and str(p["accepted"]) != "1":
                    factor = 4 * elliptic_k(math.sin(math.radians(float(p["representative_amplitude_deg"])) / 2) ** 2)
                    p.update(period_factor_model=factor, condition_alpha_s=alpha,
                             condition_k_over_i_s2=ratio, period_model_s=alpha * factor,
                             period_residual_s=float(p["period_s"]) - alpha * factor, fit_weight=0.0)
            for w in ws:
                errors = [e for p, e in zip(points, residuals) if p["segment_id"] == w["segment_id"]]
                w.update(condition_k_over_i_s2=ratio,
                         common_fit_period_rmse_ms=1000 * math.sqrt(math.fsum(e * e for e in errors) / len(errors)),
                         common_fit_period_bias_ms=1000 * math.fsum(errors) / len(errors))
            condition_rows.append(dict(
                axis=axis, configuration=config, waveforms=len(counts), accepted_periods=len(points),
                fitted_alpha_s=alpha, fitted_k_over_i_s2=ratio,
                mean_waveform_median_k_over_i_s2=math.fsum(float(w["waveform_median_k_over_i_s2"]) for w in ws) / len(ws),
                period_rmse_ms=1000 * math.sqrt(sse / math.fsum(weights)),
                period_r2=1 - sse / sst if sst > 0 else "",
                unweighted_period_rmse_ms=1000 * math.sqrt(unweighted_sse / len(points)),
                unweighted_period_r2=1 - unweighted_sse / unweighted_sst if unweighted_sst > 0 else "",
            ))
    print(f"condition_fit_seconds={time.perf_counter() - started:.6f}")
    return condition_rows

def write_outputs(output_dir, cycle_rows, wave_rows, condition_rows):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("ihb01_cycle_ratios.csv", cycle_rows),
                           ("ihb01_waveform_ratios.csv", wave_rows),
                           ("ihb01_condition_ratios.csv", condition_rows)):
        if not rows:
            raise ValueError(f"No output rows for {filename}")
        with open(out / filename, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for r in condition_rows:
        print(f'{r["axis"]},{r["configuration"]},{r["waveforms"]},{r["accepted_periods"]},{r["fitted_k_over_i_s2"]},{r["period_rmse_ms"]},{r["period_r2"]}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-csv")
    ap.add_argument("--turning-points-csv")
    ap.add_argument("--cycle-csv", help="Reuse observed period/amplitude features and accepted flags")
    ap.add_argument("--waveform-csv", help="Metadata/diagnostics accompanying --cycle-csv")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    if args.cycle_csv or args.waveform_csv:
        if not (args.cycle_csv and args.waveform_csv) or args.selection_csv or args.turning_points_csv:
            ap.error("Use both cached CSV inputs, or both selection/turning-point inputs")
        cycles, waves = read_csv(args.cycle_csv), read_csv(args.waveform_csv)
        conditions = fit_conditions(cycles, waves)
        write_outputs(args.output_dir, cycles, waves, conditions)
        return
    if not (args.selection_csv and args.turning_points_csv):
        ap.error("Both --selection-csv and --turning-points-csv are required for feature extraction")

    selected = {
        r["segment_id"]: r for r in read_csv(args.selection_csv)
        if r.get("use_for_fitting") == "1"
        and r.get("review_status") == "APPROVED"
        and r.get("configuration") in {f"SP{i:02d}" for i in range(5)}
    }
    grouped = defaultdict(list)
    for p in read_csv(args.turning_points_csv):
        if p.get("segment_id") in selected and p.get("eligible_for_later_stages") == "1":
            grouped[p["segment_id"]].append(p)

    wave_rows, cycle_rows = [], []
    for segment_id, selection in selected.items():
        peaks = sorted(grouped.get(segment_id, []), key=lambda x: int(x["peak_number"]))
        cycles = []
        for i in range(max(0, len(peaks) - 2)):
            a, b = peaks[i], peaks[i + 2]
            if a["peak_kind"] != b["peak_kind"]:
                continue
            t0, t1 = float(a["peak_time_s"]), float(b["peak_time_s"])
            period = t1 - t0
            theta0 = math.radians(abs(float(a["centered_peak_angle_deg"])))
            theta1 = math.radians(abs(float(b["centered_peak_angle_deg"])))
            rep_amp_rad = math.acos(max(-1.0, min(1.0, (math.cos(theta0) + math.cos(theta1)) / 2.0)))
            amp_deg = math.degrees(rep_amp_rad)
            m = math.sin(rep_amp_rad / 2.0) ** 2
            ratio = (4.0 * elliptic_k(m) / period) ** 2
            if period > 0 and math.isfinite(ratio):
                cycles.append({"t0": t0, "t1": t1, "period": period, "amp": amp_deg, "ratio": ratio})

        if len(cycles) < 5:
            continue
        n_reference = max(5, math.ceil(0.10 * len(cycles)))
        reference = median([x["ratio"] for x in sorted(cycles, key=lambda x: x["amp"], reverse=True)[:n_reference]])
        kept = [x for x in cycles if x["amp"] >= 25.0 and abs(x["ratio"] / reference - 1.0) <= 0.01]
        if not kept:
            continue
        wave_ratio = median([x["ratio"] for x in kept])
        base = {**selection, "reference": reference}
        for x in cycles:
            cycle_rows.append({
                "segment_id": segment_id,
                "axis": selection["axis"],
                "configuration": selection["configuration"],
                "period_start_time_s": x["t0"],
                "period_end_time_s": x["t1"],
                "period_s": x["period"],
                "representative_amplitude_deg": x["amp"],
                "k_over_i_s2": x["ratio"],
                "high_amplitude_reference_s2": reference,
                "relative_deviation": x["ratio"] / reference - 1.0,
                "accepted": int(x in kept),
            })
        wave_rows.append({
            "segment_id": segment_id,
            "axis": selection["axis"],
            "configuration": selection["configuration"],
            "candidate_periods": len(cycles),
            "accepted_periods": len(kept),
            "high_amplitude_reference_s2": reference,
            "waveform_median_k_over_i_s2": wave_ratio,
        })

    condition_rows = fit_conditions(cycle_rows, wave_rows)
    write_outputs(args.output_dir, cycle_rows, wave_rows, condition_rows)
    print(f"selected_waveforms={len(selected)} processed_waveforms={len(wave_rows)} cycles={len(cycle_rows)}")

if __name__ == "__main__":
    main()
