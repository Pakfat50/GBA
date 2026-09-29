#!/usr/bin/env python3
"""Fresh IHB-01 K/I period-ratio calculation from approved selections and peak table."""
import argparse
import csv
import math
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-csv", required=True)
    ap.add_argument("--turning-points-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

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

    condition_rows = []
    for axis in ("IN", "OUT"):
        for config in (f"SP{i:02d}" for i in range(5)):
            ws = [r for r in wave_rows if r["axis"] == axis and r["configuration"] == config]
            condition_rows.append({
                "axis": axis,
                "configuration": config,
                "waveforms": len(ws),
                "accepted_periods": sum(int(r["accepted_periods"]) for r in ws),
                "mean_waveform_median_k_over_i_s2": (
                    sum(float(r["waveform_median_k_over_i_s2"]) for r in ws) / len(ws) if ws else ""
                ),
            })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("ihb01_cycle_ratios.csv", cycle_rows),
        ("ihb01_waveform_ratios.csv", wave_rows),
        ("ihb01_condition_ratios.csv", condition_rows),
    ):
        if not rows:
            raise SystemExit(f"No output rows for {filename}; check selections and peak input.")
        with open(out / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"selected_waveforms={len(selected)} processed_waveforms={len(wave_rows)} cycles={len(cycle_rows)}")
    for r in condition_rows:
        print(f'{r["axis"]},{r["configuration"]},{r["waveforms"]},{r["accepted_periods"]},{r["mean_waveform_median_k_over_i_s2"]}')

if __name__ == "__main__":
    main()
