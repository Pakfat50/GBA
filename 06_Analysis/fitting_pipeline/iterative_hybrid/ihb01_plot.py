#!/usr/bin/env python3
"""Render the existing IHB-01 waveform fits; this script does not refit coefficients.

Run after ihb01_period_ratio.py. Requires matplotlib only for plotting.
Math labels use Matplotlib's LaTeX-style mathtext and are saved as SVG paths.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ihb01_period_ratio import elliptic_k


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cycles = read_csv(args.results_dir / "ihb01_cycle_ratios.csv")
    waves = read_csv(args.results_dir / "ihb01_waveform_ratios.csv")
    conditions = read_csv(args.results_dir / "ihb01_condition_ratios.csv")
    plt.rcParams.update({
        "svg.fonttype": "path",
        "font.size": 12,
        "mathtext.fontset": "dejavusans",
    })
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    fig, axes = plt.subplots(2, 5, figsize=(23, 10))
    for row, axis in enumerate(("IN", "OUT")):
        for col in range(5):
            config = f"SP{col:02d}"
            ax = axes[row, col]
            ws = [w for w in waves if w["axis"] == axis and w["configuration"] == config]
            points = [p for p in cycles if p["axis"] == axis
                      and p["configuration"] == config and p["accepted"] == "1"]
            metrics = next(c for c in conditions if c["axis"] == axis and c["configuration"] == config)
            upper = math.ceil(max(float(p["representative_amplitude_deg"]) for p in points) / 5) * 5
            amplitudes = [25 + (upper - 25) * i / 160 for i in range(161)]
            for index, wave in enumerate(ws):
                color = colors[index % len(colors)]
                ps = [p for p in points if p["segment_id"] == wave["segment_id"]]
                ratio = float(wave["waveform_median_k_over_i_s2"])
                model = [4 * elliptic_k(math.sin(math.radians(a) / 2) ** 2) / math.sqrt(ratio)
                         for a in amplitudes]
                ax.scatter([float(p["representative_amplitude_deg"]) for p in ps],
                           [float(p["period_s"]) for p in ps],
                           color=color, s=12, alpha=0.65, edgecolors="none", zorder=3)
                ax.plot(amplitudes, model, color=color, lw=1.5, alpha=0.9, label=f"W{index + 1}")
            ax.set_title(f"{axis} {config}\n"
                         + rf"RMSE = {float(metrics['period_rmse_ms']):.2f} ms; "
                         + rf"$R^2 = {float(metrics['period_r2']):.4f}$", fontsize=12)
            ax.set_xlabel(r"Amplitude $A_{\mathrm{obs}}$ [deg]")
            ax.set_ylabel(r"Period $T$ [s]")
            ax.set_xlim(25, upper)
            ax.grid(alpha=0.22)
            ax.legend(fontsize=9, frameon=False)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#333333", label=r"Observed: $T_{\mathrm{obs}}$"),
        Line2D([], [], lw=1.7, color="#333333", label=r"Model: $T_{\mathrm{model}}$"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=2, frameon=False)
    fig.text(0.5, 0.008,
             "Colors identify waveforms within each panel. Curves use the existing waveform-specific estimates.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=2.8, w_pad=1.4)
    output = args.output or args.results_dir / "ihb01_fit_overview.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
