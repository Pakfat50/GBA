#!/usr/bin/env python3
"""Compare monotone parametric peak-amplitude envelopes with the GCV spline."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

R_MIN = 0.99
RMSE_MAX_DEG = 0.5


def pearson_r(y, yhat):
    if len(y) < 2 or np.std(y) == 0 or np.std(yhat) == 0:
        return float('nan')
    return float(np.corrcoef(y, yhat)[0, 1])


def fit_linear(x, y):
    """Constrained decreasing line; keep its full fitted range positive."""
    xbar, ybar = float(np.mean(x)), float(np.mean(y))
    b = -float(np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2))
    bmax = max(0.0, (ybar - 1e-9) / (float(np.max(x)) - xbar))
    b = float(np.clip(b, 0.0, bmax))
    a = ybar + b * xbar
    return a - b * x, {'a_deg': a, 'b_deg_per_normalized_time': b}


def fit_nonlinear(model, p0, lower, upper, x, y):
    result = least_squares(
        lambda p: model(p, x) - y, p0, bounds=(lower, upper),
        x_scale='jac', max_nfev=50000, ftol=1e-11, xtol=1e-11, gtol=1e-11,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(result.message)
    return model(result.x, x), result.x


def candidates(x, y):
    """All functions below are positive and non-increasing for bounded params."""
    out = {}
    out['Monotone linear'], pars = fit_linear(x, y)
    parameters = {'a_deg': pars['a_deg'], 'slope_deg_per_normalized_time': -pars['b_deg_per_normalized_time']}
    out_params = {'Monotone linear': parameters}

    specs = {
        'Exponential': (
            lambda p, z: p[0] * np.exp(-p[1] * z),
            [float(y[0]), 1.0], [1e-8, 0.0], [np.inf, np.inf],
            ('A0_deg', 'k'),
        ),
        'Stretched exponential': (
            lambda p, z: p[0] * np.exp(-p[1] * np.power(z, p[2])),
            [float(y[0]), 1.0, 1.0], [1e-8, 0.0, 0.1], [np.inf, np.inf, 5.0],
            ('A0_deg', 'k', 'p'),
        ),
        'Power law': (
            lambda p, z: p[0] / np.power(1.0 + p[1] * z, p[2]),
            [float(y[0]), 1.0, 1.0], [1e-8, 0.0, 0.1], [np.inf, np.inf, 5.0],
            ('A0_deg', 'k', 'p'),
        ),
        'Log + asymptote': (
            lambda p, z: p[0] + p[1] * np.log((1.0 + p[2]) / (z + p[2])),
            [max(1e-5, float(y[-1])), max(1e-5, float(y[0] - y[-1]) / 2), 0.2],
            [1e-8, 0.0, 1e-5], [np.inf, np.inf, 100.0],
            ('A_inf_deg', 'B_deg', 'C'),
        ),
    }
    for name, (fn, p0, lo, hi, names) in specs.items():
        yh, p = fit_nonlinear(fn, p0, lo, hi, x, y)
        out[name] = yh
        out_params[name] = dict(zip(names, map(float, p)))

    # Two-parameter log curve with C fixed at 1.0; slope and asymptote are
    # obtained by linear least squares, then constrained to preserve decrease.
    c = 1.0
    z = np.log((1 + c) / (x + c))
    a, b = np.linalg.lstsq(np.column_stack((np.ones_like(z), z)), y, rcond=None)[0]
    if b < 0 or a <= 0:
        b = max(0.0, float(b))
        a = max(1e-9, float(np.mean(y)))
    out['Log + asymptote (C=1)'] = a + b * z
    out_params['Log + asymptote (C=1)'] = {'A_inf_deg': float(a), 'B_deg': float(b), 'C': c}
    return out, out_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--peak-csv', type=Path, required=True)
    ap.add_argument('--spline-metrics-csv', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.peak_csv.open(encoding='utf-8-sig', newline='') as f:
        peak_rows = list(csv.DictReader(f))
    with args.spline_metrics_csv.open(encoding='utf-8-sig', newline='') as f:
        spline_rows = list(csv.DictReader(f))
    by_segment = {}
    for r in peak_rows:
        by_segment.setdefault(r['segment_id'], []).append(r)
    spline = {r['segment_id']: r for r in spline_rows}
    records = []
    for segment_id, points in sorted(by_segment.items()):
        points.sort(key=lambda r: int(r['peak_number']))
        t = np.asarray([float(r['peak_time_s']) for r in points])
        y = np.asarray([float(r['raw_amplitude_deg']) for r in points])
        x = (t - t[0]) / (t[-1] - t[0])
        fitted, parameters = candidates(x, y)
        for model, yhat in fitted.items():
            r = pearson_r(y, yhat)
            rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
            records.append(dict(segment_id=segment_id, axis=points[0]['axis'],
                configuration=points[0]['configuration'], model=model, peaks=len(y),
                pearson_r=r, rmse_deg=rmse, pass_R_gate=r >= R_MIN,
                pass_RMSE_gate=rmse <= RMSE_MAX_DEG,
                pass_both_gates=(r >= R_MIN and rmse <= RMSE_MAX_DEG),
                parameters=repr(parameters[model])))
        r = float(spline[segment_id]['pearson_r'])
        rmse = float(spline[segment_id]['amplitude_rmse_deg'])
        records.append(dict(segment_id=segment_id, axis=points[0]['axis'],
            configuration=points[0]['configuration'], model='GCV log-spline', peaks=len(y),
            pearson_r=r, rmse_deg=rmse, pass_R_gate=r >= R_MIN,
            pass_RMSE_gate=rmse <= RMSE_MAX_DEG,
            pass_both_gates=(r >= R_MIN and rmse <= RMSE_MAX_DEG), parameters=''))

    summary = []
    names = list(dict.fromkeys(r['model'] for r in records))
    for model in names:
        rr = [r for r in records if r['model'] == model]
        rs = np.asarray([float(r['pearson_r']) for r in rr])
        es = np.asarray([float(r['rmse_deg']) for r in rr])
        summary.append(dict(model=model, waveforms=len(rr), pearson_R_min=float(rs.min()),
            pearson_R_median=float(np.median(rs)), pearson_R_max=float(rs.max()),
            rmse_mean_deg=float(es.mean()), rmse_median_deg=float(np.median(es)),
            rmse_max_deg=float(es.max()), pass_R_count=int(np.sum(rs >= R_MIN)),
            pass_RMSE_count=int(np.sum(es <= RMSE_MAX_DEG)),
            pass_both_count=int(np.sum((rs >= R_MIN) & (es <= RMSE_MAX_DEG)))))
    for name, data in (("ihb03_monotone_envelope_waveforms.csv", records),
                       ("ihb03_monotone_envelope_summary.csv", summary)):
        with (args.output_dir / name).open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
