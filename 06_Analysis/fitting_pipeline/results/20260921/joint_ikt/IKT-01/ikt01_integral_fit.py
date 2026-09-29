#!/usr/bin/env python3
"""Reproduce IKT-01 weak/integral residual fits using only Python's standard library."""
import argparse
import csv
import math
from pathlib import Path

ROD_C = 2.5486754169187504e-6
EPSILON = 0.5 * math.pi / 180.0
SMOOTHING_WINDOWS = (11, 21, 31)
PHI_FORMS = ("sin", "parabolic")


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def solve(A, b):
    n = len(b)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot][k]) < 1e-20:
            return None
        M[k], M[pivot] = M[pivot], M[k]
        div = M[k][k]
        for j in range(k, n + 1):
            M[k][j] /= div
        for i in range(n):
            if i == k:
                continue
            div = M[i][k]
            for j in range(k, n + 1):
                M[i][j] -= div * M[k][j]
    return [M[i][n] for i in range(n)]


def local_cubic_derivative(t, y, width):
    """Derivative from a centered local cubic fit; edge windows are shifted inward."""
    n = len(t)
    half = width // 2
    out = [0.0] * n
    for i in range(n):
        lo = max(0, min(i - half, n - width))
        hi = min(n, lo + width)
        lo = max(0, hi - width)
        scale = max(t[hi - 1] - t[lo], 0.01)
        M = [[0.0] * 5 for _ in range(4)]
        for j in range(lo, hi):
            x = (t[j] - t[i]) / scale
            v = (1.0, x, x*x, x*x*x)
            for r in range(4):
                for c in range(4):
                    M[r][c] += v[r] * v[c]
                M[r][4] += v[r] * y[j]
        coef = solve([row[:4] for row in M], [row[4] for row in M])
        out[i] = coef[1] / scale
    return out


def trapz(values, times):
    return sum((values[i] + values[i+1]) * (times[i+1] - times[i]) / 2.0
               for i in range(len(values) - 1))


def phi_and_derivative(s, duration, form):
    if form == "sin":
        return math.sin(math.pi*s), math.pi/duration * math.cos(math.pi*s)
    return 4.0*s*(1.0-s), 4.0*(1.0-2.0*s)/duration


def eigenvalues_symmetric_3(M):
    A = [row[:] for row in M]
    for _ in range(200):
        p, q, largest = 0, 1, 0.0
        for i in range(3):
            for j in range(i+1, 3):
                if abs(A[i][j]) > largest:
                    p, q, largest = i, j, abs(A[i][j])
        if largest < 1e-13:
            break
        tau = (A[q][q] - A[p][p]) / (2.0*A[p][q])
        tangent = math.copysign(1.0, tau if tau else 1.0) / (abs(tau) + math.sqrt(1.0+tau*tau))
        c = 1.0 / math.sqrt(1.0+tangent*tangent)
        s = tangent*c
        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        A[p][p], A[q][q], A[p][q], A[q][p] = app-tangent*apq, aqq+tangent*apq, 0.0, 0.0
        for k in range(3):
            if k in (p, q):
                continue
            x, y = A[k][p], A[k][q]
            A[k][p] = A[p][k] = c*x-s*y
            A[k][q] = A[q][k] = s*x+c*y
    return sorted((A[i][i] for i in range(3)), reverse=True)


def nnls_scaled(features, target):
    """Enumerate active sets for three nonnegative coefficients after column scaling."""
    norms = [math.sqrt(sum(row[j]**2 for row in features)) for j in range(3)]
    if any(x <= 0.0 for x in norms):
        return None
    A = [[row[j]/norms[j] for j in range(3)] for row in features]
    best = None
    for mask in range(1, 8):
        active = [j for j in range(3) if mask & (1 << j)]
        gram = [[sum(row[i]*row[j] for row in A) for j in active] for i in active]
        rhs = [sum(row[i]*y for row, y in zip(A, target)) for i in active]
        z = solve(gram, rhs)
        if z is None or any(x < -1e-10 for x in z):
            continue
        x = [0.0, 0.0, 0.0]
        for j, value in zip(active, z):
            x[j] = max(0.0, value)
        err = sum((sum(row[j]*x[j] for j in range(3))-y)**2 for row, y in zip(A, target))
        if best is None or err < best[0]:
            best = (err, x, A, norms)
    if best is None:
        return None
    err, scaled_beta, A, norms = best
    beta = [scaled_beta[j]/norms[j] for j in range(3)]
    gram = [[sum(row[i]*row[j] for row in A) for j in range(3)] for i in range(3)]
    eig = eigenvalues_symmetric_3(gram)
    cond = math.sqrt(eig[0]/max(eig[-1], 1e-30))
    rank = sum(value > eig[0]*1e-10 for value in eig)
    rms_y = math.sqrt(sum(y*y for y in target)/len(target))
    relres = math.sqrt(err/len(target))/(rms_y or 1.0)
    return beta, cond, rank, relres


def make_windows(wave, peak_rows, t, theta, omega, phi_form):
    rows = []
    peaks = sorted(peak_rows, key=lambda x: int(x["peak_number"]))
    peaks = [p for p in peaks if 0 <= int(p["sample_index_in_segment"]) < len(t)]
    for p0, p1 in zip(peaks, peaks[1:]):
        if min(float(p0["amplitude_deg"]), float(p1["amplitude_deg"])) < 4.0:
            continue
        a, b = int(p0["sample_index_in_segment"]), int(p1["sample_index_in_segment"])
        if b-a < 12:
            continue
        mid = int(math.floor((a+b)/2.0 + 0.5))
        for part, lo, hi in ((1, a, mid), (2, mid, b)):
            t0, t1 = t[lo], t[hi]
            duration = t1-t0
            ts = t[lo:hi+1]
            arrays = ([], [], [], [])
            for i in range(lo, hi+1):
                phi, dphi = phi_and_derivative((t[i]-t0)/duration, duration, phi_form)
                arrays[0].append(-dphi*omega[i])
                arrays[1].append(phi*math.sin(theta[i]))
                arrays[2].append(phi*math.tanh(omega[i]/EPSILON))
                arrays[3].append(phi*abs(omega[i])*omega[i])
            X_I, X_k, X_tau, D = [trapz(v, ts) for v in arrays]
            rows.append({
                "segment": wave["segment_id"], "config": wave["configuration"],
                "axis": wave["axis"], "peak": int(p0["peak_number"]), "part": part,
                "amp0": float(p0["amplitude_deg"]), "amp1": float(p1["amplitude_deg"]),
                "t0": t0, "t1": t1, "XI": X_I, "Xk": X_k, "Xt": X_tau, "D": D,
            })
    return rows


def fit_rows(rows):
    if len(rows) < 3:
        return None
    features = [[r["XI"], r["Xk"], r["Xt"]] for r in rows]
    target = [-ROD_C*r["D"] for r in rows]
    fit = nnls_scaled(features, target)
    if fit is None:
        return None
    beta, cond, rank, relres = fit
    I, k, tau = beta
    return {"I": I, "k": k, "tau": tau, "kI": k/I if I else "",
            "tauI": tau/I if I else "", "condition": cond, "rank": rank,
            "relres": relres, "nwindows": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, outdir = Path(args.repo_root), Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = Path("06_Analysis/fitting_pipeline")
    selection = read_csv(root/prefix/"results/20260921/waveform_review/waveform_selection.csv")
    peaks = read_csv(root/prefix/"results/20260921/hybrid_identification/01_preprocessing/turning_points.csv")
    prep = {r["segment_id"]: r for r in read_csv(root/prefix/"results/20260921/hybrid_identification/01_preprocessing/waveform_preprocessing.csv")}
    accepted = [r for r in selection if r["configuration"] != "BALL" and r["use_for_fitting"] == "1"]
    peak_groups = {}
    for row in peaks:
        peak_groups.setdefault(row["segment_id"], []).append(row)
    output, interval_output = [], []
    for wave in accepted:
        segment = wave["segment_id"]
        baseline = float(prep[segment]["envelope_center_deg"])
        start, end = int(wave["start_index"]), int(wave["end_index"])
        raw_path = root/"04_Data/05_Fitting/20260921"/wave["data_file"]
        raw = read_csv(raw_path)[start:end+1]
        t = [float(r["systime[ms]"])/1000.0 for r in raw]
        theta = [(float(r[wave["angle_column"]])-baseline)*math.pi/180.0 for r in raw]
        variants = {}
        for width in SMOOTHING_WINDOWS:
            omega = local_cubic_derivative(t, theta, width)
            for form in (PHI_FORMS if width == 21 else ("sin",)):
                rows = make_windows(wave, peak_groups.get(segment, []), t, theta, omega, form)
                fitted = fit_rows(rows)
                variants[(width, form)] = (fitted, rows)
        main_fit, main_rows = variants[(21, "sin")]
        row = {"segment": segment, "config": wave["configuration"], "axis": wave["axis"],
               "direction": wave["direction"], "rep": wave["repetition"],
               "npoints": len(t), "baseline_deg": baseline}
        for width in SMOOTHING_WINDOWS:
            fit, _ = variants[(width, "sin")]
            for key in ("nwindows", "I", "k", "tau", "kI", "tauI", "condition", "rank", "relres"):
                row[f"{key}_{width}"] = fit.get(key, "") if fit else ""
        fit, _ = variants[(21, "parabolic")]
        for key in ("I", "k", "tau", "condition", "rank", "relres"):
            row[f"{key}_phi_parabolic"] = fit.get(key, "") if fit else ""
        output.append(row)
        interval_output.extend(main_rows)
    for name, rows in (("ikt01_waveform_fits.csv", output),
                       ("ikt01_integral_windows.csv", interval_output)):
        fields = sorted({key for row in rows for key in row})
        with open(outdir/name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"waveforms={len(output)} windows={len(interval_output)}")


if __name__ == "__main__":
    main()
