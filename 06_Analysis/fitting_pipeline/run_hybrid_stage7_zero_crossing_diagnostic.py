"""Stage 7 diagnostic: measure cumulative phase drift at equilibrium crossings.

This diagnostic keeps all Stage 6 coefficients fixed. It pairs measured and
simulated zero-angle crossings in chronological order and fits the anchored
time drift y = k*x for each waveform.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from run_hybrid_stage6_validation import HERE, input_paths, load_data, selected_waveforms
from run_hybrid_stage6_validation import configuration_coefficients
from run_hybrid_rod_damping_identification import DEFAULT_ANGLE_SPEED_ATOL
from run_hybrid_rod_damping_identification import DEFAULT_RTOL
from run_hybrid_rod_damping_identification import FRICTION_EPSILON_DEG_S
from run_hybrid_rod_damping_identification import MAX_STEP_PERIOD_FRACTION


def crossing_times(times, values, root_function=None):
    """Return linearly interpolated or root-solved sign-change times."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    roots = []
    for i in range(len(times) - 1):
        left, right = values[i], values[i + 1]
        if left == 0.0:
            root = float(times[i])
        elif right == 0.0:
            root = float(times[i + 1])
        elif left * right < 0.0:
            if root_function is None:
                fraction = -left / (right - left)
                root = float(times[i] + fraction * (times[i + 1] - times[i]))
            else:
                root = float(brentq(root_function, float(times[i]), float(times[i + 1]), xtol=1e-12))
        else:
            continue
        if not roots or root - roots[-1] > 1e-8:
            roots.append(root)
    return np.asarray(roots, dtype=float)


def simulate_and_find_crossings(record, coefficients):
    """Integrate the fixed Stage 6 model and return measured/model crossings."""
    first_peak_time = float(record["peak_times_s"][0])
    last_peak_time = float(record["peak_times_s"][-1])
    sample_time = np.asarray(record["time_s"], dtype=float)
    sample_angle = np.asarray(record["centered_angle_deg"], dtype=float)
    mask = (sample_time >= first_peak_time) & (sample_time <= last_peak_time)
    measured_time = sample_time[mask] - first_peak_time
    measured_angle = sample_angle[mask]
    if len(measured_time) < 2:
        raise ValueError(record["segment_id"] + ": insufficient selected samples")

    inertia = coefficients["inertia_kg_m2"]
    restoring = coefficients["restoring_n_m_per_rad"]
    period = 2.0 * math.pi * math.sqrt(inertia / restoring)
    epsilon = math.radians(FRICTION_EPSILON_DEG_S)

    def rhs(unused_time, state):
        angle, speed = state
        acceleration = -restoring * math.sin(angle)
        acceleration -= coefficients["b_n_m_s_per_rad"] * speed
        acceleration -= coefficients["c_n_m_s2_per_rad2"] * abs(speed) * speed
        acceleration -= coefficients["tau_n_m"] * math.tanh(speed / epsilon)
        return [speed, acceleration / inertia]

    initial_angle = math.radians(float(record["peak_angles_deg"][0]))
    solution = solve_ivp(
        rhs,
        (0.0, float(measured_time[-1])),
        [initial_angle, 0.0],
        method="DOP853",
        t_eval=measured_time,
        dense_output=True,
        rtol=DEFAULT_RTOL,
        atol=[DEFAULT_ANGLE_SPEED_ATOL, DEFAULT_ANGLE_SPEED_ATOL],
        max_step=period * MAX_STEP_PERIOD_FRACTION,
    )
    if not solution.success or len(solution.t) != len(measured_time):
        raise RuntimeError(record["segment_id"] + ": continuous integration failed")

    measured_crossings = crossing_times(measured_time, measured_angle)
    model_angles = solution.sol(measured_time)[0]
    model_crossings = crossing_times(
        measured_time,
        model_angles,
        root_function=lambda t: float(solution.sol(t)[0]),
    )
    return measured_crossings, model_crossings


def run(result_root, data_root, date):
    paths = input_paths(result_root, data_root, date)
    turning, waveform_summary, parameters, selection, stage4, stage5 = load_data(paths)
    records, _, _ = selected_waveforms(turning, waveform_summary, selection, paths["data_root"])

    crossing_rows, waveform_rows = [], []
    for record in records:
        coefficients = configuration_coefficients(
            record["axis"], record["configuration"], parameters, stage4, stage5
        )
        measured, predicted = simulate_and_find_crossings(record, coefficients)
        measured_count, predicted_count = len(measured), len(predicted)
        paired_count = min(measured_count, predicted_count)
        if paired_count < 3:
            raise ValueError(f"{record['segment_id']}: fewer than three paired zero crossings")

        measured = measured[:paired_count]
        predicted = predicted[:paired_count]
        raw_error = predicted - measured  # positive means model crossing is late
        elapsed = measured - measured[0]
        drift = raw_error - raw_error[0]  # remove constant initial time offset
        denom = float(np.dot(elapsed, elapsed))
        k = float(np.dot(elapsed, drift) / denom)
        fitted = k * elapsed
        sse = float(np.sum((drift - fitted) ** 2))
        sst = float(np.sum((drift - drift.mean()) ** 2))
        r_squared = 1.0 - sse / sst if sst > 0.0 else float("nan")
        intercept_s, slope_s = np.polyfit(elapsed, drift, 1)[1], np.polyfit(elapsed, drift, 1)[0]

        base = {
            "segment_id": record["segment_id"],
            "axis": record["axis"],
            "configuration": record["configuration"],
            "direction": record["direction"],
            "repetition": record["repetition"],
        }
        waveform_rows.append({
            **base,
            "measured_crossings": measured_count,
            "predicted_crossings": predicted_count,
            "paired_crossings": paired_count,
            "unpaired_measured_crossings": max(0, measured_count - predicted_count),
            "unpaired_predicted_crossings": max(0, predicted_count - measured_count),
            "initial_crossing_offset_s": float(raw_error[0]),
            "k_slope_s_per_s": k,
            "k_slope_ms_per_s": 1000.0 * k,
            "anchored_fit_r_squared": r_squared,
            "free_intercept_s": float(intercept_s),
            "free_intercept_slope_s_per_s": float(slope_s),
            "fit_duration_s": float(elapsed[-1]),
        })
        for index, (tm, tp, x, y, yhat) in enumerate(zip(measured, predicted, elapsed, drift, fitted), 1):
            crossing_rows.append({
                **base,
                "crossing_index": index,
                "measured_crossing_time_s_from_initial_peak": float(tm),
                "predicted_crossing_time_s_from_initial_peak": float(tp),
                "elapsed_time_x_s": float(x),
                "raw_time_error_predicted_minus_measured_s": float(tp - tm),
                "anchored_drift_y_s": float(y),
                "fitted_drift_kx_s": float(yhat),
            })

    out = Path(result_root) / date / "hybrid_identification" / "07_phase_drift_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    crossing_df = pd.DataFrame(crossing_rows)
    waveform_df = pd.DataFrame(waveform_rows)
    crossing_df.to_csv(out / "zero_crossing_residuals.csv", index=False, float_format="%.10g")
    waveform_df.to_csv(out / "waveform_phase_slopes.csv", index=False, float_format="%.10g")

    summary = waveform_df.groupby(["axis", "configuration"], sort=True).agg(
        waveform_count=("segment_id", "count"),
        mean_k_ms_per_s=("k_slope_ms_per_s", "mean"),
        std_k_ms_per_s=("k_slope_ms_per_s", "std"),
        median_k_ms_per_s=("k_slope_ms_per_s", "median"),
        mean_r_squared=("anchored_fit_r_squared", "mean"),
        mean_crossings=("paired_crossings", "mean"),
    ).reset_index()
    summary.to_csv(out / "phase_slope_by_configuration.csv", index=False, float_format="%.10g")

    configs = [c for c in ["SP00", "SP01", "SP02", "SP03", "SP04", "BALL"] if c in set(waveform_df.configuration)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, axis_name in zip(axes, ["IN", "OUT"]):
        subset = waveform_df[waveform_df.axis == axis_name]
        for j, config in enumerate(configs):
            values = subset.loc[subset.configuration == config, "k_slope_ms_per_s"].to_numpy()
            if len(values):
                jitter = np.linspace(-0.11, 0.11, len(values)) if len(values) > 1 else np.array([0.0])
                ax.scatter(j + jitter, values, s=25, alpha=0.65, color="#4c78a8")
                avg = float(np.mean(values))
                ax.errorbar(j, avg, yerr=np.std(values, ddof=1) if len(values) > 1 else 0.0,
                            fmt="D", color="#d62728", capsize=4, markersize=6, zorder=3)
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
        ax.set_xticks(range(len(configs)), configs)
        ax.set_title(axis_name)
        ax.set_xlabel("configuration")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("phase drift slope k [ms/s]\npositive: model crossing is increasingly late")
    fig.suptitle("Stage 7 diagnostic — zero-crossing phase drift by configuration\nblue: individual waveform; red: mean ± SD")
    fig.tight_layout()
    fig.savefig(out / "phase_slope_by_configuration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Stage 7 診断: x軸交点における位相ズレ",
        "",
        "## 方法",
        "",
        f"Stage 6の固定係数を変更せず、承認済み{len(records)}波形について平衡角（x軸）との交点時刻を測定値・連続ODEの両方から取得した。実測交点はサンプル間を線形補間し、ODE交点は連続解の根として求めた。交点は時系列順に対応づけた。",
        "",
        "各波形の交差時刻誤差を `e_n = t_model,n - t_measured,n` と定義する（正ならモデルが遅い）。一定の時刻原点差を除くため、最初の対応交点からの経過 `x_n` と誤差変化 `y_n = e_n - e_0` を用い、原点固定で `y = kx` を最小二乗した。従って `k > 0` はモデルの遅れが時間とともに増えることを示す。",
        "",
        "係数は波形ごとにStage 6の軸・形態別割当てをそのまま使用し、フィットし直していない。交点数が異なる場合は共通する先頭から対応づけ、余った交点数を波形別CSVに記録した。",
        "",
        "## 傾きの形態別プロット",
        "",
        "![形態ごとの位相ドリフト傾き](phase_slope_by_configuration.png)",
        "",
        "## 形態別集計",
        "",
        "| 軸 | 形態 | 波形数 | k平均 [ms/s] | k標準偏差 [ms/s] | k中央値 [ms/s] | 平均R² | 平均対応交点数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        report.append(
            f"| {row.axis} | {row.configuration} | {int(row.waveform_count)} | {row.mean_k_ms_per_s:.4f} | {row.std_k_ms_per_s:.4f} | {row.median_k_ms_per_s:.4f} | {row.mean_r_squared:.3f} | {row.mean_crossings:.1f} |"
        )
    report += [
        "",
        "## 解釈上の注意",
        "",
        "`k`の形態差は周期ずれの累積率を示すが、それだけで `I`、`K`、摩擦、球後流のいずれが原因かは特定できない。波形ごとの線形適合度、初期交点オフセット、方向差、残差の時間変化と併せて次の判断材料とする。",
        "",
        "## 出力",
        "",
        "- [交点ごとの誤差とフィット値](zero_crossing_residuals.csv)",
        "- [波形ごとのkと診断値](waveform_phase_slopes.csv)",
        "- [軸・形態別集計](phase_slope_by_configuration.csv)",
        "",
    ]
    (out / "STAGE7_PHASE_DRIFT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    settings = {
        "stage": 7,
        "diagnostic_only": True,
        "coefficients_changed": False,
        "waveform_count": len(records),
        "phase_error_definition": "e = t_model - t_measured; y=e-e_first; x=t_measured-t_measured_first; fit y=k*x through origin",
        "measured_crossing_method": "linear interpolation between adjacent logger samples",
        "model_crossing_method": "root solve on continuous ODE dense output",
        "coefficient_source": "Stage 6 axis and configuration assignments, unchanged",
        "input_paths": {key: str(value) for key, value in paths.items()},
    }
    (out / "stage7_phase_drift_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 7 phase-drift diagnostic complete: {len(records)} waveforms")
    print(summary.to_string(index=False))
    print(f"Outputs: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=HERE / "results")
    parser.add_argument("--data-root", type=Path, default=HERE.parents[1] / "04_Data" / "05_Fitting")
    parser.add_argument("--date", default="20260921")
    args = parser.parse_args()
    run(args.result_root, args.data_root, args.date)
