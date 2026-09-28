"""Stage 6: 同定済み係数を固定した全自由減衰波形の連続検証。"""

import argparse
import hashlib
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

from run_hybrid_rod_damping_identification import AIR_DENSITY_KG_M3
from run_hybrid_rod_damping_identification import AIR_DYNAMIC_VISCOSITY_PA_S
from run_hybrid_rod_damping_identification import DEFAULT_ANGLE_SPEED_ATOL
from run_hybrid_rod_damping_identification import DEFAULT_RTOL
from run_hybrid_rod_damping_identification import FRICTION_EPSILON_DEG_S
from run_hybrid_rod_damping_identification import MAX_STEP_PERIOD_FRACTION
from run_hybrid_rod_damping_identification import MINIMUM_AMPLITUDE_DEG
from run_hybrid_rod_damping_identification import WAVEFORM_OVERVIEW_COLUMNS


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DATE = "20260921"
CSV_FLOAT_FORMAT = "%.10g"
OVERVIEW_DPI = 120
OVERVIEW_ROW_HEIGHT_IN = 2.25
OVERVIEW_COLUMN_WIDTH_IN = 4.3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths(result_root, data_root, date):
    result = Path(result_root) / date
    stage = result / "hybrid_identification"
    return {
        "turning": stage / "01_preprocessing" / "turning_points.csv",
        "waveforms": stage / "01_preprocessing" / "waveform_preprocessing.csv",
        "parameters": stage / "02_frequency_identification" / "identified_inertia_restoring.csv",
        "selection": result / "waveform_review" / "waveform_selection.csv",
        "stage4": stage / "04_rod_damping_identification" / "stage4_settings.json",
        "stage5": stage / "05_sphere_drag_identification" / "stage5_settings.json",
        "stage5_report": stage / "05_sphere_drag_identification" / "interval_predictions.csv",
        "data_root": Path(data_root) / date,
    }


def load_data(paths):
    required = [paths[k] for k in ["turning", "waveforms", "parameters", "selection", "stage4", "stage5", "stage5_report"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Stage 6 inputs missing:\n" + "\n".join(missing))
    turning = pd.read_csv(paths["turning"])
    waveforms = pd.read_csv(paths["waveforms"])
    parameters = pd.read_csv(paths["parameters"])
    selection = pd.read_csv(paths["selection"], encoding="utf-8-sig")
    stage4 = json.loads(paths["stage4"].read_text(encoding="utf-8"))
    stage5 = json.loads(paths["stage5"].read_text(encoding="utf-8"))
    return turning, waveforms, parameters, selection, stage4, stage5


def selected_waveforms(turning, waveform_summary, selection, date_directory):
    needed = {"segment_id", "data_file", "axis", "angle_column", "configuration", "direction", "repetition", "start_index", "end_index", "use_for_fitting", "review_status"}
    missing = sorted(needed - set(selection.columns))
    if missing:
        raise ValueError("waveform selection columns missing: " + ", ".join(missing))
    if not (selection["review_status"].astype(str).str.upper() == "APPROVED").all():
        raise ValueError("waveform selection includes an unapproved review row")
    selected = selection[(pd.to_numeric(selection["use_for_fitting"], errors="raise") == 1)].copy()
    summary = waveform_summary.set_index("segment_id")
    turning_groups = {str(k): g.sort_values("peak_number").reset_index(drop=True) for k, g in turning.groupby("segment_id", sort=False)}
    cache, records, input_files, peak_sets = {}, [], set(), {}
    for _, row in selected.iterrows():
        segment_id = str(row["segment_id"])
        if segment_id not in turning_groups or segment_id not in summary.index:
            raise ValueError(f"{segment_id}: Stage 1 summary/turning data missing")
        file_path = Path(date_directory) / str(row["data_file"])
        if file_path not in cache:
            if not file_path.exists():
                raise FileNotFoundError("raw logger file missing: " + str(file_path))
            cache[file_path] = pd.read_csv(file_path)
            input_files.add(file_path)
        data = cache[file_path]
        angle_column = str(row["angle_column"])
        if "systime[ms]" not in data or angle_column not in data:
            raise ValueError(f"{file_path}: required time/angle column missing")
        time_ms = pd.to_numeric(data["systime[ms]"], errors="coerce").to_numpy(dtype=float)
        angle_deg = pd.to_numeric(data[angle_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(time_ms) & np.isfinite(angle_deg)
        time_s = time_ms[valid] / 1000.0
        angle_deg = (angle_deg[valid] + 180.0) % 360.0 - 180.0
        start, end = int(row["start_index"]), int(row["end_index"]) + 1
        center = float(summary.loc[segment_id, "envelope_center_deg"])
        waveform = {
            "segment_id": segment_id,
            "axis": str(row["axis"]).strip().upper(),
            "configuration": str(row["configuration"]).strip().upper(),
            "direction": str(row["direction"]).strip().upper(),
            "repetition": int(row["repetition"]),
            "time_s": time_s[start:end],
            "centered_angle_deg": angle_deg[start:end] - center,
        }
        group = turning_groups[segment_id]
        initial = np.flatnonzero(group["is_initial_peak"].to_numpy(dtype=int) == 1)
        if len(initial) != 1:
            raise ValueError(f"{segment_id}: initial peak is not unique")
        group = group.iloc[int(initial[0]):].reset_index(drop=True)
        eligible = pd.to_numeric(group["eligible_for_later_stages"], errors="raise").to_numpy(dtype=int) == 1
        amplitude = pd.to_numeric(group["amplitude_deg"], errors="raise").to_numpy(dtype=float)
        accepted_pairs = [i for i in range(len(group)-1) if eligible[i] and eligible[i+1] and min(amplitude[i], amplitude[i+1]) >= MINIMUM_AMPLITUDE_DEG]
        if not accepted_pairs:
            raise ValueError(f"{segment_id}: fewer than two eligible peaks")
        first_peak_index, last_peak_index = accepted_pairs[0], accepted_pairs[-1] + 1
        peaks = group.iloc[first_peak_index:last_peak_index+1].copy()
        if len(peaks) < 2:
            raise ValueError(f"{segment_id}: fewer than two peaks in accepted interval range")
        if not (np.diff(peaks["peak_time_s"].to_numpy(dtype=float)) > 0.0).all():
            raise ValueError(f"{segment_id}: peak times are not increasing")
        waveform["peak_times_s"] = peaks["peak_time_s"].to_numpy(dtype=float)
        waveform["peak_angles_deg"] = peaks["centered_peak_angle_deg"].to_numpy(dtype=float)
        waveform["eligible_half_cycles"] = len(accepted_pairs)
        records.append(waveform)
        peak_sets[segment_id] = peaks
    if not records:
        raise ValueError("No approved waveforms were selected")
    return records, input_files, peak_sets


def configuration_coefficients(axis, configuration, parameter_table, stage4, stage5):
    row = parameter_table.set_index(["axis", "configuration"]).loc[(axis, configuration)]
    stage4_model = stage4["models_for_review"]["theoretical_c"]
    c_rod = float(stage4_model["c_rod"])
    if configuration == "BALL":
        c = float(stage5["sphere_c_n_m_s2_per_rad2"] + stage5["rod_c_n_m_s2_per_rad2_ball"])
        c_basis = "c_sphere + c_rod_ball"
    else:
        c = c_rod
        c_basis = "c_rod"
    return {
        "inertia_kg_m2": float(row["inertia_kg_m2"]),
        "restoring_n_m_per_rad": float(row["restoring_n_m_per_rad"]),
        "b_n_m_s_per_rad": 0.0,
        "c_n_m_s2_per_rad2": c,
        "c_basis": c_basis,
        "tau_n_m": float(stage4_model["tau_" + axis]),
    }


def simulate_continuous(record, coefficients):
    first_time = float(record["peak_times_s"][0])
    last_time = float(record["peak_times_s"][-1])
    sample_time = np.asarray(record["time_s"], dtype=float)
    sample_angle = np.asarray(record["centered_angle_deg"], dtype=float)
    mask = (sample_time >= first_time) & (sample_time <= last_time)
    measured_time = sample_time[mask]
    measured_angle = sample_angle[mask]
    if len(measured_time) < 2:
        raise ValueError(record["segment_id"] + ": insufficient samples in selected peak span")
    relative_time = measured_time - first_time
    initial_angle = math.radians(float(record["peak_angles_deg"][0]))
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

    solution = solve_ivp(
        rhs,
        (0.0, float(relative_time[-1])),
        [initial_angle, 0.0],
        method="DOP853",
        t_eval=relative_time,
        dense_output=True,
        rtol=DEFAULT_RTOL,
        atol=[DEFAULT_ANGLE_SPEED_ATOL, DEFAULT_ANGLE_SPEED_ATOL],
        max_step=period * MAX_STEP_PERIOD_FRACTION,
    )
    if not solution.success or len(solution.t) != len(relative_time):
        raise RuntimeError(record["segment_id"] + ": continuous integration failed")
    predicted = np.rad2deg(solution.y[0])
    residual = predicted - measured_angle
    # Locate both positive and negative turning points from the dense continuous
    # state solution. A fixed event direction would silently miss alternate peaks.
    scan_times = np.r_[0.0, relative_time]
    scan_speed = solution.sol(scan_times)[1]
    event_times = [0.0]
    for left_t, right_t, left_v, right_v in zip(scan_times[:-1], scan_times[1:], scan_speed[:-1], scan_speed[1:]):
        if left_t <= 1e-9 or left_v * right_v >= 0.0:
            continue
        root = brentq(lambda t: float(solution.sol(t)[1]), float(left_t), float(right_t), xtol=1e-12)
        if root - event_times[-1] > 1e-6:
            event_times.append(root)
    event_times = np.asarray(event_times, dtype=float)
    event_angles = np.rad2deg(solution.sol(event_times)[0])
    measured_periods = np.diff(record["peak_times_s"])
    predicted_periods = np.diff(np.r_[0.0, event_times])
    count = min(len(measured_periods), len(predicted_periods))
    measured_peak_amp = np.abs(record["peak_angles_deg"][1 : 1 + count])
    predicted_peak_amp = np.abs(event_angles[:count])
    period_error = predicted_periods[:count] - measured_periods[:count]
    amplitude_error = predicted_peak_amp - measured_peak_amp
    return {
        "time_s": relative_time,
        "measured_angle_deg": measured_angle,
        "predicted_angle_deg": predicted,
        "sample_rmse_deg": float(np.sqrt(np.mean(residual**2))),
        "sample_mae_deg": float(np.mean(np.abs(residual))),
        "sample_max_abs_error_deg": float(np.max(np.abs(residual))),
        "endpoint_error_deg": float(residual[-1]),
        "measured_peak_count": len(record["peak_times_s"]),
        "predicted_peak_count": len(event_times) + 1,
        "matched_half_period_count": count,
        "half_period_rmse_s": float(np.sqrt(np.mean(period_error**2))) if count else float("nan"),
        "peak_envelope_rmse_deg": float(np.sqrt(np.mean(amplitude_error**2))) if count else float("nan"),
        "peak_amplitude_bias_deg": float(np.mean(amplitude_error)) if count else float("nan"),
        "matched_periods_s": measured_periods[:count],
        "predicted_periods_s": predicted_periods[:count],
        "matched_amplitude_errors_deg": amplitude_error,
    }


def make_overview(records, comparisons, output_path):
    records = sorted(records, key=lambda row: (row["axis"], row["configuration"], row["direction"], row["repetition"]))
    columns = WAVEFORM_OVERVIEW_COLUMNS
    rows = int(math.ceil(len(records) / columns))
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(columns * OVERVIEW_COLUMN_WIDTH_IN, rows * OVERVIEW_ROW_HEIGHT_IN),
        squeeze=False,
    )
    for index, record in enumerate(records):
        axis = axes.flat[index]
        comp = comparisons[record["segment_id"]]
        axis.plot(comp["time_s"], comp["measured_angle_deg"], color="0.45", lw=0.55, label="measured" if index == 0 else None)
        axis.plot(comp["time_s"], comp["predicted_angle_deg"], color="#d55e00", lw=0.8, alpha=0.9, label="fixed-coefficient model" if index == 0 else None)
        axis.set_title(f"{record['segment_id']}  RMSE {comp['sample_rmse_deg']:.1f}°", fontsize=7)
        axis.grid(True, alpha=0.18)
        axis.tick_params(labelsize=6)
        if index // columns == rows - 1:
            axis.set_xlabel("time from first eligible peak [s]", fontsize=7)
        if index % columns == 0:
            axis.set_ylabel("centered angle [deg]", fontsize=7)
    for axis in axes.flat[len(records):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.997), ncol=2, frameon=False)
    figure.suptitle("Stage 6 — continuous validation of all approved waveforms", y=0.999, fontsize=15)
    figure.tight_layout(rect=(0.01, 0.01, 1.0, 0.989))
    figure.savefig(output_path, dpi=OVERVIEW_DPI, bbox_inches="tight")
    plt.close(figure)


def report_text(metrics, total_waveforms, interval_count, coefficient_table):
    overall = metrics[["sample_rmse_deg", "peak_envelope_rmse_deg", "half_period_rmse_s"]]
    sample_rmse = float(np.sqrt(np.mean(overall.sample_rmse_deg**2)))
    envelope_rmse = float(np.sqrt(np.mean(overall.peak_envelope_rmse_deg**2)))
    period_rmse_ms = float(1000.0 * np.sqrt(np.mean(overall.half_period_rmse_s**2)))
    rows = []
    for keys, group in metrics.groupby(["axis", "configuration"], sort=True):
        rows.append(
            f"| {keys[0]} | {keys[1]} | {len(group)} | {np.sqrt(np.mean(group.sample_rmse_deg**2)):.3f} | {group.sample_rmse_deg.median():.3f} | {group.sample_rmse_deg.max():.3f} | {np.sqrt(np.mean(group.peak_envelope_rmse_deg**2)):.3f} | {1000*np.sqrt(np.mean(group.half_period_rmse_s**2)):.2f} |"
        )
    return "\n".join([
        "# Stage 6: 同定済み係数による全波形連続検証",
        "",
        "## 結果",
        "",
        f"承認済み{total_waveforms}波形、{interval_count}半周期を対象に、係数を再フィットせず連続積分した。I・Kと二乗抗力係数cは形態ごと、τは軸ごとにStage 2/4/5の同定値を割り当てた。個別波形図は作成せず、すべての波形を一枚の概要図へ重ねている。",
        "",
        f"- 波形等重みサンプルRMSE：{np.sqrt(np.mean(overall.sample_rmse_deg**2)):.3f}°（波形RMSEの中央値 {overall.sample_rmse_deg.median():.3f}°、最大 {overall.sample_rmse_deg.max():.3f}°）",
        f"- 波形等重み頂点包絡線RMSE：{np.sqrt(np.mean(overall.peak_envelope_rmse_deg**2)):.3f}°（対応できた連続折返し頂点、状態リセットなし）",
        f"- 波形等重み半周期誤差RMSE：{1000*np.sqrt(np.mean(overall.half_period_rmse_s**2)):.2f} ms（対応頂点間）",
        "",
        "![全採用波形の連続検証概要](all_waveforms_overview.png)",
        "",
        "## 軸・形態別の誤差",
        "",
        "| 軸 | 形態 | 波形数 | 波形等重みサンプルRMSE [deg] | 中央値 [deg] | 最大 [deg] | 波形等重み頂点RMSE [deg] | 波形等重み半周期RMSE [ms] |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## 係数の割当てと検証条件",
        "",
        "| 軸 | 形態 | I [kg m²] | K [N m/rad] | b [N m s/rad] | c [N m s²/rad²] | τ [N m] |",
        "|---|---|---:|---:|---:|---:|---:|",
        *coefficient_table,
        "",
        "- `b_IN=b_OUT=0`、`tau_IN/tau_OUT`はStage 4代表値、球の係数はStage 5推定値で固定。波形ごとの再フィットなし。",
        "- `SP00–SP04`: IN/OUTそれぞれのStage 2 `I,K`、Stage 4の理論ロッド`c_rod`。",
        "- `BALL`: Stage 2の物理加算`I,K`、`c_sphere + c_rod_ball`（Stage 5幾何補正済み）。",
        "- 開始状態は最初の採用頂点の実測角度・ゼロ角速度。以降の実測頂点へ戻さず、最後の採用頂点まで連続積分。角度中心はStage 1確定値。",
        "- RMSEは波形ごとに算出して同じ重みで平均。半周期RMSEはODE折返しイベントとStage 1実測頂点を時系列順に対応させた。",
        f"- Stage 1採用範囲の{interval_count}半周期のうち、ODE折返し頂点と実測頂点を{int(metrics.matched_half_period_count.sum())}半周期で対応できた。残りは各記録窓の末端でモデルと実測の頂点数が一致しなかった分であり、波形サンプルRMSEにはその位相差も含めている。",
        "",
        f"頂点包絡線RMSE（{envelope_rmse:.3f}°）に比べてサンプル波形RMSE（{sample_rmse:.3f}°）が大きく、半周期誤差（{period_rmse_ms:.1f} ms）も蓄積している。これは各折返しの減衰量は概ね追う一方、周期・位相の小さなずれが連続時間とともに蓄積することを示す。指定どおり係数は再調整していないため、このStageでは「局所的な半周期再現」と「長時間の連続予測性能」を別の評価結果として残す。",
        "",
        "Stage 6は固定係数の予測性能評価であり、残差を小さくするために`I,K,b,c,tau`や初期条件を調整していない。大きな連続波形誤差は、係数同定ではなく予測・位相誤差としてそのまま報告する。",
        "",
        "## 成果物",
        "",
        "- [全波形指標](continuous_waveform_metrics.csv)",
        "- [折返し頂点・半周期比較](peak_period_metrics.csv)",
        "- [実行設定・係数対応](stage6_settings.json)",
        "",
    ])


def run(result_root, data_root, date):
    paths = input_paths(result_root, data_root, date)
    turning, waveform_summary, parameters, selection, stage4, stage5 = load_data(paths)
    records, data_files, peak_sets = selected_waveforms(turning, waveform_summary, selection, paths["data_root"])
    comparisons, metric_rows, peak_rows, interval_count = {}, [], [], 0
    for record in records:
        coefficients = configuration_coefficients(record["axis"], record["configuration"], parameters, stage4, stage5)
        comparison = simulate_continuous(record, coefficients)
        comparisons[record["segment_id"]] = comparison
        interval_count += record["eligible_half_cycles"]
        metric_rows.append({
            "segment_id": record["segment_id"], "axis": record["axis"], "configuration": record["configuration"],
            "direction": record["direction"], "repetition": record["repetition"],
            "samples": len(comparison["time_s"]), "duration_s": float(comparison["time_s"][-1]),
            **{key: comparison[key] for key in ["sample_rmse_deg", "sample_mae_deg", "sample_max_abs_error_deg", "endpoint_error_deg", "measured_peak_count", "predicted_peak_count", "matched_half_period_count", "half_period_rmse_s", "peak_envelope_rmse_deg", "peak_amplitude_bias_deg"]},
            **coefficients,
        })
        for half_index, (measured_t, predicted_t, amplitude_residual) in enumerate(zip(comparison["matched_periods_s"], comparison["predicted_periods_s"], comparison["matched_amplitude_errors_deg"]), start=1):
            peak_rows.append({"segment_id": record["segment_id"], "axis": record["axis"], "configuration": record["configuration"], "half_cycle": half_index, "measured_half_period_s": float(measured_t), "predicted_half_period_s": float(predicted_t), "half_period_residual_s": float(predicted_t-measured_t), "amplitude_residual_deg": float(amplitude_residual)})

    out = Path(result_root) / date / "hybrid_identification" / "06_all_waveform_validation"
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["axis", "configuration", "direction", "repetition"])
    peaks = pd.DataFrame(peak_rows)
    metrics.to_csv(out / "continuous_waveform_metrics.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    obsolete_samples = out / "continuous_waveform_samples.csv"
    if obsolete_samples.exists():
        obsolete_samples.unlink()
    peaks.to_csv(out / "peak_period_metrics.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    make_overview(records, comparisons, out / "all_waveforms_overview.png")
    coefficients = {}
    coefficient_table = []
    for axis in ["IN", "OUT"]:
        for configuration in ["SP00", "SP01", "SP02", "SP03", "SP04", "BALL"]:
            value = configuration_coefficients(axis, configuration, parameters, stage4, stage5)
            coefficients[f"{axis}_{configuration}"] = value
            if ((metrics.axis == axis) & (metrics.configuration == configuration)).any():
                coefficient_table.append(
                    f"| {axis} | {configuration} | {value['inertia_kg_m2']:.8e} | {value['restoring_n_m_per_rad']:.8e} | {value['b_n_m_s_per_rad']:.3e} | {value['c_n_m_s2_per_rad2']:.8e} | {value['tau_n_m']:.8e} |"
                )
    (out / "STAGE6_VALIDATION_REPORT.md").write_text(report_text(metrics, len(records), interval_count, coefficient_table), encoding="utf-8")
    settings = {
        "stage": 6, "date": date, "waveform_count": len(records), "waveform_interval_count": interval_count,
        "matched_half_cycle_count": int(metrics.matched_half_period_count.sum()),
        "predicted_half_cycle_count": int((metrics.predicted_peak_count - 1).sum()),
        "fit_parameters_changed": False, "waveform_plots": "one all-waveform summary figure; no individual figure files",
        "configuration_coefficients": coefficients,
        "fixed_model": "I*theta_ddot + K*sin(theta) + b*theta_dot + c*abs(theta_dot)*theta_dot + tau*tanh(theta_dot/epsilon) = 0",
        "initial_conditions": "first eligible peak angle relative to stage1 envelope center, zero angular velocity; no resets at later measured peaks",
        "minimum_peak_amplitude_deg": MINIMUM_AMPLITUDE_DEG, "friction_epsilon_deg_s": FRICTION_EPSILON_DEG_S,
        "ode_rtol": DEFAULT_RTOL, "angle_speed_atol": DEFAULT_ANGLE_SPEED_ATOL, "max_step_period_fraction": MAX_STEP_PERIOD_FRACTION,
        "input_sha256": {str(path.resolve().relative_to(REPOSITORY_ROOT.resolve())): sha256(path) for path in [paths[k] for k in ["turning", "waveforms", "parameters", "selection", "stage4", "stage5", "stage5_report"]] + list(data_files)},
    }
    (out / "stage6_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="Stage 6 full-waveform fixed-coefficient validation")
    parser.add_argument("--date", default=DATE)
    parser.add_argument("--result-root", type=Path, default=HERE / "results")
    parser.add_argument("--data-root", type=Path, default=HERE.parents[1] / "04_Data" / "05_Fitting")
    args = parser.parse_args()
    out = run(args.result_root, args.data_root, args.date)
    metrics = pd.read_csv(out / "continuous_waveform_metrics.csv")
    print(f"Stage 6 complete: {len(metrics)} waveforms")
    print(metrics.groupby(["axis", "configuration"])["sample_rmse_deg"].agg(["mean", "median", "max"]).to_string())
    print(f"Overall waveform-equal sample RMSE: {np.sqrt(np.mean(metrics.sample_rmse_deg**2)):.6f} deg")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
