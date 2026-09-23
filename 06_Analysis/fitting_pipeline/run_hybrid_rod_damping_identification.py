"""ハイブリッド同定Stage 4: 球なし波形から共有ロッド減衰を同定する。

Stage 1で確定した頂点とStage 2で固定したI、Kを入力にし、Stage 3の
頂点間非線形ODEと同じ運動方程式を用いる。各半周期の開始時に実測頂点へ
戻し、球なし全波形に共有するc_rodと軸別b、tauを同時同定する。
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import lsq_linear, minimize
from scipy.special import ellipe, ellipk


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
DEFAULT_RESULT_ROOT = SCRIPT_DIRECTORY / "results"
MINIMUM_AMPLITUDE_DEG = 4.0
FRICTION_EPSILON_DEG_S = 0.5
MAX_STEP_PERIOD_FRACTION = 1.0 / 80.0
MAXIMUM_SEARCH_PERIODS = 2.0
MAD_GAUSSIAN_SCALE = 1.482602218505602
CONFIDENCE_Z_95 = 1.959963984540054
RESIDUAL_AMPLITUDE_BIN_DEG = 5.0
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
FULL_FIT_MAX_ITERATIONS = 80
CROSS_VALIDATION_MAX_ITERATIONS = 8
CROSS_VALIDATION_STEP_TOLERANCE = 1.0e-6
CROSS_VALIDATION_OBJECTIVE_TOLERANCE = 1.0e-8
CROSS_VALIDATION_LINE_SEARCH_STEPS = 6
CV_LINEARIZATION_VALIDATION_FOLDS = 2
CV_RMSE_RELATIVE_TOLERANCE = 0.01
# 34枚を横4列にすると9行となり、各パネルの波形と凡例を判読できる縦横比になる。
WAVEFORM_OVERVIEW_COLUMNS = 4
# 表示曲線だけの刻み。半周期を60分割し、ODEの最大刻みT0/80と同程度以上の描画密度にする。
WAVEFORM_TRAJECTORY_SUBDIVISIONS = 60

FREE_PARAMETER_NAMES = ["b_IN", "b_OUT", "c_rod", "tau_IN", "tau_OUT"]
B_IN_FIXED_PARAMETER_NAMES = ["b_OUT", "c_rod", "tau_IN", "tau_OUT"]
NO_LINEAR_PARAMETER_NAMES = ["c_rod", "tau_IN", "tau_OUT"]
FIXED_PARAMETER_NAMES = B_IN_FIXED_PARAMETER_NAMES
MODEL_PREFIXES = {
    "b_IN_free": "free",
    "b_IN_fixed_zero": "fixed",
    "b_IN_b_OUT_fixed_zero": "no_linear",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ハイブリッド同定Stage 4: 球なし共有減衰係数同定"
    )
    parser.add_argument("--date", required=True, help="試験日。例: 20260921")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT / "04_Data" / "05_Fitting",
        help="実波形ログのルート。配下に試験日ディレクトリを置く",
    )
    parser.add_argument(
        "--waveform-selection",
        type=Path,
        default=None,
        help="waveform_selection.csv。省略時は結果ルート内のレビュー済み表を使う",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--skip-cross-validation",
        action="store_true",
        help="開発時だけ1波形除外交差検証を省略する",
    )
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def repository_input_key(path):
    """外部作業コピーでもリポジトリ内と同じ入力キーを返す。"""

    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        parts = path.parts
        for anchor in ["04_Data", "06_Analysis"]:
            if anchor in parts:
                return str(Path(*parts[parts.index(anchor) :]))
    return path.name


def save_figure(figure, output_path, dpi=150):
    temporary_path = output_path.with_name(
        output_path.stem + ".tmp" + output_path.suffix
    )
    figure.savefig(temporary_path, dpi=dpi)
    plt.close(figure)
    temporary_path.replace(output_path)


def read_inputs(parent_result):
    stage1 = parent_result / "hybrid_identification" / "01_preprocessing"
    stage2 = parent_result / "hybrid_identification" / "02_frequency_identification"
    turning_path = stage1 / "turning_points.csv"
    waveform_path = stage1 / "waveform_preprocessing.csv"
    parameter_path = stage2 / "identified_inertia_restoring.csv"
    for path in [turning_path, waveform_path, parameter_path]:
        if not path.exists():
            raise FileNotFoundError("Stage 4の入力がありません: " + str(path))
    turning = pd.read_csv(turning_path)
    waveforms = pd.read_csv(waveform_path)
    parameters = pd.read_csv(parameter_path)
    return turning, waveforms, parameters, [turning_path, waveform_path, parameter_path]


def read_waveform_plot_inputs(date_directory, selection_path, waveform_summary):
    """34波形の実測ログと、Stage 1で採用した中心・区間を読み込む。"""

    if not selection_path.exists():
        raise FileNotFoundError("波形採否表がありません: " + str(selection_path))
    selection = pd.read_csv(selection_path, encoding="utf-8-sig")
    required = {
        "segment_id",
        "data_file",
        "axis",
        "angle_column",
        "configuration",
        "direction",
        "repetition",
        "start_index",
        "end_index",
        "use_for_fitting",
        "review_status",
    }
    missing = sorted(required - set(selection.columns))
    if missing:
        raise ValueError("波形採否表の列が不足しています: " + ", ".join(missing))
    if not (selection["review_status"].astype(str).str.upper() == "APPROVED").all():
        raise ValueError("波形採否表に未承認行があります")

    summary = waveform_summary.set_index("segment_id")
    selected = selection[
        (pd.to_numeric(selection["use_for_fitting"], errors="raise") == 1)
        & (selection["configuration"].astype(str).str.upper() != "BALL")
    ].copy()
    cache = {}
    records = []
    input_paths = [selection_path]
    for unused_index, row in selected.iterrows():
        segment_id = str(row["segment_id"])
        if segment_id not in summary.index:
            raise ValueError(segment_id + " のStage 1波形情報がありません")
        data_path = date_directory / str(row["data_file"])
        if data_path not in cache:
            if not data_path.exists():
                raise FileNotFoundError("実波形ログがありません: " + str(data_path))
            cache[data_path] = pd.read_csv(data_path)
            input_paths.append(data_path)
        complete = cache[data_path]
        angle_column = str(row["angle_column"])
        if "systime[ms]" not in complete or angle_column not in complete:
            raise ValueError(str(data_path) + " に時刻または角度列がありません")
        time_ms = pd.to_numeric(complete["systime[ms]"], errors="coerce")
        angle_deg = pd.to_numeric(complete[angle_column], errors="coerce")
        valid = np.isfinite(time_ms.to_numpy()) & np.isfinite(angle_deg.to_numpy())
        complete_time_s = time_ms.to_numpy(dtype=float)[valid] / 1000.0
        complete_angle_deg = angle_deg.to_numpy(dtype=float)[valid]
        complete_angle_deg = (complete_angle_deg + 180.0) % 360.0 - 180.0
        start = int(row["start_index"])
        end = int(row["end_index"]) + 1
        center_deg = float(summary.loc[segment_id, "envelope_center_deg"])
        records.append(
            {
                "segment_id": segment_id,
                "axis": str(row["axis"]).strip().upper(),
                "configuration": str(row["configuration"]).strip().upper(),
                "direction": str(row["direction"]).strip().upper(),
                "repetition": int(row["repetition"]),
                "time_s": complete_time_s[start:end],
                "centered_angle_deg": complete_angle_deg[start:end] - center_deg,
            }
        )
    if len(records) != 34:
        raise ValueError("球なし採用波形が34件ではありません: " + str(len(records)))
    return records, input_paths


def build_intervals(turning, parameters, minimum_amplitude_deg=MINIMUM_AMPLITUDE_DEG):
    """球なし波形から同定に使う連続頂点対を作る。"""

    parameter_index = parameters.set_index(["axis", "configuration"])
    rows = []
    # 不採用頂点も時系列上は残し、その前後を誤って一つの半周期として
    # 接続しない。連続する両端が採用可能な場合だけ区間へ追加する。
    selected = turning[turning["configuration"] != "BALL"]
    for segment_id, group in selected.groupby("segment_id", sort=False):
        group = group.sort_values("peak_number").reset_index(drop=True)
        initial = np.flatnonzero(group["is_initial_peak"].to_numpy(dtype=int) == 1)
        if len(initial) != 1:
            raise ValueError(segment_id + " の最初の採用頂点が一意ではありません")
        group = group.iloc[int(initial[0]) :].reset_index(drop=True)
        axis = str(group.loc[0, "axis"])
        configuration = str(group.loc[0, "configuration"])
        physical = parameter_index.loc[(axis, configuration)]
        for index in range(len(group) - 1):
            first = group.iloc[index]
            last = group.iloc[index + 1]
            if not (
                int(first["eligible_for_later_stages"]) == 1
                and int(last["eligible_for_later_stages"]) == 1
            ):
                continue
            if min(float(first["amplitude_deg"]), float(last["amplitude_deg"])) < minimum_amplitude_deg:
                continue
            start_angle_deg = float(first["centered_peak_angle_deg"])
            measured_next_angle_deg = float(last["centered_peak_angle_deg"])
            if start_angle_deg * measured_next_angle_deg >= 0.0:
                raise ValueError(segment_id + " の連続頂点の符号が交互ではありません")
            rows.append(
                {
                    "interval_id": segment_id + "_H" + str(index + 1).zfill(3),
                    "segment_id": segment_id,
                    "axis": axis,
                    "configuration": configuration,
                    "direction": str(group.loc[0, "direction"]),
                    "interval_number": index + 1,
                    "start_peak_number": int(first["peak_number"]),
                    "end_peak_number": int(last["peak_number"]),
                    "start_time_s": float(first["peak_time_s"]),
                    "end_time_s": float(last["peak_time_s"]),
                    "measured_half_period_s": float(last["peak_time_s"] - first["peak_time_s"]),
                    "start_angle_rad": float(np.deg2rad(start_angle_deg)),
                    "measured_next_angle_rad": float(np.deg2rad(measured_next_angle_deg)),
                    "start_angle_deg": start_angle_deg,
                    "measured_next_angle_deg": measured_next_angle_deg,
                    "start_amplitude_deg": abs(start_angle_deg),
                    "end_amplitude_deg": abs(measured_next_angle_deg),
                    "transition": "+to-" if start_angle_deg > 0.0 else "-to+",
                    "inertia_kg_m2": float(physical["inertia_kg_m2"]),
                    "restoring_n_m_per_rad": float(physical["restoring_n_m_per_rad"]),
                }
            )
    if not rows:
        raise ValueError("Stage 4に使用できる頂点間隔がありません")
    counts = pd.Series([row["segment_id"] for row in rows]).value_counts().to_dict()
    for row in rows:
        row["waveform_interval_count"] = int(counts[row["segment_id"]])
    return rows


def _physical_parameters(parameter_names, scaled_values, scales):
    values = {name: 0.0 for name in FREE_PARAMETER_NAMES}
    for index, name in enumerate(parameter_names):
        values[name] = float(scaled_values[index] * scales[name])
    return values


def _local_coefficients(interval, physical):
    axis = interval["axis"]
    return (
        physical["b_" + axis],
        physical["c_rod"],
        physical["tau_" + axis],
    )


def solve_interval_with_sensitivities(interval, physical):
    """1半周期を積分し、次頂点角のb、c、tau感度も返す。"""

    inertia = interval["inertia_kg_m2"]
    restoring = interval["restoring_n_m_per_rad"]
    initial_angle = interval["start_angle_rad"]
    damping, quadratic, friction = _local_coefficients(interval, physical)
    epsilon = np.deg2rad(FRICTION_EPSILON_DEG_S)
    small_period = 2.0 * math.pi * math.sqrt(inertia / restoring)

    def differential_equation(unused_time, state):
        angle = state[0]
        speed = state[1]
        tanh_speed = np.tanh(speed / epsilon)
        acceleration = -restoring * np.sin(angle)
        acceleration -= damping * speed
        acceleration -= quadratic * abs(speed) * speed
        acceleration -= friction * tanh_speed
        acceleration /= inertia

        state_jacobian_10 = -restoring * np.cos(angle) / inertia
        state_jacobian_11 = -damping - 2.0 * quadratic * abs(speed)
        state_jacobian_11 -= friction * (1.0 - tanh_speed**2) / epsilon
        state_jacobian_11 /= inertia
        parameter_forcing = [
            -speed / inertia,
            -abs(speed) * speed / inertia,
            -tanh_speed / inertia,
        ]
        derivative = [speed, acceleration]
        for parameter_index in range(3):
            angle_sensitivity = state[2 + 2 * parameter_index]
            speed_sensitivity = state[3 + 2 * parameter_index]
            derivative.append(speed_sensitivity)
            derivative.append(
                state_jacobian_10 * angle_sensitivity
                + state_jacobian_11 * speed_sensitivity
                + parameter_forcing[parameter_index]
            )
        return derivative

    def next_turning_event(unused_time, state):
        return state[1]

    next_turning_event.terminal = True
    next_turning_event.direction = 1.0 if initial_angle > 0.0 else -1.0
    solution = solve_ivp(
        differential_equation,
        (0.0, MAXIMUM_SEARCH_PERIODS * small_period),
        [initial_angle, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        method="DOP853",
        events=next_turning_event,
        rtol=1.0e-9,
        atol=[1.0e-12] * 8,
        max_step=MAX_STEP_PERIOD_FRACTION * small_period,
    )
    if not solution.success or len(solution.t_events[0]) != 1:
        raise RuntimeError(interval["interval_id"] + " の次頂点を検出できません")
    event_state = solution.y_events[0][0]
    return {
        "predicted_next_angle_rad": float(event_state[0]),
        "predicted_half_period_s": float(solution.t_events[0][0]),
        "sensitivity_b": float(event_state[2]),
        "sensitivity_c": float(event_state[4]),
        "sensitivity_tau": float(event_state[6]),
        "function_evaluations": int(solution.nfev),
    }


def solve_interval_trajectory(
    interval,
    physical,
    sample_count=WAVEFORM_TRAJECTORY_SUBDIVISIONS + 1,
):
    """実測頂点から予測次頂点までのフィット曲線を返す。"""

    inertia = interval["inertia_kg_m2"]
    restoring = interval["restoring_n_m_per_rad"]
    initial_angle = interval["start_angle_rad"]
    damping, quadratic, friction = _local_coefficients(interval, physical)
    epsilon = np.deg2rad(FRICTION_EPSILON_DEG_S)
    small_period = 2.0 * math.pi * math.sqrt(inertia / restoring)

    def differential_equation(unused_time, state):
        angle, speed = state
        acceleration = -restoring * np.sin(angle)
        acceleration -= damping * speed
        acceleration -= quadratic * abs(speed) * speed
        acceleration -= friction * np.tanh(speed / epsilon)
        return [speed, acceleration / inertia]

    def next_turning_event(unused_time, state):
        return state[1]

    next_turning_event.terminal = True
    next_turning_event.direction = 1.0 if initial_angle > 0.0 else -1.0
    solution = solve_ivp(
        differential_equation,
        (0.0, MAXIMUM_SEARCH_PERIODS * small_period),
        [initial_angle, 0.0],
        method="DOP853",
        events=next_turning_event,
        dense_output=True,
        rtol=1.0e-9,
        atol=[1.0e-12, 1.0e-12],
        max_step=MAX_STEP_PERIOD_FRACTION * small_period,
    )
    if not solution.success or len(solution.t_events[0]) != 1:
        raise RuntimeError(interval["interval_id"] + " の軌跡終端を検出できません")
    event_time = float(solution.t_events[0][0])
    relative_time = np.linspace(0.0, event_time, int(sample_count))
    angle = solution.sol(relative_time)[0]
    return relative_time, angle


def _solve_task(task):
    interval, physical = task
    return solve_interval_with_sensitivities(interval, physical)


def evaluate_intervals(intervals, physical, pool=None):
    tasks = [(interval, physical) for interval in intervals]
    if pool is None:
        results = [_solve_task(task) for task in tasks]
    else:
        chunk_size = max(1, len(tasks) // (pool._processes * 8))
        results = pool.map(_solve_task, tasks, chunksize=chunk_size)
    residual = np.asarray(
        [
            result["predicted_next_angle_rad"] - interval["measured_next_angle_rad"]
            for interval, result in zip(intervals, results)
        ],
        dtype=float,
    )
    return results, residual


def explicit_energy_basis(amplitude_rad, inertia, restoring):
    amplitude = abs(float(amplitude_rad))
    omega_zero = np.sqrt(restoring / inertia)
    parameter = np.sin(0.5 * amplitude) ** 2
    first_kind = ellipk(parameter)
    second_kind = ellipe(parameter)
    return np.asarray(
        [
            8.0 * omega_zero * (second_kind - (1.0 - parameter) * first_kind),
            4.0 * omega_zero**2 * (np.sin(amplitude) - amplitude * np.cos(amplitude)),
            2.0 * amplitude,
        ],
        dtype=float,
    )


def initial_energy_fit(intervals, parameter_names):
    matrix = []
    target = []
    for interval in intervals:
        basis = explicit_energy_basis(
            interval["start_angle_rad"],
            interval["inertia_kg_m2"],
            interval["restoring_n_m_per_rad"],
        )
        row = np.zeros(len(parameter_names), dtype=float)
        axis = interval["axis"]
        for index, name in enumerate(parameter_names):
            if name == "b_" + axis:
                row[index] = basis[0]
            elif name == "c_rod":
                row[index] = basis[1]
            elif name == "tau_" + axis:
                row[index] = basis[2]
        start_energy = interval["restoring_n_m_per_rad"] * (
            1.0 - np.cos(interval["start_angle_rad"])
        )
        end_energy = interval["restoring_n_m_per_rad"] * (
            1.0 - np.cos(interval["measured_next_angle_rad"])
        )
        weight = 1.0 / np.sqrt(interval["waveform_interval_count"])
        matrix.append(row * weight)
        target.append((start_energy - end_energy) * weight)
    result = lsq_linear(np.asarray(matrix), np.asarray(target), bounds=(0.0, np.inf))
    return {name: float(result.x[index]) for index, name in enumerate(parameter_names)}


def parameter_scales(initial_free):
    b_scale = max(initial_free["b_IN"], initial_free["b_OUT"])
    tau_scale = max(initial_free["tau_IN"], initial_free["tau_OUT"])
    c_scale = initial_free["c_rod"]
    if min(b_scale, c_scale, tau_scale) <= 0.0:
        raise ValueError("陽エネルギー初期値から正のパラメータ尺度を作れません")
    return {
        "b_IN": b_scale,
        "b_OUT": b_scale,
        "c_rod": c_scale,
        "tau_IN": tau_scale,
        "tau_OUT": tau_scale,
    }


def robust_scale_from_initial(residual_rad):
    residual_deg = np.rad2deg(np.asarray(residual_rad, dtype=float))
    median = float(np.median(residual_deg))
    mad = float(np.median(np.abs(residual_deg - median)))
    scale_deg = MAD_GAUSSIAN_SCALE * mad
    if not np.isfinite(scale_deg) or scale_deg <= 0.0:
        raise ValueError("初期残差からロバスト尺度を決定できません")
    return scale_deg


def objective_and_gradient(
    scaled_values,
    parameter_names,
    scales,
    intervals,
    robust_scale_rad,
    pool,
):
    physical = _physical_parameters(parameter_names, scaled_values, scales)
    results, residual = evaluate_intervals(intervals, physical, pool)
    z_value = residual / robust_scale_rad
    rho_value = np.sqrt(1.0 + z_value**2) - 1.0
    psi_value = z_value / np.sqrt(1.0 + z_value**2)
    waveform_count = len({interval["segment_id"] for interval in intervals})
    objective = 0.0
    gradient_physical = {name: 0.0 for name in parameter_names}
    for index, (interval, result) in enumerate(zip(intervals, results)):
        weight = 1.0 / (
            waveform_count * interval["waveform_interval_count"]
        )
        objective += weight * rho_value[index]
        common = weight * psi_value[index] / robust_scale_rad
        axis = interval["axis"]
        for name in parameter_names:
            if name == "b_" + axis:
                gradient_physical[name] += common * result["sensitivity_b"]
            elif name == "c_rod":
                gradient_physical[name] += common * result["sensitivity_c"]
            elif name == "tau_" + axis:
                gradient_physical[name] += common * result["sensitivity_tau"]
    gradient_scaled = np.asarray(
        [gradient_physical[name] * scales[name] for name in parameter_names],
        dtype=float,
    )
    return float(objective), gradient_scaled


def fit_model(
    intervals,
    parameter_names,
    scales,
    robust_scale_deg,
    initial_physical,
    pool=None,
    max_iterations=FULL_FIT_MAX_ITERATIONS,
):
    initial_scaled = np.asarray(
        [initial_physical.get(name, 0.0) / scales[name] for name in parameter_names],
        dtype=float,
    )
    evaluation_count = 0

    def evaluate(scaled_values):
        nonlocal evaluation_count
        evaluation_count += 1
        return objective_and_gradient(
            scaled_values,
            parameter_names,
            scales,
            intervals,
            np.deg2rad(robust_scale_deg),
            pool,
        )

    result = minimize(
        evaluate,
        initial_scaled,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * len(parameter_names),
        options={
            "maxiter": max_iterations,
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
            "maxls": 30,
        },
    )
    physical = _physical_parameters(parameter_names, result.x, scales)
    predictions, residual = evaluate_intervals(intervals, physical, pool)
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "evaluations": int(evaluation_count),
        "gradient_max_abs": float(np.max(np.abs(result.jac))),
        "parameters": physical,
        "predictions": predictions,
        "residual_rad": residual,
    }


def _objective_value(intervals, residual, robust_scale_rad):
    z_value = residual / robust_scale_rad
    rho_value = np.sqrt(1.0 + z_value**2) - 1.0
    waveform_count = len({interval["segment_id"] for interval in intervals})
    return float(
        sum(
            rho_value[index]
            / (waveform_count * interval["waveform_interval_count"])
            for index, interval in enumerate(intervals)
        )
    )


def fit_model_gauss_newton(
    intervals,
    parameter_names,
    scales,
    robust_scale_deg,
    initial_physical,
    pool=None,
    max_iterations=CROSS_VALIDATION_MAX_ITERATIONS,
):
    """感度方程式を使う境界付きロバストGauss-Newton再同定。"""

    scaled_values = np.asarray(
        [initial_physical.get(name, 0.0) / scales[name] for name in parameter_names],
        dtype=float,
    )
    robust_scale_rad = np.deg2rad(robust_scale_deg)
    evaluation_count = 0
    converged = False
    message = "maximum iterations reached"
    results = None
    residual = None
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        physical = _physical_parameters(parameter_names, scaled_values, scales)
        results, residual = evaluate_intervals(intervals, physical, pool)
        evaluation_count += 1
        objective = _objective_value(intervals, residual, robust_scale_rad)
        z_value = residual / robust_scale_rad
        robust_weight = 1.0 / np.sqrt(1.0 + z_value**2)
        waveform_count = len({row["segment_id"] for row in intervals})
        jacobian = np.zeros((len(intervals), len(parameter_names)), dtype=float)
        target = np.zeros(len(intervals), dtype=float)
        for row_index, (interval, prediction) in enumerate(zip(intervals, results)):
            weight = math.sqrt(
                robust_weight[row_index]
                / (waveform_count * interval["waveform_interval_count"])
            )
            target[row_index] = -weight * residual[row_index] / robust_scale_rad
            axis = interval["axis"]
            for column, name in enumerate(parameter_names):
                sensitivity = 0.0
                if name == "b_" + axis:
                    sensitivity = prediction["sensitivity_b"]
                elif name == "c_rod":
                    sensitivity = prediction["sensitivity_c"]
                elif name == "tau_" + axis:
                    sensitivity = prediction["sensitivity_tau"]
                jacobian[row_index, column] = (
                    weight * sensitivity * scales[name] / robust_scale_rad
                )
        step_result = lsq_linear(
            jacobian,
            target,
            bounds=(-scaled_values, np.full(len(parameter_names), np.inf)),
            lsmr_tol="auto",
        )
        full_step = step_result.x
        if np.max(np.abs(full_step)) <= CROSS_VALIDATION_STEP_TOLERANCE:
            converged = True
            message = "scaled Gauss-Newton step below tolerance"
            break

        accepted = False
        for line_search_index in range(CROSS_VALIDATION_LINE_SEARCH_STEPS):
            factor = 0.5**line_search_index
            trial_scaled = np.maximum(0.0, scaled_values + factor * full_step)
            trial_physical = _physical_parameters(
                parameter_names, trial_scaled, scales
            )
            trial_results, trial_residual = evaluate_intervals(
                intervals, trial_physical, pool
            )
            evaluation_count += 1
            trial_objective = _objective_value(
                intervals, trial_residual, robust_scale_rad
            )
            if trial_objective < objective:
                scaled_values = trial_scaled
                results = trial_results
                residual = trial_residual
                accepted = True
                relative_change = (objective - trial_objective) / max(
                    abs(objective), np.finfo(float).eps
                )
                if relative_change <= CROSS_VALIDATION_OBJECTIVE_TOLERANCE:
                    converged = True
                    message = "relative objective change below tolerance"
                break
        if not accepted:
            converged = True
            message = "line search found no improving step"
            break
        if converged:
            break

    physical = _physical_parameters(parameter_names, scaled_values, scales)
    results, residual = evaluate_intervals(intervals, physical, pool)
    evaluation_count += 1
    objective = _objective_value(intervals, residual, robust_scale_rad)
    return {
        "success": bool(converged),
        "message": message,
        "objective": objective,
        "iterations": int(iteration),
        "evaluations": int(evaluation_count),
        "gradient_max_abs": np.nan,
        "parameters": physical,
        "predictions": results,
        "residual_rad": residual,
    }


def model_metrics(intervals, fit):
    table = pd.DataFrame(
        {
            "segment_id": [row["segment_id"] for row in intervals],
            "residual_deg": np.rad2deg(fit["residual_rad"]),
        }
    )
    waveform_rmse = table.groupby("segment_id")["residual_deg"].apply(
        lambda values: float(np.sqrt(np.mean(values.to_numpy() ** 2)))
    )
    return {
        "interval_rmse_deg": float(np.sqrt(np.mean(table["residual_deg"] ** 2))),
        "waveform_equal_rmse_deg": float(
            np.sqrt(np.mean(waveform_rmse.to_numpy() ** 2))
        ),
        "waveform_rmse_mean_deg": float(waveform_rmse.mean()),
        "waveform_rmse_median_deg": float(waveform_rmse.median()),
        "waveform_rmse_max_deg": float(waveform_rmse.max()),
    }


def parameter_uncertainty(intervals, fit, parameter_names, robust_scale_deg):
    residual = fit["residual_rad"]
    robust_scale_rad = np.deg2rad(robust_scale_deg)
    z_value = residual / robust_scale_rad
    robust_weight = 1.0 / np.sqrt(1.0 + z_value**2)
    waveform_count = len({row["segment_id"] for row in intervals})
    jacobian = np.zeros((len(intervals), len(parameter_names)), dtype=float)
    weighted_residual = np.zeros(len(intervals), dtype=float)
    for row_index, (interval, prediction) in enumerate(
        zip(intervals, fit["predictions"])
    ):
        base_weight = 1.0 / (
            waveform_count * interval["waveform_interval_count"]
        )
        weight = math.sqrt(base_weight * robust_weight[row_index])
        weighted_residual[row_index] = weight * residual[row_index]
        axis = interval["axis"]
        for column, name in enumerate(parameter_names):
            if name == "b_" + axis:
                jacobian[row_index, column] = weight * prediction["sensitivity_b"]
            elif name == "c_rod":
                jacobian[row_index, column] = weight * prediction["sensitivity_c"]
            elif name == "tau_" + axis:
                jacobian[row_index, column] = weight * prediction["sensitivity_tau"]
    degrees_of_freedom = max(1, len(intervals) - len(parameter_names))
    variance = float(np.sum(weighted_residual**2) / degrees_of_freedom)
    covariance = variance * np.linalg.pinv(jacobian.T @ jacobian)
    standard_error = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    denominator = np.outer(standard_error, standard_error)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    rows = []
    for index, name in enumerate(parameter_names):
        value = fit["parameters"][name]
        rows.append(
            {
                "parameter": name,
                "value": value,
                "standard_error": float(standard_error[index]),
                "ci95_lower": float(value - CONFIDENCE_Z_95 * standard_error[index]),
                "ci95_upper": float(value + CONFIDENCE_Z_95 * standard_error[index]),
            }
        )
    return rows, correlation


def cross_validate_waveforms(
    intervals,
    model_name,
    parameter_names,
    scales,
    robust_scale_deg,
    full_fit,
    pool,
):
    """全データ最適解の厳密ヤコビアンから線形化1波形除外CVを行う。"""

    rows = []
    segment_ids = list(dict.fromkeys(row["segment_id"] for row in intervals))
    full_scaled = np.asarray(
        [full_fit["parameters"][name] / scales[name] for name in parameter_names],
        dtype=float,
    )
    robust_scale_rad = np.deg2rad(robust_scale_deg)
    for fold_number, held_segment in enumerate(segment_ids, start=1):
        training_indices = [
            index
            for index, row in enumerate(intervals)
            if row["segment_id"] != held_segment
        ]
        held = [row for row in intervals if row["segment_id"] == held_segment]
        training_waveform_count = len(segment_ids) - 1
        jacobian = np.zeros((len(training_indices), len(parameter_names)), dtype=float)
        target = np.zeros(len(training_indices), dtype=float)
        for matrix_row, interval_index in enumerate(training_indices):
            interval = intervals[interval_index]
            prediction = full_fit["predictions"][interval_index]
            residual_value = full_fit["residual_rad"][interval_index]
            z_value = residual_value / robust_scale_rad
            robust_weight = 1.0 / np.sqrt(1.0 + z_value**2)
            weight = math.sqrt(
                robust_weight
                / (training_waveform_count * interval["waveform_interval_count"])
            )
            target[matrix_row] = -weight * residual_value / robust_scale_rad
            axis = interval["axis"]
            for column, name in enumerate(parameter_names):
                sensitivity = 0.0
                if name == "b_" + axis:
                    sensitivity = prediction["sensitivity_b"]
                elif name == "c_rod":
                    sensitivity = prediction["sensitivity_c"]
                elif name == "tau_" + axis:
                    sensitivity = prediction["sensitivity_tau"]
                jacobian[matrix_row, column] = (
                    weight * sensitivity * scales[name] / robust_scale_rad
                )
        step_result = lsq_linear(
            jacobian,
            target,
            bounds=(-full_scaled, np.full(len(parameter_names), np.inf)),
            lsmr_tol="auto",
        )
        updated_scaled = np.maximum(0.0, full_scaled + step_result.x)
        updated_physical = _physical_parameters(
            parameter_names, updated_scaled, scales
        )
        predictions, residual = evaluate_intervals(held, updated_physical, pool)
        residual_deg = np.rad2deg(residual)
        rows.append(
            {
                "model": model_name,
                "held_segment_id": held_segment,
                "axis": held[0]["axis"],
                "configuration": held[0]["configuration"],
                "direction": held[0]["direction"],
                "held_intervals": len(held),
                "rmse_deg": float(np.sqrt(np.mean(residual_deg**2))),
                "mean_residual_deg": float(np.mean(residual_deg)),
                "maximum_abs_residual_deg": float(np.max(np.abs(residual_deg))),
                "refit_method": "one-step robust Gauss-Newton influence update",
                "scaled_step_norm": float(np.linalg.norm(step_result.x)),
                "fit_success": int(step_result.success),
                "fit_iterations": 1,
                **{name: updated_physical[name] for name in FREE_PARAMETER_NAMES},
            }
        )
    return rows


def validate_linearized_cross_validation(
    intervals,
    cv_rows,
    model_name,
    parameter_names,
    scales,
    robust_scale_deg,
    full_fit,
    pool,
):
    """影響量最大のLOOケースを完全再同定し、線形化近似を検証する。"""

    candidates = sorted(
        [row for row in cv_rows if row["model"] == model_name],
        key=lambda row: row["scaled_step_norm"],
        reverse=True,
    )[:CV_LINEARIZATION_VALIDATION_FOLDS]
    validation_rows = []
    for number, approximate in enumerate(candidates, start=1):
        held_segment = approximate["held_segment_id"]
        print(
            "CV近似検証 "
            + model_name
            + " "
            + str(number)
            + "/"
            + str(len(candidates))
            + " "
            + held_segment,
            flush=True,
        )
        training = [row for row in intervals if row["segment_id"] != held_segment]
        held = [row for row in intervals if row["segment_id"] == held_segment]
        exact_fit = fit_model_gauss_newton(
            training,
            parameter_names,
            scales,
            robust_scale_deg,
            full_fit["parameters"],
            pool,
            max_iterations=CROSS_VALIDATION_MAX_ITERATIONS,
        )
        unused_predictions, exact_residual = evaluate_intervals(
            held, exact_fit["parameters"], pool
        )
        exact_rmse = float(
            np.sqrt(np.mean(np.rad2deg(exact_residual) ** 2))
        )
        rmse_relative_difference = abs(exact_rmse - approximate["rmse_deg"]) / max(
            exact_rmse, np.finfo(float).eps
        )
        scaled_parameter_difference = max(
            abs(exact_fit["parameters"][name] - approximate[name]) / scales[name]
            for name in parameter_names
        )
        passed = (
            exact_fit["success"]
            and rmse_relative_difference <= CV_RMSE_RELATIVE_TOLERANCE
        )
        validation_rows.append(
            {
                "model": model_name,
                "held_segment_id": held_segment,
                "selection_reason": "largest scaled influence-step norm",
                "approximate_rmse_deg": approximate["rmse_deg"],
                "exact_refit_rmse_deg": exact_rmse,
                "rmse_relative_difference": rmse_relative_difference,
                "maximum_scaled_parameter_difference": scaled_parameter_difference,
                "exact_fit_success": int(exact_fit["success"]),
                "exact_fit_iterations": exact_fit["iterations"],
                "passed": int(passed),
            }
        )
    return validation_rows


def make_interval_rows(intervals, model_fits):
    rows = []
    for index, interval in enumerate(intervals):
        row = dict(interval)
        for model_name, fit in model_fits.items():
            prefix = MODEL_PREFIXES[model_name]
            prediction = fit["predictions"][index]
            predicted_deg = float(np.rad2deg(prediction["predicted_next_angle_rad"]))
            row[prefix + "_predicted_next_angle_deg"] = predicted_deg
            row[prefix + "_residual_deg"] = (
                predicted_deg - interval["measured_next_angle_deg"]
            )
            row[prefix + "_amplitude_residual_deg"] = (
                abs(predicted_deg) - abs(interval["measured_next_angle_deg"])
            )
            row[prefix + "_predicted_half_period_s"] = prediction[
                "predicted_half_period_s"
            ]
            row[prefix + "_half_period_residual_s"] = (
                prediction["predicted_half_period_s"]
                - interval["measured_half_period_s"]
            )
        row["amplitude_bin_lower_deg"] = (
            math.floor(interval["start_amplitude_deg"] / RESIDUAL_AMPLITUDE_BIN_DEG)
            * RESIDUAL_AMPLITUDE_BIN_DEG
        )
        rows.append(row)
    return rows


def summarize_residuals(interval_rows, adopted_model):
    table = pd.DataFrame(interval_rows)
    residual_column = adopted_model + "_residual_deg"
    amplitude_residual_column = adopted_model + "_amplitude_residual_deg"
    rows = []
    groupings = {
        "axis_configuration": ["axis", "configuration"],
        "axis_transition": ["axis", "transition"],
        "axis_amplitude_bin": ["axis", "amplitude_bin_lower_deg"],
    }
    for grouping, columns in groupings.items():
        for keys, group in table.groupby(columns, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            residual = group[residual_column].to_numpy(dtype=float)
            amplitude_residual = group[amplitude_residual_column].to_numpy(
                dtype=float
            )
            row = {
                "grouping": grouping,
                "samples": len(group),
                "mean_residual_deg": float(np.mean(residual)),
                "mean_amplitude_residual_deg": float(
                    np.mean(amplitude_residual)
                ),
                "rmse_deg": float(np.sqrt(np.mean(residual**2))),
                "maximum_abs_residual_deg": float(np.max(np.abs(residual))),
                "amplitude_residual_vs_start_amplitude_correlation": float(
                    np.corrcoef(
                        group["start_amplitude_deg"].to_numpy(dtype=float),
                        amplitude_residual,
                    )[0, 1]
                ),
            }
            for column, value in zip(columns, keys):
                row[column] = value
            rows.append(row)
    return rows


def plot_results(interval_rows, cv_rows, comparison_rows, adopted_model, output_path):
    table = pd.DataFrame(interval_rows)
    cv = pd.DataFrame(cv_rows)
    comparison = pd.DataFrame(comparison_rows).set_index("model")
    adopted_prefix = MODEL_PREFIXES[adopted_model]
    residual_column = adopted_prefix + "_amplitude_residual_deg"
    prediction_column = adopted_prefix + "_predicted_next_angle_deg"
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"IN": "#1f77b4", "OUT": "#d62728"}
    for axis, group in table.groupby("axis"):
        axes[0, 0].scatter(
            group["measured_next_angle_deg"],
            group[prediction_column],
            s=8,
            alpha=0.35,
            color=colors[axis],
            label=axis,
        )
        axes[0, 1].scatter(
            group["start_amplitude_deg"],
            group[residual_column],
            s=8,
            alpha=0.35,
            color=colors[axis],
            label=axis,
        )
    limit = float(
        max(
            table["measured_next_angle_deg"].abs().max(),
            table[prediction_column].abs().max(),
        )
    )
    axes[0, 0].plot([-limit, limit], [-limit, limit], "k--", linewidth=1)
    axes[0, 0].set_xlabel("measured next peak [deg]")
    axes[0, 0].set_ylabel("predicted next peak [deg]")
    axes[0, 0].legend()
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_xlabel("start amplitude [deg]")
    axes[0, 1].set_ylabel("next-peak amplitude residual [deg]")
    axes[0, 1].legend()
    box_data = [
        cv[cv["model"] == model]["rmse_deg"].to_numpy(dtype=float)
        for model in MODEL_PREFIXES
    ]
    axes[1, 0].boxplot(
        box_data,
        tick_labels=["both b free", "b_IN=0", "both b=0"],
    )
    axes[1, 0].set_ylabel("leave-one-waveform-out RMSE [deg]")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    names = ["training", "LOO-CV"]
    x_value = np.arange(len(names))
    width = 0.25
    for offset, model in zip([-1.0, 0.0, 1.0], MODEL_PREFIXES):
        values = [
            comparison.loc[model, "waveform_equal_rmse_deg"],
            comparison.loc[model, "cv_waveform_equal_rmse_deg"],
        ]
        axes[1, 1].bar(
            x_value + offset * width,
            values,
            width,
            label=model,
        )
    axes[1, 1].set_xticks(x_value, names)
    axes[1, 1].set_ylabel("waveform-equal RMSE [deg]")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    figure.suptitle("Stage 4 shared rod damping identification: " + adopted_model)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_waveform_overlays(
    waveform_records,
    intervals,
    adopted_fit,
    output_path,
):
    """全34波形へ区間別フィット曲線と実波形を重ねて表示する。"""

    records = sorted(
        waveform_records,
        key=lambda item: (
            item["axis"],
            item["configuration"],
            item["direction"],
            item["repetition"],
        ),
    )
    interval_indices = {}
    for index, interval in enumerate(intervals):
        interval_indices.setdefault(interval["segment_id"], []).append(index)
    column_count = WAVEFORM_OVERVIEW_COLUMNS
    row_count = int(math.ceil(len(records) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.0 * column_count, 3.0 * row_count),
        squeeze=False,
    )
    for plot_number, record in enumerate(records):
        axis = axes.flat[plot_number]
        indices = interval_indices[record["segment_id"]]
        segment_intervals = [intervals[index] for index in indices]
        time_origin = segment_intervals[0]["start_time_s"]
        final_time = segment_intervals[-1]["end_time_s"]
        measured_mask = (record["time_s"] >= time_origin) & (
            record["time_s"] <= final_time
        )
        axis.plot(
            record["time_s"][measured_mask] - time_origin,
            record["centered_angle_deg"][measured_mask],
            color="0.55",
            linewidth=0.65,
            label="measured waveform" if plot_number == 0 else None,
            zorder=1,
        )
        predicted_times = []
        predicted_angles = []
        measured_peak_times = [segment_intervals[0]["start_time_s"] - time_origin]
        measured_peak_angles = [segment_intervals[0]["start_angle_deg"]]
        for interval_index in indices:
            interval = intervals[interval_index]
            relative_time, angle_rad = solve_interval_trajectory(
                interval, adopted_fit["parameters"]
            )
            axis.plot(
                interval["start_time_s"] - time_origin + relative_time,
                np.rad2deg(angle_rad),
                color="#d95f02",
                linewidth=1.05,
                alpha=0.92,
                label="interval fit" if plot_number == 0 and interval_index == indices[0] else None,
                zorder=2,
            )
            prediction = adopted_fit["predictions"][interval_index]
            predicted_times.append(
                interval["start_time_s"]
                - time_origin
                + prediction["predicted_half_period_s"]
            )
            predicted_angles.append(
                np.rad2deg(prediction["predicted_next_angle_rad"])
            )
            measured_peak_times.append(interval["end_time_s"] - time_origin)
            measured_peak_angles.append(interval["measured_next_angle_deg"])
        axis.scatter(
            measured_peak_times,
            measured_peak_angles,
            s=9,
            color="#1b1b1b",
            label="measured peaks" if plot_number == 0 else None,
            zorder=3,
        )
        axis.scatter(
            predicted_times,
            predicted_angles,
            s=12,
            marker="x",
            linewidths=0.8,
            color="#d95f02",
            label="predicted peaks" if plot_number == 0 else None,
            zorder=4,
        )
        residual_deg = np.rad2deg(adopted_fit["residual_rad"][indices])
        rmse_deg = float(np.sqrt(np.mean(residual_deg**2)))
        axis.set_title(record["segment_id"] + f"  RMSE={rmse_deg:.3f} deg", fontsize=8)
        axis.set_xlim(0.0, final_time - time_origin)
        axis.grid(True, alpha=0.2)
        axis.tick_params(labelsize=7)
    for unused_axis in axes.flat[len(records) :]:
        unused_axis.axis("off")
    figure.supxlabel("time from first fitted peak [s]")
    figure.supylabel("centered angle [deg]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.989),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Stage 4 measured waveforms and interval-reset fits (34 no-ball records)",
        y=0.999,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.972))
    save_figure(figure, output_path)


def write_report(
    output_path,
    interval_rows,
    comparison_rows,
    uncertainty_rows,
    cv_rows,
    cv_validation_rows,
    robust_scale_deg,
    adopted_model,
    decision,
    b_out_decision,
):
    intervals = pd.DataFrame(interval_rows)
    comparison = pd.DataFrame(comparison_rows).set_index("model")
    uncertainty = pd.DataFrame(uncertainty_rows)
    free_uncertainty = uncertainty[uncertainty["model"] == "b_IN_free"].set_index(
        "parameter"
    )
    fixed_uncertainty = uncertainty[
        uncertainty["model"] == "b_IN_fixed_zero"
    ].set_index("parameter")
    cv = pd.DataFrame(cv_rows)
    cv_validation = pd.DataFrame(cv_validation_rows)
    adopted = comparison.loc[adopted_model]
    adopted_prefix = MODEL_PREFIXES[adopted_model]
    amplitude_residual_column = adopted_prefix + "_amplitude_residual_deg"
    amplitude_correlations = {
        axis: float(
            np.corrcoef(
                group["start_amplitude_deg"], group[amplitude_residual_column]
            )[0, 1]
        )
        for axis, group in intervals.groupby("axis")
    }
    transition_amplitude_correlations = {
        (axis, transition): float(
            np.corrcoef(
                group["start_amplitude_deg"], group[amplitude_residual_column]
            )[0, 1]
        )
        for (axis, transition), group in intervals.groupby(["axis", "transition"])
    }
    low_amplitude = intervals[intervals["amplitude_bin_lower_deg"] == 0.0]
    largest_configuration = (
        intervals.groupby(["axis", "configuration"])[amplitude_residual_column]
        .apply(lambda values: float(np.sqrt(np.mean(values.to_numpy() ** 2))))
        .sort_values(ascending=False)
    )
    lines = [
        "# Stage 4: 球なし共有ロッド減衰係数の同定",
        "",
        "## 結論",
        "",
        f"球なし34波形、{len(intervals)}半周期を用い、Stage 2で固定したI、Kと",
        "Stage 3の厳密な頂点間ODEからc_rod、軸別b、tauを共有同定した。",
        "b_IN自由、b_IN=0、および既採用候補からb_OUTも0にした両b=0モデルを比較し、",
        f"採用候補は`{adopted_model}`となった。",
        "係数は波形別・半周期別には求めず、全対象波形で共有している。",
        "",
        "## 同定方法",
        "",
        "各実測頂点(A_n,0)から次頂点を予測し、r_n=predicted-measuredを計算した。",
        "各波形の総重みを等しくし、次のsoft-L1型ロバスト損失を最小化した。",
        "",
        "```math",
        "J=\\frac{1}{N_{wave}}\\sum_r\\frac{1}{N_r}\\sum_n",
        "[\\sqrt{1+(r_{r,n}/\\sigma_A)^2}-1]",
        "```",
        "",
        f"残差尺度sigma_Aは陽エネルギー初期値における残差のMADから{robust_scale_deg:.6f} degと決定した。",
        "ODEと同時にb、c、tauに対する感度方程式を積分し、最適化勾配に使用した。",
        "解放直後の最初の半周期はStage 1で除外済みであり、振幅4 deg以上だけを使用した。",
        "1波形除外交差検証は全データ最適解の厳密ヤコビアンから除外後係数を1回更新し、",
        "除外波形だけを更新係数で厳密ODE再計算した。影響量最大の2波形/モデルは学習側も",
        "完全再同定し、線形化近似の誤差を別途確認した。",
        "",
        "## モデル比較",
        "",
        "| モデル | 学習・波形等重みRMSE [deg] | LOO-CV・波形等重みRMSE [deg] | 最大波形RMSE [deg] |",
        "|---|---:|---:|---:|",
    ]
    for model in MODEL_PREFIXES:
        row = comparison.loc[model]
        lines.append(
            f"| {model} | {row['waveform_equal_rmse_deg']:.6f} | "
            f"{row['cv_waveform_equal_rmse_deg']:.6f} | {row['waveform_rmse_max_deg']:.6f} |"
        )
    lines.extend(
        [
            "",
            "b_IN=0固定の採用条件は、自由モデルのb_INの95%区間が0を含み、かつ固定モデルの",
            "1波形除外交差検証誤差が自由モデルより悪化しないこととした。",
            f"判定: {decision}",
            "",
            "b_OUT=0は、既採用候補のb_IN=0を維持して両軸の線形粘性を0にしたモデルとして比較した。",
            "採用条件はb_IN=0モデルにおけるb_OUTの95%区間が0を含み、かつ両b=0モデルの",
            "LOO-CV誤差がb_IN=0モデルより悪化しないことである。",
            f"判定: {b_out_decision}",
            "",
            "## 採用候補係数",
            "",
            "| 係数 | 採用候補値 | 単位 |",
            "|---|---:|---|",
            f"| b_IN | {adopted['b_IN']:.9e} | N m s/rad |",
            f"| b_OUT | {adopted['b_OUT']:.9e} | N m s/rad |",
            f"| c_rod | {adopted['c_rod']:.9e} | N m s^2/rad^2 |",
            f"| tau_IN | {adopted['tau_IN']:.9e} | N m |",
            f"| tau_OUT | {adopted['tau_OUT']:.9e} | N m |",
            "",
            "## b_IN自由モデルの係数・近似区間",
            "",
            "| 係数 | 値 | 標準誤差 | 95%下限 | 95%上限 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in free_uncertainty.iterrows():
        lines.append(
            f"| {name} | {row['value']:.9e} | {row['standard_error']:.3e} | "
            f"{row['ci95_lower']:.9e} | {row['ci95_upper']:.9e} |"
        )
    lines.extend(
        [
            "",
            "上表はb_IN自由モデルの局所ヤコビアンとロバスト重みから求めた近似区間である。",
            "境界付き推定のため厳密な確率区間ではなく、b_IN=0比較の診断値として扱う。",
            "",
            "## b_IN=0モデルの係数・近似区間",
            "",
            "| 係数 | 値 | 標準誤差 | 95%下限 | 95%上限 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in fixed_uncertainty.iterrows():
        lines.append(
            f"| {name} | {row['value']:.9e} | {row['standard_error']:.3e} | "
            f"{row['ci95_lower']:.9e} | {row['ci95_upper']:.9e} |"
        )
    lines.extend(
        [
            "",
            "上表はb_OUT=0比較の診断値であり、b_INは0に固定しているため表に含めない。",
            "",
            "## 固定数値と根拠",
            "",
            "| 数値 | 根拠 |",
            "|---|---|",
            "| epsilon=0.5 deg/s | Stage 3レビューで承認した摩擦連続化の初期値。 |",
            "| 振幅下限4 deg | 停止直前の固着・頂点検出・中心誤差を避ける初期下限。Stage 2のK範囲では復元トルクが従来暫定tauの約9～18倍。 |",
            "| MAD係数1.482602... | 正規分布でMADを標準偏差相当にする理論定数1/Phi^-1(0.75)。 |",
            "| soft-L1 | 二乗損失の局所感度を維持し、外れ値の影響を漸減する滑らかなロバスト損失。 |",
            "| 95%係数1.959964... | 標準正規分布の両側95%分位点。 |",
            "| 振幅診断幅5 deg | Stage 2の振幅安定性診断と同じ区切りを使い、段階間比較を可能にする。 |",
            "| 本同定最大反復80 | 共有係数5変数のL-BFGS-B収束上限。実際の反復数と最終勾配を保存し、上限到達時は失敗扱いにする。 |",
            "| LOO線形化1回更新 | 1波形は全34波形の約3%以下であり、全データ最適解近傍の影響関数近似を使う。全除外波形の予測は更新係数で厳密ODE再計算する。 |",
            "| 完全再同定検証2波形/モデル | 線形化更新量が最大のケースを最悪条件として各モデル2件選び、学習側も完全再同定して近似を検証する。 |",
            "| 近似RMSE差1% | LOOの目的は未使用波形の予測誤差評価なので、完全再同定とのRMSE差を1%以内に制限する。超過時は線形化LOOを採用せず処理を失敗させる。 |",
            "| LOO係数差は診断のみ | 弱識別係数の配分が変わっても予測RMSEが安定する場合があるため、無次元係数差はCSVへ保存するがLOO合否には使わない。最終係数は全データ完全最適化値を使う。 |",
            "| 完全再同定最大反復8 | 全データ最適解を初期値にする検証用ロバストGauss-Newtonの上限。合成データで汎用L-BFGS-Bと同じ係数を2%以内で回収することも自動試験する。 |",
            "| CV刻み収束1e-6、目的関数相対変化1e-8 | 無次元化係数の更新量とロバスト目的関数の二つの停止条件。係数尺度は陽エネルギー初期値から決定する。 |",
            "| CVラインサーチ最大6回 | 1、1/2、…、1/32倍のGauss-Newton刻みを試し、目的関数が減少する最大刻みを採用する。 |",
            "| 波形概要図4列 | 34波形を9行に収め、各パネルの波形と凡例を判読できる縦横比にする表示専用設定。解析値には影響しない。 |",
            "| 半周期表示60分割 | ODEの最大刻みT0/80と同程度以上の描画密度で滑らかに表示する設定。フィット計算は適応刻みの厳密ODEで行う。 |",
            "",
            "## 34波形の実測・フィット重ね合わせ",
            "",
            "![球なし34波形の実測波形と区間別フィット](waveform_fit_overview.png)",
            "",
            "橙線は採用候補係数によるフィット結果、灰線は中心補正後の実測波形である。",
            "Stage 4の目的関数に合わせ、各橙線は半周期開始時に実測頂点へリセットしている。",
            "したがって、これは1半周期先フィットの重ね合わせであり、最初から最後まで自由走行させた連続再現ではない。",
            "黒点は実測頂点、橙の×印は予測次頂点を表す。各パネルのRMSEは次頂点角の誤差である。",
            "",
            "## 残差と検証",
            "",
            f"- 採用候補の学習・波形等重みRMSE: {adopted['waveform_equal_rmse_deg']:.6f} deg",
            f"- 採用候補のLOO-CV・波形等重みRMSE: {adopted['cv_waveform_equal_rmse_deg']:.6f} deg",
            f"- LOO-CV波形数: {len(cv[cv['model'] == adopted_model])}",
            f"- 線形化LOO近似の完全再同定検証: {int(cv_validation['passed'].sum())}/{len(cv_validation)}件合格",
            f"- 完全再同定とのRMSE相対差最大: {cv_validation['rmse_relative_difference'].max():.6%}",
            f"- 完全再同定との無次元係数差最大: {cv_validation['maximum_scaled_parameter_difference'].max():.6f}（識別性診断値、合否対象外）",
            f"- 始点振幅と次頂点振幅残差の相関: IN={amplitude_correlations['IN']:.6f}、OUT={amplitude_correlations['OUT']:.6f}",
            "- 遷移方向別の同相関: "
            f"IN +to-={transition_amplitude_correlations[('IN', '+to-')]:.6f}、"
            f"IN -to+={transition_amplitude_correlations[('IN', '-to+')]:.6f}、"
            f"OUT +to-={transition_amplitude_correlations[('OUT', '+to-')]:.6f}、"
            f"OUT -to+={transition_amplitude_correlations[('OUT', '-to+')]:.6f}",
            f"- 軸×形態で最大のRMSE: {largest_configuration.index[0][0]} {largest_configuration.index[0][1]}、{largest_configuration.iloc[0]:.6f} deg",
            f"- 4～5 deg帯の次頂点振幅残差平均: IN={low_amplitude[low_amplitude['axis'] == 'IN'][amplitude_residual_column].mean():.6f} deg、OUT={low_amplitude[low_amplitude['axis'] == 'OUT'][amplitude_residual_column].mean():.6f} deg",
            "- 軸×形態、正負遷移、振幅帯別の残差をresidual_summary.csvへ保存した。",
            "- 軸全体の相関は正負遷移で相殺される一方、遷移方向別には強い逆向きの振幅依存がある。平衡中心、復元項の正負非対称、頂点抽出の影響を候補としてStage 6で確認する。",
            "- 下限直上の4～5 deg帯では負の振幅残差があり、低振幅域の摩擦・頂点検出影響を持ち越す。",
            "- 本Stageは1半周期先予測の評価であり、最初の頂点から終端までの連続波形再現はStage 6で行う。",
            "",
            "## Stage 5へ持ち越す事項",
            "",
            "- 採用したロッド係数を初期値とし、BALLを追加してc_sphereを共有同定する。",
            "- 球で隠れるロッドの寄与はalpha_rod=(129/229)^4として扱う。",
            "- 正負遷移で逆向きとなる振幅依存残差を、BALLでも方向別に分離して監視する。",
            "- Stage 5開始前に本Stageの係数、モデル選択、残差構造のレビュー承認を受ける。",
            "",
            "## 実行コマンド",
            "",
            "```bash",
            "python -m unittest discover -s 06_Analysis/fitting_pipeline/tests -p 'test_hybrid_rod_damping.py'",
            "python 06_Analysis/fitting_pipeline/run_hybrid_rod_damping_identification.py --date 20260921",
            "```",
            "",
            "## 出力",
            "",
            "- [全半周期の予測と残差](interval_predictions.csv)",
            "- [モデル比較](model_comparison.csv)",
            "- [係数と近似信頼区間](parameter_uncertainty.csv)",
            "- [b_IN=0モデルの係数相関](parameter_correlation_b_in_fixed.csv)",
            "- [1波形除外交差検証](leave_one_waveform_out.csv)",
            "- [線形化LOO近似の完全再同定検証](cv_approximation_validation.csv)",
            "- [残差集計](residual_summary.csv)",
            "- [結果概要図](rod_damping_identification.png)",
            "- [34波形の実測・フィット重ね合わせ](waveform_fit_overview.png)",
            "- [実行条件](stage4_settings.json)",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    arguments = parse_arguments()
    parent_result = arguments.result_root / arguments.date
    output_directory = (
        parent_result / "hybrid_identification" / "04_rod_damping_identification"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    turning, waveforms, parameters, input_paths = read_inputs(parent_result)
    selection_path = arguments.waveform_selection
    if selection_path is None:
        selection_path = parent_result / "waveform_review" / "waveform_selection.csv"
    intervals = build_intervals(turning, parameters)
    free_initial = initial_energy_fit(intervals, FREE_PARAMETER_NAMES)
    scales = parameter_scales(free_initial)

    worker_count = max(1, int(arguments.workers))
    pool = None
    if worker_count > 1:
        pool = multiprocessing.Pool(processes=worker_count)
    try:
        print("1/6 初期残差尺度を決定", flush=True)
        unused_initial_results, initial_residual = evaluate_intervals(
            intervals, free_initial, pool
        )
        robust_scale_deg = robust_scale_from_initial(initial_residual)
        print("2/6 b_IN自由モデルを同定", flush=True)
        free_fit = fit_model(
            intervals,
            FREE_PARAMETER_NAMES,
            scales,
            robust_scale_deg,
            free_initial,
            pool,
        )
        fixed_initial = dict(free_fit["parameters"])
        fixed_initial["b_IN"] = 0.0
        print("3/6 b_IN=0固定モデルを同定", flush=True)
        fixed_fit = fit_model(
            intervals,
            B_IN_FIXED_PARAMETER_NAMES,
            scales,
            robust_scale_deg,
            fixed_initial,
            pool,
        )
        no_linear_initial = dict(fixed_fit["parameters"])
        no_linear_initial["b_OUT"] = 0.0
        print("4/6 b_IN=b_OUT=0固定モデルを同定", flush=True)
        no_linear_fit = fit_model(
            intervals,
            NO_LINEAR_PARAMETER_NAMES,
            scales,
            robust_scale_deg,
            no_linear_initial,
            pool,
        )
        if not all(
            fit["success"] for fit in [free_fit, fixed_fit, no_linear_fit]
        ):
            raise RuntimeError(
                "共有減衰同定が収束しません: "
                + free_fit["message"]
                + " / "
                + fixed_fit["message"]
                + " / "
                + no_linear_fit["message"]
            )
        free_metrics = model_metrics(intervals, free_fit)
        fixed_metrics = model_metrics(intervals, fixed_fit)
        no_linear_metrics = model_metrics(intervals, no_linear_fit)
        free_uncertainty_rows, correlation = parameter_uncertainty(
            intervals,
            free_fit,
            FREE_PARAMETER_NAMES,
            robust_scale_deg,
        )
        fixed_uncertainty_rows, fixed_correlation = parameter_uncertainty(
            intervals,
            fixed_fit,
            B_IN_FIXED_PARAMETER_NAMES,
            robust_scale_deg,
        )
        uncertainty_rows = [
            {"model": "b_IN_free", **row} for row in free_uncertainty_rows
        ] + [
            {"model": "b_IN_fixed_zero", **row}
            for row in fixed_uncertainty_rows
        ]
        if arguments.skip_cross_validation:
            raise RuntimeError("正式なStage 4成果では交差検証を省略できません")
        print("5/6 1波形除外交差検証", flush=True)
        cv_rows = []
        cv_rows.extend(
            cross_validate_waveforms(
                intervals,
                "b_IN_free",
                FREE_PARAMETER_NAMES,
                scales,
                robust_scale_deg,
                free_fit,
                pool,
            )
        )
        cv_rows.extend(
            cross_validate_waveforms(
                intervals,
                "b_IN_fixed_zero",
                B_IN_FIXED_PARAMETER_NAMES,
                scales,
                robust_scale_deg,
                fixed_fit,
                pool,
            )
        )
        cv_rows.extend(
            cross_validate_waveforms(
                intervals,
                "b_IN_b_OUT_fixed_zero",
                NO_LINEAR_PARAMETER_NAMES,
                scales,
                robust_scale_deg,
                no_linear_fit,
                pool,
            )
        )
        cv_validation_rows = []
        cv_validation_rows.extend(
            validate_linearized_cross_validation(
                intervals,
                cv_rows,
                "b_IN_free",
                FREE_PARAMETER_NAMES,
                scales,
                robust_scale_deg,
                free_fit,
                pool,
            )
        )
        cv_validation_rows.extend(
            validate_linearized_cross_validation(
                intervals,
                cv_rows,
                "b_IN_fixed_zero",
                B_IN_FIXED_PARAMETER_NAMES,
                scales,
                robust_scale_deg,
                fixed_fit,
                pool,
            )
        )
        cv_validation_rows.extend(
            validate_linearized_cross_validation(
                intervals,
                cv_rows,
                "b_IN_b_OUT_fixed_zero",
                NO_LINEAR_PARAMETER_NAMES,
                scales,
                robust_scale_deg,
                no_linear_fit,
                pool,
            )
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    cv_table = pd.DataFrame(cv_rows)
    failed_cv = cv_table[cv_table["fit_success"] != 1]
    if len(failed_cv) > 0:
        raise RuntimeError(
            "1波形除外交差検証の再同定が収束しません: "
            + ", ".join(failed_cv["held_segment_id"].astype(str).tolist())
        )
    cv_validation_table = pd.DataFrame(cv_validation_rows)
    failed_cv_validation = cv_validation_table[
        cv_validation_table["passed"] != 1
    ]
    if len(failed_cv_validation) > 0:
        output_directory.mkdir(parents=True, exist_ok=True)
        cv_validation_table.to_csv(
            output_directory / "cv_approximation_validation_failed.csv",
            index=False,
        )
        print(cv_validation_table.to_string(index=False), flush=True)
        raise RuntimeError(
            "線形化LOO近似が完全再同定と一致しません: "
            + ", ".join(
                failed_cv_validation["held_segment_id"].astype(str).tolist()
            )
        )
    comparison_rows = []
    for model_name, fit, metrics in [
        ("b_IN_free", free_fit, free_metrics),
        ("b_IN_fixed_zero", fixed_fit, fixed_metrics),
        ("b_IN_b_OUT_fixed_zero", no_linear_fit, no_linear_metrics),
    ]:
        model_cv = cv_table[cv_table["model"] == model_name]
        comparison_rows.append(
            {
                "model": model_name,
                **metrics,
                "objective": fit["objective"],
                "iterations": fit["iterations"],
                "evaluations": fit["evaluations"],
                "gradient_max_abs": fit["gradient_max_abs"],
                "cv_waveform_equal_rmse_deg": float(
                    np.sqrt(np.mean(model_cv["rmse_deg"].to_numpy() ** 2))
                ),
                "cv_waveform_rmse_mean_deg": float(model_cv["rmse_deg"].mean()),
                "cv_waveform_rmse_median_deg": float(model_cv["rmse_deg"].median()),
                "cv_waveform_rmse_max_deg": float(model_cv["rmse_deg"].max()),
                **fit["parameters"],
            }
        )
    comparison = pd.DataFrame(comparison_rows).set_index("model")
    b_in_uncertainty = {
        row["parameter"]: row for row in free_uncertainty_rows
    }["b_IN"]
    interval_includes_zero = (
        b_in_uncertainty["ci95_lower"] <= 0.0
        and b_in_uncertainty["ci95_upper"] >= 0.0
    )
    validation_not_worse = (
        comparison.loc["b_IN_fixed_zero", "cv_waveform_equal_rmse_deg"]
        <= comparison.loc["b_IN_free", "cv_waveform_equal_rmse_deg"]
    )
    if interval_includes_zero and validation_not_worse:
        adopted_model = "b_IN_fixed_zero"
        decision = "95%区間が0を含み、固定モデルのLOO-CV誤差が悪化しないためb_IN=0を採用候補とする。"
    else:
        adopted_model = "b_IN_free"
        reasons = []
        if not interval_includes_zero:
            reasons.append("b_INの95%区間が0を含まない")
        if not validation_not_worse:
            reasons.append("b_IN=0固定でLOO-CV誤差が悪化する")
        decision = "、".join(reasons) + "ためb_IN自由モデルを採用候補とする。"

    b_out_uncertainty = {
        row["parameter"]: row for row in fixed_uncertainty_rows
    }["b_OUT"]
    b_out_interval_includes_zero = (
        b_out_uncertainty["ci95_lower"] <= 0.0
        and b_out_uncertainty["ci95_upper"] >= 0.0
    )
    b_out_validation_not_worse = (
        comparison.loc[
            "b_IN_b_OUT_fixed_zero", "cv_waveform_equal_rmse_deg"
        ]
        <= comparison.loc["b_IN_fixed_zero", "cv_waveform_equal_rmse_deg"]
    )
    if (
        adopted_model == "b_IN_fixed_zero"
        and b_out_interval_includes_zero
        and b_out_validation_not_worse
    ):
        adopted_model = "b_IN_b_OUT_fixed_zero"
        b_out_decision = (
            "b_OUTの95%区間が0を含み、両b=0でもLOO-CV誤差が悪化しないため、"
            "b_OUT=0も採用候補とする。"
        )
    else:
        b_out_reasons = []
        if adopted_model != "b_IN_fixed_zero":
            b_out_reasons.append("基準となるb_IN=0モデルが採用候補ではない")
        if not b_out_interval_includes_zero:
            b_out_reasons.append("b_IN=0モデルのb_OUT 95%区間が0を含まない")
        if not b_out_validation_not_worse:
            b_out_reasons.append("両b=0でLOO-CV誤差が悪化する")
        b_out_decision = (
            "、".join(b_out_reasons)
            + "ためb_OUT=0は採用せず、b_OUTを同定するモデルを維持する。"
        )

    # 大きい実波形配列は並列フィット終了後に読む。ワーカープロセスへ不要な
    # 配列を複製せず、同定計算のメモリ使用量とプロセス生成負荷を抑える。
    waveform_records, waveform_input_paths = read_waveform_plot_inputs(
        arguments.data_root / arguments.date,
        selection_path,
        waveforms,
    )
    input_paths.extend(waveform_input_paths)

    print("6/6 結果を保存", flush=True)
    model_fits = {
        "b_IN_free": free_fit,
        "b_IN_fixed_zero": fixed_fit,
        "b_IN_b_OUT_fixed_zero": no_linear_fit,
    }
    interval_rows = make_interval_rows(intervals, model_fits)
    residual_rows = summarize_residuals(
        interval_rows, MODEL_PREFIXES[adopted_model]
    )
    pd.DataFrame(interval_rows).to_csv(
        output_directory / "interval_predictions.csv", index=False
    )
    pd.DataFrame(comparison_rows).to_csv(
        output_directory / "model_comparison.csv", index=False
    )
    pd.DataFrame(uncertainty_rows).to_csv(
        output_directory / "parameter_uncertainty.csv", index=False
    )
    cv_table.to_csv(output_directory / "leave_one_waveform_out.csv", index=False)
    cv_validation_table.to_csv(
        output_directory / "cv_approximation_validation.csv", index=False
    )
    pd.DataFrame(residual_rows).to_csv(
        output_directory / "residual_summary.csv", index=False
    )
    np.savetxt(
        output_directory / "parameter_correlation.csv",
        correlation,
        delimiter=",",
        header=",".join(FREE_PARAMETER_NAMES),
        comments=",",
    )
    np.savetxt(
        output_directory / "parameter_correlation_b_in_fixed.csv",
        fixed_correlation,
        delimiter=",",
        header=",".join(B_IN_FIXED_PARAMETER_NAMES),
        comments=",",
    )
    plot_results(
        interval_rows,
        cv_rows,
        comparison_rows,
        adopted_model,
        output_directory / "rod_damping_identification.png",
    )
    plot_waveform_overlays(
        waveform_records,
        intervals,
        model_fits[adopted_model],
        output_directory / "waveform_fit_overview.png",
    )
    write_report(
        output_directory / "ROD_DAMPING_REPORT.md",
        interval_rows,
        comparison_rows,
        uncertainty_rows,
        cv_rows,
        cv_validation_rows,
        robust_scale_deg,
        adopted_model,
        decision,
        b_out_decision,
    )
    settings = {
        "stage": 4,
        "date": arguments.date,
        "minimum_amplitude_deg": MINIMUM_AMPLITUDE_DEG,
        "friction_epsilon_deg_s": FRICTION_EPSILON_DEG_S,
        "maximum_step_period_fraction": MAX_STEP_PERIOD_FRACTION,
        "maximum_search_periods": MAXIMUM_SEARCH_PERIODS,
        "robust_loss": "sqrt(1 + z^2) - 1",
        "robust_scale_method": "1.482602218505602 * MAD of exact-ODE residual at explicit-energy initial estimate",
        "robust_scale_deg": robust_scale_deg,
        "waveform_weighting": "equal total weight per waveform",
        "full_fit_max_iterations": FULL_FIT_MAX_ITERATIONS,
        "cross_validation_max_iterations": CROSS_VALIDATION_MAX_ITERATIONS,
        "cross_validation_optimizer": "one-step robust Gauss-Newton influence update from full-data optimum; held waveform evaluated with exact ODE",
        "cross_validation_exact_validation_optimizer": "bounded iteratively reweighted Gauss-Newton with exact ODE sensitivities",
        "cross_validation_step_tolerance": CROSS_VALIDATION_STEP_TOLERANCE,
        "cross_validation_objective_relative_tolerance": CROSS_VALIDATION_OBJECTIVE_TOLERANCE,
        "cross_validation_line_search_steps": CROSS_VALIDATION_LINE_SEARCH_STEPS,
        "cross_validation_linearization_validation_folds_per_model": CV_LINEARIZATION_VALIDATION_FOLDS,
        "cross_validation_rmse_relative_tolerance": CV_RMSE_RELATIVE_TOLERANCE,
        "workers": worker_count,
        "waveforms": len({row["segment_id"] for row in intervals}),
        "intervals": len(intervals),
        "initial_energy_parameters": free_initial,
        "parameter_scales": scales,
        "adopted_model_candidate": adopted_model,
        "decision": decision,
        "b_out_zero_decision": b_out_decision,
        "waveform_overview_columns": WAVEFORM_OVERVIEW_COLUMNS,
        "waveform_trajectory_subdivisions_per_half_cycle": WAVEFORM_TRAJECTORY_SUBDIVISIONS,
        "waveform_overlay_definition": "measured centered waveform overlaid with exact-ODE half-cycle fits reset at every measured starting peak",
        "input_sha256": {
            repository_input_key(path): file_sha256(path)
            for path in input_paths
        },
    }
    (output_directory / "stage4_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("結果: " + str(output_directory), flush=True)


if __name__ == "__main__":
    main()
