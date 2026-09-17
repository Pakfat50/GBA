"""Stage 3: tune and compare force estimators with a TWELITE angle path."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

from friction_estimators import (
    causal_luenberger_with_friction,
    kalman_rts_force_with_friction,
)
from model import PendulumParameters
from run_friction_observer_doe import friction_plant
from run_real_wind_doe import (
    CANDIDATES,
    causal_lowpass,
    force_metrics,
    wind_speed_metrics,
)
from sensor_model import (
    AngleSensorParameters,
    apply_angle_sensor_model,
    effective_uncorrelated_noise_std_deg,
)
from wind import drag_force_from_speed, synthesize_kaimal_wind


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "True force": "#111111",
    "Raw static": "#8c8c8c",
    "Causal LPF": "#8e6bbd",
    "ESO 3-state": "#2776bc",
    "ESO 4-state": "#e67e22",
    "RTS 3-state": "#159477",
    "RTS 4-state": "#006b4f",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of equally shaped dictionaries."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sensor_parameters(config: dict, case_name: str = "nominal") -> AngleSensorParameters:
    """Build the nominal, ideal, or one-factor sensitivity sensor."""
    base = dict(config["sensor"])
    case = config["sensitivity_cases"].get(case_name, {})
    if case_name == "ideal":
        base.update(
            resolution_bits=53,
            white_noise_std_deg=0.0,
            coloured_noise_std_deg=0.0,
            coloured_noise_time_constant_s=0.0,
            fixed_delay_s=0.0,
            sampling_jitter_std_s=0.0,
            gain_error_fraction=0.0,
            offset_deg=0.0,
        )
    elif not case.get("use_nominal", False):
        for key, value in case.items():
            if key != "use_nominal":
                base[key] = value
    base.pop("use_nominal", None)
    return AngleSensorParameters(sample_rate_hz=config["sample_rate_hz"], **base)


def wind_case(config: dict, duration_s: float, seed: int):
    """Create one deterministic Kaimal wind and its quasi-steady force."""
    time, speed, metadata = synthesize_kaimal_wind(
        config["sample_rate_hz"],
        duration_s,
        config["mean_wind_speed_m_s"],
        config["target_turbulence_intensity"],
        config["kaimal_integral_scale_m"],
        seed,
        config["maximum_wind_speed_m_s"],
    )
    force = drag_force_from_speed(
        speed,
        config["air_density_kg_m3"],
        config["drag_coefficient"],
        config["projected_area_m2"],
    )
    return time, speed, force, metadata


def assumed_noise_rad(parameters: AngleSensorParameters) -> float:
    """White approximation used by the Kalman/RTS measurement covariance."""
    independent = effective_uncorrelated_noise_std_deg(parameters)
    total = np.hypot(independent, parameters.coloured_noise_std_deg)
    return float(np.deg2rad(max(total, 1e-9)))


def observer_estimates(
    angle_rad: np.ndarray,
    parameters: PendulumParameters,
    dt: float,
    tuning: dict,
    sensor: AngleSensorParameters,
    friction_torque: float,
    epsilon: float,
) -> dict[str, np.ndarray]:
    """Run the four friction-aware observer candidates."""
    noise_rad = assumed_noise_rad(sensor)
    _, rts3 = kalman_rts_force_with_friction(
        angle_rad,
        parameters,
        dt,
        noise_rad,
        tuning["rts3_process_noise"],
        friction_torque,
        epsilon,
        0,
    )
    _, rts4 = kalman_rts_force_with_friction(
        angle_rad,
        parameters,
        dt,
        noise_rad,
        tuning["rts4_process_noise"],
        friction_torque,
        epsilon,
        1,
    )
    return {
        "ESO 3-state": causal_luenberger_with_friction(
            angle_rad, parameters, dt, tuning["eso3_pole_hz"],
            friction_torque, epsilon, 0,
        ),
        "ESO 4-state": causal_luenberger_with_friction(
            angle_rad, parameters, dt, tuning["eso4_pole_hz"],
            friction_torque, epsilon, 1,
        ),
        "RTS 3-state": rts3,
        "RTS 4-state": rts4,
    }


def tune_estimators(
    angle_rad: np.ndarray,
    true_force: np.ndarray,
    parameters: PendulumParameters,
    dt: float,
    mask: np.ndarray,
    config: dict,
    sensor: AngleSensorParameters,
    friction_torque: float,
    epsilon: float,
) -> tuple[dict, list[dict]]:
    """Choose each method on a training wind record only."""
    rows: list[dict] = []
    for order, method, key, grid in (
        (0, "ESO 3-state", "eso3_pole_hz", config["eso_pole_grid_hz"]),
        (1, "ESO 4-state", "eso4_pole_hz", config["eso_pole_grid_hz"]),
    ):
        for value in grid:
            estimate = causal_luenberger_with_friction(
                angle_rad, parameters, dt, value, friction_torque, epsilon, order
            )
            rows.append({
                "method": method,
                "tuning_key": key,
                "tuning_value": value,
                **force_metrics(true_force, estimate, mask),
            })

    noise_rad = assumed_noise_rad(sensor)
    for order, method, key, grid in (
        (0, "RTS 3-state", "rts3_process_noise", config["rts3_force_random_walk_grid_N_per_sample"]),
        (1, "RTS 4-state", "rts4_process_noise", config["rts4_force_rate_random_walk_grid_N_per_s_per_sample"]),
    ):
        for value in grid:
            _, estimate = kalman_rts_force_with_friction(
                angle_rad, parameters, dt, noise_rad, value,
                friction_torque, epsilon, order,
            )
            rows.append({
                "method": method,
                "tuning_key": key,
                "tuning_value": value,
                **force_metrics(true_force, estimate, mask),
            })

    raw_static = parameters.restoring_n_m_per_rad / parameters.force_lever_m * angle_rad
    for value in config["lpf_cutoff_grid_hz"]:
        estimate = causal_lowpass(raw_static, value, dt)
        rows.append({
            "method": "Causal LPF",
            "tuning_key": "lpf_cutoff_hz",
            "tuning_value": value,
            **force_metrics(true_force, estimate, mask),
        })

    tuning: dict[str, float] = {}
    for method in (*CANDIDATES, "Causal LPF"):
        selected = min(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["rmse_N"],
        )
        tuning[selected["tuning_key"]] = float(selected["tuning_value"])
    return tuning, rows


def best_lag_s(
    truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray, dt: float, limit_s: float = 2.0
) -> float:
    """Lag that maximises correlation; positive means estimate is late."""
    a = np.asarray(truth[mask])
    b = np.asarray(estimate[mask])
    a = a - np.mean(a)
    b = b - np.mean(b)
    maximum = int(round(limit_s / dt))
    correlation = signal.correlate(b, a, mode="full", method="fft")
    lags = signal.correlation_lags(len(b), len(a), mode="full")
    keep = np.abs(lags) <= maximum
    return float(lags[keep][np.argmax(correlation[keep])] * dt)


def evaluate_methods(
    time: np.ndarray,
    speed: np.ndarray,
    force: np.ndarray,
    angle: np.ndarray,
    parameters: PendulumParameters,
    sensor: AngleSensorParameters,
    tuning: dict,
    config: dict,
    mask: np.ndarray,
    friction_torque: float,
    epsilon: float,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Evaluate baselines and the four candidates on one sensor record."""
    dt = 1.0 / config["sample_rate_hz"]
    estimates = observer_estimates(
        angle, parameters, dt, tuning, sensor, friction_torque, epsilon
    )
    estimates["Raw static"] = (
        parameters.restoring_n_m_per_rad / parameters.force_lever_m * angle
    )
    estimates["Causal LPF"] = causal_lowpass(
        estimates["Raw static"], tuning["lpf_cutoff_hz"], dt
    )
    rows = []
    for method in ("Raw static", "Causal LPF", *CANDIDATES):
        _, wind_metrics = wind_speed_metrics(
            time, speed, estimates[method], mask, config
        )
        rows.append({
            "method": method,
            **force_metrics(force, estimates[method], mask),
            **wind_metrics,
            "best_lag_s": best_lag_s(force, estimates[method], mask, dt),
        })
    return estimates, rows


def zero_force_check(
    config: dict,
    sensor: AngleSensorParameters,
    parameters: PendulumParameters,
    tuning: dict,
    friction_torque: float,
    epsilon: float,
) -> list[dict]:
    """Measure false external force for a stationary zero-angle mechanism."""
    dt = 1.0 / config["sample_rate_hz"]
    time = np.arange(0.0, 60.0, dt)
    measured, _ = apply_angle_sensor_model(
        time, np.zeros_like(time), sensor, config["sensor_seed"] + 100
    )
    estimates = observer_estimates(
        measured, parameters, dt, tuning, sensor, friction_torque, epsilon
    )
    mask = time >= 10.0
    rows = []
    for method, estimate in estimates.items():
        selected = estimate[mask]
        rows.append({
            "method": method,
            "rms_false_force_N": float(np.sqrt(np.mean(selected**2))),
            "bias_false_force_N": float(np.mean(selected)),
            "p95_abs_false_force_N": float(np.quantile(np.abs(selected), 0.95)),
            "max_abs_false_force_N": float(np.max(np.abs(selected))),
        })
    return rows


def make_plots(
    output: Path,
    config: dict,
    time: np.ndarray,
    speed: np.ndarray,
    force: np.ndarray,
    true_angle: np.ndarray,
    measured_angle: np.ndarray,
    components: dict,
    estimates: dict,
    mask: np.ndarray,
    nominal_rows: list[dict],
    sensitivity_rows: list[dict],
) -> None:
    """Create the Stage 3 diagnostic figures."""
    fs = config["sample_rate_hz"]
    selected = mask & (time <= config["evaluation_start_s"] + 30.0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    axes[0].plot(time[selected], np.rad2deg(true_angle[selected]), color="black", label="True")
    axes[0].plot(time[selected], np.rad2deg(measured_angle[selected]), color="#2776bc", alpha=0.75, label="Measured")
    axes[0].set(ylabel="Angle [deg]", title="TWELITE/MT6701 sensor model")
    axes[0].legend()
    error_deg = np.rad2deg(measured_angle - components["delayed_angle_rad"])
    axes[1].plot(time[selected], error_deg[selected], color="#c23b73", linewidth=0.8)
    axes[1].set(xlabel="Time [s]", ylabel="Sensor error [deg]")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output / "sensor_timeseries.png", dpi=180)
    plt.close(fig)

    frequency, psd = signal.welch(
        error_deg - np.mean(error_deg), fs=fs, nperseg=min(len(error_deg), 8192)
    )
    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    keep = frequency > 0.0
    axis.loglog(frequency[keep], psd[keep], color="#2776bc")
    axis.set(
        xlabel="Frequency [Hz]",
        ylabel="Angle-error PSD [deg²/Hz]",
        title="Modelled sensor-error spectrum",
    )
    axis.grid(which="both", alpha=0.2)
    fig.savefig(output / "sensor_error_spectrum.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5), layout="constrained")
    axis.plot(time[selected], 1000.0 * force[selected], color="black", linewidth=2.0, label="True force")
    for method in ("Causal LPF", *CANDIDATES):
        axis.plot(
            time[selected], 1000.0 * estimates[method][selected],
            color=COLORS[method], linewidth=1.0, label=method,
        )
    axis.set(xlabel="Time [s]", ylabel="Force [mN]", title="Stage 3 force estimates")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    fig.savefig(output / "force_estimates.png", dpi=180)
    plt.close(fig)

    methods = list(CANDIDATES)
    values = [
        1000.0 * next(row["rmse_N"] for row in nominal_rows if row["method"] == method)
        for method in methods
    ]
    fig, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    axis.bar(methods, values, color=[COLORS[m] for m in methods])
    axis.set(ylabel="Force RMSE [mN]", title="Nominal sensor model")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(output / "nominal_method_comparison.png", dpi=180)
    plt.close(fig)

    cases = list(config["sensitivity_cases"])
    matrix = np.array([
        [
            1000.0 * next(
                row["rmse_N"] for row in sensitivity_rows
                if row["case"] == case and row["method"] == method
            )
            for case in cases
        ]
        for method in methods
    ])
    fig, axis = plt.subplots(figsize=(11, 4.8), layout="constrained")
    image = axis.imshow(matrix, aspect="auto", cmap="magma")
    axis.set_xticks(range(len(cases)), cases, rotation=25, ha="right")
    axis.set_yticks(range(len(methods)), methods)
    axis.set_title("Sensor one-factor sensitivity: force RMSE [mN]")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=axis, label="RMSE [mN]")
    fig.savefig(output / "sensor_sensitivity.png", dpi=180)
    plt.close(fig)


def write_report(
    output: Path,
    config: dict,
    sensor: AngleSensorParameters,
    tuning: dict,
    nominal_rows: list[dict],
    zero_rows: list[dict],
    sensitivity_rows: list[dict],
    maximum_true_angle_deg: float,
) -> None:
    """Write a concise reproducible Stage 3 result report."""
    methods = list(CANDIDATES)
    lines = [
        "# Stage 3：TWELITE BLUEセンサーモデルを含む推定器比較",
        "",
        "## 結論",
        "",
        "TWELITE BLUE上の実装、MT6701の14 bit角度出力、既存ログの100 Hz時刻列から、",
        "量子化・白色ノイズ・有色ノイズ・遅延・ジッタ・ゲイン・オフセットを独立に切替可能な",
        "センサーモデルを追加した。調整用風波形と評価用風波形は異なるseedを使用した。",
        f"センサー評価とストッパー評価を分離するため最大風速を5.0 m/sとし、最大角度は{maximum_true_angle_deg:.2f}°だった。",
        "",
        "## 公称センサー係数",
        "",
        "| 項目 | 値 | 根拠 |",
        "|---|---:|---|",
        f"| サンプル周波数 | {sensor.sample_rate_hz:g} Hz | ファームウェアと既存ログ |",
        f"| 分解能 | {sensor.resolution_bits} bit | MT6701および実装 |",
        f"| 量子化幅 | {sensor.quantisation_step_deg:.8f} deg | 360° / 2^{sensor.resolution_bits} |",
        f"| 白色ノイズσ | {sensor.white_noise_std_deg:.3f} deg | 既存PoC 0.02°を暫定分解 |",
        f"| 有色ノイズσ | {sensor.coloured_noise_std_deg:.3f} deg | BMI160補正を暫定表現 |",
        f"| 有色ノイズ時定数 | {sensor.coloured_noise_time_constant_s:.3f} s | 20 Hz、LPF係数0.1 |",
        f"| 固定遅延 | {1000*sensor.fixed_delay_s:.1f} ms | 1サンプル暫定値 |",
        f"| ジッタσ | {1000*sensor.sampling_jitter_std_s:.1f} ms | ログでは10 ms一定 |",
        "",
        "ノイズと遅延は新ハード静止ログで更新すべき暫定値である。分解能と100 Hz周期は実装から確定している。",
        "",
        "## 再調整値",
        "",
        "| 推定器 | 調整値 |",
        "|---|---:|",
        f"| ESO 3-state pole | {tuning['eso3_pole_hz']:.6g} Hz |",
        f"| ESO 4-state pole | {tuning['eso4_pole_hz']:.6g} Hz |",
        f"| RTS 3-state process noise | {tuning['rts3_process_noise']:.6g} N/sample |",
        f"| RTS 4-state process noise | {tuning['rts4_process_noise']:.6g} (N/s)/sample |",
        f"| Causal LPF cutoff | {tuning['lpf_cutoff_hz']:.6g} Hz |",
        "",
        "## 未使用風波形での評価",
        "",
        "| 推定器 | 外力RMSE [mN] | 風速RMSE [m/s] | 最良ラグ [s] |",
        "|---|---:|---:|---:|",
    ]
    for method in ("Raw static", "Causal LPF", *methods):
        row = next(item for item in nominal_rows if item["method"] == method)
        lines.append(
            f"| {method} | {1000*row['rmse_N']:.4f} | {row['rmse_m_s']:.4f} | {row['best_lag_s']:.3f} |"
        )
    lines += [
        "",
        "## 無外力時の偽外力",
        "",
        "| 推定器 | RMS [mN] | 95%絶対値 [mN] | 最大絶対値 [mN] |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        row = next(item for item in zero_rows if item["method"] == method)
        lines.append(
            f"| {method} | {1000*row['rms_false_force_N']:.4f} | "
            f"{1000*row['p95_abs_false_force_N']:.4f} | {1000*row['max_abs_false_force_N']:.4f} |"
        )
    best = min(
        (row for row in nominal_rows if row["method"] in methods),
        key=lambda row: row["rmse_N"],
    )
    lines += [
        "",
        "## 考察",
        "",
        f"公称センサー条件では **{best['method']}** が最小で、外力RMSEは{1000*best['rmse_N']:.3f} mNだった。",
        "理想センサーで選んだ高帯域設定を流用せず、センサー条件下で再調整する必要がある。",
        "一因子感度では推定器の調整値とKalman測定雑音を公称値に固定した。詳細は `sensitivity_metrics.csv` に保存した。",
        "",
        "| 推定器 | 最悪ケース | 公称比RMSE増加 |",
        "|---|---|---:|",
    ]
    for method in methods:
        nominal_rmse = next(
            row["rmse_N"] for row in sensitivity_rows
            if row["case"] == "nominal" and row["method"] == method
        )
        worst = max(
            (row for row in sensitivity_rows if row["method"] == method),
            key=lambda row: row["rmse_N"],
        )
        increase = 100.0 * (worst["rmse_N"] / nominal_rmse - 1.0)
        lines.append(f"| {method} | {worst['case']} | {increase:.1f}% |")
    lines += [
        "",
        "## 図",
        "",
        "![センサー時系列](results/sensor_model_stage3/sensor_timeseries.png)",
        "",
        "![外力推定](results/sensor_model_stage3/force_estimates.png)",
        "",
        "![センサー感度](results/sensor_model_stage3/sensor_sensitivity.png)",
        "",
        "## 係数根拠資料",
        "",
        "- [TWELITE BLUE/RED公式データシート](https://twelite.net/data-sheets/twelite/blue-red/latest.html)",
        "- [MagnTek公式サイト：MT6701の14 bit絶対角度出力](https://www.magntek.com.cn/)",
        "- [`mt6701.cpp`](https://github.com/Pakfat50/GBA/blob/feature/friction-observer-sensitivity/03_Software/GbaSoftware/src/mt6701.cpp)：14 bit値の角度変換",
        "- [`normal_mode_task.h`](https://github.com/Pakfat50/GBA/blob/feature/friction-observer-sensitivity/03_Software/GbaSoftware/src/normal_mode_task.h)：100 Hz周期",
        "- [`normal_mode_task.cpp`](https://github.com/Pakfat50/GBA/blob/feature/friction-observer-sensitivity/03_Software/GbaSoftware/src/normal_mode_task.cpp)：MT6701読出しとBMI160補正",
        "- [`utility.cpp`](https://github.com/Pakfat50/GBA/blob/feature/friction-observer-sensitivity/03_Software/GbaSoftware/src/utility.cpp)：一次LPF係数0.1",
        "- [`LOG00008_ANGLE.csv`](https://github.com/Pakfat50/GBA/blob/feature/friction-observer-sensitivity/05_Script/02_LoggerDecoder/LOG00008_ANGLE.csv)：10 ms時刻刻み",
        "",
        "## 注意",
        "",
        "今回の係数は実機静止試験前の初期モデルである。特に0.02°のノイズ分解と10 ms遅延は",
        "推測を含むため、最終帯域や方式選定の確定値にはしない。新ハードログ取得後は設定JSONだけを更新し再実行する。",
        "",
        "## 再実行",
        "",
        "```bash",
        "python 06_Analysis/simulation/src/run_sensor_model_stage3.py",
        "```",
    ]
    (ROOT / "Sensor_Model_Stage3_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path, output: Path) -> dict:
    """Run Stage 3 and return the serialisable summary."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    parameters = PendulumParameters(**config["nominal_parameters"])
    parameters.validate()
    dt = 1.0 / config["sample_rate_hz"]
    friction_torque = config["nominal_friction_torque_n_m"]
    epsilon = np.deg2rad(config["friction_epsilon_deg_s"])
    nominal_sensor = sensor_parameters(config, "nominal")

    training_time, _, training_force, _ = wind_case(
        config, config["training_duration_s"], config["training_seed"]
    )
    training_state = friction_plant(
        training_force, parameters, dt, friction_torque, epsilon
    )
    training_angle, _ = apply_angle_sensor_model(
        training_time, training_state[:, 0], nominal_sensor, config["sensor_seed"]
    )
    training_mask = (training_time >= 15.0) & (
        training_time <= config["training_duration_s"] - 10.0
    )
    tuning, tuning_rows = tune_estimators(
        training_angle,
        training_force,
        parameters,
        dt,
        training_mask,
        config,
        nominal_sensor,
        friction_torque,
        epsilon,
    )
    write_csv(output / "tuning_scan.csv", tuning_rows)

    time, speed, force, wind_metadata = wind_case(
        config, config["evaluation_duration_s"], config["evaluation_seed"]
    )
    state = friction_plant(force, parameters, dt, friction_torque, epsilon)
    true_angle = state[:, 0]
    measured_angle, components = apply_angle_sensor_model(
        time, true_angle, nominal_sensor, config["sensor_seed"] + 1
    )
    mask = (time >= config["evaluation_start_s"]) & (
        time <= config["evaluation_end_s"]
    )
    estimates, nominal_rows = evaluate_methods(
        time, speed, force, measured_angle, parameters, nominal_sensor,
        tuning, config, mask, friction_torque, epsilon,
    )
    write_csv(output / "nominal_metrics.csv", nominal_rows)

    sensitivity_rows: list[dict] = []
    for case_name in config["sensitivity_cases"]:
        current_sensor = sensor_parameters(config, case_name)
        current_angle, _ = apply_angle_sensor_model(
            time, true_angle, current_sensor, config["sensor_seed"] + 1
        )
        _, rows = evaluate_methods(
            time, speed, force, current_angle, parameters, nominal_sensor,
            tuning, config, mask, friction_torque, epsilon,
        )
        for row in rows:
            if row["method"] in CANDIDATES:
                sensitivity_rows.append({"case": case_name, **row})
    write_csv(output / "sensitivity_metrics.csv", sensitivity_rows)

    zero_rows = zero_force_check(
        config, nominal_sensor, parameters, tuning, friction_torque, epsilon
    )
    write_csv(output / "zero_force_metrics.csv", zero_rows)
    make_plots(
        output, config, time, speed, force, true_angle, measured_angle,
        components, estimates, mask, nominal_rows, sensitivity_rows,
    )

    maximum_true_angle_deg = float(np.max(np.abs(np.rad2deg(true_angle))))
    summary = {
        "config": str(config_path.relative_to(ROOT.parent.parent)),
        "sensor_parameters": nominal_sensor.__dict__,
        "quantisation_step_deg": nominal_sensor.quantisation_step_deg,
        "effective_white_plus_quantisation_std_deg": effective_uncorrelated_noise_std_deg(nominal_sensor),
        "assumed_total_angle_noise_std_deg": float(np.rad2deg(assumed_noise_rad(nominal_sensor))),
        "tuning": tuning,
        "wind_metadata": wind_metadata,
        "maximum_true_angle_deg": maximum_true_angle_deg,
        "nominal_metrics": nominal_rows,
        "zero_force_metrics": zero_rows,
        "seeds": {
            "training_wind": config["training_seed"],
            "evaluation_wind": config["evaluation_seed"],
            "sensor": config["sensor_seed"],
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        output, config, nominal_sensor, tuning, nominal_rows, zero_rows,
        sensitivity_rows,
        maximum_true_angle_deg,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "sensor_model_stage3.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sensor_model_stage3",
    )
    args = parser.parse_args()
    summary = run(args.config, args.output)
    print(json.dumps({
        "output": str(args.output),
        "tuning": summary["tuning"],
        "assumed_total_angle_noise_std_deg": summary["assumed_total_angle_noise_std_deg"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
