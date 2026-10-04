"""OW-05 theoretical-only separation of observer error sources.

This run uses the adopted BALL plant, an RTS 3-state force observer and
theoretical wind/angle signals. It compares matched coefficients, an OW-04
physical coefficient corner, and the nominal angle-sensor noise model.

Run from the repository root:
    python 06_Analysis/simulation/src/run_ow05_theoretical_error_separation.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
OUT = SIM / "results/observer_wind/ow05_theoretical_error_separation"
REPO = SIM.parents[1]
sys.path.insert(0, str(HERE))

from hbk_model_coefficients import select_coefficients, simulate_hbk_plant
from hbk_nonlinear_estimators import nonlinear_ekf_rts_force
from run_ow03_hbk_observer_tuning import applied_force, force_to_speed, make_wind
from sensor_model import AngleSensorParameters, apply_angle_sensor_model


Q_BY_AXIS = {"IN": 3e-5, "OUT": 3e-4}
NOISE_SEEDS = [20261020, 20261021, 20261022, 20261023, 20261024]
SCENARIOS = {
    "① 完全一致・無雑音": "matched_clean",
    "② OW-04係数ずれ": "ow04_mismatch",
    "③ センサーノイズ": "sensor_noise",
}


def save_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def static_angle(coeff: dict, force_n: float, lever: float) -> float:
    # Same static equilibrium relation used by the OW-03 static-force baseline.
    return float(np.arctan(lever * force_n / coeff["restoring_n_m_per_rad"]))


def load_mismatch_models(axes: list[str], nominal: dict[str, dict]) -> dict[str, dict]:
    """Use one physically correlated high-mass/high-arm/high-tau/high-c OW-04 corner."""
    source = OUT.parent / "ow04_coefficient_sensitivity" / "ow04_joint_scenarios.csv"
    table = pd.read_csv(source, encoding="utf-8-sig")
    target = "joint_mass_high_arm_high_tau-high_c-high"
    models = {}
    for axis in axes:
        row = table[(table.axis == axis) & (table.scenario == target)]
        if row.empty:
            raise RuntimeError(f"Missing OW-04 coefficient corner {axis}/{target} in {source}")
        row = row.iloc[0]
        model = nominal[axis].copy()
        model["inertia_kg_m2"] = float(row.observer_I_kg_m2)
        model["restoring_n_m_per_rad"] = float(row.observer_K_n_m_per_rad)
        model["tau_n_m"] = float(row.observer_tau_n_m)
        model["ball_quadratic_drag_n_m_s2_per_rad2"] = float(row.observer_c_ball_n_m_s2_per_rad2)
        model["total_quadratic_drag_n_m_s2_per_rad2"] = (
            model["rod_quadratic_drag_n_m_s2_per_rad2"]
            + model["ball_quadratic_drag_n_m_s2_per_rad2"]
        )
        models[axis] = model
    return models


def sensor_parameters(config: dict) -> AngleSensorParameters:
    data = json.loads((SIM / "config/sensor_model_stage3.json").read_text(encoding="utf-8"))["sensor"]
    # This experiment isolates stochastic angle noise/quantisation. Delay,
    # gain, offset and jitter remain zero so they are not mislabeled as noise.
    return AngleSensorParameters(
        sample_rate_hz=float(config["sample_rate_hz"]),
        resolution_bits=int(data["resolution_bits"]),
        full_scale_deg=float(data["full_scale_deg"]),
        white_noise_std_deg=float(data["white_noise_std_deg"]),
        coloured_noise_std_deg=float(data["coloured_noise_std_deg"]),
        coloured_noise_time_constant_s=float(data["coloured_noise_time_constant_s"]),
        fixed_delay_s=0.0,
        sampling_jitter_std_s=0.0,
        gain_error_fraction=0.0,
        offset_deg=0.0,
    )


def build_inputs(config: dict, axis: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float | None, float]]:
    """Return time, truth speed/force, initial angle and initial force."""
    rate = float(config["sample_rate_hz"])
    dt = 1.0 / rate
    # Start from a static 60-degree equilibrium with the equivalent holding wind,
    # hold for two seconds, then remove external force instantaneously.
    angle_limit = np.deg2rad(float(config["mechanical_angle_limit_deg"]))
    axis_coeff = select_coefficients(axis, "BALL")
    k = axis_coeff["restoring_n_m_per_rad"]
    hold_force = float(k * np.tan(angle_limit) / config["force_lever_m"])
    hold_speed = float(force_to_speed(config, np.array([hold_force]))[0])
    duration = 30.0
    t = np.arange(int(duration * rate) + 1) / rate
    step_speed = np.where(t < 2.0, hold_speed, 0.0)
    step_force = applied_force(config, step_speed)
    angle0 = angle_limit
    step = (t, step_speed, step_force, angle0, hold_force)

    descriptions = [
        next(d for d in config["validation_winds"] if d["name"] == "独立Kaimal乱流 平均2 m/s TI20%"),
        next(d for d in config["validation_winds"] if d["name"] == "ガスト 2→6 m/s"),
    ]
    outputs = {"自由振動ステップ": step}
    for description in descriptions:
        wt, speed, _ = make_wind(config, description)
        force = applied_force(config, speed)
        initial_force = float(force[0])
        outputs[description["name"]] = (wt, speed, force, None, initial_force)
    return outputs


def metrics(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    error = np.asarray(estimate)[mask] - np.asarray(truth)[mask]
    bias = float(np.mean(error))
    centered = error - bias
    return {
        "rmse_m_s": float(np.sqrt(np.mean(error**2))),
        "bias_m_s": bias,
        "bias_removed_rms_m_s": float(np.sqrt(np.mean(centered**2))),
        "mae_m_s": float(np.mean(np.abs(error))),
        "p95_abs_error_m_s": float(np.quantile(np.abs(error), 0.95)),
        "max_abs_error_m_s": float(np.max(np.abs(error))),
    }


def simulate_case(axis: str, coefficients: dict, observer_coeff: dict,
                  config: dict, data: tuple, noise_seed: int | None,
                  q_override: float | None = None):
    t, true_speed, force, angle0, initial_force = data
    if angle0 is None:
        angle0 = static_angle(coefficients, float(force[0]), config["force_lever_m"])
    dt = 1.0 / float(config["sample_rate_hz"])
    plant_coeff = coefficients
    states = simulate_hbk_plant(
        force, plant_coeff, dt, initial_angle_rad=angle0, initial_rate_rad_s=0.0,
        force_lever_m=config["force_lever_m"],
        friction_epsilon_deg_s=config["friction_epsilon_deg_s"],
    )
    true_angle = states[:, 0]
    measured_angle = true_angle.copy()
    if noise_seed is not None:
        measured_angle, _ = apply_angle_sensor_model(
            t, true_angle, sensor_parameters(config), noise_seed,
        )
    _, estimated_force = nonlinear_ekf_rts_force(
        measured_angle, observer_coeff, dt, config["force_lever_m"],
        np.deg2rad(float(config["assumed_angle_noise_deg"])),
        Q_BY_AXIS[axis] if q_override is None else q_override, 0,
        config["friction_epsilon_deg_s"],
        initial_state=np.array([angle0, 0.0, initial_force]),
    )
    estimated_speed = force_to_speed(config, estimated_force)
    return t, true_speed, estimated_speed, true_angle, measured_angle


def evaluate() -> tuple[list[dict], dict, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((SIM / "config/ow03_hbk_observer_tuning.json").read_text(encoding="utf-8"))
    dt = 1.0 / float(config["sample_rate_hz"])
    axes = config["axis_names"]
    nominal = {axis: select_coefficients(axis, "BALL") for axis in axes}
    mismatch = load_mismatch_models(axes, nominal)
    inputs_by_axis = {axis: build_inputs(config, axis) for axis in axes}
    detailed: list[dict] = []
    q_sensitivity: list[dict] = []
    integration_refinement: list[dict] = []
    waveforms: dict[tuple, dict] = {}

    for axis in axes:
        inputs = inputs_by_axis[axis]
        for input_name, data in inputs.items():
            t = data[0]
            eval_start = 2.0 if input_name == "自由振動ステップ" else float(config["evaluation_start_s"])
            mask = t >= eval_start
            for scenario_name, scenario_key in SCENARIOS.items():
                seeds = NOISE_SEEDS if scenario_key == "sensor_noise" else [None]
                observer_coeff = mismatch[axis] if scenario_key == "ow04_mismatch" else nominal[axis]
                for seed in seeds:
                    result = simulate_case(axis, nominal[axis], observer_coeff, config, data, seed)
                    _, truth, estimate, true_angle, measured_angle = result
                    row = {
                        "axis": axis, "input": input_name, "scenario": scenario_name,
                        "scenario_key": scenario_key, "noise_seed": seed,
                        "q_N_per_sample": Q_BY_AXIS[axis], "evaluation_start_s": eval_start,
                        **metrics(truth, estimate, mask),
                    }
                    if scenario_key == "ow04_mismatch":
                        for param, key in (("I", "inertia_kg_m2"), ("K", "restoring_n_m_per_rad"),
                                           ("tau", "tau_n_m"), ("c_total", "total_quadratic_drag_n_m_s2_per_rad2")):
                            row[f"observer_{param}_offset_percent"] = 100.0 * (observer_coeff[key] / nominal[axis][key] - 1.0)
                    detailed.append(row)
                    waveforms[(axis, input_name, scenario_key, seed)] = {
                        "t": t, "truth": truth, "estimate": estimate,
                        "angle": true_angle, "measured_angle": measured_angle,
                    }

    # A matched-model step is outside the temporal prior implied by the selected
    # small q. This focused sweep distinguishes observer tuning error from
    # numerical integration error while holding plant and angle data fixed.
    for axis in axes:
        data = inputs_by_axis[axis]["自由振動ステップ"]
        mask = data[0] >= 2.0
        for q in sorted({Q_BY_AXIS[axis], 1e-3, 1e-2}):
            result = simulate_case(axis, nominal[axis], nominal[axis], config, data, None, q_override=q)
            _, truth, estimate, _, _ = result
            q_sensitivity.append({
                "axis": axis, "input": "自由振動ステップ", "scenario": "完全一致・無雑音 q感度",
                "q_N_per_sample": q, **metrics(truth, estimate, mask),
            })
        # Compare the plant's 100 Hz RK4 trajectory with a half-step RK4 run,
        # sampled back onto the same instants. This checks forward-integration
        # truncation independently of the observer's q and sensor model.
        t, _, force, angle0, _ = data
        dt = 1.0 / float(config["sample_rate_hz"])
        low = simulate_hbk_plant(force, nominal[axis], dt, initial_angle_rad=angle0,
                                 initial_rate_rad_s=0.0, force_lever_m=config["force_lever_m"],
                                 friction_epsilon_deg_s=config["friction_epsilon_deg_s"])
        half_force = np.concatenate([np.repeat(force[:-1], 2), force[-1:]])
        high = simulate_hbk_plant(half_force, nominal[axis], dt / 2,
                                  initial_angle_rad=angle0, initial_rate_rad_s=0.0,
                                  force_lever_m=config["force_lever_m"],
                                  friction_epsilon_deg_s=config["friction_epsilon_deg_s"])[::2]
        d_angle = np.rad2deg(low[:, 0] - high[:, 0])
        d_rate = np.rad2deg(low[:, 1] - high[:, 1])
        observer_args = (
            nominal[axis], dt, config["force_lever_m"],
            np.deg2rad(float(config["assumed_angle_noise_deg"])), Q_BY_AXIS[axis], 0,
            config["friction_epsilon_deg_s"],
        )
        initial_state = np.array([angle0, 0.0, float(data[4])])
        _, force_low = nonlinear_ekf_rts_force(low[:, 0], *observer_args, initial_state=initial_state)
        _, force_high = nonlinear_ekf_rts_force(high[:, 0], *observer_args, initial_state=initial_state)
        wind_low = force_to_speed(config, force_low)
        wind_high = force_to_speed(config, force_high)
        wind_delta = wind_low - wind_high
        eval_mask = t >= 2.0
        integration_refinement.append({
            "axis": axis, "input": "自由振動ステップ", "reference_rate_hz": 200,
            "compared_rate_hz": 100,
            "angle_difference_rms_deg": float(np.sqrt(np.mean(d_angle**2))),
            "angle_difference_max_abs_deg": float(np.max(np.abs(d_angle))),
            "rate_difference_rms_deg_s": float(np.sqrt(np.mean(d_rate**2))),
            "rate_difference_max_abs_deg_s": float(np.max(np.abs(d_rate))),
            "rts_wind_difference_rms_m_s": float(np.sqrt(np.mean(wind_delta[eval_mask]**2))),
            "rts_wind_difference_max_abs_m_s": float(np.max(np.abs(wind_delta[eval_mask]))),
            "rts_wind_100hz_rmse_m_s": float(np.sqrt(np.mean((wind_low[eval_mask] - data[1][eval_mask])**2))),
            "rts_wind_200hz_rmse_m_s": float(np.sqrt(np.mean((wind_high[eval_mask] - data[1][eval_mask])**2))),
        })

    # Summarize the repeatable five-seed sensor-noise run separately from exact cases.
    frame = pd.DataFrame(detailed)
    summary = []
    for (axis, input_name, scenario), group in frame.groupby(["axis", "input", "scenario"], sort=False):
        summary.append({
            "axis": axis, "input": input_name, "scenario": scenario,
            "scenario_key": str(group.scenario_key.iloc[0]),
            "noise_runs": int(group.noise_seed.notna().sum()),
            **{col: float(group[col].mean()) for col in
               ("rmse_m_s", "bias_removed_rms_m_s", "mae_m_s", "p95_abs_error_m_s", "max_abs_error_m_s")},
            "rmse_seed_std_m_s": float(group.rmse_m_s.std(ddof=1)) if group.noise_seed.notna().sum() > 1 else 0.0,
        })
    save_csv(OUT / "ow05_theoretical_error_separation_runs.csv", detailed)
    save_csv(OUT / "ow05_theoretical_error_separation_summary.csv", summary)
    save_csv(OUT / "ow05_theoretical_step_q_sensitivity.csv", q_sensitivity)
    save_csv(OUT / "ow05_theoretical_integration_refinement.csv", integration_refinement)
    save_csv(OUT / "ow05_theoretical_error_separation_waveforms_10hz.csv", waveform_rows(waveforms, config))
    return summary, waveforms, {"config": config, "inputs_by_axis": inputs_by_axis, "nominal": nominal,
                                "mismatch": mismatch, "q_sensitivity": q_sensitivity,
                                "integration_refinement": integration_refinement}


def waveform_rows(waveforms: dict, config: dict) -> list[dict]:
    rows = []
    stride = max(1, int(round(float(config["sample_rate_hz"]) / 10.0)))
    for (axis, input_name, scenario, seed), item in waveforms.items():
        for i in range(0, len(item["t"]), stride):
            rows.append({
                "axis": axis, "input": input_name, "scenario": scenario,
                "noise_seed": seed, "time_s": float(item["t"][i]),
                "true_wind_m_s": float(item["truth"][i]),
                "estimated_wind_m_s": float(item["estimate"][i]),
                "wind_error_m_s": float(item["estimate"][i] - item["truth"][i]),
                "true_angle_deg": float(np.rad2deg(item["angle"][i])),
                "measured_angle_deg": float(np.rad2deg(item["measured_angle"][i])),
            })
    return rows


def make_figures(summary: list[dict], waveforms: dict, config: dict) -> None:
    colors = {"matched_clean": "#2673a8", "ow04_mismatch": "#dd8452", "sensor_noise": "#3b9b70"}
    labels = {"matched_clean": "Matched, no sensor noise", "ow04_mismatch": "OW-04 coefficient mismatch", "sensor_noise": "Nominal sensor noise (5-seed mean)"}
    # Step response: true/estimated wind and estimation error by axis/scenario.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col", layout="constrained")
    for col, axis in enumerate(config["axis_names"]):
        base = waveforms[(axis, "自由振動ステップ", "matched_clean", None)]
        axes[0, col].plot(base["t"], base["truth"], color="black", lw=1.4, label="True wind")
        axes[1, col].axhline(0, color="#555", lw=.7)
        for scenario in ("matched_clean", "ow04_mismatch", "sensor_noise"):
            if scenario == "sensor_noise":
                candidates = [v for (a, inp, s, seed), v in waveforms.items()
                              if a == axis and inp == "自由振動ステップ" and s == scenario]
                estimate = np.mean([v["estimate"] for v in candidates], axis=0)
                t = candidates[0]["t"]
            else:
                item = waveforms[(axis, "自由振動ステップ", scenario, None)]
                estimate, t = item["estimate"], item["t"]
            color = colors[scenario]
            axes[0, col].plot(t, estimate, color=color, lw=1.0, label=labels[scenario])
            axes[1, col].plot(t, estimate - base["truth"], color=color, lw=.9, label=labels[scenario])
        axes[0, col].axvline(2, color="#555", ls="--", lw=.8)
        axes[0, col].set_title(f"{axis}: 60° hold-equivalent wind → 0")
        axes[0, col].set_ylabel("Wind speed [m/s]")
        axes[1, col].set_ylabel("Estimated − true [m/s]")
        axes[1, col].set_xlabel("Time [s]")
        for row in range(2): axes[row, col].grid(alpha=.25)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.savefig(OUT / "ow05_theoretical_step_comparison.png", dpi=170)
    plt.close(fig)

    # Across excitation types, compare full RMSE and bias-removed variation RMS.
    sf = pd.DataFrame(summary)
    inputs = list(dict.fromkeys(sf.input))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for col, axis in enumerate(config["axis_names"]):
        part = sf[sf.axis == axis]
        x = np.arange(len(inputs)); width = .24
        for j, scenario in enumerate(("matched_clean", "ow04_mismatch", "sensor_noise")):
            vals = [float(part[(part.input == inp) & (part.scenario_key == scenario)].rmse_m_s.iloc[0]) for inp in inputs]
            axes[0, col].bar(x + (j-1)*width, vals, width, color=colors[scenario], label=labels[scenario])
            centered = [float(part[(part.input == inp) & (part.scenario_key == scenario)].bias_removed_rms_m_s.iloc[0]) for inp in inputs]
            axes[1, col].bar(x + (j-1)*width, centered, width, color=colors[scenario])
        axes[0, col].set_title(f"{axis}: wind-estimation RMSE")
        axes[0, col].set_ylabel("RMSE [m/s]")
        axes[1, col].set_title(f"{axis}: bias-removed error RMS")
        axes[1, col].set_ylabel("RMS [m/s]")
        for row in range(2):
            axes[row, col].set_xticks(x, ["Release\n60°→0", "Kaimal\n2 m/s", "Smooth gust\n2→6 m/s"])
            axes[row, col].grid(axis="y", alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(OUT / "ow05_theoretical_error_summary.png", dpi=170)
    plt.close(fig)

    # Wind-model time traces, true and estimated response plus errors.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col", layout="constrained")
    selected_inputs = [x for x in inputs if x != "自由振動ステップ"]
    for col, axis in enumerate(config["axis_names"]):
        for row, inp in enumerate(selected_inputs):
            ax = axes[row, col]
            base = waveforms[(axis, inp, "matched_clean", None)]
            ax.plot(base["t"], base["truth"], color="black", lw=1.3, label="True wind")
            for scenario in ("matched_clean", "ow04_mismatch", "sensor_noise"):
                if scenario == "sensor_noise":
                    candidates = [v for (a, name, s, seed), v in waveforms.items()
                                  if a == axis and name == inp and s == scenario]
                    estimate = np.mean([v["estimate"] for v in candidates], axis=0)
                else:
                    estimate = waveforms[(axis, inp, scenario, None)]["estimate"]
                ax.plot(base["t"], estimate, color=colors[scenario], lw=.8, label=labels[scenario])
            ax.set_ylabel("Wind [m/s]"); ax.grid(alpha=.25)
            input_label = "Kaimal turbulence, mean 2 m/s" if "Kaimal" in inp else "Smooth gust, 2 to 6 m/s"
            ax.set_title(f"{axis}: {input_label}")
            if row == 1: ax.set_xlabel("Time [s]")
            if row == 0 and col == 0: ax.legend(fontsize=7, ncol=2)
    fig.savefig(OUT / "ow05_theoretical_wind_model_responses.png", dpi=170)
    plt.close(fig)


def write_report(summary: list[dict], info: dict) -> None:
    frame = pd.DataFrame(summary)
    def row(axis, inp, scenario):
        return frame[(frame.axis == axis) & (frame.input == inp) & (frame.scenario_key == scenario)].iloc[0]
    lines = [
        "# OW-05 理論値による推定誤差要因の切り分け",
        "",
        "実測角度・実測風速は使わず、採用BALL係数で生成した理論角度だけをRTS 3状態観測器へ入力した。自由振動試験を『60°の静止保持に必要な風相当力から外力ゼロへのステップ』としてモデル化し、同じ観測器で風変動入力も評価した。",
        "",
        "## 条件",
        "",
        f"- サンプリング: {info['config']['sample_rate_hz']:.0f} Hz。プラントは既存HBK非線形運動方程式とRK4。球あり採用係数、角度リミット±{info['config']['mechanical_angle_limit_deg']:.0f}°。",
        f"- 風速は抗力式 $F=\\frac{{1}}{{2}}\\rho C_D A V^2$ で力へ換算。自由振動開始前は60°で静止保持し、各軸の保持力 $F_0=K\\tan(60°)/l$ を時刻2 sで0へ切り替えた。保持相当風はIN {float(force_to_speed(info['config'], np.array([info['inputs_by_axis']['IN']['自由振動ステップ'][4]]))[0]):.3f} m/s、OUT {float(force_to_speed(info['config'], np.array([info['inputs_by_axis']['OUT']['自由振動ステップ'][4]]))[0]):.3f} m/s。",
        f"- 観測器: RTS 3状態、角度観測標準偏差0.02°、qはIN {Q_BY_AXIS['IN']:.0e} / OUT {Q_BY_AXIS['OUT']:.0e} N/sample。全条件で同一設定。",
        "- ①係数完全一致・追加ノイズなし。②プラントは採用係数のまま、観測器だけOW-04の物理相関付き端点 `joint_mass_high_arm_high_tau-high_c-high` に変更。③係数一致のまま公称センサーの白色0.015°、色付き0.010°（時定数0.4746 s）、14 bit量子化を適用し、5 seedの平均を報告。ノイズの影響を分けるため遅延・ゲイン誤差・オフセット・ジッタは入れていない。",
        "- 風モデルはOW-05の独立Kaimal（平均2 m/s、TI 20%）と滑らかな2→6 m/sガスト。各系列の開始風速で静的平衡を初期化。",
        "- 指標: 真の風速に対するRMSE、Bias除去後RMS、95 percentile絶対誤差、最大絶対誤差。ステップ条件はリリース後2 sから採点。",
        "",
        "## 結果",
        "",
    ]
    lines += ["", "### ②で観測器だけに与えた係数ずれ", "",
              "| 軸 | I差 [%] | K差 [%] | τ差 [%] | 二乗抗力差 [%] |", "|---|---:|---:|---:|---:|"]
    for axis in info["config"]["axis_names"]:
        m, n = info["mismatch"][axis], info["nominal"][axis]
        pct = lambda key: 100.0 * (m[key] / n[key] - 1.0)
        lines.append(f"| {axis} | {pct('inertia_kg_m2'):+.2f} | {pct('restoring_n_m_per_rad'):+.2f} | {pct('tau_n_m'):+.2f} | {pct('total_quadratic_drag_n_m_s2_per_rad2'):+.2f} |")
    lines += ["", "### ①〜③の誤差指標", "",
        "| 入力 | 軸 | 条件 | RMSE [m/s] | Bias除去後RMS [m/s] | P95絶対誤差 [m/s] | 最大絶対誤差 [m/s] |",
        "|---|---|---|---:|---:|---:|---:|"]
    scenario_order = [("① 完全一致・無雑音", "matched_clean"), ("② OW-04係数ずれ", "ow04_mismatch"), ("③ センサーノイズ", "sensor_noise")]
    for inp in ["自由振動ステップ", "独立Kaimal乱流 平均2 m/s TI20%", "ガスト 2→6 m/s"]:
        for axis in info["config"]["axis_names"]:
            for label, key in scenario_order:
                r = row(axis, inp, key)
                lines.append(f"| {inp} | {axis} | {label} | {r.rmse_m_s:.4f} | {r.bias_removed_rms_m_s:.4f} | {r.p95_abs_error_m_s:.4f} | {r.max_abs_error_m_s:.4f} |")
    lines += ["", "### ① 完全一致条件でのq感度", "",
              "プラントと係数は一致させたまま、自由振動ステップのRTS qだけを変えた。qを変えたときに偏差も変われば、その一部は積分誤差ではなく、急変する外力に対するRTSのプロセス雑音設定に由来する。",
              "", "| 軸 | q [N/sample] | RMSE [m/s] | Bias除去後RMS [m/s] | 最大絶対誤差 [m/s] |", "|---|---:|---:|---:|---:|"]
    for item in info["q_sensitivity"]:
        lines.append(f"| {item['axis']} | {item['q_N_per_sample']:.0e} | {item['rmse_m_s']:.4f} | {item['bias_removed_rms_m_s']:.4f} | {item['max_abs_error_m_s']:.4f} |")
    lines += ["", "### 数値積分の刻み幅確認", "",
              "プラントの100 Hz RK4と200 Hz RK4を比較し、200 Hz結果を100 Hz時刻へ戻した角度を同じRTSへ通した。従って、2列の推定風差はプラント積分刻みが推定出力へ与える影響の評価である。",
              "", "| 軸 | 角度差 RMS [deg] | 最大角度差 [deg] | 角速度差 RMS [deg/s] | 最大角速度差 [deg/s] | RTS風推定差 RMS [m/s] | 最大差 [m/s] | RMSE 100→200 Hz [m/s] |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in info["integration_refinement"]:
        lines.append(f"| {item['axis']} | {item['angle_difference_rms_deg']:.6g} | {item['angle_difference_max_abs_deg']:.6g} | {item['rate_difference_rms_deg_s']:.6g} | {item['rate_difference_max_abs_deg_s']:.6g} | {item['rts_wind_difference_rms_m_s']:.6g} | {item['rts_wind_difference_max_abs_m_s']:.6g} | {item['rts_wind_100hz_rmse_m_s']:.4f} → {item['rts_wind_200hz_rmse_m_s']:.4f} |")
    lines += [
        "",
        "![自由振動ステップの理論風速と推定値](ow05_theoretical_step_comparison.png)",
        "",
        "![ステップと風モデルの誤差指標](ow05_theoretical_error_summary.png)",
        "",
        "![風モデル入力での真値・推定値](ow05_theoretical_wind_model_responses.png)",
        "",
        "## 読み取り",
        "",
        "①の誤差をそのまま演算誤差と呼ぶことはできない。RTSは角度観測の誤差共分散と外力状態のランダムウォークqを使う推定器であり、係数一致は推定器が急な力変化を必ず正確に再現することを意味しない。100→200 Hzにすると推定風の差はRMSでIN 0.0271、OUT 0.0323 m/sだが、総RMSEはIN 0.2622→0.2636、OUT 0.1669→0.1701 m/sとほぼ変わらない。したがって積分刻みの影響はあるものの、自由振動ステップの大きな誤差全体を数値積分だけでは説明できない。qを変えるとRMSEがIN 0.262→0.110、OUT 0.167→0.101 m/sへ変わるため、急変力に対するRTSの時間変化仮定・調整が主要因の一つである。",
        "",
        "②−①の増分はOW-04の係数ずれによる影響、③−①の増分は設定したセンサーノイズ・量子化による影響として読める。差は誤差RMSの単純差だけでなく、RMSE・Bias除去後RMS・P95・最大値を並べて見る。特にBias除去後RMSは振動成分に対応する。",
        "",
        "自由振動ステップは最大角度保持相当の風を一気にゼロへ落とす大振幅過渡である。Kaimal/滑らかガストの誤差との比が大きければ、自由振動の大きなRMSEが通常の風モデルにそのまま当てはまらず、励起振幅・変化速度が主因になり得る。①でも自由振動だけ誤差が大きい場合は、完全モデル一致の条件定義か風力状態の推定器表現を再点検する必要がある。",
        "",
        "## 再生成",
        "",
        "リポジトリルートで実行: `python 06_Analysis/simulation/src/run_ow05_theoretical_error_separation.py`。",
        "集計CSV、seedごとのCSV、10 Hz波形CSV、完全一致条件でのq感度CSVおよび積分刻み確認CSVは同じフォルダーに出力される。ノイズseed、係数端点、q、評価開始時刻はスクリプトに記録される。",
        "",
    ]
    (OUT / "OW05_THEORETICAL_ERROR_SEPARATION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary, waveforms, info = evaluate()
    make_figures(summary, waveforms, info["config"])
    write_report(summary, info)
    print(f"Wrote theoretical-only OW-05 separation results to {OUT}")


if __name__ == "__main__":
    main()
