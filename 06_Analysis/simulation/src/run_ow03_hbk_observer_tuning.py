"""OW-03: tune observers on one wind record and validate on new records.

This script uses the adopted nonlinear HBK coefficients for the BALL setup.
The plant and observer use the same mechanical coefficients (model match),
and the simulated angle sensor is ideal.  Observer settings are selected on
one Kaimal training wind record only.  They are then frozen and evaluated on
different calm, steady, turbulent, and gust inputs.

Run at the repository root:
    python 06_Analysis/simulation/src/run_ow03_hbk_observer_tuning.py

The output folder receives the full parameter scan, evaluation tables, a
JSON summary, comparison plots, and a Japanese report.  Sensor noise,
quantization, and delay are intentionally left for OW-05.  If a simulated
angle exceeds the physical +/-60 degree range, that case is marked invalid
for the hardware and is not silently clipped.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from hbk_model_coefficients import select_coefficients, simulate_hbk_plant
from hbk_nonlinear_estimators import (
    calculate_luenberger_gain,
    nonlinear_ekf_rts_force,
    nonlinear_luenberger_force,
)
from run_real_wind_doe import causal_lowpass, wind_speed_from_force
from wind import drag_force_from_speed, synthesize_kaimal_wind


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "ow03_hbk_observer_tuning.json"
DEFAULT_OUTPUT = ROOT / "results" / "observer_wind" / "ow03_tuning"
METHODS = (
    "静的換算", "因果LPF", "ESO 3状態", "ESO 4状態",
    "RTS 3状態（オフライン）", "RTS 4状態（オフライン）",
)
CASE_LABELS = {
    "無風": "Calm",
    "一定風 2 m/s": "Steady 2 m/s",
    "Kaimal乱流 平均3.75 m/s TI20% 最大6 m/s": "Kaimal 3.75 m/s, TI 20%, max 6 m/s",
    "独立Kaimal乱流 平均2 m/s TI20%": "Kaimal 2 m/s, TI 20%",
    "ガスト 2→6 m/s": "Gust 2 to 6 m/s",
}
COLORS = {
    "静的換算": "#777777",
    "因果LPF": "#8e6bbd",
    "ESO 3状態": "#2776bc",
    "ESO 4状態": "#e67e22",
    "RTS 3状態（オフライン）": "#159477",
    "RTS 4状態（オフライン）": "#006b4f",
}
LABELS = {
    "静的換算": "Static",
    "因果LPF": "Causal LPF",
    "ESO 3状態": "ESO 3-state",
    "ESO 4状態": "ESO 4-state",
    "RTS 3状態（オフライン）": "RTS 3-state (offline)",
    "RTS 4状態（オフライン）": "RTS 4-state (offline)",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a table with its column names; UTF-8 BOM opens well in Excel."""
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_wind(config: dict, description: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create one of the documented, repeatable wind records."""
    fs, duration = config["sample_rate_hz"], config["duration_s"]
    count = int(round(fs * duration))
    time = np.arange(count, dtype=float) / fs
    kind = description["kind"]
    if kind == "calm":
        speed = np.zeros(count)
        info = {"mean_m_s": 0.0, "realized_ti": 0.0, "minimum_m_s": 0.0,
                "maximum_m_s": 0.0, "seed": description["seed"]}
    elif kind == "steady":
        speed = np.full(count, description["mean_wind_speed_m_s"])
        info = {"mean_m_s": float(np.mean(speed)), "realized_ti": 0.0,
                "minimum_m_s": float(np.min(speed)), "maximum_m_s": float(np.max(speed)),
                "seed": description["seed"]}
    elif kind == "kaimal":
        time, speed, info = synthesize_kaimal_wind(
            fs, duration, description["mean_wind_speed_m_s"],
            description["target_turbulence_intensity"],
            config["kaimal_integral_scale_m"], description["seed"],
            description["maximum_wind_speed_m_s"],
        )
        info["mean_m_s"] = float(np.mean(speed))
        info["seed"] = description["seed"]
    elif kind == "gust":
        # Two smooth Gaussian gusts exercise both rising and falling response.
        speed = np.full(count, description["base_wind_speed_m_s"], dtype=float)
        for center_s, width_s in ((35.0, 5.0), (78.0, 7.0)):
            pulse = np.exp(-0.5 * ((time - center_s) / width_s) ** 2)
            speed += (description["gust_peak_m_s"] - description["base_wind_speed_m_s"]) * pulse
        speed = np.minimum(speed, description["gust_peak_m_s"])
        info = {"mean_m_s": float(np.mean(speed)), "realized_ti": float(np.std(speed) / np.mean(speed)),
                "minimum_m_s": float(np.min(speed)), "maximum_m_s": float(np.max(speed)),
                "seed": description["seed"]}
    else:
        raise ValueError(f"Unknown wind kind: {kind}")
    return time, speed, info


def applied_force(config: dict, speed: np.ndarray) -> np.ndarray:
    """Convert the known wind speed into the force applied to the plant."""
    return drag_force_from_speed(
        speed, config["air_density_kg_m3"], config["drag_coefficient"],
        config["projected_area_m2"],
    )


def metrics(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Calculate error measures on the stated evaluation interval."""
    error = np.asarray(estimate)[mask] - np.asarray(truth)[mask]
    if not np.all(np.isfinite(error)):
        return {"rmse_m_s": None, "bias_m_s": None, "mae_m_s": None,
                "p95_abs_error_m_s": None, "max_abs_error_m_s": None}
    absolute = np.abs(error)
    return {
        "rmse_m_s": float(np.sqrt(np.mean(error**2))),
        "bias_m_s": float(np.mean(error)),
        "mae_m_s": float(np.mean(absolute)),
        "p95_abs_error_m_s": float(np.quantile(absolute, 0.95)),
        "max_abs_error_m_s": float(np.max(absolute)),
    }


def peak_delay_s(time: np.ndarray, truth: np.ndarray, estimate: np.ndarray,
                 mask: np.ndarray) -> float | None:
    """Return estimated-minus-true peak time; zero-wind inputs have no peak."""
    indices = np.flatnonzero(mask)
    if not len(indices) or np.ptp(truth[mask]) < 1e-8:
        return None
    if not np.all(np.isfinite(estimate[mask])):
        return None
    truth_peak = indices[int(np.argmax(truth[mask]))]
    estimate_peak = indices[int(np.argmax(estimate[mask]))]
    return float(time[estimate_peak] - time[truth_peak])


def static_force(angle: np.ndarray, coefficients: dict, lever_m: float) -> np.ndarray:
    """Use the static restoring-torque balance as a simple baseline."""
    return coefficients["restoring_n_m_per_rad"] / lever_m * np.tan(angle)


def estimate_methods(angle: np.ndarray, coefficients: dict, config: dict,
                     tuning: dict, dt: float) -> dict[str, np.ndarray]:
    """Run all comparison methods with the currently selected settings."""
    static = static_force(angle, coefficients, config["force_lever_m"])
    def safe_force_estimate(function) -> np.ndarray:
        try:
            estimate = function()
            if np.all(np.isfinite(estimate)):
                return estimate
        except (ValueError, OverflowError, FloatingPointError):
            pass
        return np.full_like(angle, np.nan, dtype=float)

    rts3 = safe_force_estimate(lambda: nonlinear_ekf_rts_force(
        angle, coefficients, dt, config["force_lever_m"],
        np.deg2rad(config["assumed_angle_noise_deg"]), tuning["rts3_q"], 0,
        config["friction_epsilon_deg_s"],
    )[1])
    rts4 = safe_force_estimate(lambda: nonlinear_ekf_rts_force(
        angle, coefficients, dt, config["force_lever_m"],
        np.deg2rad(config["assumed_angle_noise_deg"]), tuning["rts4_q"], 1,
        config["friction_epsilon_deg_s"],
    )[1])
    eso3 = safe_force_estimate(lambda: nonlinear_luenberger_force(
        angle, coefficients, dt, config["force_lever_m"],
        tuning["eso3_hz"], 0, config["friction_epsilon_deg_s"],
    ))
    eso4 = safe_force_estimate(lambda: nonlinear_luenberger_force(
        angle, coefficients, dt, config["force_lever_m"],
        tuning["eso4_hz"], 1, config["friction_epsilon_deg_s"],
    ))
    result = {
        "静的換算": static,
        "因果LPF": causal_lowpass(static, tuning["lpf_hz"], dt),
        "ESO 3状態": eso3,
        "ESO 4状態": eso4,
        "RTS 3状態（オフライン）": rts3,
        "RTS 4状態（オフライン）": rts4,
    }
    # Preserve an unstable observer as NaNs so it is reported as a failure,
    # rather than aborting the evaluation of other estimators.
    return result


def force_to_speed(config: dict, force: np.ndarray) -> np.ndarray:
    return wind_speed_from_force(
        force, config["air_density_kg_m3"], config["drag_coefficient"],
        config["projected_area_m2"],
    )


def tune_axis(angle: np.ndarray, true_speed: np.ndarray, true_force: np.ndarray,
              coefficients: dict, config: dict, mask: np.ndarray,
              dt: float) -> tuple[dict, list[dict]]:
    """Select each tunable setting by wind-speed RMSE on training data only."""
    rows: list[dict] = []
    q_noise = np.deg2rad(config["assumed_angle_noise_deg"])
    tuning_grid = (
        ("ESO 3状態", "極周波数", "eso_pole_grid_hz", "eso3_hz", 0),
        ("ESO 4状態", "極周波数", "eso_pole_grid_hz", "eso4_hz", 1),
        ("RTS 3状態（オフライン）", "風力プロセス雑音", "rts3_force_random_walk_grid_n_per_sample", "rts3_q", 0),
        ("RTS 4状態（オフライン）", "力変化率プロセス雑音", "rts4_force_rate_random_walk_grid_n_per_s_per_sample", "rts4_q", 1),
        ("因果LPF", "遮断周波数", "lpf_cutoff_grid_hz", "lpf_hz", -1),
    )
    static_estimate = force_to_speed(config, static_force(angle, coefficients, config["force_lever_m"]))
    rows.append({"method": "静的換算", "parameter_name": "なし", "parameter": 0.0,
                 "stable": True, **metrics(true_speed, static_estimate, mask)})

    for method, parameter_name, grid_key, setting_key, order in tuning_grid:
        for value in config[grid_key]:
            try:
                if order == -1:
                    estimated_force = causal_lowpass(
                        static_force(angle, coefficients, config["force_lever_m"]), value, dt
                    )
                elif method.startswith("ESO"):
                    estimated_force = nonlinear_luenberger_force(
                        angle, coefficients, dt, config["force_lever_m"], value,
                        order, config["friction_epsilon_deg_s"],
                    )
                else:
                    _, estimated_force = nonlinear_ekf_rts_force(
                        angle, coefficients, dt, config["force_lever_m"], q_noise,
                        value, order, config["friction_epsilon_deg_s"],
                    )
                estimate = force_to_speed(config, estimated_force)
            except (ValueError, OverflowError, FloatingPointError):
                # Treat numerical blow-up as an unstable tuning candidate.
                estimate = np.full_like(true_speed, np.nan, dtype=float)
            finite = bool(np.all(np.isfinite(estimate)))
            stable = bool(finite and np.max(np.abs(estimate)) <= 20.0)
            # Large finite runaway estimates are kept in the scan as failures,
            # but cannot win parameter selection.
            row_metrics = metrics(true_speed, estimate, mask) if finite else {
                "rmse_m_s": None, "bias_m_s": None,
                "mae_m_s": None, "p95_abs_error_m_s": None,
                "max_abs_error_m_s": None,
            }
            rows.append({"method": method, "parameter_name": parameter_name,
                         "parameter": float(value), "stable": stable, **row_metrics})
    best = {"静的換算": {"parameter": None}}
    for method, _, _, setting_key, _ in tuning_grid:
        candidates = [row for row in rows if row["method"] == method]
        stable_candidates = [row for row in candidates if row["stable"]]
        pool = stable_candidates or [row for row in candidates if row["rmse_m_s"] is not None]
        if not pool:
            # The sweep had only numerical blow-ups. Keep a documented
            # diagnostic starting point to show the failure in validation.
            pool = candidates
            selected = pool[0]
        else:
            selected = min(pool, key=lambda row: row["rmse_m_s"] if row["rmse_m_s"] is not None else float("inf"))
        best[method] = {"parameter": selected["parameter"], "parameter_name": selected["parameter_name"],
                        "setting_key": setting_key, "training_rmse_m_s": selected["rmse_m_s"],
                        "stable_candidate_found": bool(stable_candidates)}
    return best, rows


def best_parameter_values(best: dict) -> dict[str, float]:
    return {
        "eso3_hz": best["ESO 3状態"]["parameter"],
        "eso4_hz": best["ESO 4状態"]["parameter"],
        "rts3_q": best["RTS 3状態（オフライン）"]["parameter"],
        "rts4_q": best["RTS 4状態（オフライン）"]["parameter"],
        "lpf_hz": best["因果LPF"]["parameter"],
    }


def save_plots(output: Path, config: dict, train_time: np.ndarray,
               train_speed: np.ndarray, scan_rows: dict[str, list[dict]],
               evaluation: dict, metrics_rows: list[dict]) -> None:
    """Create tuning curves and an independent validation waveform figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, method in zip(axes.flat, (
        "ESO 3状態", "ESO 4状態", "RTS 3状態（オフライン）",
        "RTS 4状態（オフライン）", "因果LPF", "静的換算",
    )):
        for axis_name, line_style in (("IN", "-"), ("OUT", "--")):
            rows = sorted((r for r in scan_rows[axis_name]
                           if r["method"] == method and r["parameter"] > 0 and r["stable"]),
                          key=lambda row: row["parameter"])
            if rows:
                ax.plot([r["parameter"] for r in rows], [r["rmse_m_s"] for r in rows],
                        marker="o", linestyle=line_style, label=f"{axis_name} axis")
        if method != "静的換算":
            ax.set_xscale("log")
        else:
            ax.text(0.5, 0.5, "No tuning parameter", ha="center", va="center", transform=ax.transAxes)
        ax.set(title=LABELS[method], xlabel="Tuning parameter (log scale)", ylabel="Training wind RMSE [m/s]")
        ax.grid(alpha=0.25)
        if method != "静的換算":
            ax.legend(fontsize=8)
    fig.suptitle("OW-03 tuning scan (training wind only)")
    fig.savefig(output / "ow03_tuning_scan.png", dpi=160)
    plt.close(fig)

    case_names = list(evaluation)
    fig, axes = plt.subplots(len(case_names), 2, figsize=(15, 3.0 * len(case_names)),
                             sharex=True, layout="constrained")
    for row_idx, case_name in enumerate(case_names):
        case = evaluation[case_name]
        selected = case["mask"] & (case["time"] <= config["evaluation_start_s"] + 45.0)
        local_time = case["time"][selected] - config["evaluation_start_s"]
        for col, axis_name in enumerate(("IN", "OUT")):
            ax = axes[row_idx, col]
            ax.plot(local_time, case["speed"][selected], color="black", linewidth=1.6, label="True wind")
            for method, estimate_speed in case["speed_estimates"][axis_name].items():
                ax.plot(local_time, estimate_speed[selected], color=COLORS[method],
                        linewidth=0.85, alpha=0.9, label=LABELS[method])
            ax.set_ylabel("Wind speed [m/s]")
            ax.set_title(f"{CASE_LABELS[case_name]} / {axis_name} axis")
            ax.grid(alpha=0.2)
            if row_idx == 0 and col == 0:
                ax.legend(ncol=3, fontsize=7)
            if row_idx == len(case_names) - 1:
                ax.set_xlabel("Time from evaluation-window start [s]")
    fig.suptitle("OW-03 independent wind validation (first 45 s of evaluation interval)")
    fig.savefig(output / "ow03_validation_waveforms.png", dpi=100)
    plt.close(fig)

    # A concise, axis-by-case RMSE comparison complements the waveform panels.
    cases = list(dict.fromkeys(row["case"] for row in metrics_rows))
    methods = [m for m in METHODS if m != "静的換算"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, layout="constrained")
    x = np.arange(len(cases))
    width = 0.12
    for ax, axis_name in zip(axes, ("IN", "OUT")):
        for index, method in enumerate(methods):
            values = []
            for case in cases:
                found = next(r for r in metrics_rows if r["axis"] == axis_name and r["case"] == case and r["method"] == method)
                values.append(np.nan if found["rmse_m_s"] is None else found["rmse_m_s"])
            ax.bar(x + (index - len(methods) / 2) * width, values, width, color=COLORS[method], label=LABELS[method])
        ax.set_xticks(x, [CASE_LABELS[case] for case in cases], rotation=25, ha="right")
        ax.set_ylabel("Wind-speed RMSE [m/s]")
        ax.set_title(f"{axis_name} axis")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Independent validation errors by wind case")
    fig.savefig(output / "ow03_validation_rmse.png", dpi=160)
    plt.close(fig)


def write_report(path: Path, config: dict, training_info: dict,
                 tuning_by_axis: dict, scan_rows: dict,
                 validation_info: dict, metrics_rows: list[dict],
                 max_angle: dict, gains: dict) -> None:
    """Write a Japanese report, including enough detail to reproduce tuning."""
    lines = [
        "# OW-03 新ハード係数でのオブザーバー調整と独立評価", "",
        "## 目的と結論", "",
        "OW-02で旧調整値のまま比較したESOの極周波数等を、新しいHBK係数に合わせて再調整した。調整に使う風系列と、性能を確かめる風系列は独立させた。プラントと推定モデルの機械係数は一致させ、角度センサーは理想値とした。したがって本タスクはオブザーバー調整値と入力風条件の影響を比較する基準評価であり、係数誤差はOW-04、センサー非理想性はOW-05で扱う。", "",
        "調整では推定風速RMSEが小さくなるパラメーターを各軸・各方式ごとに選んだ。評価用風系列で各方式の性能を比較し、ESO極周波数を旧設定から変更した場合の発散・追従性を確認した。RTSは将来データを用いるオフライン方式のため、オンラインESOとは分けて解釈する。", "",
        "結果として3状態ESOはIN 49.9 Hz、OUT 40 Hzが選ばれた。IN軸は探索上限49.9 Hzが選択されており、ノイズなし条件で速い応答ほど有利になる傾向が残っているため、最終値とはみなさない。4状態ESOはIN 12 Hzだったが、OUTでは探索したどの極周波数も有界性スクリーニングを通らず、3 Hzを診断用に記録した。OUT軸4状態ESOの誤差は大きく、今回の極周波数調整だけでは解消しなかった。センサー非理想性を含む最終調整はOW-05で行う。", "",
        "## 条件と仮定", "",
        f"- ハード構成：{config['configuration']}。係数：採用済みHBK台帳のIN/OUT別値。プラントと推定側で一致。",
        f"- サンプリング周波数：{config['sample_rate_hz']:.1f} Hz。各記録長：{config['duration_s']:.0f} s。先頭{config['evaluation_start_s']:.0f} sを初期過渡として評価から除外。",
        f"- 機械角リミット：±{config['mechanical_angle_limit_deg']:.0f}°。シミュレーション波形はクリップせず、超過した条件は実機で実現不能としてフラグする。",
        f"- 空力：球Cd={config['drag_coefficient']:.6f}、投影面積={config['projected_area_m2']:.8f} m²、力の作用腕={config['force_lever_m']*1000:.2f} mm。空気密度={config['air_density_kg_m3']:.3f} kg/m³。",
        "- センサー：理想角度をプラント角から直接使用。計測ノイズ・量子化・遅延は加えていない。RTS内部の観測ノイズ設定だけに角度標準偏差0.020°を仮定する。",
        "- 損失：採用HBK係数の `c` と軸別 `tau`、`b=0`。クーロン摩擦はOW-02と同じ0.5°/sのtanh近似。",
        "- 静的換算はオブザーバーではなく基準方式。LPFは静的換算力を3段因果フィルターで平滑化。RTSは全記録の未来角度を利用する。",
        "",
        "## 調整・評価データの分離", "",
        f"調整用は平均{config['tuning_wind']['mean_wind_speed_m_s']:.2f} m/s、目標TI {100*config['tuning_wind']['target_turbulence_intensity']:.1f}%、最大{config['tuning_wind']['maximum_wind_speed_m_s']:.1f} m/sのKaimal系列（seed={config['tuning_wind']['seed']}）。各軸でこれだけを使ってESO極周波数、RTSのプロセス雑音設定、LPF遮断周波数を選んだ。調整指標は評価区間の風速RMSEである。",
        "評価は調整系列と異なるseedのKaimal系列、定常風、ガスト、無風で行った。選んだ設定は評価時に固定し直さない。", "",
        "| 用途 | 入力 | 再現条件 |",
        "|---|---|---|",
        f"| 調整 | {config['tuning_wind']['kind']} | 平均{config['tuning_wind']['mean_wind_speed_m_s']:.2f} m/s、TI {100*config['tuning_wind']['target_turbulence_intensity']:.1f}%、上限{config['tuning_wind']['maximum_wind_speed_m_s']:.1f} m/s、seed {config['tuning_wind']['seed']} |",
        *[f"| 評価 | {case['name']} | {case['kind']}、seed {case['seed']} |" for case in config["validation_winds"]],
        "", "## 何を調整したか", "",
        "3状態ESOは `[theta, omega, F]`、4状態ESOは `[theta, omega, F, r_F]` を推定する。ESOでは繰返し離散極周波数を探索し、その周波数と新ハードの係数から補正ゲインベクトル `L` を計算する。RTS 3状態では風力状態に加えるプロセス雑音、RTS 4状態では風力変化率状態に加えるプロセス雑音を探索した。LPFは遮断周波数を探索した。静的換算には調整値がない。", "",
        "| 軸 | 方式 | 選択パラメーター | 調整用風速RMSE (m/s) | 実ゲインベクトル L |",
        "|---|---|---:|---:|---|",
    ]
    for axis in ("IN", "OUT"):
        for method in ("ESO 3状態", "ESO 4状態", "RTS 3状態（オフライン）", "RTS 4状態（オフライン）", "因果LPF"):
            result = tuning_by_axis[axis][method]
            value_text = "—" if result["parameter"] is None else f"{result['parameter']:.8g} ({result['parameter_name']})"
            if not result.get("stable_candidate_found", True):
                value_text += "（合格候補なし・診断用）"
            gain_text = "—"
            if method in ("ESO 3状態", "ESO 4状態"):
                vector = gains[axis][method]
                gain_text = "`[" + ", ".join(f"{v:.8g}" for v in vector) + "]^T`"
            score = "—" if result.get("training_rmse_m_s") is None else f"{result['training_rmse_m_s']:.5f}"
            lines.append(f"| {axis} | {method} | {value_text} | {score} | {gain_text} |")
    lines += [
        "", "ここでESOの極周波数は調整結果、列ベクトル `L` はその極周波数と各軸の採用 `I,K,c,tau` から計算した実際の補正ゲインである。ゲインの成分順はESO状態と同じで、角度残差（rad）に掛ける。RTSの雑音設定値は指定単位で示した。OW-02の45 Hz/20 Hzは初期値であり、ここに示すOW-03の値が調整後の値となる。", "",
        "### 調整走査", "",
        "![調整用系列だけで計算したパラメーター走査](ow03_tuning_scan.png)", "",
        "グラフの横軸は各方式の調整パラメーター（対数目盛）、縦軸は調整用風入力に対する推定風速RMSEである。選択は有界性スクリーニングを通過した点のRMSE最小値とした。破線はOUT軸、実線はIN軸を表す。", "",
        "ESO候補の有界性スクリーニングは、推定値が全標本で有限であり、かつ絶対推定風速が20 m/s以下（調整風上限5 m/sの4倍以下）であることとした。これは探索を打ち切る暴走判定の目安であり、数学的な安定性証明ではない。条件を通過しない場合は最小RMSE候補を診断用として記録し、合格した調整値とは扱わない。", "",
        "## 独立した評価結果", "",
        "風速誤差は推定風速から既知の入力風速を引いた値。`ピーク時刻差` は推定最大風速の時刻から入力風速の最大時刻を引いた値で、正なら推定が遅い。無風・一定風ケースでは真値に変動ピークがないためピーク時刻差を評価しない。", "",
        "| ケース | 軸 | 方式 | 風速RMSE (m/s) | 偏り (m/s) | MAE (m/s) | 95%絶対誤差 (m/s) | 最大絶対誤差 (m/s) | ピーク時刻差 (s) | 最大角度 | ±60°以内 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics_rows:
        max_deg = max_angle[row["case"]][row["axis"]]
        within = "はい" if max_deg <= config["mechanical_angle_limit_deg"] else "いいえ（範囲外）"
        delay = "—" if row["peak_delay_s"] is None else f"{row['peak_delay_s']:.3f}"
        cells = ["発散/算出不可" if row[key] is None else f"{row[key]:.5f}"
                 for key in ("rmse_m_s", "bias_m_s", "mae_m_s", "p95_abs_error_m_s", "max_abs_error_m_s")]
        lines.append(
            f"| {row['case']} | {row['axis']} | {row['method']} | "
            f"{cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
            f"{cells[4]} | {delay} | {max_deg:.2f}° | {within} |"
        )
    lines += [
        "", "### 入力風と推定風の重ね描き", "",
        "![異なる評価入力に対する実風と各方式の推定風](ow03_validation_waveforms.png)", "",
        "### ケースごとのRMSE", "",
        "![入力ケース・軸ごとの風速RMSE](ow03_validation_rmse.png)", "",
        "## 結果の読み方と制約", "",
        "各方式の設定は1つの調整用Kaimal系列で選んだ後に固定しているため、評価ケース上で再選択していない。ガストや別乱流で誤差が増える場合、設定の過学習・モデル近似・帯域の限界などを示す。RTSは未来サンプルを使うので、オンラインESOと誤差値だけで順位付けしない。", "",
        "無風ケースは理想センサーかつ角度ゼロ初期条件であり、ここでの誤推定ゼロはノイズ下の性能を意味しない。センサーノイズ、量子化、遅延はOW-05で評価する。係数誤差はOW-04で評価する。", "",
        "角度リミットを越えた場合、現状の非線形プラント計算は機械ストッパーによる衝突や飽和を模擬しない。該当ケースの推定性能は実機の挙動予測として採用しない。", "",
        "## 再実行", "",
        "リポジトリルートで実行する。調整格子や学習・評価風条件はJSON設定ファイルに保存されている。", "",
        "```bash", "python 06_Analysis/simulation/src/run_ow03_hbk_observer_tuning.py", "```", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["sample_rate_hz"] <= 0 or config["duration_s"] <= config["evaluation_start_s"]:
        raise ValueError("Sampling rate and evaluation duration are inconsistent")
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / config["sample_rate_hz"]
    mask = np.arange(int(round(config["duration_s"] * config["sample_rate_hz"]))) / config["sample_rate_hz"] >= config["evaluation_start_s"]

    training_time, training_speed, training_info = make_wind(config, config["tuning_wind"])
    training_force = applied_force(config, training_speed)
    tuning_by_axis, scan_by_axis, gains = {}, {}, {}
    coefficients_by_axis = {}
    for axis in config["axis_names"]:
        coefficients = select_coefficients(axis, config["configuration"])
        coefficients_by_axis[axis] = coefficients
        state = simulate_hbk_plant(
            training_force, coefficients, dt, force_lever_m=config["force_lever_m"],
            friction_epsilon_deg_s=config["friction_epsilon_deg_s"],
        )
        best, rows = tune_axis(state[:, 0], training_speed, training_force,
                               coefficients, config, mask, dt)
        tuning_by_axis[axis] = best
        scan_by_axis[axis] = rows
        settings = best_parameter_values(best)
        gains[axis] = {}
        for method, key, order in (("ESO 3状態", "eso3_hz", 0), ("ESO 4状態", "eso4_hz", 1)):
            vector = calculate_luenberger_gain(
                coefficients, config["force_lever_m"], dt, settings[key], order,
                np.deg2rad(config["friction_epsilon_deg_s"]),
            )
            gains[axis][method] = vector.reshape(-1).tolist()

    evaluation, metrics_rows, max_angle = {}, [], {}
    evaluation_info = {}
    for case_config in config["validation_winds"]:
        case_name = case_config["name"]
        time, speed, metadata = make_wind(config, case_config)
        force = applied_force(config, speed)
        evaluation_info[case_name] = metadata
        evaluation[case_name] = {"time": time, "speed": speed, "mask": mask, "speed_estimates": {}}
        max_angle[case_name] = {}
        for axis in config["axis_names"]:
            coefficients = coefficients_by_axis[axis]
            state = simulate_hbk_plant(
                force, coefficients, dt, force_lever_m=config["force_lever_m"],
                friction_epsilon_deg_s=config["friction_epsilon_deg_s"],
            )
            angle = state[:, 0]
            max_angle[case_name][axis] = float(np.max(np.abs(np.rad2deg(angle[mask]))))
            force_estimates = estimate_methods(
                angle, coefficients, config, best_parameter_values(tuning_by_axis[axis]), dt,
            )
            speed_estimates = {method: force_to_speed(config, estimate)
                               for method, estimate in force_estimates.items()}
            evaluation[case_name]["speed_estimates"][axis] = speed_estimates
            for method, estimate_speed in speed_estimates.items():
                metrics_rows.append({
                    "case": case_name, "axis": axis, "method": method,
                    **metrics(speed, estimate_speed, mask),
                    "peak_delay_s": peak_delay_s(time, speed, estimate_speed, mask),
                    "maximum_abs_angle_deg": max_angle[case_name][axis],
                    "within_mechanical_limit": max_angle[case_name][axis] <= config["mechanical_angle_limit_deg"],
                })

    scan_rows = [row for axis in config["axis_names"] for row in scan_by_axis[axis]]
    write_csv(output_dir / "ow03_tuning_scan.csv", scan_rows)
    write_csv(output_dir / "ow03_validation_metrics.csv", metrics_rows)
    save_plots(output_dir, config, training_time, training_speed, scan_by_axis,
               evaluation, metrics_rows)
    summary = {
        "task_id": "OW-03", "configuration": config["configuration"],
        "coefficient_match": True, "ideal_sensor": True,
        "tuning_wind": {**config["tuning_wind"], "metadata": training_info},
        "validation_winds": evaluation_info,
        "selected_tunings_by_axis": tuning_by_axis,
        "eso_gain_vectors_by_axis": gains,
        "validation_metrics": metrics_rows,
        "maximum_abs_angle_deg": max_angle,
        "mechanical_angle_limit_deg": config["mechanical_angle_limit_deg"],
        "interpretation": "Settings selected on the independent tuning record and held fixed for all validation cases. Sensor nonidealities and coefficient mismatch are deferred to OW-05 and OW-04 respectively.",
    }
    (output_dir / "ow03_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "OW-03_REPORT.md", config, training_info,
                 tuning_by_axis, scan_by_axis, evaluation_info, metrics_rows,
                 max_angle, gains)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OW-03 observer tuning and independent validation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.config, args.output_dir)
    for axis, result in summary["selected_tunings_by_axis"].items():
        print(f"{axis} selected settings: " + json.dumps(result, ensure_ascii=False))
    print(f"Report: {args.output_dir / 'OW-03_REPORT.md'}")


if __name__ == "__main__":
    main()
