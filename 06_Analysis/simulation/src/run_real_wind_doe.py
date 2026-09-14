"""Compare four angle-only force estimators under Kaimal-spectrum wind."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import qmc

from doe import features
from estimators import causal_luenberger, kalman_rts_force
from model import PendulumParameters, natural_characteristics
from stage2_system import plant
from wind import drag_force_from_speed, kaimal_longitudinal_psd, synthesize_kaimal_wind


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("ESO 3-state", "ESO 4-state", "RTS 3-state", "RTS 4-state")
BASELINES = ("Raw static", "Causal LPF")
COLORS = {
    "True force": "#111111",
    "Raw static": "#9e9e9e",
    "Causal LPF": "#8e6bbd",
    "ESO 3-state": "#2776bc",
    "ESO 4-state": "#e67e22",
    "RTS 3-state": "#159477",
    "RTS 4-state": "#006b4f",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decode_parameters(
    nominal: PendulumParameters, coded: np.ndarray, half_ranges: np.ndarray
) -> PendulumParameters:
    ratio = 1.0 + np.asarray(coded, dtype=float) * half_ranges
    return PendulumParameters(
        nominal.inertia_kg_m2 * ratio[0],
        nominal.damping_n_m_s_per_rad * ratio[1],
        nominal.restoring_n_m_per_rad * ratio[2],
        nominal.force_lever_m * ratio[3],
    )


def force_metrics(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    error = estimate[mask] - truth[mask]
    truth_std = float(np.std(truth[mask]))
    return {
        "rmse_N": float(np.sqrt(np.mean(error**2))),
        "nrmse_fluctuation": float(np.sqrt(np.mean(error**2)) / truth_std),
        "bias_N": float(np.mean(error)),
        "mae_N": float(np.mean(np.abs(error))),
        "p95_abs_error_N": float(np.quantile(np.abs(error), 0.95)),
        "max_abs_error_N": float(np.max(np.abs(error))),
    }


def causal_lowpass(values: np.ndarray, cutoff_hz: float, dt: float) -> np.ndarray:
    pole = np.exp(-2.0 * np.pi * cutoff_hz * dt)
    result = np.asarray(values, dtype=float)
    for _ in range(3):
        result = signal.lfilter([0.0, 1.0 - pole], [1.0, -pole], result)
    return result


def wind_case(config: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    time, speed, metadata = synthesize_kaimal_wind(
        config["sample_rate_hz"],
        config["duration_s"],
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


def four_candidates(
    angle: np.ndarray,
    parameters: PendulumParameters,
    dt: float,
    tuning: dict[str, float],
    angle_noise_rad: float,
) -> dict[str, np.ndarray]:
    _, rts3 = kalman_rts_force(
        angle, parameters, dt, angle_noise_rad, tuning["rts3_process_noise"], 0
    )
    _, rts4 = kalman_rts_force(
        angle, parameters, dt, angle_noise_rad, tuning["rts4_process_noise"], 1
    )
    return {
        "ESO 3-state": causal_luenberger(angle, parameters, dt, tuning["eso3_pole_hz"], 0),
        "ESO 4-state": causal_luenberger(angle, parameters, dt, tuning["eso4_pole_hz"], 1),
        "RTS 3-state": rts3,
        "RTS 4-state": rts4,
    }


def tune_estimators(
    angle: np.ndarray,
    force: np.ndarray,
    parameters: PendulumParameters,
    dt: float,
    mask: np.ndarray,
    config: dict,
) -> tuple[dict[str, float], list[dict]]:
    rows: list[dict] = []
    for order, label in ((0, "ESO 3-state"), (1, "ESO 4-state")):
        for value in config["eso_pole_grid_hz"]:
            estimate = causal_luenberger(angle, parameters, dt, value, order)
            rows.append(
                {
                    "method": label,
                    "parameter": value,
                    "parameter_name": "repeated_pole_hz",
                    **force_metrics(force, estimate, mask),
                }
            )
    angle_noise = np.deg2rad(config["assumed_angle_noise_deg"])
    for order, label, grid in (
        (0, "RTS 3-state", config["rts3_force_random_walk_grid_N_per_sample"]),
        (1, "RTS 4-state", config["rts4_force_rate_random_walk_grid_N_per_s_per_sample"]),
    ):
        for value in grid:
            _, estimate = kalman_rts_force(angle, parameters, dt, angle_noise, value, order)
            rows.append(
                {
                    "method": label,
                    "parameter": value,
                    "parameter_name": "force_process_noise",
                    **force_metrics(force, estimate, mask),
                }
            )
    for value in config["lpf_cutoff_grid_hz"]:
        raw = parameters.restoring_n_m_per_rad / parameters.force_lever_m * angle
        estimate = causal_lowpass(raw, value, dt)
        rows.append(
            {
                "method": "Causal LPF",
                "parameter": value,
                "parameter_name": "cutoff_hz",
                **force_metrics(force, estimate, mask),
            }
        )

    best = {}
    mapping = {
        "ESO 3-state": "eso3_pole_hz",
        "ESO 4-state": "eso4_pole_hz",
        "RTS 3-state": "rts3_process_noise",
        "RTS 4-state": "rts4_process_noise",
        "Causal LPF": "lpf_cutoff_hz",
    }
    for method, key in mapping.items():
        selected = min((row for row in rows if row["method"] == method), key=lambda r: r["nrmse_fluctuation"])
        best[key] = float(selected["parameter"])
    return best, rows


def sensitivity_tables(
    rows: list[dict], factor_names: list[str]
) -> tuple[list[dict], list[dict]]:
    main_rows: list[dict] = []
    interaction_rows: list[dict] = []
    for method in CANDIDATES:
        selected = [row for row in rows if row["method"] == method]
        overall = float(np.mean([row["nrmse_fluctuation"] for row in selected]))
        level_means: dict[tuple[int, int], float] = {}
        for index, factor in enumerate(factor_names):
            means = []
            for level in (-1, 0, 1):
                mean = float(
                    np.mean(
                        [row["nrmse_fluctuation"] for row in selected if int(row[f"coded_{factor}"]) == level]
                    )
                )
                level_means[(index, level)] = mean
                means.append(mean)
            main_rows.append(
                {
                    "method": method,
                    "factor": factor,
                    "mean_at_low": means[0],
                    "mean_at_center": means[1],
                    "mean_at_high": means[2],
                    "main_effect_range": max(means) - min(means),
                    "worst_level_mean": max(means),
                }
            )
        for first in range(len(factor_names)):
            for second in range(first + 1, len(factor_names)):
                residual = []
                for level_first in (-1, 0, 1):
                    for level_second in (-1, 0, 1):
                        cell = float(
                            np.mean(
                                [
                                    row["nrmse_fluctuation"]
                                    for row in selected
                                    if int(row[f"coded_{factor_names[first]}"]) == level_first
                                    and int(row[f"coded_{factor_names[second]}"]) == level_second
                                ]
                            )
                        )
                        additive = (
                            level_means[(first, level_first)]
                            + level_means[(second, level_second)]
                            - overall
                        )
                        residual.append(cell - additive)
                interaction_rows.append(
                    {
                        "method": method,
                        "interaction": f"{factor_names[first]}:{factor_names[second]}",
                        "interaction_rms": float(np.sqrt(np.mean(np.square(residual)))),
                        "interaction_max_abs": float(np.max(np.abs(residual))),
                    }
                )
    return main_rows, interaction_rows


def response_surface_validation(
    design: np.ndarray,
    design_rows: list[dict],
    validation: np.ndarray,
    validation_rows: list[dict],
) -> list[dict]:
    train_x, _ = features(design)
    test_x, _ = features(validation)
    output = []
    for method in CANDIDATES:
        train_y = np.log10(
            [row["rmse_N"] for row in design_rows if row["method"] == method]
        )
        test_y = np.array([row["rmse_N"] for row in validation_rows if row["method"] == method])
        coefficient = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
        train_prediction = train_x @ coefficient
        prediction = 10.0 ** (test_x @ coefficient)
        relative = np.abs(prediction - test_y) / test_y
        ss_res = float(np.sum((train_y - train_prediction) ** 2))
        ss_total = float(np.sum((train_y - np.mean(train_y)) ** 2))
        output.append(
            {
                "method": method,
                "train_log_rmse_r2": 1.0 - ss_res / ss_total,
                "validation_median_relative_error": float(np.median(relative)),
                "validation_max_relative_error": float(np.max(relative)),
                "gate_10_percent": bool(np.max(relative) <= 0.1),
            }
        )
    return output


def make_plots(
    output: Path,
    config: dict,
    time: np.ndarray,
    speed: np.ndarray,
    force: np.ndarray,
    estimates: dict[str, np.ndarray],
    mask: np.ndarray,
    wind_metadata: dict,
    main_rows: list[dict],
) -> None:
    fs = config["sample_rate_hz"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    selected = mask & (time <= config["evaluation_start_s"] + 60.0)
    axes[0].plot(time[selected], speed[selected], color="#2776bc", linewidth=1.0)
    axes[0].set(ylabel="Wind speed [m/s]", title="Bounded Kaimal-spectrum wind")
    axes[1].plot(time[selected], 1000.0 * force[selected], color="black", linewidth=2.0, label="True force")
    for method in BASELINES + CANDIDATES:
        axes[1].plot(
            time[selected],
            1000.0 * estimates[method][selected],
            color=COLORS[method],
            linewidth=1.0,
            label=method,
        )
    axes[1].set(xlabel="Time [s]", ylabel="Force [mN]")
    axes[1].legend(ncol=3, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "real_wind_timeseries.png", dpi=180)
    plt.close(fig)

    frequency, measured_psd = signal.welch(
        speed - np.mean(speed), fs=fs, window="hann", nperseg=min(len(speed), 8192)
    )
    target_sigma = config["target_turbulence_intensity"] * config["mean_wind_speed_m_s"]
    target_psd = kaimal_longitudinal_psd(
        frequency,
        config["mean_wind_speed_m_s"],
        target_sigma,
        config["kaimal_integral_scale_m"],
    ) * wind_metadata["fluctuation_scale_for_bounds"] ** 2
    keep = frequency > 0.0
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.loglog(frequency[keep], target_psd[keep], "k--", linewidth=2, label="Kaimal target")
    ax.loglog(frequency[keep], measured_psd[keep], color="#2776bc", alpha=0.8, label="Generated record (Welch)")
    ax.axvline(50.0, color="#c23b73", linestyle=":", label="100 Hz Nyquist")
    ax.set(xlabel="Frequency [Hz]", ylabel="One-sided PSD [(m/s)^2/Hz]", title="Wind spectrum check")
    ax.grid(which="both", alpha=0.2)
    ax.legend()
    fig.savefig(output / "wind_spectrum.png", dpi=180)
    plt.close(fig)

    factor_names = config["factor_names"]
    matrix = np.array(
        [
            [next(row["main_effect_range"] for row in main_rows if row["method"] == method and row["factor"] == factor) for factor in factor_names]
            for method in CANDIDATES
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(factor_names)), factor_names)
    ax.set_yticks(range(len(CANDIDATES)), CANDIDATES)
    ax.set_title("DOE main-effect range of fluctuation-normalized RMSE")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="NRMSE range")
    fig.savefig(output / "doe_main_effects.png", dpi=180)
    plt.close(fig)


def run(config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    nominal = PendulumParameters(**config["nominal_parameters"])
    dt = 1.0 / config["sample_rate_hz"]
    training_time, training_speed, training_force, training_wind = wind_case(config, config["training_seed"])
    evaluation_time, evaluation_speed, evaluation_force, evaluation_wind = wind_case(config, config["evaluation_seed"])
    training_angle = plant(training_force, nominal, dt)[:, 0]
    evaluation_state = plant(evaluation_force, nominal, dt)
    evaluation_angle = evaluation_state[:, 0]
    training_mask = (training_time >= config["evaluation_start_s"]) & (training_time <= config["evaluation_end_s"])
    evaluation_mask = (evaluation_time >= config["evaluation_start_s"]) & (evaluation_time <= config["evaluation_end_s"])
    tuning, tuning_rows = tune_estimators(
        training_angle, training_force, nominal, dt, training_mask, config
    )
    write_csv(output / "tuning_scan.csv", tuning_rows)
    angle_noise = np.deg2rad(config["assumed_angle_noise_deg"])
    candidate_nominal = four_candidates(evaluation_angle, nominal, dt, tuning, angle_noise)
    raw = nominal.restoring_n_m_per_rad / nominal.force_lever_m * evaluation_angle
    nominal_estimates = {
        "Raw static": raw,
        "Causal LPF": causal_lowpass(raw, tuning["lpf_cutoff_hz"], dt),
        **candidate_nominal,
    }
    nominal_rows = [
        {"method": method, **force_metrics(evaluation_force, values, evaluation_mask)}
        for method, values in nominal_estimates.items()
    ]
    write_csv(output / "nominal_metrics.csv", nominal_rows)

    half_ranges = np.asarray(config["factor_half_ranges"], dtype=float)
    factor_names = config["factor_names"]
    design = np.asarray(list(product((-1.0, 0.0, 1.0), repeat=4)))
    latin = 2.0 * qmc.LatinHypercube(d=4, seed=config["evaluation_seed"] + 1).random(
        config["validation_points"]
    ) - 1.0

    def evaluate_points(points: np.ndarray, kind: str) -> list[dict]:
        rows: list[dict] = []
        for run_index, coded in enumerate(points):
            parameters = decode_parameters(nominal, coded, half_ranges)
            estimates = four_candidates(evaluation_angle, parameters, dt, tuning, angle_noise)
            coded_columns = {f"coded_{name}": float(value) for name, value in zip(factor_names, coded)}
            ratio_columns = {
                f"ratio_{name}": float(1.0 + value * width)
                for name, value, width in zip(factor_names, coded, half_ranges)
            }
            for method, values in estimates.items():
                rows.append(
                    {
                        "set": kind,
                        "run": run_index,
                        "method": method,
                        **coded_columns,
                        **ratio_columns,
                        **force_metrics(evaluation_force, values, evaluation_mask),
                    }
                )
            if kind == "design" and (run_index + 1) % 10 == 0:
                print(f"DOE progress: {run_index + 1}/{len(points)}")
        return rows

    design_rows = evaluate_points(design, "design")
    validation_rows = evaluate_points(latin, "validation")
    write_csv(output / "doe_results.csv", design_rows)
    write_csv(output / "doe_validation_points.csv", validation_rows)
    main_rows, interaction_rows = sensitivity_tables(design_rows, factor_names)
    write_csv(output / "doe_main_effects.csv", main_rows)
    write_csv(output / "doe_interactions.csv", interaction_rows)
    surface_rows = response_surface_validation(design, design_rows, latin, validation_rows)
    write_csv(output / "response_surface_validation.csv", surface_rows)
    method_summary = []
    for method in CANDIDATES:
        selected = [row for row in design_rows if row["method"] == method]
        values = np.array([row["nrmse_fluctuation"] for row in selected])
        worst = max(selected, key=lambda row: row["nrmse_fluctuation"])
        method_summary.append(
            {
                "method": method,
                "minimum_nrmse": float(np.min(values)),
                "median_nrmse": float(np.median(values)),
                "maximum_nrmse": float(np.max(values)),
                "worst_parameter_ratios": {
                    name: worst[f"ratio_{name}"] for name in factor_names
                },
            }
        )

    current = config["current_parameters_for_limit_only"]
    current_angle_limit = np.deg2rad(current["angle_limit_deg"])
    current_force_limit_linear = current["restoring_n_m_per_rad"] * current_angle_limit / current["force_lever_m"]
    current_force_limit_sine = current["restoring_n_m_per_rad"] * np.sin(current_angle_limit) / current["force_lever_m"]
    aero = config["air_density_kg_m3"] * config["drag_coefficient"] * config["projected_area_m2"]
    limits = {
        "force_at_8m_s_N": float(0.5 * aero * config["maximum_wind_speed_m_s"] ** 2),
        "current_linear_force_limit_at_38deg_N": float(current_force_limit_linear),
        "current_sine_force_limit_at_38deg_N": float(current_force_limit_sine),
        "current_linear_wind_limit_m_s": float(np.sqrt(2.0 * current_force_limit_linear / aero)),
        "current_sine_wind_limit_m_s": float(np.sqrt(2.0 * current_force_limit_sine / aero)),
        "redesigned_K_ratio_to_current": float(nominal.restoring_n_m_per_rad / current["restoring_n_m_per_rad"]),
        "redesigned_b_ratio_to_current": float(nominal.damping_n_m_s_per_rad / current["damping_n_m_s_per_rad"]),
    }
    max_angle = float(np.max(np.abs(np.rad2deg(evaluation_angle[evaluation_mask]))))
    summary = {
        "scope": "Stage 2 extension: ideal angle, no sensor model, angle-only estimators",
        "wind_model": "one-sided IEC-style Kaimal along-wind spectrum with bounded random-phase realization",
        "training_wind": training_wind,
        "evaluation_wind": evaluation_wind,
        "tuning": tuning,
        "tuning_at_search_boundary": {
            "ESO 3-state": tuning["eso3_pole_hz"]
            in (min(config["eso_pole_grid_hz"]), max(config["eso_pole_grid_hz"])),
            "ESO 4-state": tuning["eso4_pole_hz"]
            in (min(config["eso_pole_grid_hz"]), max(config["eso_pole_grid_hz"])),
            "RTS 3-state": tuning["rts3_process_noise"]
            in (
                min(config["rts3_force_random_walk_grid_N_per_sample"]),
                max(config["rts3_force_random_walk_grid_N_per_sample"]),
            ),
            "RTS 4-state": tuning["rts4_process_noise"]
            in (
                min(config["rts4_force_rate_random_walk_grid_N_per_s_per_sample"]),
                max(config["rts4_force_rate_random_walk_grid_N_per_s_per_sample"]),
            ),
        },
        "nominal_metrics": nominal_rows,
        "doe": {
            "design": "four-factor three-level full factorial",
            "design_points": len(design),
            "validation_points": len(latin),
            "factor_names": factor_names,
            "factor_half_ranges": config["factor_half_ranges"],
            "method_summary": method_summary,
            "response_surface_validation": surface_rows,
        },
        "plant": {
            "parameters": config["nominal_parameters"],
            "characteristics": natural_characteristics(nominal),
            "maximum_dynamic_angle_deg_in_evaluation": max_angle,
            "design_target_static_angle_deg_at_8m_s": config["design_target_static_angle_deg_at_8m_s"],
            "limits_and_redesign": limits,
        },
        "status": "COMPLETE_AWAITING_USER_REVIEW",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_plots(
        output,
        config,
        evaluation_time,
        evaluation_speed,
        evaluation_force,
        nominal_estimates,
        evaluation_mask,
        evaluation_wind,
        main_rows,
    )
    write_report(ROOT / "Real_Wind_DOE_Report.md", config, summary, main_rows, interaction_rows)
    return summary


def write_report(
    path: Path,
    config: dict,
    summary: dict,
    main_rows: list[dict],
    interaction_rows: list[dict],
) -> None:
    metrics = {row["method"]: row for row in summary["nominal_metrics"]}
    lines = [
        "# Stage 2追加評価：実風スペクトルと4推定器の係数感度",
        "",
        "## 結論の読み方",
        "",
        "この評価は、Kaimalスペクトルに従う風速を準定常抗力へ変換し、角度だけから外力を推定するPoCである。センサノイズ、機械ストッパー、非定常Cdはまだ含めていない。4方式の比較対象は3状態ESO、4状態ESO、3状態RTSスムーザー、4状態RTSスムーザーである。生の静的換算と因果LPFを基準として併記する。",
        "",
        "## 実風スペクトル",
        "",
        "地表層の実測から整理されたKaimalスペクトルを採用した。NREL資料に記載されたIEC形式の一方向スペクトルを使う。",
        "",
        "```math",
        "S_u(f)=\\frac{4\\sigma_u^2 L_u/U}{\\left(1+6fL_u/U\\right)^{5/3}}",
        "```",
        "",
        f"ここで、$`U`$は平均風速、$`\\sigma_u`$は風速変動の標準偏差、$`L_u`$は積分長さスケールである。今回は高さ2 mの代表値として$`L_u=8.1\\times0.7z=11.34`$ m、平均5 m/s、目標乱流強度20%、上限8 m/sとした。上限を守るため変動全体へ一つの倍率を掛け、スペクトル形状を保った。評価波形で実現した乱流強度は{100*summary['evaluation_wind']['realized_turbulence_intensity']:.2f}%である。",
        "",
        "- [Kaimal et al. (1972), Spectral characteristics of surface-layer turbulence](https://doi.org/10.1002/qj.49709841707)",
        "- [NREL, Sensitivity Analysis of Wind Characteristics and Wind Turbine Properties](https://www.nrel.gov/docs/fy19osti/74876.pdf)",
        "",
        "![風速スペクトル](results/real_wind_doe/wind_spectrum.png)",
        "",
        "抗力は次の準定常式で真値を作った。Cdは入力生成だけに用い、オブザーバーはCdや風速を知らない。",
        "",
        "```math",
        "F(t)=\\frac{1}{2}\\rho C_d A U(t)\\lvert U(t)\\rvert",
        "```",
        "",
        "## 8 m/sと装置範囲",
        "",
        f"現行Kの線形モデルでは38度相当の上限は{summary['plant']['limits_and_redesign']['current_linear_wind_limit_m_s']:.2f} m/s、sinを使う静的式では{summary['plant']['limits_and_redesign']['current_sine_wind_limit_m_s']:.2f} m/sである。したがって現行機のまま8 m/sを評価することはできない。今回の仮想プラントは8 m/sの静的角度を30度に設定し、Kを現行の{summary['plant']['limits_and_redesign']['redesigned_K_ratio_to_current']:.2f}倍とした。減衰比を現行値に保つためbも{summary['plant']['limits_and_redesign']['redesigned_b_ratio_to_current']:.2f}倍とした。評価波形での最大動的角度は{summary['plant']['maximum_dynamic_angle_deg_in_evaluation']:.2f}度だった。",
        "",
        "## 4方式と調整",
        "",
        "全方式の観測は角度だけであり、角速度は入力しない。ESOは公称学習波形で反復極を探索する。RTSは同じ公称学習波形で外力または外力変化率へ与えるプロセス雑音を探索する。評価波形とDOEでは選択値を固定した。",
        "",
        "| 方式 | 状態 | 外力仮定 | 因果性 | 選択値 |",
        "|---|---|---|---|---:|",
        f"| ESO 3-state | 角度・角速度・トルク | 区間内で一定 | 因果 | {summary['tuning']['eso3_pole_hz']:.6g} Hz |",
        f"| ESO 4-state | 上記＋トルク変化率 | 区間内で一定勾配 | 因果 | {summary['tuning']['eso4_pole_hz']:.6g} Hz |",
        f"| RTS 3-state | 角度・角速度・トルク | トルクrandom walk | 非因果 | {summary['tuning']['rts3_process_noise']:.6g} N/sample |",
        f"| RTS 4-state | 上記＋トルク変化率 | 変化率random walk | 非因果 | {summary['tuning']['rts4_process_noise']:.6g} (N/s)/sample |",
        "",
        "探索上限に達した方式は、ESO 3-stateとRTS 3-stateである。理想角度・ノイズなしでは高帯域化の罰則が現れず、特にRTS 3-stateは逆動力学に近づくほど公称誤差が小さくなる。この選択値を実機の推奨値とは扱わない。センサノイズを有効にしたStage 3で再調整する。",
        "",
        "## 公称条件の結果",
        "",
        "RMSEは全外力の誤差、NRMSEは外力変動の標準偏差でRMSEを割った値である。平均抗力が大きい条件でも動的誤差を過小評価しにくい後者を主指標にする。",
        "",
        "| 方式 | RMSE [mN] | 変動基準NRMSE | Bias [mN] | 95%絶対誤差 [mN] |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in BASELINES + CANDIDATES:
        row = metrics[method]
        lines.append(
            f"| {method} | {1000*row['rmse_N']:.4f} | {row['nrmse_fluctuation']:.4f} | {1000*row['bias_N']:.4f} | {1000*row['p95_abs_error_N']:.4f} |"
        )
    lines += [
        "",
        "![時系列](results/real_wind_doe/real_wind_timeseries.png)",
        "",
        "## 係数誤差DOE",
        "",
        "I ±25%、b ±50%、K ±10%、作用距離l ±5%を低・中央・高の3水準とし、3の4乗=81条件を直接計算した。推定側の係数だけを変更し、プラント、風入力、調整値は固定した。各因子の感度値は、その因子の3水準ごとに他因子全条件を平均したNRMSEの最大値と最小値の差である。",
        "",
        "| 方式 | 81条件の最小NRMSE | 中央値 | 最大値 | 最大値となるI / b / K / l倍率 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["doe"]["method_summary"]:
        ratios = row["worst_parameter_ratios"]
        lines.append(
            f"| {row['method']} | {row['minimum_nrmse']:.4f} | {row['median_nrmse']:.4f} | {row['maximum_nrmse']:.4f} | {ratios['I']:.2f} / {ratios['b']:.2f} / {ratios['K']:.2f} / {ratios['l']:.2f} |"
        )
    lines += [
        "",
        "| 方式 | 1位 | 2位 | 3位 | 4位 |",
        "|---|---|---|---|---|",
    ]
    for method in CANDIDATES:
        ranked = sorted(
            (row for row in main_rows if row["method"] == method),
            key=lambda row: row["main_effect_range"],
            reverse=True,
        )
        cells = [f"{row['factor']} ({row['main_effect_range']:.4f})" for row in ranked]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "![主効果](results/real_wind_doe/doe_main_effects.png)",
        "",
        "交互作用は、2因子の各3×3セル平均から加法的な主効果を引いた残差で評価した。値はdoe_interactions.csvに保存した。二次応答曲面は独立Latin Hypercube点で確認し、10%最大相対誤差ゲートを満たした場合だけ補間用に使える。直接計算した81条件の感度順位は、このゲートに依存しない。",
        "",
        "| 方式 | 学習log(RMSE) R2 | 確認中央値誤差 | 確認最大誤差 | 10%ゲート |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["doe"]["response_surface_validation"]:
        lines.append(
            f"| {row['method']} | {row['train_log_rmse_r2']:.4f} | {100*row['validation_median_relative_error']:.2f}% | {100*row['validation_max_relative_error']:.2f}% | {'PASS' if row['gate_10_percent'] else 'NOT_MET'} |"
        )
    strongest = sorted(interaction_rows, key=lambda row: row["interaction_rms"], reverse=True)[:4]
    lines += [
        "",
        "強い交互作用の上位4件は次の通りである。",
        "",
        "| 方式 | 交互作用 | RMS | 最大絶対値 |",
        "|---|---|---:|---:|",
    ]
    for row in strongest:
        lines.append(
            f"| {row['method']} | {row['interaction']} | {row['interaction_rms']:.5f} | {row['interaction_max_abs']:.5f} |"
        )
    lines += [
        "",
        "## 再現方法",
        "",
        "```bash",
        "python 06_Analysis/simulation/src/run_real_wind_doe.py",
        "python -m unittest discover -s 06_Analysis/simulation/tests -v",
        "```",
        "",
        "設定はsimulation/config/real_wind_doe.json、数値結果はsimulation/results/real_wind_doeに保存する。今回の再設計K・bは実機改造値の決定ではなく、8 m/sかつ非飽和という比較条件を成立させる仮定である。Stage 3では独立したセンサモデルを追加し、同じ4方式を再調整して比較する。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/real_wind_doe.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/real_wind_doe")
    args = parser.parse_args()
    summary = run(args.config, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
