"""Stage 4: evaluate unmodelled restoring-torque backlash with Stage 3 sensor model."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKLASH_DIRECTORY = ROOT.parent / "backlash_study"
sys.path.insert(0, str(BACKLASH_DIRECTORY))

from backlash_model import update_play_state
from model import PendulumParameters
from run_sensor_model_stage3 import evaluate_methods
from run_sensor_model_stage3 import sensor_parameters
from run_sensor_model_stage3 import wind_case
from sensor_model import apply_angle_sensor_model


COLORS = {
    "ESO 3-state": "#2776bc",
    "ESO 4-state": "#e67e22",
    "RTS 3-state": "#159477",
    "RTS 4-state": "#006b4f",
}


def write_csv(path, rows):
    """辞書行をCSVへ保存する。"""

    if len(rows) == 0:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def backlash_friction_plant(
    force_n,
    parameters,
    sample_period_s,
    friction_torque_n_m,
    epsilon_rad_s,
    backlash_half_width_rad,
):
    """復元トルク側にplayバックラッシュを持つ線形振り子を計算する。"""

    force_n = np.asarray(force_n, dtype=float)
    states = np.zeros((len(force_n), 2), dtype=float)
    play_state = np.zeros(len(force_n), dtype=float)

    def derivative(state, current_play, applied_force):
        angle = state[0]
        speed = state[1]
        friction = friction_torque_n_m * np.tanh(speed / epsilon_rad_s)
        acceleration = parameters.force_lever_m * applied_force
        acceleration -= parameters.damping_n_m_s_per_rad * speed
        acceleration -= parameters.restoring_n_m_per_rad * current_play
        acceleration -= friction
        acceleration /= parameters.inertia_kg_m2
        return np.asarray([speed, acceleration])

    index = 0
    while index < len(force_n) - 1:
        state = states[index]
        current_play = play_state[index]
        applied_force = force_n[index]

        play1 = update_play_state(state[0], current_play, backlash_half_width_rad)
        k1 = derivative(state, play1, applied_force)

        state2 = state + 0.5 * sample_period_s * k1
        play2 = update_play_state(state2[0], play1, backlash_half_width_rad)
        k2 = derivative(state2, play2, applied_force)

        state3 = state + 0.5 * sample_period_s * k2
        play3 = update_play_state(state3[0], play2, backlash_half_width_rad)
        k3 = derivative(state3, play3, applied_force)

        state4 = state + sample_period_s * k3
        play4 = update_play_state(state4[0], play3, backlash_half_width_rad)
        k4 = derivative(state4, play4, applied_force)

        states[index + 1] = state + sample_period_s * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0
        play_state[index + 1] = update_play_state(
            states[index + 1, 0], play4, backlash_half_width_rad
        )
        index += 1
    return states, play_state


def make_plots(output, rows, selected_record, parameters):
    """感度曲線、代表波形、復元トルク履歴を描く。"""

    methods = list(COLORS)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        width = [row["backlash_total_width_deg"] for row in selected]
        force_rmse = [1000.0 * row["rmse_N"] for row in selected]
        wind_rmse = [row["rmse_m_s"] for row in selected]
        axes[0].plot(width, force_rmse, marker="o", color=COLORS[method], label=method)
        axes[1].plot(width, wind_rmse, marker="o", color=COLORS[method], label=method)
    axes[0].set(xlabel="Total backlash width [deg]", ylabel="Force RMSE [mN]", title="Unmodelled backlash sensitivity")
    axes[1].set(xlabel="Total backlash width [deg]", ylabel="Wind-speed RMSE [m/s]", title="Wind-speed impact")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output / "backlash_sensitivity.png", dpi=180)
    plt.close(figure)

    time = selected_record["time_s"]
    mask = selected_record["plot_mask"]
    true_force = selected_record["force_n"]
    apparent_force = selected_record["apparent_backlash_force_n"]
    estimates = selected_record["estimates"]
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    axes[0].plot(time[mask], 1000.0 * true_force[mask], color="black", linewidth=1.8, label="True force")
    axes[0].plot(time[mask], 1000.0 * apparent_force[mask], color="#c23b73", linewidth=1.0, label="Backlash-equivalent force")
    axes[0].set(ylabel="Force [mN]", title="Representative backlash stress case")
    axes[0].legend()
    for method in methods:
        axes[1].plot(time[mask], 1000.0 * estimates[method][mask], color=COLORS[method], linewidth=0.9, label=method)
    axes[1].plot(time[mask], 1000.0 * true_force[mask], color="black", linewidth=1.5, label="True force")
    axes[1].set(xlabel="Time [s]", ylabel="Estimated force [mN]")
    axes[1].legend(ncol=3, fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.savefig(output / "backlash_force_timeseries.png", dpi=180)
    plt.close(figure)

    angle_deg = np.rad2deg(selected_record["true_angle_rad"])
    restoring_mnm = 1000.0 * parameters.restoring_n_m_per_rad * selected_record["play_state_rad"]
    figure, axis = plt.subplots(figsize=(7, 5.5), layout="constrained")
    axis.plot(angle_deg[mask], restoring_mnm[mask], color="#0072b2", linewidth=0.8)
    axis.set(xlabel="Physical angle [deg]", ylabel="Restoring torque [mN m]", title="Play-model restoring hysteresis")
    axis.grid(True, alpha=0.25)
    figure.savefig(output / "backlash_hysteresis_loop.png", dpi=180)
    plt.close(figure)


def write_report(output, config, rows, selected_record, fit_summary):
    """Stage 4結果をMarkdownへまとめる。"""

    methods = list(COLORS)
    baseline = {}
    stress = {}
    for method in methods:
        baseline[method] = next(
            row for row in rows
            if row["method"] == method and row["backlash_total_width_deg"] == 0.0
        )
        stress[method] = next(
            row for row in rows
            if row["method"] == method
            and row["backlash_total_width_deg"] == config["nominal_stress_width_deg"]
        )

    lines = [
        "# Stage 4：摩擦・バックラッシュの検討",
        "",
        "## 結論",
        "",
        "クーロン摩擦はすでにプラントと推定器の双方へ実装済みである。今回、復元機構の方向反転時に履歴を持つplayバックラッシュを追加した。",
        "旧ハード自由振動ではバックラッシュ幅の同定値が試行間で安定せず、他試行へ移植すると平均予測誤差が悪化した。",
        "したがって現時点では、バックラッシュ補償を公称オブザーバーへ常時組み込む根拠は不足している。無効化可能な試験モデルとして保持し、新ハード反転試験で幅を測定する。",
        "",
        "## 既存自由振動による識別結果",
        "",
        f"- VC別試行予測RMSE平均: {fit_summary['vc_validation_mean_deg']:.4f} deg",
        f"- VCB別試行予測RMSE平均: {fit_summary['vcb_validation_mean_deg']:.4f} deg",
        f"- VCBの変化: {fit_summary['validation_change_percent']:+.1f}%",
        f"- 同定された全バックラッシュ幅: {fit_summary['identified_widths_deg']}",
        "",
        "試行内では一部区間の残差がわずかに減るが、幅がほぼ0～0.58 degへ散らばる。これはバックラッシュが一意に同定された状態ではない。",
        "",
        "## Stage 3センサーモデル込みのストレス試験",
        "",
        f"公称値ではなく、全幅{config['nominal_stress_width_deg']:.2f} degを仮定した影響を示す。推定器はバックラッシュを知らず、Stage 3の調整値を固定した。",
        "",
        "| 推定器 | 0 deg外力RMSE [mN] | ストレス時 [mN] | 変化 | 0 deg風速RMSE [m/s] | ストレス時 [m/s] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        change = 100.0 * (stress[method]["rmse_N"] / baseline[method]["rmse_N"] - 1.0)
        lines.append(
            f"| {method} | {1000*baseline[method]['rmse_N']:.4f} | {1000*stress[method]['rmse_N']:.4f} | {change:+.1f}% | "
            f"{baseline[method]['rmse_m_s']:.4f} | {stress[method]['rmse_m_s']:.4f} |"
        )

    maximum_equivalent = 1000.0 * float(np.max(np.abs(selected_record["apparent_backlash_force_n"])))
    lines += [
        "",
        f"全幅{config['nominal_stress_width_deg']:.2f} degで、復元トルク差を力換算した最大値は{maximum_equivalent:.3f} mNだった。",
        "",
        "## モデル",
        "",
        "物理角度をθ、ばね側の履歴角度をz、半幅をδとし、離散play演算子を使う。",
        "",
        "```math",
        "z_k = \\max(\\theta_k-\\delta,\\min(z_{k-1},\\theta_k+\\delta))",
        "```",
        "",
        "```math",
        "I\\ddot\\theta+b\\dot\\theta+Kz+\\tau_f\\tanh(\\dot\\theta/\\varepsilon)=lF_{aero}",
        "```",
        "",
        "δ=0で従来モデルへ完全に戻る。摩擦は速度方向に依存する散逸項、バックラッシュは反転履歴に依存する復元トルク項なので、別パラメータとして扱う。",
        "",
        "## 推奨する次の実機確認",
        "",
        "- 新ハード各軸を無風で±5 deg程度ゆっくり往復させ、角度と既知外力またはトルクを同時記録する。",
        "- 正転・逆転の同一トルクに対する角度差からバックラッシュ全幅を求める。",
        "- 幅が角度分解能0.02197 degや反復ばらつきを十分上回る場合だけ、軸別補償を有効化する。",
        "- 風速換算では、未補償バックラッシュの等価力を低風速検出限界へ加える。",
        "- 全幅2 deg条件は最大角度が45 degを超えるため、ストッパーなしモデルの参考値としてのみ扱う。",
        "- 現在の摩擦項は動摩擦であり、静止摩擦による固着やStribeck特性はまだ含まない。",
        "",
        "## 図",
        "",
        "![バックラッシュ感度](results/backlash_stage4/backlash_sensitivity.png)",
        "",
        "![代表外力波形](results/backlash_stage4/backlash_force_timeseries.png)",
        "",
        "![復元トルク履歴](results/backlash_stage4/backlash_hysteresis_loop.png)",
        "",
        "## 再実行",
        "",
        "```bash",
        "python 06_Analysis/backlash_study/analyze.py --input 04_Data/00_Calibration/swing/LOG00014.TXT",
        "python 06_Analysis/simulation/src/run_backlash_stage4.py",
        "```",
    ]
    (ROOT / "Backlash_Stage4_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path, output):
    """設定を読み、全バックラッシュ幅を同じ風・センサー条件で比較する。"""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    stage3_config = json.loads((ROOT / config["stage3_config"]).read_text(encoding="utf-8"))
    stage3_summary = json.loads((ROOT / config["stage3_summary"]).read_text(encoding="utf-8"))
    tuning = stage3_summary["tuning"]
    parameters = PendulumParameters(**stage3_config["nominal_parameters"])
    sensor = sensor_parameters(stage3_config, "nominal")
    dt = 1.0 / stage3_config["sample_rate_hz"]
    friction = stage3_config["nominal_friction_torque_n_m"]
    epsilon = np.deg2rad(stage3_config["friction_epsilon_deg_s"])
    time, speed, force, unused_metadata = wind_case(
        stage3_config,
        stage3_config["evaluation_duration_s"],
        stage3_config["evaluation_seed"],
    )
    mask = (time >= stage3_config["evaluation_start_s"]) & (
        time <= stage3_config["evaluation_end_s"]
    )

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    records = {}
    for total_width_deg in config["backlash_total_width_deg"]:
        half_width_rad = np.deg2rad(0.5 * total_width_deg)
        state, play_state = backlash_friction_plant(
            force, parameters, dt, friction, epsilon, half_width_rad
        )
        measured_angle, unused_components = apply_angle_sensor_model(
            time,
            state[:, 0],
            sensor,
            stage3_config["sensor_seed"] + 1,
        )
        estimates, metrics = evaluate_methods(
            time,
            speed,
            force,
            measured_angle,
            parameters,
            sensor,
            tuning,
            stage3_config,
            mask,
            friction,
            epsilon,
        )
        apparent_force = parameters.restoring_n_m_per_rad * (
            state[:, 0] - play_state
        ) / parameters.force_lever_m
        for row in metrics:
            if row["method"] in COLORS:
                rows.append(
                    {
                        "backlash_total_width_deg": total_width_deg,
                        "backlash_half_width_deg": 0.5 * total_width_deg,
                        "method": row["method"],
                        "rmse_N": row["rmse_N"],
                        "nrmse_fluctuation": row["nrmse_fluctuation"],
                        "bias_N": row["bias_N"],
                        "rmse_m_s": row["rmse_m_s"],
                        "best_lag_s": row["best_lag_s"],
                        "max_equivalent_backlash_force_N": float(np.max(np.abs(apparent_force[mask]))),
                        "max_true_angle_deg": float(np.max(np.abs(np.rad2deg(state[:, 0])))),
                        "within_45deg_range": int(
                            np.max(np.abs(np.rad2deg(state[:, 0]))) <= 45.0
                        ),
                    }
                )
        records[total_width_deg] = {
            "time_s": time,
            "force_n": force,
            "true_angle_rad": state[:, 0],
            "play_state_rad": play_state,
            "apparent_backlash_force_n": apparent_force,
            "estimates": estimates,
            "plot_mask": mask & (time <= stage3_config["evaluation_start_s"] + 30.0),
        }

    write_csv(output / "backlash_sensitivity.csv", rows)
    selected_record = records[config["nominal_stress_width_deg"]]

    fit_table = np.genfromtxt(
        ROOT.parent / "backlash_study" / "fits.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    validation_table = np.genfromtxt(
        ROOT.parent / "backlash_study" / "validation.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    vc_validation = validation_table[validation_table["model"] == "VC"]["rmse_deg"]
    vcb_validation = validation_table[validation_table["model"] == "VCB"]["rmse_deg"]
    identified = fit_table[fit_table["model"] == "VCB"]["backlash_total_width_deg"]
    fit_summary = {
        "vc_validation_mean_deg": float(np.mean(vc_validation)),
        "vcb_validation_mean_deg": float(np.mean(vcb_validation)),
        "validation_change_percent": float(100.0 * (np.mean(vcb_validation) / np.mean(vc_validation) - 1.0)),
        "identified_widths_deg": ", ".join(format(value, ".3f") for value in identified),
    }

    make_plots(output, rows, selected_record, parameters)
    write_report(output, config, rows, selected_record, fit_summary)
    summary = {
        "config": config,
        "stage3_tuning": tuning,
        "free_decay_fit_summary": fit_summary,
        "sensitivity_metrics": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "backlash_stage4.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "backlash_stage4",
    )
    arguments = parser.parse_args()
    summary = run(arguments.config, arguments.output)
    print(json.dumps(summary["free_decay_fit_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
