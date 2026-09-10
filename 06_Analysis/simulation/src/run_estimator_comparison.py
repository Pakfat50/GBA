"""Compare causal and future-data angle-only force estimators."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from estimators import causal_luenberger, kalman_filter_and_rts_smoother
from model import PendulumParameters, natural_characteristics
from run_stage2 import inputs, metrics
from stage2_system import estimate, plant


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "True force": "#111111",
    "Static": "#999999",
    "Low-pass 1 Hz": "#8e6bbd",
    "ESO constant 1 Hz (current)": "#2776bc",
    "ESO constant tuned": "#00a0c6",
    "ESO ramp tuned": "#e67e22",
    "Kalman causal tuned": "#c23b73",
    "RTS same Kalman tuning": "#159477",
    "RTS offline tuned": "#006b4f",
}


def score(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    return metrics(truth, prediction, mask)["nrmse"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _box(ax, xy, wh, text, color):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="#263238",
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, start, end, label=""):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#37474f"))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.02, label, ha="center", fontsize=8)


def block_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.03, 0.68), (0.14, 0.15), "True force\nF[k]", "#fff3cd")
    _box(ax, (0.24, 0.68), (0.17, 0.15), "Physical plant\nfixed nominal model", "#d9edf7")
    _box(ax, (0.48, 0.68), (0.17, 0.15), "Ideal angle\ny[k] = theta[k]", "#e2f0d9")
    _box(ax, (0.75, 0.68), (0.20, 0.15), "Causal estimators\nESO / Kalman filter", "#e8dff5")
    _box(ax, (0.48, 0.23), (0.20, 0.18), "RTS smoother\nforward + backward", "#fce1cc")
    _box(ax, (0.76, 0.23), (0.19, 0.18), "Comparison\nwith stored true F[k]", "#f8d7da")
    _arrow(ax, (0.17, 0.755), (0.25, 0.755))
    _arrow(ax, (0.41, 0.755), (0.48, 0.755), "angle")
    _arrow(ax, (0.65, 0.755), (0.75, 0.755), "angle only")
    _arrow(ax, (0.565, 0.68), (0.565, 0.41), "full record")
    _arrow(ax, (0.85, 0.68), (0.855, 0.41), "causal estimate")
    _arrow(ax, (0.68, 0.32), (0.76, 0.32), "smoothed estimate")
    ax.text(0.5, 0.94, "Angle-only estimator comparison", ha="center", fontsize=15, weight="bold")
    ax.text(
        0.5,
        0.07,
        "Angular rate is estimated internally; no angular-rate measurement is supplied",
        ha="center",
        fontsize=10,
        color="#455a64",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_mn(value_n: float) -> str:
    """Format force RMSE without hiding sub-micro-newton results as zero."""

    value_mn = 1000.0 * value_n
    return f"{value_mn:.3e}" if abs(value_mn) < 1e-6 else f"{value_mn:.6f}"


def run(config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    parameters = PendulumParameters(**config["nominal_parameters"])
    fs = config["sample_rate_hz"]
    dt = 1.0 / fs
    time, waves = inputs(config, natural_characteristics(parameters)["natural_frequency_hz"])
    evaluation = (time >= config["evaluation_start_s"]) & (time <= config["evaluation_end_s"])
    states = {name: plant(force, parameters, dt) for name, force in waves.items()}
    angles = {name: state[:, 0] for name, state in states.items()}
    training = config["training_inputs"]

    tuning_rows = []
    luenberger_cache = {}
    for order, family in ((0, "ESO constant"), (1, "ESO ramp")):
        for pole in config["luenberger_pole_grid_hz"]:
            values = {
                name: causal_luenberger(angles[name], parameters, dt, pole, order) for name in waves
            }
            luenberger_cache[(order, pole)] = values
            training_score = float(np.mean([score(waves[name], values[name], evaluation) for name in training]))
            tuning_rows.append(
                {"family": family, "parameter": pole, "parameter_unit": "pole_hz", "training_mean_nrmse": training_score}
            )

    kalman_cache = {}
    angle_noise = np.deg2rad(config["kalman_assumed_angle_noise_deg"])
    for force_rw in config["kalman_force_random_walk_grid_N_per_sample"]:
        pair = {
            name: kalman_filter_and_rts_smoother(angles[name], parameters, dt, angle_noise, force_rw)
            for name in waves
        }
        kalman_cache[force_rw] = pair
        for index, family in ((0, "Kalman causal"), (1, "RTS offline")):
            training_score = float(
                np.mean([score(waves[name], pair[name][index], evaluation) for name in training])
            )
            tuning_rows.append(
                {
                    "family": family,
                    "parameter": force_rw,
                    "parameter_unit": "force_random_walk_N_per_sample",
                    "training_mean_nrmse": training_score,
                }
            )

    def best(family):
        candidates = [row for row in tuning_rows if row["family"] == family]
        return min(candidates, key=lambda row: row["training_mean_nrmse"])

    choices = {family: best(family) for family in ("ESO constant", "ESO ramp", "Kalman causal", "RTS offline")}
    constant_pole = choices["ESO constant"]["parameter"]
    ramp_pole = choices["ESO ramp"]["parameter"]
    kalman_q = choices["Kalman causal"]["parameter"]
    rts_q = choices["RTS offline"]["parameter"]

    method_outputs = {}
    for name in waves:
        baseline = estimate(angles[name], parameters, np.ones(5), dt)
        method_outputs[name] = {
            "Static": baseline["Static"],
            "Low-pass 1 Hz": baseline["Low-pass"],
            "ESO constant 1 Hz (current)": baseline["Observer"],
            "ESO constant tuned": luenberger_cache[(0, constant_pole)][name],
            "ESO ramp tuned": luenberger_cache[(1, ramp_pole)][name],
            "Kalman causal tuned": kalman_cache[kalman_q][name][0],
            "RTS same Kalman tuning": kalman_cache[kalman_q][name][1],
            "RTS offline tuned": kalman_cache[rts_q][name][1],
        }

    metric_rows = []
    for input_name, outputs in method_outputs.items():
        for method, values in outputs.items():
            metric_rows.append({"input": input_name, "method": method, **metrics(waves[input_name], values, evaluation)})
    _write_csv(output / "metrics.csv", metric_rows)
    _write_csv(output / "tuning_scan.csv", tuning_rows)

    validation_summary = {}
    for name in config["validation_inputs"]:
        rows = {row["method"]: row for row in metric_rows if row["input"] == name}
        validation_summary[name] = {
            method: {
                "rmse_N": rows[method]["rmse_N"],
                "nrmse": rows[method]["nrmse"],
                "improvement_vs_current_percent": 100.0
                * (1.0 - rows[method]["rmse_N"] / rows["ESO constant 1 Hz (current)"]["rmse_N"]),
            }
            for method in rows
        }

    summary = {
        "scope": "nominal coefficients, ideal angle, no measured angular rate, common-time scoring",
        "training_inputs": training,
        "validation_inputs": config["validation_inputs"],
        "selected_hyperparameters": choices,
        "selection_at_grid_boundary": {
            "ESO constant": constant_pole in (min(config["luenberger_pole_grid_hz"]), max(config["luenberger_pole_grid_hz"])),
            "ESO ramp": ramp_pole in (min(config["luenberger_pole_grid_hz"]), max(config["luenberger_pole_grid_hz"])),
            "Kalman causal": kalman_q
            in (
                min(config["kalman_force_random_walk_grid_N_per_sample"]),
                max(config["kalman_force_random_walk_grid_N_per_sample"]),
            ),
            "RTS offline": rts_q
            in (
                min(config["kalman_force_random_walk_grid_N_per_sample"]),
                max(config["kalman_force_random_walk_grid_N_per_sample"]),
            ),
        },
        "validation": validation_summary,
        "rts_definition": "full-record Rauch-Tung-Striebel fixed-interval smoother",
        "result_status": "COMPARISON_COMPLETE_AWAITING_REVIEW",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_methods = [
        "Static",
        "Low-pass 1 Hz",
        "ESO constant 1 Hz (current)",
        "ESO constant tuned",
        "ESO ramp tuned",
        "Kalman causal tuned",
        "RTS same Kalman tuning",
        "RTS offline tuned",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained")
    windows = {"Random_bandlimited": (60, 80), "Gust": (64, 78)}
    for ax, name in zip(axes, config["validation_inputs"]):
        low, high = windows[name]
        selected = (time >= low) & (time <= high)
        ax.plot(time[selected], 1000 * waves[name][selected], color=COLORS["True force"], linewidth=2.4, label="True force")
        for method in plot_methods:
            ax.plot(
                time[selected],
                1000 * method_outputs[name][method][selected],
                color=COLORS[method],
                linewidth=1.1,
                alpha=0.9,
                label=method,
            )
        ax.set_title(name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Force [mN]")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=7)
    fig.suptitle("Causal observers and future-data smoothers: held-out inputs")
    fig.savefig(output / "timeseries.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, name in zip(axes, config["validation_inputs"]):
        rows = {row["method"]: row for row in metric_rows if row["input"] == name}
        values = [rows[method]["nrmse"] for method in plot_methods]
        ax.barh(plot_methods[::-1], values[::-1], color=[COLORS[m] for m in plot_methods[::-1]])
        ax.set_xlabel("NRMSE")
        ax.set_title(name)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Held-out force-estimation error; lower is better")
    fig.savefig(output / "validation_nrmse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for family, style in (("ESO constant", "o-"), ("ESO ramp", "s-")):
        rows = [row for row in tuning_rows if row["family"] == family]
        axes[0].loglog(
            [row["parameter"] for row in rows],
            [row["training_mean_nrmse"] for row in rows],
            style,
            label=family,
        )
    axes[0].set_xlabel("Repeated pole parameter [Hz]")
    axes[0].set_ylabel("Training mean NRMSE")
    axes[0].legend()
    for family, style in (("Kalman causal", "o-"), ("RTS offline", "s-")):
        rows = [row for row in tuning_rows if row["family"] == family]
        axes[1].loglog(
            [row["parameter"] for row in rows],
            [row["training_mean_nrmse"] for row in rows],
            style,
            label=family,
        )
    axes[1].set_xlabel("Assumed force random-walk std [N/sample]")
    axes[1].set_ylabel("Training mean NRMSE")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Hyperparameter selection on deterministic training inputs")
    fig.savefig(output / "tuning.png", dpi=180)
    plt.close(fig)
    block_diagram(output / "block_diagram.png")
    return summary, metric_rows


def write_report(summary: dict, metrics_rows: list[dict], path: Path) -> None:
    rows = {(row["input"], row["method"]): row for row in metrics_rows}
    methods = [
        "Static",
        "Low-pass 1 Hz",
        "ESO constant 1 Hz (current)",
        "ESO constant tuned",
        "ESO ramp tuned",
        "Kalman causal tuned",
        "RTS same Kalman tuning",
        "RTS offline tuned",
    ]
    lines = [
        "# Stage 2：オブザーバー実装比較",
        "",
        "## 観測データ",
        "",
        "現在のオブザーバーを含め、全方式が観測するのは角度だけです。角速度は測定入力ではなく内部状態として推定します。角度の数値微分値も追加していません。",
        "",
        "![構成](results/stage2_estimator_comparison/block_diagram.png)",
        "",
        "## 比較方式",
        "",
        "| 方式 | 因果性 | 外力モデル | 使用する観測 |",
        "|---|---|---|---|",
        "| 現行ESO | 因果 | トルク一定 | 現在までの角度 |",
        "| 調整ESO | 因果 | トルク一定 | 現在までの角度 |",
        "| Ramp ESO | 因果 | トルク変化率一定 | 現在までの角度 |",
        "| Kalman filter | 因果 | トルクのrandom walk | 現在までの角度 |",
        "| RTS smoother | 非因果 | Kalman filterと同じ | 記録全体の角度 |",
        "",
        "RTSは固定区間スムーザーです。時刻kの推定にkより後の角度を使うため、実時間処理には使えませんが、SDカード記録の事後解析には使用できます。",
        "",
        "## 評価方法",
        "",
        "プラント係数と推定係数は一致、センサノイズなし、100 Hzです。Step、低周波・共振付近・1 Hzの正弦波で係数を選び、選定に使っていないRandom bandlimitedとGustで評価しました。すべて同じ時刻で比較し、時間シフトは行っていません。",
        "",
        "Kalman filterとRTSでは、実データにノイズがない場合でも正則化を定義するため、仮定角度ノイズ0.02度を設定しています。この値は実センサの同定値ではありません。",
        "",
        "## 選択された設定",
        "",
        "| 推定器 | 選択値 | 探索端か |",
        "|---|---:|---|",
    ]
    choices = summary["selected_hyperparameters"]
    for family in ("ESO constant", "ESO ramp", "Kalman causal", "RTS offline"):
        choice = choices[family]
        lines.append(
            f'| {family} | {choice["parameter"]:.8g} {choice["parameter_unit"]} | {"はい" if summary["selection_at_grid_boundary"][family] else "いいえ"} |'
        )
    lines += [
        "",
        "探索端が選ばれた方式は、今回の範囲内の最良候補であり、最適値が確定したことを意味しません。ノイズなしでは帯域を上げるペナルティが現れにくいため、Stage 3のセンサモデル導入後に再調整が必要です。",
        "",
        "## 未使用入力での結果",
        "",
    ]
    for input_name in summary["validation_inputs"]:
        current = rows[(input_name, "ESO constant 1 Hz (current)")]["rmse_N"]
        lines += [
            f"### {input_name}",
            "",
            "| 方式 | RMSE [mN] | NRMSE | 現行ESO比 |",
            "|---|---:|---:|---:|",
        ]
        for method in methods:
            row = rows[(input_name, method)]
            improvement = 100.0 * (1.0 - row["rmse_N"] / current)
            lines.append(
                f'| {method} | {_format_mn(row["rmse_N"])} | {row["nrmse"]:.6g} | {improvement:.2f}% |'
            )
        lines.append("")
    lines += [
        "![時系列比較](results/stage2_estimator_comparison/timeseries.png)",
        "",
        "![NRMSE比較](results/stage2_estimator_comparison/validation_nrmse.png)",
        "",
        "![調整曲線](results/stage2_estimator_comparison/tuning.png)",
        "",
        "## 解釈上の制約",
        "",
        "この比較は外力モデルの違いと未来データ利用の効果を調べる公称・ノイズなし試験です。実機で最良の方式を決める試験ではありません。センサノイズを入れると、高帯域ESO、Ramp ESO、Kalman filterの順位が変わる可能性があります。",
        "",
        "RTS same Kalman tuningは因果Kalman filterと同じQ・Rを使うため、未来データを追加した効果を直接比較できます。RTS offline tunedはオフライン方式として独立に係数を選んだ場合の結果です。",
        "",
        "## 再現方法",
        "",
        "```bash",
        "python 06_Analysis/simulation/src/run_estimator_comparison.py",
        "python -m unittest discover -s 06_Analysis/simulation/tests -v",
        "```",
        "",
        "設定はconfig/estimator_comparison.json、数値結果はresults/stage2_estimator_comparisonに保存します。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "estimator_comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stage2_estimator_comparison")
    args = parser.parse_args()
    summary, rows = run(args.config, args.output)
    write_report(summary, rows, ROOT / "Estimator_Comparison_Report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
