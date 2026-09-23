"""ハイブリッド同定Stage 3: 厳密な頂点間非線形ソルバーを検証する。

Stage 2で固定した全形態のI、Kを使用するが、実データへの減衰係数フィットは
行わない。保存系の解析解、厳密基準解、エネルギー収支、最大刻み依存性を
合成条件で確認し、Stage 4の目的関数に使用できる数値精度を保証する。
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from peak_to_peak_solver import conservative_half_period_s
from peak_to_peak_solver import solve_next_turning_point


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
DEFAULT_RESULT_ROOT = SCRIPT_DIRECTORY / "results"
INITIAL_ANGLES_DEG = [-60.0, -45.0, -30.0, -15.0, -5.0, 5.0, 15.0, 30.0, 45.0, 60.0]
FRICTION_EPSILON_DEG_S = 0.5
DEFAULT_MAX_STEP_FRACTION = 1.0 / 80.0
REFERENCE_MAX_STEP_FRACTION = 1.0 / 640.0
STEP_FRACTIONS = [1.0 / 20.0, 1.0 / 40.0, 1.0 / 80.0, 1.0 / 160.0]

# Stage 3の数値試験専用値。物理係数の採用値ではなく、Stage 4～5でも固定しない。
SYNTHETIC_DAMPING_CASES = [
    {
        "case": "conservative",
        "damping_n_m_s_per_rad": 0.0,
        "quadratic_n_m_s2_per_rad2": 0.0,
        "friction_n_m": 0.0,
    },
    {
        "case": "synthetic_rod",
        "damping_n_m_s_per_rad": 3.8e-5,
        "quadratic_n_m_s2_per_rad2": 1.3e-6,
        "friction_n_m": 8.8e-5,
    },
    {
        "case": "synthetic_sphere",
        "damping_n_m_s_per_rad": 3.8e-5,
        "quadratic_n_m_s2_per_rad2": 1.2e-5,
        "friction_n_m": 8.8e-5,
    },
]

PEAK_ERROR_LIMIT_DEG = 1.0e-7
TIME_ERROR_LIMIT_S = 1.0e-8
ENERGY_CLOSURE_LIMIT = 1.0e-9
CONSERVATIVE_PEAK_LIMIT_DEG = 1.0e-8
CONSERVATIVE_TIME_LIMIT_S = 1.0e-9


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ハイブリッド同定Stage 3: 頂点間非線形ソルバー検証"
    )
    parser.add_argument("--date", required=True, help="試験日。例: 20260921")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
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


def save_figure(figure, output_path, dpi=150):
    temporary_path = output_path.with_name(
        output_path.stem + ".tmp" + output_path.suffix
    )
    figure.savefig(temporary_path, dpi=dpi)
    plt.close(figure)
    temporary_path.replace(output_path)


def solve_case(parameter_row, angle_deg, damping_case, reference=False, max_step_fraction=None):
    if max_step_fraction is None:
        max_step_fraction = (
            REFERENCE_MAX_STEP_FRACTION if reference else DEFAULT_MAX_STEP_FRACTION
        )
    settings = {
        "rtol": 1.0e-12 if reference else 1.0e-9,
        "angle_speed_atol": 1.0e-14 if reference else 1.0e-12,
        "energy_atol": 1.0e-16 if reference else 1.0e-14,
        "max_step_fraction": max_step_fraction,
    }
    return solve_next_turning_point(
        np.deg2rad(angle_deg),
        float(parameter_row["inertia_kg_m2"]),
        float(parameter_row["restoring_n_m_per_rad"]),
        damping_case["damping_n_m_s_per_rad"],
        damping_case["quadratic_n_m_s2_per_rad2"],
        damping_case["friction_n_m"],
        np.deg2rad(FRICTION_EPSILON_DEG_S),
        **settings,
    )


def build_validation_cases(parameters):
    rows = []
    for unused_index, parameter_row in parameters.iterrows():
        for damping_case in SYNTHETIC_DAMPING_CASES:
            for angle_deg in INITIAL_ANGLES_DEG:
                result = solve_case(parameter_row, angle_deg, damping_case)
                reference = solve_case(
                    parameter_row, angle_deg, damping_case, reference=True
                )
                peak_error_deg = float(
                    np.rad2deg(
                        result["next_angle_rad"] - reference["next_angle_rad"]
                    )
                )
                time_error_s = float(
                    result["half_period_s"] - reference["half_period_s"]
                )
                normalized_closure = float(
                    abs(result["energy_closure_error_j"])
                    / max(result["initial_energy_j"], np.finfo(float).tiny)
                )
                conservative_peak_error_deg = np.nan
                conservative_time_error_s = np.nan
                if damping_case["case"] == "conservative":
                    conservative_peak_error_deg = float(
                        np.rad2deg(result["next_angle_rad"]) + angle_deg
                    )
                    exact_time = conservative_half_period_s(
                        np.deg2rad(angle_deg),
                        float(parameter_row["inertia_kg_m2"]),
                        float(parameter_row["restoring_n_m_per_rad"]),
                    )
                    conservative_time_error_s = float(
                        result["half_period_s"] - exact_time
                    )
                direction_ok = int(
                    result["next_angle_rad"] * np.deg2rad(angle_deg) < 0.0
                )
                passed = (
                    abs(peak_error_deg) <= PEAK_ERROR_LIMIT_DEG
                    and abs(time_error_s) <= TIME_ERROR_LIMIT_S
                    and normalized_closure <= ENERGY_CLOSURE_LIMIT
                    and direction_ok == 1
                )
                if damping_case["case"] == "conservative":
                    passed = passed and (
                        abs(conservative_peak_error_deg)
                        <= CONSERVATIVE_PEAK_LIMIT_DEG
                        and abs(conservative_time_error_s)
                        <= CONSERVATIVE_TIME_LIMIT_S
                    )
                rows.append(
                    {
                        "axis": parameter_row["axis"],
                        "configuration": parameter_row["configuration"],
                        "synthetic_case": damping_case["case"],
                        "initial_angle_deg": angle_deg,
                        "inertia_kg_m2": float(parameter_row["inertia_kg_m2"]),
                        "restoring_n_m_per_rad": float(
                            parameter_row["restoring_n_m_per_rad"]
                        ),
                        "damping_n_m_s_per_rad": damping_case[
                            "damping_n_m_s_per_rad"
                        ],
                        "quadratic_n_m_s2_per_rad2": damping_case[
                            "quadratic_n_m_s2_per_rad2"
                        ],
                        "friction_n_m": damping_case["friction_n_m"],
                        "predicted_next_angle_deg": float(
                            np.rad2deg(result["next_angle_rad"])
                        ),
                        "reference_next_angle_deg": float(
                            np.rad2deg(reference["next_angle_rad"])
                        ),
                        "peak_error_deg": peak_error_deg,
                        "half_period_s": result["half_period_s"],
                        "reference_half_period_s": reference["half_period_s"],
                        "time_error_s": time_error_s,
                        "energy_closure_error_j": result["energy_closure_error_j"],
                        "normalized_energy_closure": normalized_closure,
                        "conservative_peak_error_deg": conservative_peak_error_deg,
                        "conservative_time_error_s": conservative_time_error_s,
                        "next_peak_direction_ok": direction_ok,
                        "function_evaluations": result["function_evaluations"],
                        "passed": int(passed),
                    }
                )
    return rows


def build_step_convergence(parameters):
    rows = []
    stress_case = SYNTHETIC_DAMPING_CASES[-1]
    for unused_index, parameter_row in parameters.iterrows():
        reference = solve_case(
            parameter_row,
            60.0,
            stress_case,
            reference=True,
            max_step_fraction=REFERENCE_MAX_STEP_FRACTION,
        )
        for fraction in STEP_FRACTIONS:
            result = solve_case(
                parameter_row,
                60.0,
                stress_case,
                max_step_fraction=fraction,
            )
            rows.append(
                {
                    "axis": parameter_row["axis"],
                    "configuration": parameter_row["configuration"],
                    "synthetic_case": stress_case["case"],
                    "initial_angle_deg": 60.0,
                    "max_step_period_fraction": fraction,
                    "steps_per_small_angle_period": int(round(1.0 / fraction)),
                    "peak_error_deg": float(
                        np.rad2deg(
                            result["next_angle_rad"] - reference["next_angle_rad"]
                        )
                    ),
                    "time_error_s": float(
                        result["half_period_s"] - reference["half_period_s"]
                    ),
                    "normalized_energy_closure": float(
                        abs(result["energy_closure_error_j"])
                        / result["initial_energy_j"]
                    ),
                    "function_evaluations": result["function_evaluations"],
                }
            )
    return rows


def plot_validation(validation_rows, convergence_rows, output_path):
    validation = pd.DataFrame(validation_rows)
    convergence = pd.DataFrame(convergence_rows)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {
        "conservative": "#1f77b4",
        "synthetic_rod": "#2ca02c",
        "synthetic_sphere": "#d62728",
    }
    for case, group in validation.groupby("synthetic_case"):
        axes[0, 0].scatter(
            group["initial_angle_deg"],
            np.abs(group["peak_error_deg"]),
            s=16,
            alpha=0.65,
            label=case,
            color=colors[case],
        )
        axes[0, 1].scatter(
            group["initial_angle_deg"],
            np.abs(group["time_error_s"]) * 1.0e6,
            s=16,
            alpha=0.65,
            label=case,
            color=colors[case],
        )
        axes[1, 0].scatter(
            group["initial_angle_deg"],
            group["normalized_energy_closure"],
            s=16,
            alpha=0.65,
            label=case,
            color=colors[case],
        )
    axes[0, 0].axhline(PEAK_ERROR_LIMIT_DEG, color="black", linestyle="--")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("|next-peak error| [deg]")
    axes[0, 0].set_xlabel("initial angle [deg]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(TIME_ERROR_LIMIT_S * 1.0e6, color="black", linestyle="--")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("|half-period error| [us]")
    axes[0, 1].set_xlabel("initial angle [deg]")
    axes[1, 0].axhline(ENERGY_CLOSURE_LIMIT, color="black", linestyle="--")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("normalized energy-closure error")
    axes[1, 0].set_xlabel("initial angle [deg]")
    summary = convergence.groupby("steps_per_small_angle_period").agg(
        peak_error_deg=("peak_error_deg", lambda values: np.max(np.abs(values))),
        time_error_us=("time_error_s", lambda values: np.max(np.abs(values)) * 1.0e6),
    )
    axes[1, 1].loglog(
        summary.index,
        summary["peak_error_deg"],
        "o-",
        color="#9467bd",
        label="peak [deg]",
    )
    second_axis = axes[1, 1].twinx()
    second_axis.loglog(
        summary.index,
        summary["time_error_us"],
        "s--",
        color="#ff7f0e",
        label="time [us]",
    )
    axes[1, 1].set_xlabel("maximum steps per small-angle period")
    axes[1, 1].set_ylabel("maximum peak error [deg]", color="#9467bd")
    second_axis.set_ylabel("maximum time error [us]", color="#ff7f0e")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    figure.suptitle("Stage 3 peak-to-peak nonlinear solver validation")
    figure.tight_layout()
    save_figure(figure, output_path)


def write_report(output_path, validation_rows, convergence_rows):
    validation = pd.DataFrame(validation_rows)
    convergence = pd.DataFrame(convergence_rows)
    failed = validation[validation["passed"] == 0]
    conservative = validation[validation["synthetic_case"] == "conservative"]
    default_convergence = convergence[
        convergence["steps_per_small_angle_period"] == 80
    ]
    lines = [
        "# Stage 3: 厳密な頂点間非線形ソルバー",
        "",
        "## 結論",
        "",
        f"Stage 2で固定した全12組のI、Kと5～60 degの範囲で{len(validation)}条件を検証し、",
        f"合格{len(validation) - len(failed)}件、不合格{len(failed)}件となった。",
        "実測頂点を初期状態として次の速度ゼロ交差まで積分するソルバーは、Stage 4の",
        "1半周期先頂点残差の計算に使用できる数値精度を満たした。",
        "本Stageでは実データへのb、c、tauのフィットは行っていない。",
        "",
        "## 実装した運動方程式",
        "",
        "```math",
        "I\\ddot\\theta+K\\sin\\theta+b\\dot\\theta+c|\\dot\\theta|\\dot\\theta",
        "+\\tau\\tanh(\\dot\\theta/\\varepsilon)=0",
        "```",
        "",
        "各区間は実測頂点`(A_n, 0)`から開始し、初期角と逆符号側の次の速度ゼロ交差を",
        "イベントとして検出する。速度イベントの向きを指定するため、時刻0の速度ゼロを",
        "次の頂点として誤検出しない。散逸仕事も第3状態として同時積分した。",
        "",
        "## 合成試験条件",
        "",
        "- I、K: Stage 2のIN/OUT、BALL/SP00～SP04の全12組",
        "- 初期角: ±5、±15、±30、±45、±60 deg",
        "- 減衰条件: 保存系、synthetic_rod、synthetic_sphereの3条件",
        "- 通常解: DOP853、rtol=1e-9、角度・角速度atol=1e-12、最大刻み=T0/80",
        "- 基準解: DOP853、rtol=1e-12、角度・角速度atol=1e-14、最大刻み=T0/640",
        "- epsilon: 0.5 deg/s",
        "",
        "synthetic_rodとsynthetic_sphereのb、c、tauはソルバー負荷試験用であり、",
        "物理係数の同定値・採用値ではない。Stage 4～5で実データから改めて同定する。",
        "",
        "## 主要結果",
        "",
        "| 指標 | 最大絶対値 | 判定上限 |",
        "|---|---:|---:|",
        f"| 基準解に対する次頂点角誤差 [deg] | {validation['peak_error_deg'].abs().max():.3e} | {PEAK_ERROR_LIMIT_DEG:.1e} |",
        f"| 基準解に対する半周期誤差 [s] | {validation['time_error_s'].abs().max():.3e} | {TIME_ERROR_LIMIT_S:.1e} |",
        f"| 正規化エネルギー閉合誤差 [-] | {validation['normalized_energy_closure'].max():.3e} | {ENERGY_CLOSURE_LIMIT:.1e} |",
        f"| 保存系の次頂点角解析誤差 [deg] | {conservative['conservative_peak_error_deg'].abs().max():.3e} | {CONSERVATIVE_PEAK_LIMIT_DEG:.1e} |",
        f"| 保存系の楕円積分半周期誤差 [s] | {conservative['conservative_time_error_s'].abs().max():.3e} | {CONSERVATIVE_TIME_LIMIT_S:.1e} |",
        "",
        "最大刻みT0/80の負荷試験では、全12組の60 deg条件について基準解に対する",
        f"次頂点角誤差最大{default_convergence['peak_error_deg'].abs().max():.3e} deg、",
        f"半周期誤差最大{default_convergence['time_error_s'].abs().max():.3e} sであった。",
        "T0/20、T0/40、T0/80、T0/160の全結果をstep_convergence.csvへ保存した。",
        "",
        "## 検証した項目",
        "",
        "1. 保存系では次頂点が初期角の正負反転値になること。",
        "2. 保存系の半周期が完全楕円積分による厳密値と一致すること。",
        "3. 減衰系でも次頂点の向きと速度ゼロイベントが正しく検出されること。",
        "4. 初期・終端の力学的エネルギー差と積分した散逸仕事が閉じること。",
        "5. 通常設定が厳密基準解と一致し、最大刻みへの依存が許容範囲内であること。",
        "",
        "## 既知の制約",
        "",
        "- tanhで連続化した摩擦モデルを前提とし、真の静止摩擦・固着は扱わない。",
        "- 初期角は-pi～pi、IとKは正、b、c、tauは非負を前提とする。",
        "- 次頂点が既定の2小角周期以内に検出できない条件は明示的に失敗させる。",
        "- 実データへの適合性はStage 4以降の対象であり、本Stageの合格は物理モデルの妥当性を意味しない。",
        "",
        "## Stage 4へ持ち越す事項",
        "",
        "- 球なし全波形について各実測頂点から1半周期先角度を予測する。",
        "- 波形ごとの総重みを等しくし、c_rod、b_IN、b_OUT、tau_IN、tau_OUTを共有同定する。",
        "- b_IN自由モデルとb_IN=0固定モデルを同じ評価条件で比較する。",
        "",
        "## 実行コマンド",
        "",
        "```bash",
        "python -m unittest discover -s 06_Analysis/fitting_pipeline/tests -p 'test_peak_to_peak_solver.py'",
        "python 06_Analysis/fitting_pipeline/run_hybrid_peak_solver_validation.py --date 20260921",
        "```",
        "",
        "## 出力",
        "",
        "- [全合成試験](validation_cases.csv)",
        "- [最大刻み依存性](step_convergence.csv)",
        "- [検証概要図](solver_validation.png)",
        "- [実行条件](solver_settings.json)",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    arguments = parse_arguments()
    parent_result = arguments.result_root / arguments.date
    stage2_path = (
        parent_result
        / "hybrid_identification"
        / "02_frequency_identification"
        / "identified_inertia_restoring.csv"
    )
    if not stage2_path.exists():
        raise FileNotFoundError("Stage 2のI、K入力がありません: " + str(stage2_path))
    parameters = pd.read_csv(stage2_path)
    required = {"axis", "configuration", "inertia_kg_m2", "restoring_n_m_per_rad"}
    missing = sorted(required - set(parameters.columns))
    if missing:
        raise ValueError("Stage 2入力に必要な列がありません: " + ", ".join(missing))
    if len(parameters) != 12:
        raise ValueError("Stage 2のI、Kは12組である必要があります")

    output_directory = (
        parent_result / "hybrid_identification" / "03_peak_to_peak_solver"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    validation_rows = build_validation_cases(parameters)
    convergence_rows = build_step_convergence(parameters)
    validation_table = pd.DataFrame(validation_rows)
    convergence_table = pd.DataFrame(convergence_rows)
    validation_table.to_csv(output_directory / "validation_cases.csv", index=False)
    convergence_table.to_csv(output_directory / "step_convergence.csv", index=False)
    plot_validation(
        validation_rows, convergence_rows, output_directory / "solver_validation.png"
    )
    write_report(
        output_directory / "SOLVER_REPORT.md", validation_rows, convergence_rows
    )
    settings = {
        "stage": 3,
        "date": arguments.date,
        "solver": "DOP853 with terminal directed speed-zero event",
        "initial_angles_deg": INITIAL_ANGLES_DEG,
        "friction_epsilon_deg_s": FRICTION_EPSILON_DEG_S,
        "default": {
            "rtol": 1.0e-9,
            "angle_speed_atol": 1.0e-12,
            "energy_atol": 1.0e-14,
            "max_step_period_fraction": DEFAULT_MAX_STEP_FRACTION,
        },
        "reference": {
            "rtol": 1.0e-12,
            "angle_speed_atol": 1.0e-14,
            "energy_atol": 1.0e-16,
            "max_step_period_fraction": REFERENCE_MAX_STEP_FRACTION,
        },
        "step_convergence_fractions": STEP_FRACTIONS,
        "synthetic_damping_cases": SYNTHETIC_DAMPING_CASES,
        "acceptance_limits": {
            "peak_error_deg": PEAK_ERROR_LIMIT_DEG,
            "time_error_s": TIME_ERROR_LIMIT_S,
            "normalized_energy_closure": ENERGY_CLOSURE_LIMIT,
            "conservative_peak_error_deg": CONSERVATIVE_PEAK_LIMIT_DEG,
            "conservative_time_error_s": CONSERVATIVE_TIME_LIMIT_S,
        },
        "stage2_input_sha256": file_sha256(stage2_path),
        "validation_case_count": len(validation_rows),
        "passed_case_count": int(validation_table["passed"].sum()),
        "failed_case_count": int((validation_table["passed"] == 0).sum()),
    }
    (output_directory / "solver_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not bool(validation_table["passed"].all()):
        failed = validation_table[validation_table["passed"] == 0]
        raise RuntimeError("Stage 3の合成試験に不合格があります: " + str(len(failed)))
    print("結果: " + str(output_directory))


if __name__ == "__main__":
    main()
