"""ハイブリッド同定Stage 3: 頂点間非線形ソルバーを検証する。

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
from peak_to_peak_solver import DEFAULT_ANGLE_SPEED_ATOL
from peak_to_peak_solver import DEFAULT_ENERGY_ATOL
from peak_to_peak_solver import DEFAULT_MAX_STEP_FRACTION
from peak_to_peak_solver import DEFAULT_RTOL
from peak_to_peak_solver import solve_next_turning_point


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
DEFAULT_RESULT_ROOT = SCRIPT_DIRECTORY / "results"
INITIAL_ANGLES_DEG = [-60.0, -45.0, -30.0, -15.0, -5.0, 5.0, 15.0, 30.0, 45.0, 60.0]
FRICTION_EPSILON_DEG_S = 0.5
REFERENCE_MAX_STEP_FRACTION = 1.0 / 640.0
STEP_FRACTIONS = [1.0 / 2.0, 1.0 / 5.0, 1.0 / 10.0, 1.0 / 20.0]
LEGACY_RTOL = 1.0e-9
LEGACY_ANGLE_SPEED_ATOL = 1.0e-12
LEGACY_ENERGY_ATOL = 1.0e-14
LEGACY_MAX_STEP_FRACTION = 1.0 / 80.0

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

# Stage 4で観測した約0.49 degの1半周期先RMSEに対し、数値誤差を約2%以下へ
# 制限する。時間は10 ms計測周期の1%、エネルギー閉合は0.01%を上限とする。
PEAK_ERROR_LIMIT_DEG = 1.0e-2
TIME_ERROR_LIMIT_S = 1.0e-4
ENERGY_CLOSURE_LIMIT = 1.0e-4
CONSERVATIVE_PEAK_LIMIT_DEG = PEAK_ERROR_LIMIT_DEG
CONSERVATIVE_TIME_LIMIT_S = TIME_ERROR_LIMIT_S


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


def solve_case(
    parameter_row,
    angle_deg,
    damping_case,
    reference=False,
    legacy=False,
    max_step_fraction=None,
):
    if reference and legacy:
        raise ValueError("referenceとlegacyは同時に指定できません")
    if max_step_fraction is None:
        if reference:
            max_step_fraction = REFERENCE_MAX_STEP_FRACTION
        elif legacy:
            max_step_fraction = LEGACY_MAX_STEP_FRACTION
        else:
            max_step_fraction = DEFAULT_MAX_STEP_FRACTION
    if reference:
        settings = {
            "rtol": 1.0e-12,
            "angle_speed_atol": 1.0e-14,
            "energy_atol": 1.0e-16,
        }
    elif legacy:
        settings = {
            "rtol": LEGACY_RTOL,
            "angle_speed_atol": LEGACY_ANGLE_SPEED_ATOL,
            "energy_atol": LEGACY_ENERGY_ATOL,
        }
    else:
        settings = {
            "rtol": DEFAULT_RTOL,
            "angle_speed_atol": DEFAULT_ANGLE_SPEED_ATOL,
            "energy_atol": DEFAULT_ENERGY_ATOL,
        }
    settings["max_step_fraction"] = max_step_fraction
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
                legacy = solve_case(
                    parameter_row, angle_deg, damping_case, legacy=True
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
                        "legacy_function_evaluations": legacy[
                            "function_evaluations"
                        ],
                        "reference_function_evaluations": reference[
                            "function_evaluations"
                        ],
                        "evaluation_speedup_vs_legacy": float(
                            legacy["function_evaluations"]
                            / result["function_evaluations"]
                        ),
                        "passed": int(passed),
                    }
                )
    return rows


def build_step_convergence(parameters):
    rows = []
    for unused_index, parameter_row in parameters.iterrows():
        for damping_case in SYNTHETIC_DAMPING_CASES:
            for angle_deg in INITIAL_ANGLES_DEG:
                reference = solve_case(
                    parameter_row,
                    angle_deg,
                    damping_case,
                    reference=True,
                    max_step_fraction=REFERENCE_MAX_STEP_FRACTION,
                )
                for fraction in STEP_FRACTIONS:
                    result = solve_case(
                        parameter_row,
                        angle_deg,
                        damping_case,
                        max_step_fraction=fraction,
                    )
                    peak_error_deg = float(
                        np.rad2deg(
                            result["next_angle_rad"]
                            - reference["next_angle_rad"]
                        )
                    )
                    time_error_s = float(
                        result["half_period_s"] - reference["half_period_s"]
                    )
                    normalized_closure = float(
                        abs(result["energy_closure_error_j"])
                        / result["initial_energy_j"]
                    )
                    rows.append(
                        {
                            "axis": parameter_row["axis"],
                            "configuration": parameter_row["configuration"],
                            "synthetic_case": damping_case["case"],
                            "initial_angle_deg": angle_deg,
                            "max_step_period_fraction": fraction,
                            "steps_per_small_angle_period": int(
                                round(1.0 / fraction)
                            ),
                            "peak_error_deg": peak_error_deg,
                            "time_error_s": time_error_s,
                            "normalized_energy_closure": normalized_closure,
                            "function_evaluations": result[
                                "function_evaluations"
                            ],
                            "passed": int(
                                abs(peak_error_deg) <= PEAK_ERROR_LIMIT_DEG
                                and abs(time_error_s) <= TIME_ERROR_LIMIT_S
                                and normalized_closure <= ENERGY_CLOSURE_LIMIT
                            ),
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
    axes[1, 1].axhline(PEAK_ERROR_LIMIT_DEG, color="#9467bd", linestyle=":")
    second_axis.axhline(
        TIME_ERROR_LIMIT_S * 1.0e6, color="#ff7f0e", linestyle=":"
    )
    axes[1, 1].grid(True, which="both", alpha=0.3)
    figure.suptitle("Stage 3 peak-to-peak nonlinear solver validation")
    figure.tight_layout()
    save_figure(figure, output_path)


def write_report(output_path, validation_rows, convergence_rows):
    validation = pd.DataFrame(validation_rows)
    convergence = pd.DataFrame(convergence_rows)
    failed = validation[validation["passed"] == 0]
    conservative = validation[validation["synthetic_case"] == "conservative"]
    evaluation_speedup = float(
        validation["legacy_function_evaluations"].sum()
        / validation["function_evaluations"].sum()
    )
    default_convergence = convergence[
        convergence["steps_per_small_angle_period"]
        == int(round(1.0 / DEFAULT_MAX_STEP_FRACTION))
    ]
    coarser_convergence = convergence[
        convergence["steps_per_small_angle_period"]
        < int(round(1.0 / DEFAULT_MAX_STEP_FRACTION))
    ]
    convergence_summary = (
        convergence.groupby("steps_per_small_angle_period")
        .agg(
            case_count=("passed", "size"),
            failed_count=("passed", lambda values: int((values == 0).sum())),
            peak_error_deg=("peak_error_deg", lambda values: np.max(np.abs(values))),
            time_error_s=("time_error_s", lambda values: np.max(np.abs(values))),
            normalized_energy_closure=("normalized_energy_closure", "max"),
            mean_function_evaluations=("function_evaluations", "mean"),
        )
        .sort_index()
    )
    lines = [
        "# Stage 3: 計測精度に合わせた頂点間非線形ソルバー",
        "",
        "## 結論",
        "",
        f"Stage 2で固定した全12組のI、Kと5～60 degの範囲で{len(validation)}条件を検証し、",
        f"合格{len(validation) - len(failed)}件、不合格{len(failed)}件となった。",
        "実測頂点を初期状態として次の速度ゼロ交差まで積分するソルバーは、Stage 4の",
        "1半周期先頂点残差の計算に使用できる数値精度を満たした。",
        f"従来設定に対する関数評価回数は全条件合計で1/{evaluation_speedup:.2f}となった。",
        "本Stageでは実データへのb、c、tauのフィットは行っていない。",
        "ここで報告する誤差は、理想・合成波形に対する通常解、厳密基準解、解析解の差であり、",
        "実測波形に対するモデル誤差ではない。",
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
        "## Stage 4との関係",
        "",
        "実測頂点A_nから候補係数で次頂点を予測し、半周期ごとに次の残差を生成する。",
        "",
        "```math",
        "r_{r,n}=\\hat A_{r,n+1}-A_{r,n+1}",
        "```",
        "",
        "Stage 4の係数集約・最適化方法は別途見直すが、どの方法でもODE評価が必要な場合は",
        "本Stageで検証した高速設定を使用する。各半周期の開始時に実測頂点へ戻すため、",
        "1半周期先評価では前区間の誤差は累積しない。",
        "解放直後の最初の半周期はStage 1で除外済みであり、その方針を維持する。",
        "",
        "## 半周期分割積分を採用する理由",
        "",
        "本解析は、半周期ごとに別々のb、c、tauをフィットする方法ではない。",
        "同じ波形内では共通の候補係数b、c、tauを使用し、実測波形を隣接する頂点間に",
        "分割して、各半周期を独立した初期値問題として数値積分する。",
        "",
        "```math",
        "A_n \\xrightarrow[\\dot\\theta(0)=0]{I,K,b,c,\\tau,\\varepsilon} \\hat A_{n+1},",
        "\\qquad r_n=\\hat A_{n+1}-A_{n+1}",
        "```",
        "",
        "各A_nは実測した折返し頂点であり、速度ゼロの状態である。したがって、速度を",
        "人為的にゼロへ変更する操作ではない。ここでいうリセットの本質は、前区間の予測終端",
        "値を次区間へ引き継がず、次区間を実測頂点A_{n+1}から開始することである。",
        "",
        "### 半周期をまたいで共通とする量",
        "",
        "- 同じ波形に適用する候補減衰係数b、c、tau",
        "- Stage 2で固定したI、K",
        "- 摩擦平滑化係数epsilonと数値ソルバー設定",
        "",
        "Stage 4で複数波形間の係数をどの範囲まで共通化するかは別途決定するが、少なくとも",
        "一つの波形内で半周期ごとに別のb、c、tauを与えることはしない。",
        "",
        "### 各半周期の開始時に初期化する量",
        "",
        "- 角度を、その区間の実測開始頂点A_nに設定する。",
        "- 角速度を、折返し頂点の条件dot(theta)=0に設定する。",
        "- 区間内の時刻と散逸仕事の積算値を0から開始する。",
        "- 数値積分器の内部状態と刻み幅を初期化する。",
        "- 前区間の予測誤差を次区間の初期状態へ伝播させない。",
        "",
        "### 分割積分と連続積分の得失",
        "",
        "| 方法 | 主目的 | 利点 | 制約・注意点 |",
        "|---|---|---|---|",
        "| 半周期分割積分 | 現在の実測頂点から次の頂点までの局所的な減衰を評価し、b、c、tauを同定する | 周期、位相、初期条件の小さな誤差を後続区間へ累積させず、振幅低下と散逸特性に注目できる。実測値で状態を逐次補正するオブザーバの利用形態にも近い。 | 実測頂点の検出誤差を各区間の初期値として取り込む。長時間の開ループ予測誤差や、同じ符号の小さな誤差の累積は同定残差だけでは評価できない。 |",
        "| 全区間の連続積分 | 最初の頂点から最後までの開ループ再現性と累積モデル誤差を評価する | 振幅包絡線、周期、位相の長時間整合性を一括して確認でき、系統誤差の蓄積を検出できる。 | I、K、初期条件、計測時刻のわずかな誤差による位相ずれが後半で支配的となり、b、c、tauが減衰以外の誤差まで補償する可能性がある。 |",
        "",
        "同じ時間範囲と精度であれば、両方式が積分する物理時間の合計は概ね同じである。",
        "分割積分にはソルバー初期化とイベント検出の反復負荷が加わるが、演算量が桁違いに",
        "増えるわけではない。したがって、分割積分の採用理由は高速化ではなく、減衰係数の",
        "同定から周期・位相誤差の累積を分離することにある。",
        "",
        "### 本解析での使い分け",
        "",
        "Stage 4では、半周期分割積分による1半周期先頂点残差を主目的関数として係数を同定する。",
        "その後、求めた係数を変更せず、最初の有効頂点から最後まで状態を実測値へ戻さない",
        "連続積分を行い、実波形、実測頂点列、振幅包絡線、周期および累積誤差を確認する。",
        "分割積分では良好だが連続積分で系統的なずれが生じる場合は、係数を直ちに連続波形へ",
        "合わせ直すのではなく、I、K、初期条件、頂点検出および減衰モデル不足を切り分ける。",
        "この構成により、減衰係数の局所同定とモデル全体の長時間妥当性確認を両立する。",
        "",
        "## 合成試験条件",
        "",
        "- I、K: Stage 2のIN/OUT、BALL/SP00～SP04の全12組",
        "- 初期角: ±5、±15、±30、±45、±60 deg",
        "- 減衰条件: 保存系、synthetic_rod、synthetic_sphereの3条件",
        "- 通常解: DOP853、rtol=1e-4、角度・角速度atol=1e-7、エネルギーatol=1e-11、最大刻み=T0/5",
        "- 従来解: DOP853、rtol=1e-9、角度・角速度atol=1e-12、エネルギーatol=1e-14、最大刻み=T0/80",
        "- 基準解: DOP853、rtol=1e-12、角度・角速度atol=1e-14、最大刻み=T0/640",
        "- epsilon: 0.5 deg/s",
        "",
        "synthetic_rodとsynthetic_sphereのb、c、tauはソルバー負荷試験用であり、",
        "物理係数の同定値・採用値ではない。Stage 4～5で実データから改めて同定する。",
        "",
        "## 固定数値と導出根拠",
        "",
        "| 数値 | 適用範囲 | 導出・採用根拠 |",
        "|---|---|---|",
        "| epsilon=0.5 deg/s | 摩擦のtanh連続化 | 速度ゼロの不連続を数値的に避けつつ、摩擦が0.5 deg/sで76%、1.0 deg/sで96%、1.5 deg/sで99.5%となり、平滑化を低速域へ限定する初期値。従来モデルから継承した数値正則化値で、実測同定値ではない。まずこの値でStage 4を実施する。 |",
        "| 頂点振幅4 deg以上 | Stage 4以降の減衰同定 | 停止直前の固着、頂点検出、中心誤差の影響を避ける従来解析からの初期下限。Stage 2のK範囲では4 deg時の復元トルクは約0.82～1.56 mN mで、従来の暫定tau約0.09 mN mの約9～18倍。本Stageでは使用せず、Stage 4で適用する。 |",
        "| 次頂点角誤差上限0.01 deg | 通常解 | Stage 4で観測した1半周期先RMSE約0.49 degの約2%に数値誤差を制限する。計測・モデル誤差より十分小さく、過剰精度を避ける。 |",
        "| 半周期誤差上限0.0001 s | 通常解 | ロガーの10 ms計測周期の1%に数値誤差を制限する。 |",
        "| 正規化エネルギー閉合上限1e-4 | 通常解 | 散逸仕事の数値診断を0.01%以内で閉じる。係数同定の主評価は次頂点角であり、エネルギー状態だけが過剰に刻みを細かくしない上限とする。 |",
        "| 最大刻みT0/5 | 通常解 | T0/2、T0/5、T0/10、T0/20を全360条件で基準解と比較し、T0/2より細かい候補のうち最も粗く、全受入基準を満たす設定。 |",
        "| rtol=1e-4、角度・角速度atol=1e-7、エネルギーatol=1e-11 | 通常解 | 全12組、3減衰条件、±5～±60 degの360条件で上記3基準を満たし、従来設定から関数評価回数を削減できる組合せ。エネルギーatol=1e-10ではT0/10の低振幅4条件で閉合基準を超えたため、1e-11を採用した。 |",
        "| 最大刻みT0/640、rtol=1e-12 | 基準解 | 通常解より十分厳しい独立比較基準。保存系では楕円積分解析解とも一致することを確認する。 |",
        "| 探索上限2T0 | 次頂点イベント | 通常の次頂点は概ね0.5～0.6T0で現れるため3倍以上の探索余裕を持たせ、次頂点が現れない病的条件は明示的に失敗させる。 |",
        "",
        "今後新たな固定数値を導入する場合は、数値、適用範囲、導出根拠、実測値か",
        "運用上の初期値か、必要な感度確認をレポートまたは設定ファイルへ必ず記録する。",
        "理由を記録できない数値は固定仕様として採用しない。",
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
        "### 速度比較",
        "",
        "| 指標 | 値 |",
        "|---|---:|",
        f"| 関数評価回数平均（従来） | {validation['legacy_function_evaluations'].mean():.1f} |",
        f"| 関数評価回数平均（新設定） | {validation['function_evaluations'].mean():.1f} |",
        f"| 関数評価回数による高速化倍率 | {evaluation_speedup:.2f} |",
        "",
        "### 最大刻み幅の感度",
        "",
        "| 最大刻み | 不合格/全条件 | 次頂点角誤差最大 [deg] | 半周期誤差最大 [s] | 正規化エネルギー閉合最大 | 関数評価回数平均 |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| T0/{int(steps)} | {int(row['failed_count'])}/{int(row['case_count'])} | "
            f"{row['peak_error_deg']:.3e} | {row['time_error_s']:.3e} | "
            f"{row['normalized_energy_closure']:.3e} | {row['mean_function_evaluations']:.1f} |"
            for steps, row in convergence_summary.iterrows()
        ],
        "",
        "最大刻みT0/5の全360条件試験では、基準解に対する",
        f"次頂点角誤差最大{default_convergence['peak_error_deg'].abs().max():.3e} deg、",
        f"半周期誤差最大{default_convergence['time_error_s'].abs().max():.3e} s、",
        f"正規化エネルギー閉合最大{default_convergence['normalized_energy_closure'].max():.3e}であった。",
        f"これより粗いT0/2では{int((coarser_convergence['passed'] == 0).sum())}/{len(coarser_convergence)}件が不合格となった。",
        "T0/2、T0/5、T0/10、T0/20の全結果をstep_convergence.csvへ保存した。",
        "これらは理想・合成波形に対する数値解の整合性確認であり、実測波形との",
        "一致度やb、c、tauの物理的妥当性を示す値ではない。",
        "",
        "## 検証した項目",
        "",
        "1. 保存系では次頂点が初期角の正負反転値になること。",
        "2. 保存系の半周期が完全楕円積分による厳密値と一致すること。",
        "3. 減衰系でも次頂点の向きと速度ゼロイベントが正しく検出されること。",
        "4. 初期・終端の力学的エネルギー差と積分した散逸仕事が閉じること。",
        "5. 通常設定が計測精度由来の受入基準内で基準解と一致すること。",
        "6. 従来設定より関数評価回数が減少すること。",
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
        "- 係数最適化方法の見直し後、本Stageの高速設定をODEによる最終検証へ使用する。",
        "- 厳密基準設定は収束確認専用とし、反復最適化には使用しない。",
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
            "rtol": DEFAULT_RTOL,
            "angle_speed_atol": DEFAULT_ANGLE_SPEED_ATOL,
            "energy_atol": DEFAULT_ENERGY_ATOL,
            "max_step_period_fraction": DEFAULT_MAX_STEP_FRACTION,
        },
        "legacy_default": {
            "rtol": LEGACY_RTOL,
            "angle_speed_atol": LEGACY_ANGLE_SPEED_ATOL,
            "energy_atol": LEGACY_ENERGY_ATOL,
            "max_step_period_fraction": LEGACY_MAX_STEP_FRACTION,
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
        "acceptance_basis": {
            "peak_error": "0.01 deg is about 2% of the approximately 0.49 deg Stage 4 one-half-cycle RMSE",
            "time_error": "0.0001 s is 1% of the 0.01 s logger sampling interval",
            "normalized_energy_closure": "1e-4 limits the diagnostic closure error to 0.01% without making the energy state dominate the ODE step size",
        },
        "performance": {
            "legacy_mean_function_evaluations": float(
                validation_table["legacy_function_evaluations"].mean()
            ),
            "default_mean_function_evaluations": float(
                validation_table["function_evaluations"].mean()
            ),
            "total_function_evaluation_speedup": float(
                validation_table["legacy_function_evaluations"].sum()
                / validation_table["function_evaluations"].sum()
            ),
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
    if not (
        validation_table["function_evaluations"].sum()
        < validation_table["legacy_function_evaluations"].sum()
    ):
        raise RuntimeError("新設定で関数評価回数が削減されていません")
    print("結果: " + str(output_directory))


if __name__ == "__main__":
    main()
