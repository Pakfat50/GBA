"""OW-02: rerun earlier observer candidates with adopted HBK coefficients.

The plant and nonlinear observer models use the same I, K, c and tau values
for each axis, with b fixed to zero and an ideal angle sensor. The wind input
is a bounded Kaimal-spectrum record. Existing observer tunings are retained
for this rerun so that the mechanical-model change is isolated; tuning these
values again belongs to OW-03.

Run from the repository root:
    python 06_Analysis/simulation/src/run_ow02_hbk_observer_rerun.py

The script writes a CSV, JSON summary, comparison figure and Japanese report
under results/observer_wind/ow02_baseline.
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


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = SIMULATION_ROOT / "config" / "ow02_hbk_observer_rerun.json"
DEFAULT_OUTPUT = SIMULATION_ROOT / "results" / "observer_wind" / "ow02_baseline"

METHOD_COLORS = {
    "静的換算": "#777777",
    "因果LPF": "#8e6bbd",
    "ESO 3状態": "#2776bc",
    "ESO 4状態": "#e67e22",
    "RTS 3状態（オフライン）": "#159477",
    "RTS 4状態（オフライン）": "#006b4f",
}
PLOT_LABELS = {
    "静的換算": "Static conversion",
    "因果LPF": "Causal LPF",
    "ESO 3状態": "ESO 3-state",
    "ESO 4状態": "ESO 4-state",
    "RTS 3状態（オフライン）": "RTS 3-state (offline)",
    "RTS 4状態（オフライン）": "RTS 4-state (offline)",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    """Save a table with a header, using UTF-8 so Japanese opens correctly."""
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Calculate direct estimation errors over the selected time interval."""
    error = np.asarray(estimate) - np.asarray(truth)
    absolute = np.abs(error)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(absolute)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_error": float(np.max(absolute)),
    }


def make_wind(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate the common wind record and convert it to aerodynamic force."""
    time, speed, wind_stats = synthesize_kaimal_wind(
        config["sample_rate_hz"], config["duration_s"],
        config["mean_wind_speed_m_s"], config["target_turbulence_intensity"],
        config["kaimal_integral_scale_m"], config["wind_seed"],
        config["maximum_wind_speed_m_s"],
    )
    force = drag_force_from_speed(
        speed, config["air_density_kg_m3"], config["drag_coefficient"],
        config["projected_area_m2"],
    )
    return time, speed, force, wind_stats


def estimate_all(angle: np.ndarray, coefficients: dict,
                 config: dict, sample_period_s: float) -> dict[str, np.ndarray]:
    """Run each earlier comparison method with its prior tuning value."""
    tuning = config["observer_settings_from_previous_wind_study"]
    lever = config["force_lever_m"]
    static = coefficients["restoring_n_m_per_rad"] / lever * np.tan(angle)
    causal_rts3, smoothed_rts3 = nonlinear_ekf_rts_force(
        angle, coefficients, sample_period_s, lever,
        np.deg2rad(config["assumed_angle_noise_deg"]),
        tuning["rts3_force_random_walk_n_per_sample"], 0,
        config["friction_epsilon_deg_s"],
    )
    causal_rts4, smoothed_rts4 = nonlinear_ekf_rts_force(
        angle, coefficients, sample_period_s, lever,
        np.deg2rad(config["assumed_angle_noise_deg"]),
        tuning["rts4_force_rate_random_walk_n_per_s_per_sample"], 1,
        config["friction_epsilon_deg_s"],
    )
    return {
        "静的換算": static,
        "因果LPF": causal_lowpass(static, tuning["causal_lpf_cutoff_hz"], sample_period_s),
        "ESO 3状態": nonlinear_luenberger_force(
            angle, coefficients, sample_period_s, lever,
            tuning["eso3_pole_hz"], 0, config["friction_epsilon_deg_s"],
        ),
        "ESO 4状態": nonlinear_luenberger_force(
            angle, coefficients, sample_period_s, lever,
            tuning["eso4_pole_hz"], 1, config["friction_epsilon_deg_s"],
        ),
        # RTS is deliberately reported as an offline, future-data estimate.
        "RTS 3状態（オフライン）": smoothed_rts3,
        "RTS 4状態（オフライン）": smoothed_rts4,
    }


def save_comparison_figure(path: Path, time: np.ndarray, true_speed: np.ndarray,
                           estimates_by_axis: dict[str, dict[str, np.ndarray]],
                           config: dict, evaluation_mask: np.ndarray) -> None:
    """Save two-axis overlay and error panels for the complete evaluation window."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex="col")
    local_time = time[evaluation_mask] - time[evaluation_mask][0]
    for row, axis in enumerate(("IN", "OUT")):
        speed_axis, error_axis = axes[row]
        speed_axis.plot(local_time, true_speed[evaluation_mask], color="black",
                        linewidth=1.5, label="Input wind speed")
        for method, estimated_force in estimates_by_axis[axis].items():
            # OUT 4-state ESO diverges with legacy tuning. Plot that failure
            # separately so it does not hide the remaining estimates.
            if axis == "OUT" and method == "ESO 4状態":
                continue
            estimated_speed = wind_speed_from_force(
                estimated_force, config["air_density_kg_m3"],
                config["drag_coefficient"], config["projected_area_m2"],
            )
            error = estimated_speed[evaluation_mask] - true_speed[evaluation_mask]
            color = METHOD_COLORS[method]
            speed_axis.plot(local_time, estimated_speed[evaluation_mask],
                            color=color, linewidth=0.9, alpha=0.9, label=PLOT_LABELS[method])
            error_axis.plot(local_time, error, color=color, linewidth=0.9, label=PLOT_LABELS[method])
        speed_axis.set_ylabel(f"{axis} axis wind speed [m/s]")
        error_axis.set_ylabel("Estimation error [m/s]")
        error_axis.axhline(0.0, color="black", linewidth=0.8)
        speed_axis.grid(alpha=0.22)
        error_axis.grid(alpha=0.22)
        if row == 0:
            speed_axis.legend(ncol=3, fontsize=8, loc="upper left")
            error_axis.legend(ncol=2, fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel("Elapsed time in evaluation window [s]")
    axes[-1, 1].set_xlabel("Elapsed time in evaluation window [s]")
    fig.suptitle("OW-02 Observer rerun with adopted HBK coefficients (matched model, ideal angle)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_divergence_figure(path: Path, time: np.ndarray, true_speed: np.ndarray,
                           estimated_force: np.ndarray, config: dict,
                           evaluation_mask: np.ndarray) -> None:
    """Show the OUT 4-state ESO failure separately from in-range methods."""
    estimated_speed = wind_speed_from_force(
        estimated_force, config["air_density_kg_m3"], config["drag_coefficient"],
        config["projected_area_m2"],
    )
    local_time = time[evaluation_mask] - time[evaluation_mask][0]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(local_time, true_speed[evaluation_mask], color="black", linewidth=1.4,
            label="Input wind speed")
    ax.plot(local_time, estimated_speed[evaluation_mask], color=METHOD_COLORS["ESO 4状態"],
            linewidth=0.9, label="OUT ESO 4-state (legacy tuning)")
    ax.set_xlabel("Elapsed time in evaluation window [s]")
    ax.set_ylabel("Wind speed [m/s]")
    ax.set_title("OUT-axis ESO 4-state: unstable with previous model tuning")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(path: Path, config: dict, wind_stats: dict,
                 angle_max_deg: dict[str, float], force_rows: list[dict],
                 speed_rows: list[dict], previous_case_check: dict,
                 eso_gain_vectors: dict[str, dict[str, list[float]]]) -> None:
    """Write a concise Japanese report with assumptions and evaluation limits."""
    lines = [
        "# OW-02 採用HBK係数によるオブザーバー再比較",
        "",
        "## 目的と結論",
        "",
        "従来のオブザーバー比較を、新ハードで採用した機械係数と非線形損失モデルに置き換えて再計算した。プラントと推定モデルには軸ごとに同一係数を与え、センサー角度は理想値とした。オブザーバーの係数は過去モデル用に調整した値をそのまま使ったため、今回の順位や誤差を最終選定結果とはみなさない。係数一致時の基準性能を得て、OW-03で独立した風入力を使って再調整・評価するための再実装である。",
        "",
        "## 条件",
        "",
        f"- 構成：BALL、IN軸とOUT軸を別々に評価。両軸に同じ風速時系列を与え、軸係数による差を比較。",
        f"- 標本化周波数：{config['sample_rate_hz']:.1f} Hz。計算時間：{config['duration_s']:.0f} s。評価区間：{config['evaluation_start_s']:.0f}〜{config['evaluation_end_s']:.0f} s。",
        f"- 風：Kaimal型乱流、平均{config['mean_wind_speed_m_s']:.1f} m/s、最大{config['maximum_wind_speed_m_s']:.1f} m/s、目標TI {100*wind_stats['target_turbulence_intensity']:.1f}%。生成TIは{100*wind_stats['realized_turbulence_intensity']:.2f}%。乱数seed={config['wind_seed']}。",
        f"- 抗力係数：球のIHB-05方式A等価値 Cd={config['drag_coefficient']:.4f}。球径100 mm。作用腕長={config['force_lever_m']*1000:.2f} mm。",
        f"- 実機の角度リミット：±{config['mechanical_angle_limit_deg']:.0f}°。新ハード自由振動46波形の初期振幅中央値は58.392°（Stage 1前処理レポート）で、リミット近傍まで振れている。",
        f"- センサー：角度ノイズ、量子化、遅延なし。推定器内のカルマン観測ノイズ仮定は{config['assumed_angle_noise_deg']:.3f}°で、これは計測波形に加えたノイズではない。",
        "- 損失モデル：b=0、ロッドと球の二乗抗力、および軸別tauを使用。摩擦符号関数は頂点近傍の数値積分を安定させるため0.5°/sのtanh近似。",
        "- ESO/RTS/LPFの調整値：旧モデルで得た値を固定。今回のモデル用には再調整していない。",
        "- RTSは全評価区間の将来角度データを使うオフライン推定であり、オンライン候補とは別に解釈する。",
        "",
        "## 評価結果",
        "",
        "風速RMSEは推定風速と入力した真の風速の差。力RMSEは推定抗力とプラントに与えた真の抗力の差。どちらも評価区間全体で算出した。",
        "",
        "プラントと推定モデルで共通に用いた式を示す。摩擦の符号関数は、頂点付近の数値積分を安定させるため滑らかなtanh近似とした。",
        "",
        "$$",
        r"I\ddot{\theta}+K\sin\theta+c|\dot{\theta}|\dot{\theta}+\tau\tanh\left(\frac{\dot{\theta}}{\epsilon}\right)=\ell F\cos\theta,\qquad b=0",
        "$$",
        "",
        "## 推定方式の定義と計算式",
        "",
        "角度計測を `y_k=theta_k`、推定した風力を `F_k` とする。各推定器は、このレポート冒頭の非線形運動方程式を使い、既知の `I`、`K`、`c`、`tau` と作用腕 `ell` を与える。推定した符号付き力を最後に準定常抗力式で風速へ換算する。",
        "",
        "### 静的換算（オブザーバーではない比較基準）",
        "",
        "角速度と角加速度がほぼゼロで、摩擦・空力減衰の影響を無視できる静的釣り合いを仮定する。運動方程式の復元トルクと風トルクを釣り合わせると、角度から力を直接求められる。運動の履歴や角度変化の速さは使わない。",
        "",
        "$$",
        r"\ell F\cos\theta=K\sin\theta\quad\Longrightarrow\quad\hat F_{\mathrm{static},k}=\frac{K}{\ell}\tan(y_k)",
        "$$",
        "",
        "### 因果3段LPF",
        "",
        "静的換算した力を、同じ一次ローパスフィルターに3回通す。OW-02では遮断周波数 `f_c=0.5 Hz`、標本間隔 `Delta t=0.01 s` とした。次式を各段 `j=1,2,3` に順に適用する。実装の差分式は各段で入力を1標本遅らせ、初期フィルター状態はゼロである。したがって3段の因果フィルターで遅れが生じる。風速に変換した後を平滑化するのではなく、力を平滑化してから風速へ換算する。",
        "",
        "$$",
        r"q=\exp(-2\pi f_c\Delta t),\qquad z^{(j)}_k=qz^{(j)}_{k-1}+(1-q)u^{(j)}_{k-1},\quad u^{(1)}_k=\hat F_{\mathrm{static},k},\quad u^{(j)}_k=z^{(j-1)}_k",
        "$$",
        "",
        "### 3状態ESO（非線形Luenbergerオブザーバー）",
        "",
        "状態は `x=[theta, omega, F]^T` の3つである。`F` を、短い時間では一定とみなす未知外力（風力）として状態に加える。角度だけを計測し、角度予測と計測の差（イノベーション）を使って3状態すべてを補正する。3状態とは、角度・角速度・未知風力を推定状態として持つことを指す。",
        "",
        "$$",
        r"\dot{x}=f_3(x)=\begin{bmatrix}\omega\\\frac{\ell F\cos\theta-K\sin\theta-c|\omega|\omega-\tau\tanh(\omega/\epsilon)}{I}\\0\end{bmatrix},\qquad y=Cx,\quad C=\begin{bmatrix}1&0&0\end{bmatrix}",
        "$$",
        "",
        "非線形状態方程式を1標本分RK4で予測し、静止位置の線形化モデルから離散時間の補正ゲイン `L_3` を設計する。実装上の更新は次式で、`Phi` はRK4による1標本予測である。予測で運動モデルを進め、角度残差を用いて次状態を補正する。",
        "",
        "$$",
        r"\hat{x}_{k+1}=\Phi_{\Delta t}(\hat{x}_k)+L_3\bigl(y_k-C\hat{x}_k\bigr),\qquad \hat{F}_k=(\hat{x}_k)_3",
        "$$",
        "",
        "### 4状態ESO（風力変化率を加えた非線形Luenbergerオブザーバー）",
        "",
        "3状態との違いは、風力の変化率 `r_F` を4つ目の状態に加える点である。これにより、風力が短い時間に一定値でなく、おおむね一定の傾きで変わる状況を表せる。ただし、さらに変化率の変化（風力の二階微分）はゼロと仮定する。この追加状態は変化を追える可能性を増す一方、状態とゲインが増えるため、ノイズや調整値への感度も高くなる。",
        "",
        "$$",
        r"x=\begin{bmatrix}\theta&\omega&F&r_F\end{bmatrix}^{\mathsf T},\qquad \dot{x}=f_4(x)=\begin{bmatrix}\omega\\\frac{\ell F\cos\theta-K\sin\theta-c|\omega|\omega-\tau\tanh(\omega/\epsilon)}{I}\\r_F\\0\end{bmatrix}",
        "$$",
        "",
        "更新式は3状態と同じ形だが、4次元の線形化モデルで4つの誤差極を配置して `L_4` を求める。",
        "",
        "$$",
        r"\hat{x}_{k+1}=\Phi_{\Delta t}(\hat{x}_k)+L_4\bigl(y_k-C\hat{x}_k\bigr),\qquad C=\begin{bmatrix}1&0&0&0\end{bmatrix},\qquad \hat{F}_k=(\hat{x}_k)_3",
        "$$",
        "",
        "今回のESOはどちらも角度誤差極を反復配置するが、旧比較で使った帯域は3状態45 Hz、4状態20 Hzで異なる。したがってOW-02の3状態対4状態の誤差差は、状態数だけでなく帯域設定の差も含む。今回の比較から「4状態の方が本質的に劣る」とは結論できない。",
        "",
        "両ESOのゲインは、静止位置 `(theta, omega, F, r_F)=(0,0,0,0)` で線形化し、標本時間に対して離散化したモデルから求める。原点では二乗抵抗の微分はゼロ、平滑化クーロン摩擦の傾きは `tau/epsilon` となる。各方式で `n` 状態の全誤差極が同じ `z_p` になるようにゲインを置く。",
        "",
        "$$",
        r"A_3=\begin{bmatrix}0&1&0\\-K/I&-\tau/(I\epsilon)&\ell/I\\0&0&0\end{bmatrix},\quad A_4=\begin{bmatrix}0&1&0&0\\-K/I&-\tau/(I\epsilon)&\ell/I&0\\0&0&0&1\\0&0&0&0\end{bmatrix},\quad A_{d,n}=\exp(A_n\Delta t)",
        "$$",
        "",
        "$$",
        r"z_p=\exp(-2\pi f_p\Delta t),\qquad\mathrm{eig}(A_{d,n}-L_nC)=\{z_p,\ldots,z_p\}\ (n\text{重根})",
        "$$",
        "",
        "この表の極周波数は旧調整値として引き継いだ設定である。一方、数値ゲイン `L_3`、`L_4` は、その極周波数とOW-02の採用係数（軸ごとの `I`、`K`、`c`、`tau`）から実行時に再計算した値であり、旧モデルで使った数値ゲインをそのまま流用したものではない。状態補正式に代入する列ベクトルを示す。成分の順は状態の順 `[theta, omega, F]` または `[theta, omega, F, r_F]` で、角度残差 `y_k-C xhat_k`（rad）に掛ける。OW-03でゲイン調整後に同じ表を再生成すれば、今回の基準値と比較できる。",
        "",
        "| 軸 | ESO | 極周波数 (Hz) | 離散極 `z_p` | 補正ゲイン列ベクトル `L` |",
        "|---|---|---:|---:|---|",
        *[f"| {axis} | {method} | {pole_hz:.3f} | {np.exp(-2*np.pi*pole_hz/config['sample_rate_hz']):.8g} | `[{', '.join(f'{value:.8g}' for value in eso_gain_vectors[axis][method])}]^T` |"
          for axis in ("IN", "OUT")
          for method, pole_hz in (("3状態", config["observer_settings_from_previous_wind_study"]["eso3_pole_hz"]),
                                  ("4状態", config["observer_settings_from_previous_wind_study"]["eso4_pole_hz"]))],
        "",
        "### RTS 3状態・4状態（EKF＋Rauch–Tung–Striebel平滑化）",
        "",
        "RTS候補も同じ3状態・4状態のモデルを使うが、誤差を角度残差で直接補正するESOではなく、拡張カルマンフィルター（EKF）で共分散に応じて補正する。3状態では風力 `F` にランダムウォーク雑音を、4状態では風力変化率 `r_F` にランダムウォーク雑音を与える。さらに記録の末尾から先頭へ平滑化し、未来の角度計測も使って過去の状態推定を修正するため、RTS出力はオフライン専用である。",
        "",
        "$$",
        r"\hat{x}_{k|k-1}=f_{\Delta t}(\hat{x}_{k-1|k-1}),\quad P_{k|k-1}=A_kP_{k-1|k-1}A_k^{\mathsf T}+Q,\quad S_k=CP_{k|k-1}C^{\mathsf T}+R",
        "$$",
        "",
        "$$",
        r"K_k=P_{k|k-1}C^{\mathsf T}S_k^{-1},\quad\hat{x}_{k|k}=\hat{x}_{k|k-1}+K_k(y_k-C\hat{x}_{k|k-1}),\quad P_{k|k}=(I-K_kC)P_{k|k-1}(I-K_kC)^{\mathsf T}+K_kRK_k^{\mathsf T}",
        "$$",
        "",
        "$$",
        r"G_k=P_{k|k}A_{k+1}^{\mathsf T}(P_{k+1|k})^{-1},\qquad\hat{x}_{k|N}=\hat{x}_{k|k}+G_k(\hat{x}_{k+1|N}-\hat{x}_{k+1|k})",
        "$$",
        "",
        "ここで `A_k` は非線形運動モデルの局所線形化、`Q` は状態雑音、`R` は角度計測雑音の分散、`N` は記録末尾である。OW-02では計測ノイズを波形へ加えていないが、EKF内部の仮定として標準偏差0.020°を設定した。過去設定はRTS 3状態で `F` の雑音30 N/標本、RTS 4状態で `r_F` の雑音10 N/s/標本である。",
        "",
        "### 力から風速への換算と今回の設定値",
        "",
        "全方式で、推定した符号付き力から同じ二乗抗力式を逆算する。`rho` は空気密度、`Cd` は球のIHB-05方式A等価抗力係数、`A_p` は投影面積である。力に比例して風速が変わるのではなく、力の平方根で換算される。",
        "",
        "$$",
        r"\hat v_k=\mathrm{sgn}(\hat F_k)\sqrt{\frac{2|\hat F_k|}{\rho C_d A_p}},\qquad F=\frac{1}{2}\rho C_dA_pv|v|",
        "$$",
        "",
        "| 方法 | 何を使うか | 今回の設定・注意 |",
        "|---|---|---|",
        "| 静的換算 | 角度と静的な復元力の釣り合い | 履歴・角速度・角加速度を無視 |",
        "| 3段LPF | 静的換算した力の過去値 | 0.5 Hzを3段直列。因果処理なので遅れる |",
        "| 3状態ESO | `theta, omega, F` | `F` を短時間一定と仮定。旧設定45 Hz |",
        "| 4状態ESO | `theta, omega, F, r_F` | `r_F` を追加し、短時間の力の傾きを表す。旧設定20 Hz |",
        "| RTS 3状態 | `theta, omega, F` | EKF＋後向き平滑化。30 N/標本の力雑音仮定。オフライン |",
        "| RTS 4状態 | `theta, omega, F, r_F` | EKF＋後向き平滑化。10 N/s/標本の力変化率雑音仮定。オフライン |",
        "",
        "$$",
        r"\epsilon=0.5\ ^\circ/\mathrm{s}\text{相当の角速度をラジアン毎秒で表した値},\qquad \Delta t=0.01\ \mathrm{s},\qquad f_c=0.5\ \mathrm{Hz}",
        "$$",
        "",
        "各軸で風速RMSEが最小のオンライン方式は、次のとおり。",
        "",
    ]
    for axis in ("IN", "OUT"):
        online_rows = [row for row in speed_rows if row["axis"] == axis
                       and not row["method"].startswith("RTS")]
        best = min(online_rows, key=lambda row: row["rmse"])
        lines.append(f"- {axis}軸：{best['method']}（RMSE {best['rmse']:.4f} m/s）。")
    lines += [
        "- RTS 3状態は全体で最小RMSEだが、将来データを使うオフライン評価である。",
        "- OUT軸の4状態ESOは旧設定で発散した。通常スケールの比較図には含めず、別図と誤差表に示す。これは旧モデル向け調整値を新モデルに移した場合の不安定性を示す。",
        "",
        "### 風速誤差",
        "",
        "| 軸 | 方法 | RMSE (m/s) | 偏り (m/s) | MAE (m/s) | 95%絶対誤差 (m/s) | 最大絶対誤差 (m/s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in speed_rows:
        lines.append(f"| {row['axis']} | {row['method']} | {row['rmse']:.4f} | {row['bias']:.4f} | {row['mae']:.4f} | {row['p95_absolute_error']:.4f} | {row['maximum_absolute_error']:.4f} |")
    lines += [
        "",
        "### 抗力誤差",
        "",
        "| 軸 | 方法 | RMSE (N) | 偏り (N) | MAE (N) | 95%絶対誤差 (N) | 最大絶対誤差 (N) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in force_rows:
        lines.append(f"| {row['axis']} | {row['method']} | {row['rmse']:.6g} | {row['bias']:.6g} | {row['mae']:.6g} | {row['p95_absolute_error']:.6g} | {row['maximum_absolute_error']:.6g} |")
    lines += [
        "",
        "## 作動角と波形",
        "",
        f"| 軸 | 評価区間の最大絶対角度 | ±{config['mechanical_angle_limit_deg']:.0f}°以内か |",
        "|---|---:|---|",
    ]
    for axis, value in angle_max_deg.items():
        lines.append(f"| {axis} | {value:.2f}° | {'はい' if value <= config['mechanical_angle_limit_deg'] else 'いいえ'} |")
    lines += [
        "",
        "![各軸の入力風速・推定風速と推定誤差](ow02_wind_estimates.png)",
        "",
        "![OUT軸4状態ESOの発散](ow02_out_eso4_instability.png)",
        "",
        "## 解釈と次段階",
        "",
        "この再計算で、プラントと推定側の運動モデルに採用済みのI、K、c、tauを反映し、従来候補を比較できる計算経路を用意した。旧モデル向け調整値を固定しているため、ESO/RTS間の優劣は暫定であり、新モデルに対する最適値を示していない。RTSは未来データを使うので、RMSEが小さくてもオンライン方式と同等の実装候補ではない。",
        "",
        f"平均2.0 m/s・最大3.5 m/sの基準入力は、通常域の比較用として設定した。以前の平均3.75 m/s・最大6 m/sの風系列では、評価区間の最大絶対角度はIN {previous_case_check['IN']:.2f}°、OUT {previous_case_check['OUT']:.2f}°だった。どちらも実機リミット±{config['mechanical_angle_limit_deg']:.0f}°以内だが、IN軸は上限まで約{config['mechanical_angle_limit_deg']-previous_case_check['IN']:.2f}°しか余裕がなく、リミット近傍の条件である。この単一風系列だけでは他の乱流やガストでも超過しないとは言えないため、OW-03では複数入力を評価し、リミット超過時の扱いを明記する。",
        "",
        "OW-03では調整用と評価用の入力風を分離し、候補ごとに新モデル上で調整をやり直す。OW-04では推定モデルだけのI、K、tau、cをずらし、係数誤差の影響を調べる。",
        "",
        "## 再実行",
        "",
        "リポジトリのルートで次を実行する。設定と係数台帳は別々のJSONに保存してある。",
        "",
        "```bash",
        "python 06_Analysis/simulation/src/run_ow02_hbk_observer_rerun.py",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path, output_dir: Path) -> dict:
    """Run both axes and write reproducible numeric, image and text outputs."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    time, speed, true_force, wind_stats = make_wind(config)
    sample_period = 1.0 / config["sample_rate_hz"]
    evaluation_mask = ((time >= config["evaluation_start_s"])
                       & (time <= config["evaluation_end_s"]))
    estimates_by_axis: dict[str, dict[str, np.ndarray]] = {}
    force_rows: list[dict] = []
    speed_rows: list[dict] = []
    angle_max: dict[str, float] = {}

    # Recreate the former 3.75 m/s mean, 6 m/s maximum wind input only to
    # compare this historical wind case with the device's configured angle limit.
    old_case = config["previous_wind_case_probe"]
    _, old_speed, old_wind_stats = synthesize_kaimal_wind(
        config["sample_rate_hz"], config["duration_s"],
        old_case["mean_wind_speed_m_s"], old_case["target_turbulence_intensity"],
        config["kaimal_integral_scale_m"], old_case["seed"],
        old_case["maximum_wind_speed_m_s"],
    )
    old_force = drag_force_from_speed(
        old_speed, config["air_density_kg_m3"], config["drag_coefficient"],
        config["projected_area_m2"],
    )
    previous_case_check = {}
    for axis in ("IN", "OUT"):
        old_state = simulate_hbk_plant(
            old_force, select_coefficients(axis, config["axis_configuration"]),
            sample_period, force_lever_m=config["force_lever_m"],
            friction_epsilon_deg_s=config["friction_epsilon_deg_s"],
        )
        previous_case_check[axis] = float(np.max(
            np.abs(np.rad2deg(old_state[evaluation_mask, 0]))
        ))

    for axis in ("IN", "OUT"):
        coefficients = select_coefficients(axis, config["axis_configuration"])
        # Plant is noise-free; this is an ideal-sensor model-match baseline.
        state = simulate_hbk_plant(
            true_force, coefficients, sample_period,
            force_lever_m=config["force_lever_m"],
            friction_epsilon_deg_s=config["friction_epsilon_deg_s"],
        )
        angle = state[:, 0]
        angle_max[axis] = float(np.max(np.abs(np.rad2deg(angle[evaluation_mask]))))
        force_estimates = estimate_all(angle, coefficients, config, sample_period)
        estimates_by_axis[axis] = force_estimates
        for method, estimate in force_estimates.items():
            force_rows.append({"axis": axis, "method": method,
                               **metrics(true_force[evaluation_mask], estimate[evaluation_mask])})
            estimate_speed = wind_speed_from_force(
                estimate, config["air_density_kg_m3"], config["drag_coefficient"],
                config["projected_area_m2"],
            )
            speed_rows.append({"axis": axis, "method": method,
                               **metrics(speed[evaluation_mask], estimate_speed[evaluation_mask])})

    # Record the actual correction vectors, not only the bandwidth settings.
    # Each vector is calculated from the adopted axis coefficients and the
    # legacy pole-frequency setting so future tuning results can be compared.
    tuning = config["observer_settings_from_previous_wind_study"]
    epsilon_rad_s = np.deg2rad(config["friction_epsilon_deg_s"])
    eso_gain_vectors = {}
    for axis in ("IN", "OUT"):
        coefficients = select_coefficients(axis, config["axis_configuration"])
        eso_gain_vectors[axis] = {}
        for method, pole_key, order in (("3状態", "eso3_pole_hz", 0),
                                        ("4状態", "eso4_pole_hz", 1)):
            gain = calculate_luenberger_gain(
                coefficients, config["force_lever_m"], sample_period,
                tuning[pole_key], order, epsilon_rad_s,
            )
            eso_gain_vectors[axis][method] = gain.reshape(-1).tolist()

    save_comparison_figure(output_dir / "ow02_wind_estimates.png", time, speed,
                           estimates_by_axis, config, evaluation_mask)
    save_divergence_figure(
        output_dir / "ow02_out_eso4_instability.png", time, speed,
        estimates_by_axis["OUT"]["ESO 4状態"], config, evaluation_mask,
    )
    write_csv(output_dir / "ow02_force_metrics.csv", force_rows)
    write_csv(output_dir / "ow02_wind_metrics.csv", speed_rows)
    summary = {
        "task_id": "OW-02",
        "coefficient_match": True,
        "sensor_noise_added": False,
        "plant_equation": "I*theta_ddot + K*sin(theta) + c*abs(omega)*omega + tau*tanh(omega/epsilon) = lever*F*cos(theta); b=0",
        "plant_model": "RK4 nonlinear integration",
        "estimator_model": "nonlinear Luenberger/ESO and extended Kalman RTS; adopted I, K, c and tau included",
        "wind_metadata": wind_stats,
        "previous_wind_case_safety_check": {
            "input": old_case,
            "realized_wind": old_wind_stats,
            "maximum_abs_angle_deg": previous_case_check,
        },
        "evaluation_window_s": [config["evaluation_start_s"], config["evaluation_end_s"]],
        "maximum_abs_angle_deg": angle_max,
        "mechanical_angle_limit_deg": config["mechanical_angle_limit_deg"],
        "previous_model_tunings_reused_without_retuning": config["observer_settings_from_previous_wind_study"],
        "eso_gain_vectors_recomputed_for_adopted_coefficients": eso_gain_vectors,
        "force_metrics": force_rows,
        "wind_speed_metrics": speed_rows,
        "interpretation": "Regression-style rerun with prior observer tuning values held fixed. Use OW-03 for separate-input retuning and broader wind scenarios.",
    }
    (output_dir / "ow02_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "OW-02_REPORT.md", config, wind_stats,
                 angle_max, force_rows, speed_rows, previous_case_check,
                 eso_gain_vectors)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OW-02 adopted-HBK observer rerun.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.config, args.output_dir)
    print(json.dumps({
        "result_dir": str(args.output_dir),
        "maximum_abs_angle_deg": summary["maximum_abs_angle_deg"],
        "wind_speed_metrics": summary["wind_speed_metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
