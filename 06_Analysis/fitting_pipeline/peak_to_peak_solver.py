"""ハイブリッド同定用の頂点間非線形ソルバー。

実測頂点 ``(theta, theta_dot) = (A_n, 0)`` を初期状態とし、次の
速度ゼロ交差まで運動方程式を積分する。散逸仕事も状態として同時積分し、
力学的エネルギー差との数値的な閉合を検証できるようにする。
"""

import math

import numpy as np
from scipy.integrate import solve_ivp


# Stage 3の全360条件で厳密基準解と比較し、次頂点角0.01 deg、半周期0.1 ms、
# 正規化エネルギー閉合1e-4の受入基準を満たす最も粗い検証済み設定。
DEFAULT_RTOL = 1.0e-4
DEFAULT_ANGLE_SPEED_ATOL = 1.0e-7
DEFAULT_ENERGY_ATOL = 1.0e-11
DEFAULT_MAX_STEP_FRACTION = 1.0 / 5.0
DEFAULT_MAX_PERIODS = 2.0


def _validate_inputs(
    initial_angle_rad,
    inertia_kg_m2,
    restoring_n_m_per_rad,
    damping_n_m_s_per_rad,
    quadratic_n_m_s2_per_rad2,
    friction_n_m,
    epsilon_rad_s,
):
    values = {
        "initial_angle_rad": initial_angle_rad,
        "inertia_kg_m2": inertia_kg_m2,
        "restoring_n_m_per_rad": restoring_n_m_per_rad,
        "damping_n_m_s_per_rad": damping_n_m_s_per_rad,
        "quadratic_n_m_s2_per_rad2": quadratic_n_m_s2_per_rad2,
        "friction_n_m": friction_n_m,
        "epsilon_rad_s": epsilon_rad_s,
    }
    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(name + " は有限値である必要があります")
    if abs(initial_angle_rad) <= 0.0:
        raise ValueError("initial_angle_rad は0以外である必要があります")
    if abs(initial_angle_rad) >= math.pi:
        raise ValueError("initial_angle_rad は-pi～piの範囲内である必要があります")
    if inertia_kg_m2 <= 0.0:
        raise ValueError("inertia_kg_m2 は正である必要があります")
    if restoring_n_m_per_rad <= 0.0:
        raise ValueError("restoring_n_m_per_rad は正である必要があります")
    if damping_n_m_s_per_rad < 0.0:
        raise ValueError("damping_n_m_s_per_rad は0以上である必要があります")
    if quadratic_n_m_s2_per_rad2 < 0.0:
        raise ValueError("quadratic_n_m_s2_per_rad2 は0以上である必要があります")
    if friction_n_m < 0.0:
        raise ValueError("friction_n_m は0以上である必要があります")
    if epsilon_rad_s <= 0.0:
        raise ValueError("epsilon_rad_s は正である必要があります")


def mechanical_energy_j(angle_rad, speed_rad_s, inertia_kg_m2, restoring_n_m_per_rad):
    """角度、角速度から力学的エネルギーを返す。"""

    kinetic = 0.5 * inertia_kg_m2 * speed_rad_s * speed_rad_s
    potential = restoring_n_m_per_rad * (1.0 - np.cos(angle_rad))
    return float(kinetic + potential)


def conservative_half_period_s(initial_angle_rad, inertia_kg_m2, restoring_n_m_per_rad):
    """保存系非線形振り子の頂点間半周期を楕円積分から返す。"""

    from scipy.special import ellipk

    parameter = np.sin(0.5 * abs(initial_angle_rad)) ** 2
    return float(
        2.0
        * ellipk(parameter)
        * np.sqrt(inertia_kg_m2 / restoring_n_m_per_rad)
    )


def solve_next_turning_point(
    initial_angle_rad,
    inertia_kg_m2,
    restoring_n_m_per_rad,
    damping_n_m_s_per_rad,
    quadratic_n_m_s2_per_rad2,
    friction_n_m,
    epsilon_rad_s,
    *,
    rtol=DEFAULT_RTOL,
    angle_speed_atol=DEFAULT_ANGLE_SPEED_ATOL,
    energy_atol=DEFAULT_ENERGY_ATOL,
    max_step_fraction=DEFAULT_MAX_STEP_FRACTION,
    max_periods=DEFAULT_MAX_PERIODS,
):
    """初期頂点から次の折返し頂点まで数値積分する。

    第3状態は散逸仕事の累積値である。速度イベントの向きを初期角の符号と
    逆向きに指定することで、時刻0の速度ゼロをイベントとして誤検出しない。
    """

    _validate_inputs(
        initial_angle_rad,
        inertia_kg_m2,
        restoring_n_m_per_rad,
        damping_n_m_s_per_rad,
        quadratic_n_m_s2_per_rad2,
        friction_n_m,
        epsilon_rad_s,
    )
    if rtol <= 0.0 or angle_speed_atol <= 0.0 or energy_atol <= 0.0:
        raise ValueError("数値許容誤差は正である必要があります")
    if max_step_fraction <= 0.0 or max_periods <= 0.5:
        raise ValueError("max_step_fractionとmax_periodsの指定が不正です")

    small_angle_period_s = float(
        2.0 * math.pi * math.sqrt(inertia_kg_m2 / restoring_n_m_per_rad)
    )
    maximum_step_s = small_angle_period_s * max_step_fraction
    maximum_time_s = small_angle_period_s * max_periods

    def differential_equation(unused_time, state):
        angle_rad = state[0]
        speed_rad_s = state[1]
        damping_torque = damping_n_m_s_per_rad * speed_rad_s
        quadratic_torque = (
            quadratic_n_m_s2_per_rad2 * abs(speed_rad_s) * speed_rad_s
        )
        friction_torque = friction_n_m * np.tanh(speed_rad_s / epsilon_rad_s)
        acceleration_rad_s2 = -restoring_n_m_per_rad * np.sin(angle_rad)
        acceleration_rad_s2 -= damping_torque
        acceleration_rad_s2 -= quadratic_torque
        acceleration_rad_s2 -= friction_torque
        acceleration_rad_s2 /= inertia_kg_m2
        dissipated_power_w = damping_n_m_s_per_rad * speed_rad_s**2
        dissipated_power_w += quadratic_n_m_s2_per_rad2 * abs(speed_rad_s) ** 3
        dissipated_power_w += friction_torque * speed_rad_s
        return [speed_rad_s, acceleration_rad_s2, dissipated_power_w]

    def next_turning_event(unused_time, state):
        return state[1]

    next_turning_event.terminal = True
    next_turning_event.direction = 1.0 if initial_angle_rad > 0.0 else -1.0

    solution = solve_ivp(
        differential_equation,
        (0.0, maximum_time_s),
        [initial_angle_rad, 0.0, 0.0],
        method="DOP853",
        events=next_turning_event,
        rtol=rtol,
        atol=[angle_speed_atol, angle_speed_atol, energy_atol],
        max_step=maximum_step_s,
        dense_output=False,
    )
    if not solution.success:
        raise RuntimeError("頂点間の数値積分に失敗しました: " + solution.message)
    if len(solution.t_events[0]) != 1:
        raise RuntimeError(
            "次の折返し頂点を制限時間内に一意に検出できませんでした"
        )

    event_time_s = float(solution.t_events[0][0])
    event_state = solution.y_events[0][0]
    next_angle_rad = float(event_state[0])
    next_speed_rad_s = float(event_state[1])
    integrated_loss_j = float(event_state[2])
    initial_energy_j = mechanical_energy_j(
        initial_angle_rad, 0.0, inertia_kg_m2, restoring_n_m_per_rad
    )
    final_energy_j = mechanical_energy_j(
        next_angle_rad,
        next_speed_rad_s,
        inertia_kg_m2,
        restoring_n_m_per_rad,
    )
    mechanical_loss_j = initial_energy_j - final_energy_j
    closure_error_j = mechanical_loss_j - integrated_loss_j

    return {
        "next_angle_rad": next_angle_rad,
        "next_speed_rad_s": next_speed_rad_s,
        "half_period_s": event_time_s,
        "integrated_loss_j": integrated_loss_j,
        "mechanical_loss_j": mechanical_loss_j,
        "energy_closure_error_j": closure_error_j,
        "initial_energy_j": initial_energy_j,
        "final_energy_j": final_energy_j,
        "small_angle_period_s": small_angle_period_s,
        "maximum_step_s": maximum_step_s,
        "function_evaluations": int(solution.nfev),
    }
