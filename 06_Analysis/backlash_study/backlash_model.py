"""摩擦とバックラッシュを含む1自由度自由振動モデル。

バックラッシュは、入力角度とばね側角度の差が半幅以内ではばね側角度が
動かず、境界へ達すると追従するrate-independent playモデルで表す。
"""

import numpy as np


def update_play_state(input_angle_rad, previous_output_rad, half_width_rad):
    """play演算子を1サンプル更新する。"""

    if half_width_rad < 0.0:
        raise ValueError("Backlash half width must be non-negative")
    lower = input_angle_rad - half_width_rad
    upper = input_angle_rad + half_width_rad
    return float(np.clip(previous_output_rad, lower, upper))


def initial_play_state(angle_rad, half_width_rad):
    """中心から初期角度へ動かして保持した状態を初期値とする。"""

    if angle_rad > 0.0:
        return angle_rad - half_width_rad
    if angle_rad < 0.0:
        return angle_rad + half_width_rad
    return 0.0


def simulate_decay(time_s, parameters, epsilon_rad_s):
    """摩擦・バックラッシュを含む自由振動を固定時刻で計算する。

    parametersは次の順序とする。
    [K/I, b/I, tau_f/I, backlash_half_width, offset, theta0, omega0]
    """

    time_s = np.asarray(time_s, dtype=float)
    if len(time_s) < 2 or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time_s must be strictly increasing")

    k_over_i = float(parameters[0])
    b_over_i = float(parameters[1])
    friction_over_i = float(parameters[2])
    half_width = float(parameters[3])
    offset = float(parameters[4])
    angle = float(parameters[5])
    speed = float(parameters[6])
    play_state = initial_play_state(angle, half_width)

    output = np.zeros(len(time_s), dtype=float)
    play_output = np.zeros(len(time_s), dtype=float)
    output[0] = angle + offset
    play_output[0] = play_state

    def acceleration(local_angle, local_speed, local_play):
        value = -k_over_i * np.sin(local_play)
        value -= b_over_i * local_speed
        value -= friction_over_i * np.tanh(local_speed / epsilon_rad_s)
        return value

    index = 0
    while index < len(time_s) - 1:
        dt = float(time_s[index + 1] - time_s[index])

        # RK4の各中間角度でもplay状態を更新する。履歴は時間を逆行しない。
        play1 = update_play_state(angle, play_state, half_width)
        angle_k1 = speed
        speed_k1 = acceleration(angle, speed, play1)

        angle2 = angle + 0.5 * dt * angle_k1
        speed2 = speed + 0.5 * dt * speed_k1
        play2 = update_play_state(angle2, play1, half_width)
        angle_k2 = speed2
        speed_k2 = acceleration(angle2, speed2, play2)

        angle3 = angle + 0.5 * dt * angle_k2
        speed3 = speed + 0.5 * dt * speed_k2
        play3 = update_play_state(angle3, play2, half_width)
        angle_k3 = speed3
        speed_k3 = acceleration(angle3, speed3, play3)

        angle4 = angle + dt * angle_k3
        speed4 = speed + dt * speed_k3
        play4 = update_play_state(angle4, play3, half_width)
        angle_k4 = speed4
        speed_k4 = acceleration(angle4, speed4, play4)

        angle += dt * (angle_k1 + 2.0 * angle_k2 + 2.0 * angle_k3 + angle_k4) / 6.0
        speed += dt * (speed_k1 + 2.0 * speed_k2 + 2.0 * speed_k3 + speed_k4) / 6.0
        play_state = update_play_state(angle, play4, half_width)

        output[index + 1] = angle + offset
        play_output[index + 1] = play_state
        index += 1

    return output, play_output


def information_criteria(residual_rad, parameter_count):
    """同じサンプル列でモデルを比較するAICcとBICを返す。"""

    residual = np.asarray(residual_rad, dtype=float)
    sample_count = len(residual)
    squared_error = max(float(np.sum(residual * residual)), 1e-30)
    common = sample_count * np.log(squared_error / sample_count)
    aic = common + 2.0 * parameter_count
    correction = 0.0
    if sample_count > parameter_count + 1:
        correction = 2.0 * parameter_count * (parameter_count + 1)
        correction /= sample_count - parameter_count - 1
    aicc = aic + correction
    bic = common + parameter_count * np.log(sample_count)
    return float(aicc), float(bic)

