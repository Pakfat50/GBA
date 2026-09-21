# 20260921 自由振動フィッティング結果

## 解析対象

- 検出波形：48本
- 採用波形：46本
- 除外波形：`OUT_BALL_P_R01`、`OUT_SP01_N_R03`
- フィット成功：46本
- モデル：`I*theta_ddot + K*sin(theta) + b*theta_dot + c*abs(theta_dot)*theta_dot + tau_f*tanh(theta_dot/epsilon) = 0`

全波形の比較図では実測値を青実線、フィット値を赤破線で示し、各図の表題に
RMSE、相関係数 `R`、決定係数 `R^2` を記載した。

- [全46波形のフィット一覧](waveform_fits_all.png)
- [波形別係数・RMSE・R・R^2](segment_fits.csv)
- [I0・K0の最小二乗較正曲線と残差](calibration_fit.png)
- [形態別の絶対係数](identified_parameters.csv)

## 波形フィットの適合度

| 指標 | RMSE [deg] | R | R^2 |
|---|---:|---:|---:|
| 平均 | 0.818 | 0.99928 | 0.99830 |
| 中央値 | 0.805 | 0.99933 | 0.99847 |
| 最小R/R^2の波形 | 1.907 | 0.99630 | 0.99261 |

最小値は `OUT_SP02_P_R01` であり、全46波形で `R >= 0.99630`、
`R^2 >= 0.99261` となった。局所解となった `IN_SP01_N_R02` は複数初期値で
再探索し、RMSEを3.55 degから0.75 degへ改善した。

`R` は実測角度と予測角度のPearson相関係数、`R^2` は
`1 - sum((theta_fit-theta_meas)^2) / sum((theta_meas-mean(theta_meas))^2)`
である。相関だけでは振幅誤差を見落とすため、RMSEと残差も併記する。

## 取付位置と物性値

- 支点からロッド端面：229 mm
- スペーサ重心：支点から221.5、216.5、211.5、206.5 mm
- スペーサ1個の質量：1.18 g
- 球の支柱側表面：支点から224 mm
- 球中心および風力作用点：支点から174 mm
- 球質量：3.9 g
- 球直径：100 mm

補完後の入力値は `resolved_manifest.csv`、値の出典と仮定は
`physical_input_sources.csv` に保存した。

## 最終係数の求め方

### 1. 波形ごとの正規化係数

運動方程式を `I` で割り、各採用波形に次式を非線形最小二乗フィットした。

```text
theta_ddot
+ qK*sin(theta)
+ qb*theta_dot
+ qc*abs(theta_dot)*theta_dot
+ qtau*tanh(theta_dot/epsilon) = 0

qK   = K/I
qb   = b/I
qc   = c/I
qtau = tau_f/I
```

未知数は `qK, qb, qc, qtau`、角度オフセット、初期角度、初期角速度の7個である。
大振幅点だけが支配しないよう振幅依存の重みを付け、数値積分した予測角度と
実測角度の残差を最小化した。波形別結果は `segment_fits.csv` に保存した。

### 2. 基準ハードウェアのI0とK0

各軸・各スペーサ水準について `qK=K/I` の波形平均を求めた。スペーサ追加量は
次式で計算した。`r` は支点から追加部品重心までの符号付き距離であり、今回の
スペーサは復元力を弱める側なので負である。

```text
delta_I = Jc + m*r^2
delta_K = m*g*r
qK_pred = (K0 + delta_K) / (I0 + delta_I)
```

SP00～SP04の5水準を使い、次の重み付き非線形最小二乗で `I0, K0` を同時同定した。

```text
minimize sum_j ((qK_pred_j - qK_meas_j) / sigma_j)^2
```

`sigma_j` は各水準の標準誤差と平均値の0.1%の大きい方である。初期値は
`qK_j*I0 - K0 = delta_K_j - qK_j*delta_I_j` の線形最小二乗解から与えた。

| 軸 | I0 [kg m^2] | K0 [N m/rad] | RMSE [s^-2] | R | R^2 |
|---|---:|---:|---:|---:|---:|
| IN | 2.46284e-4 | 2.25279e-2 | 0.8584 | 0.999419 | 0.998622 |
| OUT | 2.47213e-4 | 2.16553e-2 | 0.8384 | 0.999419 | 0.998604 |

較正曲線と残差は [calibration_fit.png](calibration_fit.png) に示す。両軸とも相関は
高いが、SP00とSP01に約1～1.5 s^-2の系統的残差があるため、`R` だけでなく残差も
今後のモデル改良判断に使う。

### 3. 球あり状態のIとK

球の既知質量と取付位置から、まず幾何学的な追加量を求める。

```text
K_ball       = K0 + delta_K_ball
I_ball_geom  = I0 + delta_I_ball
I_ball_final = K_ball / mean(qK_ball)
```

最終 `I` は空気付加慣性を含む実効値とするため、球あり波形の `qK=K/I` から
逆算した。したがって `I_ball_final` は幾何学値 `I_ball_geom` より大きい。

| 軸 | I_geom [kg m^2] | I_final [kg m^2] | K_final [N m/rad] | mean(K/I) [s^-2] |
|---|---:|---:|---:|---:|
| IN | 3.68260e-4 | 3.90094e-4 | 1.58709e-2 | 40.68474 |
| OUT | 3.69189e-4 | 3.90417e-4 | 1.49982e-2 | 38.41587 |

### 4. b、c、tau_fの絶対値

球ありの各波形で得た正規化係数を平均し、同じ軸の最終実効慣性を掛けた。

```text
b     = I_ball_final * mean(qb_ball)
c     = I_ball_final * mean(qc_ball)
tau_f = I_ball_final * mean(qtau_ball)
```

この計算は各波形の絶対係数 `I*q` を平均することと同じである。最終値は次のとおり。

| 軸 | I [kg m^2] | K [N m/rad] | b [N m s/rad] | c [N m s^2/rad^2] | tau_f [N m] |
|---|---:|---:|---:|---:|---:|
| IN | 3.90094e-4 | 1.58709e-2 | 1.16e-8 | 1.41e-5 | 9.72e-5 |
| OUT | 3.90417e-4 | 1.49982e-2 | 3.45e-5 | 1.27e-5 | 1.11e-4 |

IN軸では粘性項 `b` がほぼゼロとなり、二乗抵抗項 `c` 側へ減衰が配分された。
自由振動だけでは `b`、`c`、`tau_f` に相関があるため、各項の単独値よりも
合成減衰トルクと波形再現性を優先して評価する。
