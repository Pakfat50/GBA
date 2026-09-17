# 真鍮スペーサ自由減衰・係数同定パイプライン

`04_Data/05_Fitting/YYYYMMDD/` に保存した角度ログを使い、次を自動実行する。

1. 1ファイルに連続記録した自由減衰を自動分割
2. 各波形へ `K/I`、`b/I`、`c/I`、`tau_f/I` をフィット
3. 既知の追加質量・慣性を使い、複数条件の最小二乗法から絶対係数を同定
4. 自由減衰時の外力推定と、Kaimal想定風に対する真値・推定値を比較

## 入力フォルダ

```text
04_Data/05_Fitting/
└── YYYYMMDD/
    ├── LOG00001_ANGLE.csv
    ├── LOG00002_ANGLE.csv
    └── test_manifest.csv
```

角度ログは `05_Script/02_LoggerDecoder/LOG00008_ANGLE.csv` と同じ形式を使う。

- 時刻：`systime[ms]`
- 角度：`angle0[deg]` または `angle1[deg]`
- 角度はロッド鉛直を0 degとする

`test_manifest.csv` は
[`templates/test_manifest_template.csv`](templates/test_manifest_template.csv)
を試験日フォルダへコピーし、実際のファイル名と測定値に書き換える。

## 符号付き重心位置

`signed_com_radius_m` は支点から部品重心までの距離である。

- 部品を追加すると復元力が強くなる側：正
- ロッド上端や球側など、復元力が弱くなる側：負

追加部品による係数変化は次式で計算する。

```math
\Delta I=J_c+mr^2,\qquad \Delta K=mgr
```

`component_centroid_inertia_kg_m2` は、部品群の合成重心を通り、対象の振動軸と平行な軸まわりの慣性である。SP00はすべて0とする。

## 実行方法

リポジトリのトップディレクトリで実行する。

```bash
python 06_Analysis/fitting_pipeline/run_calibration.py
```

スクリプトが `04_Data/05_Fitting/` 内の8桁日付フォルダを表示するので、番号を入力する。自動実行では日付を直接指定できる。

```bash
python 06_Analysis/fitting_pipeline/run_calibration.py --date 20260916
```

結果は `06_Analysis/fitting_pipeline/results/YYYYMMDD/` に保存する。

## プロット設定

`run_calibration.py` 冒頭の次の値を `True` / `False` で切り替える。

- `PLOT_WAVEFORM_FITS`
- `PLOT_FREE_DECAY_FORCE`
- `PLOT_SIMULATED_WIND_FORCE`
- `PLOT_ESO_ESTIMATE`
- `PLOT_RTS_ESTIMATE`

CSV結果はプロット設定にかかわらず保存する。

## 自動分割の記録

自動分割結果は `segments.csv` に保存する。元の角度CSVは変更しない。分割が不正な場合は、スクリプト冒頭の分割しきい値を調整する。

各入力ファイルには、次の順序で6回の自由減衰が入っていることを前提とする。

1. +45 deg 反復1
2. -45 deg 反復1
3. +45 deg 反復2
4. -45 deg 反復2
5. +45 deg 反復3
6. -45 deg 反復3

実際の初期角度は固定せず、ログから読み取って波形ごとにフィットする。

## モデル上の注意

自由減衰モデルは次式である。

```math
I\ddot\theta+b\dot\theta+K\sin\theta+c|\dot\theta|\dot\theta
+\tau_f\tanh(\dot\theta/\varepsilon)=0
```

想定風シミュレーションでは、周囲風による水平力と球自身の運動による二乗減衰を分離した準定常モデルを使う。実機では相対風速モデルによる追加検証が必要である。
