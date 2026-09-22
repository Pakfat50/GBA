# 真鍮スペーサ自由減衰・係数同定パイプライン

`04_Data/05_Fitting/YYYYMMDD/` に保存した角度ログを使い、次を自動実行する。

1. 1ファイルに連続記録した自由減衰を自動分割
2. 各波形へ `K/I`、`b/I`、`c/I`、`tau_f/I` をフィット
3. 既知の追加質量・慣性を使い、複数条件の最小二乗法から絶対係数を同定
4. 自由減衰時の外力推定と、Kaimal想定風に対する真値・推定値を比較

全波形同時フィットとは独立に、頂点周期と陽なエネルギー損失式で係数を求め、
固定係数による元波形再現性を評価する比較スクリプトも用意している。

周期から固定する `I`, `K`、物理的に共有する `c`、厳密な半周期数値積分を組み合わせる
次段階の同定仕様と段階的コミット計画は
[`HYBRID_IDENTIFICATION_PLAN.md`](HYBRID_IDENTIFICATION_PLAN.md) を参照する。

ハイブリッド同定Stage 1として、物理係数をフィットせずに頂点、平衡中心、初期状態を
決定する場合は次を実行する。

```bash
python 06_Analysis/fitting_pipeline/run_hybrid_preprocessing.py --date 20260921
```

結果は `results/YYYYMMDD/hybrid_identification/01_preprocessing/` に保存する。
平衡中心は正負頂点包絡線の中点の算術平均から求め、全点平均と従来の終端中央値も比較用に
保存する。解放後の最初の半周期を除外し、最初の折返し頂点を時間原点、実測頂点角度を
初期角度、初期速度を0として固定する。

Stage 1で確定した頂点から、有限振幅を補正した同符号頂点間周期を用いて
`I`、`K` を決定するStage 2は次のように実行する。

```bash
python 06_Analysis/fitting_pipeline/run_hybrid_frequency_identification.py --date 20260921
```

結果は `results/YYYYMMDD/hybrid_identification/02_frequency_identification/` に保存する。
波形ごとの `K/I` は周期サンプルの中央値、形態ごとの `K/I` は波形代表値の
算術平均とする。SP00～SP04の既知慣性・復元力増分から軸別の基準 `I0`、`K0` を
同定し、BALLは重力復元力と実測 `K/I` から実効慣性を求める。減衰係数はこの段階では
同定しない。

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

20260921試験では、支点からロッド端面まで229 mmとして次の位置を使う。

- スペーサ位置の指定値はロッド端面側の端点である。長さ5 mmを考慮し、
  各重心位置を支点から221.5、216.5、211.5、206.5 mmとする。
- 球の支柱側表面は支点から224 mm、球直径は100 mmであるため、
  球中心と風力作用点は支点から174 mmとする。
- 球側は復元力を弱めるため、符号付き重心位置は負とする。

## 実行方法

リポジトリのトップディレクトリで実行する。

連続ログ、途中の再試験、3回未満の条件がある場合は、最初に確認用スクリプトを
実行する。

```bash
python 06_Analysis/fitting_pipeline/review_waveforms.py --date 20260921
```

この処理は検出した全候補を `waveform_review_all.png` と候補別PNGへ出力し、
`waveform_selection.csv` を作る。接触やリミット衝突を含む波形も消さず、
`suggested_use=0` と品質理由を残す。図を確認した後、各行の
`use_for_fitting` に1（採用）または0（除外）を入力し、確認済みの行を
`review_status=APPROVED` とする。係数同定はこの確認後に行う。

従来形式のログを直接解析する場合は次を実行する。

```bash
python 06_Analysis/fitting_pipeline/run_calibration.py
```

スクリプトが `04_Data/05_Fitting/` 内の8桁日付フォルダを表示するので、番号を入力する。自動実行では日付を直接指定できる。

```bash
python 06_Analysis/fitting_pipeline/run_calibration.py --date 20260916
```

周期から `I`, `K`、前の頂点振幅から明示した半周期エネルギー損失から
`b`, `c`, `tau_f` を同定する場合は、確認済み波形表を作成した後に次を実行する。

```bash
python 06_Analysis/fitting_pipeline/run_explicit_energy_identification.py --date 20260921
```

結果は `results/YYYYMMDD/explicit_energy/` に保存する。主な出力は次のとおり。

- `frequency_segments.csv`: 波形別の有限振幅補正済み `K/I`
- `frequency_levels.csv`: 形態別 `K/I` と既知増分による較正結果
- `energy_intervals.csv`: 頂点間エネルギー損失の実測値、予測値、残差
- `identified_parameters.csv`: 周期・陽エネルギー法による絶対係数
- `waveform_metrics.csv`: 固定係数による元波形と頂点包絡線の偏差
- `previous_representative_waveform_validation.png`: 従来法の形態別代表係数による全波形再現
- `representative_method_comparison.csv`: 今回代表係数、従来代表係数、従来個別フィットの比較
- `FITTING_REPORT.md`: 方法、係数、従来法との比較

結果は `06_Analysis/fitting_pipeline/results/YYYYMMDD/` に保存する。

## デモの実行

既存の実測自由振動を元に作成したデモ入力を
`04_Data/05_Fitting/20990101/` に収録している。`20990101` はデモ用の
予約日付であり、実際の試験日ではない。

```bash
python 06_Analysis/fitting_pipeline/run_calibration.py --date 20990101
```

生成方法と確認済み結果は
[`demo/README.md`](demo/README.md) と
[`results/20990101/DEMO_REPORT.md`](results/20990101/DEMO_REPORT.md) を参照する。

## プロット設定

`run_calibration.py` 冒頭の次の値を `True` / `False` で切り替える。

- `PLOT_WAVEFORM_FITS`
- `PLOT_CALIBRATION_FIT`
- `PLOT_FREE_DECAY_FORCE`
- `PLOT_SIMULATED_WIND_FORCE`
- `PLOT_ESO_ESTIMATE`
- `PLOT_RTS_ESTIMATE`

CSV結果はプロット設定にかかわらず保存する。

自由減衰の外力グラフでは、オブザーバー初期化過渡を除くため、既定で最初の1秒を表示・評価から除く。全サンプルは `free_decay_force.csv` に残る。除外時間は `FREE_DECAY_FORCE_PLOT_WARMUP_S` で変更できる。

RTSの外力変化幅は、自由減衰と0 N周辺の残差を確認する場合は小さく、
変動風を追従する場合は大きく設定している。それぞれ
`FREE_DECAY_ESTIMATOR_SETTINGS` と `WIND_ESTIMATOR_SETTINGS` で変更できる。
使用した値は結果フォルダの `provenance.json` に保存する。
自由減衰の既定値はESO 1 Hz、RTS外力random walk `1e-7 N/sample`であり、
外力変化への追従性よりも実測角度ノイズの抑制を優先する。

## 自動分割の記録

自動分割結果は `segments.csv` に保存する。元の角度CSVは変更しない。分割が不正な場合は、スクリプト冒頭の分割しきい値を調整する。

静止終了を検出できなかったパートは `valid=0` として `segments.csv` に残すが、係数フィットには使用しない。次の試験準備動作を自由減衰として誤ってフィットすることを防ぐためである。

従来の自動フィットは、各入力ファイルに次の順序で6回の自由減衰が入ることを
前提としている。

1. +45 deg 反復1
2. -45 deg 反復1
3. +45 deg 反復2
4. -45 deg 反復2
5. +45 deg 反復3
6. -45 deg 反復3

実際の初期角度は固定せず、ログから読み取って波形ごとにフィットする。
連続取得形式ではこの順序を仮定せず、`waveform_selection.csv` の一行を
一つの入力波形として扱う。

## モデル上の注意

自由減衰モデルは次式である。

```math
I\ddot\theta+b\dot\theta+K\sin\theta+c|\dot\theta|\dot\theta
+\tau_f\tanh(\dot\theta/\varepsilon)=0
```

想定風シミュレーションでは、周囲風による水平力と球自身の運動による二乗減衰を分離した準定常モデルを使う。実機では相対風速モデルによる追加検証が必要である。

## 主な出力ファイル

| ファイル | 内容 |
|---|---|
| `segments.csv` | 自動分割範囲と有効判定 |
| `segment_fits.csv` | 波形ごとのフィッティング結果（RMSE、R、R²を含む） |
| `waveform_fits_all.png` | 全採用波形の実測値と破線フィット、RMSE、R、R²の一覧 |
| `waveform_fits_individual.zip` | 波形ごとの実測値・フィット図 |
| `calibration_levels.csv` | スペーサ条件ごとの集計値 |
| `base_parameters.csv` | ダミーウェイトなしの `I`, `K` と較正RMSE、R、R² |
| `calibration_fit.png` | `I`, `K` の最小二乗較正曲線、残差、R、R² |
| `identified_parameters.csv` | 各形態の `I`, `b`, `K`, `c`, `tau_f` |
| `free_decay_force.csv` | 自由減衰の真値0 NとESO・RTS推定値 |
| `simulated_wind_force.csv` | 想定風外力の真値とESO・RTS推定値 |
| `*_metrics.csv` | 外力推定のRMSE、バイアス、最大誤差 |
| `provenance.json` | 入力ハッシュ、解析設定、同定係数 |
