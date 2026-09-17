# 既存自由振動ベースのデモデータ

`build_demo_data.py` は、旧ハードの実測自由振動
`04_Data/00_Calibration/swing/LOG00014.TXT` を元に、較正パイプライン用の
デモ入力を作る。

これは実測そのものではない。実測4区間から得た代表減衰係数と角度ノイズを使い、
以下の試験形式に再構成したデモデータである。

- SP00、SP01、SP02、BALLの4形態
- 各形態を1ファイルに保存
- +45 degと3回、-45 degと3回
- 1回の減衰を30秒記録し、低速域まで含める
- LoggerDecoderの `LOG00008_ANGLE.csv` と同じ列構成
- 実測ログから得た微小な角度ばらつきを付加

## 再生成

リポジトリのトップで実行する。

```bash
python 06_Analysis/fitting_pipeline/demo/build_demo_data.py
```

出力先は `04_Data/05_Fitting/20990101/` である。`20990101` はデモ用の予約日付で、
実際の試験日を表さない。

## デモ解析

```bash
python 06_Analysis/fitting_pipeline/run_calibration.py --date 20990101
```

主な確認点は次のとおり。

1. `segments.csv` に24本の波形が記録される。
2. `segment_fits.csv` の24本がフィット成功となる。
3. `identified_parameters.csv` に `I`, `b`, `K`, `c`, `tau_f` が出力される。
4. 波形フィット、自由振動外力、想定風外力のPNGが出力される。

`demo_truth.csv` はデモ作成時の既知係数であり、同定結果の照合用に使う。
`demo_provenance.json` に元ログのSHA-256、切り出し区間、乱数シードを保存する。
