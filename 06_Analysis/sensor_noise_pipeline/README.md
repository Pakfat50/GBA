# 角度センサーノイズ解析パイプライン

`04_Data/05_Fitting/YYYYMMDD/` に保存した静止角度ログを使い、次を自動実行する。

1. 1ファイルに連続記録した3回の静止測定を自動分割
2. サンプル周期、ジッタ、欠落周期、14 bit量子化格子との一致を確認
3. 一次ドリフトを除去し、白色ノイズと有色ノイズを推定
4. 自己相関、パワースペクトル密度（PSD）、誤差分布を出力
5. オブザーバーへ転記するセンサーモデル係数をCSVへ保存

## 試験方法

装置を無風環境で、ロッドが鉛直0 degとなるよう剛に固定する。1つのログに次の手順を連続して記録する。

1. 0 degで20秒以上静止する。
2. 目標角度から1 deg以上動かし、再び0 degへ固定する。
3. 1と2を繰り返し、合計3回の静止区間を記録する。

反復間でロガーを停止する必要はない。スクリプトが静止区間を分割する。角度依存性も確認する場合は、同じ手順を `+30 deg`、`-30 deg` などで追加し、対応表の `target_angle_deg` と `condition` を変更する。

静止試験に機構の微小振動が入ると、解析値にはセンサーと機構の両方が含まれる。軸を手で保持せず、治具で固定する。

## 入力フォルダ

```text
04_Data/05_Fitting/
└── YYYYMMDD/
    ├── LOG00001_ANGLE.csv
    └── sensor_noise_manifest.csv
```

角度ログは `05_Script/02_LoggerDecoder/LOG00008_ANGLE.csv` と同じ形式を使う。

- 時刻: `systime[ms]`
- 外軸角度: `angle0[deg]`
- 内軸角度: `angle1[deg]`
- 角度はロッド鉛直を0 degとする

[`templates/sensor_noise_manifest_template.csv`](templates/sensor_noise_manifest_template.csv) を試験日フォルダへコピーし、実際のファイル名へ書き換える。同じログの `angle0[deg]` と `angle1[deg]` は、対応表の2行で指定できる。

### 対応表の列

| 列 | 内容 |
|---|---|
| `data_file` | `LOG00001_ANGLE.csv` 形式のログ名 |
| `axis` | 結果に表示する軸名。`OUT` または `IN` |
| `angle_column` | `angle0[deg]` または `angle1[deg]` |
| `condition` | `FIXED_0` などの試験条件名 |
| `target_angle_deg` | 治具で固定した角度 |
| `expected_repetitions` | 通常は3 |
| `use_start_s`, `use_end_s` | ログの一部だけを使う場合の時刻。全体を使う場合は空欄 |
| `valid` | 解析対象は1、除外は0 |
| `notes` | 試験時のメモ |

## 実行方法

リポジトリのトップディレクトリで実行する。

```bash
python 06_Analysis/sensor_noise_pipeline/run_sensor_noise_analysis.py
```

スクリプトが `04_Data/05_Fitting/` 内の8桁日付フォルダを表示するので、番号を入力する。自動実行では日付を直接指定できる。

```bash
python 06_Analysis/sensor_noise_pipeline/run_sensor_noise_analysis.py --date 20260918
```

結果は `06_Analysis/sensor_noise_pipeline/results/YYYYMMDD/` に保存する。

## プロット設定

`run_sensor_noise_analysis.py` 冒頭の値を `True` / `False` で切り替える。

- `PLOT_TIME_SERIES`: 元波形と自動分割区間
- `PLOT_HISTOGRAM`: 一次ドリフト除去後の誤差分布
- `PLOT_PSD`: パワースペクトル密度
- `PLOT_AUTOCORRELATION`: 自己相関
- `PLOT_SAMPLE_INTERVALS`: サンプル周期の分布

CSV結果はプロット設定にかかわらず保存する。

## 主な出力

| ファイル | 内容 |
|---|---|
| `segment_noise_metrics.csv` | 反復ごとのノイズ・時刻品質・モデル係数 |
| `condition_summary.csv` | 3反復の平均と平均角度の再現性 |
| `recommended_sensor_model.csv` | オブザーバーへ転記する軸別代表値 |
| `psd.csv` | PSDの数値データ |
| `autocorrelation.csv` | 自己相関の数値データ |
| `SENSOR_NOISE_REPORT.md` | 主な結果と注意事項 |
| `provenance.json` | 入力ハッシュと解析設定 |

## 推定できる値とできない値

静止試験から、白色ノイズ、有色ノイズ、有色ノイズ時定数、量子化格子との一致、サンプル周期とジッタを推定する。固定遅延は静止データだけでは推定できない。遅延を更新するには、エンコーダーなどで既知角度変化を同時に記録する動的試験が必要である。

## デモ

`demo/build_demo_data.py` は、MT6701の14 bit量子化と白色・有色ノイズを含む3反復のログを `04_Data/05_Fitting/20990102/` に作る。`20990102` はデモ専用の予約日付である。

```bash
python 06_Analysis/sensor_noise_pipeline/demo/build_demo_data.py
python 06_Analysis/sensor_noise_pipeline/run_sensor_noise_analysis.py --date 20990102
```

