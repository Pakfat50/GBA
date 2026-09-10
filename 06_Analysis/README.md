# GBA 外乱オブザーバー解析

球形受風部を持つ振り子型風速計について、角度の過渡応答から外乱トルク・等価力を推定するPoCです。2025年1月14日の比較試験と既存の自由振動試験を対象に、次をまとめています。

- 1自由度振り子モデルと拡張状態オブザーバー
- 生角度の静的換算、ローパス、オブザーバーの比較
- 参照風速に対するRMSE評価
- 慣性モーメント `I`、復元係数 `K`、減衰係数 `b` の感度解析
- 約38度の機械的制限を考慮した共通除外マスク
- 100 Hzの等価トルク・力・風速推定データの生成
- 同一モデル条件でのStage 1シミュレーションとオブザーバー状態検証

理論、データ解釈、評価結果、制約、今後の試験案は [GBA_PoC_Report.md](GBA_PoC_Report.md) に記載しています。

## ディレクトリ構成

```text
06_Analysis/
├── README.md
├── GBA_PoC_Report.md
├── requirements.txt
├── config/
│   └── settings.json
├── src/
│   ├── prepare_data.py
│   ├── poc.py
│   ├── plot_wide.py
│   └── plot_free_decay_validation.py
├── simulation/
│   ├── README.md
│   ├── Stage1_Report.md
│   ├── config/stage1_nominal.json
│   ├── src/                   # 線形プラント、係数導出、オブザーバー
│   ├── tests/                 # Stage 1自動テスト
│   └── results/stage1/        # 合否JSONと検証図
├── data/
│   └── processed/              # 実行時に生成。Git管理外
└── results/
    ├── *.csv, *.json, *.png    # 評価表と主要図
    ├── wide_plots/             # 4系列の全区間・10分拡大図
    └── free_decay_validation/  # 自由振動による内部整合性確認
```

## 対象データ

| 用途 | リポジトリ内の入力 |
|---|---|
| 午前のフィールド試験 | `05_Script/01_Analysis/20250114/LOG00016.TXT`、`WS400am.txt` |
| 午後のフィールド試験 | `05_Script/01_Analysis/20250114/LOG00017.TXT`、`WS400pm.txt` |
| 内軸の自由振動 | `04_Data/00_Calibration/swing/LOG00014.TXT` |

元ログは既存ディレクトリから読み、`06_Analysis` 内には複製しません。

## 実行方法

リポジトリのトップディレクトリで実行します。Python 3.12で確認しています。

```bash
python -m pip install -r 06_Analysis/requirements.txt
python 06_Analysis/src/prepare_data.py
python 06_Analysis/src/poc.py
python 06_Analysis/src/plot_wide.py
python 06_Analysis/src/plot_free_decay_validation.py
python 06_Analysis/simulation/src/run_stage1.py
python -m unittest discover -s 06_Analysis/simulation/tests -v
```

`prepare_data.py` は中間ファイルを `06_Analysis/data/processed/` に作成します。`poc.py` は集計表・図に加え、次の100 Hzデータを `06_Analysis/results/` に生成します。

- `estimates_100hz_am.csv.gz`、`estimates_100hz_pm.csv.gz`
- `estimates_100hz_pole1hz_am.csv.gz`、`estimates_100hz_pole1hz_pm.csv.gz`

これらに加え、サンプル・評価ビンごとの詳細CSVは原ログから再生成可能で容量も大きいため、Git管理外です。集計値、条件、図はGit管理します。

## 主な結果

午前データで共通倍率・時刻補正・極を決め、午後へ固定適用した0.25秒平均の比較です。

| 方式 | 午前RMSE [m/s] | 午後RMSE [m/s] |
|---|---:|---:|
| 生角度の静的換算 | 0.327832 | 0.250151 |
| ローパス | 0.308690 | 0.230565 |
| オブザーバー | 0.306871 | 0.228622 |

誤差最小の設定ではオブザーバー固有の改善は小さく、平滑化の寄与が支配的です。極の周波数パラメータを1 Hzにして変動をより残す設定では、午後RMSEはローパス0.247048 m/s、オブザーバー0.237719 m/sでした。

自由振動4区間では、符号付き等価力のゼロからのRMSが静的換算8.631 mN、ローパス7.045 mN、オブザーバー0.690 mNでした。この4区間はモデル同定にも用いているため、独立検証ではなく内部整合性の確認です。

## 図の見方

- 青：生角度から静的換算した値
- 紫：ローパス
- 緑：オブザーバー
- 赤：参照風速
- 灰色：飽和近傍、操作、記録端などの除外区間

主要図：

- [全記録の比較](results/overview.png)
- [午後10分間の4系列比較](results/wide_plots/GBA_FourSeries_PM_10min.png)
- [午前全記録の4系列比較](results/wide_plots/GBA_FourSeries_AM_Full.png)
- [午後全記録の4系列比較](results/wide_plots/GBA_FourSeries_PM_Full.png)
- [自由振動1区間の比較](results/free_decay_validation/GBA_FreeDecay_Comparison.png)
- [自由振動4区間の比較](results/free_decay_validation/GBA_FreeDecay_AllFour.png)
- [モデル誤差の理論応答](results/theory_frequency_response.png)

## 設定

`config/settings.json` で極の探索範囲、飽和判定、前後の除外時間、手動除外区間、評価ビン幅を設定します。手動除外はJSTの午前0時からの秒数で記述します。

```json
"manual_exclusions_jst_s": {
  "am": [[36000, 36020]],
  "pm": []
}
```

100 Hzは出力間隔です。参照器は約4 Hzであり、このデータだけでは100 Hzの風変動の再現精度を検証できません。

## シミュレーションPoC

[Stage 1シミュレーション](simulation/README.md)では、プラントとオブザーバーに同じ1軸線形モデルを使用し、理想角度観測下で状態一致と収束を検証します。結果と合否判定は[Stage 1検証結果](simulation/Stage1_Report.md)に記載しています。

[Stage 2：DOEによる係数誤差評価](simulation/Stage2_Report.md)では、同一入力に対する静的換算・LPF・オブザーバーの比較と、5因子の主効果・交互作用・応答曲面の検証を行っています。

[Stage 2：オブザーバー実装比較](simulation/Estimator_Comparison_Report.md)では、角度だけを観測する現行ESO、Ramp ESO、因果Kalman filterと、未来の角度も使うRTS smootherを公称・ノイズなし条件で比較しています。
