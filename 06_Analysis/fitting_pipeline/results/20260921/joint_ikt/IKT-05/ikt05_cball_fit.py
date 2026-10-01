#!/usr/bin/env python3
"""IKT-05: 固定済みの球ありI・Kと球なしτを使い、球抗力係数を同定する。

このスクリプトは既存の Stage 5 解析を呼び出します。元の入力CSVや設定は
書き換えず、一時コピー上でτだけを今回採用するIHB-03値に差し替えます。
計算結果は --output-dir で指定したフォルダーへ保存します。

実行例（リポジトリのルートで）:
  python 06_Analysis/fitting_pipeline/results/20260921/joint_ikt/IKT-05/ikt05_cball_fit.py \
    --result-root 06_Analysis/fitting_pipeline/results \
    --date 20260921 \
    --output-dir 06_Analysis/fitting_pipeline/results/20260921/joint_ikt/IKT-05/output

既定のτはIHB-03の「半周期ODE次頂点角フィット」で選んだ値です。
IHB-03の異なる目的関数（エネルギー損失に直接フィット）の値を試す場合は、
--tau-in と --tau-out で値を明示できます。
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

# 既存Stage 5スクリプトがあるfitting_pipelineをimport対象へ追加します。
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# 同じ解析フォルダーにある、承認波形の選択・エネルギー法・ODE検証を行う既存コード。
import run_hybrid_sphere_drag_identification as stage5


def main():
    parser = argparse.ArgumentParser(description="球あり自由振動データから c_ball を同定します")
    parser.add_argument("--result-root", type=Path, required=True,
                        help="前処理結果が入っている results フォルダー")
    parser.add_argument("--date", default="20260921", help="解析データの日付フォルダー")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="IKT-05のCSV・設定ファイルを書き出す場所")
    parser.add_argument("--tau-in", type=float, default=8.8238e-5,
                        help="IN軸の固定τ [N m]（既定値はIHB-03 ODE頂点法）")
    parser.add_argument("--tau-out", type=float, default=1.5633e-4,
                        help="OUT軸の固定τ [N m]（既定値はIHB-03 ODE頂点法）")
    args = parser.parse_args()

    source_date = args.result_root / args.date
    if not source_date.is_dir():
        parser.error(f"日付フォルダーが見つかりません: {source_date}")

    # 元データを変更しないため、Stage 5が必要とする結果一式を一時領域に複製します。
    with tempfile.TemporaryDirectory(prefix="ikt05_cball_") as temporary:
        temporary_root = Path(temporary) / "results"
        temporary_date = temporary_root / args.date
        temporary_date.mkdir(parents=True)
        for folder in ("waveform_review", "hybrid_identification"):
            source = source_date / folder
            if source.exists():
                shutil.copytree(source, temporary_date / folder)

        # 一時コピー内のτのみを更新。I、K、ロッド抗力、採否表はそのまま固定します。
        settings_path = temporary_date / "hybrid_identification" / "04_rod_damping_identification" / "stage4_settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        selected = settings["models_for_review"]["theoretical_c"]
        selected["tau_IN"] = args.tau_in
        selected["tau_OUT"] = args.tau_out
        settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        # 既存のStage 5計算を再利用し、球cの推定・波形クラスタ区間推定・ODE検証を行います。
        old_argv = sys.argv[:]
        try:
            sys.argv = ["run_hybrid_sphere_drag_identification.py", "--date", args.date,
                        "--result-root", str(temporary_root)]
            stage5.main()
        finally:
            sys.argv = old_argv

        generated = temporary_date / "hybrid_identification" / "05_sphere_drag_identification"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for file in generated.iterdir():
            if file.is_file():
                shutil.copy2(file, args.output_dir / file.name)

    print(f"IKT-05の出力先: {args.output_dir}")


if __name__ == "__main__":
    main()
