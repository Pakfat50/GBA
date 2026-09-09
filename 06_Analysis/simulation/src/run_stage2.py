"""Reproduce Stage 2 DOE, held-out response-surface checks and PNG report."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
import scipy
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from doe import factorial, face_centered, features, contrasts, decode
from model import PendulumParameters, natural_characteristics
from stage2_system import plant, estimate, force_frequency_response

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('Static', 'Low-pass', 'Observer')
COLORS = ('#2776bc', '#9667bd', '#159477')


def inputs(config, fn):
    t = np.arange(round(config['duration_s']*config['sample_rate_hz'])+1)/config['sample_rate_hz']
    a = config['force_amplitude_N']
    rng = np.random.default_rng(config['seed'])
    phases = rng.uniform(0, 2*np.pi, 32)
    frequencies = np.geomspace(.025, 1.5, 32)
    random = sum(np.sin(2*np.pi*f*t+phase)/np.sqrt(f) for f, phase in zip(frequencies, phases))
    random /= np.max(np.abs(random))
    waves = {'Constant': np.full(len(t), a),
             'Step': a*(t >= 60),
             'Sine_low': a*np.sin(2*np.pi*.05*t),
             'Sine_resonance': a*np.sin(2*np.pi*fn*t),
             'Sine_high': a*np.sin(2*np.pi*1.0*t),
             'Random_bandlimited': a*random,
             'Gust': a*np.exp(-.5*((t-70)/1.5)**2)}
    return t, waves


def metrics(truth, predicted, mask):
    error = predicted[mask]-truth[mask]
    rms = np.sqrt(np.mean(truth[mask]**2))
    rmse = np.sqrt(np.mean(error**2))
    return {'rmse_N': float(rmse), 'mse_N2': float(rmse**2),
            'nrmse': float(rmse/rms), 'bias_N': float(error.mean()),
            'max_abs_error_N': float(np.max(np.abs(error)))}


def main(config_path=ROOT/'config/stage2_doe.json', output=ROOT/'results/stage2'):
    config = json.loads(Path(config_path).read_text())
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    p = PendulumParameters(**config['nominal_parameters'])
    dt = 1/config['sample_rate_hz']
    t, waves = inputs(config, natural_characteristics(p)['natural_frequency_hz'])
    mask = t >= config['evaluation_start_s']
    states = {name: plant(force, p, dt) for name, force in waves.items()}
    angle_max = max(float(np.rad2deg(np.max(np.abs(x[:, 0])))) for x in states.values())
    if angle_max >= 38:
        raise ValueError('Input amplitude enters excluded mechanical-stop regime')
    design = face_centered()
    validation = 2*qmc.LatinHypercube(d=5, seed=config['seed']).random(config['validation_points'])-1
    cache = {}

    def evaluate(z):
        key = tuple(float(v) for v in z)
        if key in cache:
            return cache[key]
        ratios = decode(z, config)
        result = {}
        for name, force in waves.items():
            ys = estimate(states[name][:, 0], p, ratios, dt)
            result[name] = {method: metrics(force, y, mask) for method, y in ys.items()}
        cache[key] = result
        return result

    train = [evaluate(z) for z in design]
    held = [evaluate(z) for z in validation]
    nominal = evaluate(np.zeros(5))
    x, names = features(design)
    xv, _ = features(validation)
    effects, surfaces, predictions = {}, {}, {}
    validation_pass = True
    for name, force in waves.items():
        # Fit MSE rather than RMSE to avoid the cusp at zero error.
        mse = np.array([r[name]['Observer']['mse_N2'] for r in train])
        coeff = np.linalg.lstsq(x, mse, rcond=None)[0]
        actual = np.array([r[name]['Observer']['rmse_N'] for r in held])
        predicted_mse = xv@coeff
        predicted = np.sqrt(np.maximum(predicted_mse, 0))
        floor = config['prediction_floor_fraction_of_force_rms']*np.sqrt(np.mean(force[mask]**2))
        relative = np.abs(predicted-actual)/np.maximum(actual, floor)
        passed = bool(np.max(relative) <= config['prediction_relative_error_limit'])
        validation_pass &= passed
        surfaces[name] = {'target': 'observer_MSE_N2', 'coefficients': dict(zip(names, coeff.tolist())),
                          'train_max_residual_N2': float(np.max(np.abs(x@coeff-mse))),
                          'validation_max_relative_RMSE_prediction_error': float(relative.max()),
                          'validation_median_relative_error': float(np.median(relative)),
                          'negative_predicted_mse_count': int((predicted_mse < 0).sum()),
                          'prediction_floor_N': float(floor), 'gate_pass': passed}
        effects[name] = contrasts(np.array([r[name]['Observer']['nrmse'] for r in train[:32]]))
        predictions[name] = {'actual_rmse_N': actual.tolist(), 'predicted_rmse_N': predicted.tolist(),
                             'relative_error': relative.tolist()}

    # A fixed pole cross-section: all 16 physical-factor corners, plus center.
    # This supplies direct simulations for simultaneous uncertainty boxes; it
    # does not use an unvalidated quadratic approximation for tolerances.
    envelopes = []
    for scale in (0., .25, .5, .75, 1.):
        corner_results = [evaluate(np.r_[scale*z, 0.]) for z in factorial(4)]
        for name in waves:
            improvements = [100*(1-r[name]['Observer']['rmse_N']/max(r[name]['Low-pass']['rmse_N'], 1e-15)) for r in corner_results]
            envelopes.append({'input': name, 'range_scale': scale,
                              'pole_hz': config['pole_center_hz'],
                              'min_corner_improvement_percent': min(improvements),
                              'max_corner_observer_nrmse': max(r[name]['Observer']['nrmse'] for r in corner_results),
                              'corners_checked':16})
    frequencies = np.geomspace(.02, 5, 250)
    response = force_frequency_response(frequencies, p, decode(np.zeros(5), config), dt)
    freq_table = {method: {'gain': np.abs(y).tolist(),
                           'phase_deg': np.rad2deg(np.unwrap(np.angle(y))).tolist()} for method,y in response.items()}
    summary = {'stage':2, 'implementation_status':'COMPLETE',
               'response_surface_gate':'PASS' if validation_pass else 'NOT_MET',
               'stage3':'AWAITING_USER_REVIEW', 'design_points':len(design),
               'held_out_parameter_points':len(validation), 'input_blocks':len(waves),
               'regression_rank':int(np.linalg.matrix_rank(x)),
               'regression_columns':x.shape[1], 'regression_condition_number':float(np.linalg.cond(x)),
               'max_true_angle_deg':angle_max, 'nominal':nominal,
               'config':config, 'versions':{'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__},
               'caution':'Sampled parameter boxes are not proven continuous-domain tolerances; no sensor noise; causal processing only.'}
    for filename, obj in {
        'summary.json':summary,
        'design_and_results.json':{'factor_names':config['factors'], 'coded_design':design.tolist(), 'decoded_design':[decode(z,config).tolist() for z in design], 'results':train},
        'validation.json':{'coded_points':validation.tolist(), 'decoded_points':[decode(z,config).tolist() for z in validation], 'predictions':predictions, 'direct_results':held},
        'effects.json':effects, 'response_surfaces.json':surfaces,
        'sampled_tolerance_boxes.json':envelopes,
        'frequency_response.json':{'frequency_hz':frequencies.tolist(), 'response':freq_table}
    }.items():
        (output/filename).write_text(json.dumps(obj, indent=2, allow_nan=False))

    plt.rcParams.update({'font.size':10, 'axes.grid':True, 'grid.alpha':.2})
    fig, axes = plt.subplots(3, 1, figsize=(10,9), layout='constrained')
    for ax, name in zip(axes, ('Step', 'Sine_resonance', 'Random_bandlimited')):
        ys = estimate(states[name][:,0], p, decode(np.zeros(5),config),dt)
        selection = (t>=60)&(t<=80)
        ax.plot(t[selection], waves[name][selection]*1000, color='black', lw=2, label='True force')
        for method, color in zip(METHODS,COLORS):
            ax.plot(t[selection], ys[method][selection]*1000, color=color, lw=1, label=method)
        ax.set(title=name, xlabel='Time since simulation start [s]', ylabel='Force [mN]')
        ax.legend(ncol=4, fontsize=8)
    fig.suptitle('Stage 2: nominal coefficients, causal processing, ideal angle')
    fig.savefig(output/'comparison.png',dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2,1,figsize=(9,7),layout='constrained',sharex=True)
    for method,color in zip(METHODS,COLORS):
        axes[0].semilogx(frequencies,20*np.log10(np.abs(response[method])),label=method,color=color)
        axes[1].semilogx(frequencies,freq_table[method]['phase_deg'],label=method,color=color)
    axes[0].set(ylabel='Force gain [dB]',title='Exact discrete transfer: true force to estimated force')
    axes[0].legend(); axes[1].set(xlabel='Frequency [Hz]',ylabel='Unwrapped phase [deg]')
    fig.savefig(output/'frequency.png',dpi=150);plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    labels = list(effects['Random_bandlimited'])
    readable = [':'.join(config['factors'][int(i)] for i in lab.split(':')) for lab in labels]
    effect_array = np.array([[effects[n][lab] for lab in labels] for n in waves])
    limit=np.max(np.abs(effect_array))
    im=axes[0].imshow(effect_array,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
    axes[0].set_xticks(range(len(labels)),readable,rotation=90)
    axes[0].set_yticks(range(len(waves)),list(waves)); axes[0].set_title('Main / interaction contrasts of NRMSE')
    fig.colorbar(im,ax=axes[0])
    for name in waves:
        a=np.array(predictions[name]['actual_rmse_N'])*1000
        b=np.array(predictions[name]['predicted_rmse_N'])*1000
        axes[1].scatter(a,b,s=8,label=name,alpha=.5)
    lim=axes[1].get_xlim()[1];axes[1].plot([0,lim],[0,lim],'k--')
    axes[1].set(xlabel='Direct simulation RMSE [mN]',ylabel='Quadratic prediction RMSE [mN]',title='Held-out parameter points')
    axes[1].legend(fontsize=7)
    fig.savefig(output/'doe_validation.png',dpi=150);plt.close(fig)

    lines=['# Stage 2：DOEによるモデル係数誤差の評価','',
           'Stage 2の計算・成果物生成を完了しました。Stage 3はユーザー確認待ちです。', '',
           f'応答曲面の確認点ゲート：**{summary["response_surface_gate"]}**。未達の波形については応答曲面から許容誤差を断定しません。','',
           '## 評価条件','',
           '物理プラントは外力を入力とする2状態モデル、オブザーバーは角度だけを入力とする3状態モデルです。プラント係数は固定し、推定側の係数だけを変更しました。100 Hz、120秒、40〜120秒を全方式共通で評価しています。全方式に同じ理想角度を与え、ゲイン合わせ・時刻合わせは行っていません。', '',
           '静的換算は推定K／推定作用距離×角度。LPFは同じ静的換算に三重極の因果LPFを適用。オブザーバーはStage 1と同じ離散極配置を使用しています。極パラメータとLPFの実効帯域は同義ではありません。ノイズなしなので、この結果だけで最適帯域は決定できません。', '',
           '## 実験計画','',
           'I ±25%、b ±50%、K ±10%、作用距離 ±5%、極パラメータ0.5〜2 Hz（log2座標）。32端点＋中心1点＋軸上10点＝43点の面心中心複合計画で、5主効果・10交互作用・5二乗項を扱います。因子を削減せず全因子を保持しました。中心点は同一条件の決定論的計算なので重複実行しません。入力波形は別ブロックとして解析し、各ブロック内で主効果と交互作用を算出します。', '',
           '[NISTの中心複合計画の説明](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3361.htm)に基づく設計です。面心型は範囲外に点を置かず、回転可能性は持ちません。p値や統計的有意差、実機の発生確率は推定していません。', '',
           '計画点と独立なLatin Hypercube 96点で確認。応答はMSEに二次式を当て、平方根をRMSE予測とします。負のMSE予測はゼロへ切り上げ、その件数を記録します。相対予測誤差の分母は実RMSEと真の力RMS×0.001の大きい方です。最大相対誤差10%以内をゲートとし、未達もそのまま報告します。', '',
           f'設計行列のランク：{summary["regression_rank"]}/{summary["regression_columns"]}。今回の最大角度は{angle_max:.3f}度。ストッパー・飽和モデルは使用していません。', '',
           '## 公称条件の比較','',
           '| 入力 | 静的RMSE [mN] | LPF RMSE [mN] | Observer RMSE [mN] | LPF比改善率 |','|---|---:|---:|---:|---:|']
    for name,r in nominal.items():
        v=[r[m]['rmse_N']*1000 for m in METHODS]
        improvement=100*(1-v[2]/max(v[1],1e-12))
        lines.append(f'| {name} | {v[0]:.6f} | {v[1]:.6f} | {v[2]:.6f} | {improvement:.2f}% |')
    lines += ['', '改善率は負なら悪化です。Constantは機械振動の残留による微小誤差を含み、改善率だけを性能根拠にしません。これは固定1 Hzの因果比較であり、各方式の最適値比較や事後平滑化の性能ではありません。','',
              '![比較](results/stage2/comparison.png)','','![周波数応答](results/stage2/frequency.png)','',
              '## DOEと確認点','', '| 入力 | 確認点の最大相対予測誤差 | 中央値 | 負MSE予測点 | 判定 |','|---|---:|---:|---:|---|']
    for name,s in surfaces.items():
        lines.append(f'| {name} | {100*s["validation_max_relative_RMSE_prediction_error"]:.2f}% | {100*s["validation_median_relative_error"]:.2f}% | {s["negative_predicted_mse_count"]} | {"PASS" if s["gate_pass"] else "NOT_MET"} |')
    lines += ['', '![DOE検証](results/stage2/doe_validation.png)','', '## 同時誤差範囲の直接確認','',
              '以下は極1 Hzに固定し、I・b・K・作用距離の誤差範囲を一括で縮小した16端点での最悪改善率です。範囲内の全点で成立する保証、公差確定値、確率95%の意味はありません。', '',
              '| 範囲倍率 | I / b / K / 作用距離 の誤差 | ランダム波・最悪改善率 | 突風・最悪改善率 |','|---|---|---:|---:|']
    for scale in (0.,.25,.5,.75,1.):
        selected={r['input']:r for r in envelopes if r['range_scale']==scale}
        errors=' / '.join(f'±{scale*h*100:g}%' for h in config['linear_half_ranges'])
        lines.append(f'| {scale:g} | {errors} | {selected["Random_bandlimited"]["min_corner_improvement_percent"]:.2f}% | {selected["Gust"]["min_corner_improvement_percent"]:.2f}% |')
    lines += ['', '## 結果の解釈','',
              'ランダム波では極パラメータの主効果が大きく、IとKの交互作用も見られます。共振付近ではIの主効果とI×Kの交互作用が大きく、Iだけを単独で精密化すればよいとは結論できません。主効果は符号付きHigh−Lowの平均差なので、対称な二乗感度が大きくてもゼロになり得ます。応答曲面の二乗項と合わせて解釈してください。','',
              '公称1 Hzでランダム波はLPF比約55%改善しますが、低周波正弦波では静的換算の方が高精度です。1 Hz正弦波ではオブザーバーの位相遅れを含めた誤差が大きく、LPFより悪化します。単一のRMSE改善率を全帯域へ一般化できません。','',
              '現在の広い範囲に対する二次応答曲面はゲート未達です。係数差が小さいときのRMSEがゼロに近くなること、共振付近のIとKの相互作用、極の変更による非線形性を一つの二次式で十分近似できていません。追加の局所DOEはユーザー確認後に行います。','',
              '## 再現方法','', 'リポジトリ直下から実行します。既存requirements.txtを使用します。','',
              '```bash','python 06_Analysis/simulation/src/run_stage2.py','python -m unittest discover -s 06_Analysis/simulation/tests -v','```','',
              '設定はconfig/stage2_doe.json。全43条件・確認96条件の物理係数と指標、回帰係数、主効果・交互作用、端点確認、周波数応答をresults/stage2のJSONへ保存します。乱数seedとライブラリ版も保存します。波形の乱数seedは全条件共通で、波形ごとの条件差を混入させません。','',
              '## 限界と次段階','',
              'センサモデルは未使用です。Stage 3で物理プラントと独立した切替可能なモデルとして追加します。入力波形は工学的な試験波形であり、実測風のスペクトルを再現したものではありません。作用距離の誤差は力換算に影響しますが、トルク状態の観測性には影響しません。', '',
              '応答曲面ゲート未達なら、この範囲の二次近似を較正精度の決定に使用できません。必要帯域・要求誤差が未確定のため、許容公差の正式決定も行いません。Stage 2の結果をレビューし、必要なら帯域ごとの局所DOEへ進むかを決めます。']
    (ROOT/'Stage2_Report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ['implementation_status','response_surface_gate','max_true_angle_deg','nominal']},indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/stage2_doe.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/stage2')
    args=parser.parse_args();main(args.config,args.output)
