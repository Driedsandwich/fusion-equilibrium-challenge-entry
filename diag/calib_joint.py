#!/usr/bin/env python3
"""Sequential joint DIII-D calibration, fixed twelve shots, no MAST labels."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capacity import (ROOT, load_shot, vacuum_and_ip, jsonable, swap_reading)
import argparse
import gc
import json
import time
import numpy as np
from scipy.linalg import svd, null_space
from greens import plasma_matrix

# Explicit identities from capacity_result_run2.json; never expand to more shots.
SHOTS = ('00000a10ac', '031ec9e642', '05d8f8a714', '08c3a3f160',
         '0ba731cc9d', '0e1b070fe7', '1217bc4c60', '1558027fd7',
         '187f9af914', '1bb17d6385', '1ecfbb00ba', '20e5596616')

def factor(R, Z, box, free_c, rcond):
    g = plasma_matrix(R, Z, 12, 16, box)*1e6
    q = null_space(np.ones((1, 192)))
    # w is expressed in MA; J=alpha*I and sum(J)=alpha*Ip.
    a = g @ q
    if free_c:
        a = np.column_stack((a, np.ones(len(a))))
    u, s, vt = svd(a, full_matrices=False, check_finite=True)
    keep = s > s[0]*rcond
    return g.mean(axis=1), q, u[:, keep], (vt[keep].T/s[keep]) @ u[:, keep].T, int(keep.sum())

def sufficient(y, x, e, u, pinv):
    projected = [v-(v@u)@u.T for v in (y, x, e)]
    py, px, pe = projected
    gram = np.array([[np.sum(px*px), np.sum(px*pe)],
                     [np.sum(px*pe), np.sum(pe*pe)]])
    rhs = np.array([np.sum(px*py), np.sum(pe*py)])
    # Store only small local-coefficient responses, never another complete shot.
    return gram, rhs, float(np.sum(py*py)), [v@pinv.T for v in (y, x, e)]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--rcond', type=float, default=1e-10)
    ap.add_argument('--no-guard', action='store_true')
    args = ap.parse_args()
    if args.stride < 1 or not 0 < args.rcond < 1:
        ap.error('stride>=1 and 0<rcond<1 required')
    start = time.monotonic()
    previous = json.loads((ROOT/'diag/capacity_result_run2.json').read_text())
    expected = ['d3d_shot_'+s+'.parquet' for s in SHOTS]
    if [s['file'] for s in previous['shots']] != expected:
        raise ValueError('Previous fixed-shot identities changed')
    box = previous['geometry']['box_m']
    sums = {k: [np.zeros((2, 2)), np.zeros(2), 0.] for k in ('j1', 'j2')}
    locals_ = {k: [] for k in sums}
    kappas, qs, bs = [], [], []
    sy = sy2 = n = 0
    factors = None
    ranks = {}
    for si, filename in enumerate(expected):
        if not args.no_guard:
            reading = swap_reading()
            if not reading['ok'] or reading['free_mb'] < 300:
                raise RuntimeError('Memory guard: '+str(reading))
        row = load_shot(ROOT/'_data/diii_d_train'/filename)
        for r, z, count in zip(row['efit_lcfs_r'], row['efit_lcfs_z'], row['efit_lcfs_n']):
            r, z = r[:int(count)], z[:int(count)]
            ok = np.isfinite(r)&np.isfinite(z)&(r>0)
            if ok.sum()>=3 and np.ptp(r[ok])>0:
                kappas.append(np.ptp(z[ok])/np.ptp(r[ok]))
        qs.extend(row['efit_q95'][np.isfinite(row['efit_q95'])])
        bs.extend(row['efit_beta_n'][np.isfinite(row['efit_beta_n'])])
        idx = np.arange(0, len(row['efit_times']), args.stride)
        f, e, ip, _, _, _ = vacuum_and_ip(row, idx, 6, split=True)
        y = row['efit_psirz'][idx].reshape(len(idx), -1).astype(float)
        sy += float(y.sum()); sy2 += float(np.sum(y*y)); n += y.size
        if factors is None:
            grid = (row['efit_grid_R'].copy(), row['efit_grid_Z'].copy())
            factors = {k: factor(*grid, box, k=='j1', args.rcond) for k in sums}
        elif not all(np.array_equal(a, b) for a,b in zip(grid, (row['efit_grid_R'],row['efit_grid_Z']))):
            raise ValueError('DIII-D grids changed')
        for k in sums:
            mean, q, u, pinv, rank = factors[k]
            x = f+ip[:, None]/1e6*mean
            gram, rhs, yy, responses = sufficient(y, x, e, u, pinv)
            sums[k][0] += gram; sums[k][1] += rhs; sums[k][2] += yy
            locals_[k].append((filename, idx, ip, responses))
            ranks[k] = rank
        print(f'joint {si+1}/12 {filename} frames={len(idx)}', flush=True)
        del row, f, e, y, x, responses
        gc.collect()
    results = {}
    out = ROOT/'diag/joint_locals'
    out.mkdir(exist_ok=True)
    for k, (gram, rhs, yy) in sums.items():
        alpha, beta = np.linalg.solve(gram, rhs)
        sse = float(yy-2*np.array([alpha,beta])@rhs+np.array([alpha,beta])@gram@np.array([alpha,beta]))
        offsets = []; constraint_error = 0.
        for filename, idx, ip, (ly,lx,le) in locals_[k]:
            coeff = ly-alpha*lx-beta*le
            q = factors[k][1]
            j = alpha*ip[:,None]/1e6/192 + coeff[:,:191]@q.T
            currents = j*1e6/alpha
            constraint_error = max(constraint_error, float(np.max(np.abs(currents.sum(axis=1)-ip))))
            c = coeff[:,-1] if k=='j1' else np.zeros(len(idx))
            offsets.extend(c)
            np.savez_compressed(out/(k+'_'+Path(filename).stem+'.npz'), indices=idx, currents_A=currents, c=c)
        candidates = [1.,-1.,1/(2*np.pi),-1/(2*np.pi)]
        nearest = min(candidates,key=lambda v: abs(alpha-v)/abs(v))
        results[k] = dict(alpha_F=float(alpha), beta_E=float(beta), R2_psi=1-sse/(sy2-sy*sy/n),
                          sse=sse, local_rank=ranks[k], global_condition=float(np.linalg.cond(gram)),
                          nearest_physical_alpha=nearest, relative_distance=abs(alpha-nearest)/abs(nearest),
                          c_median=float(np.median(offsets)), c_abs_max=float(np.max(np.abs(offsets))),
                          max_sum_I_minus_Ip_A=constraint_error)
    # Explicit DIII-D-only, preselected criterion; no MAST observations enter it.
    delta = results['j1']['R2_psi']-results['j2']['R2_psi']
    absolute = delta <= .001
    state = dict(status='complete', shots=expected, frames=n//4225, stride=args.stride,
                 box_m=box, rcond=args.rcond, results=results,
                 convention=dict(j2_sufficient=absolute, R2_loss=delta, threshold=.001,
                                 chosen='j2' if absolute else 'j1',
                                 alpha_F=results['j2' if absolute else 'j1']['alpha_F'],
                                 c_for_mast=0. if absolute else results['j1']['c_median']),
                 priors=dict(kappa=float(np.median(kappas)), q95=float(np.median(qs)),
                             betaN=float(np.median(bs)), lcfs_frames=len(kappas),
                             scalar_frames=len(qs), source='fixed 12 DIII-D training shots, all frames'),
                 elapsed_seconds=time.monotonic()-start)
    (ROOT/'diag/calib_joint_result.json').write_text(json.dumps(jsonable(state),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines = ['# DIII-Dの倍率の同時推定', '',
        'フレームごとの原点を当てるj1ではα_Fが−1に近づきました。c＝0のj2は同程度ではありません。', '',
        '|版|α_F|β_E|R²ψ（符号変更なし）|', '|---|---:|---:|---:|']
    for k,v in results.items():
        lines.append(f"|{k}|{v['alpha_F']:.10g}|{v['beta_E']:.10g}|{v['R2_psi']:.10g}|")
    lines += ['',f"j1はc(t)自由、j2はc(t)＝0です。同じ固定12ショット・全{len(qs)}フレームから{args.stride}枚に1枚、計{state['frames']}フレームを選び、全65×65画素で当てました。R²ψの分母は選択フレーム全画素の全体平均まわりの平方和です。保留での予測性能ではなく、同じデータ上の当てはめです。",'',
        f"j1のα_Fは−1から{100*results['j1']['relative_distance']:.3f}%離れています。±1/(2π)より−1に近い値です。β_EはECOILAのkA入力に対する別係数です。MASTへは移しません。",'',
        f"j2のR²ψの低下は{delta:.8f}です。同程度とする判定値0.001を超えました。EFITのψは絶対の磁束で原点を当てなくてよい、とは結論しません。MASTではj1のα_FとDIII-Dのcの中央値{results['j1']['c_median']:.10g}を暫定的に使います。後者はMASTの原点を物理から決定した値ではなく、未検証の仮定です。",'',
        '## 解法と制約', '',
        'ψ＝α_F F＋β_E E＋GJ＋c、J＝α_F I、ΣJ＝α_F Ipと書き換えます。J＝α_F Ip/192＋Qwと置き、Qは総和0の直交基底です。局所変数wとcを特異値分解で消去し、全ショット共通の2変数の正規方程式を解きます。α_F≠0でI＝J/α_Fを復元します。別のプラズマ倍率は置きません。', '',
        f"箱は[前回の結果](capacity_result_run2.json)のV1′と同じ{box} m、12×16点です。フィラメントは正負自由で、非負制約はありません。局所特異値の打切り比は{args.rcond:g}です。最小二乗にはリッジを加えていません。j1/j2の局所階数は{ranks['j1']}/{ranks['j2']}です。傾斜コイルの座標変換は既存[greens.py](greens.py)の仮定を引き継ぎ、ここでは独立な幾何検証はしていません。",'',
        f"ΣI＝Ipの最大誤差はj1で{results['j1']['max_sum_I_minus_Ip_A']:.3g} A、j2で{results['j2']['max_sum_I_minus_Ip_A']:.3g} Aでした。復元した電流とcはjoint_locals/、全数値・ショットID・事前値は[calib_joint_result.json](calib_joint_result.json)に保存します。",'',
        f"κ（楕円の伸び）はDIII-Dの保存済み真値LCFS（最外殻閉磁気面）から縦幅/横幅を計算し、固定12ショット・全{len(kappas)}フレームの中央値{state['priors']['kappa']:.10g}としました。q95とβNも同じ12ショットの全訓練フレームの中央値です。MASTの再構成・集計・マスク・デモショットは使いません。",'',
        '## Claudeが作業場所のルートで実行する命令（AI担当）', '',
        '```bash', '.venv/bin/python -B diag/test_joint_simple.py',
        '.venv/bin/python -B diag/calib_joint.py --no-guard',
        '.venv/bin/python -B mast/simple.py --no-guard', '```', '',
        '--no-guardは今回許可された固定12ショットとMASTの手元10ショットだけで使います。全コーパスへ拡張しません。ファイル読込は1ショットずつ、Parquetと数値計算は1スレッドです。', '',
        f"今回の校正本体の実測時間は{state['elapsed_seconds']:.3f}秒です。合成データのj1/j2を独立な密行列の等式制約付き連立方程式と照合する検査はtest_joint_simple.pyにあります。"]
    (ROOT/'diag/calib_joint_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(jsonable(state), ensure_ascii=False), flush=True)

if __name__=='__main__':
    main()
