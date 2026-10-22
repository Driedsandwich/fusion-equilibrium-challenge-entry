#!/usr/bin/env python3
"""Offline DIII-D physics gate, using every fourth true-flux frame."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'mast'), str(ROOT)]
import qb_io as io
from qb_physics import (TF_TURNS, vacuum_F, flux_geometry, q_at, volume_average_pressure, beta_n,
                        in_range, fill_out_of_range)
from data_fixes import fix_d3d_ip_times
import argparse
import gc
import json
import re
import numpy as np

COLUMNS = ['efit_times', 'efit_grid_R', 'efit_grid_Z', 'efit_psirz', 'efit_q95',
           'efit_beta_n', 'magnetics_time', 'magnetics_bcoil', 'magnetics_plasma_current',
           'magnetics_plasma_current_times', 'thomson_core_times', 'thomson_core_R',
           'thomson_core_Te', 'thomson_core_ne', 'thomson_chord_name',
           'thomson_chord_R', 'thomson_chord_Z']


def core_positions(row):
    """Map channel j by the numeric suffix of TS_core_<region>_j, per shot.

    Region flags r+1/r+0/r-1 are not channel numbers and are not filters.
    Reject missing/duplicate indices or a radial-position mismatch.
    """
    core_R = np.asarray(row['thomson_core_R'], float)
    names = row['thomson_chord_name']
    r, z = np.asarray(row['thomson_chord_R']), np.asarray(row['thomson_chord_Z'])
    n = len(core_R)
    if len(names) != len(r) or len(r) != len(z):
        raise ValueError('Chord coordinate counts differ')
    positions = {}
    for name, rr, zz in zip(names, r, z):
        if not str(name).startswith('TS_core_'):
            continue
        match = re.fullmatch(r'TS_core_.+_(\d+)', str(name))
        if match is None or int(match[1]) in positions:
            raise ValueError('Ambiguous core chord channel')
        positions[int(match[1])] = (rr, zz)
    if set(positions) != set(range(n)):
        raise ValueError('Core chord indices must be exactly 0..C-1')
    points = np.asarray([positions[j] for j in range(n)], float)
    if not np.isfinite(points).all() or not np.allclose(points[:, 0], core_R, rtol=0, atol=1e-5):
        raise ValueError('Core chord R does not match profile R')
    if np.asarray(row['thomson_core_Te']).shape != (len(row['thomson_core_times']), n):
        raise ValueError('Te/channel count mismatch')
    if np.asarray(row['thomson_core_ne']).shape != np.asarray(row['thomson_core_Te']).shape:
        raise ValueError('Te/ne layout mismatch')
    return points[:, 0], points[:, 1]


def metric_state():
    return dict(frames=0, truth_missing=0, failures=0, sum_y=0., sum_y2=0.,
                ssres=0., failed_y=0., failed_y2=0.)


def metric_add(s, y, pred):
    s['frames'] += 1
    if not np.isfinite(y):
        s['truth_missing'] += 1
        return
    s['sum_y'] += float(y)
    s['sum_y2'] += float(y*y)
    if np.isfinite(pred):
        s['ssres'] += float((pred-y)**2)
    else:
        s['failures'] += 1
        s['failed_y'] += float(y)
        s['failed_y2'] += float(y*y)


def metric_result(s):
    n = s['frames']-s['truth_missing']
    mean = s['sum_y']/n if n else 0.
    sstot = s['sum_y2']-n*mean**2
    # Same scorer policy: failed prediction = the pooled holdout target mean.
    ssres = s['ssres']+s['failed_y2']-2*mean*s['failed_y']+s['failures']*mean**2
    return dict(frames=s['frames'], finite_targets=n, failures=s['failures'],
                failure_fraction=s['failures']/n if n else None,
                r2=1-ssres/sstot if sstot > 0 else None,
                SSres=ssres, SStot=sstot, target_mean=mean if n else None)


def snapshot(state, output):
    state['metrics'] = {k: metric_result(v) for k, v in state['metric_sums'].items()}
    io.json_write(output, state)
    text = [f"# DIII-D q95・βN の物理式の確認（{state['status']}）", '',
            f"処理済みは学習 {state['train_processed']} ショット・保留 {state['holdout_processed']} ショットです。4枚に1枚を使います。",
            f"全圧力係数 c_p は {state['c_p']} です。名目値は2です。",
            '巻数144は今回の依頼で指定された装置定数です。READMEは電流の単位kAを確認する根拠です。', '',
            'DIII-Dのcoreは縦の測定です。README 587–596・633–634・668–676行を根拠に、各ショットの',
            '`TS_core_<region>_<j>` の末尾番号jをTe/neの列jへ対応させます。regionの札では絞りません。',
            '番号の重複・欠落、チャネル数、core_Rとの半径の一致（許容差10⁻⁵ m）を検査します。',
            '曖昧なショットはβNをNaNとして数えます。別ショットの位置は再利用しません。', '',
            'Ipは波形全体へ公式fix_d3d_ip_timesを適用してから補間します。βNの分母には電流の大きさを使います。磁場は真空Fから求めます。',
            '電子圧力は20個のψ_N区間の測定平均です。空の区間は補間し、未測定の両端は直近値を使います。',
            '各区間は磁束面の体積差で重み付けします。真空F一定・電子圧力から全圧力への共通係数は物理近似です。', '',
            'R²は保留全フレームの1−SSres/SStotです。導出失敗は保留の真値平均で置換して集計します。',
            '学習・保留の境界はbase/v2_full_result.jsonに従います。試運転は合計2ショットなら学習1・保留1です。', '',
            '|項目|フレーム数|失敗数|R²|', '|---|---:|---:|---:|']
    for k, v in state['metrics'].items():
        text.append(f"|{k}|{v['frames']}|{v['failures']}|{v['r2']}|")
    text += ['', f"位置の対応が決まらないショット数: {state['unmapped_shots']}。",
             f"見張り無効: {state['guard_disabled']}。停止理由: {state.get('stop_reason')}。",
             '試運転の係数を本番用の校正値とは扱いません。本番の学習60ショットと全保留は未実行です。'
             if state['limit_shots'] else '指定された学習60ショット・全保留の処理状態はJSONのstatusを確認してください。', '']
    io.workspace_path(output.with_suffix('.md') if output.name != 'qb_gate_result.json'
                      else output.with_name('qb_gate_report.md'), output=True).write_text('\n'.join(text))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit-shots', type=int, default=0,
                    help='total cap across train and holdout; 2 means 1+1 (0=full)')
    ap.add_argument('--no-guard', action='store_true', help='smoke runs only')
    ap.add_argument('--output', default='diag/qb_gate_result.json')
    args = ap.parse_args(argv)
    if args.limit_shots < 0 or args.limit_shots == 1 or (args.no_guard and not args.limit_shots):
        ap.error('Use a limit >=2 for smoke/no-guard, or 0 for guarded full evaluation')
    split = json.loads((ROOT/'base/v2_full_result.json').read_text())
    train, hold = split['train_shots'][:60], split['holdout_shots']
    if set(train) & set(hold):
        raise ValueError('Overlapping train and holdout')
    if args.limit_shots:
        train = train[:max(1, args.limit_shots//2)]
        hold = hold[:max(1, args.limit_shots-len(train))]
    output = io.workspace_path(args.output, output=True)
    with np.load(ROOT/'starter_ref/fusion_scoring/masks/d3d_envelope.npz', allow_pickle=False) as f:
        R, Z, mask = f['grid_R'], f['grid_Z'], f['mask_coarse'].astype(bool)
    mask_f = mask.astype(float)  # exactly the published local scorer usage
    state = dict(status='running', limit_shots=args.limit_shots, guard_disabled=args.no_guard,
                 train_shots=train, holdout_shots=hold, stride=4, pressure_bins=20,
                 train_processed=0, holdout_processed=0, unmapped_shots=0,
                 c_p=None, calibration_complete=False, fit=dict(xy=0., xx=0., pairs=0),
                 code_sha256={p.name: io.digest(p) for p in [Path(__file__), ROOT/'mast/qb_physics.py',
                              ROOT/'mast/qb_io.py', ROOT/'starter_ref/data_fixes.py']},
                 metric_sums={k: metric_state() for k in ['q95', 'betaN_fitted', 'betaN_nominal2']}, shots=[])
    frames_out = []
    try:
        for phase, shots in [('train', train), ('holdout', hold)]:
            for name in shots:
                io.resource_guard(args.no_guard)
                row = io.read_columns(ROOT/'_data/diii_d_train'/name, COLUMNS)
                if not np.array_equal(row['efit_grid_R'], R) or not np.array_equal(row['efit_grid_Z'], Z):
                    raise ValueError('DIII-D mask grid mismatch')
                idx = np.arange(0, len(row['efit_times']), 4)
                times = row['efit_times'][idx]
                # Correction must see the entire Ip waveform, not a sliced prefix.
                ip, F = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
                current_errors = {}
                try:
                    fixed = fix_d3d_ip_times(row)
                    ip[:] = io.interp(fixed, row['magnetics_plasma_current'], times)
                except (ValueError, TypeError, IndexError) as exc:
                    current_errors['Ip'] = str(exc)
                try:
                    F[:] = vacuum_F(io.interp(row['magnetics_time'], row['magnetics_bcoil'], times)/1e3,
                                    TF_TURNS['DIII-D'])
                except (ValueError, TypeError, IndexError) as exc:
                    current_errors['TF'] = str(exc)
                mapping_error = None
                try:
                    sr, sz = core_positions(row)
                    profiles = io.repair_profiles(row['thomson_core_times'], row['thomson_core_Te'],
                                                 row['thomson_core_ne'], sr, sz)
                except (ValueError, TypeError, IndexError) as exc:
                    profiles = None
                    mapping_error = str(exc)
                    state['unmapped_shots'] += 1
                qp, bp = [], []
                for j, frame in enumerate(idx):
                    if j % 16 == 0:
                        io.resource_guard(args.no_guard)
                    psi = row['efit_psirz'][frame]
                    geom = flux_geometry(psi, R, Z, 'DIII-D', mask, mask_f)
                    q = q_at(psi, R, Z, F[j], machine='DIII-D', geometry=geom)
                    pe = np.nan
                    if profiles is not None:
                        ts, te, ne, sr, sz = profiles
                        k = io.nearest_index(ts, times[j])
                        pe = volume_average_pressure(psi, R, Z, sr, sz, te[k], ne[k],
                                                     'DIII-D', geometry=geom)
                    # Normalised beta uses current strength, not circuit polarity.
                    b1 = beta_n(pe, geom['a'], abs(F[j])/geom['R_geo'], abs(ip[j]))
                    qp.append(q); bp.append(b1)
                    y = float(row['efit_beta_n'][frame])
                    # Fit c_p only on frames whose nominal estimate (c_p = 2) is physical,
                    # the same range the MAST fill policy applies (label-free screen).
                    if phase == 'train' and np.isfinite(y) and in_range('betaN', 2*b1):
                        state['fit']['xy'] += float(b1*y)
                        state['fit']['xx'] += float(b1*b1)
                        state['fit']['pairs'] += 1
                    if phase == 'holdout':
                        metric_add(state['metric_sums']['q95'], float(row['efit_q95'][frame]), q)
                        cp = state['c_p'] if state['c_p'] is not None else np.nan
                        metric_add(state['metric_sums']['betaN_fitted'], y, cp*b1)
                        metric_add(state['metric_sums']['betaN_nominal2'], y, 2*b1)
                if phase == 'holdout':
                    frames_out.append(dict(q=np.asarray(qp, float), b1=np.asarray(bp, float),
                                           yq=np.asarray(row['efit_q95'][idx], float),
                                           yb=np.asarray(row['efit_beta_n'][idx], float)))
                state[phase+'_processed'] += 1
                state['shots'].append(dict(phase=phase, file=name, frames=len(idx),
                    q95=io.finite_summary(qp), betaN_cp1=io.finite_summary(bp),
                    q95_failures=int(np.sum(~np.isfinite(qp))),
                    betaN_failures=int(np.sum(~np.isfinite(bp))), mapping_error=mapping_error,
                    current_errors=current_errors))
                if phase == 'train':
                    fit = state['fit']
                    state['c_p'] = fit['xy']/fit['xx'] if fit['xx'] > 0 else None
                snapshot(state, output)
                print(json.dumps(state['shots'][-1]), flush=True)
                del row, profiles, psi
                gc.collect()
        state['status'] = 'smoke_complete' if args.limit_shots else 'complete'
        state['calibration_complete'] = not args.limit_shots and len(train) == 60 and state['c_p'] is not None
    except io.ResourceStop as exc:
        state.update(status='stopped_resource', stop_reason=str(exc))
        snapshot(state, output)
        print(str(exc), file=sys.stderr)
        return 2
    state['metrics_range_filled'] = range_filled_metrics(frames_out, state['c_p'])
    snapshot(state, output)
    np.savez_compressed(io.workspace_path(output.with_name(output.stem+'_frames.npz'), output=True),
                        **{f'shot_{i:03d}_{k}': v for i, d in enumerate(frames_out) for k, v in d.items()})
    print(json.dumps(dict(status=state['status'], c_p=state['c_p'], metrics=state['metrics'],
                          metrics_range_filled=state['metrics_range_filled'])), flush=True)
    return 0


def range_filled_metrics(frames_out, cp):
    """Holdout R2 after the MAST fill policy: failed or out-of-range frames take the
    in-range median of their shot, else the in-range median over all holdout shots."""
    out = {}
    for field, key, scale in [('q95', 'q', 1.), ('betaN', 'b1', cp)]:
        if scale is None or not frames_out:
            out[field] = None
            continue
        preds = [scale*d[key] for d in frames_out]
        pooled = np.concatenate([p[in_range(field, p)] for p in preds])
        fallback = float(np.median(pooled)) if len(pooled) else np.nan
        filled, fills = [], 0
        for p in preds:
            f, n = fill_out_of_range(field, p, fallback)
            filled.append(f); fills += n
        y = np.concatenate([d['yq' if field == 'q95' else 'yb'] for d in frames_out])
        pred = np.concatenate(filled)
        ok = np.isfinite(y)
        sstot = float(np.sum((y[ok]-y[ok].mean())**2))
        ssres = float(np.sum((pred[ok]-y[ok])**2))
        out[field] = dict(frames=int(ok.sum()), filled=fills, r2=1-ssres/sstot if sstot > 0 else None,
                          SSres=ssres, SStot=sstot)
    return out


if __name__ == '__main__':
    raise SystemExit(main())
