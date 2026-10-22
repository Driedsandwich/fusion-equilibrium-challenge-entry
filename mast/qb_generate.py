#!/usr/bin/env python3
"""Sequential, resumable, label-free MAST q95/betaN generation from final6 psi."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'mast'))
import qb_io as io
from qb_physics import (TF_TURNS, in_range, vacuum_F, flux_geometry, q_at,
                        volume_average_pressure, beta_n)
import argparse
import gc
import json
import zipfile
import numpy as np


def save_chunk(path, arrays):
    tmp = path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def source_signature(paths, psi_path, calibration_path, cp, limit, axis_sign):
    stat = psi_path.stat()
    return dict(files=[p.name for p in paths], limit_shots=limit, axis_sign=axis_sign,
                psi_source=str(psi_path.relative_to(ROOT)), psi_size=stat.st_size,
                psi_mtime_ns=stat.st_mtime_ns, calibration_sha256=io.digest(calibration_path), c_p=cp,
                code_sha256={p.name: io.digest(p) for p in [Path(__file__),
                    ROOT/'mast/qb_physics.py', ROOT/'mast/qb_io.py', ROOT/'mast/simple.py',
                    ROOT/'starter_ref/data_fixes.py']})


def recover_pending(directory, state):
    """Complete a recorded atomic chunk replacement after any interruption."""
    pending = state.get('pending_update')
    if pending is None:
        return
    shot = state['shots'][pending['index']]
    path = directory/shot['chunk']
    actual = io.digest(path)
    if actual == shot['sha256']:
        tmp = path.with_name(path.name+'.partial')
        if not tmp.exists() or io.digest(tmp) != pending['sha256']:
            raise ValueError('Interrupted chunk replacement has no matching temporary file')
        tmp.replace(path)
    elif actual != pending['sha256']:
        raise ValueError('Interrupted chunk replacement identity mismatch')
    shot['sha256'] = pending['sha256']
    state.pop('pending_update')
    io.json_write(directory/'manifest.json', state)


def finish_chunks(directory, state, no_guard):
    """Pool only successful raw estimates; reapply fills on resume from raw arrays."""
    successful = {f: [] for f in ['q95', 'betaN']}
    for shot in state['shots']:
        io.resource_guard(no_guard)
        path = directory/shot['chunk']
        if io.digest(path) != shot['sha256']:
            raise ValueError(f'Changed chunk: {path}')
        with np.load(path, allow_pickle=False) as z:
            for field in successful:
                a = z['raw_'+field]
                successful[field].append(a[in_range(field, a)])
    medians = {}
    for field, pieces in successful.items():
        values = np.concatenate(pieces) if pieces else np.array([])
        medians[field] = float(np.median(values)) if len(values) else None
    state['global_success_medians'] = medians
    if any(v is None for v in medians.values()):
        state['status'] = 'no_successful_estimates'
        return False
    counts = {f: dict(failed=0, shot_median=0, global_median=0) for f in successful}
    for shot in state['shots']:
        io.resource_guard(no_guard)
        path = directory/shot['chunk']
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        for field in successful:
            raw = arrays['raw_'+field]
            fail = ~in_range(field, raw)
            good = raw[~fail]
            value = float(np.median(good)) if len(good) else medians[field]
            filled = np.where(fail, value, raw)
            with np.errstate(over='ignore', invalid='ignore'):
                stored = filled.astype(np.float16)
            if not np.isfinite(stored).all() or stored.shape != (shot['frames'],):
                raise ValueError(f'Estimate outside finite float16 range: {path} {field}')
            arrays[field] = stored
            arrays['failed_'+field] = fail
            n = int(fail.sum())
            counts[field]['failed'] += n
            counts[field]['shot_median' if len(good) else 'global_median'] += n
            shot[field] = io.finite_summary(stored)
        tmp = path.with_name(path.name+'.partial')
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
        state['pending_update'] = dict(index=shot['index'], sha256=io.digest(tmp))
        # Record both identities before replacing; recover either side of the rename.
        io.json_write(directory/'manifest.json', state)
        recover_pending(directory, state)
    state['fill_counts'] = counts
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit-shots', type=int, default=0)
    ap.add_argument('--no-guard', action='store_true', help='smoke runs only')
    ap.add_argument('--work-dir', help='default: _work/final7/qb[/smokeN]')
    ap.add_argument('--calibration', default='diag/qb_gate_result.json')
    ap.add_argument('--axis-sign', type=float, choices=[-1., 1.], default=-1.,
                    help='final6 uses negative DIII-D alpha_F: phi=-psi, without modifying psi')
    args = ap.parse_args(argv)
    if args.limit_shots < 0 or (args.no_guard and not args.limit_shots):
        ap.error('--no-guard requires a positive smoke limit')
    paths = sorted((ROOT/'_data/final/mast_private_test').glob('*.parquet'))
    if len(paths) != 1205:
        raise ValueError(f'Expected 1205 sorted MAST input files, got {len(paths)}')
    if args.limit_shots:
        paths = paths[:args.limit_shots]
    calibration_path = io.workspace_path(args.calibration)
    calibration = json.loads(calibration_path.read_text())
    cp = calibration.get('c_p')
    if cp is None or not np.isfinite(cp) or cp <= 0:
        raise ValueError('Require a finite positive DIII-D fitted c_p')
    if not args.limit_shots and not (calibration.get('calibration_complete')
                                   and calibration.get('status') == 'complete'):
        raise ValueError('Full generation requires the completed 60-shot DIII-D calibration')
    default = '_work/final7/qb'+(f'/smoke{args.limit_shots}' if args.limit_shots else '')
    directory = io.workspace_path(args.work_dir or default, output=True)
    directory.mkdir(parents=True, exist_ok=True)
    psi_path = ROOT/'_work/final6/out/mast_private_test.npz'
    signature = source_signature(paths, psi_path, calibration_path, cp, args.limit_shots, args.axis_sign)
    manifest = directory/'manifest.json'
    if manifest.exists():
        state = json.loads(manifest.read_text())
        if state['signature'] != signature:
            raise ValueError('Resume signature mismatch; use a fresh --work-dir')
        recover_pending(directory, state)
    else:
        state = dict(status='running', signature=signature, shots=[], c_p=cp,
                     calibration_complete=calibration.get('calibration_complete', False),
                     guard_disabled=args.no_guard, expected_shots=len(paths),
                     production_complete=False, pressure_bins=20)
    try:
        with np.load(psi_path, allow_pickle=False) as archive, zipfile.ZipFile(psi_path) as zpsi:
            for i, path in enumerate(paths):
                io.resource_guard(args.no_guard)
                # Schema guard runs even on resumed shots, before any projected read.
                names = io.pq.ParquetFile(path).schema_arrow.names
                if any(n.startswith('efit_') and n not in ('efit_times', 'efit_grid_R', 'efit_grid_Z')
                       for n in names):
                    raise ValueError('Refuse MAST equilibrium label columns')
                member = f'shot_{i:04d}_psirz'
                info = zpsi.getinfo(member+'.npy')
                identity = dict(file=path.name, input_sha256=io.digest(path),
                                psi_crc=info.CRC, psi_member_size=info.file_size)
                chunk = directory/f'shot_{i:04d}.npz'
                if i < len(state['shots']):
                    shot = state['shots'][i]
                    if (shot['index'] != i or shot['source'] != identity
                            or not chunk.exists() or io.digest(chunk) != shot['sha256']):
                        raise ValueError(f'Resume shot/chunk identity mismatch: {i}')
                    continue
                # A completed chunk without a manifest entry can result from interruption.
                # It is recomputed from the immutable inputs; no other shot is overwritten.
                row = io.read_mast(path)
                times, R, Z = [np.asarray(row[k]) for k in ['efit_times', 'efit_grid_R', 'efit_grid_Z']]
                psi = archive[member]  # exactly one shot, never all 1205 flux arrays
                if psi.shape != (len(times), len(Z), len(R)):
                    raise ValueError('final6 psi/input frame mismatch')
                n = len(times)
                F, ip = np.full(n, np.nan), np.full(n, np.nan)
                current_errors = {}
                for field, dest in [('magnetics_tf_current', F), ('magnetics_plasma_current', ip)]:
                    try:
                        vals = io.interp(row['magnetics_time'], row[field], times)
                        dest[:] = vacuum_F(vals/1e3, TF_TURNS['MAST']) if field.endswith('tf_current') else vals
                    except (ValueError, TypeError, IndexError) as exc:
                        current_errors[field] = str(exc)
                try:
                    profiles = io.mast_profiles(row)
                    profile_error = None
                except (ValueError, TypeError, IndexError) as exc:
                    profiles = None
                    profile_error = str(exc)
                q, b = np.full(n, np.nan), np.full(n, np.nan)
                for j, t in enumerate(times):
                    if j % 16 == 0:
                        io.resource_guard(args.no_guard)
                    # final6's predictor uses DIII-D j1 alpha_F < 0 (predict.py:109;
                    # diag/calib_joint_result.json). Pin that input convention via
                    # the scorer's axis_sign interface. Native MAST's +1 would
                    # pick coil-region maxima before trying the physical minimum.
                    # This convention comes from the producer, not MAST labels.
                    geom = flux_geometry(psi[j], R, Z, 'MAST', axis_sign=args.axis_sign)
                    q[j] = q_at(psi[j], R, Z, F[j], machine='MAST', geometry=geom)
                    if profiles is not None:
                        ts, te, ne, sr, sz = profiles
                        k = io.nearest_index(ts, t)
                        pe = volume_average_pressure(psi[j], R, Z, sr, sz, te[k], ne[k],
                                                     'MAST', geometry=geom)
                        b[j] = beta_n(cp*pe, geom['a'], abs(F[j])/geom['R_geo'], abs(ip[j]))
                # Non-representable outputs are numerical failures too. Keep the
                # failure flags and use the same estimate-only median policy.
                overflow = {}
                for field, values in [('q95', q), ('betaN', b)]:
                    bad = np.isfinite(values) & (np.abs(values) > np.finfo(np.float16).max)
                    overflow[field] = int(bad.sum())
                    values[bad] = np.nan
                arrays = dict(raw_q95=q, raw_betaN=b, q95=q, betaN=b,
                              failed_q95=~np.isfinite(q), failed_betaN=~np.isfinite(b))
                save_chunk(chunk, arrays)
                shot = dict(index=i, source=identity, chunk=chunk.name, frames=n,
                            sha256=io.digest(chunk), profile_error=profile_error,
                            current_errors=current_errors, float16_overflow=overflow,
                            raw_q95=io.finite_summary(q),
                            raw_betaN=io.finite_summary(b),
                            failed_q95=int(np.sum(~np.isfinite(q))),
                            failed_betaN=int(np.sum(~np.isfinite(b))))
                state['shots'].append(shot)
                state['status'] = 'running'
                io.json_write(manifest, state)
                print(json.dumps(shot), flush=True)
                del row, psi, profiles, arrays
                gc.collect()
        if state['status'] in ('complete', 'smoke_complete'):
            print('All selected shots already complete; verified resume identities.', flush=True)
        elif finish_chunks(directory, state, args.no_guard):
            state['status'] = 'smoke_complete' if args.limit_shots else 'complete'
            state['production_complete'] = not args.limit_shots and len(state['shots']) == 1205
    except io.ResourceStop as exc:
        state.update(status='stopped_resource', stop_reason=str(exc))
        io.json_write(manifest, state)
        print(str(exc), file=sys.stderr)
        return 2
    io.json_write(manifest, state)
    print(json.dumps({k: state.get(k) for k in ['status', 'global_success_medians', 'fill_counts']}), flush=True)
    return 0 if state['status'] in ('complete', 'smoke_complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
