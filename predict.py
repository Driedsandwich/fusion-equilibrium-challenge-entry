#!/usr/bin/env python3
"""Sequential, resumable submission inference and bounded-memory NPZ assembly."""
from __future__ import annotations
import submission_io as io
import argparse
import gc
import json
import re
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from features_v2 import INPUT_COLUMNS, build_features, feature_names

def assemble(work, output, machine, no_guard=False, config=None):
    manifest = json.loads((work/'manifest.json').read_text())
    if manifest['source']['machine'] != machine or manifest['source']['config'] != config:
        raise ValueError('Work directory machine/config mismatch')
    count = manifest.get('shot_count')
    if count is None or count < 1:
        raise ValueError('Source was not exhausted: run inference through the final shot first')
    expected = [f'shot_{i:04d}.npz' for i in range(count)]
    actual = sorted(p.name for p in work.glob('shot_*.npz'))
    if set(actual) != set(expected):
        raise ValueError('Missing or extra shot chunks; finish all ranges before assembly')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing finished submissions.
    if output.exists():
        raise FileExistsError(f'Output exists; choose a new --output: {output}')
    temporary = output.with_suffix(output.suffix+'.partial')
    with io.zipfile.ZipFile(temporary, 'w', compression=io.zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for i, name in enumerate(expected):
            io.guard(no_guard)
            with np.load(work/name, allow_pickle=False) as chunk:
                frames = int(chunk['frames'])
                arrays = {f: chunk[f] for f in io.FIELDS}
                io.check_arrays(arrays, frames, machine, True)
                for field, arr in arrays.items():
                    io.store_member(archive, f'shot_{i:04d}_{field}', arr)
            del arrays, arr
    temporary.replace(output)
    print(f'assembled {count} shots: {output}', flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    io.source_arguments(ap)
    ap.add_argument('--weights', default='weights')
    ap.add_argument('--work-dir')
    ap.add_argument('--output', help='default: <machine>_private_test.npz')
    ap.add_argument('--start', type=int, default=0, help='inclusive stream shot index')
    ap.add_argument('--stop', type=int, help='exclusive stream shot index')
    ap.add_argument('--assemble', action='store_true', help='assemble only; no dataset access')
    ap.add_argument('--no-guard', action='store_true', help='offline tests in restricted sandbox only')
    args = ap.parse_args()
    machine = io.machine_of(args)
    if args.start < 0 or (args.stop is not None and args.stop < args.start):
        ap.error('Require 0 <= start <= stop')
    if args.no_guard and not args.local_dir and not args.assemble:
        ap.error('--no-guard is only permitted with --local-dir')
    work = io.output_path(args.work_dir or '_work/'+re.sub(r'[^A-Za-z0-9_.-]', '_', args.config))
    work.mkdir(parents=True, exist_ok=True)
    if args.assemble:
        assemble(work, io.output_path(args.output or machine+'_private_test.npz'), machine, args.no_guard, args.config)
        return
    identity = dict(repo=args.repo, config=args.config, split=args.split, machine=machine,
        local_dir=str(args.local_dir.resolve()) if args.local_dir else None, limit_shots=args.limit_shots,
        weights=str(io.Path(args.weights).resolve()) if machine == 'diii_d' else None,
        mast_policy='j1_alpha;c=0;solenoid=1')
    mp = work/'manifest.json'
    manifest = json.loads(mp.read_text()) if mp.exists() else dict(source=identity)
    if manifest['source'] != identity:
        raise ValueError('Work directory belongs to a different source/model; choose another --work-dir')
    io.write_json(mp, manifest)
    if machine == 'diii_d':
        model = joblib.load(io.Path(args.weights)/'model.joblib')
        if model['features'] != feature_names() or model['variant'] != 'd_hist20':
            raise ValueError('Incompatible model/features')
        columns = INPUT_COLUMNS
    else:
        import simple
        calibration = json.loads((io.ROOT/'diag/calib_joint_result.json').read_text())
        calibration['convention']['alpha_F'] = calibration['results']['j1']['alpha_F']
        calibration['convention']['c_for_mast'] = 0.
        columns = simple.INPUTS
    total = 0
    exhausted = True
    with threadpool_limits(limits=1):
        for index, source, row in io.source_rows(args, columns, args.no_guard):
            total = index+1
            if args.stop is not None and index >= args.stop:
                exhausted = False
                break
            if index < args.start:
                del row
                continue
            dst = work/f'shot_{index:04d}.npz'
            frames = len(row['efit_times'])
            if dst.exists():
                with np.load(dst, allow_pickle=False) as chunk:
                    if str(chunk['source']) != source or int(chunk['frames']) != frames:
                        raise ValueError(f'Existing chunk identity mismatch: {dst}')
                    io.check_arrays({f: chunk[f] for f in io.FIELDS}, frames, machine, True)
                print(f'skip shot_{index:04d}', flush=True)
                del row
                continue
            if machine == 'diii_d':
                x = model['scaler'].transform(build_features(row, row['efit_times']))
                coeff = np.column_stack([m.predict(x) for m in model['coefficients']])
                psi = model['pca'].inverse_transform(coeff).reshape(frames, 65, 65)
                arrays = dict(psirz=psi.astype(np.float16), q95=model['scalars'][0].predict(x).astype(np.float16),
                              betaN=model['scalars'][1].predict(x).astype(np.float16))
                del x, coeff, psi
            else:
                g, _ = simple.coil_groups(row)
                predictions, _ = simple.predict(row, calibration, g)
                arrays = predictions[1.]
                del predictions, g
            io.check_arrays(arrays, frames, machine, True)
            # A complete chunk becomes visible only after close; interrupted partial files are not skipped.
            temporary = dst.with_suffix('.partial')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **arrays, frames=np.int64(frames), source=np.asarray(source))
            temporary.replace(dst)
            print(f'predicted shot_{index:04d}: {frames} frames', flush=True)
            del row, arrays
            gc.collect()
    if exhausted:
        if total < 1:
            raise ValueError('Empty source')
        manifest['shot_count'] = total
        io.write_json(mp, manifest)

if __name__ == '__main__':
    main()
