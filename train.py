#!/usr/bin/env python3
"""Train chosen DIII-D variant d on every manifest shot, without a holdout."""
from __future__ import annotations
import submission_io as io
import argparse
import gc
import platform
import time
from datetime import datetime, timezone
import joblib
import numpy as np
import pyarrow.parquet as pq
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from features_v2 import INPUT_COLUMNS, build_features, feature_names

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shots', default='train_shots.txt')
    ap.add_argument('--data-dir', default='_data/diii_d_train')
    ap.add_argument('--weights', default='weights')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit-shots', type=int, default=0, help='offline smoke only')
    ap.add_argument('--no-guard', action='store_true')
    args = ap.parse_args()
    if args.limit_shots < 0 or (args.no_guard and not 1 <= args.limit_shots <= 3):
        ap.error('--no-guard requires --limit-shots 1..3')
    names = [s.strip() for s in io.Path(args.shots).read_text().splitlines() if s.strip()]
    if len(names) != 312 or len(set(names)) != 312 or any(io.Path(n).name != n or not n.endswith('.parquet') for n in names):
        raise ValueError('Manifest must contain exactly 312 unique parquet basenames')
    if args.limit_shots:
        names = names[:args.limit_shots]
    paths = [io.Path(args.data_dir)/n for n in names]
    if not all(p.is_file() for p in paths):
        raise ValueError('Missing training files; run fetch_train.py first')
    out = io.output_path(args.weights)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'model.joblib').exists() or (out/'train_meta.json').exists():
        raise FileExistsError('Weights already exist; choose a new --weights directory')
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    pca_frames_per_shot = max(8, int(np.ceil(21/len(paths))))
    with threadpool_limits(limits=1):
        sample = []
        for path in paths:
            io.guard(args.no_guard)
            row = io.read_columns(path, ['efit_times', 'efit_psirz'])
            nt = len(row['efit_times'])
            idx = np.unique(np.linspace(0, nt-1, pca_frames_per_shot, dtype=int))
            sample.append(row['efit_psirz'][idx].reshape(len(idx), -1).astype(np.float32))
            del row
            gc.collect()
        sample = np.concatenate(sample)
        pca_sample_frames = len(sample)
        pca = PCA(n_components=20, svd_solver='randomized', random_state=args.seed, copy=False).fit(sample)
        del sample
        xp, cp, sp = [], [], []
        available = 0
        for path in paths:
            io.guard(args.no_guard)
            cols = INPUT_COLUMNS+['efit_psirz', 'efit_q95', 'efit_beta_n']
            present = pq.ParquetFile(path).schema_arrow.names
            row = io.read_columns(path, [c for c in cols if c in present])
            nt = len(row['efit_times']); available += nt
            idx = np.unique(np.linspace(0, nt-1, min(nt, 64), dtype=int))
            xp.append(build_features(row, row['efit_times'][idx]))
            cp.append(pca.transform(row['efit_psirz'][idx].reshape(len(idx), -1)))
            sp.append(np.column_stack((row['efit_q95'][idx], row['efit_beta_n'][idx])))
            print(f'train inputs: {path.name} ({len(idx)}/{nt} frames)', flush=True)
            del row
            gc.collect()
        x, c, scalars = np.concatenate(xp), np.concatenate(cp), np.concatenate(sp)
        del xp, cp, sp
        if not np.isfinite(c).all():
            raise ValueError('Nonfinite PCA training coefficients')
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)
        def fit(y, iterations):
            io.guard(args.no_guard)
            valid = np.isfinite(y)
            if not valid.any():
                raise ValueError('No finite training target')
            return HistGradientBoostingRegressor(max_iter=iterations, max_leaf_nodes=15,
                l2_regularization=1., early_stopping=False, random_state=args.seed).fit(x[valid], y[valid])
        coefficients = [fit(c[:, j], 60) for j in range(20)]
        scalars = [fit(scalars[:, j], 100) for j in range(2)]
        model = dict(format_version=1, variant='d_hist20', features=feature_names(),
                     pca=pca, scaler=scaler, coefficients=coefficients, scalars=scalars)
        joblib.dump(model, out/'model.joblib', compress=3)
    meta = dict(status='complete_smoke' if args.limit_shots else 'complete',
        seed=args.seed, train_shots=names, holdout_shots=[], sampled_train_frames=len(x),
        available_train_frames=available, pca_components=20,
        pca_frames_per_shot=pca_frames_per_shot, pca_sample_frames=pca_sample_frames,
        train_frames_per_shot=64, hist_iterations=60, scalar_iterations=100,
        features=feature_names(), started_utc=started, elapsed_seconds=time.monotonic()-start,
        machine=dict(system=platform.platform(), architecture=platform.machine(),
                     processor=platform.processor(), cpu_count=io.os.cpu_count(), threads=1),
        versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__))
    io.write_json(out/'train_meta.json', meta)
    print(f'{meta["status"]}: {len(names)} shots, {meta["elapsed_seconds"]:.2f} seconds, {out}')

if __name__ == '__main__':
    main()
