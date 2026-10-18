"""Shared sequential I/O for submission entrypoints (no network on local paths)."""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode = True
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import json
import hashlib
import subprocess
import zipfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'base'), str(ROOT/'diag'), str(ROOT/'mast')]
from capacity import read_columns, swap_reading

FIELDS = ('psirz', 'q95', 'betaN')

def output_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to(ROOT) or any(path.is_relative_to(ROOT/d) for d in ('_data', 'starter_ref', '.git', '.venv')):
        raise ValueError(f'Output must be inside workspace, outside protected inputs: {path}')
    return path

def input_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Input must be inside workspace: {path}')
    return path

def sha256(path):
    with input_path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

# Override inherited settings before either Hub library can be imported.
for _key, _suffix in {
    'HF_HOME': '', 'HF_HUB_CACHE': 'hub', 'HF_DATASETS_CACHE': 'datasets',
    'HF_MODULES_CACHE': 'modules', 'HF_XET_CACHE': 'xet',
}.items():
    os.environ[_key] = str(output_path(Path('_work/hf_home')/_suffix))

def guard(disabled=False):
    if disabled:
        return
    swap = swap_reading()
    df = subprocess.run(['df', '-k', str(ROOT)], capture_output=True, text=True, timeout=5)
    try:
        free = int(df.stdout.splitlines()[-1].split()[3])*1024
    except (IndexError, ValueError):
        free = None
    if not swap['ok'] or df.returncode or free is None:
        raise RuntimeError('Resource reading unavailable; stopping before next shot')
    if swap['free_mb'] < 300 or free < 12*1024**3:
        raise RuntimeError(f'Resource guard: swap free={swap["free_mb"]} MB, disk free={free} bytes')

def source_arguments(parser):
    parser.add_argument('--repo', default='Sophelio/fusion-equilibrium-challenge')
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', default='public_test')
    parser.add_argument('--revision', help='dataset revision; use a fixed commit SHA for reproduction')
    parser.add_argument('--machine', choices=('diii_d', 'mast'), help='required for names without a machine prefix')
    parser.add_argument('--local-dir', type=Path, help='offline: sorted *.parquet, one row per file')
    parser.add_argument('--limit-shots', type=int, default=0, help='offline smoke subset, 0=all')

def machine_of(args):
    if args.machine:
        return args.machine
    if args.config.startswith('diii_d'):
        return 'diii_d'
    if args.config.startswith('mast'):
        return 'mast'
    raise ValueError('Unknown machine: pass --machine diii_d or mast')

def source_rows(args, columns, no_guard=True):
    """Yield (stream index, identity, projected row); never shuffle the Hub stream."""
    if args.limit_shots < 0:
        raise ValueError('--limit-shots must be nonnegative')
    if args.local_dir:
        directory = input_path(args.local_dir)
        if not directory.is_relative_to(ROOT):
            raise ValueError('Local input must be in workspace')
        paths = sorted(directory.glob('*.parquet'))
        if args.limit_shots:
            paths = paths[:args.limit_shots]
        if not paths:
            raise ValueError(f'No parquet shots in {directory}')
        for i, path in enumerate(paths):
            guard(no_guard)
            names = pq.ParquetFile(path).schema_arrow.names
            if machine_of(args) == 'mast' and any(n.startswith('efit_') and n not in ('efit_times', 'efit_grid_R', 'efit_grid_Z') for n in names):
                raise ValueError('Refuse MAST equilibrium label columns')
            selected = [c for c in columns if c in names]
            yield i, path.name, read_columns(path, selected)
    else:
        if args.limit_shots:
            raise ValueError('--limit-shots is only for offline tests')
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError('Install datasets before using the Hugging Face streaming path') from exc
        guard(no_guard)
        cache = output_path(os.environ['HF_DATASETS_CACHE'])
        ds = load_dataset(args.repo, args.config, split=args.split, revision=args.revision,
                          streaming=True, cache_dir=str(cache))
        # Project before iteration: validator reads only times, MAST never decodes EFIT targets.
        names = ds.column_names
        selected = [c for c in columns if names is None or c in names]
        ds = ds.select_columns(selected)
        iterator = iter(ds)
        i = 0
        while True:
            guard(no_guard)
            try:
                raw = next(iterator)
            except StopIteration:
                break
            row = {k: np.asarray(v) for k, v in raw.items() if v is not None}
            yield i, f'{args.repo}/{args.config}/{args.split}:{i}', row
            del raw, row
            i += 1

def check_arrays(arrays, frames, machine, require_all_finite=False):
    if set(arrays) != set(FIELDS):
        raise ValueError('Expected exactly psirz, q95, betaN')
    if frames < 1:
        raise ValueError('Empty shot')
    for field in FIELDS:
        arr = arrays[field]
        expected = (frames, 65, 65) if field == 'psirz' else (frames,)
        if arr.shape != expected or arr.dtype != np.float16:
            raise ValueError(f'{field}: got {arr.shape}/{arr.dtype}, expected {expected}/float16')
        if (machine == 'diii_d' or require_all_finite) and not np.isfinite(arr).all():
            raise ValueError(f'{field}: contains NaN/Inf')

def store_member(archive, key, array):
    with archive.open(key+'.npy', 'w', force_zip64=True) as member:
        np.lib.format.write_array(member, array, allow_pickle=False)

def write_json(path, value):
    output_path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
