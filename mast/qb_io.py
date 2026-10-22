"""Local-only helpers shared by the q95/betaN command-line tools."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'diag'), str(ROOT/'starter_ref'), str(ROOT/'mast')]
from capacity import read_columns, swap_reading
from data_fixes import align_mast_thomson_core
from simple import interp, _repair_thomson
import numpy as np
import pyarrow.parquet as pq

MAST_COLUMNS = ['efit_times', 'efit_grid_R', 'efit_grid_Z', 'magnetics_time',
                'magnetics_tf_current', 'magnetics_plasma_current', 'thomson_core_times',
                'thomson_core_R', 'thomson_core_Te', 'thomson_core_ne']


class ResourceStop(RuntimeError):
    pass


def workspace_path(value, output=False):
    p = Path(value)
    p = (p if p.is_absolute() else ROOT/p).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f'Outside workspace: {p}')
    if output and any(p.is_relative_to(ROOT/d) for d in
                      ['_data', 'weights', 'starter_ref', '.git', '.venv']+
                      [f'_work/final{i}' for i in range(2, 7)]):
        raise ValueError(f'Protected output path: {p}')
    return p


def digest(path):
    with workspace_path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def json_write(path, data):
    p = workspace_path(path, output=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.partial')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def resource_guard(disabled=False):
    if disabled:
        return
    mem = swap_reading()
    if not mem['ok'] or mem['free_mb'] < 300:
        raise ResourceStop('Swap guard: '+json.dumps(mem, ensure_ascii=False))


def read_mast(path):
    names = pq.ParquetFile(workspace_path(path)).schema_arrow.names
    if any(n.startswith('efit_') and n not in ('efit_times', 'efit_grid_R', 'efit_grid_Z')
           for n in names):
        raise ValueError('Refuse MAST equilibrium label columns')
    return read_columns(path, MAST_COLUMNS)


def repair_profiles(times, Te, ne, R, Z):
    """One shared row selection for Te/ne; same malformed-time repair as simple.py."""
    times, R, Z = [np.asarray(v, float) for v in (times, R, Z)]
    Te, ne = np.atleast_2d(Te), np.atleast_2d(ne)
    n = min(len(times), Te.shape[0], ne.shape[0])
    m = min(len(R), len(Z), Te.shape[1], ne.shape[1])
    if not n or not m:
        raise ValueError('Empty Thomson acquisition')
    # Repair a row-index column so both measured quantities retain identical times.
    index, times, _ = _repair_thomson(np.arange(n, dtype=float)[:, None],
                                     times[:n], np.array([0.0]))
    idx = index[:, 0].astype(int)
    if not len(times):
        raise ValueError('No finite Thomson timestamps')
    return times, Te[:n, :m][idx], ne[:n, :m][idx], R[:m], Z[:m]


def mast_profiles(row):
    R, te, ne = align_mast_thomson_core(row)
    return repair_profiles(row['thomson_core_times'], te, ne, R, np.zeros(len(R)))


def nearest_index(times, target):
    right = min(int(np.searchsorted(times, target)), len(times)-1)
    left = max(right-1, 0)
    return right if abs(times[right]-target) < abs(times[left]-target) else left


def finite_summary(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return dict(count=int(len(a)), min=float(a.min()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                max=float(a.max()) if len(a) else None)
