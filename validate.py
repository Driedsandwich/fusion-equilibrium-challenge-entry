#!/usr/bin/env python3
"""Submission validator adapted from starter_ref/validate_submission.py (MIT).

Projects reference inputs to efit_times; never reads MAST equilibrium truth.
Requires float16, exact shot keys and every frame, without loading the whole NPZ.
"""
from __future__ import annotations
import submission_io as io
import argparse
import re
import numpy as np

def validate_archive(path, machine, rows=None):
    """Shared structural check; optional rows also verify dataset shot count/T."""
    path = io.input_path(path)
    with io.zipfile.ZipFile(path) as z:
        members = z.namelist()
        if len(members) != len(set(members)):
            raise ValueError('Duplicate ZIP members')
    ids = set()
    for member in members:
        match = re.fullmatch(r'shot_(\d{4,})_(psirz|q95|betaN)\.npy', member)
        if not match:
            raise ValueError(f'Unexpected NPZ member: {member}')
        ids.add(int(match[1]))
    if not ids or ids != set(range(len(ids))):
        raise ValueError('Expected nonempty contiguous shots starting at zero')
    expected = {f'shot_{i:04d}_{f}.npy' for i in ids for f in io.FIELDS}
    if set(members) != expected:
        raise ValueError('Missing or extra NPZ keys')
    count = 0
    with np.load(path, allow_pickle=False) as archive:
        if rows is None:
            rows = ((i, None, None) for i in range(len(ids)))
        for i, _, row in rows:
            arrays = {f: archive[f'shot_{i:04d}_{f}'] for f in io.FIELDS}
            frames = len(row['efit_times']) if row is not None else arrays['q95'].shape[0] if arrays['q95'].ndim == 1 else 0
            io.check_arrays(arrays, frames, machine, True)
            count += 1
        if count != len(ids):
            raise ValueError('Dataset/NPZ shot count mismatch')
    return count

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('npz', type=io.Path)
    io.source_arguments(ap)
    args = ap.parse_args()
    machine = io.machine_of(args)
    count = validate_archive(args.npz, machine, io.source_rows(args, ['efit_times']))
    print(f'OK: {args.npz}, {count} shots, exact names/shapes/T/float16; all finite')

if __name__ == '__main__':
    main()
