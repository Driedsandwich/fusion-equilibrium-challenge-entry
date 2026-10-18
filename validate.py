#!/usr/bin/env python3
"""Submission validator adapted from starter_ref/validate_submission.py (MIT).

Projects reference inputs to efit_times; never reads MAST equilibrium truth.
Requires float16, exact shot keys and every frame, without loading the whole NPZ.
"""
from __future__ import annotations
import submission_io as io
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('npz', type=io.Path)
    io.source_arguments(ap)
    args = ap.parse_args()
    machine = io.machine_of(args)
    expected = set()
    count = 0
    nonfinite_mast = 0
    with io.zipfile.ZipFile(args.npz) as z:
        members = z.namelist()
        if len(members) != len(set(members)):
            raise ValueError('Duplicate ZIP members')
    with np.load(args.npz, allow_pickle=False) as archive:
        for i, _, row in io.source_rows(args, ['efit_times']):
            count += 1
            frames = len(row['efit_times'])
            arrays = {}
            for field in io.FIELDS:
                key = f'shot_{i:04d}_{field}'
                expected.add(key+'.npy')
                if key not in archive.files:
                    raise ValueError(f'Missing {key}')
                arrays[field] = archive[key]
            io.check_arrays(arrays, frames, machine)
            if machine == 'mast':
                nonfinite_mast += sum(int((~np.isfinite(a)).sum()) for a in arrays.values())
            del arrays, row
    if count < 1 or set(members) != expected:
        raise ValueError(f'Shot count/name mismatch: expected {count} shots/{len(expected)} arrays, got {len(members)} members')
    print(f'OK: {args.npz}, {count} shots, exact names/shapes/T/float16; MAST nonfinite={nonfinite_mast}')

if __name__ == '__main__':
    main()
