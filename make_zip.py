#!/usr/bin/env python3
"""Package exactly two prediction NPZ files at the ZIP root, without recompression."""
import submission_io as io
import argparse
from validate import validate_archive

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diii-d', default='diii_d_private_test.npz')
    ap.add_argument('--mast', default='mast_private_test.npz')
    ap.add_argument('--output', default='submission.zip')
    ap.add_argument('--no-guard', action='store_true', help='skip resource guard for local reproduction')
    args = ap.parse_args()
    paths = [io.input_path(args.diii_d), io.input_path(args.mast)]
    for path, machine in zip(paths, ('diii_d', 'mast')):
        validate_archive(path, machine)
    io.guard(args.no_guard)
    output = io.output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with io.zipfile.ZipFile(output, 'x', compression=io.zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path, name in zip(paths, ('diii_d_private_test.npz', 'mast_private_test.npz')):
            archive.write(path, arcname=name)
    with io.zipfile.ZipFile(output) as archive:
        if archive.namelist() != ['diii_d_private_test.npz', 'mast_private_test.npz'] or archive.testzip() is not None:
            raise ValueError('Submission ZIP verification failed')
    print(f'OK: {output}, two NPZ files at archive root')

if __name__ == '__main__':
    main()
