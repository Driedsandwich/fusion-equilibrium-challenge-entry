#!/usr/bin/env python3
"""Copy original compressed psi members and replace only q95/betaN; offline QA."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'mast'))
import qb_io as io
import argparse
import copy
import hashlib
import json
import subprocess
import zipfile
import numpy as np

D3_SHA256 = 'b2af65a4244ede7cd5f3c8083dc17e4d708143b2295838eaf74be30f1a717702'


def raw_member_copy(source, dest, member):
    """Copy local header + compressed payload + optional descriptor, byte for byte.

    zipfile has no public raw-copy API. Only its central-directory bookkeeping
    is updated here; no numpy decoding or recompression is used for this copy.
    """
    original = source.getinfo(member)
    offsets = [i.header_offset for i in source.infolist() if i.header_offset > original.header_offset]
    end = min(offsets) if offsets else source.start_dir
    info = copy.copy(original)
    dest.fp.seek(dest.start_dir)
    info.header_offset = dest.fp.tell()
    source.fp.seek(original.header_offset)
    remaining = end-original.header_offset
    while remaining:
        block = source.fp.read(min(1024*1024, remaining))
        if not block:
            raise ValueError('Truncated source ZIP member')
        dest.fp.write(block)
        remaining -= len(block)
    dest.start_dir = dest.fp.tell()
    dest.filelist.append(info)
    dest.NameToInfo[info.filename] = info
    dest._didModify = True


def store_array(dest, key, array):
    with dest.open(key+'.npy', 'w', force_zip64=True) as f:
        np.lib.format.write_array(f, array, allow_pickle=False)


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def assemble_mast(source_path, output, directory, state, count, no_guard):
    shots = state['shots']
    if len(shots) != count or [s['index'] for s in shots] != list(range(count)):
        raise ValueError('Require exactly the selected contiguous shot chunks')
    if output.exists():
        raise FileExistsError(f'Output already exists; use a fresh --out-dir: {output}')
    partial = output.with_name(output.name+'.partial')
    with zipfile.ZipFile(source_path) as src, zipfile.ZipFile(partial, 'x', allowZip64=True,
                                                        compression=zipfile.ZIP_DEFLATED) as dst:
        for i, shot in enumerate(shots):
            io.resource_guard(no_guard)
            chunk = directory/shot['chunk']
            if io.digest(chunk) != shot['sha256']:
                raise ValueError('Chunk digest mismatch')
            prefix = f'shot_{i:04d}_'
            raw_member_copy(src, dst, prefix+'psirz.npy')
            with np.load(chunk, allow_pickle=False) as z:
                for field in ['q95', 'betaN']:
                    a = z[field]
                    if a.dtype != np.float16 or a.shape != (shot['frames'],) or not np.isfinite(a).all():
                        raise ValueError(f'Invalid generated {field} for shot {i}')
                    store_array(dst, prefix+field, a)
    partial.replace(output)
    matches, hashes = 0, []
    with np.load(source_path, allow_pickle=False) as src, np.load(output, allow_pickle=False) as dst:
        for i, shot in enumerate(shots):
            io.resource_guard(no_guard)
            key = f'shot_{i:04d}_psirz'
            a, b = src[key], dst[key]
            h1, h2 = array_digest(a), array_digest(b)
            same = a.dtype == b.dtype and a.shape == b.shape and h1 == h2
            if not same:
                raise ValueError(f'Psi array changed: {key}')
            matches += 1
            hashes.append(dict(index=i, sha256=h1))
    return matches, hashes


def clone_d3d(source, target):
    """Required APFS clone, then verify the exact submission-1 file digest."""
    if target.exists():
        raise FileExistsError(target)
    subprocess.run(['cp', '-c', str(source), str(target)], check=True)
    actual = io.digest(target)
    if actual != D3_SHA256:
        raise ValueError(f'DIII-D reference digest mismatch: {actual}')
    return actual


def smoke_d3d_prefix(source, output, count):
    """Two-shot packaging fixture only; the verified full clone is retained too."""
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(output, 'x', allowZip64=True) as dst:
        for i in range(count):
            for field in ['psirz', 'q95', 'betaN']:
                raw_member_copy(src, dst, f'shot_{i:04d}_{field}.npy')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit-shots', type=int, default=0)
    ap.add_argument('--no-guard', action='store_true', help='smoke runs only')
    ap.add_argument('--qb-dir', help='default: _work/final7/qb[/smokeN]')
    ap.add_argument('--out-dir', help='default: _work/final7/out or _work/final7/smokeN/out')
    args = ap.parse_args(argv)
    if args.limit_shots < 0 or (args.no_guard and not args.limit_shots):
        ap.error('--no-guard requires a positive smoke limit')
    directory = io.workspace_path(args.qb_dir or ('_work/final7/qb'+
                                  (f'/smoke{args.limit_shots}' if args.limit_shots else '')))
    default_out = f'_work/final7/smoke{args.limit_shots}/out' if args.limit_shots else '_work/final7/out'
    out = io.workspace_path(args.out_dir or default_out, output=True)
    out.mkdir(parents=True, exist_ok=True)
    state = json.loads((directory/'manifest.json').read_text())
    count = args.limit_shots or 1205
    if state['status'] != ('smoke_complete' if args.limit_shots else 'complete'):
        raise ValueError('Generation is not complete for this scope')
    if state['signature']['limit_shots'] != args.limit_shots:
        raise ValueError('Generation/assembly scope mismatch')
    if not args.limit_shots and not state['production_complete']:
        raise ValueError('Full 1205-shot generation is required')
    source = ROOT/'_work/final6/out/mast_private_test.npz'
    if (source.stat().st_size != state['signature']['psi_size']
            or source.stat().st_mtime_ns != state['signature']['psi_mtime_ns']):
        raise ValueError('final6 psi identity changed')
    report = dict(status='running', smoke=bool(args.limit_shots), expected_shots=count,
                  guard_disabled=args.no_guard)
    try:
        io.resource_guard(args.no_guard)
        mast_out = out/'mast_private_test.npz'
        matches, hashes = assemble_mast(source, mast_out, directory, state, count, args.no_guard)
        report.update(psi_sha256_matches=matches, psi_array_sha256=hashes)
        io.json_write(out/'assembly_manifest.json', report)
        io.resource_guard(args.no_guard)
        d3_out = out/'diii_d_private_test.npz'
        full_clone = out/'diii_d_reference_clone.npz' if args.limit_shots else d3_out
        report['diii_d_reference_sha256'] = clone_d3d(ROOT/'_work/final6/out/diii_d_private_test.npz', full_clone)
        if args.limit_shots:
            smoke_d3d_prefix(full_clone, d3_out, count)
            report['diii_d_scope'] = 'prefix fixture; full verified cp -c clone retained separately'
        else:
            report['diii_d_scope'] = 'unchanged full reference file'
        command = [sys.executable, '-B', str(ROOT/'make_zip.py'), '--diii-d', str(d3_out),
                   '--mast', str(mast_out), '--output', str(out/'submission.zip')]
        if args.no_guard:
            command.append('--no-guard')
        io.resource_guard(args.no_guard)
        subprocess.run(command, cwd=ROOT, check=True)
        command = [sys.executable, '-B', str(ROOT/'validate.py'), str(mast_out),
                   '--config', 'mast_private_test', '--machine', 'mast',
                   '--local-dir', str(ROOT/'_data/final/mast_private_test')]
        if args.limit_shots:
            command += ['--limit-shots', str(args.limit_shots)]
        io.resource_guard(args.no_guard)
        subprocess.run(command, cwd=ROOT, check=True)
        report.update(status='smoke_complete' if args.limit_shots else 'complete',
                      mast_validation=True, submission_sha256=io.digest(out/'submission.zip'))
    except io.ResourceStop as exc:
        report.update(status='stopped_resource', stop_reason=str(exc))
        io.json_write(out/'assembly_manifest.json', report)
        print(str(exc), file=sys.stderr)
        return 2
    io.json_write(out/'assembly_manifest.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'psi_array_sha256'}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
