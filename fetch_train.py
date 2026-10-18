#!/usr/bin/env python3
"""Fetch only the 312 training shots named in train_shots.txt (organizer rebuild)."""
from __future__ import annotations
import submission_io as io
import argparse
import shutil

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', default='Sophelio/fusion-equilibrium-challenge')
    ap.add_argument('--revision', default='main')
    ap.add_argument('--shots', default='train_shots.txt')
    ap.add_argument('--remote-dir', default='data/diii_d_train')
    args = ap.parse_args()
    names = [s.strip() for s in io.input_path(args.shots).read_text().splitlines() if s.strip()]
    if len(names) != 312 or len(set(names)) != 312 or any(io.Path(n).name != n or not n.endswith('.parquet') for n in names):
        raise ValueError('Expected 312 unique parquet basenames')
    from huggingface_hub import hf_hub_download, list_repo_files
    remote = list_repo_files(args.repo, repo_type='dataset', revision=args.revision)
    selected = {}
    for name in names:
        matches = [p for p in remote if io.Path(p).name == name and p.startswith(args.remote_dir.rstrip('/')+'/')]
        if len(matches) != 1:
            raise ValueError(f'Expected one remote match for {name}: {matches}')
        selected[name] = matches[0]
    target = io.input_path('_data/diii_d_train')
    target.mkdir(parents=True, exist_ok=True)
    cache = io.output_path('_work/hf_fetch_cache')
    for name, remote_path in selected.items():
        dst = io.input_path(target/name)
        if dst.exists():
            if io.pq.ParquetFile(dst).metadata.num_rows != 1:
                raise ValueError(f'Invalid existing training shot: {dst}')
            print(f'skip {name}', flush=True)
            continue
        io.guard()
        src = hf_hub_download(args.repo, remote_path, repo_type='dataset', revision=args.revision, cache_dir=str(cache))
        stage = io.output_path('_work/fetch_stage')
        stage.mkdir(parents=True, exist_ok=True)
        temporary = io.output_path(stage/(name+'.partial'))
        with open(src, 'rb') as source, temporary.open('xb') as output:
            shutil.copyfileobj(source, output, length=1024*1024)
        if io.pq.ParquetFile(temporary).metadata.num_rows != 1:
            raise ValueError(f'Invalid downloaded training shot: {name}')
        if dst.exists():
            raise FileExistsError(f'Training destination appeared during fetch: {dst}')
        temporary.replace(dst)
        print(f'fetched {name}', flush=True)

if __name__ == '__main__':
    main()
