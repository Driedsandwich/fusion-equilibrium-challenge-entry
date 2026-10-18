# fusion-equilibrium-entry

Entry for the Fusion Equilibrium Challenge (NeurIPS 2026). Work in progress.

- `base/` — DIII-D model: PCA of the flux map, with the PCA coefficients and the two scalars (q95, betaN) regressed from non-magnetic inputs (coil currents, Ip, Thomson summaries) by gradient-boosted trees or a small MLP.
- `diag/` — diagnostics, including a check of how well a coil-plus-plasma-filament Green's-function representation can fit DIII-D flux maps.
- `starter_ref/` — unmodified files from the official starter kit (MIT, see `starter_ref/LICENSE-starter-MIT`), including the official scorer `fusion_scoring/`.

Data is not included. It is the public challenge dataset on Hugging Face (`Sophelio/fusion-equilibrium-challenge`, CC BY 4.0).

## Environment and working directory

Run all commands from the repository root. Relative inputs and outputs resolve from that root. Use `.venv/bin/python`.

```bash
uv venv --python 3.12.13
uv pip install -r requirements-lock.txt
```

The tools override inherited `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `HF_MODULES_CACHE` and `HF_XET_CACHE` before importing Hub libraries. All five locations are under `_work/hf_home/`. macOS reads swap via `sysctl`; Linux reads `SwapFree` from `/proc/meminfo`. Unavailable readings stop execution. The guard requires 300 MB free swap and 12 GiB free disk. Training, local prediction, assembly and packaging support `--no-guard` for controlled reproduction.

## Rebuild both models with one command

Set `TRAIN_REVISION` to a fixed dataset commit SHA. Use a fresh output directory. Existing `_data/diii_d_train` files are reused by `fetch_train.py`; their skip check verifies Parquet row count, not remote content identity. For a fresh download, use a new workspace copy without `_data/`.

```bash
export TRAIN_REVISION='REPLACE_WITH_FIXED_DATASET_COMMIT_SHA'
sh -eu -c '
  .venv/bin/python -B fetch_train.py --revision "$TRAIN_REVISION"
  .venv/bin/python -B train.py --weights _work/rebuilt/weights
  .venv/bin/python -B diag/calib_joint.py --rebuild-capacity-input --capacity-result _work/rebuilt/capacity_input.json
'
```

`train.py` refuses to overwrite an existing model. The separate `--weights` output preserves bundled `weights/`. Calibration writes `diag/calib_joint_result.json`, `diag/joint_locals/` and its report; finish calibration before starting a new prediction run. To use rebuilt DIII-D weights, set `WEIGHTS=_work/rebuilt/weights` below. Full training and calibration also accept `--no-guard`; the default remains fail closed.

`diag/capacity_result_run2.json` is the archived second capacity diagnostic on the fixed twelve DIII-D shots listed in `diag/calib_joint.py`. Its settings were stride 4, quadrature order 6, relative ridge 1e-10 and a 520-second budget; the diagnostic took 44.948 seconds. Calibration consumes only its shot identities and `geometry.box_m`. The box is the union of stored LCFS contours over all 2,342 frames of those twelve shots, padded by 0.05 m on each side. No MAST equilibrium enters this calculation.

The command above regenerates these consumed fields into `_work/rebuilt/capacity_input.json` and uses that input for calibration. It does not regenerate the historical scoring report. `--capacity-result` also accepts an existing input. Running `diag/capacity.py` uses all training files currently present, so with 312 files it does not recreate the historical twelve-shot report.

Recorded computation: `weights/train_meta.json` reports 312 training shots, 101.226 seconds, macOS 27.0.1 arm64, 10 logical CPUs, numerical computation limited to one thread, Python 3.12.13. `diag/calib_joint_result.json` reports 4.569 seconds for calibration (474 sampled frames, stride 5). These are separate timings and exclude environment setup/downloads. Runtime on other machines may differ.

## Regenerate both NPZ files and the ZIP with one command

Set repo, fixed dataset revision, both config names and split to the organizer's actual values. The development names below must be replaced with the final-phase names when announced. Use a fresh `RUN_DIR`.

```bash
export DATASET_REPO='Sophelio/fusion-equilibrium-challenge'
export DATASET_REVISION='REPLACE_WITH_FIXED_DATASET_COMMIT_SHA'
export DATASET_SPLIT='public_test'
export D3_CONFIG='diii_d_public_test'
export MAST_CONFIG='mast_public_test'
export RUN_DIR='_work/reproduced'
export WEIGHTS='weights'
sh -eu -c '
  for machine in diii_d mast; do
    config="$D3_CONFIG"
    if [ "$machine" = mast ]; then config="$MAST_CONFIG"; fi
    .venv/bin/python -B predict.py --repo "$DATASET_REPO" --revision "$DATASET_REVISION" --config "$config" --split "$DATASET_SPLIT" --machine "$machine" --weights "$WEIGHTS" --work-dir "$RUN_DIR/$machine"
    .venv/bin/python -B predict.py --repo "$DATASET_REPO" --revision "$DATASET_REVISION" --config "$config" --split "$DATASET_SPLIT" --machine "$machine" --weights "$WEIGHTS" --work-dir "$RUN_DIR/$machine" --assemble --output "$RUN_DIR/${machine}_private_test.npz"
    .venv/bin/python -B validate.py "$RUN_DIR/${machine}_private_test.npz" --repo "$DATASET_REPO" --revision "$DATASET_REVISION" --config "$config" --split "$DATASET_SPLIT" --machine "$machine"
  done
  .venv/bin/python -B make_zip.py --diii-d "$RUN_DIR/diii_d_private_test.npz" --mast "$RUN_DIR/mast_private_test.npz" --output "$RUN_DIR/submission.zip"
'
```

`manifest.json` binds repo/config/split/revision, local input hashes when applicable, weights and MAST calibration SHA256, inference Python-file SHA256 values, and each shot's source and frame count. Chunks carry the run signature. Resume and assembly reject mismatches. Assembly requires the same source/model arguments, including `--local-dir` and `--limit-shots` for local tests. Old manifests require a new work directory. Interrupted `.partial` files are retained; select a new work directory/output if one blocks exclusive creation. A completed source scan is required before assembly.

Packaging uses the same NPZ checker as `validate.py` for exact keys, contiguous shot numbering, shapes, float16 and finite values. `validate.py` additionally checks dataset shot/frame counts. ZIP root members are exactly `diii_d_private_test.npz` and `mast_private_test.npz`, stored without recompression.

Local two-shot, development and production commands are in `SUBMISSION_COMMANDS.md`. No upload is performed. Before final delivery, the publisher must identify the fixed source commit or a versioned source archive and its SHA256; this repair does not publish a source version. See `DATA_PROVENANCE.md` for per-machine data use.
