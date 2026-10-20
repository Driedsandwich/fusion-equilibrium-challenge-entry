# fusion-equilibrium-entry

Entry for the Fusion Equilibrium Challenge (NeurIPS 2026). Final-phase submission #1 was made on 19 October 2026 (UTC); see "Submission #1 record" below.

- `base/` — DIII-D model: PCA of the flux map, with the PCA coefficients and the two scalars (q95, betaN) regressed from non-magnetic inputs (coil currents, Ip, Thomson summaries) by gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`, used by `train.py` for the submitted weights). `base/train_eval_v2.py` also compares ridge and a small MLP on held-out shots.
- `diag/` — diagnostics, including a check of how well a coil-plus-plasma-filament Green's-function representation can fit DIII-D flux maps.
- `starter_ref/` — unmodified files from the official starter kit (MIT, see `starter_ref/LICENSE-starter-MIT`), including the official scorer `fusion_scoring/`.

Data is not included. Training uses the public challenge dataset on Hugging Face (`Sophelio/fusion-equilibrium-challenge`, CC BY 4.0). Final-phase test inputs come from `ganatma/fusion-equilibrium-challenge-final`.

## Submission #1 record

- Platform: Codabench competition 17739, Final phase, submission ID 1003290, created 2026-10-19 15:35 UTC, scored (status Finished).
- Code: commit `c337865f9e4810c47570d73aa0a4d5da4c9e4070`, tag `submission-1`. Check out this commit to reproduce submission #1.
- Environment: `requirements-lock.txt`, Python 3.12.13.
- Weights: `weights/model.joblib`, SHA-256 `03d504a862419f1c5d55266a38bffa67ed37d498f8e0c2650cf27990542f99b7`. MAST calibration: `diag/calib_joint_result.json`, SHA-256 `29498713233df2396c0e85f4e418aac151678b83e79af394922531391374cfab`. MAST policy: j1 alpha_F, c=0, solenoid multiplier 1.
- Training data: `Sophelio/fusion-equilibrium-challenge` at revision `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb`, the 312 files of `data/diii_d_train` named in `train_shots.txt`. On 2026-10-20 the SHA-256 of all 312 local files matched the Hub's LFS records at that revision.
- Test inputs: `ganatma/fusion-equilibrium-challenge-final` at revision `965040c12b814f33c8e7209f6b1799932247e793`, configs `diii_d_private_test` (879 files) and `mast_private_test` (1,205 files), split `private_test`. Submission #1 read local copies of these files with `--local-dir _data/final/<config>` and `--config <config>`. The run manifests store a SHA-256 for every input file. On 2026-10-20 all 2,084 values matched the Hub's LFS records at that revision.
- Output: `submission.zip`, 1,918,727,900 bytes, SHA-256 `5e753ee20c4aa03723ad395d0655b8972aed445747490b2dcbbabca499ffe1f3`. Its root holds only `diii_d_private_test.npz` and `mast_private_test.npz`.

## Environment and working directory

Run all commands from the repository root. Relative inputs and outputs resolve from that root. Use `.venv/bin/python`.

```bash
uv venv --python 3.12.13
uv pip install -r requirements-lock.txt
```

The tools override inherited `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `HF_MODULES_CACHE` and `HF_XET_CACHE` before importing Hub libraries. All five locations are under `_work/hf_home/`. macOS reads swap via `sysctl`; Linux reads `SwapFree` from `/proc/meminfo`. Unavailable readings stop execution. The guard requires 300 MB free swap and 12 GiB free disk. Training, local prediction, assembly and packaging support `--no-guard` for controlled reproduction.

## Rebuild both models with one command

`TRAIN_REVISION` below is the dataset revision behind the submitted weights. Use a fresh output directory. Existing `_data/diii_d_train` files are reused by `fetch_train.py`; their skip check verifies Parquet row count, not remote content identity. For a fresh download, use a new workspace copy without `_data/`.

```bash
export TRAIN_REVISION='1e280905b85f2a6fdde7e06fca8cf3a1edf447cb'
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

The organizer's rebuild tolerance (19 October 2026 announcement): a retrained model must score within 0.01 of the submitted composite S on each machine, and shipped weights must reproduce the submitted score to within 1e-4. A model rebuilt from scratch with the commands above has not yet been scored against that tolerance. The calibration step rewrites `diag/calib_joint_result.json`; submission #1 used the bundled file with the SHA-256 listed above.

## Regenerate both NPZ files and the ZIP with one command

The values below are the Final-phase inputs of submission #1. Use a fresh `RUN_DIR`. Submission #1 itself read local copies of the same files with `--local-dir` (see "Submission #1 record"); this streaming form has not been re-run on the Final-phase inputs.

```bash
export DATASET_REPO='ganatma/fusion-equilibrium-challenge-final'
export DATASET_REVISION='965040c12b814f33c8e7209f6b1799932247e793'
export DATASET_SPLIT='private_test'
export D3_CONFIG='diii_d_private_test'
export MAST_CONFIG='mast_private_test'
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

`manifest.json` binds repo/config/split/revision, local input hashes when applicable, weights and MAST calibration SHA256, inference Python-file SHA256 values, and each shot's source and frame count. Chunks carry the run signature. Resume and assembly reject mismatches. Assembly requires the same source/model arguments, including `--local-dir` and `--limit-shots` for local tests. Old manifests require a new work directory. Interrupted `.partial` files are retained; select a new work directory/output if one blocks exclusive creation. A completed source scan is required before assembly. With `--local-dir`, the repo, split and revision fields keep their defaults; submission #1's manifests show `Sophelio/fusion-equilibrium-challenge`, `public_test` and `None`. In that case the per-file SHA-256 values identify the inputs.

Packaging uses the same NPZ checker as `validate.py` for exact keys, contiguous shot numbering, shapes, float16 and finite values. `validate.py` additionally checks dataset shot/frame counts. ZIP root members are exactly `diii_d_private_test.npz` and `mast_private_test.npz`, stored without recompression.

Local two-shot, development and production commands are in `SUBMISSION_COMMANDS.md`. No upload is performed. Under the organizer's 19 October 2026 participant guide, the whole Final allows three successfully scored submissions per team; a failed submission does not use one of those three. A Code URL and a public repository are not required at submission time. Keep the exact code, pinned environment, weights and data provenance for every submission, with its fixed commit or versioned archive and SHA256, input revision, inference settings and ZIP identity. After the Final, the top five entries per challenge have seven days to supply a reproduction package when requested. The selected repository and its visibility remain unchanged; upload and submission remain human actions. See `DATA_PROVENANCE.md` for per-machine data use.
