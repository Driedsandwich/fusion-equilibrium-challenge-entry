# fusion-equilibrium-entry

Entry for the Fusion Equilibrium Challenge (NeurIPS 2026). Final-phase submissions #1, #2 and #3 were made on 19, 21 and 26 October 2026 (UTC); see "Submission #1 record", "Submission #2 record" and "Submission #3 record" below. In the final results published on 14 November 2026 (https://fusion-equilibrium-challenge.sophelio.io/results/), the entry placed 46th of 57 in Challenge 1 (DIII-D composite S 0.920579) and 30th of 41 in Challenge 2 (G 0.551062, MAST S 0.507296), and won no award.

- `base/` — DIII-D model: PCA of the flux map, with the PCA coefficients and the two scalars (q95, betaN) regressed from non-magnetic inputs (coil currents, Ip, Thomson summaries) by gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`, used by `train.py` for the submitted weights). `base/train_eval_v2.py` also compares ridge and a small MLP on held-out shots.
- `diag/` — diagnostics, including a check of how well a coil-plus-plasma-filament Green's-function representation can fit DIII-D flux maps.
- `starter_ref/` — unmodified files from the official starter kit (MIT, see `starter_ref/LICENSE-starter-MIT`), including the official scorer `fusion_scoring/`.

Data is not included. Training uses the public challenge dataset on Hugging Face (`Sophelio/fusion-equilibrium-challenge`, CC BY 4.0). Final-phase test inputs come from `ganatma/fusion-equilibrium-challenge-final`.

## About this public copy

This repository is a public copy of the private development repository. Before publication, one sentence was edited in the version of `SUBMISSION_COMMANDS.md` stored in three earlier commits (it named a local file path and an email from the organizers); no other file differs in any commit. Commits up to and including submission #1's `c337865` keep their IDs. Later commits have new IDs here:

| Original commit | In this copy | Note |
|---|---|---|
| `e5aaaae` | `5dba2dd` | inference code of submission #2 |
| `f16f76c` | `9805e8d` | tag `submission-2` |
| `f4a610d` | `ffb3213` | |
| `7b84dc9` | `36aa572` | tag `submission-3` |
| `7af13e0` | `9558761` | |

## Submission #1 record

- Platform: Codabench competition 17739, Final phase, submission ID 1003290, created 2026-10-19 15:35 UTC, scored (status Finished).
- Code: commit `c337865f9e4810c47570d73aa0a4d5da4c9e4070`, tag `submission-1`. Check out this commit to reproduce submission #1.
- Environment: `requirements-lock.txt`, Python 3.12.13.
- Weights: `weights/model.joblib`, SHA-256 `03d504a862419f1c5d55266a38bffa67ed37d498f8e0c2650cf27990542f99b7`. MAST calibration: `diag/calib_joint_result.json`, SHA-256 `29498713233df2396c0e85f4e418aac151678b83e79af394922531391374cfab`. MAST policy: j1 alpha_F, c=0, solenoid multiplier 1.
- Training data: `Sophelio/fusion-equilibrium-challenge` at revision `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb`, the 312 files of `data/diii_d_train` named in `train_shots.txt`. On 2026-10-20 the SHA-256 of all 312 local files matched the Hub's LFS records at that revision.
- Test inputs: `ganatma/fusion-equilibrium-challenge-final` at revision `965040c12b814f33c8e7209f6b1799932247e793`, configs `diii_d_private_test` (879 files) and `mast_private_test` (1,205 files), split `private_test`. Submission #1 read local copies of these files with `--local-dir _data/final/<config>` and `--config <config>`. The run manifests store a SHA-256 for every input file. On 2026-10-20 all 2,084 values matched the Hub's LFS records at that revision.
- Output: `submission.zip`, 1,918,727,900 bytes, SHA-256 `5e753ee20c4aa03723ad395d0655b8972aed445747490b2dcbbabca499ffe1f3`. Its root holds only `diii_d_private_test.npz` and `mast_private_test.npz`.

## Submission #2 record

- Platform: Codabench competition 17739, Final phase, submission ID 1007599, created 2026-10-21 07:12 UTC, scored (status Finished).
- Code: tag `submission-2`. Inference code is that of commit `5dba2dd` (`e5aaaae` in the original history); compared with submission #1 (`c337865`) it adds only the `--mast-sol-scale` option (recorded as a separate run-identity field), whose default 1.0 reproduces submission #1. The environment (`requirements-lock.txt`, Python 3.12.13; on 2026-10-20 the `.venv` package list matched the lock file) and `weights/model.joblib` and `diag/calib_joint_result.json` (same SHA-256 in the run manifest) are those listed under submission #1.
- DIII-D: `diii_d_private_test.npz` is submission #1's file, extracted from submission #1's ZIP: 1,397,867,474 bytes, SHA-256 `b2af65a4244ede7cd5f3c8083dc17e4d708143b2295838eaf74be30f1a717702`; the DIII-D members of both ZIPs have this size and SHA-256. This submission therefore leaves the entry's best DIII-D score unchanged.
- MAST: regenerated from the same 1,205 local Final-phase input files with `--mast-sol-scale 656`; no other setting changed (origin c = 0 as in submission #1). The starter-kit README's geometry section states that MAST ships one geometry row per conductor turn and that all 656 solenoid rows share `magnetics_sol_current`, the solenoid feed current in kA. Submission #1 applied that current once (multiplier 1); submission #2 applies it in each of the 656 turns. The README's units note forbids turn multipliers on the shaping-coil columns, which are already in kA·turn; its reasons for keeping other channels in plain kA name `ECOILA`, `bcoil` and `tf`, not the solenoid. The value is the shipped row count; no MAST equilibrium, MAST score or statistic of either was used to choose it.
- Difference from submission #1 (prediction archives only, no truth): over 1,205 shots and 68,373 frames, absolute ψ difference median 0.0306 Wb/rad and maximum 0.1212 Wb/rad; q95 and βN bitwise equal.
- Output: `submission.zip`, 1,876,764,675 bytes, SHA-256 `eeead2a18e2fa5a708b4331220bd122ed50e4f4ff4d0e89ac3cd9fe16a20d65e`; MAST NPZ SHA-256 `b5bffe9f3bffe80ea3c438b3ec617b3cabb526049bd977bc19cdaac16288c6a5`. Root members: `diii_d_private_test.npz` and `mast_private_test.npz`.
- Commands (repository root):

      .venv/bin/python -B predict.py --config mast_private_test --local-dir _data/final/mast_private_test --weights weights --work-dir _work/final6/mast --mast-sol-scale 656
      .venv/bin/python -B predict.py --config mast_private_test --local-dir _data/final/mast_private_test --weights weights --work-dir _work/final6/mast --mast-sol-scale 656 --assemble --output _work/final6/out/mast_private_test.npz
      .venv/bin/python -B validate.py _work/final6/out/mast_private_test.npz --config mast_private_test --local-dir _data/final/mast_private_test
      .venv/bin/python -B make_zip.py --diii-d _work/final6/out/diii_d_private_test.npz --mast _work/final6/out/mast_private_test.npz --output _work/final6/out/submission.zip

## Submission #3 record

- Platform: the organizers' Final-phase submission page (https://fusion-equilibrium-challenge.sophelio.io/final-submission/), used because Codabench was unreachable from 22 October (organizers' announcement of 23 October). Receipt FW-B0CA2C4F, received 2026-10-26 23:12:02 UTC; the page reported both `diii_d_private_test.npz` and `mast_private_test.npz` found. The page takes a pointer to a private Hugging Face dataset repository: the two NPZ members of the output below were pushed to its root at commit `25269b4235d8374ab95b604f64b1d82dbaed1b18`, where their sizes and LFS SHA-256 equal those recorded in this section. The ZIP itself was not uploaded.
- Code: tag `submission-3`. It adds `mast/qb_physics.py`, `mast/qb_io.py`, `mast/qb_generate.py`, `mast/qb_assemble.py`, `diag/qb_gate_diiid.py` and `diag/test_qb_physics.py`; no earlier code file changes. The environment, `weights/model.joblib` and `diag/calib_joint_result.json` are those of submissions #1 and #2.
- DIII-D: the same `diii_d_private_test.npz` as submissions #1 and #2 (1,397,867,474 bytes, SHA-256 `b2af65a4244ede7cd5f3c8083dc17e4d708143b2295838eaf74be30f1a717702`), so the entry's best DIII-D score does not change.
- MAST flux maps: every `shot_XXXX_psirz` member is copied from submission #2's `mast_private_test.npz` as its compressed ZIP member, without re-encoding. For all 1,205 shots the CRC, compressed size and method are equal, and so are the SHA-256 of the decoded arrays.
- MAST q95 and betaN (the only change; in submissions #1 and #2 they were DIII-D medians): estimated without MAST equilibrium data. q95 = (|F|/2π)∮ dl/(R·|∇ψ|) on the ψ_N = 0.95 surface of the copied flux map, with the vacuum F = μ0·24·I_TF/(2π) from `magnetics_tf_current` (kA). The turn count 24 is MAST hardware: A. C. Darke et al., "MAST: The Mega Amp Spherical Tokamak" ("MAST uses same LVPS supply, in 24 turns giving 0.6T"), consistent with the 2.2 MA TF rod current and 0.52 T at R = 0.85 m in A. Sykes, "First Results From MAST", IAEA Fusion Energy Conference 2000, OV4/1, Table 1. betaN = 100·2μ0⟨p⟩·a/(B0·|Ip| [MA]), where ⟨p⟩ is c_p times the volume average of the Thomson electron pressure n_e·T_e mapped to ψ_N, a and R_geo come from the LCFS and B0 = |F|/R_geo. c_p = 2.42 was fitted on the first 60 shots of the DIII-D training list (DIII-D: 144 TF turns, General Atomic report GA-A-13668, 1975).
- Physical ranges: frames that fail or fall outside 2 < q95 < 20 or 0 < betaN < 7 take the median of the same shot's in-range estimates (q95: 16,422 of 68,373 frames; betaN: 22,632 frames, of which 193 in shots with no in-range estimate take the median of all in-range MAST estimates of this submission, 1.54).
- Check on DIII-D (EFIT flux maps of the 52 held-out training shots, every fourth frame, 3,101 frames): pooled R² 0.903 for q95 and 0.866 for betaN. MAST cannot be checked without MAST equilibria.
- Output: `submission.zip`, 1,876,964,972 bytes, SHA-256 `f115dbd957f16c075c46497890515700cc239041bfc51aae7b453b134f0eb21d`; MAST NPZ 479,097,236 bytes, SHA-256 `22abe08c10c9bad84f78dad02056d6ac17f9fa73db9aae9429ec93a31e2747e6`. Root members: `diii_d_private_test.npz` and `mast_private_test.npz`.
- Commands (repository root; `qb_assemble.py` runs `make_zip.py` and `validate.py`). They read submission #2's `_work/final6/out/mast_private_test.npz` and `_work/final6/out/diii_d_private_test.npz`, produced by the commands in "Submission #2 record":

      .venv/bin/python -B diag/qb_gate_diiid.py
      .venv/bin/python -B mast/qb_generate.py --work-dir _work/final7/qb/full
      .venv/bin/python -B mast/qb_assemble.py --qb-dir _work/final7/qb/full --out-dir _work/final7/out

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

For submission #3, also refit the MAST pressure factor c_p on DIII-D training shots with `.venv/bin/python -B diag/qb_gate_diiid.py`; it writes `diag/qb_gate_result.json`, which `mast/qb_generate.py` reads. `train.py` refuses to overwrite an existing model. The separate `--weights` output preserves bundled `weights/`. Calibration writes `diag/calib_joint_result.json`, `diag/joint_locals/` and its report; finish calibration before starting a new prediction run. To use rebuilt DIII-D weights, set `WEIGHTS=_work/rebuilt/weights` below. Full training and calibration also accept `--no-guard`; the default remains fail closed.

`diag/capacity_result_run2.json` is the archived second capacity diagnostic on the fixed twelve DIII-D shots listed in `diag/calib_joint.py`. Its settings were stride 4, quadrature order 6, relative ridge 1e-10 and a 520-second budget; the diagnostic took 44.948 seconds. Calibration consumes only its shot identities and `geometry.box_m`. The box is the union of stored LCFS contours over all 2,342 frames of those twelve shots, padded by 0.05 m on each side. No MAST equilibrium enters this calculation.

The command above regenerates these consumed fields into `_work/rebuilt/capacity_input.json` and uses that input for calibration. It does not regenerate the historical scoring report. `--capacity-result` also accepts an existing input. Running `diag/capacity.py` uses all training files currently present, so with 312 files it does not recreate the historical twelve-shot report.

Recorded computation: `weights/train_meta.json` reports 312 training shots, 101.226 seconds, macOS 27.0.1 arm64, 10 logical CPUs, numerical computation limited to one thread, Python 3.12.13. `diag/calib_joint_result.json` reports 4.569 seconds for calibration (474 sampled frames, stride 5). These are separate timings and exclude environment setup/downloads. Runtime on other machines may differ.

The organizer's rebuild tolerance (19 October 2026 announcement): a retrained model must score within 0.01 of the submitted composite S on each machine, and shipped weights must reproduce the submitted score to within 1e-4. A model rebuilt from scratch with the commands above has not yet been scored against that tolerance. The calibration step rewrites `diag/calib_joint_result.json`; submission #1 used the bundled file with the SHA-256 listed above.

## Regenerate both NPZ files and the ZIP with one command

The values below are the Final-phase inputs of submission #1. This form regenerates submission #1; submission #2 adds `--mast-sol-scale 656` to the MAST commands, and submission #3 then replaces only q95 and betaN with the commands in "Submission #3 record". Use a fresh `RUN_DIR`. Submission #1 itself read local copies of the same files with `--local-dir` (see "Submission #1 record"); this streaming form has not been re-run on the Final-phase inputs.

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

Local two-shot, development and production commands are in `SUBMISSION_COMMANDS.md`. No upload is performed. Under the organizer's 19 October 2026 participant guide, the whole Final allows three successfully scored submissions per team; a failed submission does not use one of those three. A Code URL and a public repository are not required at submission time. Keep the exact code, pinned environment, weights and data provenance for every submission, with its fixed commit or versioned archive and SHA256, input revision, inference settings and ZIP identity. After the Final, the top five entries per challenge have seven days to supply a reproduction package when requested. See `DATA_PROVENANCE.md` for per-machine data use.
