# Data provenance (per machine)

Covers Final-phase submissions #1 (code tag `submission-1`) and #2 (code tag `submission-2`). Rule 6 asks for a data-provenance statement per machine. Update this file for later submissions.

## DIII-D
- Training and model selection: `Sophelio/fusion-equilibrium-challenge` DIII-D train split (inputs and EFIT targets) at revision `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb`, with the starter kit's `data_fixes.fix_d3d_ip_times` applied. The submitted weights use the 312 files named in `train_shots.txt`.
- In `base/features_v2.py`, the DIII-D Ip time correction is applied within the causal input window/common past interval used to construct each feature. It is not a correction of the entire waveform before feature extraction.
- Local evaluation: during model development, 47 DIII-D training shots were held out from fitting and scored with the starter kit's `local_score.py` (unmodified `fusion_scoring/`). The submitted weights were then refitted on all 312 files (`weights/train_meta.json` records 0 held-out shots).
- Final-phase inputs: `ganatma/fusion-equilibrium-challenge-final` at revision `965040c12b814f33c8e7209f6b1799932247e793`, config `diii_d_private_test` (879 files, inputs only).
- Submission #2 reuses submission #1's DIII-D predictions unchanged (same file, same SHA-256).

## MAST
- No MAST equilibrium reconstruction of any kind (flux maps, LCFS contours, EFIT-derived scalars, or aggregates of them) was used to train, fine-tune, calibrate, validate or select any component that produces MAST predictions, including hyperparameter search and checkpoint selection.
- Final-phase inputs: `ganatma/fusion-equilibrium-challenge-final` at revision `965040c12b814f33c8e7209f6b1799932247e793`, config `mast_private_test` (1,205 files, inputs only).
- Used: MAST inputs only (coil currents, Ip, Thomson profiles), MAST machine geometry shipped in each row (coil rectangles, Thomson chord positions), and the starter kit's `data_fixes.align_mast_thomson_core`.
- MAST predictions use alpha_F (j1), kappa, q95 and betaN calculated only from the fixed twelve DIII-D training shots in `diag/calib_joint.py`. They use no MAST equilibrium data. Both submissions set c=0 (the DIII-D median origin that `diag/calib_joint.py` proposes is not used). Submission #1 applies the solenoid feed current once (multiplier 1); submission #2 applies it in each of the 656 solenoid turns (`--mast-sol-scale 656`), a count taken from the MAST geometry shipped with the inputs (one row per turn, per the starter-kit README), not from MAST equilibria or scores.
- The three labelled MAST demo shots in the starter kit were used for format inspection only.
- `fusion_scoring/masks/mast_envelope.npz` is used only inside the unmodified official scoring derivation, never as a model input, feature, loss mask or prior.
- Development-phase leaderboard feedback: not used (no Development-phase submission was made).

## External data and pre-trained models
- None. (Update this section if that changes.)
