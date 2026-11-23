# Data provenance (per machine)

Covers Final-phase submissions #1 (code tag `submission-1`), #2 (code tag `submission-2`) and #3 (code tag `submission-3`). Rule 6 asks for a data-provenance statement per machine. These were all of the entry's Final-phase submissions.

## DIII-D
- Training and model selection: `Sophelio/fusion-equilibrium-challenge` DIII-D train split (inputs and EFIT targets) at revision `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb`, with the starter kit's `data_fixes.fix_d3d_ip_times` applied. The submitted weights use the 312 files named in `train_shots.txt`.
- In `base/features_v2.py`, the DIII-D Ip time correction is applied within the causal input window/common past interval used to construct each feature. It is not a correction of the entire waveform before feature extraction.
- Local evaluation: during model development, 52 DIII-D training shots were held out from fitting; the run stopped at its resource guard after scoring 47 of them, every eighth frame. Scoring used the starter kit's `local_score.py` functions (`build_reference`, `score_shot`, `psi_residuals`) and the unmodified `fusion_scoring/` package (`Accum`, `finalize_machine`), called from the entry's own loop in `base/train_eval_v2.py`. The submitted weights were then refitted on all 312 files (`weights/train_meta.json` records 0 held-out shots).
- Final-phase inputs: `ganatma/fusion-equilibrium-challenge-final` at revision `965040c12b814f33c8e7209f6b1799932247e793`, config `diii_d_private_test` (879 files, inputs only).
- Submissions #2 and #3 reuse submission #1's DIII-D predictions unchanged (same file, same SHA-256).
- For submission #3's MAST q95/betaN, DIII-D training shots (EFIT flux maps, q95 and betaN, `magnetics_bcoil`, Ip with `fix_d3d_ip_times` applied to the whole waveform, Thomson profiles with `thomson_chord_R`/`_Z` positions) were used to fit the pressure factor c_p (the first 60 shots of the training list) and to check the estimates (the 52 held-out shots above), in `diag/qb_gate_diiid.py`.

## MAST
- No MAST equilibrium reconstruction of any kind (flux maps, LCFS contours, EFIT-derived scalars, or aggregates of them) was used to train, fine-tune, calibrate, validate or select any component that produces MAST predictions, including hyperparameter search and checkpoint selection.
- Final-phase inputs: `ganatma/fusion-equilibrium-challenge-final` at revision `965040c12b814f33c8e7209f6b1799932247e793`, config `mast_private_test` (1,205 files, inputs only).
- Used: MAST inputs only (coil currents, Ip, toroidal-field coil current, Thomson profiles), MAST machine geometry shipped in each row (coil rectangles, Thomson chord positions), the published MAST toroidal-field turn count (24), and the starter kit's `data_fixes.align_mast_thomson_core`.
- MAST flux maps use alpha_F (j1) and kappa, and submissions #1 and #2 also use q95 and betaN, calculated only from the fixed twelve DIII-D training shots in `diag/calib_joint.py`. They use no MAST equilibrium data. All three submissions set c=0 (submission #3 copies submission #2's flux maps) (the DIII-D median origin that `diag/calib_joint.py` proposes is not used). Submission #1 applies the solenoid feed current once (multiplier 1); submission #2 applies it in each of the 656 solenoid turns (`--mast-sol-scale 656`), a count taken from the MAST geometry shipped with the inputs (one row per turn, per the starter-kit README), not from MAST equilibria or scores.
- Submission #3 keeps submission #2's MAST flux maps (copied unchanged) and replaces only q95 and betaN by estimates computed from those flux maps, the MAST inputs above, the turn count 24 and the DIII-D pressure factor c_p (`mast/qb_generate.py`; README "Submission #3 record"). No MAST equilibrium data, statistic of it or MAST score was used.
- The three labelled MAST demo shots in the starter kit were used for format inspection only.
- `fusion_scoring/masks/mast_envelope.npz` is not in this repository (ignored by `.gitignore`) and was never used: the entry never scored MAST predictions, and no MAST prediction path loads an envelope mask.
- Development-phase leaderboard feedback: not used (no Development-phase submission was made).

## External data and pre-trained models
- No external datasets or pre-trained models. Submission #3 uses two published hardware values, the toroidal-field turn counts of MAST (24) and DIII-D (144); sources are listed in the README's "Submission #3 record".
