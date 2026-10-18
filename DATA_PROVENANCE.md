# Data provenance (per machine)

Draft — to be finalized before the first Final-phase submission. Required by rule 6.

## DIII-D
- Training and model selection: `Sophelio/fusion-equilibrium-challenge` DIII-D train split (inputs and EFIT targets), with the starter kit's `data_fixes.fix_d3d_ip_times` applied.
- In `base/features_v2.py`, the DIII-D Ip time correction is applied within the causal input window/common past interval used to construct each feature. It is not a correction of the entire waveform before feature extraction.
- Local evaluation: held-out DIII-D training shots, scored with the starter kit's `local_score.py` (unmodified `fusion_scoring/`).

## MAST
- No MAST equilibrium reconstruction of any kind (flux maps, LCFS contours, EFIT-derived scalars, or aggregates of them) was used to train, fine-tune, calibrate, validate or select any component that produces MAST predictions.
- Used: MAST inputs only (coil currents, Ip, Thomson profiles), MAST machine geometry shipped in each row (coil rectangles, Thomson chord positions), and the starter kit's `data_fixes.align_mast_thomson_core`.
- MAST predictions use alpha_F (j1), kappa, q95 and betaN calculated only from the fixed twelve DIII-D training shots in `diag/calib_joint.py`. They use no MAST equilibrium data. The submission path sets c=0 and solenoid multiplier=1.
- The three labelled MAST demo shots in the starter kit were used for format inspection only.
- `fusion_scoring/masks/mast_envelope.npz` is used only inside the unmodified official scoring derivation, never as a model input, feature, loss mask or prior.
- Development-phase leaderboard feedback: not used (no Development-phase submission was made).

## External data and pre-trained models
- None. (Update this section if that changes.)
