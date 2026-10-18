# Methods report (draft outline, 1–2 pages)

1. Summary — one paragraph: machine-agnostic physics skeleton (coil Green's functions + Ip-scaled plasma current distribution placed from Thomson geometry), the same model applied to both machines.
2. Inputs and preprocessing — data_fixes; resampling inputs onto efit_times; no target interpolation.
3. Model — vacuum flux from coil rectangles and ampere-turn currents; plasma current distribution parameterization; the regressor and its machine-agnostic features; q95 and betaN from the predicted flux.
4. Zero-shot transfer to MAST — what changes between machines (geometry and inputs only); statement that no MAST equilibrium data was used (rule 3).
5. Safeguards — derivation-failure gate, float16 without rounding, fallback solution, in-shot smoothing.
6. Local evaluation — DIII-D held-out scores with local_score.py; extrapolation test across DIII-D bands.
7. Compute — hardware, wall-clock for training and any CPU-only generation; compute-light badge claim if applicable.
8. AI assistance disclosure (rule 4) — which AI coding assistants were used, for which parts, and that the entrant takes responsibility for the submission.
9. Reproduction — repository URL and pinned commit SHA; training and inference entrypoints.
