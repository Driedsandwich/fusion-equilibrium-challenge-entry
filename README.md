# fusion-equilibrium-entry

Entry for the Fusion Equilibrium Challenge (NeurIPS 2026). Work in progress.

- `base/` — DIII-D model: PCA of the flux map, with the PCA coefficients and the two scalars (q95, betaN) regressed from non-magnetic inputs (coil currents, Ip, Thomson summaries) by gradient-boosted trees or a small MLP.
- `diag/` — diagnostics, including a check of how well a coil-plus-plasma-filament Green's-function representation can fit DIII-D flux maps.
- `starter_ref/` — unmodified files from the official starter kit (MIT, see `starter_ref/LICENSE-starter-MIT`), including the official scorer `fusion_scoring/`.

Data is not included. It is the public challenge dataset on Hugging Face (`Sophelio/fusion-equilibrium-challenge`, CC BY 4.0).

Training and inference entrypoints, a pinned environment and the data-provenance statement required by the challenge rules will be added before the first Final-phase submission.
