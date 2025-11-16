Path Interference Testbed
-------------------------

Hypothesis
----------
- Î£-Gravity coherence is limited by quantum path interference whose visibility shrinks as velocity dispersion rises.

Data Status
-----------
- No real galaxy data ingested yet; use synthetic or preprocessed arrays from outside this folder and document the source explicitly when experiments begin.

Planned Analyses
----------------
- Fit `path_interference_multiplier` parameters (`A`, `ell0`, `p`, `n_coh`, `beta_sigma`, `sigma_ref`) against existing Î£-Gravity residual kernels.
- Compare recovered `L_coh(Ïƒ_v)` trends to observed Ïƒ-scaling from SPARC and Gaia catalogs.

Result Log Template
-------------------
- Record each run in `results_log.md` (to be created when the first experiment executes) with: data source (real vs toy), parameter fit summary, Ï‡Â² metrics, and interpretation of coherence-length scaling.
