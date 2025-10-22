# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- Purpose: paper-release bundle for Σ‑Gravity figures, derivations, and tables.
- Structure:
  - data/Rotmod_LTG: SPARC rotation-curve catalog used by galaxy figure scripts.
  - scripts/: figure and table generators. Many rely on external modules imported as many_path_model.* and core.* via sys.path tricks.
  - derivations/: mathematical verification script and manuscripts.
  - config/: hyperparameters and bars overrides used by galaxy scripts.
  - figures/, tables/: generated outputs.
- External code expected but not included here:
  - many_path_model/* and core/* modules, and projects/SigmaGravity assets. Several scripts assume these exist at a higher-level repo root (three directories above scripts/) as coded via Path(__file__).resolve().parents[3]. Place required repos accordingly or adjust import paths.

Dependencies
- Python 3.10+ with: numpy, scipy, matplotlib, pandas, sympy, markdown, mpmath. Optional: corner, cupy, arviz, pymc (needed only by external pipelines referenced in README).
- For PDF building: Google Chrome or Microsoft Edge installed (used headlessly by scripts/make_pdf.py).

Common commands
- Install core Python packages (adjust to your environment):
  - python -m pip install -U numpy scipy matplotlib pandas sympy markdown mpmath

- Derivation validation (acts as a sanity-test suite; writes results under results/derivations):
  - python derivations/derivation_validation.py

- Build paper-style PDF from a Markdown file (example uses README):
  - python scripts/make_pdf.py --md README.md --out figures/paper_formatted.pdf

- Self-contained theory and bounds figures (do not require external repos; override defaults to write inside figures/):
  - python scripts/generate_pn_bounds_plot.py --rotmod data/Rotmod_LTG/NGC2403_rotmod.dat --out figures/pn_bounds_ngc2403.png
  - python scripts/generate_theory_ring_kernel_check.py
  - python scripts/generate_theory_window_check.py

- Galaxy figures requiring external many_path_model present at expected path:
  - python scripts/generate_rc_gallery.py --sparc_dir data/Rotmod_LTG --master data/Rotmod_LTG/MasterSheet_SPARC.mrt --hp config/hyperparams_track2.json --out figures/rc_gallery.png
  - python scripts/generate_rar_plot.py --sparc_dir data/Rotmod_LTG --hp config/hyperparams_track2.json --out figures/rar_sparc_validation.png

- Cluster figures requiring external repos and data (projects/SigmaGravity paths):
  - python scripts/generate_cluster_kappa_panels.py --clusters A2261,MACS1149 --catalog projects/SigmaGravity/data/clusters/master_catalog_paper.csv --posterior projects/SigmaGravity/output/pymc_mass_scaled/flat_samples_from_pymc.npz --out figures/cluster_kappa_panels.png

- Convenience wrapper (Windows PowerShell; expects external repos):
  - pwsh -File scripts/make_all_figures.ps1

Notes on tests and linting
- No formal test runner or linter is configured in this repository. Use derivations/derivation_validation.py as the primary regression check. There is no single-test selection flag; run individual lightweight scripts in scripts/ as targeted checks when needed.

Path and layout assumptions used by scripts
- Many scripts compute ROOT = Path(__file__).resolve().parents[3] and then refer to ROOT/many_path_model/paper_release/... and ROOT/projects/SigmaGravity/.... To run those scripts as-is:
  - Place this repository so that the directory three levels above scripts/ contains many_path_model and projects/SigmaGravity.
  - Or modify sys.path insertions and default paths in the scripts to point to your local clones of those repos.

Configuration used by galaxy scripts
- config/hyperparams_track2.json: baseline hyperparameters for the Track-2 galaxy kernel.
- config/bars_override.json: optional SPARC bar-class overrides (e.g., UGC02953: SB).

Outputs and provenance
- Figure commands write into figures/ when you pass --out there. Some scripts hardcode many_path_model/paper_release/figures; adjust arguments or the script constants to keep outputs inside this repo.
- derivations/derivation_validation.py writes JSON and PNGs under results/derivations.

What to read first
- README.md sections on reproducibility and methods for additional context on the galaxy RAR workflow and the cluster Σ‑kernel pipeline (note that those workflows rely on external repos not included here).
