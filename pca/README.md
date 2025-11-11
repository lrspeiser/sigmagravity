# Rotation-Curve PCA Toolkit

## ✅ STATUS: INTEGRATED WITH SIGMAGRAVITY DATA

This toolkit provides an end-to-end pipeline to discover **latent structure** in galaxy rotation curves (SPARC, Gaia/MW) using:
- Weighted, radius-aligned **PCA** (curves only or curves + scalar physics features).
- **Diagnostics**: Scree curves, radial loading profiles, biplots, correlations (PCs ↔ Σ₀, R_d, M_bar, V_f).
- **Clustering** in PC space (BIC-chosen Gaussian Mixture).
- **Subset stability** (principal angles) across LSB/HSB, dwarfs/massive, etc.
- Optional **GPU autoencoder** (PyTorch) to check non-linear structure vs PCA.

**Hardware:** Designed to benefit from your NVIDIA GPU (e.g., 5090). PCA runs fine on CPU; the autoencoder will use GPU automatically if available.

> **DONE**: Successfully integrated with SPARC data from `data/Rotmod_LTG/`. See `ANALYSIS_SUMMARY.md` for results!
>
> If you already have outputs from your Σ‑Gravity repo (e.g., `data/gaia/outputs/mw_gaia_full_coverage_suggested.csv`), you can plug them into `08_compare_models.py` to study how your model residuals align with PCA axes. See usage there.


## Quick start (Already Done!)

The pipeline has already been run on your SPARC data! To explore the results:

```bash
# View the summary
cat pca/ANALYSIS_SUMMARY.md

# Interactive exploration (plots require GUI)
python -i pca/explore_results.py
# Then use: plot_cluster_curves(5), plot_pc_loadings(0), etc.

# View generated figures
ls pca/outputs/figures/
```

Key findings:
- **PC1-3 capture 96.5% of variance** in 170 galaxy rotation curves
- **6 clusters** identified (1 dominant cluster of 141 galaxies)
- **HSB/LSB galaxies share similar PC1** (4.1° principal angle)

## Pipeline Steps (for reference)

To re-run or modify the analysis:

```bash
# 0) Create a Python environment (conda recommended)
conda create -n rc-pca python=3.11 -y
conda activate rc-pca

# 1) Install deps (CPU baseline)
pip install -r requirements.txt

# For GPU autoencoder (optional):
#   - PyTorch with CUDA: see https://pytorch.org/get-started/locally/ for the right command on your machine.
#   - Optional cupy/cuml if you want GPU PCA: pip install cupy-cuda12x cuml-cu12 --extra-index-url https://pypi.nvidia.com

# 2) Prepare SPARC data (see below for expected formats)
# Example: curves as per-galaxy CSVs in data/raw/sparc_curves/, metadata in data/raw/metadata/sparc_meta.csv
python scripts/01_ingest_sparc.py   --curves_dir data/raw/sparc_curves   --meta_csv data/raw/metadata/sparc_meta.csv   --out_npz data/processed/sparc_curvematrix.npz   --grid_min 0.2 --grid_max 6.0 --grid_k 50 --norm_radius Rd --norm_velocity Vf

# 3) Build feature matrices (curve-only and curves+scalars)
python scripts/02_build_curve_matrix.py   --npz data/processed/sparc_curvematrix.npz   --scalars_json configs/scalars_sparc.json   --out_prefix data/processed/sparc_features

# 4) Run weighted PCA and make diagnostics
python scripts/03_run_weighted_pca.py   --features_npz data/processed/sparc_features_curve_only.npz   --n_components 10   --out_dir outputs

python scripts/04_plot_diagnostics.py   --pca_npz outputs/pca_results_curve_only.npz   --grid_json data/processed/sparc_curvematrix_grid.json   --out_dir outputs/figures

# 5) Clustering in PC space and correlations with physics
python scripts/05_cluster_pc_space.py   --pca_npz outputs/pca_results_curve_only.npz   --features_npz data/processed/sparc_features_curve_plus_scalars.npz   --out_dir outputs

# 6) Subset stability (e.g., HSB vs LSB)
python scripts/06_subset_stability.py   --features_npz data/processed/sparc_features_curve_only.npz   --subset_csv data/raw/metadata/sparc_meta.csv   --subset_column HSB_LSB   --n_components 3   --out_dir outputs

# 7) Optional: train a GPU autoencoder on curves
python scripts/07_autoencoder_train.py   --features_npz data/processed/sparc_features_curve_only.npz   --latent_dim 3 --epochs 50 --batch 256   --out_dir outputs/models

# 8) Optional: align Σ‑Gravity residuals or model params with PCs
python scripts/08_compare_models.py   --pca_npz outputs/pca_results_curve_only.npz   --model_csv path/to/mw_gaia_full_coverage_suggested.csv   --out_dir outputs/tables
```

---

## Expected data formats

We support **flexible** SPARC-like inputs. You can adapt column names via the `--map_*` flags in `01_ingest_sparc.py`.

### A) Rotation curves (one CSV per galaxy)
Required columns (case-insensitive, can be remapped):
- `R_kpc`: radius (kpc)
- `V_obs`: observed circular speed (km/s)
- `eV_obs`: 1σ uncertainty in `V_obs` (km/s)

Optional columns (if you want RAR diagnostics):
- `V_bar`: baryon-contributed circular speed (km/s). If absent but `V_star` and `V_gas` are present, we use `V_bar = sqrt(V_star**2 + V_gas**2)`.

### B) Metadata CSV (one row per galaxy)
Required columns (case-insensitive, can be remapped):
- `name`: galaxy identifier (must match curve filename stem or be mapped via `--id_column`)
- `Rd`: disk scale length (kpc)
- `Vf`: flat-part velocity (km/s), optional but helpful for velocity normalization
- `Sigma0`: central surface density (M_sun/pc^2) — optional
- `Mbar`: total baryonic mass (M_sun) — optional
- `HSB_LSB`: category string (e.g., "HSB" or "LSB") — optional for subset tests

**Curve files:** Place per-galaxy CSVs under `data/raw/sparc_curves/`. Filenames should include the galaxy name or you can supply a mapping CSV.

---

## Outputs

- `data/processed/sparc_curvematrix.npz` — normalized, resampled curve matrix (N×K) + weights (N×K) + scalars.
- `data/processed/sparc_features_*.npz` — feature matrices (curve-only; curve+scalars).
- `outputs/pca_results_*.npz` — PCA scores/loadings/EVR + scalers and sample weights.
- Figures in `outputs/figures/`: scree plots, radial loadings per PC, PC scatter with clusters, correlations.
- Tables in `outputs/tables/`: correlations (PCs ↔ physics), cluster memberships, subspace angles.

---

## Notes

- Weighted PCA here uses **per-feature weighted standardization** (from per-point σ) and **sample-weighted SVD** with row weights = median of per-feature weights per galaxy. This closely approximates full per-point weighting while remaining numerically stable.
- You can switch to **cuML PCA** if you want GPU-accelerated PCA; or keep CPU for reproducibility.
- Autoencoder training automatically uses CUDA if available (PyTorch).

For integration with your Σ‑Gravity repo outputs (e.g., Milky Way star-level residuals or SPARC RAR tables), see `08_compare_models.py`.


## License
MIT
