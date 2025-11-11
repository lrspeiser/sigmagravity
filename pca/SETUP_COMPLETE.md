# PCA Toolkit - Integration Complete! ✅

## What Was Done

Successfully integrated the rotation-curve PCA toolkit with your SPARC dataset from `data/Rotmod_LTG/`.

### 1. Data Conversion
- Created `scripts/00_convert_sparc_to_csv.py` to convert SPARC `.dat` files to CSV format
- Converted **175 rotation curve files** 
- Parsed the SPARC MasterSheet metadata for **174 galaxies**
- Generated CSV files in `pca/data/raw/sparc_curves/` and `pca/data/raw/metadata/`

### 2. PCA Pipeline Execution
Ran the complete 6-step pipeline:

1. **Ingest & Normalize** (`01_ingest_sparc.py`) - ✅ Complete
   - Normalized 170 rotation curves to common R/Rd grid (0.2-6.0, 50 points)
   - Applied velocity normalization by Vf (flat velocity)
   - Output: `pca/data/processed/sparc_curvematrix.npz`

2. **Build Feature Matrices** (`02_build_curve_matrix.py`) - ✅ Complete
   - Created curve-only and curve+scalars feature matrices
   - Output: `pca/data/processed/sparc_features_*.npz`

3. **Weighted PCA** (`03_run_weighted_pca.py`) - ✅ Complete
   - Computed 10 principal components
   - Used per-point weighting based on observational uncertainties
   - Output: `pca/outputs/pca_results_curve_only.npz`

4. **Diagnostics** (`04_plot_diagnostics.py`) - ✅ Complete
   - Generated scree plot (cumulative variance)
   - Created radial loading profiles for PC1-3
   - Output: `pca/outputs/figures/*.png`

5. **Clustering** (`05_cluster_pc_space.py`) - ✅ Complete
   - Identified 6 clusters via Gaussian Mixture Model (BIC selection)
   - Generated PC scatter plot with cluster colors
   - Output: `pca/outputs/clusters.csv`, `pca/outputs/pc_scatter_clusters.png`

6. **Subset Stability** (`06_subset_stability.py`) - ✅ Complete
   - Compared HSB vs LSB galaxy PC subspaces
   - Computed principal angles
   - Output: `pca/outputs/subset_principal_angles.txt`

### 3. Created Helper Scripts
- **`explore_results.py`** - Interactive exploration script with plotting functions
- **`ANALYSIS_SUMMARY.md`** - Comprehensive summary of results and findings

## Key Results

### Variance Explained
- **PC1**: 79.5% (dominant mode of variation)
- **PC1-3**: 96.5% (captures nearly all structure)
- **PC1-5**: 99.9% (essentially complete)

### Clustering
- **6 clusters** identified (k selected by BIC)
- **141/170 galaxies** in dominant cluster (5)
- Smaller clusters (1-22 galaxies each) represent morphological outliers

### HSB vs LSB Comparison
Principal angles between 3D subspaces:
- **1st angle: 4.1°** - Nearly identical primary variation mode
- **2nd angle: 18.6°** - Moderate divergence in secondary mode
- **3rd angle: 44.5°** - Significant divergence in tertiary mode

**Interpretation**: HSB and LSB galaxies share the same dominant rotation curve shape (PC1), but differ in secondary structural features.

## How to Use

### View Results
```bash
# Read the analysis summary
cat pca/ANALYSIS_SUMMARY.md

# Check generated figures
ls pca/outputs/figures/
# - scree_cumulative.png
# - pc1_radial_loading.png
# - pc2_radial_loading.png
# - pc3_radial_loading.png

# View cluster visualization
# open pca/outputs/pc_scatter_clusters.png
```

### Interactive Exploration
```bash
# Run in interactive mode
python -i pca/explore_results.py

# Then use these functions:
>>> plot_cluster_curves(cluster_id=5)        # Plot curves from cluster 5
>>> plot_pc_loadings(pc_idx=0)               # Show PC1 loading profile
>>> plot_galaxy_vs_mean('NGC3198')           # Compare galaxy to mean
>>> plot_pc_scatter(pc_x=0, pc_y=1)          # Scatter plot in PC1-PC2 space
```

### Access Raw Data
```python
import numpy as np
import pandas as pd

# Load PCA results
pca = np.load('pca/outputs/pca_results_curve_only.npz')
components = pca['components']  # [10, 50] - 10 PCs x 50 radial points
scores = pca['scores']          # [170, 10] - galaxy projections onto PCs
evr = pca['evr']                # [10] - explained variance ratios

# Load rotation curves
curves = np.load('pca/data/processed/sparc_curvematrix.npz', allow_pickle=True)
curve_mat = curves['curve_mat']  # [170, 50] - normalized V(R)
x_grid = curves['x_grid']        # [50] - radial grid (R/Rd)
names = curves['names']          # [170] - galaxy names

# Load clusters
clusters = pd.read_csv('pca/outputs/clusters.csv')
# Columns: name, cluster

# Load metadata
meta = pd.read_csv('pca/data/raw/metadata/sparc_meta.csv')
# Columns: name, Rd, Vf, Mbar, Sigma0, HSB_LSB, etc.
```

## Next Steps (Optional Research)

1. **Correlate with Physics**
   - Examine how PC scores relate to Mbar, Σ₀, Rd, Vf
   - Identify which physical parameters drive each PC

2. **Model Comparison** (Step 8)
   ```bash
   python pca/scripts/08_compare_models.py \
       --pca_npz pca/outputs/pca_results_curve_only.npz \
       --model_csv data/gaia/outputs/mw_gaia_full_coverage_predicted.csv \
       --out_dir pca/outputs/tables
   ```
   This will show how Σ-Gravity model residuals project onto the empirical PCs.

3. **Non-linear Structure** (Step 7)
   ```bash
   python pca/scripts/07_autoencoder_train.py \
       --features_npz pca/data/processed/sparc_features_curve_only.npz \
       --latent_dim 3 --epochs 50 --batch 256 \
       --out_dir pca/outputs/models
   ```
   Train a neural network autoencoder to test if there's non-linear structure beyond PCA.

4. **Mass-based Subsets**
   - Repeat subset stability analysis for dwarf vs massive galaxies
   - Test if Mbar (baryonic mass) divides galaxies into distinct PC subspaces

## Files Generated

```
pca/
├── ANALYSIS_SUMMARY.md          # Complete analysis writeup
├── SETUP_COMPLETE.md            # This file
├── explore_results.py            # Interactive exploration script
├── data/
│   ├── raw/
│   │   ├── sparc_curves/        # 175 CSV rotation curve files
│   │   └── metadata/
│   │       └── sparc_meta.csv   # Galaxy metadata (Rd, Vf, Mbar, etc.)
│   └── processed/
│       ├── sparc_curvematrix.npz            # Normalized curves [170 x 50]
│       ├── sparc_features_curve_only.npz     # Feature matrix (curves)
│       └── sparc_features_curve_plus_scalars.npz  # Feature matrix (curves+scalars)
└── outputs/
    ├── pca_results_curve_only.npz       # PCA components, scores, EVR
    ├── pca_explained_curve_only.csv     # Explained variance table
    ├── clusters.csv                     # Cluster assignments
    ├── subset_principal_angles.txt      # HSB/LSB comparison
    ├── pc_scatter_clusters.png          # PC1-PC2 scatter with clusters
    └── figures/
        ├── scree_cumulative.png         # Cumulative variance plot
        ├── pc1_radial_loading.png       # PC1 vs R/Rd
        ├── pc2_radial_loading.png       # PC2 vs R/Rd
        └── pc3_radial_loading.png       # PC3 vs R/Rd
```

## Technical Details

- **Weighting**: Per-point weights derived from observational uncertainties (1/σ²)
- **Normalization**: 
  - Radial: R → R/Rd (scale-invariant)
  - Velocity: V → V/Vf (amplitude-invariant)
- **Grid**: 50 logarithmically-spaced points from 0.2 to 6.0 R/Rd
- **Sample size**: 170 galaxies (4 skipped due to missing Rd or insufficient data)
- **PCA method**: Weighted SVD with per-feature standardization

## Questions?

- See `pca/README.md` for full toolkit documentation
- See `pca/ANALYSIS_SUMMARY.md` for detailed results interpretation
- Run `python -i pca/explore_results.py` for interactive data exploration

**All PCA analysis is complete and ready for your research!** 🎉


