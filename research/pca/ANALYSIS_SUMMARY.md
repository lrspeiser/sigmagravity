# PCA Analysis of SPARC Rotation Curves

## Summary

Successfully integrated the SPARC rotation curve dataset with the PCA toolkit. Analyzed 170 galaxies (out of 174 total, 4 skipped due to missing data).

## Data Preparation

### Conversion
- Converted 175 SPARC `.dat` rotation curve files to CSV format
- Parsed the SPARC MasterSheet metadata for 174 galaxies
- Created normalized, resampled curve matrices on a common R/Rd grid (K=50 points, 0.2-6.0 R/Rd)

### Dataset Statistics
- **Rd range**: 0.18 - 18.76 kpc (disk scale lengths)
- **Vf range**: 0.00 - 332.00 km/s (flat velocities)
- **Mbar range**: 0.04 - 262.94 × 10⁹ M_sun (baryonic masses)
- **HSB galaxies**: 98 (high surface brightness)
- **LSB galaxies**: 76 (low surface brightness)

## PCA Results

### Variance Explained
The first 10 principal components explain the following fractions of variance:

| PC  | Variance Explained | Cumulative |
|-----|-------------------|------------|
| PC1 | 79.48%            | 79.48%     |
| PC2 | 11.88%            | 91.36%     |
| PC3 | 5.16%             | 96.51%     |
| PC4 | 3.13%             | 99.65%     |
| PC5 | 0.22%             | 99.87%     |
| PC6 | 0.09%             | 99.96%     |
| PC7 | 0.03%             | 99.99%     |
| PC8 | 0.01%             | 100.00%    |
| PC9 | 0.00%             | 100.00%    |
| PC10| 0.00%             | 100.00%    |

**Key Finding**: The first 3 PCs capture **96.5%** of the variance in galaxy rotation curves, suggesting a low-dimensional latent structure.

## Clustering Analysis

### Gaussian Mixture Model in PC Space
- **Optimal clusters (k)**: 6 (selected by BIC = 905.8)
- **Cluster distribution**:
  - Cluster 0: 22 galaxies
  - Cluster 1: 1 galaxy
  - Cluster 2: 3 galaxies
  - Cluster 3: 1 galaxy
  - Cluster 4: 2 galaxies
  - Cluster 5: 141 galaxies (dominant cluster)

**Note**: One dominant cluster (5) contains most galaxies, suggesting a common rotation curve morphology with a few outliers in smaller clusters.

## Subset Stability Analysis

### HSB vs LSB Galaxies
Tested whether high surface brightness (HSB) and low surface brightness (LSB) galaxies span similar PC subspaces.

**Principal angles** (3-dimensional subspace comparison):
- Angle 1: 0.071 radians (4.1°) - **very small**, near-aligned
- Angle 2: 0.325 radians (18.6°)
- Angle 3: 0.777 radians (44.5°)

**Interpretation**: The first principal angle is very small, indicating that the primary mode of variation (PC1) is nearly identical between HSB and LSB galaxies. The higher-order PCs show more divergence, suggesting different secondary structure.

## Output Files

### Data Files
- `pca/data/processed/sparc_curvematrix.npz` - Normalized curve matrix (170 × 50)
- `pca/data/processed/sparc_features_curve_only.npz` - Feature matrix (curves only)
- `pca/data/processed/sparc_features_curve_plus_scalars.npz` - Feature matrix (curves + scalars)
- `pca/outputs/pca_results_curve_only.npz` - PCA results (components, scores, loadings)

### Analysis Results
- `pca/outputs/pca_explained_curve_only.csv` - Explained variance ratios
- `pca/outputs/clusters.csv` - Cluster assignments for each galaxy
- `pca/outputs/subset_principal_angles.txt` - HSB/LSB subspace comparison

### Figures
- `pca/outputs/figures/scree_cumulative.png` - Scree plot showing cumulative variance
- `pca/outputs/figures/pc1_radial_loading.png` - PC1 radial loading profile
- `pca/outputs/figures/pc2_radial_loading.png` - PC2 radial loading profile
- `pca/outputs/figures/pc3_radial_loading.png` - PC3 radial loading profile
- `pca/outputs/pc_scatter_clusters.png` - 2D scatter plot of galaxies in PC space with cluster colors

## Next Steps (Optional)

1. **Correlations with physics**: Examine how PC scores correlate with Mbar, Σ₀, Rd, Vf
2. **Autoencoder comparison**: Train a non-linear autoencoder (PyTorch) to test if non-linear structure exists beyond PCA
3. **Model comparison**: Use `08_compare_models.py` to see how Σ-Gravity model residuals project onto the PCA axes
4. **Dwarf vs massive**: Repeat subset stability analysis for dwarf vs massive galaxies

## Interpretation for Σ-Gravity

The PCA analysis reveals:

1. **Low-dimensional structure**: Only 3 PCs needed to explain 96.5% of rotation curve variance
2. **Dominant morphology**: Most galaxies (141/170) fall into one main cluster
3. **Universal first mode**: PC1 is nearly identical between HSB and LSB galaxies (4.1° separation)
4. **Secondary variations**: PC2 and PC3 show more morphological diversity

This suggests that:
- There may be a **universal rotation curve shape** (PC1) with systematic variations (PC2-3)
- Σ-Gravity predictions could be tested against these empirical PCs
- Model residuals could reveal whether Σ-Gravity captures the main modes of variation

## References

- **SPARC dataset**: Lelli, McGaugh, Schombert (2016) - "SPARC: Mass Models for 175 Disk Galaxies"
- **PCA toolkit**: Custom rotation-curve PCA implementation with weighted standardization


