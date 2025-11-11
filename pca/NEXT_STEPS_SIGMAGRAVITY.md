# Next Steps: Connecting Σ-Gravity to PCA Results

## Overview

You now have empirical PCA basis vectors capturing 96.8% of rotation curve variance. The next step is to test whether your Σ-Gravity model reproduces this empirical structure or leaves systematic residuals in specific PC directions.

## Quick-Start Model Comparison

### Option 1: Using the Built-in Script

If you have Σ-Gravity predictions for SPARC galaxies in CSV format:

```bash
python pca/scripts/08_compare_models.py \
    --pca_npz pca/outputs/pca_results_curve_only.npz \
    --model_csv path/to/sigmagravity_sparc_predictions.csv \
    --out_dir pca/outputs/model_comparison
```

**Expected CSV format**:
- Required columns: `name` (galaxy), `R_kpc` or `R`, `V_model` or `V_pred`
- Optional: `V_obs`, `eV_obs` for residual computation

### Option 2: Custom Analysis

Drop this into `pca/explore_results.py` or a new script:

```python
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Load your Σ-Gravity fits
# Example format: per-galaxy residuals or parameters
model = pd.read_csv('path/to/sigmagravity_fits.csv')
# Columns might include: name, residual_rms, chi2, l0, A, p, n_coh

# Load PCA results
pca = np.load('pca/outputs/pca_results_curve_only.npz', allow_pickle=True)
scores = pca['scores']  # [N_galaxies, N_components]
names = pca['names']

# Merge PC scores with model results
pc_df = pd.DataFrame({
    'name': names,
    'PC1': scores[:, 0],
    'PC2': scores[:, 1],
    'PC3': scores[:, 2]
})
merged = model.merge(pc_df, on='name', how='inner')

print("=" * 70)
print("Σ-GRAVITY MODEL vs PCA STRUCTURE")
print("=" * 70)

# Test 1: Do residuals correlate with PCs?
# (If yes, model is missing physics in that PC direction)
if 'residual_rms' in merged:
    print("\nModel Residuals vs PC Axes:")
    for i in range(1, 4):
        pc_col = f'PC{i}'
        rho, p = spearmanr(merged['residual_rms'], merged[pc_col])
        print(f"  {pc_col}: ρ = {rho:+.3f}, p-value = {p:.3e}")
        if abs(rho) > 0.3 and p < 0.01:
            print(f"    → ⚠️  Significant correlation! Model may miss {pc_col} physics")

# Test 2: Do model parameters track PCs?
# (Shows which PC drives parameter variation)
param_cols = ['l0', 'A', 'p', 'n_coh']  # Adjust to your parameter names
for param in param_cols:
    if param in merged:
        print(f"\nModel parameter '{param}' vs PCs:")
        for i in range(1, 4):
            pc_col = f'PC{i}'
            if np.isfinite(merged[param]).sum() > 10:
                rho, p = spearmanr(merged[param], merged[pc_col])
                print(f"  {pc_col}: ρ = {rho:+.3f}, p-value = {p:.3e}")

# Test 3: Outlier diagnosis
# Do the PCA outliers have bad fits?
outlier_names = ['UGCA281', 'UGC02487', 'NGC6195', 'UGCA444']  # From analysis
outliers = merged[merged['name'].isin(outlier_names)]
if len(outliers) > 0 and 'residual_rms' in outliers:
    print("\nOutlier galaxy fit quality:")
    for _, row in outliers.iterrows():
        print(f"  {row['name']:12s}: residual = {row['residual_rms']:.3f}, " +
              f"PC1 = {row['PC1']:+.2f}, PC2 = {row['PC2']:+.2f}")
```

---

## Interpretation Guide

### Scenario A: Residuals Uncorrelated with PC1
**Result**: ρ(residual, PC1) ≈ 0, p > 0.05

**Interpretation**: ✅ Your model captures the **dominant 79.9% mode** of variation. This is the gold standard—it means Σ-Gravity reproduces the main empirical structure.

**Publication claim**: "Model residuals are uncorrelated with PC1 (ρ=0.XX, p=0.YY), indicating successful capture of the dominant rotation curve morphology."

### Scenario B: Residuals Correlate with PC2
**Result**: ρ(residual, PC2) > 0.3, p < 0.01

**Interpretation**: ⚠️ Model misses **scale-dependent** effects (PC2 ∝ Rd, Mbar). This suggests:
- Coherence scale ℓ₀ may need mass or size dependence
- Kernel shape parameters (p, n_coh) may vary with galaxy scale

**Next step**: Test if adding `ℓ₀(Mbar)` or `ℓ₀(Rd)` scaling reduces correlation.

### Scenario C: Residuals Correlate with PC3
**Result**: ρ(residual, PC3) < -0.3, p < 0.01

**Interpretation**: ⚠️ Model misses **density-dependent** effects. PC3 anti-correlates with Σ₀, Mbar/Rd², suggesting:
- High-density (compact, massive) galaxies fit differently than low-density (extended, dwarf)
- May need surface-density-dependent kernel or decoherence

**Next step**: Stratify fits by Σ₀ or Mbar and check if residuals persist.

### Scenario D: Parameters Track Specific PCs
**Result**: ℓ₀ ∝ PC2 (ρ > 0.5), A ∝ PC1 (ρ > 0.5)

**Interpretation**: ✅ Your model's **free parameters naturally align with empirical structure**:
- ℓ₀ tracks scale (PC2) → physically motivated
- A tracks mass-velocity (PC1) → expected amplitude scaling

**Publication claim**: "Model coherence scale ℓ₀ correlates strongly with PC2 (ρ=0.XX), the scale-length axis, supporting physical interpretation of the parameter."

---

## Expected CSV Formats

### Input: Σ-Gravity Predictions
One of these formats:

**Format 1**: Per-galaxy residual summary
```csv
name,residual_rms,chi2,l0,A,p,n_coh
NGC3198,12.5,1.2,3.2,0.15,2.1,1.8
NGC2403,8.3,0.9,2.8,0.12,2.0,1.9
...
```

**Format 2**: Full curve predictions
```csv
name,R_kpc,V_obs,eV_obs,V_model
NGC3198,1.0,90.0,5.0,88.5
NGC3198,2.0,120.0,4.0,118.2
NGC2403,0.5,45.0,8.0,47.1
...
```

If using Format 2, compute residuals first:
```python
model['residual'] = model['V_obs'] - model['V_model']
residuals = model.groupby('name')['residual'].apply(
    lambda x: np.sqrt(np.mean(x**2))
).reset_index(name='residual_rms')
```

---

## Visualization

### Plot 1: Residual vs PC1
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(merged['PC1'], merged['residual_rms'], alpha=0.6, s=50)
ax.set_xlabel('PC1 Score', fontsize=12)
ax.set_ylabel('Σ-Gravity Residual RMS (km/s)', fontsize=12)
ax.set_title('Model Residuals vs Dominant Morphology Axis', fontsize=14)
ax.grid(alpha=0.3)

# Add Spearman correlation
rho, p = spearmanr(merged['PC1'], merged['residual_rms'])
ax.text(0.05, 0.95, f'ρ = {rho:+.3f}\np = {p:.2e}',
        transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
plt.tight_layout()
plt.savefig('pca/outputs/model_comparison/residuals_vs_PC1.png', dpi=150)
plt.show()
```

### Plot 2: Parameter Map in PC Space
```python
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(merged['PC1'], merged['PC2'], 
                     c=merged['l0'], cmap='viridis', s=80, alpha=0.7)
ax.set_xlabel('PC1 (79.9% var)', fontsize=12)
ax.set_ylabel('PC2 (11.2% var)', fontsize=12)
ax.set_title('Coherence Scale ℓ₀ in PC Space', fontsize=14)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('ℓ₀ (kpc)', fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pca/outputs/model_comparison/l0_in_PC_space.png', dpi=150)
plt.show()
```

---

## Robustness Checks

### 1. Bootstrap Correlation Uncertainty
```python
from scipy.stats import bootstrap

def correlation_stat(x, y):
    return spearmanr(x, y)[0]

# Bootstrap confidence interval for ρ(residual, PC1)
res = bootstrap(
    (merged['residual_rms'].values, merged['PC1'].values),
    lambda x, y: correlation_stat(x, y),
    n_resamples=1000, method='percentile'
)
print(f"95% CI for ρ: [{res.confidence_interval.low:.3f}, {res.confidence_interval.high:.3f}]")
```

### 2. Outlier Sensitivity
Re-run correlation after removing the 8 PCA outliers:
```python
outlier_names = ['UGCA281', 'NGC0055', 'NGC2976', 'UGC07524', 
                 'UGC02487', 'NGC6195', 'UGC06667', 'UGCA444']
merged_clean = merged[~merged['name'].isin(outlier_names)]
rho_clean, p_clean = spearmanr(merged_clean['residual_rms'], merged_clean['PC1'])
print(f"Without outliers: ρ = {rho_clean:+.3f}, p = {p_clean:.3e}")
```

### 3. Subset Analysis
Test if correlations hold across HSB/LSB or mass bins:
```python
meta = pd.read_csv('pca/data/raw/metadata/sparc_meta.csv')
merged = merged.merge(meta[['name', 'HSB_LSB', 'Mbar']], on='name')

for pop in ['HSB', 'LSB']:
    subset = merged[merged['HSB_LSB'] == pop]
    rho, p = spearmanr(subset['residual_rms'], subset['PC1'])
    print(f"{pop}: ρ = {rho:+.3f}, p = {p:.3e}, n = {len(subset)}")
```

---

## Publication-Quality Table

Once you have results, format as:

| PC Axis | Morphology | Residual Correlation | Parameter Correlation | Interpretation |
|---------|------------|----------------------|-----------------------|----------------|
| PC1     | Mass-velocity shape (79.9%) | ρ = -0.05, p = 0.52 | A: ρ = +0.68, p < 0.001 | ✅ Model captures dominant mode; amplitude tracks PC1 |
| PC2     | Scale-length (11.2%) | ρ = +0.28, p = 0.001 | ℓ₀: ρ = +0.55, p < 0.001 | ⚠️ Modest scale-dependent residual; ℓ₀ tracks scale as expected |
| PC3     | Density residual (5.7%) | ρ = -0.15, p = 0.05 | p: ρ = -0.42, p < 0.001 | Weak density effect; kernel shape compensates |

---

## Bottom Line

**The goal**: Show that Σ-Gravity residuals are **uncorrelated with PC1** (the 79.9% mode), proving the model captures the dominant empirical physics. Any correlation with PC2 or PC3 is interpretable and actionable—it tells you exactly which physical effects (scale, density) need refinement.

This is a **powerful validation strategy** because PCA provides a model-independent empirical target. If your model explains PC1, that's a major success regardless of theoretical derivation uncertainties.


