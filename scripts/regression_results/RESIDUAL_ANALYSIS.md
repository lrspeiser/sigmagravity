# Residual Analysis: Flow Coherence vs Baseline

## Overview

Analysis of pointwise residuals comparing baseline (C coherence) vs flow coherence model to identify which flow invariants correlate with prediction errors.

## Key Findings

### 1. Flow Invariant Correlations with Residuals

**Correlation with absolute residual:**
- `omega2` (vorticity): r=0.151 (p=1e-18) - **Strongest positive correlation**
- `shear2`: r=0.109 (p=2e-10) - Moderate correlation
- `tidal2`: r=0.114 (p=4e-11) - Moderate correlation
- `C_term` (flow coherence): r=-0.148 (p=6e-18) - **Negative correlation** (higher coherence = lower residual)

**Correlation with dSigma (Sigma_req - Sigma_pred):**
- `C_term`: r=-0.171 (p=2e-23) - **Strongest correlation**
- Flow invariants show stronger correlations with dSigma than with velocity residuals
- Suggests flow coherence affects the enhancement factor more directly

**Key Insight**: Higher flow coherence (C_term) correlates with **lower** residuals, confirming the model is working as intended.

### 2. Bulge vs Disk Differences

**Bulge galaxies (n=799):**
- `omega2`: r=0.174 (stronger than disk) - Vorticity is more important for bulges
- `shear2`: r=-0.002 (no correlation) - Shear doesn't matter for bulges
- `tidal2`: r=0.067 (weak) - Tidal effects present but weak
- `C_term`: r=-0.143 (negative, as expected)

**Disk galaxies (n=2574):**
- `omega2`: r=0.101 (weaker than bulge) - Vorticity less important
- `shear2`: r=0.141 (stronger than bulge) - **Shear matters more for disks**
- `tidal2`: r=0.001 (no correlation) - Tidal effects negligible
- `C_term`: r=-0.164 (stronger negative correlation)

**Key Insight**: Bulge and disk galaxies have **different flow topology signatures**:
- Bulges: Vorticity-dominated, shear irrelevant
- Disks: Shear matters, vorticity less important

### 3. Baseline vs Flow Performance

- Flow performs within 0.67 km/s (3.8%) of baseline overall
- Flow improves predictions at ~40-50% of individual points
- Improvement varies by galaxy type (bulge vs disk)

### 4. Composite Features

- `omega_over_shear`: Ratio of vorticity to shear - r=-0.008 (no correlation)
- `vorticity_dominance`: Fraction of energy in vorticity - r=-0.143 (p=8e-17)
- `C_term`: r=-0.148 (p=6e-18) - Best predictor

**Worst 10% residuals:**
- Mean omega2: 31,585 vs overall: 10,174 (3× higher!)
- Mean shear2: 6,073 vs overall: 2,697 (2.2× higher)
- Mean tidal2: 6,602 vs overall: 2,867 (2.3× higher)
- Mean C_term: 0.920 vs overall: 0.958 (lower coherence)

**Key Insight**: Worst residuals occur where flow invariants are **high** but coherence is **low** - suggesting the flow coherence model is correctly identifying problematic regions.

## Next Steps

1. **Gradient analysis**: Investigate correlation between flow invariants and gradient proxies (dlnVbar_dlnR, dlnGbar_dlnR)
2. **Radial trends**: Analyze how flow invariants vary with radius and correlate with residuals
3. **Galaxy-specific patterns**: Identify which galaxies benefit most from flow coherence
4. **Feature engineering**: Create better composite features from flow invariants

