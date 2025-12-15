# Residual Discovery Pipeline - Summary

## Overview

Three scripts have been added to discover what drives residuals in SPARC and Gaia data:

1. **`discover_sparc_residual_drivers.py`** - Feature importance discovery on SPARC pointwise residuals
2. **`export_gaia_pointwise_features.py`** - Export Gaia star-level flow features (vorticity, shear, etc.)
3. **`discover_gaia_flow_drivers.py`** - Discover which flow features drive Gaia residuals

## Key Hypothesis

**Coherence is not set by speed or dispersion alone — it's set by 3D flow topology**

- Rotating disk: laminar, single-sign vorticity (aligned angular momentum vectors)
- Bulge: multi-stream, pressure-supported, anisotropic (many orbit families, phase-mixing)

The "universal principle" candidate:
> **Gravity coheres when the local baryonic flow is vorticity-dominated and slowly varying across the coherence length; it decoheres (screens) when shear/tidal gradients dominate.**

This would naturally explain:
- **Cassini/Solar system**: static bulk flow → vorticity ~ 0 → coherence ~ 0 → no modification
- **Disks**: strong organized rotation → high vorticity fraction → coherence on
- **Bulges**: high shear / multi-stream / rapid phase mixing → coherence reduced or sign-reversed (screening)
- **Clusters**: bulk flows / coherent infall → nonzero vorticity-dominated regions

## Usage

### SPARC Residual Discovery

```bash
# All SPARC points
python scripts/discover_sparc_residual_drivers.py \
  scripts/regression_results/sparc_pointwise_baseline.csv \
  --subset all --target dSigma --model xgb --splits 3 --perm-repeats 3 \
  --outdir scripts/regression_results/residual_drivers

# High-bulge regions only
python scripts/discover_sparc_residual_drivers.py \
  scripts/regression_results/sparc_pointwise_baseline.csv \
  --subset highbulge --target dSigma --model xgb --splits 3 --perm-repeats 3 \
  --outdir scripts/regression_results/residual_drivers

# Bulge galaxies
python scripts/discover_sparc_residual_drivers.py \
  scripts/regression_results/sparc_pointwise_baseline.csv \
  --subset bulgegal --target dSigma --model xgb --splits 3 --perm-repeats 3 \
  --outdir scripts/regression_results/residual_drivers
```

### Gaia Flow Feature Export

```bash
python scripts/export_gaia_pointwise_features.py \
  data/gaia/eilers_apogee_6d_disk.csv \
  --out scripts/regression_results/gaia_pointwise_features.csv \
  --k 64 \
  --compute-sigma-gravity \
  --regression-script scripts/run_regression_experimental.py
```

### Gaia Flow Driver Discovery

```bash
python scripts/discover_gaia_flow_drivers.py \
  scripts/regression_results/gaia_pointwise_features.csv \
  --target resid_kms \
  --splits 5 --perm-repeats 6 \
  --outdir scripts/regression_results/gaia_flow_drivers \
  --shap
```

## What to Look For

### SPARC Discovery

If bulges are revealing a missing variable, residuals should correlate with:
- **Curvature/gradient information**: `dlnGbar_dlnR`, `dlnVbar_dlnR`
- **Morphology**: `f_bulge_r`, `f_bulge_global`
- **Timescale**: `Omega`, `tau_dyn`
- **Tidal/decoherence proxy**: `tidal_L0`

### Gaia Discovery

If "coherence" is fundamentally a **flow-topology** phenomenon, residuals should correlate more with:
- **Vorticity vs shear**: `log10_omega` vs `log10_shear`
- **Anisotropy of dispersion tensor**: `anisotropy_sigma`
- **Local density/separation**: `log10_n_star` / `log10_d_star`

...than with just `R` and `|z|`.

## Next Steps

Once we identify which invariants matter, we can implement a new coherence mode:

- `--coherence=flow`
- Replace scalar coherence with a **vorticity fraction**:
  ```
  C_flow = ω² / (ω² + α·s² + β·θ² + γ·T² + ...)
  ```
  where:
  - ω² from vorticity (curl)
  - s² from shear
  - θ² from divergence/expansion
  - T² from tidal/dephasing proxy

This would be a structural change: coherence becomes a field-property of the **velocity-gradient tensor**, not a scalar based on "how fast are things moving".

## Files Created

- `scripts/discover_sparc_residual_drivers.py`
- `scripts/export_gaia_pointwise_features.py`
- `scripts/discover_gaia_flow_drivers.py`
- Output directories: `scripts/regression_results/residual_drivers/` and `scripts/regression_results/gaia_flow_drivers/`


