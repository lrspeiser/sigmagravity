# BRAVA 6D Flow Topology Workflow

Complete workflow for processing BRAVA-matched Gaia 6D data and computing flow topology invariants for Σ-Gravity validation.

## Overview

This workflow transforms BRAVA-matched Gaia 6D stars (positions + velocities) into:
1. **Galactocentric coordinates** (R, φ, z, vR, vφ, vz)
2. **Flow topology invariants** (ω², θ²) from velocity gradients
3. **Covariant coherence** (C_cov) and enhancement factor (Σ)
4. **Regression metrics** comparing predicted vs observed velocity dispersions

## Prerequisites

- BRAVA catalog: `data/bulge_kinematics/BRAVA/brava_catalog.tbl`
- Python packages: `astropy`, `pandas`, `numpy`, `scikit-learn` (for 3D gradients)
- Gaia TAP access (for downloading 6D stars)

## Step 1: Download BRAVA-Matched Gaia 6D Stars

Download only Gaia stars that match BRAVA via 2MASS IDs:

```bash
python scripts/download_gaia_6d_full.py \
    --brava-catalog data/bulge_kinematics/BRAVA/brava_catalog.tbl \
    --output data/gaia/6d_brava_full.fits
```

**Expected result:**
- ~6,189 stars (72% of BRAVA catalog)
- All stars have complete 6D information (radial velocity required)
- Output: `data/gaia/6d_brava_full.fits`

## Step 2: Sanity Check Downloaded Data

Verify the downloaded FITS file has all required columns:

```bash
python scripts/check_brava_gaia.py data/gaia/6d_brava_full.fits
```

**Expected output:**
- All 6D columns present and valid (ra, dec, parallax, pmra, pmdec, radial_velocity)
- tmass_id column present for cross-reference

## Step 3: Process 6D Data → Flow Topology

Transform to Galactocentric coordinates and compute flow invariants.

### Option A: Axisymmetric Approximation (Fast, Robust)

```bash
python scripts/process_brava_6d_flow.py \
    --input data/gaia/6d_brava_full.fits \
    --output data/gaia/6d_brava_galcen.parquet \
    --min-stars-per-bin 50
```

**What it does:**
- Transforms to Galactocentric (R, φ, z, vR, vφ, vz)
- Bins stars in (R,z) with minimum 50 stars per bin
- Computes ω² ≈ (v_φ/R)² (axisymmetric approximation)
- Computes θ² ≈ 0 (steady-state assumption)
- Computes baryonic density ρ(R,z) from MW model
- Computes C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)
- Computes Σ = 1 + A₀ × C_cov × h(g)

**Output:** `data/gaia/6d_brava_galcen.parquet` with binned data

### Option B: 3D Velocity Gradients (Level B - Full Topology)

```bash
python scripts/process_brava_6d_flow.py \
    --input data/gaia/6d_brava_full.fits \
    --output data/gaia/6d_brava_galcen_3d.parquet \
    --min-stars-per-bin 50 \
    --use-3d-gradients \
    --n-neighbors 50
```

**What it does:**
- Same as Option A, but computes ω² and θ² from actual 3D velocity gradients
- Uses KNN to find nearest neighbors for each star
- Fits local linear velocity field: v(x) ≈ v₀ + ∇v · (x - x₀)
- Extracts vorticity (ω² = |∇×v|²) and expansion (θ² = (∇·v)²) from gradient tensor
- Captures local flow structure beyond axisymmetric approximation

**Output:** `data/gaia/6d_brava_galcen_3d.parquet` with binned data

**Note:** 3D gradients take longer to compute (~5-10 minutes for 3k stars) but provide more detailed flow topology.

## Step 4: Run Regression Test

Test Σ-Gravity predictions against observed velocity dispersions (appropriate for bulge kinematics).

### Axisymmetric Test

```bash
python scripts/test_brava_6d_covariant.py
```

### 3D Gradient Test

```bash
python scripts/test_brava_6d_covariant.py --3d
```

**What it does:**
- Loads binned data from Step 3
- Computes predicted velocity dispersion: σ_tot = 0.51 × V_circ (calibrated from BRAVA)
- Compares predicted vs observed σ_tot across bins
- Reports RMS residual and improvement over baseline

**Expected results:**
- RMS(σ_tot) ≈ 15-16 km/s (much better than v_φ comparison ~209 km/s)
- Both methods perform similarly (axisymmetric slightly better)
- Small improvement from Σ enhancement (~1.009-1.024) is consistent with physics

## Step 5: Compare Methods

Compare axisymmetric vs 3D gradient results side-by-side:

```bash
python scripts/compare_flow_methods.py
```

**Output shows:**
- Mean/std/min/max for ω², θ², C_cov, Σ
- Ratios and differences between methods
- Helps decide if 3D gradients add real signal or mostly noise

## Output Files

### Processed Data (Parquet)

**Columns in binned output:**
- `R_kpc`, `z_kpc` - Bin centers
- `R_min`, `R_max`, `z_min`, `z_max` - Bin boundaries
- `n_stars` - Number of stars per bin
- `vR_mean`, `vR_std` - Radial velocity statistics
- `vphi_mean`, `vphi_std` - Azimuthal velocity statistics
- `vz_mean`, `vz_std` - Vertical velocity statistics
- `omega2` - Vorticity squared (km/s/kpc)²
- `theta2` - Expansion squared (km/s/kpc)²
- `rho_kg_m3` - Baryonic density (kg/m³)
- `C_cov` - Covariant coherence scalar [0,1]
- `Sigma` - Enhancement factor

### Test Results

**Metrics reported:**
- `RMS(sigma_tot)` - Primary metric (velocity dispersion)
- `RMS(v_phi)` - Secondary metric (mean streaming velocity)
- `improvement` - Difference from baseline (measures Σ effects)
- `mean_C_cov`, `mean_Sigma` - Average coherence and enhancement

## Interpretation

### Flow Invariants

- **ω² (vorticity)**: Measures local rotation strength
  - Axisymmetric: ω² ≈ (v_φ/R)²
  - 3D gradients: ω² = |∇×v|² (captures local structure)
  
- **θ² (expansion)**: Measures flow divergence
  - Axisymmetric: θ² ≈ 0 (steady-state)
  - 3D gradients: θ² = (∇·v)² (captures expansion/contraction)

### Covariant Coherence

- **C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)**
  - Measures flow coherence relative to density and expansion
  - Higher C_cov → more coherent flow → larger Σ enhancement
  - Axisymmetric: C_cov ≈ 0.25
  - 3D gradients: C_cov ≈ 0.68 (captures more local structure)

### Enhancement Factor

- **Σ = 1 + A₀ × C_cov × h(g)**
  - Small enhancement in bulge: Σ ≈ 1.009-1.024
  - Consistent with bulge kinematics (non-circular orbits, high dispersion)

### Test Results

- **RMS(σ_tot) ≈ 15-16 km/s**: Good fit to observed dispersions
- **Improvement ≈ -0.2 to -0.5 km/s**: Small, consistent with small Σ
- **Baseline uses same calibration**: Test isolates Σ effects, not calibration differences

## Next Steps

1. **Use BRAVA bins as calibration target** for universal coherence parameterization
2. **Fit one parameterization** across:
   - Gaia disk (Eilers)
   - BRAVA bulge (this workflow)
   - SPARC galaxies (via proxies)
3. **Refine gradient computation** if needed (distance weighting, regularization)
4. **Add to main regression suite** for automated testing

## Troubleshooting

### Large gradient values (3D method)

If ω² or θ² are very large (>10⁶), check:
- Neighbor distances (may be too small)
- Coordinate units (should be kpc for positions, km/s for velocities)
- Numerical stability (try fewer neighbors or add regularization)

### Low match rate

If <50% of BRAVA stars match:
- Check 2MASS ID format in BRAVA catalog
- Verify Gaia cross-match table is accessible
- Check quality cuts (RUWE, visibility periods)

### No bins created

If processing creates 0 bins:
- Reduce `--min-stars-per-bin` (try 30 or 20)
- Check distance filtering (parallax > 0.1 mas)
- Verify coordinate transformation succeeded

## References

- BRAVA catalog: [Howard et al. 2008, 2009]
- Gaia DR3: [Gaia Collaboration 2023]
- Covariant coherence theory: See `docs/sigmagravity_paper.tex`

