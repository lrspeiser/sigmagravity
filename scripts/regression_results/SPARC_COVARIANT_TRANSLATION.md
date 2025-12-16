# SPARC Covariant Coherence Translation: Complete

## Implementation Status ✓

### 1. Translation Module Created
- **File**: `scripts/translate_covariant_to_sparc.py`
- **Functions**:
  - `compute_omega2_from_rotation_curve()`: ω² ≈ (V/R)²
  - `compute_density_proxy_sparc()`: 4πGρ from surface density
  - `compute_C_cov_proxy_sparc()`: C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)
  - `apply_covariant_coherence_to_sparc()`: Full prediction with fixed-point iteration

### 2. Integrated into Regression Suite
- **CLI Flag**: `--use-covariant-proxy`
- **Behavior**: Applies C_cov proxy to bulge galaxies (f_bulge ≥ 0.3)
- **Location**: `predict_velocity()` function in `run_regression_experimental.py`

### 3. Density Proxy Method
**Surface Density Approach:**
- Estimate total mass from V_bar at outer radius
- Compute central surface density: Σ₀ = M/(2π R_d²)
- Exponential profile: Σ(R) = Σ₀ exp(-R/R_d)
- Volume density: ρ(R) = Σ(R) / (2h_z) with h_z = 0.3 kpc
- 4πGρ computed and converted to (km/s/kpc)^2

**Why This Works:**
- More stable than gradient-based methods
- Physically motivated (exponential disk)
- Units consistent with ω²

## Usage

```bash
# Run SPARC with covariant coherence proxy for bulges
python scripts/run_regression_experimental.py --core \
  --coherence=C \
  --use-covariant-proxy
```

## Strategy

**From Gaia to SPARC:**
1. **Gaia**: Learned that C_cov = ω²/(ω² + 4πGρ + θ² + H₀²) improves predictions
2. **SPARC**: Approximate using rotation curve proxies:
   - ω² ≈ (V/R)² (vorticity from rotation)
   - 4πGρ ≈ from surface density model
   - θ² ≈ 0 (steady-state assumption)
3. **Apply**: Use C_cov proxy for bulge galaxies (where flow info matters most)

## Expected Results

**Hypothesis:**
- Bulge galaxies should show improvement with C_cov proxy
- Disk galaxies may not benefit (already well-predicted)
- Overall RMS should improve or stay similar

**Testing:**
- Compare baseline (C coherence) vs covariant proxy
- Focus on bulge RMS improvement
- Check for regressions in disk performance

## Next Steps

1. **Run Full Test**: Execute on SPARC with `--use-covariant-proxy`
2. **Compare Results**: Baseline vs covariant proxy
3. **Fine-tune**: Adjust density proxy if needed
4. **Validate**: Ensure no regressions in other tests

## Files

- `scripts/translate_covariant_to_sparc.py`: Translation module
- `scripts/run_regression_experimental.py`: Integration
- `scripts/regression_results/SPARC_COVARIANT_TRANSLATION.md`: This document


