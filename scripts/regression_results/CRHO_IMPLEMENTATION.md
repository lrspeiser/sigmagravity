# CRHO (Density/Vorticity Coherence) Implementation

## Concept

The coherence scalar C = v²/(v²+σ²) is too easy to saturate to C≈1 in dense, compact, high-shear regions (bulges). The covariant form should include a density term:

**C ≈ ω²/(ω² + 4πGρ + H₀²)**

For circular flow with ω ~ v/r, this becomes:

**C ≈ v²/(v² + (4πGρ + H₀²)r²)**

Therefore, we define a density-dependent effective dispersion:

**σ_ρ²(r) = (4πGρ(r) + H₀²)r²**

And use:

**σ_eff²(r) = σ_base² + σ_ρ²(r)**

This suppresses C in high-density (bulge) regions while keeping it high in low-density (disk outskirts) regions.

## Implementation

### 1. `sigma_rho_profile_kms()` function
- Computes σ_ρ from baryonic density proxy (reuses `rho_proxy_from_vbar()`)
- Applies scaling factor `CRHO_SCALE = 0.05` to prevent over-suppression
- Caps at 150 km/s to prevent complete suppression of C

### 2. Density-dependent application
- Only applies σ_ρ in high-density regions (bulges)
- Uses density threshold `rho_threshold = 5e-19 kg/m³` (between disk and bulge)
- Scales σ_ρ by `density_factor = min(rho/rho_threshold, 2.0)`

### 3. CLI integration
- `--coherence=crho` enables CRHO mode
- Also accepts `--coherence=c_rho` or `--coherence=rho`

## Current Results

**Baseline (C mode):**
- Overall RMS: 17.42 km/s
- Bulge RMS: 28.93 km/s
- Disk RMS: 16.06 km/s

**CRHO mode (with density-dependent scaling):**
- Overall RMS: 17.41 km/s (slight improvement)
- Bulge RMS: 28.93 km/s (no change)
- Disk RMS: 16.06 km/s (no change)

## Status

CRHO is implemented and working, but **not yet improving bulge predictions**. The density-dependent scaling ensures it doesn't hurt, but we need to tune parameters to get improvement.

## Next Steps

1. **Tune CRHO_SCALE**: Try values 0.1, 0.2, 0.3 to increase suppression in bulges
2. **Tune rho_threshold**: Lower it to affect more regions, or raise it to be more selective
3. **Remove density-dependent scaling**: Apply σ_ρ everywhere and see if that helps
4. **Check density proxy**: Verify that `rho_proxy_from_vbar()` correctly identifies bulge regions

## Files Modified

- `scripts/run_regression_experimental.py`:
  - Added `sigma_rho_profile_kms()` function
  - Added CRHO to coherence model selection
  - Modified `predict_velocity()` to use σ_eff when CRHO is selected



