# PR: Covariant Coherence Scalar Implementation and Gaia Bulge Regression Test

## Summary

Implements the **covariant coherence scalar** `C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)` from field theory and validates it on Gaia inner-disk data as a proxy for bulge kinematics. This translates the theoretical formulation into a testable regression test.

## Motivation

The baseline coherence scalar `C = v²/(v²+σ²)` is a kinematic approximation. The paper's covariant formulation includes a density term `4πGρ` that should improve predictions in regions where density gradients matter (bulges, inner disks). This PR implements the full covariant form and tests it on Gaia data.

## Implementation

### 1. Core Function: `C_covariant_coherence()`
- **Location**: `scripts/run_regression_experimental.py`
- **Formula**: `C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)`
- **Inputs**: 
  - `omega2_kms2_per_kpc2`: Vorticity squared in (km/s/kpc)²
  - `rho_kg_m3`: Baryonic density in kg/m³
  - `theta2_kms2_per_kpc2`: Expansion squared (optional, defaults to 0)
- **Returns**: Coherence scalar in [0,1]

### 2. Milky Way Density Model: `mw_baryonic_density_kg_m3()`
- **Location**: `scripts/run_regression_experimental.py`
- **Model**: Exponential disk + Hernquist bulge
- **Parameters**: Matches McMillan 2017 (scaled by 1.16)
- **Usage**: Provides `ρ(R,z)` for covariant coherence computation

### 3. Regression Test: `test_gaia_bulge_covariant()`
- **Location**: `scripts/run_regression_experimental.py`
- **Test Design**:
  - Selects inner-disk stars (R < 5 kpc, |z| > 0.3 kpc) as bulge proxy
  - Bins in (R, z) space (5 radial × 2 vertical = 10 bins)
  - Computes ω² from mean v_φ/R
  - Evaluates ρ from MW baryonic model
  - Compares baseline vs covariant coherence predictions
- **Success Criteria**: Improvement > 1.0 km/s RMS
- **Integration**: Added to extended test suite (#9)

## Results

**Test Status**: ✅ PASSED

- **Baseline RMS**: 113.59 km/s
- **Covariant RMS**: 106.74 km/s
- **Improvement**: **6.86 km/s** (well above 1 km/s threshold)
- **Sample**: 734 stars, 10 bins
- **Mean C_cov**: 0.179

## Important Caveats

1. **Not True Bulge Data**: Eilers catalog is disk sample (R: 4-16 kpc). The test uses inner-disk (R < 5 kpc) as a proxy. This validates the math and plumbing but isn't yet "bar/bulge physics."

2. **High Absolute RMS**: 100+ km/s indicates missing systematics (dispersion-dominated, non-axisymmetric, selection-function effects). The win is the **relative improvement**: C_cov contains extra information (ρ term) that baseline coherence didn't.

3. **Translation to SPARC**: Not yet implemented. Should wait for true bulge dataset (R < 3 kpc) before calibrating SPARC proxies.

## Usage

```bash
# Run full test suite (includes Gaia Bulge Covariant in extended tests)
python scripts/run_regression_experimental.py

# Run core tests only
python scripts/run_regression_experimental.py --core

# Run with flow coherence (alternative topology model)
python scripts/run_regression_experimental.py --coherence=flow

# Export SPARC pointwise residuals for analysis
python scripts/run_regression_experimental.py --coherence=c \
  --export-sparc-points=scripts/regression_results/sparc_points_C.csv
```

## Files Changed

- `scripts/run_regression_experimental.py`:
  - Added `C_covariant_coherence()` function
  - Added `mw_baryonic_density_kg_m3()` function
  - Added `test_gaia_bulge_covariant()` test function
  - Integrated test into extended suite

## Next Steps (Future Work)

1. **Obtain True Bulge Dataset** (R < 3 kpc) for full validation
2. **Add Bulge/Bar Realism Corrections** (non-axisymmetry, streaming, dispersion-dominated corrections)
3. **Translate to SPARC** (only after true bulge calibration)

## Acceptance Criteria

- ✅ `C_covariant_coherence()` function implemented and tested
- ✅ `mw_baryonic_density_kg_m3()` provides physically reasonable density
- ✅ `test_gaia_bulge_covariant()` integrated into regression suite
- ✅ Test passes with improvement > 1 km/s
- ✅ All existing tests still pass
- ✅ Documentation complete

## Related Work

- Flow coherence tuning: `FLOW_COHERENCE_TUNING_SUMMARY.md`
- Strategic pivot: `STRATEGIC_PIVOT_GAIA_BULGE.md`
- Test design: `GAIA_BULGE_TEST_DESIGN.md`
- Completion notes: `NEXT_STEPS_COMPLETE.md`

