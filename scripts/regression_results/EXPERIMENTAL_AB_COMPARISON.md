# Experimental A/B Test: C vs JJ Coherence Models

**Date:** 2025-12-12  
**Test:** Core regression suite (8 tests)  
**Script:** `run_regression_experimental.py`

## Summary

This document compares the baseline **C coherence model** (`C = v²/(v²+σ²)`) with the experimental **JJ coherence model** (current-current correlation `Q_JJ(r)`).

## Key Results

### SPARC Galaxies (171 rotation curves)

| Metric | C Model (Baseline) | JJ Model | Change |
|--------|-------------------|---------|--------|
| **Mean RMS** | 17.42 km/s | 22.94 km/s | +31.7% (worse) |
| **Win Rate vs MOND** | 42.7% | 26.3% | -16.4% (worse) |
| **RAR Scatter** | 0.100 dex | 0.151 dex | +51% (worse) |
| **Status** | ✓ Passed | ✗ Failed | |

**Benchmarks:**
- MOND RMS: 17.15 km/s
- ΛCDM RMS: ~15 km/s (with 2-3 params/galaxy)
- Target: < 20 km/s

### Galaxy Clusters (42 Fox+ 2022)

| Metric | C Model | JJ Model | Change |
|--------|---------|---------|--------|
| **Median Ratio** | 0.987 | 0.987 | No change |
| **Scatter** | 0.132 dex | 0.132 dex | No change |
| **Status** | ✓ Passed | ✓ Passed | |

**Note:** Clusters use `q_cluster = 1.0` for both models (large correlation volume).

### Other Tests

| Test | C Model | JJ Model |
|------|---------|---------|
| Cluster Holdout | ✓ Passed | ✓ Passed |
| Gaia/MW (28,368 stars) | ✓ RMS=29.8 km/s | ✓ RMS=28.1 km/s |
| Redshift Evolution | ✓ Passed | ✓ Passed |
| Solar System (Cassini) | ✓ |γ-1|=1.77e-09 | ✓ |γ-1|=0.00e+00 |
| Counter-Rotation | ✓ Passed | ✓ Passed |
| Tully-Fisher | ✓ Passed | ✓ Passed |

**Total:** C Model: 8/8 passed | JJ Model: 7/8 passed

## Physical Interpretation

### C Model (Baseline)
- **Coherence:** `C = v²/(v²+σ²)` - instantaneous ordered/total kinetic energy ratio
- **Interpretation:** Measures local phase-space coherence from velocity dispersion
- **Performance:** Matches MOND on SPARC, works well for galaxies

### JJ Model (Experimental)
- **Coherence:** `Q_JJ(r) = <J>_K² / <J² + J_rand²>_K` - scale-dependent current-current correlation
- **Interpretation:** Measures long-range correlation of baryonic mass-current
- **Performance:** Worse on SPARC, but explicitly suppresses Solar System (coherence → 0)
- **Parameters:**
  - `JJ_XI_MULT = 1.0` (correlation scale = R_d/(2π))
  - `JJ_SMOOTH_M_POINTS = 5` (smoothing for density proxy)

## Discussion

### Why JJ Model Performs Worse on SPARC

1. **Different physical mechanism:** JJ model depends on spatial correlation of mass-current, not just local velocity dispersion
2. **Density proxy limitations:** Uses spherical inversion `ρ ~ dM_enc/dr` from `V_bar`, which is approximate for disk galaxies
3. **Parameter tuning:** Current `JJ_XI_MULT = 1.0` may not be optimal for galaxy-scale correlations

### Potential Improvements

1. **Tune correlation scale:** Try `JJ_XI_MULT = 0.5` or `2.0` to adjust `ξ_corr`
2. **Better density proxy:** Use actual disk density profiles instead of spherical inversion
3. **Hybrid approach:** Use JJ for large-scale systems (clusters), C for galaxies

### Solar System Suppression

The JJ model explicitly sets `γ-1 = 0` for Solar System (no macroscopic baryonic current network), which is physically motivated. The C model relies on small `h(g_N)` alone.

## Conclusion

**Current recommendation:** Keep the **C model** as baseline for SPARC galaxies. The JJ model shows promise for:
- Explicit Solar System suppression (coherence → 0)
- Potential for better cluster physics (though current results are identical)
- Different physical interpretation (current-current correlations)

**Next steps:**
1. Tune `JJ_XI_MULT` and smoothing parameters
2. Test with component-mixed σ profiles (`--sigma-components`)
3. Consider hybrid model: C for galaxies, JJ for clusters/solar system

## Files

- Baseline report: `experimental_report_C.json`
- JJ model report: `experimental_report_JJ.json`
- Script: `scripts/run_regression_experimental.py`

## Usage

```bash
# Baseline (C model)
python scripts/run_regression_experimental.py --core

# JJ model
python scripts/run_regression_experimental.py --core --coherence=jj
```

