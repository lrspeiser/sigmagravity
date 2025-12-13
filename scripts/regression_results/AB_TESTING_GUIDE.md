# A/B Testing: Does Anisotropic Gravity Predict Better?

## Overview

The `--compare-guided` mode runs **all 17 regression tests twice** to answer the question:

**"Does anisotropic gravity (κ>0) predict star motion and light lensing better than baseline (κ=0)?"**

## How It Works

1. **Baseline run (κ=0)**: Standard Σ-Gravity predictions
2. **Anisotropic run (κ>0)**: Guided-gravity predictions with stream-seeking
3. **Comparison**: Side-by-side metrics showing which model better matches observations

## Usage

```bash
# Compare baseline vs anisotropic on all 17 tests
python scripts/run_regression_experimental.py --compare-guided --guided-kappa 0.1

# Core tests only (faster)
python scripts/run_regression_experimental.py --core --compare-guided --guided-kappa 0.1
```

## What Gets Compared

The comparison shows metrics for each test:

| Test | Baseline (κ=0) | Anisotropic (κ>0) | Better? |
|------|----------------|-------------------|---------|
| **SPARC Galaxies** | RMS (km/s) | RMS (km/s) | Lower is better |
| **Clusters** | Mass ratio | Mass ratio | Closer to 1.0 is better |
| **Gaia/MW** | RMS (km/s) | RMS (km/s) | Lower is better |
| **Bullet Cluster** | M_pred/M_bar | M_pred/M_bar | Closer to obs is better |
| ... | ... | ... | ... |

## Example Output

```
COMPARISON: baseline → guided
================================================================================
SPARC Galaxies            ✓     17.4  →  ✓     17.6
Clusters                  ✓    0.987  →  ✓        1
Cluster Holdout           ✓     1.02  →  ✓     1.02
Gaia/MW                   ✓     29.8  →  ✓     30.2
Redshift Evolution        ✓     2.97  →  ✓     2.97
Solar System              ✓ 1.77e-09  →  ✓ 1.77e-09
Counter-Rotation          ✓  0.00392  →  ✓  0.00392
Tully-Fisher              ✓     0.87  →  ✓     0.87
```

## Interpretation

- **SPARC (17.4 → 17.6)**: Baseline slightly better (lower RMS)
- **Clusters (0.987 → 1.0)**: Anisotropic better (perfect ratio vs slightly under-predicting)
- **Gaia/MW (29.8 → 30.2)**: Baseline slightly better (lower RMS)
- **Solar System**: Both identical (no coherent streams → no effect)

## Key Tests for Anisotropy

Tests that measure **directional effects** are most relevant:

1. **Bullet Cluster**: Lensing offset (directional)
2. **Galaxy-Galaxy Lensing**: Shear anisotropy (directional)
3. **SPARC Galaxies**: Star motion in disks (could show stream effects)
4. **Clusters**: Lensing + dynamics (directional)

## Current Results (κ=0.1, optimal from tuning)

- **SPARC**: Baseline better by ~1.2% (17.4 vs 17.6 km/s)
- **Clusters**: Anisotropic better (0.987 → 1.0, perfect ratio)
- **Overall**: Mixed results - anisotropy helps clusters but slightly hurts SPARC

## Next Steps

1. **Tune κ per test**: Different κ values might optimize different observables
2. **Add directional observables**: Weak lensing shear, merger offsets, stream tracks
3. **Test with 2D anisotropic solver**: Full PDE solver vs guided-gravity approximation

## Files

- Comparison report: `regression_results/experimental_report_compare_C.json`
- Contains full metrics for both models and comparison statistics

