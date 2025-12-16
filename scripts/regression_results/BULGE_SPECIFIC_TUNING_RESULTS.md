# Bulge-Specific Flow Coherence Tuning: Results

## Implementation

Added bulge-specific flow coherence parameters that use different weights for bulge vs disk galaxies based on residual analysis findings:

- **Bulges** (f_bulge ≥ 0.3): Vorticity-dominated, shear irrelevant
  - `alpha_bulge = 0.0` (ignore shear)
  - `gamma_bulge = 0.01` (minimal tidal)
  
- **Disks** (f_bulge < 0.3): Shear matters, vorticity less important
  - `alpha_disk = 0.02` (use shear)
  - `gamma_disk = 0.005` (minimal tidal)

## Results

### Performance Comparison

| Model | Overall RMS | Bulge RMS | Disk RMS | Improvement |
|-------|-------------|-----------|----------|-------------|
| **Baseline (C)** | 17.42 km/s | 28.93 km/s | 16.06 km/s | - |
| **Flow (uniform)** | 18.09 km/s | 30.03 km/s | 16.69 km/s | - |
| **Flow (bulge-specific)** | **17.97 km/s** | **28.85 km/s** | 16.69 km/s | ✓ |

### Key Improvements

1. **Overall RMS**: 17.97 km/s (vs 18.09 uniform, vs 17.42 baseline)
   - Improvement: **0.12 km/s** better than uniform flow
   - Only **0.55 km/s** worse than baseline (3.2% difference)

2. **Bulge RMS**: 28.85 km/s (vs 30.03 uniform, vs 28.93 baseline)
   - Improvement: **1.18 km/s** better than uniform flow
   - **Better than baseline** by 0.08 km/s! ✓

3. **Disk RMS**: 16.69 km/s (same as uniform)
   - No change (expected, since disk parameters unchanged)

### All Tests: PASSED ✓

All 8 core tests pass with bulge-specific tuning.

## Usage

```bash
python scripts/run_regression_experimental.py --core \
  --coherence=flow \
  --flow-bulge-specific \
  --flow-alpha-bulge=0.0 \
  --flow-gamma-bulge=0.01 \
  --flow-alpha-disk=0.02 \
  --flow-gamma-disk=0.005 \
  --flow-smooth=5 \
  --flow-no-tidal
```

## Key Insights

1. **Bulge-specific tuning works**: Successfully addresses the bulge degradation observed with uniform flow parameters

2. **Bulges benefit from ignoring shear**: Setting alpha=0.0 for bulges improves their RMS by 1.18 km/s

3. **Disk performance unchanged**: Using the same disk parameters as uniform flow maintains disk performance

4. **Overall improvement**: Bulge-specific tuning brings flow coherence within 3.2% of baseline (vs 3.8% with uniform)

## Next Steps

1. **Fine-tune bulge parameters**: Test gamma_bulge variations (0.005, 0.015, 0.02)
2. **Fine-tune disk parameters**: Test alpha_disk variations (0.01, 0.03, 0.04)
3. **Radial-dependent weights**: Consider making weights depend on radius as well
4. **Hybrid approach**: Combine baseline C coherence with flow coherence as correction term

## Files Generated

- `sparc_pointwise_flow_bulge_specific.csv`: Pointwise data with bulge-specific tuning
- `experimental_report_FLOW.json`: Test results (updated)


