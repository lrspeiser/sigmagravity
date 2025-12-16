# SPLITG (Split-Field Enhancement) Results

## Implementation Complete

SPLITG mode successfully implements the split-field enhancement where:
- **h(g)** is computed from **g_coh** (coherent acceleration from disk+gas only)
- Enhancement is applied only to coherent part: **V_pred² = V_bulge² + Σ_coh × V_coh²**

This breaks the structural coupling where bulge mass suppresses disk enhancement.

## Results

### Baseline (C mode)
- Overall RMS: **17.42 km/s**
- Bulge RMS: **28.93 km/s**
- Disk RMS: **16.06 km/s**

### SPLITG mode
- Overall RMS: **17.47 km/s** (+0.05 km/s, **slightly worse**)
- Bulge RMS: **29.09 km/s** (+0.16 km/s, **worse**)
- Disk RMS: **16.10 km/s** (+0.04 km/s, **slightly worse**)

## Key Findings

### 1. SPLITG Has Large Effect on h
For a high-bulge galaxy (NGC4217, f_bulge=0.832):
- **g_coh/g_bar ratio**: 0.016-0.224 (g_coh is much smaller)
- **h(g_coh)/h(g_bar) ratio**: 2.6-54× (h is much larger with g_coh)
- This confirms the structural coupling exists and SPLITG breaks it

### 2. But It Doesn't Improve Predictions
SPLITG makes predictions **slightly worse**, not better. This suggests:
- The problem is **not** "disk needs more enhancement"
- Breaking the coupling doesn't help because the issue is elsewhere

### 3. Overshoot is Moderate
Across all bulge galaxies (f_bulge > 0.3):
- **Mean overshoot fraction**: 11.91% (points where V_bar > V_obs + 5 km/s)
- **Median**: 8.39%
- **Max**: 43.33%
- **None > 50%**

Overshoot is present but not extreme, so it's not the only issue.

## Interpretation

The fact that SPLITG makes things worse (even slightly) is **useful information**:

1. **Structural coupling exists** (confirmed by h ratios)
2. **But breaking it doesn't help** - suggests the problem is not about suppression
3. **Overshoot is moderate** - not the primary issue, but contributes

## Possible Explanations

1. **Wrong direction**: Maybe we need to *suppress* enhancement more in bulges, not boost it
2. **Coherence proxy issue**: The C(v) proxy may be wrong for bulge systems
3. **Fixed M/L assumptions**: The M/L corrections may be off for bulge galaxies
4. **Need Σ < 1**: Some regions may need true screening (de-enhancement), not just less enhancement

## Next Steps

1. **Test with Σ < 1 capability** to see if de-enhancement helps overshoot cases
2. **Check if coherence proxy is wrong** for bulge systems (maybe need different σ)
3. **Investigate M/L assumptions** - are bulge M/L values correct?
4. **Try combining approaches** - SPLITG + other modifications

## Files

- **Implementation**: `scripts/run_regression_experimental.py` (SPLITG mode)
- **Results**: `scripts/regression_results/experimental_report_SPLITG.json`
- **Documentation**: `scripts/regression_results/SPLITG_IMPLEMENTATION.md`



