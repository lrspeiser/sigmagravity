# SPLITG (Split-Field Enhancement) Implementation

## Concept

The bulge problem exposes a **structural coupling**: bulge mass increases g_N everywhere, which suppresses h(g_N), which then throttles the enhancement of the disk too.

**Solution**: Use "split-field" enhancement where:
- Compute h from **g_coh** (coherent acceleration from disk+gas only), not g_total
- Apply enhancement only to the coherent part: **V_pred² = V_bulge² + Σ_coh × V_coh²**

This breaks the coupling where bulge mass suppresses the disk enhancement.

## Implementation

### 1. SPLITG Coherence Mode
- CLI: `--coherence=splitg` (also accepts `split-g`, `cohacc`, `scg`)
- Computes g_coh = V_coh²/r where V_coh² = V_disk² + V_gas²
- Uses g_coh (not g_bar) in h(g) function
- Applies enhancement only to coherent part: V_pred² = V_bulge² + Σ_coh × V_coh²

### 2. Component Curves
- Requires V_gas, V_disk_scaled, V_bulge_scaled from SPARC data
- Already available in `load_sparc()` output
- Passed to `predict_velocity()` via existing parameters

### 3. Overshoot Diagnostic
- Tracks fraction of points where V_bar > V_obs + 5 km/s
- Indicates if baryons already exceed observations (hard ceiling)
- Stored in `bulge_overshoot_frac_mean` for bulge galaxies

## Results

### Baseline (C mode)
- Overall RMS: 17.42 km/s
- Bulge RMS: 28.93 km/s
- Disk RMS: 16.06 km/s

### SPLITG mode
- Overall RMS: 17.47 km/s (+0.05 km/s, slightly worse)
- Bulge RMS: 29.09 km/s (+0.16 km/s, worse)
- Disk RMS: 16.10 km/s (+0.04 km/s, slightly worse)

## Key Finding

**SPLITG makes predictions slightly worse, not better.** This is actually **useful information** because it tells us:

1. **The structural coupling exists** (g_coh is much smaller than g_bar, h(g_coh) is 2.6-54× larger)
2. **But breaking it doesn't help** - suggests the problem is not "disk needs more enhancement"
3. **Overshoot is significant** - 36.84% of points in high-bulge galaxies have V_bar > V_obs + 5 km/s

## Interpretation

The fact that SPLITG makes things worse suggests:

1. **Overshoot is the real problem**: Many bulge regions already have V_bar > V_obs, so increasing enhancement makes it worse
2. **Need Σ < 1 capability**: The model needs to be able to *reduce* enhancement (true screening) in high-density regions, not just add boost
3. **Wrong root cause**: The bulge problem may not be about suppression of disk enhancement, but about needing de-enhancement where baryons overshoot

## Next Steps

1. **Check overshoot across all bulge galaxies** to confirm this is a systematic issue
2. **If overshoot is high**, consider allowing Σ < 1 in some regimes (major theoretical change)
3. **If overshoot is low**, then the issue is elsewhere (coherence proxy, σ model, etc.)

## Files Modified

- `scripts/run_regression_experimental.py`:
  - Added SPLITG to coherence model selection
  - Modified `predict_velocity()` to compute g_coh and use it for h
  - Updated fixed-point map to combine bulge + enhanced coherent
  - Added overshoot diagnostic in `test_sparc()`



