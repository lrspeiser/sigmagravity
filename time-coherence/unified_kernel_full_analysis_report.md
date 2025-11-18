# Unified Kernel Full Analysis Report

## Executive Summary

The unified kernel combining roughness (time-coherence) and mass-coherence effects was tested on **165 SPARC galaxies**. Results show:

- **60% of galaxies improved** (99 galaxies) with mean improvement of -13.14 km/s
- **40% of galaxies worsened** (66 galaxies) with mean worsening of +51.97 km/s
- **Median delta_RMS: -5.04 km/s** (good overall performance)
- **Mean delta_RMS: +12.91 km/s** (pulled up by outliers)

## Key Findings

### 1. F_missing Saturation

**98.8% of galaxies** (163 out of 165) have `F_missing >= 4.99`, hitting the `F_max=5.0` cap. This indicates:
- The mass-coherence model is producing values that exceed the current maximum
- Most galaxies are being treated identically in terms of F_missing scaling
- The model may need recalibration or a higher F_max

### 2. Performance by Galaxy Type

**Improved galaxies** (99 galaxies):
- Mean M_baryon: 8.01×10¹⁰ M☉
- Mean R_disk: 13.73 kpc
- Mean sigma_v: 13.04 km/s
- Mean K_total: 0.867
- Mean Xi: 0.175

**Worsened galaxies** (66 galaxies):
- Mean M_baryon: 2.87×10¹¹ M☉ (3.6× larger)
- Mean R_disk: 27.81 kpc (2.0× larger)
- Mean sigma_v: 25.56 km/s (2.0× larger)
- Mean K_total: 1.127 (30% higher)
- Mean Xi: 0.261 (49% higher)

**Conclusion**: The model over-predicts for massive, extended, high-dispersion galaxies.

### 3. Correlations with Performance

Strong correlations with `delta_RMS`:
- **sigma_v: 0.598** (strongest - higher dispersion → worse performance)
- **R_disk: 0.322** (larger disks → worse performance)
- **M_baryon: 0.318** (more massive → worse performance)
- **K_total_mean: 0.359** (higher total kernel → worse performance)

**Conclusion**: The model needs to account for the fact that high-dispersion, massive galaxies require less enhancement than predicted.

### 4. Distribution Analysis

- **25th percentile**: -13.84 km/s (good improvement)
- **Median**: -5.04 km/s (modest improvement)
- **75th percentile**: +31.06 km/s (worsening)
- **Standard deviation**: 40.53 km/s (large scatter)

The distribution is **highly skewed** with a long tail of large positive values (outliers).

### 5. Best and Worst Performers

**Top 10 Improvements** (delta_RMS < -24 km/s):
- UGC12506: -51.04 km/s improvement
- NGC5985: -36.67 km/s improvement
- UGC00128: -33.49 km/s improvement
- All have F_missing = 5.0 (saturated)

**Top 10 Worst Performers** (delta_RMS > 97 km/s):
- NGC6195: +142.31 km/s worsening
- NGC5371: +141.19 km/s worsening
- NGC2955: +129.25 km/s worsening
- All have F_missing = 5.0 (saturated) and high sigma_v

## Physical Interpretation

### What's Working

1. **Roughness component (K_rough)**: Provides a baseline enhancement (~0.65) that works well for typical galaxies
2. **Median performance**: Negative median delta_RMS indicates the model improves most galaxies
3. **Smaller galaxies**: The model performs well for galaxies with M_baryon < 10¹¹ M☉, R_disk < 15 kpc, sigma_v < 15 km/s

### What's Not Working

1. **F_missing saturation**: Almost all galaxies hit the F_max cap, losing the ability to differentiate
2. **Massive galaxies**: The model over-predicts for high-mass, high-dispersion systems
3. **High sigma_v correlation**: The model doesn't account for the fact that high velocity dispersion may indicate different physics (e.g., pressure support, different dark matter fraction, or different baryon-dark matter coupling)

## Recommendations

### Immediate Actions

1. **Increase F_max or remove cap**: Since 98.8% of galaxies are saturated, consider:
   - Increasing F_max to 7.0 or 10.0
   - Removing the cap entirely and letting the model run free
   - Using a softer saturation function (e.g., tanh instead of hard clamp)

2. **Add sigma_v dependence to F_missing**: The strong correlation (0.598) suggests F_missing should decrease with increasing sigma_v:
   ```
   F_missing_adj = F_missing * f(sigma_v)
   ```
   where `f(sigma_v)` decreases for high sigma_v.

3. **Parameter tuning**: Run a parameter sweep on:
   - `extra_amp`: Currently 1.0, try 0.5-0.75 to reduce F_missing contribution
   - `f_amp`: Currently 1.0, try 0.5-0.8 to blend roughness and F_missing
   - `F_max`: Increase from 5.0 to 7.0-10.0

### Longer-term Improvements

1. **Revisit mass-coherence model**: The current model may not capture the physics correctly for high-mass systems. Consider:
   - Adding a mass-dependent saturation
   - Including a sigma_v-dependent term
   - Using a more sophisticated potential depth calculation

2. **Outlier analysis**: Investigate the worst-performing galaxies (NGC6195, NGC5371, etc.) to understand:
   - Are they barred galaxies?
   - Do they have unusual morphologies?
   - Are there data quality issues?

3. **Two-component model**: Consider splitting galaxies into two populations:
   - Low-mass, low-dispersion: Current model works well
   - High-mass, high-dispersion: Needs different treatment

## Next Steps

1. **Run parameter sweep** on `extra_amp`, `f_amp`, and `F_max` to find optimal values
2. **Implement sigma_v correction** to F_missing calculation
3. **Analyze outliers** to identify systematic issues
4. **Test on Milky Way and clusters** to ensure consistency across scales

## Data Files

- Results: `time-coherence/unified_kernel_sparc_results.csv`
- Summary: `time-coherence/unified_kernel_summary.json`
- Analysis: `time-coherence/unified_kernel_analysis.json`

