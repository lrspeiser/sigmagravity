# h(g) High-Acceleration Tail Softening - Sweep Results Summary

## Best Configuration

**p_hi = 1.2, x0 = 1.0**

- **Overall SPARC RMS**: 17.411 km/s (baseline: 17.415 km/s)
- **Bulge RMS**: 28.754 km/s (baseline: 28.926 km/s) - **improvement of 0.17 km/s**
- **Disk RMS**: 16.077 km/s (baseline: 16.061 km/s) - slight increase
- **Clusters ratio**: 1.106 (baseline: 0.987) - slightly high
- **Solar System**: |γ-1| = 0.000 < 2.3e-05 ✓
- **All tests passed**: ✓

## Key Findings

### 1. Best Overall Performance
- **p_hi=1.2, x0=1** gives the best SPARC RMS (17.411 km/s) and best bulge RMS (28.754 km/s)
- This is a **gentle modification** (p_hi=1.2 is close to baseline 1.5) with **early activation** (x0=1)

### 2. Bulge Improvement is Small
- Best bulge RMS improvement: **0.17 km/s** (28.754 vs 28.926)
- This is a **0.6% improvement**, which is measurable but not dramatic
- Suggests that **h(g) modification alone may not be sufficient** to solve the bulge problem

### 3. Parameter Sensitivity
- **x0=1 (early activation)**: Best for bulge improvement, but slightly worsens clusters
- **x0=3**: Moderate improvement, maintains cluster ratio
- **x0=10, 30**: No improvement (modification doesn't activate for most bulge regions)

### 4. p_hi Sensitivity
- **p_hi < 1.0**: Too aggressive, worsens predictions and can fail Solar System test
- **p_hi = 1.0-1.2**: Best range for improvement
- **p_hi = 1.5 (baseline)**: No modification (baseline behavior)

### 5. Solar System Safety
- All configurations with **p_hi ≥ 0.8** pass the Cassini bound
- **p_hi=0.7** can fail Solar System test (especially with x0=1 or x0=3)

### 6. Cluster Impact
- Most configurations maintain cluster ratio ~0.987 (good)
- **x0=1** configurations have higher cluster ratios (1.106-1.253), suggesting the modification affects cluster-scale accelerations

## Top 5 Configurations

| Rank | p_hi | x0 | SPARC RMS | Bulge RMS | Disk RMS | Clusters | Solar | Status |
|------|------|----|-----------|-----------|----------|----------|-------|--------|
| 1 | 1.2 | 1 | **17.411** | **28.754** | 16.077 | 1.106 | ✓ | ✓ |
| 2 | 1.5 | 1 | 17.415 | 28.926 | 16.061 | 0.987 | ✓ | ✓ |
| 3 | 1.5 | 3 | 17.415 | 28.926 | 16.061 | 0.987 | ✓ | ✓ |
| 4 | 1.5 | 10 | 17.415 | 28.926 | 16.061 | 0.987 | ✓ | ✓ |
| 5 | 1.5 | 30 | 17.415 | 28.926 | 16.061 | 0.987 | ✓ | ✓ |

## Interpretation

### Why the Improvement is Small

1. **Bulge accelerations are moderate**: Mean x = g/g† ≈ 3.8, not >> 10
2. **Modification activates early**: With x0=1, the modification affects most bulge regions
3. **Gentle modification needed**: p_hi=1.2 (vs baseline 1.5) is a small change
4. **Limited room for improvement**: The baseline h(g) may already be near optimal

### Why p_hi=1.2, x0=1 Works Best

- **Early activation (x0=1)**: Catches bulge regions where x ≈ 3.8
- **Gentle softening (p_hi=1.2)**: Small enough to not break Solar System, large enough to help bulges
- **Trade-off**: Slightly worsens clusters (ratio 1.106 vs 0.987) but improves bulges

## Recommendations

1. **Use p_hi=1.2, x0=1** if:
   - You want the best bulge RMS improvement
   - Slightly high cluster ratio (1.106) is acceptable
   - Overall SPARC RMS improvement is the priority

2. **Use baseline (p_hi=1.5)** if:
   - Cluster ratio must be exactly ~1.0
   - The small bulge improvement (0.17 km/s) is not worth the cluster trade-off

3. **Consider alternative approaches** if:
   - The bulge problem requires larger improvements (>1 km/s)
   - The h(g) modification alone is insufficient

## Next Steps

1. **Test p_hi=1.1, 1.3** to see if there's a better sweet spot
2. **Investigate why clusters are affected** when x0=1 (may need coherence-dependent modification)
3. **Consider combining with other modifications** (e.g., geometry-based A, coherence-dependent h)
4. **Analyze individual bulge galaxies** to understand why improvement is limited

## Files

- **Full results**: `scripts/regression_results/H_HI_POWER_SWEEP.json`
- **Markdown table**: `scripts/regression_results/H_HI_POWER_SWEEP.md`
- **Individual reports**: `scripts/regression_results/experimental_report_h_p*.json`


