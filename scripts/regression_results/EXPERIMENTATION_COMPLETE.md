# Flow Coherence Experimentation: Complete Summary

## Overview

Comprehensive experimentation with flow coherence model on SPARC galaxies and analysis of residual patterns. All core tests pass with optimal parameters.

## Achievements

### 1. Parameter Tuning ✓
- **Optimal configuration**: `alpha=0.02, gamma=0.005, smooth=5, no-tidal`
- **Performance**: 18.09 km/s RMS (within 3.8% of baseline 17.42 km/s)
- **All 8 core tests**: PASSED

### 2. Residual Analysis ✓
- **Flow coherence (C_term)**: Negative correlation with residuals (r=-0.148, p=6e-18)
  - Higher coherence → Lower residuals ✓
- **dSigma correlation**: Stronger (r=-0.171, p=2e-23)
  - Flow coherence directly affects enhancement factor

### 3. Bulge vs Disk Discovery ✓
- **Bulge galaxies**: Vorticity-dominated (r=0.174), shear irrelevant (r=-0.002)
- **Disk galaxies**: Shear matters (r=0.141), vorticity less important (r=0.101)
- **Implication**: Different flow topology signatures require different treatment

### 4. Radial Trends Identified ✓
- Inner regions (0-2 kpc): Very high vorticity (44,274), highest coherence (0.9886)
- Mid regions (5-10 kpc): Largest residuals (7.45 km/s), moderate invariants
- Outer regions (20+ kpc): Low invariants, lower coherence (0.9312)

### 5. Galaxy-Specific Patterns ✓
- Flow helps 50% of disk points, 33% of bulge points
- Top galaxies: Up to 6.4 km/s improvement (all disk-dominated)
- Bottom galaxies: Up to 10.5 km/s degradation (also mostly disk-dominated)

## Key Insights

### Flow Coherence Works as Intended
- Model correctly identifies high-coherence regions (lower residuals)
- Worst residuals occur where invariants are high but coherence is low
- Confirms flow coherence captures meaningful topology

### Bulge vs Disk: Different Physics
- **Bulges**: Vorticity-dominated flows, shear doesn't matter
- **Disks**: Shear-sensitive flows, vorticity less important
- **Recommendation**: Implement bulge-specific tuning

### Worst Residuals Pattern
- Top 10% worst residuals have:
  - 3× higher omega2 (31,585 vs 10,174)
  - 2.2× higher shear2 (6,073 vs 2,697)
  - Lower coherence (0.920 vs 0.958)
- Suggests complex flow topology where coherence model struggles

## Files Generated

### Pointwise Data
- `sparc_pointwise_baseline.csv`: Baseline (C coherence) pointwise data
- `sparc_pointwise_flow_optimal.csv`: Flow coherence pointwise data

### Documentation
- `FLOW_COHERENCE_TUNING_SUMMARY.md`: Parameter tuning results
- `RESIDUAL_ANALYSIS.md`: Detailed residual analysis
- `FLOW_COHERENCE_INSIGHTS.md`: Key insights and recommendations
- `EXPERIMENTATION_COMPLETE.md`: This summary

### Test Results
- `experimental_report_C.json`: Baseline test results
- `experimental_report_FLOW.json`: Flow coherence test results

## Next Steps

### Immediate (High Priority)
1. **Implement bulge-specific tuning**
   - Bulges: alpha=0.0 (ignore shear), gamma=0.01
   - Disks: alpha=0.02, gamma=0.005
   - Expected improvement: Better bulge performance

2. **Analyze galaxy properties**
   - What distinguishes flow-helpers from flow-hurters?
   - Investigate NGC3992 (10.5 km/s degradation)

### Medium Priority
3. **Radial-dependent weights**
   - Inner regions: Higher vorticity weight
   - Outer regions: Lower overall weights

4. **Generate full Gaia flow features**
   - Current file has 5,000/28,368 stars
   - Need full file to test with Gaia 6D features

### Future
5. **Composite features**
   - `vorticity_dominance`: Already shows promise (r=-0.143)
   - `coherence_gradient`: Rate of change with radius
   - `omega_shear_ratio`: Flow regime transitions

6. **Hybrid model**
   - Combine baseline C coherence with flow coherence
   - Use flow as correction term rather than replacement

## Usage

### Best performing configuration:
```bash
python scripts/run_regression_experimental.py --core \
  --coherence=flow \
  --flow-alpha=0.02 \
  --flow-gamma=0.005 \
  --flow-smooth=5 \
  --flow-no-tidal
```

### With pointwise export:
```bash
python scripts/run_regression_experimental.py --core \
  --coherence=flow \
  --flow-alpha=0.02 \
  --flow-gamma=0.005 \
  --flow-smooth=5 \
  --flow-no-tidal \
  --export-sparc-points=scripts/regression_results/sparc_pointwise_flow_optimal.csv
```

## Conclusion

Flow coherence model is **functional and validated**:
- ✓ Performs within 4% of baseline
- ✓ Captures meaningful topology (negative correlation with residuals)
- ✓ Identifies distinct bulge vs disk signatures
- ✓ All core tests pass
- ✓ Ready for further refinement (bulge-specific tuning, radial weights)

The model provides topology information (vorticity, shear, tidal) that correlates with residuals, enabling future improvements through targeted tuning.

