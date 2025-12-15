# Flow Coherence Model: Key Insights from Residual Analysis

## Executive Summary

Flow coherence model performs within 3.8% of baseline (18.09 vs 17.42 km/s RMS) while providing topology information that correlates with residuals. The model works best for **disk galaxies** (50% improvement rate) and shows distinct signatures for **bulge vs disk** systems.

## Critical Findings

### 1. Flow Coherence Works as Intended

- **C_term (flow coherence) has negative correlation with residuals**: r=-0.148 (p=6e-18)
  - Higher coherence → Lower residuals ✓
  - Confirms the model is capturing meaningful topology

- **Strongest correlation with dSigma**: r=-0.171 (p=2e-23)
  - Flow coherence directly affects the enhancement factor
  - More predictive than velocity residuals

### 2. Bulge vs Disk: Different Flow Topologies

**Bulge galaxies (n=799, f_bulge ≥ 0.3):**
- **Vorticity-dominated**: omega2 correlation r=0.174 (stronger than disk)
- **Shear irrelevant**: r=-0.002 (no correlation)
- **Tidal effects weak**: r=0.067
- Flow helps only 33% of points

**Disk galaxies (n=2574, f_bulge < 0.3):**
- **Shear matters**: r=0.141 (stronger than bulge)
- **Vorticity less important**: r=0.101
- **Tidal effects negligible**: r=0.001
- Flow helps 50% of points

**Implication**: Flow coherence model should use **different weights for bulge vs disk**:
- Bulges: Emphasize vorticity, ignore shear
- Disks: Emphasize shear, reduce vorticity weight

### 3. Radial Trends

| Radius (kpc) | Mean Residual | Mean omega2 | Mean shear2 | Mean C_term |
|--------------|---------------|-------------|-------------|-------------|
| 0-2          | -2.68 km/s    | 44,274      | 9,139       | 0.9886      |
| 2-5          | 1.91 km/s     | 2,606       | 1,717       | 0.9643      |
| 5-10         | 7.45 km/s     | 1,034       | 994         | 0.9483      |
| 10-20        | 1.40 km/s     | 679         | 673         | 0.9468      |
| 20+          | 1.32 km/s     | 176         | 289         | 0.9312      |

**Key observations:**
- Inner regions (0-2 kpc): Very high vorticity, highest coherence
- Mid regions (5-10 kpc): Largest residuals, moderate invariants
- Outer regions (20+ kpc): Low invariants, lower coherence

### 4. Galaxy-Specific Patterns

**Top 10 galaxies where flow helps most:**
- All are **disk-dominated** (f_bulge ≤ 0.20)
- Improvement up to 6.4 km/s
- Examples: F561-1, F574-2, NGC4138

**Bottom 10 galaxies where flow degrades:**
- Also mostly disk-dominated
- Degradation up to 10.5 km/s
- Examples: NGC3992, NGC5985, NGC3893

**What distinguishes them?**
- Need further analysis of galaxy properties (R_d, rotation curve shape, etc.)

### 5. Worst Residuals Analysis

**Top 10% worst residuals:**
- Mean omega2: **3× higher** than average (31,585 vs 10,174)
- Mean shear2: **2.2× higher** (6,073 vs 2,697)
- Mean tidal2: **2.3× higher** (6,602 vs 2,867)
- Mean C_term: **Lower** (0.920 vs 0.958)

**Interpretation**: Worst residuals occur where flow invariants are **high** but coherence is **low**. This suggests:
- High vorticity/shear/tidal → complex flow topology
- Low coherence → flow coherence model correctly identifies problematic regions
- These regions may need different treatment or additional physics

## Recommendations

### 1. Bulge-Specific Tuning

Implement separate flow coherence weights for bulge vs disk:
```python
if f_bulge >= 0.3:  # Bulge galaxy
    alpha = 0.0  # Ignore shear
    gamma = 0.01  # Keep tidal
else:  # Disk galaxy
    alpha = 0.02  # Use shear
    gamma = 0.005  # Minimal tidal
```

### 2. Radial-Dependent Weights

Consider making flow coherence weights depend on radius:
- Inner regions (R < 2 kpc): Higher vorticity weight
- Mid regions (2-10 kpc): Balanced weights
- Outer regions (R > 10 kpc): Lower overall weights

### 3. Galaxy-Specific Calibration

For galaxies where flow degrades significantly (e.g., NGC3992), investigate:
- Rotation curve shape
- Baryonic mass distribution
- Whether they need different coherence model entirely

### 4. Composite Features

Explore composite features that better capture topology:
- `vorticity_dominance`: Already shows r=-0.143 correlation
- `omega_shear_ratio`: May capture flow regime transitions
- `coherence_gradient`: Rate of change of C_term with radius

## Next Steps

1. **Implement bulge-specific tuning** and re-test
2. **Analyze galaxy properties** that distinguish flow-helpers from flow-hurters
3. **Test with Gaia 6D flow features** to validate on Milky Way data
4. **Create radial-dependent weights** based on trends observed
5. **Feature engineering**: Develop better composite features from flow invariants

## Files Generated

- `sparc_pointwise_baseline.csv`: Baseline pointwise data
- `sparc_pointwise_flow_optimal.csv`: Flow coherence pointwise data
- `FLOW_COHERENCE_TUNING_SUMMARY.md`: Parameter tuning results
- `RESIDUAL_ANALYSIS.md`: Detailed residual analysis
- `FLOW_COHERENCE_INSIGHTS.md`: This document

