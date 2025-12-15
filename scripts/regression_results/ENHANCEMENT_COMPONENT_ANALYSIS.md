# Enhancement Component Analysis: What Moves Σ the Most?

## Formula

**Σ = 1 + A × C × h**

where:
- **A** = amplitude (from path length L: `A = A₀ × (L/L₀)^n`)
- **C** = coherence (`C = v²/(v²+σ²)` or other models)
- **h** = acceleration function (`h = √(g†/g) × g†/(g†+g)`)

## Sensitivity Analysis

### Parameter Variation Impact (Baseline: A=1.17, C=0.8, h=0.3 → Σ=1.28)

| Component | Variation | Σ Range | Impact on Σ |
|-----------|-----------|---------|-------------|
| **h** | 0.1 → 0.5 | 1.09 → 1.47 | **29.3%** (biggest) |
| **A** | ±50% | 1.14 → 1.42 | 22.0% |
| **C** | 0.5 → 1.0 | 1.18 → 1.35 | 13.7% (smallest) |

**Conclusion**: **h (acceleration function) moves the enhancement the most** - a 4× change in h (0.1→0.5) produces a 29% change in Σ.

## Real Data Analysis (SPARC galaxies)

### Variation Across Galaxies

| Component | Mean | Std Dev | Range | Correlation with Σ |
|-----------|------|---------|-------|-------------------|
| **A** | 1.1725 | 0.0000 | [1.17, 1.17] | N/A (constant) |
| **C** | 0.7376 | 0.1977 | [0.18, 0.99] | **-0.352** (negative!) |
| **h** | 3.1798 | 1.7592 | [0.22, 8.30] | **+0.754** (strongest) |
| **Σ** | 3.4155 | 1.0973 | [1.26, 5.55] | - |

### Key Findings

1. **h is the primary driver**:
   - Strongest correlation with Σ (0.754)
   - Largest variation (std=1.76, range spans 8×)
   - This makes sense: h depends on acceleration, which varies dramatically across radii and galaxies

2. **C has negative correlation**:
   - Correlation = -0.352 (higher C → lower Σ?)
   - This is counterintuitive but may be because:
     - High-C galaxies (fast rotation) tend to have high acceleration
     - High acceleration → low h → lower Σ
     - The h effect dominates over the C effect

3. **A is constant** (in baseline model):
   - All galaxies use A = A₀ = 1.1725
   - No variation, so it doesn't drive differences between galaxies

## Disk vs Bulge Comparison

### Disk Galaxy (CamB, f_bulge=0.000)
- **C**: mean=0.24, std=0.10, range=[0.03, 0.32] (moderate variation)
- **h**: mean=4.47, std=0.93, range=[3.69, 6.58] (large variation)
- **Σ**: mean=2.20, std=0.51, range=[1.20, 2.87]
- **h variation effect**: 0.49 (dominates)
- **C variation effect**: 0.26

### Bulge Galaxy (NGC4217, f_bulge=0.832)
- **C**: mean=0.98, std=0.006, range=[0.98, 0.99] (nearly constant, very high)
- **h**: mean=0.53, std=0.43, range=[0.05, 1.34] (large variation)
- **Σ**: mean=1.61, std=0.49, range=[1.06, 2.53]
- **h variation effect**: 0.49 (dominates)
- **C variation effect**: 0.004 (negligible)

### Key Insight

**For bulge galaxies, C is nearly constant (~0.98)**, so:
- C cannot be the lever to fix bulge predictions
- h still varies significantly and drives Σ variation
- But h is lower on average for bulge galaxies (0.53 vs 4.47 for disks)

## Why This Matters

1. **A-suppression didn't work** because A is constant and doesn't vary much even when changed
2. **C-based fixes (SRC2) didn't work** because:
   - For bulge galaxies, C is already very high (~0.98) and nearly constant
   - The problem isn't that C is wrong, it's that h is too low
3. **h is the real lever**:
   - It varies the most (8× range across galaxies)
   - It has the strongest correlation with Σ
   - But h is determined by acceleration, which is fixed by the baryonic mass distribution

## Implications

The enhancement is primarily controlled by **h(g)**, which depends on the **acceleration scale**:
- High acceleration (inner regions, massive galaxies) → low h → low enhancement
- Low acceleration (outer regions, low-mass galaxies) → high h → high enhancement

**The bulge problem**: Bulge galaxies have high acceleration (compact, massive) → low h → low enhancement → under-prediction.

**Potential solutions**:
1. Make h depend on something other than just g (e.g., geometry, coherence)
2. Add a bulge-specific correction to h
3. Change how h is computed for bulge-dominated regions


