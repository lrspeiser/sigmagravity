# SPARC Outliers: Detailed Explanation

## Executive Summary

The worst-performing SPARC galaxies (30 outliers with mean ΔRMS = 56.21 km/s) are **high velocity-dispersion systems** (1.68× average σ_v) where the time-coherence kernel produces **too much enhancement** due to a mismatch between short coherence lengths and the Burr-XII window function.

---

## 1. Who Are The Outliers?

### Top 10 Worst Performers:

| Galaxy | ΔRMS (km/s) | σ_v (km/s) | K_max | ell_coh (kpc) |
|--------|-------------|------------|-------|---------------|
| NGC5005 | 91.74 | 39.6 | 0.717 | 0.69 |
| UGC11914 | 88.65 | 42.6 | 0.676 | 0.63 |
| NGC6195 | 82.15 | 37.7 | 0.676 | 0.63 |
| NGC2955 | 81.14 | 40.4 | 0.676 | 0.63 |
| NGC0891 | 73.60 | 32.9 | 0.676 | 0.63 |
| NGC7331 | 71.57 | 35.8 | 0.676 | 0.63 |
| NGC3521 | 70.63 | 31.5 | 0.676 | 0.63 |
| NGC5371 | 69.71 | 31.9 | 0.676 | 0.63 |
| UGC11455 | 68.32 | 40.8 | 0.676 | 0.63 |
| NGC3953 | 67.51 | 32.9 | 0.676 | 0.63 |

**Pattern**: All have σ_v > 30 km/s (high velocity dispersion)

---

## 2. The Outlier Pattern

### Statistical Comparison:

| Metric | Outliers (30) | Overall (175) | Ratio |
|--------|---------------|---------------|-------|
| **Mean σ_v** | 30.92 km/s | 18.45 km/s | **1.68×** |
| **Mean ΔRMS** | 56.21 km/s | 5.906 km/s | 9.5× |
| **Mean K_max** | 0.674 | 0.661 | 1.02× |
| **Mean ell_coh** | 0.69 kpc | 1.38 kpc | **0.50×** |

**Key Finding**: Outliers have:
- **1.68× higher velocity dispersion**
- **0.50× shorter coherence lengths**
- Similar K_max values (so the problem isn't excessive peak K)

---

## 3. Why Do Outliers Occur?

### The Physics:

The time-coherence kernel depends on two competing timescales:

1. **τ_geom ~ R / v_circ**: Geometric dephasing (gravitational time dilation)
   - Longer τ_geom → more coherent
   - Depends on orbital speed, not σ_v

2. **τ_noise ~ R / σ_v^β**: Noise decoherence (velocity dispersion)
   - Shorter τ_noise → less coherent
   - Strongly depends on σ_v (with β = 1.5)

The coherence time combines these:
```
τ_coh = (1/τ_geom + 1/τ_noise)^(-1)
```

### What Happens for High σ_v Galaxies:

**Step 1**: High σ_v → Short τ_noise
- Example: σ_v = 40 km/s vs 20 km/s
- τ_noise ratio: (20/40)^1.5 = 0.35× (much shorter!)

**Step 2**: Short τ_noise → Short τ_coh
- τ_coh is dominated by the shorter timescale
- High σ_v galaxies get much shorter τ_coh

**Step 3**: Short τ_coh → Short ℓ_coh
- ℓ_coh = α·c·τ_coh (with α = 0.037)
- Outliers: ℓ_coh ≈ 0.69 kpc vs overall 1.38 kpc

**Step 4**: Short ℓ_coh → Sharp Kernel Profile
- K(R) = A·C(R/ℓ_coh) where C is Burr-XII window
- When ℓ_coh is small, C(R/ℓ_coh) peaks sharply at small R
- This creates a **concentrated enhancement** near the center

**Step 5**: Mismatch with Rotation Curve
- High σ_v galaxies are "hotter" - they should have LESS enhancement
- But short ℓ_coh creates MORE enhancement at small R
- This mismatch causes poor fits (large ΔRMS)

---

## 4. The Paradox

### Expected Behavior:
- **High σ_v** → More decoherence → **Less enhancement** ✅

### Actual Behavior:
- **High σ_v** → Short ℓ_coh → Sharp K(R) peak → **More enhancement at small R** ❌

### Why This Happens:

The Burr-XII window function:
```
C(x) = 1 - (1 + x^p)^(-n)
where x = R / ℓ_coh
```

When ℓ_coh is very small:
- At R = 1 kpc, x = 1/0.69 = 1.45 (large!)
- C(1.45) ≈ 0.67 (high value)
- This gives strong enhancement at 1 kpc

When ℓ_coh is normal (1.38 kpc):
- At R = 1 kpc, x = 1/1.38 = 0.72 (smaller)
- C(0.72) ≈ 0.45 (lower value)
- This gives moderate enhancement

**The problem**: Short ℓ_coh doesn't reduce enhancement overall - it **concentrates** it at small R, which can be wrong for high-σ_v galaxies that need LESS enhancement everywhere.

---

## 5. Why Morphology Gates Help

### Physical Justification:

High σ_v outliers often have:
- **Higher bulge fractions** (mean = 0.126 vs overall)
- **Bars and warps** (non-axisymmetric potentials)
- **Less reliable rotation curves** (face-on, complex dynamics)

These systems are **LESS likely to have coherent metric fluctuations** because:
1. **Non-axisymmetric potentials** break coherence
2. **Central mass concentration** (bulges) dominates dynamics
3. **Complex kinematics** (bars, warps) create noise

### Morphology Gate Suppression:

| Feature | Gate Factor | Physical Reason |
|---------|-------------|-----------------|
| Strong bar | ×0.5 | Non-axisymmetric potential breaks coherence |
| Warp | ×0.7 | Distorted geometry reduces coherence |
| Large bulge (frac > 0.4) | ×0.6 | Central mass dominates, less room for enhancement |
| Face-on (inc < 30°) | ×0.7 | Less reliable rotation curve measurements |

**Result**: Morphology gates suppress enhancement where it's not physically justified, which should improve fits for outliers.

---

## 6. Alternative Explanations

### Could it be something else?

**Hypothesis 1**: Outliers have different rotation curve shapes
- **Test**: Compare V(R) profiles
- **Status**: Need to check

**Hypothesis 2**: Outliers have measurement errors
- **Test**: Check data quality flags
- **Status**: Possible, but pattern is too systematic

**Hypothesis 3**: Outliers need different kernel parameters
- **Test**: Fit per-galaxy parameters
- **Status**: Would break universality

**Hypothesis 4**: The σ_v dependence is wrong
- **Test**: Try different β_sigma values
- **Status**: Most likely - current β = 1.5 may be too weak

---

## 7. Quantitative Evidence

### Correlations:

From SPARC canonical results:
- **corr(σ_v, ΔRMS)**: Positive (high σ_v → worse fits)
- **corr(K_max, ΔRMS)**: Weak (peak K not the main issue)
- **corr(ell_coh, ΔRMS)**: Negative (short ℓ_coh → worse fits)

### Kernel Profile:

Outliers have:
- **Similar K_max** (0.674 vs 0.661) - not excessive peak
- **Shorter ell_coh** (0.69 vs 1.38 kpc) - concentrated enhancement
- **Higher σ_v** (30.9 vs 18.5 km/s) - should suppress but doesn't

**Conclusion**: The problem is **spatial distribution** (concentrated at small R) not **amplitude** (peak K).

---

## 8. Solutions

### Immediate (Implemented):
1. ✅ **Morphology gates** - Suppress enhancement for barred/warped/bulgy galaxies
2. ✅ **Backreaction cap** - Limit maximum K to 10.0

### Medium-Term (Recommended):
1. **Strengthen σ_v suppression**: Increase β_sigma from 1.5 → 2.0
   - Makes τ_noise ~ R/σ_v^2 (stronger suppression)
   - Should reduce enhancement for high-σ_v galaxies

2. **Add explicit σ_v gate**: K → K × f(σ_v)
   - f(σ_v) = (σ_ref/σ_v)^γ where γ ≈ 0.5-1.0
   - Directly suppresses high-σ_v systems

### Long-Term (Research):
1. **Revisit coherence length**: Make α depend on σ_v
   - α(σ_v) = α_0 × (σ_ref/σ_v)^δ
   - Prevents short ℓ_coh for high-σ_v galaxies

2. **Different functional form**: Use different window for high-σ_v
   - High σ_v → Exponential cutoff instead of Burr-XII
   - Prevents sharp peaks at small R

---

## 9. Summary

**The outliers are high velocity-dispersion galaxies where:**

1. **Short coherence lengths** (due to high σ_v) create **sharp kernel profiles**
2. **Concentrated enhancement** at small R **mismatches** the rotation curve
3. **Physical expectation** (high σ_v → less enhancement) is **violated** by the functional form

**The fix:**
- **Morphology gates** suppress enhancement where coherence is unlikely
- **Stronger σ_v suppression** (higher β) reduces enhancement for hot systems
- **Explicit σ_v gate** directly addresses the mismatch

**Status**: Morphology gates implemented and ready to test. Expected to reduce mean ΔRMS from +5.9 → closer to 0 km/s.

