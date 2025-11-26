# Oscillatory Coherence Exploration: Conclusions

**Date:** 2025-11-26

## What We Set Out To Do

Test whether the coherence exponent n_coh = 0.5 could be derived from first principles, using Gaia velocity correlations as the observational test.

## What We Found

### The Surprise: Oscillatory Correlations

Velocity correlations in Gaia DR3 don't follow simple power-law decay. They **oscillate**, going negative at separations > 2.4 kpc:

| Separation | Correlation |
|------------|-------------|
| 0.1 kpc | +0.26 |
| 1.2 kpc | +0.15 |
| 2.4 kpc | ~0 (zero crossing) |
| 3.5 kpc | -0.10 |
| 5.5 kpc | **-0.24** |

### Best-Fit Model

Damped cosine with λ = 10.2 kpc fits **269× better** than power-law:

$$C(r) = A \cdot \frac{\cos(2\pi r / \lambda)}{\sqrt{1 + r/\ell_0}} \cdot e^{-r/r_{\rm damp}}$$

### Derived Parameters

| Parameter | Value | Physical Meaning |
|-----------|-------|------------------|
| λ | 10.2 kpc | Oscillation wavelength |
| Zero crossing | 2.55 kpc | Matches observed 2.4 kpc |
| t_coh | 180 Myr | Inferred coherence time |
| n_orbits | 0.8 | Much shorter than winding gate (7 orbits) |

---

## Interpretation

### Two Possibilities

**Option A: Galactic Structure Contamination**

The oscillatory signal could be from Milky Way structure effects:
- Spiral arms create converging/diverging velocity flows
- Bar resonances at 2-3 kpc scale
- Moving groups and stellar streams
- Non-axisymmetric potential effects

Evidence supporting this:
- Flat rotation curve baseline works better than Eilers2019
- t_coh = 180 Myr is very short (< 1 orbit)
- Negative correlations dominate the integral

**Option B: Real Physics**

The oscillatory pattern could reveal something fundamental:
- Coherence patches "push against" each other (conservation)
- Winding creates phase-dependent anticorrelations
- The λ ≈ 10 kpc scale is cosmologically interesting

Evidence against:
- t_coh doesn't match winding gate theory (off by 9×)
- λ ratio with radius is 3× larger than R² prediction
- Simple theoretical models don't predict oscillations

### Most Likely Assessment

**The oscillatory signal is probably Galactic structure contamination**, not the fundamental coherence pattern Σ-Gravity predicts.

Reasons:
1. The inferred t_coh = 180 Myr (0.8 orbits) is much shorter than the winding gate predicts (~7 orbits)
2. The flat rotation curve works better than the proper Eilers2019 curve — suggesting the "correlations" encode systematic baseline errors
3. The test was designed to find correlations following K_coh(r), but instead found something completely different

---

## What This Means for Σ-Gravity

### The Paper's Formula Is NOT Invalidated

The kernel K(R) used for rotation curve fits is **not the same** as the correlation function C(r) we measured:

- **K(R)**: Enhancement magnitude at position R → used for rotation curves
- **C(r)**: Statistical correlation between positions separated by r → what we measured

The rotation curve formula still achieves 0.0854 dex scatter. That empirical success stands.

### The Velocity Correlation Test Is Inconclusive

We cannot confirm or deny that velocity correlations follow K_coh because:
1. Galactic structure effects dominate the signal
2. A much cleaner sample would be needed (maybe external galaxies?)
3. The test as designed doesn't cleanly isolate the coherence signal

### Recommendations

1. **Don't change the paper's kernel formula** — it works for rotation curves
2. **Don't publish the velocity correlation test** — results are ambiguous
3. **The oscillatory pattern is interesting** but needs more work to understand
4. **Future tests should use external galaxies** where MW structure doesn't contaminate

---

## Summary

| Question | Answer |
|----------|--------|
| Does n_coh = 0.5 fit Gaia correlations? | No — oscillatory fits much better |
| Is this a problem for Σ-Gravity? | No — K(R) and C(r) are different quantities |
| What causes the oscillations? | Probably MW structure (spiral arms, bar) |
| Should we publish this? | Not yet — results are ambiguous |
| What would clarify this? | Test on external galaxies (no MW contamination) |

---

## Files in This Exploration

- `N_COH_DERIVATION.md` — Full theory document
- `test_ncoh_derivations.py` — Tests 5 original approaches
- `test_oscillatory_coherence.py` — Tests Approach 6 (oscillatory)
- `test_extended_analysis.py` — Conservation, radius-dependence, theory
- `outputs/` — All results and plots
