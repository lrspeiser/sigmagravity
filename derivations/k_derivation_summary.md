# Derivation of k ≈ 0.24: Summary of Findings

## Executive Summary

We attempted to derive the coherence scale coefficient k in ξ = k × σ_eff/Ω_d from first principles. The key finding is that **k ≈ 1/3 emerges naturally from matching the coherence window W(r) to the covariant coherence scalar C(r)**, but the empirical k = 0.24 suggests additional effects.

## Theoretical Framework

### 1. The Covariant Coherence Scalar

The local coherence scalar from §2.6.1:

$$\mathcal{C} = \frac{\omega^2}{\omega^2 + 4\pi G\rho + \theta^2 + H_0^2}$$

In the non-relativistic limit for steady-state circular rotation:

$$\mathcal{C} \approx \frac{v_{\rm rot}^2}{v_{\rm rot}^2 + \sigma^2}$$

The transition C = 1/2 occurs when v_rot = σ, giving r_transition = σ/Ω.

### 2. Matching to the Coherence Window

The phenomenological coherence window:

$$W(r) = 1 - \left(\frac{\xi}{\xi+r}\right)^{0.5}$$

has W = 1/2 at r = 3ξ.

**Naive matching:** If W = 1/2 at the same radius as C = 1/2:
- 3ξ = σ/Ω
- ξ = σ/(3Ω)
- **k = 1/3 ≈ 0.333**

## Numerical Results

### From Theoretical Analysis

| Approach | Predicted k | Notes |
|----------|------------|-------|
| Naive matching (W=1/2 at C=1/2) | 0.333 | Direct transition matching |
| Orbit-averaged matching | 0.332 | Epicyclic averaging has negligible effect |
| Mass-weighted MSE | 0.274 | Exponential disk weighting |
| Average matching (⟨W⟩ = ⟨C⟩) | 0.187 | Mass-weighted average |
| Dispersion ratio (σ_φ/σ_total) | 0.178 | If coherence uses σ_φ only |

### From SPARC Data Fitting

Per-galaxy optimal k fitting on 171 galaxies:

| Statistic | Value |
|-----------|-------|
| Mean k | 0.43 |
| Median k | 0.37 |
| Std k | 0.34 |
| Interior solutions (not at bounds) | 26% |
| Interior median k | 0.33 |

**Key observation:** Interior solutions have median k ≈ 0.33, matching the naive theoretical prediction!

## Key Findings

### 1. Epicyclic Orbit Averaging Has Negligible Effect

For a flat rotation curve (κ/Ω = √2), orbit averaging changes k by < 1%. This is because the coherence window W(r) is nearly linear over the epicyclic excursion range.

### 2. The Functional Form Mismatch Matters

W(r) and C(r) have different functional forms:
- W(r) = 1 - (ξ/(ξ+r))^0.5 (slower rise at small r)
- C(r) = r²/(r² + ξ_C²) (faster rise at small r)

This mismatch means no single k can match both functions perfectly. The best k depends on the radial weighting.

### 3. The Empirical k = 0.24 May Reflect σ_eff Definition

The theoretical k ≈ 0.33 assumes σ in the coherence physics equals σ_eff in the formula. If the effective σ differs:

- If σ_eff uses only azimuthal dispersion (σ_φ) but physics uses total (σ_total):
  - σ_total/σ_φ ≈ √3 for flat RC
  - k = 0.33 × (σ_φ/σ_total) = 0.33/√3 ≈ 0.19

- The empirical k = 0.24 falls between 0.19 and 0.33, suggesting partial averaging.

### 4. Per-Galaxy k Shows Wide Variation

The per-galaxy optimal k ranges from 0.05 to 0.80, with 73% hitting the fitting bounds. This suggests:

1. k may not be the right parameter to vary per-galaxy
2. The σ_eff estimation has significant uncertainty
3. Other parameters (A, G) may need to vary instead

### 5. k Has Weak Correlations with Galaxy Properties

| Property | Correlation with k |
|----------|-------------------|
| V/σ | +0.16 |
| R_d | -0.06 |
| V(R_d) | +0.07 |
| σ_eff | -0.04 |
| gas_frac | +0.04 |

The weak correlations suggest k is approximately universal, but with significant scatter.

## Implications for the Theory

### What We Can Claim

1. **k ≈ 1/3 is theoretically motivated** from matching the coherence window to the covariant scalar
2. **The exponent 0.5 in W(r)** (from decoherence statistics) is consistent with this matching
3. **The empirical k = 0.24** is within the range of theoretical predictions (0.19 to 0.33)

### What Remains Uncertain

1. **The exact value of k** depends on:
   - How σ_eff is defined (which components, how weighted)
   - The radial weighting in the matching integral
   - Galaxy-specific factors (not yet identified)

2. **Whether k should vary per-galaxy** or remain fixed

### Recommended Path Forward

1. **Derive k from the orbit-averaging integral** with proper mass weighting:
   $$k = \frac{\int_0^\infty W(r) \Sigma(r) r \, dr}{\int_0^\infty C(r) \Sigma(r) r \, dr} \times \frac{1}{3}$$

2. **Test alternative σ_eff definitions**:
   - Line-of-sight dispersion
   - Inclination-corrected dispersion
   - Component-weighted dispersion

3. **Check if k correlates with morphology** (not yet tested)

## Conclusion

The coefficient k ≈ 0.24 in the dynamical coherence scale ξ = k × σ_eff/Ω_d is **consistent with theoretical expectations** (k ≈ 1/3 from naive matching, reduced by σ_eff definition effects). The derivation is not yet complete, but the empirical value falls within the theoretically motivated range.

**Status:** Dynamically motivated, with partial theoretical support. Full derivation requires:
1. Proper treatment of the σ_eff definition
2. Mass-weighted matching integral
3. Understanding of galaxy-to-galaxy variation

---

*Generated by `derivations/derive_k_from_epicycles.py`, `derivations/derive_k_from_dispersion_ratio.py`, and `derivations/test_k_derivation_on_sparc.py`*

