# Verification Results for Σ-Gravity Theoretical Derivations

**Date**: 2025-01-XX  
**Status**: VERIFIED - Multiple issues found, DO NOT incorporate unverified claims into paper

## Executive Summary

The claimed "theoretical derivations" were tested independently. Results:

| Parameter | Claimed Status | Verified Status | Key Finding |
|-----------|----------------|-----------------|-------------|
| n_coh = k/2 | RIGOROUS | **RIGOROUS** ✓ | Gamma-exponential identity verified |
| A₀ = 1/√e | RIGOROUS | **NUMERIC** ○ | Correct math, but Gaussian assumption unproven |
| p = 3/4 | IMPROVED | **MOTIVATED** △ | p=1/2 verified, p=1/4 NOT verified |
| g† = cH₀/(2e) | IMPROVED | **MOTIVATED** △ | Scale correct, coefficients not unique |
| f_geom = π×2.5 | PARTIAL | **EMPIRICAL** ✗ | NFW gives 0.80, NOT 2.5 (arithmetic error!) |

**Bottom line**: 1 rigorous, 1 numeric, 2 motivated, 1 empirical.  
This is significantly LESS than the claimed "all parameters derived rigorously".

---

## Detailed Findings

### 1. n_coh = k/2 ✓ RIGOROUS

**Claim**: The coherence exponent n_coh = k/2 follows from Gamma-exponential conjugacy.

**Verification**: 
- The Gamma-exponential identity S(R) = (θ/(θ+R))^k is a mathematical theorem
- Monte Carlo verification shows average error ~0.2%, max error ~1.6%
- For k=1 (single decoherence channel), n_coh = 0.5 exactly matches SPARC fit

**Verdict**: **VERIFIED** - This IS a rigorous derivation.

---

### 2. A₀ = 1/√e ○ NUMERIC

**Claim**: A₀ = exp(-1/2) from Gaussian phase statistics.

**Verification**:
- The formula A = exp(-σ²/2) for Gaussian phases IS correct (error <0.5%)
- Agreement with fit: 0.607 derived vs 0.591 fitted (97.4%)

**Issues**:
- The assumption that gravitational phases are Gaussian is NOT proven
- The definition σ² = 1 at coherence scale is a CONVENTION, not derived

**Verdict**: **NUMERIC** - Well-defined calculation but assumptions not rigorously justified.

---

### 3. p = 3/4 △ MOTIVATED

**Claim**: p = 1/2 + 1/4 from random phase addition + Fresnel mode counting.

**Verification**:
- p = 1/2 from random phases: **VERIFIED** (matches MOND deep limit exactly)
- p = 1/4 from Fresnel zones: **NOT VERIFIED**
  - Fresnel amplitude approaches constant √(2/π) ≈ 0.798, NOT √N_zones
  - The scaling behavior is fundamentally different from claimed

**Additional Issues**:
- The decomposition 0.75 = 0.5 + 0.25 is NOT unique
- Alternative decompositions: 3/8 + 3/8, 2/3 + 1/12, etc.

**Verdict**: **MOTIVATED** - The p=1/2 part is solid physics (MOND), but p=1/4 mechanism unproven.

---

### 4. g† = cH₀/(2e) △ MOTIVATED

**Claim**: g† arises from horizon decoherence with factors from polarization averaging (1/2) and characteristic coherence (1/e).

**Verification**:
- The scale cH₀ IS well-motivated (MOND coincidence known since 1983)
- Agreement: 1.25×10⁻¹⁰ derived vs 1.2×10⁻¹⁰ fitted (96%)

**Issues**:
- Factor 1/2 from "graviton polarization" - plausible but not uniquely derived
- Factor 1/e from "horizon coherence" - plausible but not unique
- OTHER expressions also work:
  - cH₀/6 (Verlinde): 5.6% error
  - cH₀×ln(2)/4: **1.8% error** (better than 2e!)

**Verdict**: **MOTIVATED** - Correct scale, but coefficients are not uniquely derived.

---

### 5. f_geom = π × 2.5 ✗ EMPIRICAL

**Claim**: f_geom = π (3D/2D geometry) × 2.5 (NFW projection)

**Verification**:

**CRITICAL ERROR**: The NFW projection formula 2ln(1+c)/c gives:
- c=2: 1.10
- c=4: **0.80** (typical cluster)
- c=10: 0.48

**The NFW formula NEVER gives 2.5** - the maximum is 1.39 at c→1.

**This is the same arithmetic error we already fixed in the paper!**

**Factor π analysis**:
- Solid angle ratio gives 2, not π
- Path integral measure gives √π ≈ 1.77, not π
- The claimed derivation does not hold

**Verdict**: **EMPIRICAL** - No valid derivation. The factor 2.5 is unexplained.

---

### 6. Verlinde Connection

**Claim**: g† ≈ a_V = cH₀/6 (Verlinde's emergent scale) suggests common physics.

**Analysis**:
- g†/a_V ratio = 1.10 (close but not identical)
- All three scales (g†, a_V, a₀_MOND) are ~cH₀/O(1)
- This is the MOND coincidence, not new physics

**Verdict**: **INTERESTING COINCIDENCE** - Not evidence for common mechanism.

---

## Recommendations

### Do NOT Incorporate Into Paper:
1. **f_geom derivation** - Contains arithmetic error
2. **p = 1/4 from Fresnel** - Physics not supported
3. **Specific g† coefficients** - Not uniquely derived
4. **"All parameters rigorous" claims** - Not supported by verification

### Safe to Keep/Discuss:
1. **n_coh = k/2** - Rigorously derived
2. **A₀ = 1/√e** - Numerically sound with stated assumptions
3. **p = 1/2 as MOND limit** - Well-established physics
4. **g† ~ cH₀** - Known MOND coincidence

### Paper Already Updated Correctly
The honest parameter assessment in commit b696371 already reflects these findings:
- 1 rigorous (n_coh)
- 2 numeric (A₀, ℓ₀/R_d)  
- 2 motivated (p, g†)
- 1 empirical (f_geom)

---

## Technical Notes

Verification script: `derivations/connections/verify_theoretical_claims.py`

Tests performed:
1. Monte Carlo sampling (10⁵-10⁶ samples) for statistical identities
2. Direct numerical evaluation of Fresnel integrals
3. Systematic coefficient search for g†
4. NFW projection calculation for range of concentrations

All verification code is available for independent checking.
