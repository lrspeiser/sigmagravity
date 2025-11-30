# Σ-Gravity: Derivation Summary and Replication Guide

**Date**: November 2025  
**Status**: Honest Assessment After Rigorous Verification

---

## Executive Summary

This document summarizes our attempts to derive the Σ-Gravity parameters from first principles, and provides honest assessments of what is actually derived versus what is calibrated.

### Parameter Status Overview

| Parameter | Formula | Fitted Value | Derived Value | Agreement | Status |
|-----------|---------|--------------|---------------|-----------|--------|
| **n_coh** | k/2 (Gamma-exponential) | 0.5 | 0.5 | 100% | ✓ **RIGOROUS** |
| **A₀** | 1/√e (Gaussian phases) | 0.591 | 0.606 | 97.4% | ○ NUMERIC |
| **ℓ₀/R_d** | Monte Carlo disk | ~1.6 | 1.42 | ~85% | ○ NUMERIC |
| **p** | 1/2 + 1/4 (claimed) | 0.757 | 0.75 | 99.1% | △ MOTIVATED |
| **g†** | cH₀/(2e) | 1.2×10⁻¹⁰ | 1.25×10⁻¹⁰ | 96% | △ MOTIVATED |
| **A_max** | √2 (polarizations) | √2 | √2 | 100% | △ MOTIVATED |
| **f_geom** | π × 2.5 (claimed) | 7.78 | 7.85 | 99% | ✗ EMPIRICAL |
| **ξ** | (2/3)×R_d (claimed) | ~5 kpc | 2 kpc | ~40% | △ MOTIVATED |

**Legend**:
- ✓ **RIGOROUS**: Mathematical theorem, independently verifiable
- ○ **NUMERIC**: Well-defined calculation with stated assumptions
- △ **MOTIVATED**: Plausible physical story, not unique derivation
- ✗ **EMPIRICAL**: Fits data but no valid derivation

---

## Detailed Parameter Derivations

### 1. n_coh = k/2 — RIGOROUS ✓

**Claim**: The coherence exponent n_coh = k/2 follows from Gamma-exponential conjugacy.

**Derivation**:
1. Assume decoherence rate λ follows Gamma distribution: λ ~ Gamma(k, θ)
2. Survival probability: S(R) = E[exp(-λR)]
3. Using Gamma-exponential identity: S(R) = (θ/(θ+R))^k
4. Amplitude A(R) = √S(R) = (θ/(θ+R))^(k/2)
5. Therefore n_coh = k/2

**Verification**: Monte Carlo simulation with 100,000 samples confirms identity to <1% error.

**For k=1** (single decoherence channel): n_coh = 0.5 exactly matches SPARC fit.

**Status**: This IS a rigorous mathematical derivation.

**Replication**:
```bash
python derivations/connections/verify_theoretical_claims.py
# See TEST 1 output
```

---

### 2. A₀ = 1/√e — NUMERIC ○

**Claim**: A₀ = exp(-1/2) from Gaussian phase statistics.

**Derivation**:
1. Assume gravitational phases φ are Gaussian distributed with variance σ²
2. Coherent amplitude: A = |⟨exp(iφ)⟩| = exp(-σ²/2)
3. Define coherence length where σ² = 1
4. Therefore A₀ = exp(-1/2) = 1/√e ≈ 0.606

**Verification**: Monte Carlo confirms A = exp(-σ²/2) to <0.5% error.

**Caveats**:
- Gaussian phase assumption is plausible (CLT) but not proven
- σ² = 1 at coherence scale is a DEFINITION, not derived

**Agreement**: 0.606 derived vs 0.591 fitted (97.4%)

**Status**: Numeric calculation, assumptions not rigorously justified.

**Replication**:
```bash
python derivations/connections/verify_theoretical_claims.py
# See TEST 2 output
```

---

### 3. p = 3/4 — MOTIVATED △

**Claim**: p = 1/2 + 1/4 from random phase addition + Fresnel mode counting.

**What's Verified**:
- p = 1/2 from random phases: **VERIFIED** ✓
  - N paths with random phases → amplitude ~ 1/√N
  - If N ~ g†/g, then enhancement ~ (g†/g)^(1/2)
  - This IS the MOND deep limit

**What's NOT Verified**:
- p = 1/4 from Fresnel zones: **NOT VERIFIED** ✗
  - Fresnel amplitude approaches constant √(2/π) ≈ 0.798
  - Does NOT scale as √N_zones
  - The claimed mechanism doesn't produce the claimed scaling

**Additional Issue**: The decomposition 0.75 = 0.5 + 0.25 is NOT unique.
Other decompositions (3/8 + 3/8, 2/3 + 1/12, etc.) also work.

**Agreement**: 0.75 derived vs 0.757 fitted (99.1%)

**Status**: p=1/2 is solid physics (MOND limit), p=1/4 mechanism unproven.

**Replication**:
```bash
python derivations/connections/verify_theoretical_claims.py
# See TEST 3 output
```

---

### 4. g† = cH₀/(2e) — MOTIVATED △

**Claim**: g† arises from horizon decoherence with factors from polarization averaging (1/2) and characteristic coherence (1/e).

**What's Well-Motivated**:
- The scale cH₀ is cosmological (Hubble acceleration)
- This explains the "MOND coincidence" a₀ ~ cH₀ known since 1983

**What's NOT Uniquely Derived**:
- Factor 1/2 from "graviton polarization" — plausible but not proven
- Factor 1/e from "horizon coherence" — plausible but not unique
- Other expressions also work:
  - cH₀/6 (Verlinde): 5.6% error
  - cH₀×ln(2)/4: **1.8% error** (better than 2e!)

**Agreement**: 1.25×10⁻¹⁰ derived vs 1.2×10⁻¹⁰ fitted (96%)

**Status**: Correct scale, but coefficients not uniquely determined.

**Replication**:
```bash
python derivations/connections/verify_theoretical_claims.py
# See TEST 4 output
```

---

### 5. A_max = √2 — MOTIVATED △

**Claim**: A_max = √2 from teleparallel gravity / graviton polarizations.

**Multiple Convergent Arguments**:

1. **Mode Counting**: Gravitational waves have 2 polarizations (+ and ×). If both contribute in the coherent regime, amplitudes add in quadrature: A_total = √(1² + 1²) = √2

2. **Torsion Fluctuations**: At the transition point where T_classical = T_critical, the effective torsion is T_eff = √(T_cl² + T_crit²) = √2 × T_cl

3. **Statistical Mechanics**: For χ²(k=2) distribution with 2 degrees of freedom, RMS amplitude = √2

4. **Dimensional Analysis**: √2 is the simplest number arising from quadrature addition

**Comparison to BTFR Calibration**:
The BTFR approach gives A_max² = 1/A_TF where A_TF ≈ 0.5, so A_max = √2. But this is calibration, not derivation, because A_TF is measured from data.

**Physical Picture**: The enhancement amplitude √2 comes from both graviton polarization modes contributing to the coherent torsion field, whereas classical gravity effectively uses only one.

**Status**: Physically motivated, multiple convergent arguments, but not a rigorous derivation from the teleparallel action.

**Replication**:
```bash
python derivations/connections/derive_A_max_teleparallel.py
```

---

### 6. f_geom = π × 2.5 — EMPIRICAL ✗

**Claim**: f_geom = π (3D/2D geometry) × 2.5 (NFW projection)

**CRITICAL ERROR FOUND**:

The NFW projection formula 2ln(1+c)/c gives:
- c=2: 1.10
- c=4: **0.80** (typical cluster)
- c=10: 0.48

**The NFW formula NEVER gives 2.5** — maximum is 1.39 at c→1.

**Factor π Analysis**:
- Solid angle ratio gives 2, not π
- Path integral measure gives √π ≈ 1.77, not π

**The factor 2.5 is UNEXPLAINED**. The claimed derivation contains an arithmetic error.

**Agreement**: 7.85 claimed vs 7.78 fitted — but derivation is invalid

**Status**: No valid derivation exists. Parameter is empirical.

**Replication**:
```bash
python derivations/connections/verify_theoretical_claims.py
# See TEST 5 output - shows NFW formula gives 0.80, not 2.5
```

---

### 7. h(g) = √(g†/g) × g†/(g†+g) — MOTIVATED △

**Claim**: The acceleration function h(g) is derived from "geometric mean of torsion".

**Derivation Sketch**:
1. Classical torsion T_local ~ g
2. Critical torsion T_crit ~ g†
3. Effective torsion as geometric mean: T_eff = √(T_local × T_crit)
4. Enhancement: Σ - 1 ~ T_eff/T_local = √(g†/g)
5. Add high-g cutoff: multiply by g†/(g†+g)
6. Result: h(g) = √(g†/g) × g†/(g†+g)

**What Works**:
- Produces flat rotation curves (v_flat ≈ 228 km/s, variation <2%)
- Different from MOND by ~7% in transition region (testable!)

**What's NOT Rigorous**:
- The "geometric mean" assumption is NOT derived from any Lagrangian
- It's CHOSEN to produce the g^(-1/2) scaling required for flat curves
- The cutoff factor g†/(g†+g) is standard interpolation, not derived

**Status**: Provides a physical story, not a mathematical proof.

**Replication**:
```bash
python derivations/connections/verify_teleparallel_h_derivation.py
```

---

### 8. ξ = (2/3)×R_d — MOTIVATED △

**Claim**: The coherence length ξ = (2/3)×R_d from torsion gradient analysis.

**What the Derivation Actually Shows**:

The `derive_xi.py` script attempts 18 different approaches, most of which fail:
- Approach 1-2: Hubble-scale lengths (4 Gpc) — way too large
- Approach 3: Gradient condition gives ξ ~ R_d/3 (1 kpc) — too small
- Approach 6: Phase accumulation gives ξ ~ 750 Mpc — way too large
- Approach 7: ξ ~ 2 R_d (6 kpc) — too large
- Approach 11: Coherence variance gives ξ ~ 0.4 R_d (1.2 kpc) — too small
- Approach 16-18: Trial and error to fit rotation curve shapes

**CRITICAL ISSUE**: The final formula ξ = (2/3)×R_d is CHOSEN, not derived.

**Claimed justification**: "The factor 2/3 comes from the requirement that Σ ≈ 2 at r ≈ 2R_d"

**BUT**: The script's own output shows:
```
Testing β = 0.667 (so ξ = 0.667 × R_d):
R_d (kpc)    ξ (kpc)      Σ(2Rd)
3.0          2.00         1.37      ← NOT ~2!
```

**The derivation claims Σ(2R_d) ≈ 2, but actually gets Σ(2R_d) = 1.37**

**What IS Established**:
- ξ scales with galaxy properties (NOT universal)
- The coherence concept is physically motivated
- ξ ~ R_d is dimensionally sensible

**What's NOT Established**:
- The coefficient 2/3 is arbitrary
- The "torsion gradient" argument is hand-waving
- The phenomenological ξ ~ 5 kpc disagrees with derived ξ ~ 2 kpc by factor of 2.5

**Comparison**:
- Phenomenological fit to SPARC: ξ ≈ 5 kpc
- Derived from (2/3)×R_d with R_d=3 kpc: ξ = 2 kpc
- **Disagreement: ~40%**

**Status**: The scaling ξ ∝ R_d is motivated, but the coefficient is fitted.

**Replication**:
```bash
python derivations/connections/derive_xi.py
# Watch for the 18 failed approaches before the "final" formula
# Note the Σ(2Rd) = 1.37, not ~2 as claimed
```

---

## Summary: What's Actually Derived?

### Genuinely Derived (1 parameter):
- **n_coh = k/2**: Mathematical theorem (Gamma-exponential identity)

### Numerically Constrained (2 parameters):
- **A₀ = 1/√e**: Correct math, but Gaussian assumption unproven
- **ℓ0/R_d ≈ 1.42**: Monte Carlo calculation, geometry-dependent

### Physically Motivated (4 parameters):
- **p ≈ 3/4**: p=1/2 verified (MOND), p=1/4 not verified
- **g† = cH₀/(2e)**: Right scale, coefficients not unique
- **A_max = √2**: Multiple convergent arguments, not rigorous
- **ξ = (2/3)×R_d**: Scaling motivated, coefficient fitted (40% discrepancy)

### Empirical (1 parameter):
- **f_geom ≈ 7.8**: No valid derivation, claimed NFW formula is wrong

---

## Comparison to Other Theories

| Theory | Free Parameters | Derivation Status |
|--------|-----------------|-------------------|
| ΛCDM | 6 cosmological + per-halo c-M | All fitted |
| MOND | 1 (a₀) | Fitted to BTFR |
| Σ-Gravity | 7 (g†, A₀, p, ℓ0, n_coh, f_geom, ξ) | 1 rigorous, 2 numeric, 4 motivated |

**Honest Assessment**: Σ-Gravity has MORE theoretical structure than MOND (which just posits a₀) but LESS than claimed. The "all parameters derived" claim is not supported. The recent ξ derivation adds to the motivated parameters but does not achieve rigorous status.

---

## Replication Instructions

### Prerequisites
```bash
pip install numpy scipy matplotlib
```

### Run All Verifications
```bash
cd C:\Users\henry\dev\sigmagravity

# 1. Verify all theoretical claims
python derivations/connections/verify_theoretical_claims.py

# 2. Verify teleparallel h(g) derivation
python derivations/connections/verify_teleparallel_h_derivation.py

# 3. Verify A_max derivation attempts
python derivations/connections/verify_A_max_derivation.py

# 4. Explore A_max from teleparallel gravity
python derivations/connections/derive_A_max_teleparallel.py
```

### Key Output to Check

1. **n_coh verification**: Should show Monte Carlo error <1%
2. **A₀ verification**: Should show 97.4% agreement
3. **p verification**: Should show p=1/2 verified, p=1/4 NOT verified
4. **g† verification**: Should show cH₀×ln(2)/4 is better fit than cH₀/(2e)
5. **f_geom verification**: Should show NFW gives 0.80, NOT 2.5

---

## Files Reference

| File | Purpose |
|------|---------|
| `verify_theoretical_claims.py` | Tests all parameter derivations |
| `verify_teleparallel_h_derivation.py` | Tests h(g) function derivation |
| `verify_A_max_derivation.py` | Tests A_max = √2 BTFR claim |
| `derive_A_max_teleparallel.py` | Explores A_max from polarizations |
| `derive_xi.py` | Attempts ξ derivation (18 approaches) |
| `VERIFICATION_RESULTS.md` | Summary of verification findings |

---

## Recommendations for the Paper

### DO Say:
- "n_coh = k/2 is rigorously derived from Gamma-exponential statistics"
- "A₀ ≈ 1/√e follows from Gaussian phase averaging"
- "g† ~ cH₀ is cosmologically motivated (MOND coincidence)"
- "The theory predicts ~7% difference from MOND in transition region"

### DON'T Say:
- "All parameters are derived from first principles"
- "No free parameters" (still have f_geom empirical, and others are only motivated)
- "f_geom = π × 2.5 from NFW projection" (arithmetic error)
- "p = 1/4 from Fresnel zones" (physics not verified)
- "ξ = (2/3)×R_d from first principles" (coefficient is fitted, 40% error from data)

### Honest Framing:
"Σ-Gravity provides a theoretical framework where the structure of the enhancement function emerges from coherence physics, with one rigorously derived parameter (n_coh), two numerically constrained parameters (A₀, ℓ0), and four physically motivated parameters (p, g†, A_max, ξ). One parameter (f_geom) remains empirical."

### Regarding ξ:
"ξ is expected to scale with the disk scale radius R_d based on coherence arguments. The phenomenological value ξ ≈ 5 kpc is consistent with typical SPARC galaxies having R_d ≈ 3-8 kpc, though the precise coefficient remains to be determined from data."

---

## Version History

- **v1.0** (Nov 2025): Initial honest assessment after rigorous verification
- Commits: `b696371` (honest parameter status), `c946c3e` (verification scripts), `7a05e29` (teleparallel h(g)), `90fba50` (A_max BTFR), `2fec074` (A_max teleparallel)
