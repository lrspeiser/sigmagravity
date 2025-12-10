# R_c Problem Diagnosis

**Date**: November 19, 2025  
**Status**: IN PROGRESS - Diagnosing why halos are too extended

---

## Problem Statement

Field-driven halos are consistently too extended:
- Fitted R_c often hits 50 kpc upper bound
- Phenomenological R_c typically 1-5 kpc
- Ratio: R_c^(field) / R_c^(phen) ≈ 10-100x

---

## Theoretical Framework

For the static KG equation:
\[
\phi'' + \frac{2}{r}\phi' = \frac{dV_{\rm eff}}{d\phi}
\]

The core size is set by the effective mass:
\[
m_{\rm eff}^2 = \frac{d^2 V_{\rm eff}}{d\phi^2}, \quad R_c \sim m_{\rm eff}^{-1}
\]

If R_c is too large, m_eff is too small (field is too light).

---

## Measurements

### Effective Mass Analysis

For DDO064 (lambda=1.2, beta=0.05):
- **m_eff at r=2*R_disk**: 6.66e-6 kpc⁻¹
- **R_c^(theory) = 1/m_eff**: 150,156 kpc (!)
- **R_c^(fitted)**: 50 kpc (hitting bound)
- **R_c^(phen)**: 1.79 kpc

**Interpretation**: 
- Field is extremely light (m_eff ≈ 7e-6 kpc⁻¹)
- Theoretical core radius is huge (150k kpc)
- Fitted R_c is much smaller than theoretical, suggesting the pseudo-isothermal fit doesn't match the actual field profile

---

## Root Cause Analysis

### 1. Effective Mass Too Small

The effective mass squared is:
\[
m_{\rm eff}^2 = \frac{d^2 V}{d\phi^2} + \beta^2 e^{\beta\phi} \cdot \rho_b
\]

For our parameters:
- V(φ) = V₀ e^(-λφ), so d²V/dφ² = λ² V₀ e^(-λφ) ≈ λ² V₀ (if φ small)
- Coupling term: β² e^(βφ) · ρ_b

At r ≈ 2*R_disk:
- ρ_b is relatively small (exponential decay)
- φ is small (near cosmological value)
- So m_eff² is dominated by λ² V₀ ≈ 1e-6

This gives m_eff ≈ 1e-3 in H0 units, which converts to ~7e-6 kpc⁻¹.

### 2. Why Field is So Light

The field is light because:
1. **V₀ is small**: V₀ = 1e-6 (set by cosmology)
2. **Coupling term is small**: β² e^(βφ) · ρ_b is small at r ≈ 2*R_disk
3. **No screening**: Without chameleon term, field doesn't get heavy in dense regions

### 3. Why Fitted R_c is Smaller Than Theoretical

The fitted R_c (50 kpc) is much smaller than theoretical (150k kpc) because:
- The pseudo-isothermal profile fit is trying to match the rotation curve
- The actual field profile may have a different shape
- The fit is hitting the upper bound, so it's not a true fit

---

## Potential Solutions

### 1. Increase Effective Mass

**Option A: Larger V₀**
- Problem: V₀ is constrained by cosmology (dark energy density)
- May not be viable

**Option B: Stronger Coupling (larger β)**
- Tested: β = 0.05 gives R_c ratio ≈ 15x (still too large)
- May need β >> 1, but this could break solar system tests

**Option C: Chameleon Screening**
- Add M⁴/φ term to potential
- Field gets heavy in dense regions: m_eff² ~ M⁴/φ³
- This is the standard solution for scalar-tensor theories

### 2. Different Potential Form

**Option A: Power-law potential**
- V(φ) = M⁴/φⁿ (chameleon)
- Field gets heavy in dense regions

**Option B: Symmetron**
- V(φ) = -μ²/2 φ² + λ/4 φ⁴ + ρ/M² φ²
- Symmetry breaking in dense regions

### 3. Mass-Dependent Coupling

As suggested in the user's message:
\[
\beta_{\rm eff}(M_{\rm disk}) = \beta_0 \left(\frac{M_{\rm disk}}{M_0}\right)^\alpha
\]

This could help with the dwarf vs spiral tension.

---

## Next Steps

1. **Test chameleon term**: Add M⁴/φ to potential and see if it increases m_eff
2. **Test stronger β**: Try β = 0.1, 0.2, 0.5 and see if R_c improves
3. **Test mass-dependent coupling**: Implement β_eff(M_disk) and see if it helps
4. **Check field profile shape**: Plot actual ρ_φ(r) vs pseudo-isothermal to see if shape mismatch is the issue

---

## Current Status

- ✅ Instrumentation complete: Can measure m_eff and R_c^(theory)
- ✅ R_c penalty implemented: Can penalize oversized halos
- ⚠️ Problem identified: Field is too light (m_eff too small)
- ⚠️ Solution needed: Either increase m_eff or change approach

**Key Insight**: The field needs to be heavier in galaxy interiors. This likely requires screening (chameleon term) or much stronger coupling.

