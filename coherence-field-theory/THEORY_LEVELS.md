# Theory Levels: Fundamental vs Effective

**Date**: November 19, 2025  
**Purpose**: Clarify what is field theory and what is phenomenology

---

## Overview

This document separates "what we claim to be a fundamental field theory" from "what we're using as an effective/phenomenological tool to learn about needed physics."

---

## Level 0: Canonical Scalar-Tensor Field Theory (FUNDAMENTAL)

### Action

```
S = ∫ d⁴x √(-g) [M_Pl²/2 R - 1/2 (∇φ)² - V(φ)] + S_m[A²(φ) g_μν, ψ_m]
```

Where:
- **V(φ)** = potential (to be determined)
- **A(φ)** = conformal coupling (to be determined)

### Field Equations

**Modified Einstein equations:**
```
G_μν = 8πG [T_μν^(m) + T_μν^(φ)]
```

**Klein-Gordon equation:**
```
□φ = dV/dφ - β A dA/dφ ρ_matter
```

**Static, weak-field limit (galaxies):**
```
1/r² d/dr(r² φ') = -λV₀e^(-λφ) + βe^(βφ)ρ_b(r)
```

### Status

✅ This structure is **standard scalar-tensor gravity** (Brans-Dicke family)  
✅ Field equations are correct for this class of theories  
✅ Cosmology, galaxy, and PPN modules all implement this correctly

**What's NOT decided yet:** the exact forms of V(φ) and A(φ).

---

## Level 1: Specific Potential/Coupling Choices (TESTABLE HYPOTHESES)

### Current Baseline: Exponential + Chameleon

```
V(φ) = V₀ exp(-λφ) + M⁵/φ     (constant M, not M(ρ))
A(φ) = exp(βφ)
```

**Parameters**: V₀, λ, M, β (all constants)

### Physics Expectations

1. **Cosmology**: Exponential V(φ) gives quintessence-like dark energy
   - Target: Ω_m0 ≈ 0.3, Ω_φ0 ≈ 0.7

2. **Galaxies**: Chameleon M⁵/φ term screens in dense regions
   - Target: R_c ~ few kpc in galaxies, >> Mpc at cosmic density

3. **Solar System**: Same screening protects from fifth force
   - Target: |γ-1| < 2.3×10⁻⁵, |β-1| < 8×10⁻⁵

### Current Status

⚠️ **Problem discovered**: Pure exponential gives R_c ~ 10⁶ kpc (too light)  
⚠️ **Problem discovered**: Naive chameleon M₄ ~ 0.05 fixes galaxies but kills cosmology (Ω_m ~ 10⁻⁴)

**This is NOT a bug** — it's the field theory telling us this particular V(φ) may not work globally.

### Decisive Test

**Goal**: Find ANY (V₀, λ, M, β) that simultaneously passes:
1. Cosmology cuts: Ω_m0 ∈ [0.25, 0.35]
2. Screening cuts: R_c^spiral ≤ 10 kpc, R_c^cosmic ≥ 1000 kpc  
3. PPN cuts: |γ-1| < 2.3×10⁻⁵

**Tool**: `analysis/global_viability_scan.py`

**Possible Outcomes**:
- ✅ **Found viable region** → This V(φ) works! Proceed to full fits.
- ❌ **No viable region** → This V(φ) is ruled out. Try next potential form.

---

## Level 2: Density-Dependent M₄(ρ) (EFFECTIVE/DIAGNOSTIC)

### What It Is

Instead of constant M, we write:
```
M₄(ρ) = {
    0           for cosmology (ρ ~ 10⁻²⁶ kg/m³)
    0.05        for galaxies (ρ ~ 10⁻²⁰ kg/m³)
}
```

### Status

🔧 **This is NOT a fundamental field theory**  
🔧 **This is a phenomenological tool** to explore what environmental dependence is needed

### Purpose

By using M₄(ρ), we can:
1. Learn what kind of screening is required
2. Fit galaxies to understand ρ_c0(M_disk), R_c(ρ_b) relations
3. Use those relations to **constrain** what a real microphysical V(φ) must produce

### What It Is NOT

❌ The final Lagrangian  
❌ A claim that M varies with density in the action  
❌ Something we'd publish as a fundamental theory

### How to Interpret Results

If field-driven fits with M₄(ρ) work well:
- ✅ "Nature likes a field that screens strongly in galaxies"
- ✅ "We need V(φ) with environment-dependent effective mass"
- ⚠️ "Next: find a fundamental mechanism that generates M_eff(ρ)"

### Path Forward

Once we know the needed M_eff(ρ) profile from phenomenology, we can:
1. Try different fundamental potentials (symmetron, k-mouflage, etc.)
2. Test if they naturally produce that M_eff(ρ)
3. Select the most fundamental theory that matches data

---

## Next Potential Forms to Try (If Chameleon Fails)

### 1. Symmetron

```
V(φ) = -μ²φ²/2 + λφ⁴/4 + V₀
A(φ) = exp(βφ²)
```

**Screening**: φ → 0 in high density (symmetry restoration)  
**Advantage**: Naturally has two minima, density-dependent vacuum

### 2. K-mouflage (Non-canonical Kinetic)

```
L = -1/2 K(X) - V(φ)    where X = (∇φ)²
```

**Screening**: Higher derivatives suppressed in dense regions  
**Advantage**: Different screening mechanism than chameleon

### 3. Vainshtein (Derivative Interactions)

```
L = -1/2(∇φ)² - V(φ) + 1/M³ (∇φ)² □φ
```

**Screening**: Strong coupling in high curvature regions  
**Advantage**: Very effective at Solar System scales

---

## Summary Table

| Level | What It Is | Parameters | Status | Use Case |
|-------|-----------|------------|--------|----------|
| **Level 0** | GR + canonical scalar | Structure only | ✅ Correct | Foundation |
| **Level 1** | Exponential + chameleon | V₀, λ, M, β (constant) | 🔬 Testing | Viability scan |
| **Level 2** | Density-dependent M₄ | M₄(ρ) | 🔧 Diagnostic | Learn needed screening |

---

## How to Know If We Have "The Right" Field Equation

### Decision Flowchart

```
1. Run global viability scan (Level 1) with constant parameters
   ├─ Found viable region?
   │  ├─ YES → ✅ This V(φ) works! Use those parameters.
   │  └─ NO  → ❌ This V(φ) ruled out. Go to step 2.
   │
2. Try next potential form (symmetron, k-mouflage, Vainshtein)
   └─ Repeat step 1
   
3. If multiple potentials work:
   ├─ Compare Bayesian evidence (WAIC, AIC)
   ├─ Check unique predictions
   └─ Pick simplest/most fundamental

4. If NONE work:
   └─ Revisit Level 0 structure (maybe need higher-order terms, multiple fields, etc.)
```

---

## Current Work Plan

### Immediate (This Week)

1. ✅ Clarify theory levels (this document)
2. 🔬 Run global viability scan for exponential + chameleon
3. 📊 Analyze results:
   - If viable region found → characterize it, run full fits
   - If no viable region → document failure, move to symmetron

### Short Term (2-3 Weeks)

- Implement symmetron potential (if needed)
- Run viability scan for symmetron
- Compare multiple potential forms

### Medium Term (1-2 Months)

- Once we have a globally viable Level 1 theory:
  - Full SPARC sample fits
  - Cosmology + galaxy joint inference
  - PPN verification
  - Publication preparation

---

## Key Takeaway

**We are NOT moving away from field theory.**

We are in the normal **theory exploration phase**:
1. Start with a clean field equation class (Level 0: ✅)
2. Test specific potentials systematically (Level 1: 🔬 in progress)
3. Use phenomenology to learn what's needed (Level 2: 🔧 diagnostic tool)
4. Iterate until we find a viable, fundamental V(φ)

The M₄(ρ) work is **part of the scientific method**, not a departure from it.

---

**Next Script to Run**: `python analysis/global_viability_scan.py`

This will tell us definitively whether exponential + chameleon can work globally, or if we need to move on to the next potential form.
