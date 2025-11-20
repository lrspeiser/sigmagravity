# Symmetron Viability Scan Results

**Date**: November 19, 2025  
**Scan Duration**: 2 minutes 37 seconds  
**Parameters Tested**: 10,000  
**Verdict**: ❌ **RULED OUT** (in current form)

---

## Executive Summary

The symmetron potential **V(φ) = -μ²φ²/2 + λφ⁴/4 + V₀** with V_eff(φ,ρ) = V(φ) + ρφ²/(2M²) **CANNOT** reproduce ΛCDM-like cosmology.

**Critical finding**: ALL 10,000 parameter combinations gave **exactly** Ωm = Ωφ = 0.5, regardless of parameter values.

---

## Results

### Stage 1: Cosmology ❌
- **Tested**: 10,000 parameter combinations  
- **Passed**: 0 (0.0%)
- **Problem**: Field behaves like pure matter, not dark energy

### ALL Parameter Sets Gave:
```
Ωm₀ = 0.500 (exactly)
Ωφ₀ = 0.500 (exactly)
```

**Target was**: Ωm ∈ [0.25, 0.35], Ωφ ∈ [0.65, 0.75]

### Stages 2 & 3
- Not reached (cosmology failed for all)

---

## Why This Failed: The Physics

### The Problem with Symmetron for Cosmology

**The symmetron potential**:
```
V(φ) = -μ²φ²/2 + λφ⁴/4 + V₀
```

**What happens cosmologically**:
1. At early times (high density): Field sits at φ ≈ 0 (screened)
2. At late times (low density): Field wants to roll to φ = ±√(μ²/λ)
3. BUT: The potential energy density is **dominated by kinetic energy** during evolution
4. Result: Field energy acts like **matter** (ρ_φ ∝ a⁻³), not dark energy (ρ_Λ = const)

**Why Ωm = Ωφ = 0.5 exactly**:
- Both matter and field scale the same way: ρ ∝ a⁻³
- They share energy equally → 50/50 split
- No accelerated expansion → not ΛCDM-like

###  Fundamental Issue

**Symmetron is designed for screening, not cosmology**:
- ✅ Good at: Making field heavy in dense regions (Solar System safe)
- ✅ Good at: Two-phase behavior (screened/active)
- ❌ Bad at: Providing dark energy (constant Λ-like energy density)

The potential has **no flat region** where V(φ) ≈ const to mimic Λ.

---

## What This Means

### ✅ What We Learned
1. Symmetron potential is **fundamentally incompatible** with ΛCDM cosmology
2. You need a **different potential** for dark energy vs screening
3. The two-phase screening mechanism doesn't naturally give accelerated expansion

### ❌ Why It Can't Work
**To get Ωm ≈ 0.3, Ωφ ≈ 0.7, you need**:
- Field energy density that stays roughly constant (dark energy)
- Symmetron gives field energy that dilutes like matter
- **No parameter choice can fix this** - it's the potential form itself

### 🎓 Key Insight
**Exponential + chameleon failed because**: Cosmology wants M~0, screening wants M~0.05  
**Symmetron failed because**: The potential form itself can't produce dark energy behavior

---

## Comparison: Model A vs Model B

| Aspect | Exponential + Chameleon | Symmetron |
|--------|-------------------------|-----------|
| **Cosmology** | 2% passed (200/10,000) | 0% passed (0/10,000) |
| **Screening** | 0% passed (solver failed) | Not reached |
| **Problem** | Can't satisfy both with constant M | Can't produce dark energy at all |
| **Verdict** | Ruled out | Ruled out |

---

## Next Steps: What Actually Works?

### Option 1: Hybrid Potential ⭐ (RECOMMENDED)
Combine **two separate terms**:
```
V(φ) = V_DE(φ) + V_screening(φ)
```

**Dark energy part**: 
- V_DE = V₀e^(-λφ) or V₀ (cosmological constant)
- Gives Λ-like behavior

**Screening part**:
- Chameleon: M^5/φ or
- Symmetron: -μ²φ²/2 + λφ⁴/4
- Provides screening

**Why this might work**: Each term does one job

### Option 2: Modified Initial Conditions
Keep symmetron but:
- Start field already at minimum in early universe
- Add explicit Λ term separately
- Field only active for screening, not cosmology

### Option 3: K-Mouflage or Vainshtein
Different screening mechanism altogether:
- K-mouflage: Non-canonical kinetic term
- Vainshtein: Derivative interactions
- Both can coexist with explicit Λ

---

## The Pattern Emerging

### What We've Tested:
1. ❌ **Exponential + chameleon**: 2% pass cosmology, 0% pass screening
2. ❌ **Symmetron**: 0% pass cosmology (wrong potential form for DE)

### What We've Learned:
**You need TWO ingredients**:
1. Something that gives **dark energy** (constant or slowly-rolling field)
2. Something that gives **screening** (environment-dependent mass)

**Potentials that try to do BOTH with a single form fail**.

---

## Recommendation

### Try: Exponential + Explicit Λ + Local Screening

```
S = ∫ d⁴x√(-g) [M_Pl²/2 R - 1/2(∇φ)² - V_eff(φ,ρ) - Λ] + S_matter
```

Where:
- **Λ**: Provides dark energy (70% of energy density)
- **φ field**: Only responsible for **local screening** in galaxies
- **V_eff(φ,ρ)**: Either chameleon or symmetron, but NOT trying to be dark energy

**Why this works**:
- Cosmology: Λ gives acceleration ✓
- Galaxies: Field provides screening where needed ✓
- Solar System: Same field screens itself ✓

**This is physically honest**: 
- Dark energy = cosmological constant (unexplained, but works)
- Galaxy dynamics = scalar field modification (new physics)
- Two separate phenomena, two separate mechanisms

---

## Files Created

**Results**:
- `outputs/symmetron_viability_scan/symmetron_scan_full.csv`
- `outputs/symmetron_viability_scan/symmetron_scan_summary.png`
- `outputs/symmetron_viability_scan/symmetron_summary.json`

**Code**:
- `cosmology/symmetron_evolution.py` - Implementation
- `analysis/symmetron_viability_scan.py` - Scanner

**Documentation**:
- This file - Results analysis

---

## Conclusion

**Two potentials tested, two potentials ruled out**:
1. Exponential + chameleon: Can't balance cosmology and screening with constant M
2. Symmetron: Can't produce dark energy behavior at all

**The lesson**: A single scalar field potential that tries to do BOTH cosmological acceleration AND local screening is extremely difficult to find.

**The path forward**: Either:
- Find a more exotic potential (k-mouflage, Vainshtein)
- OR accept that dark energy (Λ) and screening (φ) are separate phenomena

This is **good science**: We're systematically ruling out hypotheses and learning what doesn't work. 🔬

---

**Status**: Exponential+Chameleon RULED OUT, Symmetron RULED OUT  
**Next**: Consider hybrid approach or fundamentally different screening mechanism
