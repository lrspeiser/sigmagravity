# Potential Forms for Coherence Scalar Field

## Overview

The coherence scalar field potential V(φ) must satisfy multiple constraints across different scales:

1. **Cosmology**: Provide late-time acceleration (dark energy)
2. **Galaxies**: Cluster to form halos with appropriate profiles
3. **Clusters**: Enhance lensing masses
4. **Solar System**: Be screened or negligible

Here we catalog different potential forms and their properties.

## 1. Exponential (Quintessence)

### Form
```
V(φ) = V₀ exp(-λφ / M_Pl)
```

### Properties
- **Tracker behavior**: Equation of state evolves slowly
- **w_φ**: Can range from -1 to 0 depending on λ
- **Cosmology**: Natural dark energy candidate

### Parameter Ranges
- V₀ ~ 10⁻¹²⁰ M_Pl⁴ (to match ρ_Λ ~ (10⁻³ eV)⁴)
- λ ~ 0.1 - 10 (dimensionless slope)

### Pros
- Simple, well-studied
- Naturally gives w ≈ -1 for steep slope
- Tracker solutions avoid fine-tuning

### Cons
- No natural screening
- Needs additional mechanism for solar system
- May not cluster enough for galaxy halos

## 2. Inverse Power Law

### Form
```
V(φ) = M⁴⁺ⁿ / φⁿ
```

### Properties
- **n > 0**: Runaway potential (no minimum)
- **Cosmology**: Can mimic dark energy if field rolls slowly
- **Screening**: Natural chameleon behavior

### Parameter Ranges
- n = 1, 2, 4 common choices
- M⁴ ~ (meV)⁴ for n=1

### Pros
- Provides chameleon screening automatically
- Well-studied in literature
- Simple functional form

### Cons
- Runaway (no stable minimum)
- Initial conditions matter
- May not give exactly w = -1

## 3. Combined Exponential + Power Law (Our Baseline)

### Form
```
V(φ) = V₀ exp(-λφ / M_Pl) + M⁴⁺ⁿ / φⁿ
```

### Properties
- **Exponential**: Drives cosmic acceleration
- **Power law**: Provides screening in dense regions
- **Combined**: Best of both worlds

### Strategy
1. Exponential dominates at cosmological scales → dark energy
2. Power law dominates in dense environments → screening
3. Field settles to minimum of V_eff(φ, ρ) locally

### Parameter Ranges
- V₀ ~ 10⁻¹²⁰ M_Pl⁴
- λ ~ 0.1 - 10
- M⁴⁺ⁿ tuned for solar system screening
- n = 1 or 2

### Pros
- Addresses both cosmology and local tests
- Flexible enough to fit observations
- Physical motivation (two distinct mechanisms)

### Cons
- More parameters (4 instead of 2)
- Must tune to avoid tensions

## 4. Polynomial (Symmetron-like)

### Form
```
V(φ) = -μ²/2 φ² + λ/4 φ⁴ + V₀
```

### Properties
- **Z₂ symmetry**: φ → -φ
- **Spontaneous symmetry breaking**: In low density
- **Screening**: Field pinned at φ=0 in high density

### Phase Transition
- **High ρ**: Minimum at φ = 0 (symmetric phase)
- **Low ρ**: Minima at φ = ±μ/√λ (broken phase)

### Critical Density
```
ρ_crit ~ μ² M_Pl² / M²
```

Above ρ_crit: screened
Below ρ_crit: active fifth force

### Pros
- Natural screening transition
- Clear physical picture (symmetry breaking)
- No runaway issues

### Cons
- 4 parameters (μ, λ, M, V₀)
- May not give exactly w = -1 for dark energy
- Symmetry somewhat arbitrary

## 5. Hilltop Potential

### Form
```
V(φ) = V₀[1 - (φ/μ)^p] + V₁
```

### Properties
- **p = 2**: Quadratic hilltop
- **p > 2**: Steeper
- Near φ=0: acts like cosmological constant

### Pros
- Can give w ≈ -1 easily
- Stable minimum

### Cons
- Less natural for screening
- Less physical motivation in our context

## 6. Double Exponential

### Form
```
V(φ) = V₁ exp(-λ₁φ) + V₂ exp(-λ₂φ)
```

### Properties
- Two scales: λ₁ for cosmology, λ₂ for local
- Can interpolate between regimes

### Pros
- Flexible
- Each exponential serves different purpose

### Cons
- Many parameters (4)
- Not clear which regime dominates where

## 7. Logarithmic

### Form
```
V(φ) = V₀ ln(φ / φ₀)
```

### Properties
- Slow roll naturally
- Used in some inflation models

### Pros
- Simple
- Natural slow evolution

### Cons
- Singular at φ=0
- Not well-suited for screening

## Recommended Strategy

### Phase 1: Exponential Only
Start with simplest:
```
V(φ) = V₀ exp(-λφ / M_Pl)
```

**Test**:
- Can it reproduce ΛCDM expansion? ✓
- Can it produce galaxy halos? Test
- Does it pass solar system tests? ✗ (Expected to fail)

### Phase 2: Add Chameleon Term
```
V(φ) = V₀ exp(-λφ / M_Pl) + M⁴ / φ
```

**Test**:
- Cosmology still works? Should
- Galaxy halos still fit? Check
- Solar system safe? Tune M⁴

### Phase 3: Optimize
Fit {V₀, λ, M⁴} globally across:
- Pantheon SNe (cosmology)
- SPARC galaxies (halos)
- Abell clusters (lensing)
- Cassini (PPN)

### Phase 4 (If Needed): Try Alternative
If no good fit found, try symmetron:
```
V(φ) = -μ²/2 φ² + λ/4 φ⁴
```

## Connection to GW Coherence

Ultimately, V(φ) should be derived from GW microphysics:

1. **Start with GW spectrum**: P(k, t) for stochastic background
2. **Compute Isaacson stress-energy**: ⟨T_μν^(GW)⟩
3. **Identify coherence parameter**: φ ~ amplitude or correlation length
4. **Effective action**: Integrate out high-frequency modes
5. **Result**: V(φ) emerges as effective potential

**Example heuristic**:
- Coherent GWs: φ large, V(φ) small → field active
- Incoherent GWs: φ small, V(φ) large → field massive/frozen

**Exponential form might arise from**:
- Entropy of GW modes: V ~ exp(-S(φ))
- Or thermal distribution: V ~ exp(-E(φ)/T)

**Power-law form might arise from**:
- Dimensional analysis with cutoff scale Λ
- Or matching to effective field theory expansion

This is the **key theoretical challenge**: make the connection rigorous.

## Fitting Parameters to Data

### Cosmology Constraints
From Pantheon SNe:
- Today: Ω_φ ≈ 0.7
- Requires: ρ_φ(a=1) ≈ 0.7 ρ_crit,0
- Sets scale: V₀ ~ 10⁻¹²⁰ M_Pl⁴ ~ (2.4 meV)⁴

### Galaxy Constraints
From SPARC rotation curves:
- Need: ρ_φ(r) ~ few M_☉/pc³ at r ~ 10 kpc
- Core radius: R_c ~ few kpc
- Sets: Field value and gradients at galaxy scale

### Cluster Constraints
From Abell lensing:
- Need: ρ_φ ~ 10⁸ - 10⁹ M_☉/kpc³ at r ~ 100 kpc
- Core radius: R_c ~ 100-500 kpc
- Consistent with galaxy scaling?

### Solar System Constraints
From Cassini:
- Need: |γ - 1| < 2×10⁻⁵
- Requires: χ < 0.01 (strong screening)
- Sets: M⁴ in chameleon term

### Challenge
Can one V(φ) satisfy all constraints?
- If yes: Coherence field theory viable ✓
- If no: Need to understand why, or modify approach

## Next Steps

1. Implement potential class with multiple forms
2. Create parameter fitting framework
3. Run optimization across all scales
4. Assess goodness of fit for each form
5. Derive best-fit potential from data
6. Then work backwards to GW interpretation

