# Σ-Gravity Cosmological Expansion Analysis
## Baryons-Only Cosmology: No Dark Matter, No Dark Energy

## Overview

This analysis tests whether Σ-Gravity can explain cosmic expansion using **only baryons**
— no dark matter, no dark energy. We compare three dark-stuff-free models against ΛCDM.

**Key Question:**
> "With ONLY baryons (Ω_b ≈ 0.05) and NO dark energy, how much of the observed
> redshift–distance relation can Σ-Gravity explain?"

---

## Results Summary (Pantheon+ SN Ia Data)

| Model | χ² | Description |
|-------|-----|-------------|
| A: GR + baryons | 1294.6 | Baseline (no dark stuff, no Σ) |
| **B: Σ + baryons** | **835.2** | **Your model** |
| C: GR + eff. matter | 1294.6 | Control (Ω_m free) |
| ΛCDM (ref) | 710.5 | Sanity check |

**Key finding:** Σ-Gravity improves χ² by **459** over GR+baryons, closing **~78%** of
the gap to ΛCDM with only baryons and coherence.

- **A_cos = 8.83** (about 2× the cluster value π√2 ≈ 4.44)
- **w_eff ≈ −1.53** (phantom-like, "super-Λ" acceleration)

---

## 1. The Three Baryons-Only Models

All models use **Ω_m = Ω_baryon ≈ 0.05** (Planck value) and **Ω_Λ = 0**.

### Model A: GR + Baryons Only
- Ω_m = 0.05 (fixed), Ω_Λ = 0, Σ = 1
- The baseline "nothing works" model
- Free parameters: H₀, M (absolute magnitude)

### Model B: Σ-Gravity + Baryons Only (YOUR MODEL)
- Ω_m = 0.05 (fixed), Ω_Λ = 0, Σ = Σ_cos(z) with A_cos free
- No dark matter, no dark energy, only baryons + coherence
- Free parameters: H₀, A_cos, M

### Model C: GR + Effective Matter (Control)
- Ω_m = free, Ω_Λ = 0, Σ = 1
- Shows how badly GR fails even if Ω_m floats
- Free parameters: H₀, Ω_m, M

**Operationally required:**

1. A **background expansion law** H(z) predicted by Σ-Gravity
2. The corresponding **distance–redshift relations** (luminosity distance, angular-diameter distance)
3. A comparison to SN Ia, BAO, and H(z) data

---

## 2. Promoting Σ to a Cosmological "Effective G(z)"

The paper defines Σ as a *local* enhancement of gravity for extended, low-acceleration systems:

```
Σ = 1 + A · W(r) · h(g)

h(g) = sqrt(g†/g) · g† / (g† + g)

g† = c·H₀ / (2e)
```

For cosmology, build a **homogeneous version** Σ_cos(z) that modifies the Friedmann equations:

```
G_eff(z) ≡ G · Σ_cos(z)
```

### Simple Cosmology Ansatz

- At the **Hubble scale**, relevant "acceleration" is of order `g_H(z) ~ c·H(z)`
- Coherence is **maximal on horizon-scale / very large modes**, so for background FRW: `W → 1`

**Definition:**

```
Σ_cos(z) ≡ 1 + A_cos · h(g_H(z))

g_H(z) ≡ c · H(z)
```

where `A_cos` is an order-unity parameter (could be related to cluster A ≈ π√2 ≈ 4.44, or treated as a fit parameter).

Note: Σ depends on H, and H depends on Σ – this creates a nonlinear Friedmann equation to solve numerically.

---

## 3. Modified Friedmann Equations with Σ_cos

For a flat FRW universe with only matter (and radiation), **no cosmological constant**:

```
H²(z) = (8πG/3) · ρ_m(z) · Σ_cos(z) + (8πG/3) · ρ_r(z)
```

where:
```
ρ_m(z) = ρ_m0 · (1+z)³
ρ_r(z) = ρ_r0 · (1+z)⁴
```

### Dimensionless Form

Define:
```
E(z) ≡ H(z) / H₀
Ω_m0 = (8πG·ρ_m0) / (3H₀²)
Ω_r0 = (8πG·ρ_r0) / (3H₀²)
```

Then:
```
E²(z) = Ω_m0 · (1+z)³ · Σ_cos(z) + Ω_r0 · (1+z)⁴
```

### The Core Equation

Insert Σ_cos:
```
Σ_cos(z) = 1 + A_cos · h(c·H₀·E(z))
```

with h(g) unchanged:
```
h(g) = sqrt(g†/g) · g† / (g† + g)
g† = c·H₀ / (2e)
```

**Core nonlinear equation to solve:**
```
E²(z) = Ω_m0·(1+z)³ · [1 + A_cos · h(c·H₀·E(z))] + Ω_r0·(1+z)⁴
```

This is solved by **fixed-point iteration** at each z, or by root-finding methods.

---

## 4. Observable Quantities

Once E(z) is computed, calculate what SN/BAO observe.

### 4.1 Comoving Distance
```
χ(z) = c ∫₀ᶻ dz' / H(z') = (c/H₀) ∫₀ᶻ dz' / E(z')
```

### 4.2 Luminosity Distance
```
D_L(z) = (1+z) · χ(z)
```

### 4.3 Distance Modulus (for SN Ia)
```
μ(z) = 5·log₁₀(D_L(z) / 10 pc)
```

### 4.4 BAO Observables

Volume-averaged distance:
```
D_V(z) = [(1+z)² · D_A²(z) · c·z / H(z)]^(1/3)
```

where `D_A(z) = D_L(z) / (1+z)²`

Compare `D_V(z)/r_s` to BAO measurements (BOSS, eBOSS).

---

## 5. Fitting Procedure

### 5.1 Free Parameters

**Minimal set:**
- H₀ (Hubble constant)
- Ω_m0 (matter density parameter)
- A_cos (cosmological coherence amplitude – could be tied to cluster A ≈ 4.44, or left free)
- M (nuisance magnitude for SN Ia)

Everything else (g†, h(g)) is **fixed by Σ theory**.

### 5.2 Datasets

- **SN Ia**: Pantheon+ (distance moduli μ(z) vs z) – available at `data/pantheon/`
- **Cosmic chronometers**: H(z) data points from differential ages
- **BAO**: D_V(z)/r_s, and optionally anisotropic D_M(z), H(z)
- **CMB shift parameter** (optional): R and l_A for crude CMB constraint

### 5.3 Likelihood

For each dataset, construct χ²:
```
χ² = (d - m(θ))ᵀ · C⁻¹ · (d - m(θ))
```

where θ = (H₀, Ω_m0, A_cos, M, …)

Total likelihood is product of all (or sum of all χ²s).

**MCMC sampling:**
- Use emcee / dynesty / existing Σ-Gravity pipeline
- Uniform priors:
  - 0.2 < Ω_m0 < 0.4
  - 0.5 < A_cos < 6 (around galaxy/cluster A values)
  - 60 < H₀ < 80

### 5.4 What to Measure / Report

1. **Best-fit A_cos with Λ = 0**
   - If Λ=0, Σ-only model fits SN+BAO almost as well as ΛCDM → "Σ can replace most of dark energy"

2. **Effective w_eff(z)**
   
   Define equivalent dark-energy density:
   ```
   ρ_DE,eff(z) ≡ 3H²(z)/(8πG) - ρ_m(z) - ρ_r(z)
   ```
   
   Compute:
   ```
   w_eff(z) = -1 - (1/3) · d ln(ρ_DE,eff) / d ln(1+z)
   ```
   
   If w_eff ≈ −1 over 0 < z < 1, Σ is effectively *doing* dark energy.

3. **Fraction of ΛCDM's Ω_Λ mimicked by Σ**
   - Compare distance moduli μ(z) to standard Ω_Λ ≈ 0.7 model
   - If residuals are tiny → "replaced" most of Ω_Λ

---

## 6. Data Location

Pantheon+ data is located at:
- `../data/pantheon/Pantheon+SH0ES.dat`
- `../data/pantheon/Pantheon+SH0ES_STAT+SYS.cov`

---

## References

- Pantheon+ SN Ia compilation
- BOSS/eBOSS BAO measurements
- Cosmic chronometer H(z) data
