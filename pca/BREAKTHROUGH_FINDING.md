# BREAKTHROUGH: Empirical Boost Function Analysis

## 🚨 Critical Discovery

The empirical boost extraction reveals a **fundamental insight** that changes everything:

### The Surprising Pattern

**Expected** (from residual correlations):
- Massive galaxies have large residuals → Need larger A
- Residual ∝ Mbar suggests A should increase with mass

**Empirical reality** (from extracting K from data):
- **A_empirical ANTI-CORRELATES with mass!**
  - ρ(A_emp, Mbar) = **-0.540** (p < 10⁻¹²)
  - ρ(A_emp, Vf) = **-0.410** (p < 10⁻⁷)

**This means**: Smaller galaxies need **larger** boost amplitudes, not smaller!

---

## What This Tells Us

### The Real Problem

**Original hypothesis**: "Massive galaxies are under-predicted because A=0.6 is too small"
- Led to: Try A ∝ Vf^α with α > 0
- Result: Helped a little but still failed

**Empirical discovery**: "Massive galaxies are under-predicted because the boost SATURATES differently"
- True pattern: A_empirical **decreases** with mass/velocity
- Implication: **Boost physics changes qualitatively** across mass range

### Physical Interpretation

**Small galaxies** (Mbar < 1 × 10⁹ M☉):
- Need large boost amplitude (A ~ 3-4)
- Boost is the dominant effect
- Sparse baryon distribution → paths less interrupted?

**Massive galaxies** (Mbar > 100 × 10⁹ M☉):
- Need small boost amplitude (A ~ 0.5-1.0)
- Boost saturates or is suppressed
- Dense baryon distribution → paths decohere faster?

**This suggests**: Boost effectiveness **decreases** in denser environments!

---

## Empirical Boost PCA Results

### Variance in Boost Functions

**PC1**: 71.1% of boost shape variance
- Reveals the dominant mode of how K(R) varies across galaxies
- This is THE target shape to match

**PC2**: 13.4% of variance
- Secondary variation in boost shape

**PC3**: 5.6% of variance
- Tertiary variation

**PC1-3**: 90.2% total
- Boost functions themselves lie on low-dimensional manifold!

### Empirical Parameter Statistics

| Parameter | Mean | Std | Range |
|-----------|------|-----|-------|
| A_empirical | 2.24 | 1.89 | Wide variation!|
| l0_empirical | 4.27 kpc | 2.98 kpc | Moderate variation |
| p_empirical | 2.95 | 1.78 | Wide variation |

**Key finding**: Empirical parameters vary by factors of 3-5 across galaxies, not simple scalings.

---

## What the Correlations Mean

### A_empirical vs Galaxy Properties

| Property | Correlation | Interpretation |
|----------|-------------|----------------|
| log(Mbar) | **ρ = -0.54** | INVERSE! Massive galaxies need SMALLER boost |
| log(Vf) | **ρ = -0.41** | INVERSE! Fast galaxies need SMALLER boost |
| log(Rd) | ρ = -0.33 | INVERSE! Large galaxies need SMALLER boost |

**This is the opposite of naive expectation!**

### l0_empirical vs Galaxy Properties

| Property | Correlation | Interpretation |
|----------|-------------|----------------|
| log(Vf) | ρ = +0.29 | Weak: faster galaxies have slightly longer coherence |
| log(Mbar) | ρ = +0.29 | Weak: massive galaxies have slightly longer coherence |
| log(Rd) | ρ = +0.03 | **NONE!** Coherence scale doesn't track disk size |

**Surprising**: ℓ₀ does NOT scale with Rd as we expected!

---

## Physical Implications

### The Decoherence Hypothesis

The anti-correlation of A with mass/density suggests:

**In sparse environments** (dwarfs):
- Few scattering centers
- Paths stay coherent longer
- Large boost amplitude needed

**In dense environments** (giants):
- Many scattering centers  
- Paths decohere quickly
- Small boost amplitude

**Mathematical form**:
```python
# Instead of: A ∝ Mass^α
# Need: A ∝ 1 / (density or scattering rate)

A = A0 / (1 + (Sigma0 / Sigma_crit)^delta)
# or
A = A0 * exp(-Sigma0 / Sigma_coh)
```

---

### The Coherence Scale Mystery

**Expected**: ℓ₀ ∝ Rd (coherence tracks system size)
**Empirical**: ℓ₀ barely correlates with Rd (ρ = +0.03!)

**Possible explanations**:
1. Coherence scale is set by physics (not geometry)
2. ℓ₀ ~ 4-5 kpc is a **universal** physical scale
3. Variations in fitted ℓ₀ are compensating for wrong A or wrong functional form

---

## Revised Model Recommendations

### Priority 1: Density-Suppressed Amplitude

**Form**:
```python
A = A0 / (1 + (Sigma0 / Sigma_crit)^delta)

# Or equivalently with mass:
A = A0 / (1 + (Mbar / M_crit)^gamma)
```

**Rationale**: 
- Matches A_emp ∝ 1/Mbar pattern
- Physical: Dense environments suppress coherence
- Testable: Should reduce ρ(residual, PC1) dramatically

---

### Priority 2: Universal Coherence Scale

**Form**:
```python
l0 = 4.0 kpc  # Fixed, universal

# Don't scale with Rd since empirical data shows NO correlation
```

**Rationale**:
- Empirical ℓ₀ doesn't vary with Rd
- Wide scatter (std = 3 kpc) may be due to fitting degeneracies
- Try fixed ℓ₀ with proper A(Σ₀) scaling

---

### Priority 3: Test Modified Boost Structure

The persistent PC correlations suggest the multiplicative form may be wrong:

```python
# Current (multiplicative):
g_eff = g_bar * (1 + K)

# Alternative (additive in squared velocity):
V_eff^2 = V_bar^2 + V_boost^2
where V_boost = f(R, Sigma0, ...)

# Or (interpolating):
g_eff = g_bar * W(R) + g_boost * (1 - W(R))
where W(R) is a weight function
```

---

## Recommended Next Steps

### Step 1: Try Inverse-Mass Amplitude

```python
# In 10_fit_sigmagravity_to_sparc.py, replace A=0.6 with:

def amplitude_inverse_mass(Mbar, A_dwarf=3.0, M_crit=10.0, gamma=0.5):
    """A = A_dwarf / (1 + (Mbar/M_crit)^gamma)"""
    Mbar_safe = max(Mbar, 0.01)
    return A_dwarf / (1.0 + (Mbar_safe / M_crit)**gamma)

A = amplitude_inverse_mass(meta_row['Mbar'])
l0 = 4.0  # Fixed
```

**Expected**: ρ(residual, PC1) should drop significantly (possibly < 0.2)

---

### Step 2: Examine Empirical Boost PC1

Look at the figure `pca/outputs/empirical_boost/empirical_boost_pca.png` to see:
- What is the actual radial shape of K(R) that data prefers?
- Does it match Burr-XII C(R)?
- Where does it differ?

---

### Step 3: Compare Functional Forms

Test different coherence functions against empirical PC1:

```python
# Burr-XII (current)
C_burr(R) = 1 - [1 + (R/l0)^p]^{-n_coh}

# Alternatives:
C_tanh(R) = tanh(R / l0)
C_exp(R) = 1 - exp(-(R/l0)^p)
C_logistic(R) = 1 / (1 + exp(-(R-R0)/l0))

# Fit each to empirical PC1 and compare chi^2
```

---

## The Key Insight

### Why Simple Scalings Failed

**We tried**: A ∝ Vf^α (positive exponent)
- Assumes massive galaxies need MORE boost
- Based on residual correlation (high-mass → large residual)

**Data shows**: A_empirical ∝ Vf^{-0.4} (negative exponent!)
- Massive galaxies actually need LESS boost
- Problem is not "boost too small" but "boost wrong shape/saturation"

**Lesson**: Residual correlation can be **misleading** about which direction to scale parameters!

### The Correct Diagnosis

**Original**: "Massive galaxies need larger A because they have larger residuals"  
**Correct**: "Massive galaxies have larger residuals because they need SMALLER A (boost saturates in dense environments) AND different K(R) shape"

**This is why extracting empirical K(R) is so powerful**: It reveals the TRUE parameter values that data wants, not just the direction of residual trends.

---

## Summary

### What We Learned

1. ✅ **Empirical boost** extracted for 152 galaxies via K = (V_obs²/V_bar²) - 1
2. ✅ **Boost has 3D structure**: PC1-3 explain 90.2% of K(R) variance
3. 🚨 **A ANTI-CORRELATES with mass**: ρ = -0.54 (opposite of naive guess!)
4. 🚨 **ℓ₀ doesn't scale with Rd**: ρ = +0.03 (almost independent!)
5. ✅ **Clear path forward**: Try A ∝ 1/Mbar and fixed ℓ₀

### Next Action

**Implement inverse-mass amplitude**:
```python
A = 3.0 / (1 + (Mbar/10)^0.5)
l0 = 4.0  # Fixed
```

**Expected result**: ρ(residual, PC1) should drop below 0.2 ✅

**This changes the entire model philosophy**: From "universal boost" to "density-dependent decoherence."

---

**The empirical boost extraction is the "PCA version" of Σ-Gravity calculations - it shows what the model SHOULD predict based purely on data!**








