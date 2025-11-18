# PCA + Σ-Gravity: Deeper Diagnosis

## What the Calibration Revealed

### Simple Scalings Don't Work

**Tried**: A = A₀ × (Vf/100)^α, ℓ₀ = ℓ₀,₀ × (Rd/5)^β

**Best found**: 
- A₀ = 0.10, α = 0.30
- ℓ₀,₀ = 4.0 kpc, β = 0.30

**Result**:
- ρ(residual, PC1): 0.459 → 0.417 (9% improvement, still FAIL)
- Mean RMS: 33.85 → 33.29 km/s (1.6% improvement)

**Conclusion**: Power-law parameter scalings are **not sufficient** to capture the empirical structure.

---

## Why This is Actually Revealing

### The Deeper Problem

The correlation structure suggests the issue is **not just parameter values** but the **model structure itself**:

**Direct correlations with physics**:
- ρ(residual, Vf) = **+0.781** (strongest!)
- ρ(residual, Mbar) = +0.707
- ρ(residual, Rd) = +0.524

**After adding A(Vf), ℓ₀(Rd) scalings**:
- ρ(residual, PC1) only drops to 0.417 (still strong)

**Implication**: The problem is not that A and ℓ₀ have the wrong values - it's that the **functional form** g_eff = g_bar × (1 + K) may be too simple.

---

## What the PCA Structure Suggests

### PC1 Loadings (Need to Check)

Let's examine **where** in radius the PC1 loading is strongest. This tells us where the model is failing.

**Hypothesis 1**: Inner region dominance
- If PC1 loading peaks at small R/Rd (< 1)
- Model fails in inner regions where g_bar transitions to Σ-Gravity boost
- **Fix**: Need different inner/outer boost structure

**Hypothesis 2**: Outer region dominance  
- If PC1 loading peaks at large R/Rd (> 3)
- Model fails in outer regions where dark matter should dominate
- **Fix**: Need stronger boost at large radii or different R-dependence

**Hypothesis 3**: Transition region
- If PC1 loading peaks at R/Rd ~ 1-2
- Model fails at transition between baryon-dominated and boost-dominated
- **Fix**: Need different coherence function shape

---

## Diagnostic Actions

### Action 1: Examine PC1 Radial Loading Profile

```python
import numpy as np
import matplotlib.pyplot as plt

pca = np.load('pca/outputs/pca_results_curve_only.npz')
curve_data = np.load('pca/data/processed/sparc_curvematrix.npz', allow_pickle=True)

components = pca['components']
x_grid = curve_data['x_grid']  # R/Rd grid

# Plot PC1 loading
plt.figure(figsize=(10, 6))
plt.plot(x_grid, components[0, :], 'o-', lw=2, ms=4)
plt.axhline(0, color='k', ls='--', alpha=0.3)
plt.xlabel('R / Rd', fontsize=12)
plt.ylabel('PC1 Loading', fontsize=12)
plt.title('PC1 Radial Loading Profile (79.9% variance)', fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# Find where loading is strongest
max_idx = np.argmax(np.abs(components[0, :]))
print(f"PC1 loading peaks at R/Rd = {x_grid[max_idx]:.2f}")
```

**This tells us WHERE the model is failing most.**

---

### Action 2: Residual Analysis by Radial Bin

Check if residuals are localized to specific radii:

```python
# For each galaxy, compute residual vs radius
# then average over galaxies in high-residual vs low-residual groups

# High-residual galaxies (worst 25%)
# Low-residual galaxies (best 25%)

# Plot average residual(R) for each group
# This shows if the failure is inner-dominated, outer-dominated, or everywhere
```

---

### Action 3: Test Alternative Model Structures

The PCA suggests trying:

#### Option A: Two-Component Boost
```python
# Instead of: g_eff = g_bar * (1 + K)
# Try: g_eff = g_bar * (1 + K_inner + K_outer)

K_inner = A_inner * C(R / l0_inner) * exp(-R/R_trans)
K_outer = A_outer * C(R / l0_outer) * (1 - exp(-R/R_trans))
```

#### Option B: Baryonic-Matter Dependent Boost
```python
# Make boost depend on local baryonic content
K = A * C(R / l0) * f(g_bar / g_crit)

# Where f(x) is a function that modulates the boost based on local g_bar
```

#### Option C: Radial-Form Modification
```python
# Try different coherence functions
# Current: Burr-XII
# Alternatives: tanh(), 1 - exp(), logistic, etc.
```

---

## PCA-Informed Next Steps

### Strategy 1: Data-Driven Functional Form

Use PCA to **learn the boost function** empirically:

1. For each galaxy, compute: K_empirical = (V_obs² / V_bar²) - 1
2. Stack K_empirical(R) for all galaxies
3. Run PCA on K_empirical to find empirical boost modes
4. Fit parametric form to match empirical PC1 of boost

**This lets PCA tell you what K(R) should look like!**

---

### Strategy 2: Hybrid Approach

**Current problem**: g_bar itself has structure that doesn't match Σ-Gravity boost structure

**Solution**: 
1. Compute "ideal" g_total = g that would give V_obs
2. Extract K_ideal = (g_total / g_bar) - 1  
3. Fit Σ-Gravity kernel form to match K_ideal across population
4. Check if ANY parametric form K(R; params) can match empirical structure

**This tests whether the Burr-XII coherence function is the right shape.**

---

### Strategy 3: Physical Refinements

Based on "many-path" reasoning:

#### Path Density Weighting
```python
# Paths scale with enclosed baryonic mass
# Weight boost by local baryon fraction
n_paths ∝ M_bar_enclosed(R)
A_eff(R) = A0 * (M_bar(<R) / M_bar_total)^gamma
```

#### Decoherence Rate
```python
# Decoherence stronger in dense regions
# Modify coherence function to depend on local Σ
C(R) = C_base(R) * exp(-Sigma(R) / Sigma_coh)
```

---

## Recommended Next Actions

### Priority 1: Examine Where Model Fails

```python
# Run this to see radial structure of failures
python -i pca/explore_results.py

# Then:
>>> import matplotlib.pyplot as plt
>>> # Plot PC1 loading
>>> x_grid = np.load('pca/data/processed/sparc_curvematrix.npz', allow_pickle=True)['x_grid']
>>> components = np.load('pca/outputs/pca_results_curve_only.npz')['components']
>>> plt.figure(figsize=(10, 6))
>>> plt.plot(x_grid, components[0, :], 'o-', lw=2)
>>> plt.axhline(0, color='k', ls='--')
>>> plt.xlabel('R/Rd'); plt.ylabel('PC1 Loading')
>>> plt.title('PC1: Where is the dominant variance?')
>>> plt.grid(alpha=0.3)
>>> plt.show()
```

**This shows you where (in radius) the model needs to improve.**

---

### Priority 2: Extract Empirical Boost Function

Create a script that learns K(R) from data:

```python
# For each SPARC galaxy:
# 1. V_obs(R), V_bar(R) are known
# 2. g_obs = V_obs^2 / R
# 3. g_bar = V_bar^2 / R  
# 4. K_empirical = (g_obs / g_bar) - 1
# 5. Normalize: K_norm = K / A_fit (where A_fit chosen per galaxy)

# Then: Run PCA on K_norm(R/Rd) curves
# This tells you the empirical shape of the boost function!
```

This would reveal if Burr-XII C(R) is the right functional form.

---

### Priority 3: Consider Fundamental Model Revision

The persistent correlations suggest the multiplicative form g_eff = g_bar × (1 + K) may be limiting.

**Alternatives to explore**:

#### Additive Boost
```python
g_eff = g_bar + g_boost
# where g_boost = A * C(R) * some_function(R, Sigma)
```

#### Interpolating Form
```python
g_eff = g_bar * f(K) + g_DM * (1 - f(K))
# Interpolates between baryonic and modified gravity
```

#### Radius-Dependent Mixing
```python
g_eff = g_bar * (1 + K(R, Sigma(R), M_enc(R)))
# K depends on LOCAL conditions, not just global galaxy properties
```

---

## Physical Interpretation

### What We're Learning

The PCA test is revealing that:

1. **Simple global scalings insufficient**: A(Vf), ℓ₀(Rd) only reduce ρ from 0.46 → 0.42
2. **Structure matters more than parameters**: The functional form g_bar × (1+K) may be limiting
3. **Local vs global**: May need boost to depend on LOCAL baryon distribution, not just global Vf, Rd

### Implications for "Many-Path" Physics

If the boost really comes from "coherent superposition of paths":
- **Path statistics** should depend on enclosed mass M(<R), not just total mass
- **Coherence** should depend on local density Σ(R), not just global Σ₀
- **Decoherence** should vary with environment (dense inner disk vs sparse outer disk)

**This suggests**: Need to make K = K(R, M_enc(R), Σ(R)) rather than K(R; A(Mbar), ℓ₀(Rd))

---

## Bottom Line

### What the Grid Search Showed

Simple power-law parameter scalings are **necessary but not sufficient**:
- ✅ They help (9% improvement in ρ)
- ❌ They don't solve the problem (still ρ = 0.42 >> 0.2)

### The Real Issue

The PCA diagnostic suggests the problem is **structural**, not parametric:
- The boost form g_bar × (1 + A·C(R/ℓ₀)) may be too rigid
- Need boost to adapt to LOCAL baryonic distribution
- May need R-dependent amplitude or multi-component boost

### Recommended Path Forward

**Option 1** (Conservative): Extract empirical K(R) via PCA and fit better functional form

**Option 2** (Radical): Revise model to make boost depend on local baryon density, not global parameters

**Option 3** (Pragmatic): Accept that fixed/scaled parameters won't fully pass the test; report that model captures ~60% of structure but misses systematic mass-velocity effects

---

## What to Do Next

### Immediate: Examine PC1 Loading Profile
```bash
# Check the radial loading plot that was already generated
# (or regenerate with script above)
```

**This tells you**: Does PC1 variance come from inner regions, outer regions, or transition?

### Then: Choose Strategy

**If PC1 peaks at inner R**: Focus on improving inner boost
**If PC1 peaks at outer R**: Focus on improving outer boost  
**If PC1 is broad**: Need to revise overall functional form

### Finally: Decide on Model Philosophy

**Path A**: Keep pursuing parameter scalings (more complex forms)
**Path B**: Accept "partial capture" and report diagnostics
**Path C**: Fundamental model revision based on PCA-extracted K(R)

---

**Current status**: Simple scalings tested, found insufficient. Deeper diagnosis needed to choose next approach.






