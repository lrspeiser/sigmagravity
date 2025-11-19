# PCA Methodology Alternatives for SPARC Data

## What We Currently Did

**Standard approach**:
- **Weighted PCA** (uncertainty-based weights)
- **Normalization**: R → R/Rd, V → V/Vf
- **Standardization**: Z-score per feature (weighted mean/std)
- **Algorithm**: SVD on weighted, standardized matrix
- **Features**: 50 velocity points on R/Rd grid (0.2-6.0)

**Result**: PC1-3 explain 96.8% of variance, ρ(residual, PC1) = 0.44

---

## Alternative Approaches (8 Categories)

### 1. Weighting Schemes

#### A) **Unweighted PCA** (Equal importance to all points)
```python
# Current: Weight by 1/σ²
# Alternative: All points equally weighted
```

**Pros**: 
- Simpler, more interpretable
- Not biased toward low-error regions
- Tests if findings are robust to weighting choice

**Cons**:
- Ignores measurement quality
- Noisy points get same weight as precise ones

**When to use**: Robustness check - if unweighted gives similar PCs, findings are solid

---

#### B) **Robust Weighting** (Down-weight outliers)
```python
# Use Huber weights or iterative reweighting
# Reduces influence of extreme points
```

**Pros**:
- Robust to data quality issues
- Less sensitive to a few bad measurements

**When to use**: If you suspect outliers are driving PCs

---

#### C) **Radial Weighting** (Emphasize different regions)
```python
# Inner-weighted: More weight to R/Rd < 2
# Outer-weighted: More weight to R/Rd > 3
# Transition-weighted: More weight to R/Rd ~ 1-2
```

**Pros**:
- Tests which radial regions drive the structure
- Can isolate inner vs outer physics

**When to use**: To understand WHERE in radius the PC variance comes from

**We should do this!** - It would show if PC1 correlation comes from inner, outer, or everywhere.

---

### 2. Normalization Variations

#### A) **No Velocity Normalization** (Raw V instead of V/Vf)
```python
# Current: V → V/Vf
# Alternative: Use raw km/s
```

**Pros**: Tests if V/Vf normalization is creating artificial structure

**We already tested this!** - Our robustness script `09_robustness_tests.py` includes this

---

#### B) **Log-Space PCA** (PCA on log(V) instead of V)
```python
# Feature: log(V) vs log(R/Rd)
# Tests scale-invariant structure
```

**Pros**:
- Emphasizes fractional changes, not absolute
- More sensitive to shape than amplitude

**When to use**: If you care about relative variations more than absolute

---

#### C) **Acceleration PCA** (PCA on g = V²/R instead of V)
```python
# Feature: g(R) or even dg/dR
# Directly tests force law
```

**Pros**:
- More directly related to physics (acceleration vs velocity)
- Derivatives can reveal transition points

**We should try this!** - Would show if problem is in V(R) or g(R)

---

### 3. PCA Algorithm Variants

#### A) **Sparse PCA** (Find interpretable loadings)
```python
from sklearn.decomposition import SparsePCA
# Forces loadings to be sparse (many zeros)
```

**Pros**:
- More interpretable - each PC uses fewer features
- Can identify specific radii that matter most

**When to use**: If standard PCA loadings are hard to interpret

---

#### B) **Kernel PCA** (Non-linear structure)
```python
from sklearn.decomposition import KernelPCA
# Uses RBF, polynomial, or sigmoid kernel
```

**Pros**:
- Captures non-linear relationships
- Can find curved manifolds

**Cons**:
- Less interpretable
- More parameters to tune

**When to use**: If you suspect rotation curves lie on curved manifold, not linear subspace

**We should try this!** - Tests if linear PCA is missing non-linear structure

---

#### C) **Incremental PCA** (Memory efficient)
```python
from sklearn.decomposition import IncrementalPCA
# Processes data in batches
```

**When to use**: Huge datasets (not needed for 170 galaxies)

---

### 4. Alternative Decompositions

#### A) **Independent Component Analysis (ICA)**
```python
from sklearn.decomposition import FastICA
# Finds statistically independent sources
```

**Pros**:
- Finds non-Gaussian structure
- Components are independent, not just orthogonal
- Can reveal different physical processes

**Cons**:
- Harder to interpret
- No variance explained metric

**When to use**: If rotation curves are mixtures of independent physical processes

**We should try this!** - Might reveal inner vs outer physics as separate components

---

#### B) **Non-negative Matrix Factorization (NMF)**
```python
from sklearn.decomposition import NMF
# Enforces non-negative components and scores
```

**Pros**:
- Components are additive (easier to interpret)
- No negative values (physical for some quantities)

**When to use**: When features are inherently non-negative

**Could work for g(R) or K(R) but not normalized V(R)**

---

#### C) **Autoencoder** (Deep learning)
```python
# Neural network: Encoder → latent → Decoder
# Already in your toolkit! (script 07)
```

**Pros**:
- Captures non-linear structure
- Can be more flexible than kernel PCA

**When to use**: If PCA misses important non-linear relationships

**You already have the script!** - Just haven't run it yet

---

### 5. Feature Engineering Variants

#### A) **PCA on Residuals** (Not raw curves)
```python
# 1. Compute mean rotation curve
# 2. Residuals = V(R) - V_mean(R)
# 3. PCA on residuals
```

**Pros**:
- Removes trivial "average shape" mode
- Focuses on deviations from mean

**When to use**: If you want to understand variations, not the mean

**This could help!** - Might reduce PC1 dominance and show secondary structure better

---

#### B) **Multi-Scale PCA** (Separate inner/outer)
```python
# PCA on R/Rd < 2 (inner regions)
# PCA on R/Rd > 2 (outer regions)
# Compare the subspaces
```

**Pros**:
- Tests if inner and outer physics are independent
- Can reveal region-specific structure

**When to use**: If you suspect different physics at different radii

**We should do this!** - Would show if PC1 correlation comes from specific radial range

---

#### C) **Derivative/Slope PCA**
```python
# Features: dV/dR or d²V/dR²
# Tests shape curvature, not absolute values
```

**Pros**:
- Emphasizes transitions and inflection points
- Less sensitive to overall normalization

**When to use**: Looking for structural features like "rise", "turnover", "asymptote"

---

### 6. Subset PCA (What Drives Structure?)

#### A) **By Mass Bins**
```python
# Separate PCA for:
# - Dwarfs (Mbar < 1e9)
# - Intermediate (1e9 < Mbar < 10e9)
# - Massive (Mbar > 10e9)
```

**Pros**:
- Tests if structure is mass-dependent
- Can reveal if PCs mean different things in different mass ranges

**We should do this!** - Would clarify if PC1 is "universal" or mass-dependent

---

#### B) **By Surface Brightness**
```python
# We already did HSB vs LSB!
# Result: PC1 nearly identical (4.1° angle)
```

**Already complete** ✅

---

### 7. Grid/Sampling Variations

#### A) **Different Radial Grids**
```python
# Current: Linear grid in R/Rd space (0.2-6.0, 50 points)
# Alternatives:
# - Logarithmic grid: log(R/Rd)
# - Finer inner: More points at small R/Rd
# - Extended outer: Go to R/Rd = 10
```

**Pros**:
- Tests sensitivity to grid choice
- Can emphasize different radial regions

**When to use**: Robustness check

---

#### B) **Varying Number of Components**
```python
# Current: 10 components computed
# Alternative: Analyze stability of PC1-3 as n_components varies
```

**Pros**:
- Tests if PCs are stable
- Can show if 3 is truly sufficient

---

### 8. Model-Residual PCA

#### A) **PCA on Model Residuals** (Not observed curves)
```python
# 1. Compute Σ-Gravity predictions for all galaxies
# 2. Residuals = V_obs - V_model
# 3. PCA on residuals(R/Rd)
```

**Pros**:
- Shows what model systematically misses
- PC1 of residuals = biggest systematic error pattern

**When to use**: Understanding model failures

**We should do this!** - Would directly show what Σ-Gravity is missing

---

## Recommended Next Tests (Priority Order)

### Priority 1: **Radial-Region PCA** (Most Informative)
```python
# Separate PCA for:
# - Inner (R/Rd < 1): Where baryons dominate
# - Transition (1 < R/Rd < 3): Where boost kicks in
# - Outer (R/Rd > 3): Where boost should dominate

# Then check: Which region drives the PC1-Σ-Gravity correlation?
```

**Expected insight**: If correlation comes from outer region, boost shape is wrong. If from inner, transition between g_bar and boost is wrong.

---

### Priority 2: **Acceleration-Space PCA**
```python
# PCA on g(R) = V²/R instead of V(R)
# More directly tests force law
```

**Expected insight**: Σ-Gravity predicts g_eff = g_bar × (1+K). If acceleration-space PCA gives different results, suggests velocity vs acceleration discrepancy.

---

### Priority 3: **Residual-Space PCA** (Direct Diagnostic)
```python
# PCA on (V_obs - V_Σ-Gravity) residuals
# Shows dominant mode of what model misses
```

**Expected insight**: PC1 of residuals directly shows systematic error pattern.

---

### Priority 4: **Kernel PCA or ICA** (Non-linear Check)
```python
# Test if linear PCA is missing non-linear structure
```

**Expected insight**: If kernel PCA or ICA give very different results, linear subspace assumption is wrong.

---

### Priority 5: **Mass-Stratified PCA** (Population Variation)
```python
# Separate PCA for dwarfs vs giants
# Check if PC meanings change
```

**Expected insight**: If dwarf-PC1 ≠ giant-PC1, confirms that "universal" boost structure doesn't exist.

---

## Quick Implementation

### Let me create a script to test the top priorities:

**Script idea**: `15_alternative_pca_methods.py`

**Tests**:
1. Unweighted vs weighted PCA
2. Inner/outer/transition region PCA
3. Acceleration-space PCA
4. Model-residual PCA
5. Mass-stratified PCA

**Output**: Comparison table showing how PCs change across methods

**This would answer**: "Are our findings robust to methodology choice?"

---

## Expected Results

### If All Methods Agree

**PC1-3 capture ~97% variance** across all methods
→ Finding is **robust**, not method-dependent

**ρ(residual, PC1) ~ 0.4-0.5** across all methods  
→ Σ-Gravity failure is **real**, not artifact

**Interpretation**: Results are trustworthy, model really does fail

---

### If Methods Disagree

**Weighted vs unweighted give different PCs**
→ Structure is **driven by high-precision regions**

**Inner vs outer PCA very different**
→ Physics is **radially stratified**, not universal

**Kernel PCA finds more variance**
→ **Non-linear structure** missed by linear PCA

**Interpretation**: Current PCA may be incomplete, try alternatives

---

## Recommendation

### Should we test alternatives?

**YES** - because:
1. **Robustness check**: Confirms findings aren't method artifacts
2. **Diagnostic power**: Shows WHERE problems occur (inner vs outer)
3. **Physical insight**: Different methods reveal different physics
4. **Publication strength**: "Robust across multiple PCA methods" is powerful

**Priority order**:
1. **Radial-region PCA** (where does PC1 come from?)
2. **Residual PCA** (what does model miss?)
3. **Acceleration PCA** (test force law directly)
4. **Unweighted PCA** (robustness check)
5. **Mass-stratified** (population variation)

**Time investment**: ~2-3 hours to implement and run all five
**Value**: High - either confirms findings or reveals new insights

---

## Bottom Line

**Current PCA is ONE valid approach**, but there are **many alternatives** that could:
- Confirm the findings (robustness)
- Reveal where problems are localized (radial regions)
- Show different aspects of structure (acceleration, residuals)
- Test non-linear structure (kernel PCA, autoencoders)

**Should we try them?** 
- **Yes for robustness** (unweighted, different normalizations)
- **Yes for diagnostics** (radial regions, residuals, mass-stratified)
- **Maybe for completeness** (kernel PCA, ICA)

Want me to implement the top 3-5 alternative methods and see what they reveal?









