# Two Types of PCA: What We Did vs What the Templates Offer

## TL;DR

**We ran one type** (curve-shape PCA) successfully and found major insights.  
**The templates offer another type** (parameter-space PCA) that's complementary.  
**Both are worth doing** - they answer different questions!

---

## Type 1: Curve-Shape PCA (What We Already Did)

### What It Analyzes
**Features**: 50 velocity points along R/Rd grid for each galaxy
- Example: V(0.2 Rd), V(0.3 Rd), ..., V(6.0 Rd)
- Matrix: 170 galaxies × 50 radial points

### What It Finds
**Dominant shape modes** across the population:
- PC1: Mass-velocity shape (79.9% variance)
- PC2: Scale-length mode (11.2% variance)
- PC3: Density mode (5.7% variance)

### What It Tells You
- Which **shapes** are common vs rare
- Whether galaxies lie on a low-dimensional manifold
- If model residuals align with empirical modes
- **Tests**: Does model capture population structure?

### Key Results
✅ 96.8% variance in 3 dimensions
✅ HSB/LSB share PC1 (universal)
🚨 Dwarfs and giants are orthogonal (78.7° angle!)
❌ Σ-Gravity fails (ρ = 0.44 with PC1)

---

## Type 2: Parameter-Space PCA (What Templates Offer)

### What It Analyzes
**Features**: Per-galaxy summary statistics
- Structural: Rd, Mstar, Mgas, T-type, inclination
- Kernel: K(2 kpc), K(5 kpc), K(10 kpc)
- Fields: g_bar at 2/5/10 kpc
- Gates: Mean G_bulge, G_bar, G_shear in inner region
- Outcomes: RAR bias/scatter, BTFR residual, Vflat

**Matrix**: N galaxies × ~15-20 summary features

### What It Finds
**Which model features cluster together**:
- PC1 might be: "High K everywhere" vs "Low K everywhere"
- PC2 might be: "Strong inner gates" vs "Weak inner gates"
- PC3 might be: "Baryonic structure" (Mstar, Rd, etc.)

### What It Tells You
- Which **kernel features** drive good performance
- Whether RAR success correlates with K at specific radii
- If gates (Gbulge, Gshear) improve fits
- **Tests**: What makes Σ-Gravity work when it works?

---

## Why Both Are Valuable

### Curve-Shape PCA (Our Analysis)
**Question**: "What empirical structure exists in rotation curves?"
**Answer**: "3D manifold with mass-velocity, scale, density axes"
**Use**: Model-independent empirical target

### Parameter-Space PCA (Templates)
**Question**: "What model features explain variance in outcomes?"
**Answer**: "Which K values, gates, baryonic properties matter most?"
**Use**: Interpret your model's successes and failures

---

## Concrete Example

### Same Galaxy, Different Features

**NGC3198 in curve-shape PCA**:
- Features: [V(0.2Rd), V(0.4Rd), ..., V(6.0Rd)] = [45.2, 68.8, ..., 152.0] km/s
- 50 numbers describing the curve shape

**NGC3198 in parameter-space PCA**:
- Features: [Rd, Mstar, K(2kpc), K(5kpc), RAR_bias, RAR_scatter, ...] 
- ~15-20 numbers describing model behavior and outcomes

---

## What Parameter-Space PCA Would Reveal

### For Galaxies (SPARC)

**Expected PC1**: "Kernel strength axis"
- High loadings: K(2kpc), K(5kpc), K(10kpc) all positive
- Meaning: Galaxies with strong boost everywhere vs weak boost everywhere
- **Tests**: Does PC1 correlate with good RAR scatter?

**Expected PC2**: "Inner vs outer physics"
- High loadings: K(2kpc) positive, K(10kpc) negative (or vice versa)
- Meaning: Boost concentrated in inner vs outer regions
- **Tests**: Does PC2 correlate with morphology (barred, T-type)?

**Key diagnostic**: 
- If **low RAR scatter correlates with PC1** → Boost strength drives success
- If **low RAR scatter correlates with PC2** → Boost distribution drives success

---

### For Clusters (Your Template)

**Expected PC1**: "Geometry vs baryons"
- Either: q_plane, q_LOS, Sigma_crit (geometry)
- Or: fgas, BCG_mass (baryons)
- **Tests**: What drives Einstein radius predictions?

**Expected PC2**: "Amplitude scaling"
- A_c variations across clusters
- **Tests**: Does A scale with mass/geometry as expected?

---

## Recommendation: YES, Run Both!

### What You'd Gain

**From parameter-space PCA** (templates):
1. Understand which kernel features drive your 0.087 dex RAR success
2. Test if gates (Gbulge, Gbar, Gshear) improve fits
3. See if K(5kpc) matters more than K(2kpc) or K(10kpc)
4. Check if RAR/BTFR outcomes cluster with specific model features

**This answers**: "WHY does Σ-Gravity work well on RAR?"

**Combined with curve-shape PCA**:
- Curve PCA: "Model fails population structure test"
- Parameter PCA: "But succeeds because [specific kernel features]"
- **Synthesis**: "Model captures global amplitudes (parameter PCA) but misses shape variations (curve PCA)"

---

## Quick Implementation Plan

### Step 1: Build Real Features (Fix Current Script)
The current feature builder needs better radial interpolation. Let me fix it to handle varying radial coverage.

### Step 2: Run Parameter-Space PCA
Use the uploaded `sigma_pca_runner.py` or our own implementation

### Step 3: Interpret Results
- Which features load on PC1?
- Does RAR scatter correlate with any PC?
- Do K values at different radii cluster together or separately?

### Step 4: Compare to Curve-Shape PCA
- Do they agree on mass-dependence?
- Do they reveal different aspects?
- Can you combine insights?

---

## Expected Timeline

**Feature building**: 10 minutes (fix interpolation issues)
**Running PCA**: 1 minute  
**Analysis**: 30 minutes (interpret loadings, correlations)
**Documentation**: 30 minutes

**Total**: ~1.5 hours for complete parameter-space analysis

---

##Bottom Line

**Your question**: "Maybe worth trying with real data?"

**Answer**: **YES - absolutely!**

**The templates offer a complementary PCA** that tests:
- Which **model parameters** matter (our curve PCA tested shapes)
- What drives **RAR/BTFR success** (our curve PCA tested failures)
- Whether **kernel features** cluster meaningfully

**This would show WHY your model works on global metrics** even though it fails the curve-shape manifold test.

Want me to fix the feature builder and run the full parameter-space PCA analysis?




