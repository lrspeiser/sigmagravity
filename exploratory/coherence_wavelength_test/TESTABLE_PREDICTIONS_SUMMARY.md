# Σ-Gravity: Testable Predictions and Data Requirements

**Date:** December 2025  
**Status:** Summary of what we can test now vs. what data we need

---

## Executive Summary

Σ-Gravity makes **specific, quantitative predictions** that can be tested against observations. We have data for some tests; others require downloading additional datasets.

| Test | Have Data? | Status |
|------|------------|--------|
| g† = cH₀/(4√π) value | ✓ YES (SPARC) | **Can test NOW** |
| g†(z) redshift evolution | ✗ NO | **NEED DATA - KEY TEST** |
| h(g) functional form | ✓ YES (SPARC RAR) | **Can test NOW** |
| Cluster lensing | ✓ YES (Fox+ 2022) | **Can test NOW** |
| Milky Way | ✓ YES (Gaia) | **Can test NOW** |
| Counter-rotating systems | ✗ NO | **NEED DATA - UNIQUE TEST** |
| Wide binaries | ✗ NO | **NEED DATA - CONTROVERSIAL** |

---

## PREDICTION 1: Critical Acceleration Value

### The Prediction
$$g^\dagger = \frac{cH_0}{4\sqrt{\pi}} = 9.59 \times 10^{-11} \text{ m/s}^2$$

This is **20% lower** than MOND's empirical a₀ = 1.2 × 10⁻¹⁰ m/s².

### Data Available
- **SPARC database:** 175 galaxy rotation curves ✓
- **Location:** `/data/Rotmod_LTG/`

### Test
Compare best-fit g† from rotation curves:
- If best fit ≈ 9.6 × 10⁻¹¹ → Supports Σ-Gravity
- If best fit ≈ 1.2 × 10⁻¹⁰ → Supports MOND empirical value

### Status: **CAN TEST NOW**

---

## PREDICTION 2: Redshift Evolution (THE KEY TEST)

### The Prediction
$$g^\dagger(z) = \frac{cH(z)}{4\sqrt{\pi}} = g^\dagger(0) \times \frac{H(z)}{H_0}$$

| z | H(z)/H₀ | g†(z)/g†(0) | Effect on Dark Matter Fraction |
|---|---------|-------------|--------------------------------|
| 0 | 1.0 | 1.0 | Baseline |
| 1 | 1.8 | 1.8 | ~30% less enhancement |
| 2 | 3.0 | 3.0 | ~45% less enhancement |
| 3 | 4.5 | 4.5 | ~55% less enhancement |

### Why This Is The Key Test
- **ΛCDM predicts:** Roughly constant dark matter fraction at fixed halo mass
- **MOND predicts:** No natural redshift evolution (a₀ is constant)
- **Σ-Gravity predicts:** Specific H(z) scaling from first principles

This is **UNIQUE** to Σ-Gravity. If confirmed, it's strong evidence for the postulates.

### Data Needed
1. **KMOS³D Survey**
   - URL: https://www.mpe.mpg.de/ir/KMOS3D
   - Contains: Rotation curves for z ~ 0.6-2.7 galaxies
   - ~600 galaxies with spatially-resolved kinematics

2. **Genzel et al. (2017, 2020) Data**
   - Paper: Nature 543, 397 (2017)
   - Contains: Dark matter fractions at z = 0, 1, 2
   - Key result: f_DM decreases from ~50% (z=0) to ~20% (z=2)
   - Need: Supplementary data tables

3. **SINS Survey**
   - URL: https://www.mpe.mpg.de/ir/SINS
   - Contains: z ~ 2 star-forming galaxies

### Existing Evidence (Not Yet Downloaded)
Genzel et al. (2020) found f_DM decreases with z, qualitatively matching our prediction. But we need the actual data to test quantitatively.

### Status: **NEED TO DOWNLOAD DATA**

---

## PREDICTION 3: Enhancement Function h(g)

### The Prediction
$$h(g) = \sqrt{\frac{g^\dagger}{g}} \times \frac{g^\dagger}{g^\dagger + g}$$

This differs from MOND's interpolating functions:
- MOND simple: ν(x) = 1/(1-e^(-√x))
- MOND standard: ν(x) = (1 + √(1+4/x))/2

### Data Available
- **SPARC RAR data:** McGaugh et al. (2016) ✓
- **Location:** Can be extracted from SPARC rotation curves

### Test
Fit the Radial Acceleration Relation with:
1. Σ-Gravity h(g)
2. MOND simple ν(x)
3. MOND standard ν(x)

Compare scatter and residuals.

### Status: **CAN TEST NOW**

---

## PREDICTION 4: Galaxy Clusters

### The Prediction
Clusters require A = π√2 ≈ 4.44 (vs A = √3 ≈ 1.73 for disks).

### Data Available
- **Fox+ 2022 clusters:** 75 unique clusters ✓
- **Location:** `/data/clusters/fox2022_unique_clusters.csv`
- Contains: M500, MSL_200kpc, redshifts

### Test
Compare enhanced baryonic mass to lensing mass:
- M_enhanced = M_bar × Σ(cluster)
- Should match MSL_200kpc

### Status: **CAN TEST NOW** (already done, see earlier validation)

---

## PREDICTION 5: Counter-Rotating Systems

### The Prediction
Counter-rotating disks should show **REDUCED enhancement** because:
- Co-rotating material builds coherence together
- Counter-rotating material doesn't add coherently
- Unique prediction not shared by MOND or ΛCDM

### Data Needed
Detailed kinematics of:
- **NGC 4550:** Two counter-rotating stellar disks
- **NGC 7217:** Counter-rotating gas disk
- **NGC 4138:** Counter-rotating gas disk

Need:
- Separate rotation curves for each component
- Mass models for baryonic components
- Comparison of "dark matter" in co- vs counter-rotating regions

### Why This Is Important
This would be a **decisive test** of coherence-based theories. Neither MOND nor ΛCDM predicts different behavior for counter-rotating components.

### Status: **NEED SPECIALIZED DATA**

---

## PREDICTION 6: Milky Way

### The Prediction
Same formula applies to Milky Way with g† = cH₀/(4√π).

### Data Available
- **Gaia DR3 data:** ✓
- **Location:** `/vendor/maxdepth_gaia/gaia_bin_residuals.csv`

### Status: **CAN TEST NOW** (already done, see Gaia validation)

---

## PREDICTION 7: Wide Binary Stars

### The Prediction
At separations > 7000 AU, binary orbits should show MOND-like deviations because g < g† ≈ 10⁻¹⁰ m/s².

### Data Needed
- **Gaia DR3 wide binary catalog**
- **Chae (2023) analysis:** Claims detection
- **Banik et al. (2024) reanalysis:** Disputes detection

### Status: **NEED DATA - CONTROVERSIAL**

This test is currently disputed in the literature. Results depend on selection criteria and systematic error treatment.

---

## Priority Data Downloads

### 1. HIGH-Z ROTATION CURVES (CRITICAL)

**Source:** KMOS³D Survey
**URL:** https://www.mpe.mpg.de/ir/KMOS3D
**What to get:**
- Rotation curve catalogs
- Galaxy properties (mass, size, redshift)
- Kinematic maps if available

**Why critical:** This is the **unique prediction** that distinguishes Σ-Gravity from both MOND and ΛCDM.

### 2. GENZEL ET AL. DATA

**Source:** Nature 543, 397 (2017) supplementary materials
**What to get:**
- Dark matter fractions vs redshift
- Individual galaxy measurements
- Error bars and selection criteria

**Why important:** Direct comparison to our H(z) scaling prediction.

### 3. COUNTER-ROTATING GALAXY DATA

**Source:** Literature / Author requests
**What to get:**
- NGC 4550, NGC 7217, NGC 4138 kinematics
- Separate rotation curves for each component
- Mass models

**Why important:** Unique test of coherence mechanism.

---

## What Would Prove the Theory?

### Strong Evidence
1. **g† = 9.6 × 10⁻¹¹** fits better than a₀ = 1.2 × 10⁻¹⁰
2. **g†(z) follows H(z)** quantitatively across z = 0, 1, 2, 3
3. **h(g) form** fits RAR better than MOND functions

### Decisive Evidence
4. **Counter-rotating systems** show reduced enhancement
5. **Time-dependent effects** in recently disturbed systems

### What Would Falsify It
- g† significantly different from cH₀/(4√π)
- No redshift evolution of g†
- Counter-rotating systems show same enhancement as normal
- h(g) form inconsistent with RAR

---

## Immediate Next Steps

1. **Run g† comparison test** on SPARC data (have data)
2. **Run h(g) vs MOND** comparison on RAR (have data)
3. **Download KMOS³D data** for redshift test
4. **Find Genzel supplementary data** for f_DM vs z
5. **Search literature** for counter-rotating galaxy kinematics

---

## Conclusion

Σ-Gravity makes **falsifiable predictions**. The key test is **redshift evolution of g†**, which is:
- Unavoidable from the postulates
- Quantitatively specific
- Different from all competing theories
- Becoming testable with JWST-era data

The theory stands or falls on whether g†(z) = cH(z)/(4√π) is confirmed.

