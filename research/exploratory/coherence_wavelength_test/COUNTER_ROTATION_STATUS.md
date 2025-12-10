# Counter-Rotating Galaxy Test: Current Status

## The Problem

You're absolutely right - our other tests involve:
- **175 SPARC galaxies** for rotation curves
- **612 KMOS³D galaxies** for high-z evolution  
- **42 clusters** for lensing
- **108,000 Gaia stars** for Milky Way

A single-galaxy test (NGC 4550) is **not sufficient** to draw conclusions.

## What We Have

### Available Data Sources

1. **Bevacqua et al. 2022 (MaNGA)**: 64 counter-rotating galaxies
   - Has: stellar mass, morphology, age, metallicity
   - **Missing**: dynamical masses, rotation curves
   - VizieR: J/MNRAS/511/139

2. **ATLAS3D (Krajnović et al. 2011)**: 11 counter-rotating systems
   - Has: IFU kinematics
   - **Missing**: systematic dynamical mass analysis

3. **NGC 4550 (Johnston et al. 2013)**: Single galaxy
   - Has: kinematics, stellar populations
   - **Missing**: robust M/L ratio, dynamical mass

### The Gap

To properly test the counter-rotation prediction, we need:
- **M_dyn** (dynamical mass from rotation curves)
- **M_bar** (baryonic mass from photometry)
- For **both** counter-rotating AND matched normal galaxies

This data combination doesn't exist in a ready-to-use form.

## Current NGC 4550 Result

Our analysis found M_dyn/M_star ≈ 0.72, which is:
- Less than 1 (problematic - suggests M/L overestimated)
- Closer to Σ-Gravity (1.84) than MOND (2.56)
- But with ~40% uncertainty

**This is NOT a robust result** - it's a single galaxy with large uncertainties.

## What Would Be Needed for a Proper Test

### Option 1: Cross-match MaNGA data
- Get dynamical masses for the 64 Bevacqua et al. counter-rotating galaxies
- Compare to matched control sample of normal galaxies
- This is a significant analysis project

### Option 2: Literature search
- Find published M_dyn/M_bar ratios for counter-rotating galaxies
- Compare to normal galaxies from same studies
- May not exist

### Option 3: Wait for future surveys
- DESI, 4MOST will provide more kinematic data
- May enable statistical counter-rotation tests

## Recommendation

**Do NOT add the NGC 4550 single-galaxy result to the main documentation.**

The counter-rotation prediction remains a **testable prediction** that distinguishes Σ-Gravity from MOND, but we don't yet have the data to test it properly.

Keep the prediction in SUPPLEMENTARY_INFORMATION.md §6.4 as a "future test" without claiming any observational support.

## Summary Table

| Test | Sample Size | Status |
|------|-------------|--------|
| SPARC rotation curves | 175 | ✅ Complete |
| RAR scatter | 175 | ✅ Complete |
| High-z evolution | 612 | ✅ Complete |
| Clusters | 42 | ✅ Complete |
| Milky Way | 108k | ✅ Complete |
| **Counter-rotation** | **1** | ❌ **Insufficient** |

The counter-rotation test requires a proper statistical sample, not a single galaxy.

