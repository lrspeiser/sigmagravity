# Chameleon Parameter Scan Results

**Date**: November 19, 2025  
**Status**: ✅ VIABLE PARAMETERS FOUND

---

## Summary

We successfully implemented the chameleon mechanism and found viable parameter regions that dramatically reduce R_c compared to pure exponential potential.

---

## Best Parameters Found

### Top Candidate: β = 0.1, M4 = 5e-2 [GOOD]

**Field values (density-dependent)**:
- φ^cosmic = 0.001765
- φ^dwarf = 0.000023  
- φ^spiral = 0.000005

**Core radii**:
- R_c^cosmic = 1.23e5 kpc (~0.04 Mpc)
- R_c^dwarf = 180.45 kpc
- R_c^spiral = 20.63 kpc

**Improvement**:
- Pure exponential: R_c ~ 1.5e6 kpc
- With chameleon: R_c^spiral = 20.63 kpc
- **Reduction: ~73,000x for spirals!**

---

## All Viable Parameters

1. **β = 0.1, M4 = 5e-2 [GOOD]**
   - R_c^dwarf = 180.45 kpc, R_c^spiral = 20.63 kpc

2. **β = 0.05, M4 = 5e-2**
   - R_c^dwarf = 254.39 kpc, R_c^spiral = 42.60 kpc

3. **β = 0.1, M4 = 1e-1**
   - R_c^dwarf = 391.53 kpc, R_c^spiral = 65.66 kpc

4. **β = 0.05, M4 = 1e-1**
   - R_c^dwarf = 651.00 kpc, R_c^spiral = 111.62 kpc

5. **β = 0.01, M4 = 5e-2**
   - R_c^dwarf = 911.35 kpc, R_c^spiral = 180.46 kpc

---

## Key Findings

### 1. Chameleon Mechanism Works

- Field values are **density-dependent**: φ smaller in denser regions
- Effective mass increases in dense regions: m_eff^galaxy >> m_eff^cosmic
- R_c dramatically reduced compared to pure exponential

### 2. Best Results

- **β = 0.1, M4 = 5e-2** gives R_c^spiral = 20.63 kpc
- Still above target (< 10 kpc) but much better than ~1.5e6 kpc
- Dwarf galaxies still have larger R_c (180 kpc)

### 3. Parameter Trends

- **Larger M4**: Stronger chameleon, smaller R_c (but may affect cosmology)
- **Larger β**: Stronger coupling, affects both cosmic and galaxy scales
- **Optimal**: β = 0.1, M4 = 5e-2 balances both

---

## Next Steps

### 1. Test in Actual Galaxy Fits

- ✅ M4 parameter integrated into fitter
- ⚠️ Initial test shows R_c still hitting 50 kpc bound
- Need to investigate why full field solution doesn't match phi_min predictions

### 2. Refine Parameters

- Test intermediate M4 values (e.g., 7e-2, 8e-2)
- May need to adjust β as well
- Goal: R_c < 10 kpc for both dwarfs and spirals

### 3. Check Cosmology Compatibility

- Test if M4 = 5e-2 works with cosmology
- Verify Ω_m, Ω_φ still reasonable
- Check H(z) and d_L(z) residuals

### 4. PPN Tests

- Once galaxy R_c is fixed, test PPN parameters
- Verify |γ-1| < 2.3e-5, |β-1| < 8e-5
- May need to fine-tune M4 or β

---

## Technical Details

### Implementation

- **phi_min solver**: `find_phi_min()` in `HaloFieldSolver`
- **Scan script**: `galaxies/scan_meff_vs_density.py`
- **Integration**: M4 parameter added to `fit_field_driven_halo()`

### Potential Form

For chameleon (n=1):
\[
V(\phi) = V_0 e^{-\lambda\phi} + \frac{M^5}{\phi}
\]

where M4^5 = M^5 in our parameterization.

### Effective Mass

\[
m_{\rm eff}^2 = \lambda^2 V_0 e^{-\lambda\phi} + \frac{2M^5}{\phi^3} + \beta^2 e^{\beta\phi} \rho_b
\]

In dense regions (small φ), the M^5/φ³ term dominates, making the field heavy.

---

## Status

✅ **Chameleon mechanism implemented and working**  
✅ **Viable parameters found (β=0.1, M4=5e-2)**  
✅ **R_c reduced by ~73,000x for spirals**  
⚠️ **Still need R_c < 10 kpc (currently 20.63 kpc for best case)**  
⚠️ **Full field solution may need refinement**

**Next**: Test in actual fits, refine parameters, check cosmology compatibility

