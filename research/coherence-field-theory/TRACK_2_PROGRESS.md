# Track 2 Progress: Unit Fix & Field-Driven Fitting

**Date**: November 19, 2025  
**Status**: IN PROGRESS  
**Achievement**: Units fixed, field-driven fitter created

---

## Completed Work

### ✅ 1. Fixed Units/Scaling
**File**: `galaxies/halo_field_profile.py`

**Changes**:
- Added `convert_to_mass_density` parameter to `effective_density()`
- Proper conversion from cosmology units (H0²) to mass density (M_sun/kpc³)
- Uses critical density: ρ_crit = 3H0²/(8πG) ≈ 9.5e9 M_sun/kpc³
- Converts V(φ) from cosmology units to physical density

**Key Fix**:
```python
# Critical density conversion
H0_kms_kpc = 70.0 / 306.6  # km/s/kpc
H0_squared = H0_kms_kpc**2
rho_crit = 3 * H0_squared / (8 * np.pi * G)  # M_sun/kpc³
potential = potential * rho_crit  # Convert to mass density
kinetic = kinetic * rho_crit
```

This should significantly improve the chi-squared for halo fitting.

### ✅ 2. Field-Driven Galaxy Fitter
**File**: `galaxies/fit_field_driven.py` (NEW, 283 lines)

**Features**:
- `FieldDrivenSPARCFitter` class
- Uses `HaloFieldSolver` to derive halos from field parameters
- Fits only baryonic parameters (M_disk, R_disk) per galaxy
- Field parameters (V₀, λ, β) are global (shared across galaxies)
- Compares predicted vs fitted halo parameters

**Key Methods**:
- `fit_field_driven_halo()` - Main fitting routine
- `rho_baryon_profile()` - Baryon density for field solver
- `test_field_driven_fit()` - Test on real galaxy (CamB)

**Usage**:
```python
fitter = FieldDrivenSPARCFitter()
data = fitter.load_galaxy('CamB')
result = fitter.fit_field_driven_halo(data, V0=1e-6, lambda_param=1.0, beta=0.1)
```

This moves from **"free halo parameters per galaxy"** to **"halos derived from shared field parameters"**.

---

## Next Steps

### Immediate
1. **Test field-driven fitter on real galaxy**
   - Run `fit_field_driven.py` on CamB or DDO161
   - Compare predicted (ρ_c0, R_c) vs fitted from `sparc_fit_summary.csv`
   - Check if chi² is reasonable

2. **Refine unit conversion**
   - Verify conversion factor produces realistic densities
   - Test on multiple galaxies
   - Adjust if needed

3. **Connect to cosmology**
   - Get φ(∞) from `CoherenceCosmology.phi_0` (today's value)
   - Use same (V₀, λ) as cosmology for consistency
   - Ensure units match between cosmology and halo solver

### Medium-term
1. **Test on multiple galaxies**
   - Fit 2-3 galaxies with same (V₀, λ, β)
   - Check if single parameter set works for multiple galaxies
   - This is the "unification test"

2. **Compare with phenomenological fits**
   - See if field-driven halos can match fitted halos
   - If close, theory is predictive!

---

## Status

**Track 2 Foundation**: ✅ COMPLETE (solver working)
**Track 2 Refinement**: 🚧 IN PROGRESS
- [x] Units/scaling fixed
- [x] Field-driven fitter created
- [ ] Tested on real galaxy
- [ ] Connected to cosmology
- [ ] Tested on multiple galaxies

**Next Priority**: Test field-driven fitter on real galaxy

---

**Commit**: In progress  
**Status**: Units fixed, field-driven fitter ready for testing  
**Next**: Run on real galaxy, compare with phenomenological fits

