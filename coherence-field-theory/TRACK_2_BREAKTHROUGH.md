# Track 2 Breakthrough: Field Now Responds to Baryons!

**Date**: November 19, 2025  
**Status**: ✅ MAJOR PROGRESS - Field coupling fixed!

---

## The Problem

The field solution was producing constant, very low density (~1.37e-3 M_sun/kpc³) because the coupling term wasn't creating a force on the field.

**Root Cause**: Incorrect coupling form
- Used: `V_eff = V(φ) + (d ln A/dφ) * ρ_b = V(φ) + β * ρ_b`
- This is **constant in φ**, so `dV_eff/dφ = dV/dφ` (no coupling term!)
- Field doesn't respond because there's no φ-dependent force

---

## The Solution

**Corrected coupling form**:
- `V_eff(φ) = V(φ) + A(φ) * ρ_b` where `A(φ) = e^(βφ)`
- `dV_eff/dφ = dV/dφ + β * e^(βφ) * ρ_b`
- Now the coupling **depends on φ**, creating a force!

**Implementation**:
- Updated `Veff()` and `dVeff_dphi()` in `halo_field_profile.py`
- Added overflow protection: `phi_safe = np.clip(phi, -10.0, 10.0)`
- Use smaller β values (0.01) to prevent numerical instabilities

---

## Results (CamB Galaxy)

### Before Fix:
- Field-driven: ρ_c0 = 1.37e-3 M_sun/kpc³ (constant, ~6 million times too low!)
- Field didn't respond to baryons at all

### After Fix (β = 0.01):
- Field-driven: ρ_c0 = 1.05e6 M_sun/kpc³, R_c = 11.7 kpc
- Phenomenological: ρ_c0 = 8.16e6 M_sun/kpc³, R_c = 3.08 kpc
- **Ratio: 0.13x (density), 3.8x (radius)**
- **Field density now varies by 5 orders of magnitude!** (3e1 to 2e6)

### Comparison:
| Metric | Field-Driven | Phenomenological | Ratio |
|--------|--------------|-------------------|-------|
| ρ_c0   | 1.05e6       | 8.16e6            | 0.13x |
| R_c    | 11.7 kpc     | 3.08 kpc          | 3.8x  |
| χ²_red | 0.923        | 0.027              | 34x   |

---

## What This Means

✅ **Field is now responding to baryons!**
- Density varies with radius (not constant)
- Field solution creates realistic halo profiles
- Within order of magnitude of phenomenological fits

⚠️ **Still needs tuning:**
- Density is ~8x too low (need stronger coupling or different β)
- R_c is ~4x too large (field extends too far)
- Chi² is 34x worse (but field-driven has fewer free parameters)

---

## Next Steps

1. **Tune β parameter**
   - Try β = 0.05, 0.1, 0.2
   - Find optimal value for density match

2. **Tune V₀ and λ**
   - May need different potential parameters
   - Test with cosmology constraints

3. **Test on multiple galaxies**
   - See if single (V₀, λ, β) works for multiple galaxies
   - This is the key test for predictive power

4. **Improve numerical stability**
   - Current clipping may be too restrictive
   - Consider better normalization

---

## Files Changed

- `galaxies/halo_field_profile.py` - Fixed Veff and dVeff_dphi
- `galaxies/debug_field_response.py` - Debug script
- `examples/test_field_driven_galaxy.py` - Updated test

---

**Status**: Field coupling working! Now tuning parameters for best fit.  
**Commit**: Fixed coupling form - field now responds to baryons

