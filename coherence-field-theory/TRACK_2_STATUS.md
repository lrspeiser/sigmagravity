# Track 2 Status: Field-Driven Halos

**Date**: November 19, 2025  
**Status**: IN PROGRESS - Solver working, needs parameter tuning

---

## Current Status

### ✅ Completed
1. **Halo field solver working** (`galaxies/halo_field_profile.py`)
   - Solves static Klein-Gordon equation
   - Produces φ(r) solutions
   - Computes effective density ρ_φ(r)

2. **Unit conversion implemented**
   - Converts from cosmology units (H0²) to mass density (M_sun/kpc³)
   - Uses critical density: ρ_crit = 3H0²/(8πG)

3. **Field-driven fitter created** (`galaxies/fit_field_driven.py`)
   - Fits galaxies with halos derived from field theory
   - Uses shared field parameters (V₀, λ, β)
   - Fits only baryonic parameters per galaxy

4. **Connected to cosmology** (`cosmology/background_evolution.py`)
   - Added `get_phi_0()` method to get today's field value
   - Uses same (V₀, λ) as cosmology for consistency

### ⚠️ Current Issue

**Problem**: Field solution produces very low, constant density
- Field-driven: ρ_c0 ≈ 1.37e-3 M_sun/kpc³ (constant)
- Phenomenological: ρ_c0 ≈ 8.16e6 M_sun/kpc³
- Ratio: ~6 million times too low!

**Root Cause**: Field isn't responding to baryons
- φ(r) is nearly constant (minimal gradient)
- Density is dominated by V(φ) which is constant
- Coupling term (β * ρ_b / ρ_crit) may be too weak or incorrectly applied

**Evidence**:
- Field solution density is flat (min = max = median)
- Chi² is 34× worse than phenomenological fit
- R_c hits upper bound (50 kpc) because density is too low

---

## Next Steps to Fix

### 1. Debug Coupling Term
- [ ] Verify coupling term (β * ρ_b / ρ_crit) is correct
- [ ] Check if coupling is actually affecting the field solution
- [ ] Print Veff values to see if coupling term dominates

### 2. Test Stronger Coupling
- [ ] Try β = 10, 100, 1000 (very strong coupling)
- [ ] See if field responds more to baryons
- [ ] Check if density increases

### 3. Check Field Solution
- [ ] Verify φ(r) actually varies with radius
- [ ] Check if dφ/dr is non-zero
- [ ] Plot φ(r) and ρ_φ(r) to visualize

### 4. Alternative: Different Potential
- [ ] Try different V₀ values (larger?)
- [ ] Try different λ values
- [ ] May need to adjust potential form

### 5. Alternative: Different Approach
- [ ] Consider if we need a different coupling form
- [ ] May need A(φ) = e^(βφ) with different normalization
- [ ] Check literature on scalar-tensor galaxy halos

---

## Test Results (CamB)

**Field Parameters**: V₀ = 1e-6, λ = 1.0, β = 1.0

**Field-Driven Fit**:
- M_disk = 2.81e8 M_sun
- R_disk = 1.08 kpc
- ρ_c0 = 1.37e-3 M_sun/kpc³ (constant, too low)
- R_c = 50.0 kpc (hit upper bound)
- χ²_red = 0.924

**Phenomenological Fit**:
- M_disk = 9.82e6 M_sun
- R_disk = 1.08 kpc
- ρ_c0 = 8.16e6 M_sun/kpc³
- R_c = 3.08 kpc
- χ²_red = 0.027

**Gap**: Field-driven density is ~6 million times too low

---

## Hypothesis

The field solution is dominated by the cosmological V(φ) term, and the coupling to baryons is too weak to create a significant halo. Possible fixes:

1. **Stronger coupling**: β >> 1 (but may break solar system tests)
2. **Different potential**: Larger V₀ or different form
3. **Different coupling form**: A(φ) = e^(βφ/M_Pl) with proper normalization
4. **Screening**: May need chameleon mechanism to create density-dependent response

---

## Files

- `galaxies/halo_field_profile.py` - Solver (needs tuning)
- `galaxies/fit_field_driven.py` - Fitter (working, needs better field solution)
- `examples/test_field_driven_galaxy.py` - Test script
- `galaxies/debug_units.py` - Unit debugging

---

**Status**: Solver works, but field solution needs refinement  
**Next**: Debug coupling term, test stronger β, check field response  
**Priority**: HIGH (this is the core of Track 2)

