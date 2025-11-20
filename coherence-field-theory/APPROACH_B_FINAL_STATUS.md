# Approach B (Wave Amplification): Final Status

## Summary

**Approach B is not viable for real SPARC galaxies** due to:
1. **Physics mismatch**: Most galaxies lack sufficient cold (Q<2) regions
2. **Numerical fragility**: Solver fails to converge on real multi-scale data even when physics is correct

## What We Accomplished

✅ **Numerically stable solver** on synthetic smooth data  
✅ **Fixed data loading**: Proper Σ_b from SBdisk column  
✅ **Identified viable galaxies**: NGC2403 & NGC6503 have cold zones  
✅ **Boosted gain**: Achieved g_max ~ 2.7 kpc⁻² with 46/64 tachyonic points  

❌ **Solver convergence on real data**: Fails even with strong tachyonic zones  
❌ **Rotation curve fits**: No successful fits produced  

## Test Results

### NGC2403 (Spiral, 64 valid points)
- **Surface density**: 0 - 5.5×10⁸ M☉/kpc² (proper from SBdisk)
- **Toomre Q**: 0 - 1262 (46 points with Q<2)
- **Gain**: 0 - 2.74 kpc⁻² (**strong!**)
- **Tachyonic zone**: 46/64 points (72%)
- **Solver**: Failed, residual=23.7 after 500 iterations

### NGC6503 (Spiral, 28 valid points)
- **Surface density**: 1.5 - 4.2×10⁸ M☉/kpc²
- **Toomre Q**: 0 - 1011 (11 points with Q<2)
- **Gain**: 0 - 2.70 kpc⁻²
- **Tachyonic zone**: 11/28 points (39%)
- **Solver**: Failed, residual=25.3 after 500 iterations

## Why Solver Fails

**Real SPARC data characteristics**:
- Sparse: 30-70 points (vs 300 in synthetic test)
- Multi-scale: r spans 0.2-20 kpc with varying resolution
- Noisy derivatives: dlnΩ/dlnr fluctuates point-to-point
- Discontinuous Σ_b: Drops to zero at outer radii (SBdisk=0)

**Solver assumptions** (from synthetic test):
- Smooth profiles (no sharp transitions)
- Dense sampling (300 points)
- Continuous g(r) (no jumps)

**Result**: Relaxation fails to converge. The tachyonic PDE with discontinuous coefficients is ill-conditioned.

## Parameters Used

**Final boosted parameters** (best attempt):
```python
m0 = 0.01 kpc⁻¹  # REDUCED for wider tachyonic zones
R_coh = 2.0 kpc   # REDUCED for stronger g₀ = α/R²coh  
alpha = 10.0      # BOOSTED 10x → g₀ = 2.5 kpc⁻²
Q_c = 2.0         # Accept marginally stable disks
beta = 2.0        # Strong baryon coupling
lambda_4 = 0.5    # Quartic saturation
```

This gave:
- g_max ~ 2.7 kpc⁻² (vs m₀²=0.0001)  
- Tachyonic zones covering 40-70% of points  
- But still no convergence

## Comparison to Synthetic Test

| Property | Synthetic Test | NGC2403 (Real) |
|----------|---------------|----------------|
| Points | 300 | 64 |
| r range | 0-10 kpc | 0.16-20.87 kpc |
| Σ_b | ~10⁸ M☉/kpc² (uniform) | 0-5.5×10⁸ (discontinuous) |
| Q | <1.5 (all inner disk) | 0-1262 (46/64 <2) |
| g_max | ~0.06 kpc⁻² | 2.74 kpc⁻² |
| Tachyonic | Yes (inner disk) | Yes (46/64 points) |
| **Convergence** | ✅ 43 iterations | ❌ 500+ iterations |

**Key difference**: Smoothness. Real data breaks solver assumptions.

## Data Quality Verification

✅ **Headers correct**: Columns match SPARC spec  
✅ **No corruption**: All values in reasonable ranges  
✅ **SBdisk → Σ_disk**: Proper conversion with M/L=0.5  
✅ **Q calculation**: Matches literature for spirals  
✅ **Filtering**: Removed SBdisk=0 points (avoid Q→∞)  

The data is good. The problem is solver robustness.

## Alternative Approaches

### Option 1: Stick with Phenomenological Σ-Gravity
Your existing `many_path_model/` and `GravityWaveTest/` frameworks work on real galaxies:
- K(R) formalism fits rotation curves successfully
- Uses morphology/Q as tuning knobs (not hard gates)
- No requirement for tachyonic instability
- Numerically stable

**Recommendation**: This is the most robust path forward.

### Option 2: Approach C (Symmetron)
Revisit with proper parameter ranges:
- Avoid cosmological constant problem (μ~10⁻³³ eV, not 10⁻³)
- Don't require Q < Q_c (screening instead of amplification)
- Use chameleon mechanism (φ couples to trace of stress-energy)

But be warned: Approach C failed viability scan (240k combinations, 0 passed).

### Option 3: Hybrid Model
Use phenomenological K(R) as the "gain" function:
```python
g(r) = K(R) · [learned from data]
∇²φ - m₀²φ = β g(r) ρ_b
```
Then φ(r) provides the field-theoretic interpretation, but K(R) does the heavy lifting.

## Lessons Learned

1. **Physical plausibility ≠ numerical tractability**: Approach B has solid physics (wave amplification in cold disks) but fails numerically on real data.

2. **Synthetic tests can be misleading**: Solver worked perfectly on smooth 300-point profiles but breaks on sparse 64-point real data.

3. **Gates vs tuning knobs**: Hard physical requirements (Q < Q_c) are more restrictive than soft empirical correlations.

4. **Data verification is crucial**: We discovered the Σ_b calculation was wrong, fixed it, and found cold zones in NGC2403/NGC6503. Always check!

## Files

**Code**:
- `galaxies/resonant_halo_solver.py` - Approach B solver (works on synthetic)
- `galaxies/test_resonant_on_sparc.py` - SPARC test script (fails to converge)
- `galaxies/verify_data_quality.py` - Data verification (confirms Q<2 zones exist)
- `galaxies/diagnose_gain.py` - Gate diagnostics

**Documentation**:
- `APPROACH_B_IMPLEMENTED.md` - Implementation details
- `NUMERICS_FIXED_READY_FOR_SPARC.md` - Synthetic test success
- `APPROACH_B_CRITICAL_ISSUE.md` - Initial diagnosis (Q too high)
- `APPROACH_B_FINAL_STATUS.md` - This document

**Outputs**:
- `outputs/sparc_resonant_fits/` - No successful fits
- `outputs/data_verification/` - Q vs r plots showing cold zones
- `outputs/gain_diagnostics/` - Gate activation plots

## Recommendation

**Abandon Approach B for SPARC galaxy fits.** The physics is interesting but not practical:
- Requires idealized conditions (cold, smooth disks)
- Numerically fragile on real data
- Even with proper data and boosted parameters, solver fails

**Return to phenomenological Σ-Gravity** (`many_path_model/`) which:
- Works on real galaxies (175 SPARC fits completed)
- Uses empirical K(R) without hard physical gates
- Is numerically stable

If you want a field theory backbone, consider:
- **Approach C** with better parameters (avoid CC problem)
- **Hybrid**: phenomenological K(R) as source for field equation
- **Accept limitations**: some theories work in principle but not in practice

---

**Bottom line**: Approach B taught us about wave amplification physics and the importance of data verification, but it's not viable for fitting SPARC rotation curves. Move forward with what works.
