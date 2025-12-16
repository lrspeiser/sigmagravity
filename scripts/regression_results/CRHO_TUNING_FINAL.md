# CRHO Parameter Tuning - Final Results

## Summary

After extensive parameter tuning (35 combinations tested), **CRHO shows no measurable improvement** over baseline:
- All tested combinations: RMS=17.41, Bulge=28.93, Disk=16.06
- Baseline: RMS=17.42, Bulge=28.93, Disk=16.06

## Key Findings

### 1. Density Proxy Values Are Too Low
- Actual density values: 1e-20 to 7e-20 kg/m³
- Initial threshold: 5e-19 kg/m³ (10-50x too high)
- Even with threshold=1e-21, density_factor caps at 10.0

### 2. CRHO Does Affect C Values
- Test shows C reduction of 16-30% with scale=0.2-0.5
- But regression results don't change
- Suggests the effect is either:
  - Too small to matter
  - Compensated by other factors
  - Applied incorrectly

### 3. Parameter Sweep Results
- Tested: CRHO_SCALE = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
- Tested: rho_threshold = [1e-21, 1e-20, 3e-20, 5e-20, 1e-19, 3e-19, 5e-19, 1e-18, 3e-18]
- **All combinations give identical results**

## Possible Issues

1. **Density proxy is wrong**: `rho_proxy_from_vbar()` uses spherical approximation which may not capture true bulge density
2. **Effect is too small**: Even with 30% C reduction, the impact on Σ and V_pred is negligible
3. **Fixed-point iteration compensates**: When C is reduced, V_pred changes, which changes the density proxy, creating a feedback loop
4. **Wrong density indicator**: Maybe need component-based density (bulge vs disk) instead of total

## Recommendations

1. **Try component-based density**: Compute ρ_bulge and ρ_disk separately
2. **Use acceleration-based indicator**: Instead of density, use g_bar as a proxy for bulge regions
3. **Move CRHO inside iteration**: Compute sigma_rho from V_pred (not V_bar) inside the fixed-point loop
4. **Check if effect is real**: Verify that reducing C actually changes V_pred in the full pipeline

## Current Status

CRHO is **implemented and working** (affects C values), but **not improving predictions**. The density/vorticity approach may need refinement or a different implementation strategy.



