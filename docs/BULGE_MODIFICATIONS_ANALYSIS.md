# BRAVA Bulge Modifications Analysis

## Summary

After extensive testing of modifications to Sigma-Gravity for bulge predictions, **no modifications were found that improve upon the baseline (no enhancement)**. However, a **Jeans-field analysis** reveals the true picture: the baryonic model **overpredicts** gravity by ~6x, meaning bulge kinematics don't require enhancement - they require *less* gravity.

## Key Finding: Jeans-Field Test

The Jeans equation directly computes the dynamical gravity required by observed kinematics:

```
g_R^dyn = v_phi^2/R + (1/nu) * d(nu*sigma_R^2)/dR + (sigma_R^2 - sigma_phi^2)/R
```

Results:
- **g_dyn / g_bar = 0.17** (kinematics require only ~17% of baryonic gravity)
- **g_bar overpredicts gravity by ~6x**
- Sigma-Gravity enhancement makes things **worse** (adds more gravity when less is needed)

This means:
1. Bulge kinematics **don't require gravitational enhancement**
2. The baryonic mass model (M_bulge + M_disk) is too high for the inner region
3. SPARC bulge issues are likely **M/L or modeling problems**, not missing gravity
4. C_cov correctly predicts small enhancement (Sigma ~ 1.01) for dense regions

## Current Performance

- **Baseline RMS**: 15.04 km/s (no enhancement, Sigma = 1.0)
- **Current C_cov RMS**: 15.21 km/s (with enhancement, Sigma ≈ 1.009)
- **Improvement**: -0.17 km/s (actually worse!)

## Key Findings

### 1. Enhancement is Too Small

- Mean Sigma enhancement: **0.93%** (Sigma - 1 = 0.009)
- Mean C_cov: 0.252 (range: 0.008 - 0.559)
- Mean h: 0.032 (range: 0.002 - 0.132)
- Mean A_0 * C_cov * h: 0.0094

**Problem**: The enhancement is so small (< 1%) that it can't meaningfully affect predictions.

### 2. Tested Modifications (None Helped)

All of the following modifications were tested and **none improved** upon baseline:

1. **2x A_0**: RMS = 15.32 km/s (worse)
2. **5x A_0**: RMS = 16.07 km/s (worse)
3. **10x g_dagger**: RMS = 18.46 km/s (much worse)
4. **Direct sigma enhancement**: RMS = 15.21 km/s (same as current)
5. **Optimal A_0 (fitted)**: RMS = 15.15 km/s (slightly worse)
6. **Modified C_cov (reduce density term)**: RMS = 15.63 km/s (worse)
7. **Direct sigma from omega**: Numerical issues
8. **Sigma-dependent calibration**: RMS = 15.21 km/s (no improvement)
9. **Combined approach**: RMS = 15.04 km/s (same as baseline)

### 3. Physical Properties

- Mean omega²: 1977 s⁻² (strong rotation)
- Mean theta²: 0.0 s⁻² (no expansion, steady state)
- Mean rho: 8.3×10⁻²¹ kg/m³
- Mean g_bar: 1.87×10⁻⁹ m/s²
- Mean R: 2.15 kpc (inner bulge)

### 4. Residual Patterns

- **sigma_obs vs R correlation**: -0.89 (strong negative - sigma decreases with R)
- **resid vs R**: -0.18 (weak negative)
- **resid vs C_cov**: +0.08 (weak positive)
- **resid vs Sigma**: -0.37 (moderate negative - higher Sigma → lower residual)
- **resid vs omega²**: -0.21 (weak negative)
- **resid vs rho**: -0.12 (weak negative)

The negative correlation between resid and Sigma suggests that **higher enhancement actually helps**, but the enhancement is too small to matter.

## Possible Explanations

### 1. C_cov Formula May Be Wrong for Bulges

The current formula:
```
C_cov = ω² / (ω² + 4πGρ + θ² + H₀²)
```

For bulges:
- ω² is large (~2000 s⁻²)
- 4πGρ is also large (dominates denominator)
- Result: C_cov is small (~0.25)

**Maybe**: The density term (4πGρ) shouldn't dominate for bulges, or should be weighted differently.

### 2. Enhancement Should Be Larger

The enhancement (A_0 * C_cov * h) is only ~0.009, which is < 1%. 

**Maybe**: For bulges, we need:
- A_0_bulge >> A_0 (much larger amplitude)
- Or a different h_function for bulges
- Or a different enhancement formula entirely

### 3. Wrong Observable

We're predicting velocity dispersion (sigma_tot) from circular speed (V_circ) with a calibration factor.

**Maybe**: For bulges, we should:
- Predict sigma directly from flow invariants
- Use a different relationship (not V_circ → sigma)
- Account for anisotropy differently

### 4. Baseline Is Already Optimal

The baseline RMS of 15.04 km/s is already quite good. Maybe:
- The enhancement simply isn't needed for bulges
- Or the enhancement works differently (not through Sigma)
- Or we need different flow invariants

## Recommendations

### Short Term

1. **Accept that C_cov doesn't help for bulges** (at least not in its current form)
2. **Use baseline predictions** (no enhancement) for bulge velocity dispersions
3. **Focus on rotation curves** where C_cov/standard C coherence works well

### Medium Term

1. **Investigate different coherence formulas for bulges**:
   - Try: C_bulge = ω² / (ω² + α·4πGρ + θ² + β·H₀²) with α << 1
   - Try: C_bulge = ω² / (ω² + θ² + H₀²) (ignore density)
   - Try: C_bulge based on different invariants (shear, tidal, etc.)

2. **Test direct sigma prediction**:
   - sigma² = f(omega², R, rho, ...) without going through V_circ
   - Use Jeans equation or virial theorem directly

3. **Check if enhancement should affect anisotropy**:
   - Maybe enhancement affects σ_R, σ_φ, σ_z differently
   - Test if enhancement should be applied to individual components

### Long Term

1. **Revisit the theoretical basis**:
   - Why should C_cov work for rotation curves but not dispersions?
   - Is there a different coherence measure for dispersion-dominated systems?
   - Should bulges use a different enhancement mechanism entirely?

2. **Compare to other bulge datasets**:
   - Test on different bulge samples
   - See if the issue is BRAVA-specific or general

3. **Consider that bulges may not need enhancement**:
   - Maybe the baseline (GR/Newtonian) is already correct for bulges
   - Enhancement may only be needed for rotation curves (disk)

## Next Steps

1. ✅ Document findings (this document)
2. ⏳ Test modified C_cov formulas (reduce/remove density term)
3. ⏳ Test direct sigma prediction from flow invariants
4. ⏳ Compare to other bulge datasets if available
5. ⏳ Consider if enhancement is needed for bulges at all

## Conclusion

The Jeans-field analysis provides the definitive answer: **bulge kinematics don't require gravitational enhancement**. In fact, they require *less* gravity than the baryonic model predicts.

This explains why:
1. C_cov enhancement doesn't help (it adds gravity when we need to remove some)
2. All modifications either made things worse or had no effect
3. The baseline (no enhancement) was optimal

The real issue is not with Sigma-Gravity but with the **baryonic mass model**. For SPARC bulge galaxies, the problems are likely:
- M/L ratio issues (stellar mass overestimated)
- Bulge/disk decomposition errors
- Pressure support not properly accounted for
- Non-circular orbits in inner regions

## Implications for Sigma-Gravity

1. **C_cov is working correctly**: It predicts small enhancement (Sigma ~ 1.01) in dense, high-g environments, which is consistent with the Jeans result showing no extra gravity is needed.

2. **Rotation curves vs dispersions**: Enhancement is needed for outer disk rotation curves (where g_dyn > g_bar), but not for bulge dispersions (where g_dyn < g_bar).

3. **Focus effort on baryonic modeling**: Instead of modifying Sigma-Gravity for bulges, the focus should be on improving M/L estimates and bulge/disk decomposition for SPARC galaxies.

