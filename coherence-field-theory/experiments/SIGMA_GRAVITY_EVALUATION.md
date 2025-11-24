# Vacuum-Hydrodynamic Σ-Gravity: Evaluation Summary

## Implementation Status

✅ **Test script created**: `test_sigma_gravity_first_principles.py`

The script implements the core Σ-gravity mapping and integrates with your existing SPARC data infrastructure.

## What Was Implemented

### Core Functionality

1. **Σ-gravity force rescaling**:
   ```python
   g_eff = g_bar * [ 1 + alpha_vac * I_geo * (1 - exp(-(R/L_grad)^p)) ]
   ```

2. **State variable computation**:
   - `I_geo = 3σ²/(v_bar² + 3σ²)` - pressure-support fraction
   - `L_grad = |Φ_bar / g_bar|` - field flatness scale
   - `Φ_bar` computed via stable outward integration from `g_bar`

3. **Integration with existing codebase**:
   - Uses `EnhancedSPARCFitter.load_galaxy()` for SPARC data
   - Compatible with existing rotation curve utilities
   - RMS error comparison vs. baryons-only baseline

4. **Command-line interface**:
   - Test individual or multiple galaxies
   - Adjustable parameters (alpha_vac, sigma, p)
   - CSV output for batch analysis

## Current Limitations (As Documented)

1. **Single σ per galaxy**: Currently uses a fixed `sigma_kms` parameter. Should be replaced with `sigma_v(R)` from:
   - SPARC SBdisk + epicyclic frequency κ(R) via Toomre Q
   - Your `EnvironmentEstimator` class (already exists in codebase)

2. **1D potential integration**: Uses simple 1D integration. For disk geometry, should use axisymmetric machinery (Bessel K_0 convolution).

3. **No Lagrangian/action**: Force rescaling only. Lensing and PPN need metric/effective stress-energy mapping.

4. **Fixed p=0.75**: Should test p ∈ [0.6, 1.0] in model-selection run.

5. **No cosmology/PPN validation**: Should route through existing PPN and background-cosmology harnesses.

## Next Steps for Full Evaluation

### Immediate (Easy Wins)

1. **Integrate EnvironmentEstimator**:
   - Replace fixed `sigma_kms` with `sigma_v(R)` from `EnvironmentEstimator.estimate_from_sparc()`
   - Use morphology classification for better priors
   - This will make `I_geo` radius-dependent and more realistic

2. **Batch testing**:
   - Run on 20-40 galaxy hold-out set
   - Compare RMS improvement rates vs. GPM baseline
   - Target: ≥70% improvement rate, ≥35-40% median RMS reduction

### Medium-Term (Requires More Work)

3. **Axisymmetric potential**:
   - Replace 1D integration with disk convolution (Bessel K_0)
   - Use existing axisymmetric machinery if available
   - This should reduce scatter driven by geometry

4. **Model selection on p**:
   - Test p ∈ [0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
   - One-time run to find optimal value
   - Document as fixed theoretical constant

5. **Ellipticals vs spirals test**:
   - With measured σ_v, verify ellipticals show larger I_geo
   - This is the "smoking gun" prediction

### Long-Term (Theoretical Development)

6. **Effective Lagrangian**:
   - Derive action whose weak-field limit reproduces the mapping
   - Ensures energy-momentum conservation
   - Defines lensing/PPN unambiguously

7. **PPN & cosmology validation**:
   - Route through `multiscale/ppn_safety.py`
   - Verify Solar System safety (high gradients → small L_grad → enhancement ≈ 0)
   - Check FRW decoupling with `cosmology/background_evolution.py`

8. **Lensing mapping**:
   - Decide how force rescaling maps to lensing potential (Φ+Ψ)
   - Conservative: treat as effective density → cluster lensing module
   - Compare with observed κ(R)

## How to Run Initial Tests

```bash
cd coherence-field-theory/experiments

# Basic test on a few galaxies
python test_sigma_gravity_first_principles.py --galaxies DDO154,NGC2403,NGC3198

# Test with different parameters
python test_sigma_gravity_first_principles.py --galaxies DDO154 --alpha_vac 5.0 --sigma 25.0

# Save results for analysis
python test_sigma_gravity_first_principles.py --galaxies DDO154,NGC2403,NGC3198 --output initial_results.csv
```

## Expected Behavior

### Solar System
- High gradients → small L_grad
- Cold/Keplerian → I_geo ≪ 1
- **Expected**: enhancement ≈ 0 (GR recovered)

### Cold, Thin Disks
- Small I_geo but finite
- L_grad grows with radius
- **Expected**: modest outer-disk boost

### Ellipticals/Clusters
- σ_v dominates v_circ → I_geo → 1
- Large-scale flat potentials → L_grad large
- **Expected**: strong enhancement

## Comparison with Existing Models

| Feature | GPM (existing) | Σ-gravity (new) |
|---------|---------------|------------------|
| **Parameters** | 3-4 tuned per galaxy | 1 global constant |
| **State variables** | Environment estimators | Field flatness + pressure support |
| **Predictions** | Empirical gates | Morphology-dependent (ellipticals > spirals) |
| **Parsimony** | Moderate | High (single α_vac) |

## Files Created

- `test_sigma_gravity_first_principles.py`: Main test script
- `README_SIGMA_GRAVITY.md`: Detailed documentation
- `SIGMA_GRAVITY_EVALUATION.md`: This summary

## Assessment

**Verdict**: The concept is **worth testing**. It's not a rehash of MOND (no universal a₀; disk-aligned & pressure-aware) and not particle DM (response is baryon-conditioned). It is a compact closure that can sit on top of your existing data plumbing.

**Key strengths**:
- Parsimony (single global constant)
- State variables, not fits
- Clear, falsifiable predictions

**Key gaps** (all fixable):
- No action/Lagrangian yet
- L_grad numerically delicate
- I_geo needs measured kinematics
- p=3/4 is a hypothesis
- Cosmology & Solar System need validation

The implementation provides a solid foundation for testing. The next step is to integrate with `EnvironmentEstimator` for realistic σ_v(R) profiles and run on a larger galaxy sample.

