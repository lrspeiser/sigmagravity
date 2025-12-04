# Vacuum-Hydrodynamic Σ-Gravity: First-Principles Test

## Overview

This directory contains an experimental implementation of the **vacuum-hydrodynamic Σ-gravity** concept, a parsimonious modification to gravity that rescales the force based on field flatness and pressure support rather than introducing multiple tuned parameters.

## The Concept

The Σ-gravity mapping modifies the gravitational acceleration as:

```
g_eff = g_bar * [ 1 + alpha_vac * I_geo * (1 - exp(-(R/L_grad)^p)) ]
```

where:

- **I_geo** = 3σ²/(v_bar² + 3σ²) is the pressure-support fraction (state variable)
- **L_grad** = |Φ_bar / ∇Φ_bar| = |Φ_bar / g_bar| is the field flatness scale (state variable)
- **alpha_vac** is a single global vacuum coupling constant (free parameter)
- **p** = 0.75 is the coherence exponent (theoretical constant)

### Key Features

1. **Parsimony**: Collapses 3-4 tuned parameters into a single global constant (`alpha_vac`)
2. **State variables, not fits**: The amplitude and scale depend on *measurable* properties (field flatness and pressure support) rather than free parameters
3. **Clear predictions**: Hot/pressure-supported systems (ellipticals, clusters) should show larger enhancement than cold, rotation-supported disks of the same mass

## Implementation Status

### ✅ Implemented

- Basic Σ-gravity force rescaling
- Integration with existing SPARC data loader (`EnhancedSPARCFitter`)
- Potential integration from acceleration
- RMS error comparison vs. baryons-only baseline
- Command-line interface for testing

### ⚠️ Known Gaps

1. **No action/Lagrangian yet**: Lensing & energy accounting undefined
   - The formula rescales the *force*; for light-deflection and PPN you still need a metric (or effective stress-energy) consistent with that rescaling

2. **L_grad is numerically delicate**: 
   - Near the center (|∇Φ| → 0) and SPARC radii can start very small
   - Current implementation uses smoothing and regularization
   - May need axisymmetric treatment for disk geometry

3. **I_geo needs measured kinematics**:
   - Currently uses a single fiducial `sigma` (velocity dispersion) per galaxy
   - Should be replaced with `sigma_v(R)` estimated from SPARC SBdisk + epicyclic frequency κ(R) via Toomre Q
   - Add priors/caps (e.g., σ_v ≤ 50 km/s) to avoid catastrophic outliers

4. **The p=3/4 exponent is a hypothesis**:
   - Motivated by anomalous diffusion/fractal transport
   - Should be documented as a *fixed theoretical constant* and test alternatives (p ∈ [0.6, 1.0]) in a one-time model-selection run

5. **Cosmology & Solar System**:
   - Should demonstrate that the mapping keeps H(z) and PPN within bounds
   - High gradients → small L_grad, cold/Keplerian → I_geo ≪ 1, so enhancement ≈ 0 (expected)
   - Must verify with existing PPN harness

## Usage

### Basic Test

```bash
cd coherence-field-theory/experiments
python test_sigma_gravity_first_principles.py --galaxies DDO154,NGC2403,NGC3198
```

### Custom Parameters

```bash
# Test with different vacuum coupling
python test_sigma_gravity_first_principles.py --galaxies DDO154 --alpha_vac 5.0

# Test with different velocity dispersion
python test_sigma_gravity_first_principles.py --galaxies NGC2403 --sigma 25.0

# Test with different coherence exponent
python test_sigma_gravity_first_principles.py --galaxies NGC3198 --p 0.8

# Save results to CSV
python test_sigma_gravity_first_principles.py --galaxies DDO154,NGC2403 --output results.csv
```

### Command-Line Options

- `--galaxies`: Comma-separated list of galaxy names (default: DDO154,NGC2403,NGC3198)
- `--alpha_vac`: Global vacuum coupling constant (default: 4.6)
- `--sigma`: Velocity dispersion in km/s (default: 20.0)
- `--p`: Coherence exponent (default: 0.75)
- `--output`: Optional CSV file to save results

## Expected Results

### "Good" Performance Criteria

1. **Hold-out quality**: On a 20-40 galaxy hold-out, aim for:
   - ≥70% RMS improvement rate
   - ≥35-40% median RMS reduction
   - Comparable to best GPM runs when quality-filtered

2. **Ellipticals vs spirals (smoking gun)**:
   - With measured σ_v, ellipticals (pressure supported) should show larger I_geo and stronger enhancement than spirals at the same mass

3. **Axisymmetry**: 
   - If scatter is driven by geometry, swap 1D potential integral for axisymmetric machinery (Bessel K_0 disk convolution)

4. **PPN & cosmology sanity**:
   - Feed effective potential into PPN and background-cosmology harnesses
   - Verify Solar-System safety and FRW decoupling

5. **Lensing**:
   - Decide how force rescaling maps to lensing potential (Φ+Ψ)
   - Conservative choice: treat enhancement as *effective density* and pass through cluster lensing module

## Comparison with Other Models

| Situation | GR (baseline) | Σ-gravity mapping |
|-----------|---------------|-------------------|
| **Solar System** | Metric from baryonic mass; PPN γ=β=1 | High gradients → small L_grad, cold/Keplerian → I_geo ≪ 1 ⇒ enhancement ≈ 0 |
| **Cold, thin disks** | Declining v(r) without halo | Small I_geo but finite; if L_grad grows with radius, factor tends to 1, giving modest *outer-disk* boost |
| **Ellipticals / clusters** | Need DM halo to match dynamics/lensing | σ_v dominates v_circ → I_geo → 1; large-scale flat potentials → L_grad large ⇒ strong enhancement |

## Next Steps

1. **Replace crude σ with σ_v(R)**:
   - Estimate from SPARC SBdisk + epicyclic κ(R) via Toomre Q
   - Use existing `EnvironmentEstimator` if available

2. **Write effective Lagrangian**:
   - Derive weak-field limit that reproduces the mapping
   - Ensures energy-momentum conservation

3. **Route through existing pipelines**:
   - Cosmology: background evolution & H(z) checks
   - Lensing: cluster module for κ(R) comparison
   - PPN: Solar System safety verification

4. **Model selection on p**:
   - Test p ∈ [0.6, 1.0] in one-time run
   - Document as fixed theoretical constant

## Files

- `test_sigma_gravity_first_principles.py`: Main test script
- `README_SIGMA_GRAVITY.md`: This file

## References

This implementation is based on the assessment provided in the user query, which evaluates the concept's strengths, gaps, and practical test design.




