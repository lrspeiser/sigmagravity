# Anisotropic Gravity Testing Against Real Observational Data

## Overview

This document describes the framework for testing anisotropic gravity (κ>0) against real directional observables, not just synthetic tests. The goal is to determine whether anisotropic gravity **predicts object impact better** than the baseline isotropic model (κ=0).

## Key Insight

The synthetic test (`test_synthetic_stream_lensing`) answers:
- ✅ **"Can the anisotropic operator produce directional focusing that scalar rescaling cannot mimic?"**

But it does NOT answer:
- ❌ **"Does it match reality better?"**

To answer the second question, we need **A/B prediction scoring** against real observational targets.

## Framework

### 1. Define Observable "Impact"

Pick directional observables where gravity's impact is directly measurable:

**For Light:**
- Deflection/image positions (strong lensing)
- Tangential shear vs azimuth (weak lensing quadrupole)
- Peak offsets in mergers (Bullet-like systems)

**For Massive Objects:**
- Trajectory/scattering angle of test particles
- Orbit precession or stream track in known potential

### 2. A/B Test Structure

For the same input scene (ρ, ŝ, w), compute predictions with:
- **Baseline:** κ = 0 (isotropic)
- **Stream-seeking:** κ > 0 (anisotropic)

Then compare predictions to observational target (y_obs):
- RMS error
- χ² / log-likelihood (if uncertainties available)
- Holdout evaluation (fit κ on one set, evaluate on another)

### 3. Implementation

#### A. Bullet Cluster Offset Test

**File:** `scripts/test_bullet_anisotropic.py`

**Observable:** Lensing peak offset from gas peak (~150 kpc)

**Method:**
1. Build 2D scene from Bullet Cluster data:
   - Gas mass: 1.5×10¹⁴ M☉ at x=-50 kpc
   - Stellar mass: 0.3×10¹⁴ M☉ at x=-200 kpc
   - Stream direction: from gas toward stars (collision direction)
2. Solve anisotropic Poisson for κ=0 and κ>0
3. Find lensing peak from potential field
4. Compare predicted offset vs observed offset

**Status:** Implemented, but currently shows no difference between κ=0 and κ>0. Needs debugging.

**Integration:** Added to regression suite as `test_bullet_anisotropic_ab()`

#### B. Synthetic Ray-Tracing Test

**File:** `scripts/score_stream_lensing.py`

**Observable:** Ray endpoint positions

**Method:**
1. Build synthetic scene with mass blob and stream
2. Generate "observations" from known κ_true
3. Predict with κ=0 and κ>0
4. Compare RMS error

**Status:** Implemented, uses synthetic data (can be replaced with real observations)

#### C. Particle Trajectory Test

**File:** `scripts/test_bullet_anisotropic.py` (has `integrate_particle` function)

**Observable:** Scattering angle, final position, closest approach

**Method:**
1. Solve potential for κ=0 and κ>0
2. Integrate particle trajectory: `r̈ = -∇Φ(r)`
3. Compare final state to observations

**Status:** Framework exists, needs integration with real data

## Real-World Tests Needed

### 1. Weak Lensing Shear Anisotropy

**Data:** Shear as function of azimuth relative to galaxy major axis

**Test:** Compare predicted shear pattern γ_t(θ), γ_x(θ) to observations

**Metric:** χ² of quadrupole amplitude vs observations

### 2. Bullet-Like Merger Offsets

**Data:** Offset between baryon peak and lensing peak

**Test:** Compare predicted offset to observed offset

**Metric:** RMS error in offset prediction

**Status:** Partially implemented (needs debugging)

### 3. Stellar Stream Track Bending

**Data:** Measured stream track (great circle deviations)

**Test:** Compare predicted stream track and precession to observations

**Metric:** RMS deviation from observed track

## Reality Checks

To claim "better," ensure:

1. **κ=0 sanity:** κ→0 must recover isotropic behavior (unit test)
2. **Resolution convergence:** N=128, 256, 512 should give stable predictions
3. **Holdout evaluation:** Fit κ on one set, evaluate on another
4. **Complexity penalty:** Require meaningful Δχ² improvement, not tiny overfitting

## Current Status

- ✅ Framework implemented
- ✅ Bullet Cluster test structure created
- ⚠️ Bullet Cluster test shows no difference (needs debugging)
- ⚠️ Need to integrate real observational data
- ⚠️ Need to test with different κ values and stream configurations

## Next Steps

1. **Debug Bullet Cluster test:** Why doesn't κ>0 create a different offset?
   - Check if stream direction/weight is being used correctly
   - Verify anisotropy is actually affecting the solution
   - Test with different κ values

2. **Add real weak lensing data:** Test against actual shear measurements

3. **Add stellar stream data:** Test against measured stream tracks

4. **Systematic κ tuning:** Find optimal κ for each observable type

## Files

- `scripts/test_bullet_anisotropic.py` - Bullet Cluster A/B test
- `scripts/score_stream_lensing.py` - General ray-tracing A/B framework
- `scripts/stream_seeking_anisotropic.py` - Anisotropic Poisson solver
- `scripts/run_regression_experimental.py` - Integration into regression suite

