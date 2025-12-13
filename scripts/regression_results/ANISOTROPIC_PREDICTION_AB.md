# Anisotropic Gravity Prediction A/B Testing

## Overview

This framework tests whether **anisotropic gravity** (κ>0) predicts how gravity impacts objects **better** than the baseline isotropic model (κ=0). Unlike the synthetic stream lensing test which only validates that directional effects *can* occur, this test compares predictions against **observational targets** to determine if anisotropy improves real-world predictions.

## Key Question

**Does anisotropic gravity predict how gravity impacts real objects (light rays, particles, etc.) better than the baseline?**

## Framework Components

### 1. Scoring Module (`score_stream_lensing.py`)

Provides:
- **Scene building**: Create gravitational scenes (density, stream directions, weights)
- **Potential solving**: Solve `∇·[(I + κ w(x) ŝ ŝᵀ) ∇Φ] = 4πGρ` for different κ values
- **Prediction functions**: 
  - `predict_ray_endpoints()`: Light ray deflection predictions
  - `integrate_particle_trajectory()`: Test particle trajectory predictions
- **Scoring**: RMS, χ², mean absolute error, max error
- **A/B comparison**: Direct comparison of baseline vs anisotropic predictions

### 2. Regression Test (`test_anisotropic_prediction_ab()`)

Currently uses synthetic observations but designed to accept real data. The test:
1. Builds a gravitational scene
2. Generates/loads observational targets
3. Predicts with baseline (κ=0) and anisotropic (κ>0) models
4. Scores both against observations
5. Reports which model performs better

**Pass criteria**: Anisotropic model must improve RMS prediction error by at least 5% vs baseline.

## Using Real Observational Data

### Current Status: Synthetic Observations

The test currently uses synthetic observations generated from a known `κ_true=6.0`. This validates the framework but doesn't test against reality.

### How to Add Real Data

Replace the synthetic observation generation in `test_anisotropic_prediction_ab()` with real data:

```python
# Instead of:
Phi_true, _ = solve_potential(scene, kappa=kappa_true)
x0s, x_obs = predict_ray_endpoints(Phi_true, scene, ...)
x_obs = x_obs + noise  # synthetic

# Use:
x_obs = load_real_observations()  # Your data loading function
obs_uncertainty = load_observation_uncertainties()
```

### Suitable Observables

The framework supports any observable where "gravity impact" is measurable:

#### A) Light Deflection (Strong Lensing)
- **Data**: Image positions, deflection angles
- **Prediction**: `predict_ray_endpoints()` gives final ray positions
- **Example**: Galaxy-galaxy lensing image positions

#### B) Weak Lensing Shear Anisotropy
- **Data**: Shear as function of azimuth relative to galaxy major axis
- **Prediction**: Compute shear from potential gradient
- **Example**: Quadrupole amplitude vs observations

#### C) Bullet-like Merger Offsets
- **Data**: Offset between baryon peak and lensing peak
- **Prediction**: Convergence/shear field morphology
- **Example**: Bullet Cluster offset measurements

#### D) Stellar Stream Tracks
- **Data**: Stream track positions (great circle deviations)
- **Prediction**: `integrate_particle_trajectory()` for stream particles
- **Example**: GD-1, Palomar 5 stream tracks

#### E) Test Particle Scattering
- **Data**: Scattering angles, trajectory endpoints
- **Prediction**: `integrate_particle_trajectory()` + `compute_scattering_angle()`
- **Example**: Wide binary separations, high-velocity star trajectories

## Example: Adding Real Weak Lensing Data

```python
def test_weak_lensing_shear_ab() -> TestResult:
    """A/B test using real weak lensing shear anisotropy data."""
    from score_stream_lensing import build_scene, solve_potential, compare_models_ab
    
    # Build scene from galaxy data
    scene = build_scene_from_galaxy(galaxy_id="NGC1234")
    
    # Load real weak lensing observations
    # Format: shear measurements as function of azimuth angle
    obs_data = load_weak_lensing_shear("NGC1234")
    azimuth_angles = obs_data['azimuth']
    gamma_t_obs = obs_data['gamma_t']  # Tangential shear
    gamma_x_obs = obs_data['gamma_x']  # Cross shear
    obs_uncertainty = obs_data['uncertainty']
    
    # Predict shear from potentials
    Phi_baseline, _ = solve_potential(scene, kappa=0.0)
    Phi_aniso, _ = solve_potential(scene, kappa=6.0)
    
    gamma_t_baseline = compute_shear_from_potential(Phi_baseline, azimuth_angles)
    gamma_t_aniso = compute_shear_from_potential(Phi_aniso, azimuth_angles)
    
    # Score both models
    results = compare_models_ab(
        scene=scene,
        kappa_baseline=0.0,
        kappa_aniso=6.0,
        observations=gamma_t_obs,
        obs_uncertainty=obs_uncertainty,
        # ... custom prediction function
    )
    
    return TestResult(...)
```

## Validation Requirements

Before claiming "anisotropic is better," ensure:

1. **κ→0 sanity check**: κ=0 must recover isotropic behavior (unit test)
2. **Resolution convergence**: N=128, 256, 512 should give stable predictions
3. **Holdout evaluation**: Fit κ on training set, evaluate on test set
4. **Complexity penalty**: Require meaningful Δχ² improvement (not tiny overfitting)
5. **Multiple observables**: Test across different types of observations

## Current Test Results

**Synthetic test** (κ_true=6.0):
- Baseline RMS: ~0.15
- Anisotropic RMS: ~0.002
- **RMS improvement: ~98%** ✓
- **Anisotropic better: True** ✓

This validates the framework works correctly, but **does not test against reality**.

## Next Steps

1. **Identify real observational targets**:
   - Weak lensing shear anisotropy catalogs
   - Bullet-like merger offset measurements
   - Stellar stream track data

2. **Build data loading functions**:
   - Convert observations to format expected by scoring framework
   - Handle coordinate transformations, uncertainties

3. **Add custom prediction functions**:
   - Weak lensing: Compute shear from potential
   - Stream tracks: Integrate stream particle trajectories
   - Merger offsets: Compute convergence field

4. **Run A/B tests**:
   - Compare baseline vs anisotropic predictions
   - Report RMS, χ² improvements
   - Determine if anisotropy improves real-world predictions

## Files

- `scripts/score_stream_lensing.py`: Scoring framework
- `scripts/run_regression_experimental.py`: Regression test integration
- `scripts/stream_seeking_anisotropic.py`: Anisotropic solver

## References

- Synthetic test validates directional effects can occur
- A/B test determines if anisotropy improves predictions
- Real data integration required to test against reality

