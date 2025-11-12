# Gravitational Wave Baseline Analysis

## Overview

This pipeline tests whether a gravitational enhancement multiplier based on stellar "periods" (characteristic wavelengths) can explain the observed Milky Way rotation curve using 1.8 million Gaia stars.

## Hypothesis

Standard Newtonian gravity from visible matter cannot explain observed stellar velocities. We test if a multiplier function of the form:

```
F_enhanced = F_newtonian × f(λ, r)
```

where:
- `λ` = characteristic wavelength/period of the source star
- `r` = distance between source and observer
- `f(λ, r)` = multiplier function (to be determined)

can reproduce observations.

## Pipeline

### Step 1: Calculate Period Hypotheses

`calculate_periods.py` computes 8 different period length hypotheses for each star:

1. **Orbital**: λ = 2πR (orbital circumference)
2. **Dynamical**: λ = R (characteristic scale)
3. **Jeans**: λ = c_s / √(Gρ) (pressure support scale)
4. **Mass-dependent**: λ = M^0.5 (mass scaling)
5. **Hybrid**: λ = √(M×R) (combined)
6. **GW frequency**: λ = v/f_GW (gravitational wave analog)
7. **Scale height**: λ = h(R) (vertical oscillation)
8. **Toomre**: λ from stability criterion

**Output**: `gaia_with_periods.parquet` (~200-300 MB)

**Runtime**: ~5-10 minutes for 1.8M stars

### Step 2: Inverse Multiplier Optimization

`inverse_multiplier_calculation.py` works *backwards* from observations:

1. For each observation point with known velocity v_obs
2. Try different multiplier functions f(λ, r)
3. Optimize parameters to minimize: Σ(v_model - v_obs)²

Tests multiplier functions:
- Linear: f = 1 + A(λ/r)
- Power law: f = 1 + A(λ/λ₀)^α
- Saturating: f = 1 + A[1 - 1/(1 + (λ/λ₀)^p)]
- Distance-modulated: f = 1 + A λ/(r+λ)
- Inverse square: f = 1 + A λ²/(r²+λ²)
- Exponential: f = 1 + A exp(-r/λ)
- Resonant: f = 1 + A exp(-(r-λ)²/σ²)

**Output**: 
- `inverse_multiplier_results.json` (rankings)
- `inverse_multiplier_results.png` (visualizations)

**Runtime**: ~1-3 hours with GPU (depends on number of combinations tested)

## Hardware Requirements

- **GPU**: NVIDIA RTX 5090 (24GB) via CuPy
- **CPU**: 10-core Intel (used for KDTree, sorting, I/O)
- **RAM**: 16GB+ recommended
- **Disk**: ~1GB free space

## Installation

```bash
# Install dependencies
pip install numpy pandas scipy matplotlib cupy-cuda12x

# For parquet support
pip install pyarrow
```

## Usage

```bash
# Step 1: Calculate periods (5-10 min)
cd gravitywavebaseline
python calculate_periods.py

# Step 2: Optimize multipliers (1-3 hours)
python inverse_multiplier_calculation.py
```

## Computational Strategy

For 1.8M stars, full N×N gravity calculation = 3.2 trillion pairs!

We use:
1. **Stratified sampling**: Representative subset of source stars by radius
2. **GPU batching**: Process observation points in batches on GPU
3. **Statistical methods**: KDTree for local density, sorting for enclosed mass
4. **Limited optimization**: ~20 iterations × 6 population (expensive function calls)

Trade-off: Speed vs accuracy. Increase `n_source_samples` in code if you want higher accuracy at cost of runtime.

## Expected Results

If a multiplier exists that can reproduce the observed rotation curve:
- **Good fit**: RMS < 20 km/s
- **Acceptable**: RMS < 50 km/s
- **Poor**: RMS > 50 km/s

The best-fit multiplier function and parameters tell us:
1. Which characteristic wavelength matters (orbital? dynamical? Jeans?)
2. How gravity enhancement scales with period and distance
3. Whether this is consistent with a gravitational wave interpretation

## Output Interpretation

**inverse_multiplier_results.json** contains ranked results:
```json
{
  "period_name": "orbital",
  "multiplier_func": "power_law",
  "params": [1.5, 10.2, 0.8],
  "rms": 15.3,
  "chi_squared": 234567.8
}
```

Lower RMS = better fit to observations.

## Next Steps

If a good multiplier is found:
1. Test on other galaxies (rotation curves, clusters)
2. Check if parameters are universal or galaxy-dependent
3. Derive theoretical foundation for the multiplier function
4. Make predictions for other observables

## Notes

- First run will be slower (JIT compilation for GPU kernels)
- Increase sampling for convergence tests (slower but more accurate)
- Can parallelize across multiple GPUs if available (modify code)
- Results are stochastic (sampling-based) - run multiple times for robustness

