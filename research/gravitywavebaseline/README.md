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

# Optional: Analytic backbone + λ_gw perturbations
python backbone_analysis.py
```

### Analytic Backbone + λ_gw Perturbations

The file `backbone_analysis.py` implements the two-stage approach described in the λ_gw reference docs:

1. **Analytic backbone** – Miyamoto–Nagai disk + Hernquist bulge (+ optional NFW halo) carries the full Milky Way mass and reproduces ≈220 km/s without any multipliers.
2. **Stellar perturbation** – a sampled subset of Gaia stars supplies the λ_gw-dependent enhancement so multipliers act as coherent perturbations, not substitutes for missing mass.

Key scripts/documentation:

- `backbone_analysis.py` – runs the analytic backbone optimization and writes `backbone_analysis_results.json`.
- `LAMBDA_GW_PHYSICS.md` – deep dive on the gravitational-wave wavelength hypothesis and expected parameter ranges.
- `LAMBDA_GW_QUICK_REF.md` – one-page checklist for verifying short-wavelength boosters and λ_gw usage.

Run it via:

```bash
python gravitywavebaseline/backbone_analysis.py
```

Look for low RMS (<30 km/s) in the “NO HALO” config to demonstrate that λ_gw multipliers can replace dark matter, and compare with the “WITH HALO” config to see how the perturbation behaves on top of ΛCDM.

### GR Baseline vs λ_gw Enhancement (Corrected Workflow)

To avoid the earlier “fit-the-answer” baseline, the Σ-Gravity test now runs in two explicit phases using observed baryonic masses only:

1. **GR baseline (no dark matter, no multipliers)**  
   ```bash
   python gravitywavebaseline/calculate_gr_baseline.py
   ```
   - Loads `gaia_with_periods.parquet`, applies a Miyamoto–Nagai disk (4×10¹⁰ Msun) + Hernquist bulge (1.5×10¹⁰ Msun)
   - Writes `gaia_with_gr_baseline.parquet` and `gr_baseline_plot.png`
   - Reports the per-ring gap `v_obs - v_GR`; outer disk currently shows ⟨gap⟩ ≈ 104 km/s, RMS ≈ 109 km/s

2. **λ_gw enhancement on top of the fixed GR baseline**  
   ```bash
   python gravitywavebaseline/test_lambda_enhancement.py \
       --r-min 12 --r-max 16 \
       --n-obs 1000 \
       --stellar-scale 10 \
       --disk-mass 4e10 \
       --capacity-model surface_density \
       --capacity-alpha 0.5
   ```
   - Samples 50 k Gaia stars, rescales them so the sampled mass represents the 4×10¹⁰ Msun disk (× user multiplier)
   - Optional `--capacity-model` applies spillover constraints (`surface_density`, `velocity_dispersion`, `flatness`, `wavelength`) with geometry (`disk`/`sphere`) and strength (`--capacity-alpha`), limiting how much enhancement each shell can hold before spilling outward
   - `--force-shell-match` scales each observation after spillover so `v_total` exactly equals `v_observed` (the “infinite boost per sphere” thought experiment)
   - Runs the short-λ booster, saturating booster, and constant multiplier on the outer disk observations after the spillover step
   - Outputs `lambda_enhancement_results.json` with baseline vs λ_gw RMS, best-fit parameters, and improvement percentages

3. **Capacity-law solver**  
   ```bash
   python gravitywavebaseline/fit_capacity_profile.py \
       --r-min 12 --r-max 16 \
       --n-obs 1000 \
       --stellar-scale 5 \
       --disk-mass 4e10 \
       --capacity-model surface_density
   ```
   - Computes the required per-star enhancement `sqrt(v_obs^2 - v_GR^2)` and fits a growth law `capacity ∝ Σ(r) × (r/r₀)^γ` whose spillover reproduces the data
   - Writes `capacity_profile_fit.json` with the best-fit `(alpha, gamma)`, the residual RMS, and the per-shell capacity table for downstream modeling

4. **SPARC validation (per-galaxy)**  
   ```bash
   python gravitywavebaseline/sparc_capacity_test.py \
       --galaxy NGC2403 \
       --alpha 2.8271 \
       --gamma 0.8579
   ```
   - Reads `data/Rotmod_LTG/<galaxy>_rotmod.dat`, builds the baryonic GR baseline from the SPARC-provided `Vgas`, `Vdisk`, `Vbulge`
   - Applies the Milky Way capacity law (or any `(alpha, gamma)` you supply) to the SPARC radii and reports the resulting RMS error
   - Outputs `gravitywavebaseline/sparc_results/<galaxy>_capacity_test.json` for record-keeping inside this folder

5. **Batch SPARC sweep**  
   ```bash
   python gravitywavebaseline/run_sparc_capacity_batch.py \
       --alpha 2.8271 --gamma 0.8579 \
       --force-match \
       --results-file gravitywavebaseline/sparc_results_batch.csv
   ```
   - Loops through every entry in `data/sparc/sparc_combined.csv`, runs `sparc_capacity_test.py` per galaxy, and aggregates the RMS results
   - Keeps all outputs under `gravitywavebaseline/sparc_results/` so nothing touches the core code/paper

Latest run (12–16 kpc, stellar_scale=10, disk_mass=4×10¹⁰ Msun):
- GR-only RMS: 113 km/s
- Best λ_gw RMS: 52.5 km/s
- Improvement: 60.4 km/s (53.5 %) with a simple constant multiplier, the short-λ forms land at the same RMS

The detailed rationale, parameters, and interpretation are captured in:
- `CORRECTED_WORKFLOW.md` — step-by-step rationale for the corrected baseline
- `QUICK_REFERENCE.md` — one-page checklist for rerunning the test
- `EXECUTIVE_SUMMARY.md` — narrative explaining why the new workflow matters

### Correcting Gaia v_phi

The supplied 1.8 M Gaia catalogue originally contained a simplified `v_phi = sqrt(v_ra^2 + v_dec^2)` estimate. Run `python gravitywavebaseline/recompute_gaia_velocities.py` to rebuild `data/gaia/gaia_processed_corrected.csv` using Astropy’s full Galactocentric transformation from the raw DR3 sample (`data/gaia/gaia_large_sample_raw.csv`). `calculate_periods.py` automatically prefers this corrected file when present.

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

