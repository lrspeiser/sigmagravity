# Time-Coherence Σ-Gravity

Testing a first-principles approach to Σ-Gravity based on **coherence time** τ_coh(R).

## Core Concept

Enhancement is controlled by coherence time τ_coh(R), set by competition between:

1. **τ_geom**: Geometry-driven dephasing (gravitational time dilation)
2. **τ_noise**: Noise-driven decoherence (velocity dispersion/turbulence)

The coherence length is **ℓ_coh(R) = c · τ_coh(R)**, which feeds the Burr-XII kernel.

## Key Equations

```
1/τ_coh = 1/τ_geom + 1/τ_noise
ℓ_coh = c · τ_coh
K(R) = A_global · C(R / ℓ_coh(R))
```

## Current Results

### SPARC Performance ✅
- **Mean ΔRMS**: +0.113 km/s (vs +5.25 km/s for previous theory kernel)
- **Improved**: 130/175 (74.3%) vs 24% before
- **Median ΔRMS**: -0.561 km/s (negative = improvement)

### Coherence Scales ⚠️
- **MW**: `ℓ_coh ≈ 140 kpc` (target: ~5 kpc)
- **SPARC mean**: `ℓ_coh ≈ 135 kpc`
- **Need tuning**: Scales too large, but concept works!

## Files

### Core Implementation
- `coherence_time_kernel.py`: Core kernel functions
  - `compute_tau_geom()`: Geometric dephasing time
  - `compute_tau_noise()`: Noise decoherence time
  - `compute_tau_coh()`: Combined coherence time
  - `compute_coherence_kernel()`: Full kernel computation

### Test Scripts
- `test_mw_coherence.py`: Test on Milky Way
- `test_sparc_coherence.py`: Test on SPARC galaxies (175 galaxies)
- `test_cluster_coherence.py`: Test on galaxy clusters (lensing)

### Analysis & Fitting
- `analyze_results.py`: Quick summary of test results
- `analyze_full_coherence_scaling.py`: Cross-system scaling analysis
- `fit_time_coherence_hyperparams.py`: Joint MW+SPARC hyperparameter fit

### Documentation
- `COMPARISON_AND_NEXT_STEPS.md`: Detailed comparison and roadmap

## Quick Start

```bash
# 1. Run tests
python time-coherence/test_mw_coherence.py
python time-coherence/test_sparc_coherence.py

# 2. Analyze results
python time-coherence/analyze_results.py
python time-coherence/analyze_full_coherence_scaling.py

# 3. Fit hyperparameters (tune scales)
python time-coherence/fit_time_coherence_hyperparams.py \
    --mw-parquet gravitywavebaseline/gaia_with_gr_baseline.parquet \
    --sparc-rotmod-dir data/Rotmod_LTG \
    --sparc-summary data/sparc/sparc_combined.csv
```

## Next Steps

1. ✅ **Concept validated**: 74% improvement on SPARC
2. 🔄 **Tune scales**: Fit hyperparameters to move `ℓ_coh` from ~140 kpc → ~5-20 kpc
3. 🔄 **Cluster validation**: Test if naturally gives `ℓ_coh ~ 200 kpc` for lensing
4. 🔄 **First-principles story**: Document how τ-based microphysics explains Σ-Gravity

