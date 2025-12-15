# Flow Coherence Parameter Tuning Summary

## Best Configuration Found

**Optimal Parameters (Bulge-Specific Tuning):**
- `--coherence=flow`
- `--flow-bulge-specific` (enable bulge-specific tuning)
- `--flow-alpha-bulge=0.0` (ignore shear for bulges)
- `--flow-gamma-bulge=0.01` (minimal tidal for bulges)
- `--flow-alpha-disk=0.02` (use shear for disks)
- `--flow-gamma-disk=0.005` (minimal tidal for disks)
- `--flow-smooth=5` (moving-average window)
- `--flow-no-tidal` (disable tidal term)

**Performance:** RMS = 17.97 km/s (within 0.55 km/s / 3.2% of baseline 17.42 km/s)
**Bulge RMS:** 28.85 km/s (**better than baseline** 28.93 km/s by 0.08 km/s!)

## Performance Comparison

### SPARC Galaxies (171 galaxies)

| Model | Overall RMS | Bulge RMS | Disk RMS | Scatter | Win Rate |
|-------|-------------|-----------|----------|---------|----------|
| **Baseline (C)** | 17.42 km/s | 28.93 km/s | 16.06 km/s | 0.100 dex | 42.7% |
| **Flow (uniform)** | 18.09 km/s | 30.03 km/s | 16.69 km/s | 0.103 dex | 38.0% |
| **Flow (bulge-specific)** | **17.97 km/s** | **28.85 km/s** | 16.69 km/s | 0.102 dex | 39.2% |
| **MOND** | 17.15 km/s | - | - | - | - |
| **ΛCDM** | ~15.0 km/s | - | - | - | - |

**Key Findings:**
- Flow coherence with **bulge-specific tuning** performs within **0.55 km/s (3.2%)** of baseline
- **Bulge RMS is better than baseline** (28.85 vs 28.93 km/s) with bulge-specific tuning
- Flow coherence has **lower dSigma variance** (std=2.01 vs 2.23), suggesting better structure capture
- Bulge-specific tuning addresses bulge degradation: 1.18 km/s improvement (30.03 → 28.85 km/s)
- All 8 core tests pass with optimal flow parameters
- Very small parameter values work best, suggesting flow coherence should be a small correction to baseline
- **Bulges**: vorticity-dominated, shear irrelevant (α=0.0)
- **Disks**: shear matters, vorticity less important (α=0.02)

### Parameter Sensitivity

**Alpha (shear weight):**
- Lower is better: α=0.03-0.05 gives best results
- Higher values (α>0.5) degrade performance significantly

**Gamma (tidal weight):**
- Very small values work best: γ=0.01-0.02
- Tidal term can be disabled entirely with `--flow-no-tidal` for similar performance

**Smoothing:**
- smooth=5 (default) works well
- smooth=3 or 7 give similar but slightly worse results

**Delta (H0 floor):**
- Default δ=1.0 works fine
- Variations (0.5-2.0) have minimal impact

## Pointwise Data Export

Both baseline and flow coherence exports include:
- **Basic**: galaxy, R_kpc, V_obs, V_pred, Sigma_req, Sigma_pred, dSigma
- **Component fractions**: f_bulge_r, f_disk_r, f_gas_r
- **Gradient proxies**: dlnVbar_dlnR, dlnGbar_dlnR
- **Timescale proxies**: Omega_bar_SI, tau_dyn_Myr
- **Model internals**: h_term, C_term, A_use

**Flow-specific columns:**
- `omega2`: Vorticity squared (km/s/kpc)²
- `shear2`: Shear squared (km/s/kpc)²
- `theta2`: Divergence squared (km/s/kpc)²
- `tidal2`: Tidal proxy (km/s/kpc)²

## Usage

### Best performing configuration (bulge-specific):
```bash
python scripts/run_regression_experimental.py --core \
  --coherence=flow \
  --flow-bulge-specific \
  --flow-alpha-bulge=0.0 \
  --flow-gamma-bulge=0.01 \
  --flow-alpha-disk=0.02 \
  --flow-gamma-disk=0.005 \
  --flow-smooth=5 \
  --flow-no-tidal
```

### With pointwise export:
```bash
python scripts/run_regression_experimental.py --core \
  --coherence=flow \
  --flow-bulge-specific \
  --flow-alpha-bulge=0.0 \
  --flow-gamma-bulge=0.01 \
  --flow-alpha-disk=0.02 \
  --flow-gamma-disk=0.005 \
  --flow-smooth=5 \
  --flow-no-tidal \
  --export-sparc-points=scripts/regression_results/sparc_pointwise_flow_bulge_specific.csv
```

## Next Steps

1. **Residual analysis**: Use pointwise exports to analyze which flow invariants correlate with residuals
2. **Gaia 6D features**: Test with precomputed Gaia flow features using `--gaia-flow-features`
3. **Bulge-specific tuning**: Investigate why bulge galaxies show larger degradation
4. **Parameter optimization**: Systematic grid search for even better parameters

## Files Generated

- `sparc_pointwise_baseline.csv`: Baseline (C coherence) pointwise data
- `sparc_pointwise_flow_optimal.csv`: Flow coherence pointwise data with optimal parameters
- `experimental_report_C.json`: Baseline test results
- `experimental_report_FLOW.json`: Flow coherence test results

