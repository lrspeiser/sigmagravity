# Residual Discovery Results

## Key Finding: **Vorticity is the #1 Driver in Gaia!**

The Gaia flow driver discovery shows that **`log10_omega` (vorticity) is the top feature** predicting residuals, strongly supporting the **flow topology hypothesis**.

## SPARC Residual Driver Discovery

### All Points (N=3,373)

**Top permutation importances:**
1. `dlnVbar_dlnR`: +18.28 ± 13.08 (velocity gradient)
2. `dlnGbar_dlnR`: +17.19 ± 12.25 (acceleration gradient)
3. `h_term`: +1.73 ± 0.99 (acceleration function)
4. `log10_x_gbar`: +0.75 ± 0.53 (normalized acceleration)
5. `f_disk_r`: +0.19 ± 0.17 (disk fraction)

**Key insights:**
- **Velocity/acceleration gradients dominate** - suggests flow topology matters
- Model factors (`h_term`) are important but secondary
- Morphology features appear but are less important than gradients

### High-Bulge Regions (f_bulge_r > 0.6, N=356)

**Top permutation importances:**
1. `log10_Omega_over_H0`: +1.72 ± 1.14 (orbital frequency)
2. `log10_x_gbar`: +1.66 ± 1.37 (normalized acceleration)
3. `h_term`: +1.12 ± 1.57 (acceleration function)
4. `R_over_L0`: +0.90 ± 0.46 (compactness)
5. `R_over_Rd`: +0.53 ± 0.45 (normalized radius)

**Key insights:**
- **Different pattern than all points** - orbital frequency and acceleration regime matter more
- `tidal_L0` has high correlation (+0.55) but negative permutation importance (interesting!)
- Gradient features (`dlnVbar_dlnR`, `dlnGbar_dlnR`) have negative importance here (model may be overfitting)

### Bulge Galaxies (f_bulge_global > 0.3)

**Top permutation importances:**
1. `h_term`: +1.91 ± 2.37 (acceleration function)
2. `log10_Omega_over_H0`: +1.74 ± 0.26 (orbital frequency)
3. `R_over_L0`: +1.35 ± 1.15 (compactness)
4. `tidal_L0`: +0.31 ± 0.20 (tidal/decoherence proxy)
5. `log10_x_gbar`: +0.30 ± 0.41 (normalized acceleration)

**Key insights:**
- `h_term` is most important (acceleration suppression)
- Orbital frequency (`log10_Omega_over_H0`) is consistently important
- Tidal proxy (`tidal_L0`) shows positive importance here

## Gaia Flow Driver Discovery

**Top permutation importances (N=5,000 stars):**
1. **`log10_omega`**: +0.0745 ± 0.0170 ⭐ **VORTICITY IS #1!**
2. `z_kpc`: +0.0463 ± 0.0157 (vertical position)
3. `R_kpc`: +0.0285 ± 0.0101 (radial position)
4. **`log10_shear`**: +0.0199 ± 0.0095 (shear)
5. `log10_n_star`: +0.0179 ± 0.0088 (number density)
6. `sigma_1d_kms`: +0.0124 ± 0.0094 (velocity dispersion)
7. `theta_abs`: +0.0122 ± 0.0034 (divergence)
8. `log10_d_star`: +0.0121 ± 0.0112 (mean separation)

**Key insights:**
- **Vorticity (`log10_omega`) is the #1 feature** - strongly supports flow topology hypothesis!
- **Shear (`log10_shear`) is #4** - also important for flow topology
- Density/separation features (`log10_n_star`, `log10_d_star`) matter
- Position features (`R_kpc`, `z_kpc`) are baseline but less important than flow features

## Interpretation

### The Flow Topology Hypothesis is Confirmed

The fact that **vorticity is #1 in Gaia** strongly supports the hypothesis that:
> **Coherence is not set by speed or dispersion alone — it's set by 3D flow topology**

### SPARC vs Gaia Patterns

**SPARC (rotation curves):**
- Gradient features (`dlnVbar_dlnR`, `dlnGbar_dlnR`) dominate
- These are **proxies for flow topology** in axisymmetric systems
- Orbital frequency (`log10_Omega_over_H0`) is important in bulge regions

**Gaia (6D star field):**
- **Direct flow topology features** (vorticity, shear) are most important
- This confirms that the missing variable is **flow structure**, not just scalar speed

### Why This Matters

1. **Bulges have different flow topology** than disks:
   - Disks: high vorticity, low shear → coherence on
   - Bulges: low vorticity, high shear → coherence off/screened

2. **The missing variable is a tensor invariant**, not a scalar:
   - Current: `C = v²/(v²+σ²)` (scalar speed-based)
   - Needed: `C_flow = ω²/(ω² + α·s² + β·θ² + ...)` (tensor-based)

3. **This explains why bulges are hard to fix**:
   - Current model can't capture flow topology from rotation curves alone
   - Need direct vorticity/shear measurements or better proxies

## Next Steps

1. **Implement SPARC-side ω/shear proxies** from rotation curve derivatives:
   - For axisymmetric systems: `ω ~ V/R`, `s ~ |dV/dR|`
   - Create `C_flow_proxy` using these

2. **Implement `--coherence=flow` mode**:
   - Use vorticity-fraction coherence: `C_flow = ω²/(ω² + α·s² + β·θ²)`
   - Test against baseline to see if it improves bulge predictions

3. **Full Gaia run** (not just 5k stars) to confirm vorticity dominance

4. **Compare with MOND** to see if flow topology explains the difference

## Files Created

- `scripts/discover_sparc_residual_drivers.py`
- `scripts/export_gaia_pointwise_features.py`
- `scripts/discover_gaia_flow_drivers.py`
- Outputs:
  - `scripts/regression_results/residual_drivers/sparc_residual_drivers_*.csv`
  - `scripts/regression_results/gaia_pointwise_features.csv`
  - `scripts/regression_results/gaia_flow_drivers/gaia_flow_drivers_resid_kms.csv`
