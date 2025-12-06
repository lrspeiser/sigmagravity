# Changelog

This document tracks major revisions to the Σ-Gravity framework and documentation.

---

## December 2025

### Theoretical Advances

1. **Dynamical coherence scale**
   - Replaced phenomenological ξ = (2/3)R_d with dynamically motivated ξ = k×σ_eff/Ω_d (k ≈ 0.24)
   - 16% improvement in rotation curve predictions
   - Validated with baryons-only computation (not circular)

2. **Unified amplitude formula**
   - A(G) = √(1.6 + 109×G²) connects galaxies (G=0.038) and clusters (G=1.0)
   - Single calibrated formula replaces separate galaxy/cluster amplitudes

3. **Conservation and fifth-force resolution**
   - Stress-energy conservation established via dynamical coherence field
   - Fifth-force concern eliminated via QUMOND-like formulation with minimal matter coupling
   - Matter Lagrangian convention explicitly specified as L_m = -ρc²

### Validation Updates

- Full regression test suite (`derivations/full_regression_test.py`)
- Star-by-star Milky Way validation (28,368 Eilers-APOGEE-Gaia stars)
- Robustness tests against multiple MOND variants

---

## November 2025

### Critical Acceleration Formula

- Updated g† = cH₀/(2e) → g† = cH₀/(4√π)
- 14.3% improvement in rotation curve fits
- Geometric derivation from spherical coherence

### Cluster Amplitude Derivation

- Mode counting: π√2/√3 ≈ 2.57 (3D vs 2D geometry)
- Coherence saturation: 1/⟨W⟩ ≈ 1.9 (W=1 at lensing radii)
- Combined: A_cluster/A_galaxy ≈ 4.9

### SPARC Calibration

- Corrected M/L ratios: 0.5 (disk), 0.7 (bulge) per Lelli+ 2016
- Consistent with Milky Way at R = 8 kpc

---

## Earlier Versions

### Original Framework

- Phenomenological enhancement Σ = 1 + A × f(r) × h(g)
- Separate fitted amplitudes for galaxies and clusters
- Critical acceleration from dimensional analysis (cH₀ scale)

### Known Issues (Resolved)

- Stress-energy conservation: Resolved via dynamical field
- Fifth force: Eliminated via minimal coupling
- Cluster amplitude: Now derived from geometry

