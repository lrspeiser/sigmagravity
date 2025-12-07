# Changelog

This document tracks major revisions to the Σ-Gravity framework and documentation.

---

## December 2025 (Latest)

### 2D Coherence Framework Adoption

**Major theoretical advance:** Disk galaxies have 2D coherence (not 1D), leading to derived parameters:

1. **Coherence exponent k = 1**
   - Derived from 2D disk geometry: k = ν/2 where ν = 2 (spatial dimensions)
   - Simplifies W(r) to rational function: W(r) = r/(ξ+r)
   - For clusters (3D): k = 1.5

2. **Coherence scale ξ = R_d/(2π)**
   - Derived from one azimuthal wavelength at disk scale length
   - Replaces phenomenological ξ = (1/2)R_d or (2/3)R_d
   - Typical value: ~0.5 kpc for disk galaxies

3. **Amplitude A = e^(1/(2π)) ≈ 1.173**
   - Derived from inverse density ratio at coherence boundary
   - Replaces phenomenological A = √e or √3

### Performance Improvements

| Test | Previous | Current |
|------|----------|---------|
| SPARC RMS | 20.2 km/s | 17.5 km/s |
| SPARC Win Rate | 42% | 48% |
| Cluster Ratio | 0.955 | 0.955 |
| Cluster Scatter | 0.133 dex | 0.133 dex |
| MW RMS | 29.0 km/s | 29.4 km/s |

### Derivation Status

All galaxy parameters are now derived from first principles:
- ✓ k = 1 (2D geometry)
- ✓ ξ = R_d/(2π) (azimuthal wavelength)
- ✓ A = e^(1/(2π)) (inverse density at ξ)
- ✓ W(r) = r/(ξ+r) (superstatistics with k=1)
- ✓ h(g) = √(g†/g) × g†/(g†+g) (enhancement × coherence probability)

---

## December 2025 (Earlier)

### Theoretical Advances

1. **Dynamical coherence scale (superseded)**
   - Replaced phenomenological ξ = (2/3)R_d with ξ = k×σ_eff/Ω_d (k ≈ 0.24)
   - Now superseded by 2D-derived ξ = R_d/(2π)

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

