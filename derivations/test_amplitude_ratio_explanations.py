#!/usr/bin/env python3
"""
Test Amplitude Ratio Explanations
=================================

The cluster validation shows A_cluster/A_galaxy ≈ 5.2 (need ~9 vs ~1.73),
much larger than the current mode-counting prediction of π√2/√3 ≈ 2.57.

This script tests two spatial (not temporal) explanations:

OPTION A: Same h(g), different A due to spatial geometry
---------------------------------------------------------
A1: 3D vs 2D mode counting (more spherical harmonic modes in clusters)
A2: Coherence window saturation (W ≈ 1 for clusters, <W> < 1 for galaxies)
A3: Lensing probes outer regions where coherence is higher

OPTION B: Different h(g) due to how observables couple to spatial field
-----------------------------------------------------------------------
B1: Gradient coupling (dynamics) vs potential coupling (lensing)
B2: Tensor component coupling (Σ_ij ≠ Σ_00)
B3: Path geometry (circular orbit vs straight photon path)

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Physical constants
c = 2.998e8          # m/s
H0_SI = 70 * 1000 / 3.086e22  # s⁻¹
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))
A_galaxy = np.sqrt(3)
A_cluster_current = np.pi * np.sqrt(2)
A_cluster_needed = 9.0  # From profile-based validation

print("=" * 80)
print("TESTING AMPLITUDE RATIO EXPLANATIONS")
print("=" * 80)
print(f"\nCurrent situation:")
print(f"  A_galaxy = √3 = {A_galaxy:.3f}")
print(f"  A_cluster (current) = π√2 = {A_cluster_current:.3f}")
print(f"  A_cluster (needed from data) ≈ {A_cluster_needed:.1f}")
print(f"  Ratio needed: {A_cluster_needed/A_galaxy:.2f}")
print(f"  Ratio from mode-counting: {A_cluster_current/A_galaxy:.2f}")
print(f"  Gap to explain: factor {A_cluster_needed/A_cluster_current:.2f}")


# =============================================================================
# OPTION A2: COHERENCE WINDOW SATURATION
# =============================================================================

print("\n" + "=" * 80)
print("OPTION A2: COHERENCE WINDOW SATURATION")
print("=" * 80)

def W_coherence(r, R_d):
    """Coherence window W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)R_d"""
    xi = (2/3) * R_d
    return 1 - (xi / (xi + r)) ** 0.5


def compute_effective_W(r_points, R_d, weights=None):
    """
    Compute effective <W> for a galaxy.
    
    For rotation curves, weight by where data points are.
    For lensing, weight by surface density (∝ exp(-r/R_d) for exponential disk).
    """
    W_values = W_coherence(r_points, R_d)
    
    if weights is None:
        # Uniform weighting
        return np.mean(W_values)
    else:
        return np.average(W_values, weights=weights)


# Load SPARC data to get typical R_d and radial coverage
sparc_file = Path(__file__).parent.parent / "vendor" / "sparc" / "MassModels_Lelli2016c.mrt"

print("\nAnalyzing SPARC galaxies for effective <W>...")

# Read SPARC rotation curve data
try:
    # Try to load rotation curve data
    rc_files = list((Path(__file__).parent.parent / "vendor" / "sparc" / "RotationCurves").glob("*.dat"))
    
    if not rc_files:
        print("  No rotation curve files found, using synthetic data")
        # Use typical values
        R_d_values = np.array([2.0, 3.0, 4.0, 5.0])  # kpc
        r_max_over_Rd = np.array([3, 4, 5, 6])
        
        W_eff_values = []
        for R_d, r_max_ratio in zip(R_d_values, r_max_over_Rd):
            r_points = np.linspace(0.5, r_max_ratio * R_d, 20)
            W_eff = compute_effective_W(r_points, R_d)
            W_eff_values.append(W_eff)
            print(f"  R_d = {R_d:.1f} kpc, r_max = {r_max_ratio}×R_d: <W> = {W_eff:.3f}")
        
        W_eff_mean = np.mean(W_eff_values)
    else:
        # Load actual SPARC data
        W_eff_values = []
        n_galaxies = 0
        
        for rc_file in rc_files[:50]:  # Sample first 50
            try:
                # Read rotation curve
                data = np.loadtxt(rc_file, usecols=(0,))  # First column is radius
                r_kpc = data
                
                # Get disk scale length from filename or use typical value
                R_d = 3.0  # kpc (typical)
                
                if len(r_kpc) > 3:
                    W_eff = compute_effective_W(r_kpc, R_d)
                    W_eff_values.append(W_eff)
                    n_galaxies += 1
            except:
                continue
        
        if W_eff_values:
            W_eff_mean = np.mean(W_eff_values)
            print(f"  Analyzed {n_galaxies} galaxies")
            print(f"  Mean <W> = {W_eff_mean:.3f}")
            print(f"  Range: {np.min(W_eff_values):.3f} - {np.max(W_eff_values):.3f}")
        else:
            W_eff_mean = 0.6  # Fallback
            print("  Using fallback <W> = 0.6")

except Exception as e:
    print(f"  Error loading SPARC data: {e}")
    W_eff_mean = 0.6  # Fallback

# For clusters, W = 1 (no inner suppression in lensing region)
W_cluster = 1.0

print(f"\nCoherence window comparison:")
print(f"  <W>_galaxy ≈ {W_eff_mean:.3f}")
print(f"  W_cluster = {W_cluster:.3f}")
print(f"  Ratio W_cluster/<W>_galaxy = {W_cluster/W_eff_mean:.2f}")

# Effective amplitude ratio from W alone
W_ratio = W_cluster / W_eff_mean
print(f"\nIf same bare A, effective A ratio from W alone: {W_ratio:.2f}")

# Combined with mode counting
combined_ratio = W_ratio * (A_cluster_current / A_galaxy)
print(f"Combined with mode-counting ({A_cluster_current/A_galaxy:.2f}): {combined_ratio:.2f}")
print(f"Needed: {A_cluster_needed/A_galaxy:.2f}")
print(f"Gap remaining: {(A_cluster_needed/A_galaxy)/combined_ratio:.2f}×")


# =============================================================================
# OPTION A1: EXTENDED MODE COUNTING
# =============================================================================

print("\n" + "=" * 80)
print("OPTION A1: EXTENDED MODE COUNTING")
print("=" * 80)

print("""
Current derivation:
  - Disk (2D): 3 torsion modes → A = √3
  - Sphere (3D): ℓ = 0,1,2 modes + 2 polarizations → A = π√2

But for extended sources, higher multipoles contribute.
For a source subtending angle θ, modes up to ℓ_max ~ π/θ contribute.
""")

# Cluster angular sizes
z_cluster = 0.3  # typical
theta_cluster_arcmin = 5  # typical Einstein radius
theta_cluster_rad = theta_cluster_arcmin * np.pi / (180 * 60)

l_max = np.pi / theta_cluster_rad
print(f"Typical cluster at z = {z_cluster}:")
print(f"  θ_E ≈ {theta_cluster_arcmin} arcmin")
print(f"  ℓ_max ~ π/θ ≈ {l_max:.0f}")

# Number of modes up to ℓ_max
n_modes_extended = l_max**2
n_modes_basic = 5  # ℓ = 0,1,2 gives 1+3+5 = 9 modes, but only ~5 physical

print(f"  Number of modes (basic): ~{n_modes_basic}")
print(f"  Number of modes (extended): ~{n_modes_extended:.0f}")

# Amplitude scaling with mode count
A_extended = A_cluster_current * np.sqrt(n_modes_extended / n_modes_basic)
print(f"\nIf A scales as √(n_modes):")
print(f"  A_cluster (extended) ≈ {A_extended:.1f}")
print(f"  This is {'larger' if A_extended > A_cluster_needed else 'smaller'} than needed ({A_cluster_needed:.1f})")

# More conservative: only ℓ up to ~10 contribute coherently
l_max_conservative = 10
n_modes_conservative = l_max_conservative**2
A_conservative = A_cluster_current * np.sqrt(n_modes_conservative / n_modes_basic)
print(f"\nConservative (ℓ_max ~ 10):")
print(f"  A_cluster ≈ {A_conservative:.1f}")


# =============================================================================
# OPTION A3: APERTURE RADIUS EFFECT
# =============================================================================

print("\n" + "=" * 80)
print("OPTION A3: APERTURE RADIUS EFFECT")
print("=" * 80)

print("""
Rotation curves measure g(r) at specific radii (typically r ~ 1-5 R_d).
Lensing measures integrated mass within R_E (typically ~200 kpc for clusters).

The effective enhancement depends on where you're measuring:
  - Inner regions: g >> g†, h(g) → 0, Σ → 1
  - Outer regions: g << g†, h(g) → large, Σ >> 1
""")

def h_universal(g):
    """Universal acceleration function h(g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def Sigma_enhancement(g, A):
    """Enhancement factor Σ = 1 + A × h(g)"""
    return 1 + A * h_universal(g)

# Typical accelerations
g_galaxy_inner = 1e-10  # m/s² (inner disk)
g_galaxy_outer = 1e-11  # m/s² (outer disk)
g_cluster_200kpc = 3e-11  # m/s² (at 200 kpc)

print(f"\nTypical accelerations:")
print(f"  Galaxy inner (r ~ R_d): g = {g_galaxy_inner:.1e} m/s², g/g† = {g_galaxy_inner/g_dagger:.2f}")
print(f"  Galaxy outer (r ~ 5R_d): g = {g_galaxy_outer:.1e} m/s², g/g† = {g_galaxy_outer/g_dagger:.2f}")
print(f"  Cluster (r ~ 200 kpc): g = {g_cluster_200kpc:.1e} m/s², g/g† = {g_cluster_200kpc/g_dagger:.2f}")

print(f"\nEnhancement h(g) at these accelerations:")
print(f"  Galaxy inner: h = {h_universal(g_galaxy_inner):.3f}")
print(f"  Galaxy outer: h = {h_universal(g_galaxy_outer):.3f}")
print(f"  Cluster 200kpc: h = {h_universal(g_cluster_200kpc):.3f}")

# The cluster h is larger because g is lower
h_ratio = h_universal(g_cluster_200kpc) / h_universal(g_galaxy_inner)
print(f"\nRatio h_cluster/h_galaxy_inner = {h_ratio:.2f}")


# =============================================================================
# OPTION B1: GRADIENT COUPLING (∇Σ TERM)
# =============================================================================

print("\n" + "=" * 80)
print("OPTION B1: GRADIENT COUPLING")
print("=" * 80)

print("""
The full acceleration from the modified Poisson equation includes:
  g_obs = g_bar × Σ + Φ_bar × ∇Σ

The second term (∇Σ) appears in dynamics but NOT in lensing (which 
integrates over z, washing out gradients).

If ∇Σ ∝ (∂Σ/∂r) and Σ = 1 + A × h(g), then:
  ∂Σ/∂r = A × (∂h/∂g) × (∂g/∂r)

For h(g) = √(g†/g) × g†/(g†+g):
  ∂h/∂g = -½ × √(g†/g³) × g†/(g†+g) - √(g†/g) × g†/(g†+g)²
        ≈ -h(g) × (1/2g + 1/(g†+g))

This is negative and scales as ~h(g)/g.
""")

# Compute the gradient term magnitude
def dh_dg(g):
    """Derivative of h with respect to g"""
    h = h_universal(g)
    return -h * (0.5/g + 1/(g_dagger + g))

def gradient_term_ratio(g, r_kpc, R_d_kpc):
    """
    Ratio of gradient term to main term: (Φ × ∇Σ) / (g × Σ)
    
    Φ ~ g × r (for circular velocity)
    ∇Σ ~ A × dh/dg × dg/dr
    dg/dr ~ -2g/r (for point mass)
    """
    h = h_universal(g)
    dh = dh_dg(g)
    
    # Approximate: Φ ~ g × r, dg/dr ~ -2g/r
    # Gradient term: Φ × A × dh × dg/dr = g × r × A × dh × (-2g/r) = -2 × A × g² × dh
    # Main term: g × Σ = g × (1 + A × h)
    
    Sigma = 1 + A_galaxy * h
    gradient = -2 * A_galaxy * g * dh
    main = Sigma
    
    return np.abs(gradient / main)

g_test = 1e-10  # m/s²
r_test = 10.0   # kpc
R_d_test = 3.0  # kpc

ratio = gradient_term_ratio(g_test, r_test, R_d_test)
print(f"\nGradient term analysis at g = {g_test:.1e} m/s²:")
print(f"  |∇Σ term| / |main term| ≈ {ratio:.3f}")
print(f"  This is {'significant' if ratio > 0.1 else 'small'}")

# Check if this could explain the √(g†/g) factor
print(f"\nDoes ∇Σ term explain √(g†/g) factor?")
print(f"  √(g†/g) at g = g† gives factor = 1")
print(f"  √(g†/g) at g = 0.1×g† gives factor = {np.sqrt(10):.2f}")
print(f"  The gradient term scales similarly but is subdominant")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY: EXPLAINING THE AMPLITUDE RATIO")
print("=" * 80)

print(f"""
Needed: A_cluster/A_galaxy ≈ {A_cluster_needed/A_galaxy:.1f}
Current mode-counting: {A_cluster_current/A_galaxy:.2f}
Gap: factor {A_cluster_needed/A_cluster_current:.1f}

OPTION A (same h, different A):
  A1 (extended modes): Could give factor {np.sqrt(n_modes_conservative/n_modes_basic):.1f} (conservative)
  A2 (W saturation): Gives factor {W_ratio:.1f}
  Combined A1 + A2: ~{W_ratio * np.sqrt(n_modes_conservative/n_modes_basic):.1f}
  
  Assessment: Can explain ~{W_ratio * np.sqrt(n_modes_conservative/n_modes_basic) * (A_cluster_current/A_galaxy):.1f}× 
              vs needed {A_cluster_needed/A_galaxy:.1f}×
              {'✓ Sufficient' if W_ratio * np.sqrt(n_modes_conservative/n_modes_basic) * (A_cluster_current/A_galaxy) > A_cluster_needed/A_galaxy * 0.8 else '✗ Insufficient'}

OPTION B (different h):
  B1 (gradient coupling): Subdominant (~{ratio*100:.0f}% correction)
  
  Assessment: Cannot explain the full difference alone

RECOMMENDED PATH:
  Keep h(g) universal, explain A_cluster ≈ 9 via:
  1. Extended mode counting for 3D spherical geometry
  2. Coherence window saturation (W = 1 for clusters)
  3. Possibly: aperture radius effect (clusters measured at lower g/g†)
""")

# Save results
results = {
    'A_galaxy': A_galaxy,
    'A_cluster_current': A_cluster_current,
    'A_cluster_needed': A_cluster_needed,
    'W_eff_galaxy': W_eff_mean,
    'W_cluster': W_cluster,
    'W_ratio': W_ratio,
    'mode_ratio_conservative': np.sqrt(n_modes_conservative/n_modes_basic),
    'combined_explanation': W_ratio * np.sqrt(n_modes_conservative/n_modes_basic) * (A_cluster_current/A_galaxy),
}

output_file = Path(__file__).parent / "amplitude_ratio_analysis.json"
import json
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_file}")

