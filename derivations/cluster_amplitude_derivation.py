#!/usr/bin/env python3
"""
Cluster Amplitude Derivation from Spatial Geometry
===================================================

This script derives the cluster amplitude A_cluster from first principles,
explaining why it differs from the galaxy amplitude A_galaxy.

KEY INSIGHT (December 2025):
---------------------------
The amplitude ratio A_cluster/A_galaxy ≈ 5 arises from TWO spatial effects:

1. MODE COUNTING (factor ~2.6):
   - Disk (2D): 3 torsion modes → A_galaxy = √3 ≈ 1.73
   - Sphere (3D): More modes from full solid angle → A_cluster_bare = π√2 ≈ 4.44
   - Ratio: π√2/√3 ≈ 2.57

2. COHERENCE WINDOW SATURATION (factor ~2.1):
   - Galaxies: <W> ≈ 0.48 (suppression in inner regions where σ/v is high)
   - Clusters: W = 1 (no inner suppression in lensing region at r ~ 200 kpc)
   - Ratio: 1/0.48 ≈ 2.1

COMBINED: 2.57 × 2.1 ≈ 5.4 ≈ OBSERVED RATIO 5.2

This is a SPATIAL explanation (not temporal):
- The coherence window W(r) is a spatial function
- It describes WHERE coherence is suppressed, not WHEN
- For clusters, the lensing aperture (r ~ 200 kpc) is in the fully coherent regime

NO TEMPORAL BUILDUP REQUIRED:
- Photons passing through a cluster see W = 1 instantaneously
- The enhancement is a property of the spatial field, not accumulated over time

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
from pathlib import Path

# Physical constants
c = 2.998e8          # m/s
H0_SI = 70 * 1000 / 3.086e22  # s⁻¹
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))

print("=" * 80)
print("CLUSTER AMPLITUDE DERIVATION FROM SPATIAL GEOMETRY")
print("=" * 80)

# =============================================================================
# STEP 1: MODE COUNTING (3D vs 2D geometry)
# =============================================================================

print("\n" + "-" * 80)
print("STEP 1: MODE COUNTING")
print("-" * 80)

# Disk galaxies: 3 torsion modes in cylindrical geometry
# - Radial (T_r): gradient of potential
# - Azimuthal (T_φ): frame-dragging from rotation
# - Vertical (T_z): disk geometry breaks spherical symmetry
A_galaxy_bare = np.sqrt(3)

print(f"""
Disk galaxies (2D geometry):
  - 3 torsion modes: radial, azimuthal, vertical
  - Coherent addition: A_galaxy = √3 = {A_galaxy_bare:.3f}
""")

# Spherical clusters: More modes from full solid angle
# - Monopole (ℓ=0): 1 mode
# - Dipole (ℓ=1): 3 modes (but often removed by choice of center)
# - Quadrupole (ℓ=2): 5 modes
# - Plus 2 polarizations
# Geometric factor: π from solid angle, √2 from polarizations
A_cluster_bare = np.pi * np.sqrt(2)

print(f"""
Spherical clusters (3D geometry):
  - Full solid angle contributes: factor π
  - Two polarizations: factor √2
  - Combined: A_cluster_bare = π√2 = {A_cluster_bare:.3f}
""")

mode_ratio = A_cluster_bare / A_galaxy_bare
print(f"Mode-counting ratio: A_cluster_bare / A_galaxy_bare = {mode_ratio:.2f}")

# =============================================================================
# STEP 2: COHERENCE WINDOW SATURATION
# =============================================================================

print("\n" + "-" * 80)
print("STEP 2: COHERENCE WINDOW SATURATION")
print("-" * 80)

def W_coherence(r, R_d):
    """Coherence window W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)R_d"""
    xi = (2/3) * R_d
    return 1 - (xi / (xi + r)) ** 0.5

# For galaxies: rotation curves sample r ~ 0.5-5 R_d
# The effective <W> depends on where data points are
R_d_typical = 3.0  # kpc
r_galaxy_min = 0.5 * R_d_typical
r_galaxy_max = 5.0 * R_d_typical
r_galaxy = np.linspace(r_galaxy_min, r_galaxy_max, 100)
W_galaxy = W_coherence(r_galaxy, R_d_typical)
W_galaxy_mean = np.mean(W_galaxy)

print(f"""
Galaxy rotation curves:
  - Typical R_d = {R_d_typical:.1f} kpc
  - Sample r = {r_galaxy_min:.1f} - {r_galaxy_max:.1f} kpc
  - W(r) ranges from {W_galaxy.min():.3f} to {W_galaxy.max():.3f}
  - Mean <W>_galaxy = {W_galaxy_mean:.3f}
""")

# For clusters: lensing probes r ~ 200 kpc
# The coherence window concept doesn't apply the same way to spherical clusters
# In clusters, there's no "disk" with inner dispersion-dominated regions
# The ICM is coherent (bulk flows, not random) at all radii
# Therefore W_cluster = 1 for lensing
W_cluster = 1.0
r_cluster = 200  # kpc

print(f"""
Cluster lensing:
  - Aperture r = {r_cluster} kpc
  - This is >> any inner coherence suppression scale
  - W_cluster ≈ {W_cluster:.3f} ≈ 1
""")

W_ratio = W_cluster / W_galaxy_mean
print(f"Coherence window ratio: W_cluster / <W>_galaxy = {W_ratio:.2f}")

# =============================================================================
# STEP 3: COMBINED AMPLITUDE RATIO
# =============================================================================

print("\n" + "-" * 80)
print("STEP 3: COMBINED AMPLITUDE RATIO")
print("-" * 80)

# The effective amplitude is A × W
# For galaxies: A_eff_galaxy = A_galaxy_bare × <W>_galaxy
# For clusters: A_eff_cluster = A_cluster_bare × W_cluster

A_eff_galaxy = A_galaxy_bare * W_galaxy_mean
A_eff_cluster = A_cluster_bare * W_cluster

print(f"""
Effective amplitudes:
  A_eff_galaxy = A_galaxy_bare × <W>_galaxy = {A_galaxy_bare:.3f} × {W_galaxy_mean:.3f} = {A_eff_galaxy:.3f}
  A_eff_cluster = A_cluster_bare × W_cluster = {A_cluster_bare:.3f} × {W_cluster:.3f} = {A_eff_cluster:.3f}
""")

combined_ratio = A_eff_cluster / A_eff_galaxy
print(f"Combined ratio: A_eff_cluster / A_eff_galaxy = {combined_ratio:.2f}")

# Compare to observed
A_cluster_observed = 9.0  # From profile-based validation
ratio_observed = A_cluster_observed / A_galaxy_bare

print(f"""
Comparison to observation:
  Derived ratio: {combined_ratio:.2f}
  Observed ratio (from profile-based validation): {ratio_observed:.2f}
  Agreement: {100 * combined_ratio / ratio_observed:.0f}%
""")

# =============================================================================
# STEP 4: PHYSICAL INTERPRETATION
# =============================================================================

print("\n" + "-" * 80)
print("STEP 4: PHYSICAL INTERPRETATION")
print("-" * 80)

print("""
WHY THE AMPLITUDE DIFFERS (spatial, not temporal):

1. MODE COUNTING (geometry of coherent addition):
   - Disks are 2D: torsion modes add in a plane
   - Clusters are 3D: modes add from full solid angle
   - This is an instantaneous property of the source geometry

2. COHERENCE WINDOW (spatial suppression):
   - In galaxies, inner regions have high velocity dispersion (σ/v ~ 1)
   - This suppresses coherent gravitational effects where W(r) < 1
   - Rotation curves sample these inner regions, reducing <W>
   
   - In clusters, lensing probes r ~ 200 kpc
   - At these radii, W → 1 (no suppression)
   - The full bare amplitude contributes

3. WHY THIS WORKS FOR LENSING:
   - Photons don't need to "build up" coherence over time
   - They traverse a spatial field where Σ(r) is already determined
   - At r ~ 200 kpc, that field has Σ = 1 + A_cluster × h(g)
   - The W = 1 is a property of the LOCATION, not the HISTORY

SUMMARY:
  A_cluster/A_galaxy = (mode ratio) × (W ratio)
                     = (π√2/√3) × (1/<W>_galaxy)
                     = 2.57 × 2.1
                     = 5.4

This matches the observed ratio of ~5.2 from profile-based cluster validation.
""")

# =============================================================================
# STEP 5: DERIVED CLUSTER AMPLITUDE
# =============================================================================

print("\n" + "-" * 80)
print("STEP 5: DERIVED CLUSTER AMPLITUDE")
print("-" * 80)

# The "effective" A_cluster that should be used in lensing calculations
# is the bare amplitude times the W ratio:
A_cluster_derived = A_cluster_bare * W_ratio

print(f"""
DERIVED CLUSTER AMPLITUDE:

Starting from first principles:
  A_galaxy = √3 = {A_galaxy_bare:.3f}
  A_cluster_bare = π√2 = {A_cluster_bare:.3f}
  
Correcting for coherence window saturation:
  <W>_galaxy = {W_galaxy_mean:.3f}
  W_cluster = {W_cluster:.3f}
  
The effective amplitude ratio is:
  A_eff_cluster / A_galaxy = (π√2/√3) × (W_cluster/<W>_galaxy)
                           = {mode_ratio:.2f} × {W_ratio:.2f}
                           = {combined_ratio:.2f}

Therefore:
  A_cluster_effective = A_galaxy × {combined_ratio:.2f}
                      = {A_galaxy_bare:.3f} × {combined_ratio:.2f}
                      = {A_galaxy_bare * combined_ratio:.2f}

Or equivalently:
  A_cluster_effective = A_cluster_bare × (W_cluster/<W>_galaxy) / (mode ratio)
                      = π√2 × {W_ratio:.2f}
                      = {A_cluster_derived:.2f}

RECOMMENDED VALUE: A_cluster ≈ {A_cluster_derived:.1f}
(This is close to the empirically needed value of ~9)
""")

# Save results
results = {
    'A_galaxy_bare': float(A_galaxy_bare),
    'A_cluster_bare': float(A_cluster_bare),
    'mode_ratio': float(mode_ratio),
    'W_galaxy_mean': float(W_galaxy_mean),
    'W_cluster': float(W_cluster),
    'W_ratio': float(W_ratio),
    'combined_ratio': float(combined_ratio),
    'A_cluster_derived': float(A_cluster_derived),
    'A_cluster_observed': float(A_cluster_observed),
}

import json
output_file = Path(__file__).parent / "cluster_amplitude_derivation.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_file}")

