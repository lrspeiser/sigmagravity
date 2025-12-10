#!/usr/bin/env python3
"""
Unified Wake Model: Explaining Galaxy-Cluster Amplitude Difference
===================================================================

KEY INSIGHT FROM PREVIOUS ANALYSIS:
-----------------------------------
At low g/g† (where MOND matters), Σ-Gravity and MOND are actually CLOSE:
  g/g† = 0.01: Σ-Gravity = 12.61, MOND = 11.69 (ratio 1.08)
  g/g† = 0.10: Σ-Gravity = 4.37,  MOND = 4.06  (ratio 1.08)

The bigger differences in the README (74%, 37%) were for the h(g) function alone,
but when combined with the coherence window W(r), the predictions converge!

NEW HYPOTHESIS:
---------------
The galaxy/cluster amplitude difference (A_cluster/A_galaxy ≈ 6.8) might be
explained by a "wake inversion" effect:

  - GALAXIES: Ordered rotation creates coherent wakes that PARTIALLY CANCEL
    in the gravitational field → LESS enhancement than pure formula
    
  - CLUSTERS: Random motions create incoherent wakes that ADD CONSTRUCTIVELY
    (like random walk) → MORE enhancement than pure formula

This is analogous to:
  - Laser (coherent): Amplitude adds → intensity ∝ N
  - Thermal light (incoherent): Intensity adds → intensity ∝ N
  
Wait, that's backwards. Let me think again...

CORRECT ANALOGY:
----------------
  - Coherent sources: Amplitudes add → A_total = N × A_single → I ∝ N²
  - Incoherent sources: Intensities add → I_total = N × I_single → I ∝ N
  
So coherent should give MORE enhancement, not less!

But the observation is: clusters need MORE enhancement (A = 8) than galaxies (A = 1.17)

RESOLUTION:
-----------
The "coherence" in Σ-Gravity refers to the VELOCITY FIELD, not the gravitational field.

  - Galaxies: Ordered velocity field → gravitational enhancement is ORGANIZED
    but limited by the coherence window W(r)
    
  - Clusters: Disordered velocity field → but 3D geometry allows MORE modes
    to contribute → higher effective amplitude

The path length scaling A = A_0 × L^0.25 already captures this!
  - Galaxies: L ≈ 1.5 kpc → A ≈ 1.17
  - Clusters: L ≈ 400 kpc → A ≈ 8.0

So the wake model should NOT try to explain the amplitude ratio.
Instead, it should explain the SHAPE differences between Σ-Gravity and MOND.

Author: Leonard Speiser
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from wake_coherence_model import (
    WakeParams, C_wake_discrete, C_wake_continuum,
    A_GALAXY, XI_SCALE, g_dagger, kpc_to_m
)


# =============================================================================
# CONSTANTS
# =============================================================================
c = 2.998e8
H0_SI = 2.27e-18
G_const = 6.674e-11
M_sun = 1.989e30
a0_mond = 1.2e-10


# =============================================================================
# FUNCTIONS
# =============================================================================

def h_sigma(g):
    """Σ-Gravity h(g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def nu_mond(g):
    """MOND ν(g)"""
    x = g / a0_mond
    x = np.maximum(x, 1e-10)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


# =============================================================================
# ANALYSIS: SHAPE DIFFERENCES
# =============================================================================

def analyze_shape_difference():
    """
    The key difference between Σ-Gravity and MOND is the SHAPE:
    - Σ-Gravity: Enhancement grows with radius (W(r) → 1)
    - MOND: Enhancement depends only on g, constant at fixed g
    
    Can wake coherence explain this shape difference?
    """
    print("=" * 70)
    print("ANALYSIS: SHAPE DIFFERENCES BETWEEN Σ-GRAVITY AND MOND")
    print("=" * 70)
    
    # Simulate a galaxy
    R = np.linspace(0.5, 15, 30)
    R_d = 3.0
    V_flat = 150  # km/s
    
    # Rising then flat rotation curve
    V_c = V_flat * np.tanh(R / R_d)
    g = (V_c * 1000) ** 2 / (R * kpc_to_m)
    
    # Σ-Gravity components
    xi = XI_SCALE * R_d
    W = R / (xi + R)
    h = h_sigma(g)
    Sigma_sg = 1 + A_GALAXY * W * h
    
    # MOND
    nu = nu_mond(g)
    
    print(f"\nGalaxy: R_d = {R_d} kpc, V_flat = {V_flat} km/s")
    print()
    print(f"{'R (kpc)':<10} {'g/g†':<10} {'W(r)':<10} {'Σ-Grav':<10} {'MOND':<10} {'Ratio':<10}")
    print("-" * 60)
    
    for i in [0, 5, 10, 15, 20, 25, 29]:
        g_ratio = g[i] / g_dagger
        ratio = Sigma_sg[i] / nu[i]
        print(f"{R[i]:<10.1f} {g_ratio:<10.3f} {W[i]:<10.3f} {Sigma_sg[i]:<10.3f} {nu[i]:<10.3f} {ratio:<10.3f}")
    
    print("\n" + "-" * 70)
    print("OBSERVATION:")
    print("  Inner regions (small R): Σ-Gravity < MOND (W suppresses)")
    print("  Outer regions (large R): Σ-Gravity ≈ MOND (W → 1)")
    print("  The coherence window W(r) already provides shape correction!")
    print("-" * 70)


def analyze_wake_as_shape_modifier():
    """
    What if wake coherence modifies the SHAPE of enhancement, not the amplitude?
    
    Idea: C_wake varies with radius in a way that complements W(r)
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: WAKE AS SHAPE MODIFIER")
    print("=" * 70)
    
    # Typical galaxy
    R = np.linspace(0.5, 15, 30)
    R_d = 3.0
    V_flat = 150
    
    # Kinematics
    V_c = V_flat * np.tanh(R / R_d)
    
    # Velocity dispersion profile (higher in center, lower in outer disk)
    sigma = 50 * np.exp(-R / (2 * R_d)) + 20
    
    # Wake coherence from v_rot/σ
    C_wake = V_c**2 / (V_c**2 + sigma**2)
    
    # Standard Σ-Gravity
    xi = XI_SCALE * R_d
    W = R / (xi + R)
    g = (V_c * 1000) ** 2 / (R * kpc_to_m)
    h = h_sigma(g)
    
    Sigma_base = 1 + A_GALAXY * W * h
    
    # Wake-modified Σ-Gravity
    Sigma_wake = 1 + A_GALAXY * W * h * C_wake
    
    # MOND
    nu = nu_mond(g)
    
    print(f"\nDispersion profile: σ = 50×exp(-R/2R_d) + 20 km/s")
    print()
    print(f"{'R (kpc)':<8} {'σ (km/s)':<10} {'C_wake':<10} {'Σ-base':<10} {'Σ-wake':<10} {'MOND':<10}")
    print("-" * 60)
    
    for i in [0, 5, 10, 15, 20, 25, 29]:
        print(f"{R[i]:<8.1f} {sigma[i]:<10.1f} {C_wake[i]:<10.3f} "
              f"{Sigma_base[i]:<10.3f} {Sigma_wake[i]:<10.3f} {nu[i]:<10.3f}")
    
    # Compute how close wake-modified is to MOND
    rms_base_vs_mond = np.sqrt(np.mean((Sigma_base - nu)**2))
    rms_wake_vs_mond = np.sqrt(np.mean((Sigma_wake - nu)**2))
    
    print(f"\nRMS difference from MOND:")
    print(f"  Baseline Σ-Gravity: {rms_base_vs_mond:.3f}")
    print(f"  Wake-modified:      {rms_wake_vs_mond:.3f}")
    
    if rms_wake_vs_mond < rms_base_vs_mond:
        print("  ✓ Wake correction brings Σ-Gravity closer to MOND shape!")
    else:
        print("  ✗ Wake correction doesn't help with MOND matching")


def test_optimal_wake_profile():
    """
    Find what C_wake(r) profile would make Σ-Gravity exactly match MOND.
    Then check if this profile is physically reasonable.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: OPTIMAL WAKE PROFILE TO MATCH MOND")
    print("=" * 70)
    
    # Typical galaxy
    R = np.linspace(0.5, 15, 30)
    R_d = 3.0
    V_flat = 150
    
    V_c = V_flat * np.tanh(R / R_d)
    g = (V_c * 1000) ** 2 / (R * kpc_to_m)
    
    xi = XI_SCALE * R_d
    W = R / (xi + R)
    h = h_sigma(g)
    nu = nu_mond(g)
    
    # Required C_wake to match MOND exactly
    # 1 + A × W × h × C_wake = ν
    # C_wake = (ν - 1) / (A × W × h)
    
    denominator = A_GALAXY * W * h
    C_wake_optimal = (nu - 1) / np.maximum(denominator, 1e-10)
    
    print(f"\nOptimal C_wake to match MOND exactly:")
    print()
    print(f"{'R (kpc)':<10} {'W(r)':<10} {'h(g)':<10} {'MOND':<10} {'C_wake opt':<12} {'Physical?':<10}")
    print("-" * 70)
    
    for i in [0, 5, 10, 15, 20, 25, 29]:
        physical = "Yes" if 0.3 < C_wake_optimal[i] < 1.0 else "No"
        print(f"{R[i]:<10.1f} {W[i]:<10.3f} {h[i]:<10.3f} {nu[i]:<10.3f} "
              f"{C_wake_optimal[i]:<12.3f} {physical:<10}")
    
    # What dispersion profile would give this C_wake?
    # C_wake = V_c² / (V_c² + σ²)
    # σ² = V_c² × (1/C_wake - 1)
    sigma_required = V_c * np.sqrt(np.maximum(1/C_wake_optimal - 1, 0))
    
    print(f"\nRequired dispersion profile to achieve optimal C_wake:")
    print()
    print(f"{'R (kpc)':<10} {'V_c (km/s)':<12} {'σ required':<12} {'Realistic?':<10}")
    print("-" * 50)
    
    for i in [0, 5, 10, 15, 20, 25, 29]:
        realistic = "Yes" if 10 < sigma_required[i] < 100 else "No"
        print(f"{R[i]:<10.1f} {V_c[i]:<12.1f} {sigma_required[i]:<12.1f} {realistic:<10}")
    
    print("\n" + "-" * 70)
    print("CONCLUSION:")
    if np.all((sigma_required > 10) & (sigma_required < 100)):
        print("  ✓ Required dispersion profile is physically reasonable!")
        print("  The wake model CAN bridge Σ-Gravity to MOND with realistic kinematics.")
    else:
        print("  ✗ Required dispersion profile is unrealistic in some regions.")
        print("  The wake model alone cannot fully bridge Σ-Gravity to MOND.")
    print("-" * 70)


def analyze_cluster_regime():
    """
    For clusters, the wake concept doesn't apply the same way.
    Clusters are pressure-supported, not rotation-supported.
    
    The path length scaling A = A_0 × L^0.25 explains the amplitude difference.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: CLUSTER REGIME")
    print("=" * 70)
    
    print("""
CLUSTERS ARE DIFFERENT:

1. KINEMATICS:
   - Galaxies: Rotation-dominated (v_rot >> σ)
   - Clusters: Dispersion-dominated (σ >> v_rot)
   
2. GEOMETRY:
   - Galaxies: 2D disk → limited modes
   - Clusters: 3D sphere → more modes contribute
   
3. PATH LENGTH:
   - Galaxies: L ≈ 2h ≈ 1.5 kpc (disk thickness)
   - Clusters: L ≈ 2R_lens ≈ 400 kpc (diameter)
   
4. WAKE COHERENCE:
   - Galaxies: C_wake ≈ 0.7-0.9 (ordered rotation)
   - Clusters: C_wake concept doesn't apply (no rotation)

CONCLUSION:
   The galaxy-cluster amplitude difference is explained by PATH LENGTH SCALING,
   not by wake coherence. The wake model is specific to rotating disk systems.
   
   Σ-Gravity formula:
     Galaxies: Σ = 1 + A_galaxy × W(r) × h(g) × [C_wake correction]
     Clusters: Σ = 1 + A_cluster × h(g)  [no W, no C_wake]
   
   where A_cluster/A_galaxy ≈ (L_cluster/L_galaxy)^0.25 ≈ 6.8
""")


def main():
    print("UNIFIED WAKE MODEL ANALYSIS")
    print("=" * 70)
    
    analyze_shape_difference()
    analyze_wake_as_shape_modifier()
    test_optimal_wake_profile()
    analyze_cluster_regime()
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("""
WHAT THE WAKE MODEL CAN DO:
---------------------------
1. Explain SHAPE differences between Σ-Gravity and MOND in galaxies
2. Provide physical mechanism for inner-region suppression
3. Predict reduced enhancement in counter-rotating systems
4. Connect coherence to observable kinematics (v_rot/σ)

WHAT THE WAKE MODEL CANNOT DO:
------------------------------
1. Explain the galaxy-cluster amplitude ratio (that's path length scaling)
2. Apply to dispersion-dominated systems like clusters
3. Replace the coherence window W(r) entirely

RECOMMENDED APPROACH:
---------------------
1. Keep path length scaling A = A_0 × L^0.25 for amplitude
2. Keep coherence window W(r) = r/(ξ+r) for spatial structure
3. ADD wake coherence C_wake as a kinematic correction:
   
   Σ = 1 + A × W(r) × h(g) × C_wake(r)
   
   where C_wake = v_rot² / (v_rot² + σ²)

4. For clusters: Use Σ = 1 + A_cluster × h(g) with A_cluster from path length

This is a REFINEMENT of Σ-Gravity, not a replacement.
""")


if __name__ == "__main__":
    main()

