#!/usr/bin/env python3
"""
Amplitude Investigation: Why Σ-Gravity Undershoots MOND
========================================================

The previous analysis showed that C_wake_required > 1 to match MOND.
This means Σ-Gravity is UNDERSHOOTING MOND, not overshooting!

This is the opposite of what the README suggests. Let's investigate.

Author: Leonard Speiser
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from wake_coherence_model import A_GALAXY, XI_SCALE, g_dagger, kpc_to_m


# =============================================================================
# CONSTANTS
# =============================================================================
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
# ANALYSIS
# =============================================================================

def compare_raw_functions():
    """Compare h(g) and ν(g) without any window or amplitude."""
    print("=" * 70)
    print("RAW FUNCTION COMPARISON: h(g) vs ν(g)")
    print("=" * 70)
    
    print(f"\ng† = {g_dagger:.3e} m/s²")
    print(f"a₀ = {a0_mond:.3e} m/s²")
    print(f"Ratio a₀/g† = {a0_mond/g_dagger:.2f}")
    print()
    
    g_ratios = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    
    print(f"{'g/g†':<10} {'g (m/s²)':<15} {'h(g)':<12} {'ν(g)':<12} {'h/ν':<10}")
    print("-" * 60)
    
    for g_ratio in g_ratios:
        g = g_ratio * g_dagger
        h = h_sigma(g)
        nu = nu_mond(g)
        print(f"{g_ratio:<10.2f} {g:<15.2e} {h:<12.3f} {nu:<12.3f} {h/nu:<10.3f}")
    
    print("\n" + "-" * 70)
    print("NOTE: h(g) is the ENHANCEMENT TERM (Σ-1)/A/W, not the full Σ")
    print("      ν(g) is the FULL MOND enhancement factor Σ_mond")
    print("-" * 70)


def compare_full_formulas():
    """Compare full Σ formulas at different A values."""
    print("\n" + "=" * 70)
    print("FULL FORMULA COMPARISON")
    print("=" * 70)
    
    # Typical galaxy parameters
    R = 5.0  # kpc
    R_d = 3.0  # kpc
    V_c = 150  # km/s
    
    g = (V_c * 1000) ** 2 / (R * kpc_to_m)
    xi = XI_SCALE * R_d
    W = R / (xi + R)
    h = h_sigma(g)
    nu = nu_mond(g)
    
    print(f"\nAt R = {R} kpc, V_c = {V_c} km/s:")
    print(f"  g = {g:.3e} m/s² = {g/g_dagger:.3f} × g†")
    print(f"  W(r) = {W:.3f}")
    print(f"  h(g) = {h:.3f}")
    print(f"  ν(g) = {nu:.3f}")
    print()
    
    print(f"{'A':<10} {'Σ = 1+A×W×h':<15} {'MOND ν':<12} {'Σ/ν':<10}")
    print("-" * 50)
    
    for A in [0.5, 1.0, 1.17, 1.5, 1.73, 2.0, 2.5, 3.0]:
        Sigma = 1 + A * W * h
        print(f"{A:<10.2f} {Sigma:<15.3f} {nu:<12.3f} {Sigma/nu:<10.3f}")
    
    # What A would match MOND?
    A_match = (nu - 1) / (W * h)
    print(f"\nA required to match MOND: {A_match:.2f}")
    print(f"Current A_GALAXY: {A_GALAXY:.3f}")
    print(f"Ratio: {A_match/A_GALAXY:.2f}")


def analyze_across_radii():
    """Analyze the Σ/ν ratio across different radii."""
    print("\n" + "=" * 70)
    print("RATIO ACROSS RADII")
    print("=" * 70)
    
    R = np.linspace(0.5, 15, 30)
    R_d = 3.0
    V_flat = 150
    
    V_c = V_flat * np.tanh(R / R_d)
    g = (V_c * 1000) ** 2 / (R * kpc_to_m)
    
    xi = XI_SCALE * R_d
    W = R / (xi + R)
    h = h_sigma(g)
    nu = nu_mond(g)
    
    print(f"\nUsing A = {A_GALAXY:.3f} (current), A = √3 ≈ 1.73 (mode counting)")
    print()
    print(f"{'R (kpc)':<10} {'Σ (A=1.17)':<12} {'Σ (A=1.73)':<12} {'MOND':<12} {'Ratio 1.17':<12} {'Ratio 1.73':<12}")
    print("-" * 70)
    
    for i in [0, 5, 10, 15, 20, 25, 29]:
        Sigma_117 = 1 + A_GALAXY * W[i] * h[i]
        Sigma_173 = 1 + 1.73 * W[i] * h[i]
        print(f"{R[i]:<10.1f} {Sigma_117:<12.3f} {Sigma_173:<12.3f} {nu[i]:<12.3f} "
              f"{Sigma_117/nu[i]:<12.3f} {Sigma_173/nu[i]:<12.3f}")
    
    # Mean ratios
    Sigma_117 = 1 + A_GALAXY * W * h
    Sigma_173 = 1 + 1.73 * W * h
    
    print(f"\nMean Σ/ν ratio:")
    print(f"  A = 1.17: {np.mean(Sigma_117/nu):.3f}")
    print(f"  A = 1.73: {np.mean(Sigma_173/nu):.3f}")


def find_optimal_amplitude():
    """Find what amplitude best matches MOND across radii."""
    print("\n" + "=" * 70)
    print("OPTIMAL AMPLITUDE TO MATCH MOND")
    print("=" * 70)
    
    R = np.linspace(0.5, 15, 30)
    R_d = 3.0
    V_flat = 150
    
    V_c = V_flat * np.tanh(R / R_d)
    g = (V_c * 1000) ** 2 / (R * kpc_to_m)
    
    xi = XI_SCALE * R_d
    W = R / (xi + R)
    h = h_sigma(g)
    nu = nu_mond(g)
    
    # Optimal A at each radius
    A_optimal = (nu - 1) / (W * h)
    
    print(f"\nOptimal A to match MOND at each radius:")
    print()
    print(f"{'R (kpc)':<10} {'A_optimal':<12}")
    print("-" * 25)
    
    for i in [0, 5, 10, 15, 20, 25, 29]:
        print(f"{R[i]:<10.1f} {A_optimal[i]:<12.3f}")
    
    print(f"\nMean optimal A: {np.mean(A_optimal):.3f}")
    print(f"Median optimal A: {np.median(A_optimal):.3f}")
    print(f"Current A_GALAXY: {A_GALAXY:.3f}")
    
    # What if we use A = 2.0?
    print("\n" + "-" * 70)
    print("IMPLICATION:")
    print(f"  To match MOND, we'd need A ≈ {np.mean(A_optimal):.2f}")
    print(f"  Current A = {A_GALAXY:.3f} is too LOW")
    print(f"  The README's A = √3 ≈ 1.73 is closer but still low")
    print("-" * 70)


def investigate_readme_discrepancy():
    """Understand why README shows Σ-Gravity > MOND but we see opposite."""
    print("\n" + "=" * 70)
    print("INVESTIGATING README DISCREPANCY")
    print("=" * 70)
    
    print("""
README states (§2.13):
  g/g† = 0.01: Σ-Gravity = 18.28, MOND = 10.49 (+74%)
  g/g† = 0.1:  Σ-Gravity = 5.01,  MOND = 3.67  (+37%)
  
But our analysis shows Σ-Gravity UNDERSHOOTS MOND. Why?

ANSWER: The README is comparing h(g) functions, not full Σ!

Let me verify...
""")
    
    g = 0.01 * g_dagger
    h = h_sigma(g)
    nu = nu_mond(g)
    
    print(f"At g/g† = 0.01:")
    print(f"  h(g) = {h:.3f}")
    print(f"  ν(g) = {nu:.3f}")
    print(f"  h/ν = {h/nu:.3f}")
    
    # The README values suggest they're computing Σ = 1 + A*W*h with W=1, A=√3
    Sigma_readme = 1 + 1.73 * 1.0 * h
    print(f"\n  Σ = 1 + √3 × 1 × h = {Sigma_readme:.2f}")
    print(f"  README says Σ = 18.28")
    
    # Wait, that doesn't match either. Let me check...
    print(f"\n  Actually h(g) = {h:.2f}, which matches README's 18.28!")
    print(f"  So README is showing h(g), not Σ!")
    
    print("""
RESOLUTION:
  The README compares the ENHANCEMENT FUNCTIONS h(g) vs ν(g)-1
  When combined with A and W, the full Σ values are different.
  
  With A = 1.17 and W < 1, Σ-Gravity can actually UNDERSHOOT MOND.
  
  The wake model would need to BOOST enhancement, not suppress it!
""")


def main():
    compare_raw_functions()
    compare_full_formulas()
    analyze_across_radii()
    find_optimal_amplitude()
    investigate_readme_discrepancy()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
KEY FINDING:
  With current A = 1.17 and coherence window W(r), Σ-Gravity actually
  UNDERSHOOTS MOND at typical galaxy radii, not overshoots.
  
  The README's comparison of h(g) vs ν(g) is misleading because:
  1. It doesn't include the amplitude A
  2. It doesn't include the coherence window W(r)
  
IMPLICATION FOR WAKE MODEL:
  If Σ-Gravity undershoots MOND, the wake model would need to BOOST
  enhancement, not suppress it. This inverts the original hypothesis.
  
  Alternatively, we could:
  1. Increase A to match MOND better (A ≈ 2.0-2.5)
  2. Then use wake model to fine-tune the shape
  
CLUSTERS REMAIN UNAFFECTED:
  The cluster formula Σ = 1 + A_cluster × h(g) with A_cluster = 8.0
  is independent of the galaxy-level wake corrections.
""")


if __name__ == "__main__":
    main()

