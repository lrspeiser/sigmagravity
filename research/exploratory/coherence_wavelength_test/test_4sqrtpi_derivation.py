#!/usr/bin/env python3
"""
Test: Can we derive the factor 4√π from coherence physics?

This is an EXPLORATORY test - not incorporated into main theory.
We're checking if the geometric arguments in the hypothesis actually work out.

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')

# Physical constants
c = 2.998e8          # m/s
H0 = 70 * 1000 / 3.086e22  # 1/s (70 km/s/Mpc)
cH0 = c * H0

print("=" * 80)
print("TESTING THE 4√π DERIVATION HYPOTHESIS")
print("=" * 80)
print("\nThis is an exploratory test - checking if the math works out.")
print("NOT incorporated into main Σ-Gravity theory.\n")

# =============================================================================
# PART 1: What is 4√π and related factors?
# =============================================================================

print("=" * 80)
print("PART 1: GEOMETRIC FACTORS")
print("=" * 80)

factors = {
    "4√π": 4 * np.sqrt(np.pi),
    "2√(4π)": 2 * np.sqrt(4 * np.pi),
    "4π × √π / π": 4 * np.pi * np.sqrt(np.pi) / np.pi,
    "4/√π": 4 / np.sqrt(np.pi),
    "√(4π)": np.sqrt(4 * np.pi),
    "2e": 2 * np.e,
    "2π": 2 * np.pi,
    "π": np.pi,
}

print(f"\n{'Expression':<20} {'Value':<15} {'= 4√π?':<10}")
print("-" * 50)
for name, val in factors.items():
    match = "✓" if abs(val - 4*np.sqrt(np.pi)) < 0.01 else ""
    print(f"{name:<20} {val:<15.4f} {match:<10}")

print(f"\nTarget: 4√π = {4*np.sqrt(np.pi):.6f}")
print(f"Note: 4π × √π / π = 4√π ✓ (this is the key identity)")

# =============================================================================
# PART 2: Test the Gaussian integral derivation
# =============================================================================

print("\n" + "=" * 80)
print("PART 2: GAUSSIAN INTEGRAL DERIVATION")
print("=" * 80)

print("""
The hypothesis claims:
  g† = cH₀/(4√π) arises from:
  
  1. Solid angle integration: 4π
  2. 1D Gaussian integral: √π  (from ∫exp(-r²/σ²)dr = √π × σ)
  3. Area normalization: 1/π
  
  Combined: (4π × √π) / π = 4√π
""")

# Test the integrals
sigma = 1.0  # Normalized coherence radius

# 1D Gaussian integral: ∫exp(-x²/σ²)dx from -∞ to ∞
def gaussian_1d(x, sigma):
    return np.exp(-x**2 / sigma**2)

integral_1d, _ = integrate.quad(gaussian_1d, -np.inf, np.inf, args=(sigma,))
expected_1d = np.sqrt(np.pi) * sigma

print(f"1D Gaussian integral ∫exp(-r²/σ²)dr:")
print(f"  Numerical: {integral_1d:.6f}")
print(f"  Expected (√π × σ): {expected_1d:.6f}")
print(f"  Match: {'✓' if abs(integral_1d - expected_1d) < 0.001 else '✗'}")

# 3D Gaussian integral with spherical coordinates: ∫exp(-r²/σ²) × 4πr² dr
def gaussian_3d_radial(r, sigma):
    return np.exp(-r**2 / sigma**2) * 4 * np.pi * r**2

integral_3d, _ = integrate.quad(gaussian_3d_radial, 0, np.inf, args=(sigma,))
expected_3d = np.pi**(3/2) * sigma**3

print(f"\n3D Gaussian integral ∫exp(-r²/σ²) × 4πr² dr:")
print(f"  Numerical: {integral_3d:.6f}")
print(f"  Expected (π^(3/2) × σ³): {expected_3d:.6f}")
print(f"  Match: {'✓' if abs(integral_3d - expected_3d) < 0.001 else '✗'}")

# =============================================================================
# PART 3: Test the proposed derivation step by step
# =============================================================================

print("\n" + "=" * 80)
print("PART 3: STEP-BY-STEP DERIVATION TEST")
print("=" * 80)

print("""
Proposed derivation from the hypothesis:

Step 1: Coherent field amplitude integrated over solid angle
  g_coh = g₀ × exp(-r²/σ²) × 4π

Step 2: Radial integration with Gaussian weighting
  G_coh = ∫ g_coh(r) dr = g₀ × 4π × √π × σ

Step 3: Normalize per unit transverse area (πσ²)
  Ḡ_coh = G_coh / (πσ²) = g₀ × 4π × √π × σ / (πσ²) = g₀ × 4√π / σ

Step 4: Critical acceleration
  g† = cH₀ / (4√π)  when σ = c/H₀
""")

# Verify step by step
g0 = 1.0  # Normalized amplitude
sigma = 1.0  # Normalized coherence radius

# Step 1: At r = 0, the angular integral gives 4π
step1 = g0 * np.exp(0) * 4 * np.pi
print(f"Step 1 (at r=0): g_coh = g₀ × 4π = {step1:.4f}")

# Step 2: Radial integral
step2 = g0 * 4 * np.pi * np.sqrt(np.pi) * sigma
print(f"Step 2: G_coh = g₀ × 4π × √π × σ = {step2:.4f}")

# Step 3: Normalize by area
step3 = step2 / (np.pi * sigma**2)
print(f"Step 3: Ḡ_coh = G_coh / (πσ²) = {step3:.4f}")

# Check if step 3 = g₀ × 4√π / σ
expected_step3 = g0 * 4 * np.sqrt(np.pi) / sigma
print(f"Expected: g₀ × 4√π / σ = {expected_step3:.4f}")
print(f"Match: {'✓' if abs(step3 - expected_step3) < 0.001 else '✗'}")

# =============================================================================
# PART 4: Does this give the right g†?
# =============================================================================

print("\n" + "=" * 80)
print("PART 4: CRITICAL ACCELERATION CALCULATION")
print("=" * 80)

# If σ = c/H₀ (Hubble radius), what is g†?
sigma_hubble = c / H0
print(f"\nCoherence radius σ = c/H₀ = {sigma_hubble:.3e} m")

# From the derivation: g† = cH₀ / (4√π)
g_dagger_derived = cH0 / (4 * np.sqrt(np.pi))
print(f"\nDerived g† = cH₀/(4√π) = {g_dagger_derived:.4e} m/s²")

# Compare to MOND a₀
a0_mond = 1.2e-10
print(f"MOND a₀ = {a0_mond:.4e} m/s²")
print(f"Ratio g†/a₀ = {g_dagger_derived/a0_mond:.3f}")

# =============================================================================
# PART 5: Test alternative derivations
# =============================================================================

print("\n" + "=" * 80)
print("PART 5: ALTERNATIVE DERIVATION ATTEMPTS")
print("=" * 80)

print("\nTrying different geometric combinations to get 4√π:")

attempts = [
    ("4π × √π / π", 4*np.pi * np.sqrt(np.pi) / np.pi),
    ("2 × √(4π)", 2 * np.sqrt(4*np.pi)),
    ("4 × √π", 4 * np.sqrt(np.pi)),
    ("√(16π)", np.sqrt(16*np.pi)),
    ("2π / √π", 2*np.pi / np.sqrt(np.pi)),
    ("4π / √(4π)", 4*np.pi / np.sqrt(4*np.pi)),
    ("(4π)^(3/4)", (4*np.pi)**(3/4)),
    ("π × 4^(1/2) × π^(-1/2)", np.pi * 4**(1/2) * np.pi**(-1/2)),
]

target = 4 * np.sqrt(np.pi)
print(f"\nTarget: 4√π = {target:.6f}\n")
print(f"{'Expression':<25} {'Value':<15} {'Match':<10}")
print("-" * 50)

for name, val in attempts:
    match = "✓ EXACT" if abs(val - target) < 1e-10 else ""
    print(f"{name:<25} {val:<15.6f} {match:<10}")

# =============================================================================
# PART 6: The Fresnel zone argument
# =============================================================================

print("\n" + "=" * 80)
print("PART 6: FRESNEL ZONE ARGUMENT")
print("=" * 80)

print("""
The Fresnel zone radius at distance L with wavelength λ is:
  r_F = √(λL)

For coherent enhancement across a galaxy of radius R:
  R ~ √(λ × c/H₀)
  
Solving for λ:
  λ ~ R² × H₀/c
""")

R_galaxy = 10e3 * 3.086e19  # 10 kpc in meters
lambda_fresnel = R_galaxy**2 * H0 / c
f_fresnel = c / lambda_fresnel
period_fresnel = 1 / f_fresnel

print(f"For R = 10 kpc:")
print(f"  λ = R² × H₀/c = {lambda_fresnel:.3e} m = {lambda_fresnel/3.086e19:.3e} kpc")
print(f"  f = c/λ = {f_fresnel:.3e} Hz")
print(f"  Period = {period_fresnel:.3e} s = {period_fresnel/(3600*24):.1f} days")

# =============================================================================
# PART 7: Mode density argument
# =============================================================================

print("\n" + "=" * 80)
print("PART 7: MODE DENSITY ARGUMENT")
print("=" * 80)

print("""
Hypothesis: The critical scale is where the number of Gaussian modes = 1

For a Gaussian volume:
  N_modes = π^(3/2) × σ³ / λ³ = 1
  
Solving:
  λ_crit = π^(1/2) × σ = √π × σ
  k_crit = 2π / λ_crit = 2√π / σ
""")

sigma_norm = 1.0
lambda_crit = np.sqrt(np.pi) * sigma_norm
k_crit = 2 * np.pi / lambda_crit

print(f"For σ = 1 (normalized):")
print(f"  λ_crit = √π × σ = {lambda_crit:.4f}")
print(f"  k_crit = 2π/λ = 2√π/σ = {k_crit:.4f}")
print(f"  Expected 2√π = {2*np.sqrt(np.pi):.4f}")
print(f"  Match: {'✓' if abs(k_crit - 2*np.sqrt(np.pi)) < 0.001 else '✗'}")

# =============================================================================
# PART 8: Summary and verdict
# =============================================================================

print("\n" + "=" * 80)
print("PART 8: SUMMARY AND VERDICT")
print("=" * 80)

print("""
WHAT THE HYPOTHESIS CLAIMS:
  g† = cH₀/(4√π) arises from Gaussian coherence geometry:
  - Solid angle: 4π
  - 1D Gaussian: √π
  - Area normalization: 1/π
  - Combined: (4π × √π)/π = 4√π

MATHEMATICAL CHECK:
""")

# Check the key identity
identity_lhs = 4 * np.pi * np.sqrt(np.pi) / np.pi
identity_rhs = 4 * np.sqrt(np.pi)
print(f"  (4π × √π)/π = {identity_lhs:.6f}")
print(f"  4√π = {identity_rhs:.6f}")
print(f"  Identity holds: {'✓' if abs(identity_lhs - identity_rhs) < 1e-10 else '✗'}")

print("""
PHYSICAL PLAUSIBILITY:
""")

issues = [
    "1. The 'transverse area normalization by πσ²' is ad hoc - why this specific normalization?",
    "2. The derivation assumes a specific Gaussian profile exp(-r²/σ²) without justification",
    "3. Why would gravitational coherence have a Gaussian radial profile?",
    "4. The 'coherence radius = c/H₀' assumption is not derived",
    "5. No mechanism for how coherence actually works in gravity",
    "6. Standard QFT gives 10⁻⁷⁰ corrections, not O(1)",
]

print("  ISSUES:")
for issue in issues:
    print(f"    {issue}")

print("""
VERDICT:
  The identity (4π × √π)/π = 4√π is MATHEMATICALLY CORRECT.
  
  However, the PHYSICAL DERIVATION is problematic:
  - It's a post-hoc construction to match the desired result
  - The steps (solid angle, Gaussian, area normalization) are chosen to give 4√π
  - There's no independent physical justification for each step
  - This is "numerology dressed as derivation"
  
  CONCLUSION: The math works, but the physics is not established.
  
  This is similar to noting that α ≈ 1/137 ≈ π/(2 × 69) - true but not meaningful.
""")

# =============================================================================
# PART 9: What would make this a real derivation?
# =============================================================================

print("\n" + "=" * 80)
print("PART 9: WHAT WOULD MAKE THIS A REAL DERIVATION?")
print("=" * 80)

print("""
For the 4√π factor to be physically meaningful, we would need:

1. INDEPENDENT DERIVATION of the Gaussian profile
   - Why exp(-r²/σ²) and not exp(-r/σ) or 1/(1+r²/σ²)?
   - What physics produces this specific shape?

2. PHYSICAL ORIGIN of the area normalization
   - Why normalize by πσ² specifically?
   - What does "per unit transverse area" mean for gravity?

3. CONNECTION TO KNOWN PHYSICS
   - How does this relate to the Einstein equations?
   - Where does the coherence come from in QFT?
   - Why doesn't standard quantum gravity give this?

4. TESTABLE PREDICTIONS
   - If the profile is Gaussian, what observational signatures follow?
   - How would we distinguish this from other functional forms?

WITHOUT THESE, the derivation is just a mathematical construction
that happens to give the right answer, not a physical explanation.
""")

print("=" * 80)
print("END OF EXPLORATORY TEST")
print("=" * 80)

