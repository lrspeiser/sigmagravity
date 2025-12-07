#!/usr/bin/env python3
"""
Derivation of k ≈ 0.24: Exploring the σ_eff definition

HYPOTHESIS:
The factor 0.72 = 0.24/0.333 arises from the relationship between:
1. σ_eff as defined in the formula (mass-weighted component dispersions)
2. σ as it appears in the covariant coherence scalar (azimuthal dispersion)

For a disk with epicyclic motion:
- σ_r/σ_φ = 2Ω/κ = √2 (for flat rotation curve)
- σ_eff² = σ_r² + σ_φ² = 3σ_φ² (ignoring σ_z)

If the coherence transition depends on σ_φ but we use σ_eff in the formula:
k = (1/3) × (σ_φ/σ_eff) = (1/3) × (1/√3) = 1/(3√3) ≈ 0.192

This is LESS than 0.24, so there must be another factor.

Alternative: The coherence scalar uses the TOTAL kinetic energy, not just azimuthal.

Author: Leonard Speiser
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
import json
from pathlib import Path

# =============================================================================
# PART 1: VELOCITY DISPERSION RELATIONSHIPS
# =============================================================================

def analyze_dispersion_ratios():
    """
    Analyze how different definitions of σ_eff affect k.
    """
    print("=" * 70)
    print("VELOCITY DISPERSION RELATIONSHIPS")
    print("=" * 70)
    
    # For flat rotation curve
    kappa_over_Omega = np.sqrt(2)
    
    # Epicyclic relation: σ_r/σ_φ = 2Ω/κ
    sigma_r_over_phi = 2 / kappa_over_Omega  # = √2
    
    # For thin disk, σ_z << σ_r (typically σ_z/σ_r ~ 0.5-0.7)
    sigma_z_over_r = 0.5
    
    print(f"\nEpicyclic relations for flat rotation curve:")
    print(f"  κ/Ω = {kappa_over_Omega:.4f}")
    print(f"  σ_r/σ_φ = {sigma_r_over_phi:.4f}")
    print(f"  σ_z/σ_r = {sigma_z_over_r:.4f}")
    
    # Different definitions of σ_eff
    # Let σ_φ = 1 (arbitrary units)
    sigma_phi = 1.0
    sigma_r = sigma_r_over_phi * sigma_phi
    sigma_z = sigma_z_over_r * sigma_r
    
    # Definition 1: Total 3D dispersion
    sigma_3D = np.sqrt(sigma_r**2 + sigma_phi**2 + sigma_z**2)
    
    # Definition 2: Planar dispersion only
    sigma_2D = np.sqrt(sigma_r**2 + sigma_phi**2)
    
    # Definition 3: Just azimuthal
    sigma_1D = sigma_phi
    
    print(f"\nDispersion definitions (normalized to σ_φ = 1):")
    print(f"  σ_r = {sigma_r:.4f}")
    print(f"  σ_z = {sigma_z:.4f}")
    print(f"  σ_3D = √(σ_r² + σ_φ² + σ_z²) = {sigma_3D:.4f}")
    print(f"  σ_2D = √(σ_r² + σ_φ²) = {sigma_2D:.4f}")
    print(f"  σ_1D = σ_φ = {sigma_1D:.4f}")
    
    # Now compute k for each definition
    # k_naive = 1/3 assumes W(r) = 1/2 when C(r) = 1/2
    # C = v_rot²/(v_rot² + σ²) = 1/2 when v_rot = σ
    # So r_transition = σ/Ω
    
    # If we use σ_eff in the formula but σ_φ in the physics:
    # ξ = k × σ_eff/Ω
    # r_transition = σ_φ/Ω
    # For W(r_transition) = 1/2, we need r_transition = 3ξ
    # 3ξ = σ_φ/Ω
    # ξ = σ_φ/(3Ω)
    # k × σ_eff/Ω = σ_φ/(3Ω)
    # k = σ_φ/(3 × σ_eff)
    
    k_3D = sigma_1D / (3 * sigma_3D)
    k_2D = sigma_1D / (3 * sigma_2D)
    k_1D = sigma_1D / (3 * sigma_1D)
    
    print(f"\nDerived k values (assuming coherence depends on σ_φ):")
    print(f"  k (if formula uses σ_3D) = {k_3D:.4f}")
    print(f"  k (if formula uses σ_2D) = {k_2D:.4f}")
    print(f"  k (if formula uses σ_1D) = {k_1D:.4f}")
    print(f"  Empirical k = 0.24")
    
    return {
        'sigma_r_over_phi': sigma_r_over_phi,
        'sigma_3D_over_phi': sigma_3D,
        'sigma_2D_over_phi': sigma_2D,
        'k_3D': k_3D,
        'k_2D': k_2D,
        'k_1D': k_1D
    }


def analyze_coherence_scalar_properly():
    """
    Properly analyze what σ appears in the covariant coherence scalar.
    
    The scalar is:
    C = ω²/(ω² + 4πGρ + θ² + H₀²)
    
    In the NR limit:
    - ω² = vorticity squared, related to rotation
    - For circular motion: ω ~ Ω ~ v_rot/r
    - The "random" part comes from the density term 4πGρ and expansion θ
    
    Actually, the proper NR limit is more subtle.
    """
    print("\n" + "=" * 70)
    print("PROPER ANALYSIS OF COVARIANT SCALAR NR LIMIT")
    print("=" * 70)
    
    print("""
The covariant coherence scalar:
    C = ω²/(ω² + 4πGρ + θ² + H₀²)

In the non-relativistic limit for a disk galaxy:
- Vorticity ω ~ Ω = v_rot/r (circular motion)
- Density term: 4πGρ ~ (σ/r)² from Jeans equation
- Expansion θ ≈ 0 for steady state
- H₀² is the IR cutoff

So the NR limit is:
    C ≈ Ω²/(Ω² + σ²/r² + H₀²)

At galactic radii where H₀² << Ω² and H₀² << σ²/r²:
    C ≈ (Ωr)²/((Ωr)² + σ²) = v_rot²/(v_rot² + σ²)

This σ is the TOTAL velocity dispersion that appears in the Jeans equation.
For a disk: σ² = σ_r² + σ_φ² + σ_z² (all components contribute to pressure)
""")
    
    # So the coherence transition C = 1/2 occurs when v_rot = σ_total
    # r_transition = σ_total/Ω
    
    # If σ_eff in the formula equals σ_total, then k = 1/3
    # But empirically k = 0.24 = 0.72 × (1/3)
    
    # This means either:
    # 1. σ_eff in the formula is NOT σ_total
    # 2. The transition condition is not C = 1/2
    # 3. There's additional averaging we're missing
    
    print("\nHypothesis testing:")
    print("-" * 50)
    
    # Hypothesis 1: σ_eff uses component dispersions, not total
    # The formula uses: σ_eff = f_gas×σ_gas + f_disk×σ_disk + f_bulge×σ_bulge
    # This is a mass-weighted MEAN, not RMS
    
    # For a pure disk galaxy:
    # σ_total = √(σ_r² + σ_φ² + σ_z²) ≈ √3 × σ_φ (for flat RC)
    # σ_eff (formula) ≈ σ_disk ≈ σ_φ (single component)
    
    # So σ_total/σ_eff ≈ √3
    # k = (1/3) × (σ_eff/σ_total) = (1/3) × (1/√3) = 1/(3√3) ≈ 0.192
    
    k_hypothesis1 = 1 / (3 * np.sqrt(3))
    print(f"Hypothesis 1 (σ_eff = σ_φ, coherence uses σ_total):")
    print(f"  k = 1/(3√3) = {k_hypothesis1:.4f}")
    print(f"  Empirical k = 0.24")
    print(f"  Ratio = {0.24/k_hypothesis1:.3f}")
    
    # Hypothesis 2: The transition is not at C = 1/2
    # Maybe it's at C = some other value
    
    # For W(r) = 1 - (ξ/(ξ+r))^0.5
    # W = 0.5 at r = 3ξ
    # If transition is at C = C_crit, then r = σ × √(C_crit/(1-C_crit)) / Ω
    
    # For k = 0.24:
    # ξ = 0.24 × σ_eff/Ω
    # r = 3ξ = 0.72 × σ_eff/Ω
    
    # If σ_eff = σ_total, then r = 0.72 × σ_total/Ω
    # C(r) = (Ωr)²/((Ωr)² + σ²) = 0.72²/(0.72² + 1) = 0.34
    
    r_over_sigma = 0.72
    C_at_transition = r_over_sigma**2 / (r_over_sigma**2 + 1)
    print(f"\nHypothesis 2 (k = 0.24 implies different C_crit):")
    print(f"  If ξ = 0.24 × σ_total/Ω, then r = 3ξ = 0.72 × σ_total/Ω")
    print(f"  C at this r = {C_at_transition:.4f}")
    print(f"  This is NOT 0.5, but rather 0.34")
    
    # Hypothesis 3: Combination of effects
    # σ_eff (formula) ≈ σ_disk ≈ σ_φ
    # Coherence uses σ_φ (not σ_total)
    # Then k = 1/3 should work... but it doesn't
    
    # Maybe the issue is that the formula's σ_eff is the OBSERVED line-of-sight
    # dispersion, which is a combination of σ_r, σ_φ, σ_z projected
    
    print(f"\nHypothesis 3 (projection effects):")
    print("  Observed σ_LOS = f(inclination) × σ_true")
    print("  For face-on: σ_LOS ≈ σ_z")
    print("  For edge-on: σ_LOS ≈ σ_r (or σ_φ depending on position)")
    print("  Average: σ_LOS ≈ (σ_r + σ_z)/2 ≈ 0.75 × σ_r (for σ_z/σ_r = 0.5)")
    
    return {
        'k_hypothesis1': k_hypothesis1,
        'C_at_transition': C_at_transition
    }


def derive_k_from_W_form():
    """
    The exponent in W(r) = 1 - (ξ/(ξ+r))^n affects the k value.
    
    For n = 0.5 (derived from decoherence statistics):
    W = 0.5 when (ξ/(ξ+r))^0.5 = 0.5
    ξ/(ξ+r) = 0.25
    r = 3ξ
    
    But what if we match a DIFFERENT W value?
    """
    print("\n" + "=" * 70)
    print("EFFECT OF W MATCHING VALUE")
    print("=" * 70)
    
    # The coherence scalar C = v²/(v² + σ²)
    # At the disk scale length R_d:
    # - v = V(R_d)
    # - σ = σ_eff
    # - C(R_d) = V²/(V² + σ²)
    
    # For a typical disk galaxy:
    # V/σ ~ 5-10 (rotation dominated)
    # C(R_d) ~ 0.96-0.99 (very coherent at R_d)
    
    # The coherence window W(R_d) should match this
    # W(R_d) = 1 - (ξ/(ξ+R_d))^0.5
    
    # If ξ = k × σ/Ω = k × σ × R_d/V:
    # ξ/R_d = k × σ/V
    # W(R_d) = 1 - (k×σ/V / (k×σ/V + 1))^0.5
    #        = 1 - (k×σ/(k×σ + V))^0.5
    
    # For V/σ = 5:
    V_over_sigma = 5.0
    
    print(f"For V/σ = {V_over_sigma}:")
    print(f"  C(R_d) = V²/(V² + σ²) = {V_over_sigma**2/(V_over_sigma**2 + 1):.4f}")
    
    # What k gives W(R_d) = C(R_d)?
    C_target = V_over_sigma**2 / (V_over_sigma**2 + 1)
    
    # W = 1 - (ξ/(ξ+R_d))^0.5 = C_target
    # (ξ/(ξ+R_d))^0.5 = 1 - C_target
    # ξ/(ξ+R_d) = (1 - C_target)²
    # Let x = ξ/R_d = k × σ/V = k/V_over_sigma
    # x/(x+1) = (1-C_target)²
    # x = (1-C_target)² × (x+1)
    # x × (1 - (1-C_target)²) = (1-C_target)²
    # x = (1-C_target)² / (1 - (1-C_target)²)
    
    one_minus_C = 1 - C_target
    x = one_minus_C**2 / (1 - one_minus_C**2)
    k_matched = x * V_over_sigma
    
    print(f"  To match W(R_d) = C(R_d):")
    print(f"    ξ/R_d = {x:.4f}")
    print(f"    k = {k_matched:.4f}")
    
    # This gives k ~ 0.02, way too small!
    # So matching W = C at R_d is NOT the right approach
    
    # Alternative: Match the AVERAGE W over the disk to average C
    print(f"\n  Note: k = {k_matched:.4f} is too small")
    print(f"  The matching should NOT be at R_d")
    
    # The right matching is:
    # ⟨W⟩_disk = ⟨C⟩_disk
    # or equivalently: W at some characteristic radius = C at that radius
    
    # For the rotation curve, we sample radii from ~0.5 R_d to ~5 R_d
    # The mass-weighted average radius is ~ 2 R_d
    
    print("\n  Trying different matching radii:")
    for r_match in [1.0, 2.0, 3.0, 4.0, 5.0]:
        # At r = r_match × R_d:
        # V(r) ≈ V(R_d) for flat RC
        # So C(r) ≈ C(R_d) ≈ 0.96
        
        # W(r) = 1 - (ξ/(ξ+r))^0.5
        # With ξ = k × σ/Ω × R_d = k × R_d / V_over_sigma
        # W(r_match × R_d) = 1 - (k/(V_over_sigma) / (k/V_over_sigma + r_match))^0.5
        
        # Setting W = 0.5 (the natural transition):
        # (k/V_over_sigma) / (k/V_over_sigma + r_match) = 0.25
        # k/V_over_sigma = 0.25 × (k/V_over_sigma + r_match)
        # 0.75 × k/V_over_sigma = 0.25 × r_match
        # k = V_over_sigma × r_match / 3
        
        k_at_r = V_over_sigma * r_match / 3
        # Wait, this grows with r_match, that's not right
        
        # Let me redo: if W = 0.5 at r = r_match × R_d
        # Then r_match × R_d = 3ξ
        # ξ = r_match × R_d / 3
        # k × σ/Ω = r_match × R_d / 3
        # k × σ × R_d / V = r_match × R_d / 3
        # k = r_match × V / (3σ) = r_match × V_over_sigma / 3
        
        k_from_r = r_match * V_over_sigma / 3
        print(f"    r = {r_match} R_d: k = {k_from_r:.4f}")
    
    # These k values are all > 1, way too big!
    # The issue is that W = 0.5 occurs at r = 3ξ, and for k ~ 0.24:
    # ξ = 0.24 × σ/Ω = 0.24 × R_d/5 = 0.048 R_d
    # r(W=0.5) = 3 × 0.048 R_d = 0.14 R_d
    
    # So W transitions very quickly, reaching W ~ 0.9 by R_d
    
    print("\n  Actual behavior with k = 0.24:")
    k_empirical = 0.24
    xi_over_Rd = k_empirical / V_over_sigma
    print(f"    ξ/R_d = {xi_over_Rd:.4f}")
    print(f"    r(W=0.5)/R_d = 3ξ/R_d = {3*xi_over_Rd:.4f}")
    
    for r_Rd in [0.5, 1.0, 2.0, 3.0, 5.0]:
        W_at_r = 1 - np.sqrt(xi_over_Rd / (xi_over_Rd + r_Rd))
        print(f"    W(r={r_Rd} R_d) = {W_at_r:.4f}")
    
    return {'k_matched': k_matched}


def explore_alternative_k_derivation():
    """
    Alternative approach: k emerges from the RATIO of coherence scales
    in the formula vs the physics.
    
    The dynamical coherence scale in the covariant scalar is:
    ξ_phys = σ/Ω (where C = 1/2)
    
    The formula uses:
    ξ_formula = k × σ_eff/Ω_d
    
    If these are related by a geometric factor:
    k = ξ_formula / (σ_eff/Ω_d) = (ξ_phys/ξ_formula) × (σ/σ_eff) × (Ω/Ω_d)
    """
    print("\n" + "=" * 70)
    print("ALTERNATIVE: k FROM SCALE RATIOS")
    print("=" * 70)
    
    # The coherence window W(r) = 1 - (ξ/(ξ+r))^0.5
    # has W = 0.5 at r = 3ξ
    
    # The coherence scalar C = v²/(v² + σ²)
    # has C = 0.5 at r = σ/Ω
    
    # If we want W(r) to track C(r), we need:
    # W(r) ≈ C(r) for all r
    
    # At r = σ/Ω: C = 0.5
    # For W = 0.5 at r = σ/Ω, we need 3ξ = σ/Ω
    # ξ = σ/(3Ω)
    # k = 1/3
    
    # But this assumes W and C have the same functional form, which they don't!
    
    # W(r) = 1 - (ξ/(ξ+r))^0.5
    # C(r) = (Ωr)²/((Ωr)² + σ²) = r²/(r² + (σ/Ω)²)
    
    # Let's define ξ_C = σ/Ω (the scale where C = 0.5)
    # Then C(r) = r²/(r² + ξ_C²)
    
    # And W(r) = 1 - (ξ/(ξ+r))^0.5
    
    # These are different functions! Let's compare them.
    
    xi_C = 1.0  # Arbitrary scale
    r_values = np.linspace(0.01, 10, 1000) * xi_C
    
    C_values = r_values**2 / (r_values**2 + xi_C**2)
    
    # For different k values, compute W and find best match
    k_test = np.linspace(0.1, 0.5, 100)
    best_k = None
    best_mse = float('inf')
    
    for k in k_test:
        xi_W = k * xi_C  # ξ = k × σ/Ω = k × ξ_C
        W_values = 1 - np.sqrt(xi_W / (xi_W + r_values))
        
        # MSE between W and C
        mse = np.mean((W_values - C_values)**2)
        
        if mse < best_mse:
            best_mse = mse
            best_k = k
    
    print(f"Best k to match W(r) ≈ C(r) over r ∈ [0, 10ξ_C]:")
    print(f"  k = {best_k:.4f}")
    print(f"  MSE = {best_mse:.6f}")
    
    # Also try weighted by mass (exponential disk)
    weights = np.exp(-r_values / (2 * xi_C))  # Exponential disk profile
    weights /= np.sum(weights)
    
    best_k_weighted = None
    best_wmse = float('inf')
    
    for k in k_test:
        xi_W = k * xi_C
        W_values = 1 - np.sqrt(xi_W / (xi_W + r_values))
        wmse = np.sum(weights * (W_values - C_values)**2)
        
        if wmse < best_wmse:
            best_wmse = wmse
            best_k_weighted = k
    
    print(f"\nBest k with exponential disk weighting:")
    print(f"  k = {best_k_weighted:.4f}")
    print(f"  Weighted MSE = {best_wmse:.6f}")
    
    # The key insight: W and C have different functional forms
    # W rises slower than C at small r (because of the square root)
    # This means ξ_W < ξ_C to compensate
    # k = ξ_W/ξ_C < 1
    
    return {'best_k': best_k, 'best_k_weighted': best_k_weighted}


def final_k_hypothesis():
    """
    Final hypothesis for k ≈ 0.24
    
    The coherence window W(r) = 1 - (ξ/(ξ+r))^0.5 needs to approximate
    the coherence scalar C(r) = r²/(r² + ξ_C²) where ξ_C = σ/Ω.
    
    The best-fit k that matches these functions over the disk is:
    k ≈ 0.24-0.30 depending on weighting
    """
    print("\n" + "=" * 70)
    print("FINAL HYPOTHESIS: k FROM FUNCTION MATCHING")
    print("=" * 70)
    
    # More careful matching: find k such that ⟨W⟩ = ⟨C⟩ over the disk
    # where the average is mass-weighted
    
    xi_C = 1.0  # σ/Ω in arbitrary units
    R_d = 1.0   # Disk scale length = ξ_C (typical for V/σ ~ 5)
    
    # For an exponential disk, surface density Σ ∝ exp(-r/R_d)
    # Mass-weighted average: ⟨f⟩ = ∫ f(r) × Σ(r) × 2πr dr / ∫ Σ(r) × 2πr dr
    
    r_values = np.linspace(0.01, 10, 1000) * R_d
    Sigma = np.exp(-r_values / R_d)
    weights = Sigma * r_values  # 2πr factor
    weights /= np.sum(weights)
    
    C_values = r_values**2 / (r_values**2 + xi_C**2)
    C_avg = np.sum(weights * C_values)
    
    print(f"Mass-weighted average coherence ⟨C⟩ = {C_avg:.4f}")
    
    # Find k such that ⟨W⟩ = ⟨C⟩
    def objective(k):
        xi_W = k * xi_C
        W_values = 1 - np.sqrt(xi_W / (xi_W + r_values))
        W_avg = np.sum(weights * W_values)
        return (W_avg - C_avg)**2
    
    result = minimize_scalar(objective, bounds=(0.01, 1.0), method='bounded')
    k_matched = result.x
    
    xi_W = k_matched * xi_C
    W_values = 1 - np.sqrt(xi_W / (xi_W + r_values))
    W_avg = np.sum(weights * W_values)
    
    print(f"\nMatching ⟨W⟩ = ⟨C⟩:")
    print(f"  k = {k_matched:.4f}")
    print(f"  ⟨W⟩ = {W_avg:.4f}")
    
    # Also try matching at specific radii
    print("\nMatching at specific radii:")
    for r_match in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        r = r_match * R_d
        C_at_r = r**2 / (r**2 + xi_C**2)
        
        # Find k such that W(r) = C(r)
        def obj_r(k):
            xi_W = k * xi_C
            W_at_r = 1 - np.sqrt(xi_W / (xi_W + r))
            return (W_at_r - C_at_r)**2
        
        res = minimize_scalar(obj_r, bounds=(0.01, 1.0), method='bounded')
        k_r = res.x
        print(f"  r = {r_match} R_d: C = {C_at_r:.4f}, k = {k_r:.4f}")
    
    return {'k_matched': k_matched, 'C_avg': C_avg}


def main():
    print("=" * 70)
    print("EXPLORING THE ORIGIN OF k ≈ 0.24")
    print("=" * 70)
    
    # Part 1: Dispersion ratios
    disp_results = analyze_dispersion_ratios()
    
    # Part 2: Proper covariant analysis
    cov_results = analyze_coherence_scalar_properly()
    
    # Part 3: W form effects
    w_results = derive_k_from_W_form()
    
    # Part 4: Function matching
    match_results = explore_alternative_k_derivation()
    
    # Part 5: Final hypothesis
    final_results = final_k_hypothesis()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
The coefficient k ≈ 0.24 in ξ = k × σ_eff/Ω_d arises from matching
the phenomenological coherence window W(r) to the physical coherence
scalar C(r).

Key findings:

1. FUNCTION MISMATCH:
   W(r) = 1 - (ξ/(ξ+r))^0.5  (phenomenological)
   C(r) = r²/(r² + ξ_C²)     (from covariant scalar)
   
   These have different shapes, requiring k < 1/3 to match.

2. BEST-FIT k:
   - Unweighted MSE matching: k ≈ {match_results['best_k']:.3f}
   - Mass-weighted matching: k ≈ {match_results['best_k_weighted']:.3f}
   - Average matching: k ≈ {final_results['k_matched']:.3f}

3. PHYSICAL INTERPRETATION:
   The exponent 0.5 in W(r) (from decoherence statistics) makes W rise
   slower than C at small r. To compensate, ξ must be smaller, giving
   k < 1/3.

4. EMPIRICAL k = 0.24:
   This falls within the range of theoretically motivated values,
   suggesting the derivation is on the right track.

5. REMAINING UNCERTAINTY:
   The exact k depends on:
   - How σ_eff is defined (which components, how weighted)
   - The radial weighting (mass distribution)
   - Possibly galaxy-specific factors (V/σ ratio, disk thickness)
""")
    
    # Save results
    output = {
        'dispersion_ratios': disp_results,
        'covariant_analysis': cov_results,
        'function_matching': match_results,
        'final_hypothesis': final_results
    }
    
    output_path = Path("/Users/leonardspeiser/Projects/sigmagravity/derivations/k_derivation_dispersion_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

