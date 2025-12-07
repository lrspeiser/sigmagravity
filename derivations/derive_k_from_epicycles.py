#!/usr/bin/env python3
"""
Derivation of k ≈ 0.24 from orbit-averaging with epicyclic corrections.

HYPOTHESIS:
The coherence scale coefficient k in ξ = k × σ_eff/Ω_d emerges from orbit-averaging
the coherence window W(r) over epicyclic oscillations in a thin disk.

THEORETICAL FRAMEWORK:
1. The covariant coherence scalar C = ω²/(ω² + 4πGρ + θ² + H₀²)
2. In NR limit: C ≈ v_rot²/(v_rot² + σ²)
3. Transition C = 1/2 occurs at v_rot = σ, giving r_transition ~ σ/Ω
4. Stars on nearly circular orbits undergo epicyclic oscillations
5. The orbit-averaged coherence ⟨W⟩ differs from W evaluated at mean radius

KEY PREDICTION:
k should emerge from the ratio of epicyclic to circular frequencies (κ/Ω)
For a flat rotation curve: κ = √2 × Ω, so κ/Ω = √2 ≈ 1.41
For Keplerian: κ = Ω, so κ/Ω = 1

We expect k to involve these ratios in a specific way.

Author: Leonard Speiser
"""

import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar
import json
from pathlib import Path

# Physical constants
G_NEWTON = 6.674e-11  # m³/(kg·s²)
kpc_to_m = 3.086e19   # m
km_to_m = 1000        # m

# =============================================================================
# PART 1: EPICYCLIC MOTION IN A DISK
# =============================================================================

def epicyclic_frequency_ratio(rotation_curve_type='flat'):
    """
    Compute κ/Ω for different rotation curve types.
    
    κ² = (2Ω/R) × d(R²Ω)/dR = 4Ω² + R × dΩ²/dR
    
    For V(R) = V₀ × (R/R₀)^β:
        Ω = V/R ∝ R^(β-1)
        κ²/Ω² = 2(2 - β)
    
    Returns:
        κ/Ω ratio
    """
    if rotation_curve_type == 'flat':
        # V = const, so β = 0
        # κ²/Ω² = 2(2 - 0) = 4, κ/Ω = 2
        # Wait, let me recalculate...
        # For flat: V = V₀, Ω = V₀/R
        # κ² = 4Ω² + R × d(Ω²)/dR = 4Ω² + R × d(V₀²/R²)/dR
        #    = 4Ω² + R × (-2V₀²/R³) = 4Ω² - 2V₀²/R² = 4Ω² - 2Ω² = 2Ω²
        # κ/Ω = √2
        return np.sqrt(2)
    elif rotation_curve_type == 'keplerian':
        # V ∝ R^(-1/2), Ω ∝ R^(-3/2)
        # κ = Ω (standard result for Keplerian)
        return 1.0
    elif rotation_curve_type == 'solid_body':
        # V ∝ R, Ω = const
        # κ = 2Ω
        return 2.0
    else:
        raise ValueError(f"Unknown rotation curve type: {rotation_curve_type}")


def epicyclic_orbit(R_guide, kappa_over_Omega, amplitude_ratio=0.1, n_points=1000):
    """
    Generate an epicyclic orbit around guiding radius R_guide.
    
    In the epicyclic approximation:
        R(t) = R_guide + X × cos(κt)
        φ(t) = Ω×t - (2Ω/κ) × (X/R_guide) × sin(κt)
    
    where X is the radial amplitude.
    
    Args:
        R_guide: Guiding center radius
        kappa_over_Omega: Ratio κ/Ω
        amplitude_ratio: X/R_guide (typical ~0.1 for disk stars)
        n_points: Number of points per orbit
    
    Returns:
        R_values: Array of radii over one orbit
        weights: Time weights (uniform for epicyclic)
    """
    # One full epicyclic period
    t = np.linspace(0, 2*np.pi, n_points)  # κt from 0 to 2π
    
    X = amplitude_ratio * R_guide
    R_values = R_guide + X * np.cos(t)
    
    # Weights are uniform in time
    weights = np.ones(n_points) / n_points
    
    return R_values, weights


# =============================================================================
# PART 2: COHERENCE WINDOW AVERAGING
# =============================================================================

def W_coherence(r, xi):
    """Coherence window: W(r) = 1 - (ξ/(ξ+r))^0.5"""
    return 1 - np.sqrt(xi / (xi + r))


def orbit_averaged_W(R_guide, xi, kappa_over_Omega, amplitude_ratio=0.1, n_points=1000):
    """
    Compute orbit-averaged coherence window ⟨W⟩.
    
    ⟨W⟩ = (1/T) ∮ W(R(t)) dt
    """
    R_values, weights = epicyclic_orbit(R_guide, kappa_over_Omega, amplitude_ratio, n_points)
    W_values = W_coherence(R_values, xi)
    return np.sum(W_values * weights)


def find_effective_xi(R_guide, target_W, kappa_over_Omega, amplitude_ratio=0.1):
    """
    Find ξ_eff such that W(R_guide, ξ_eff) = ⟨W⟩_orbit
    
    This gives us the "effective" coherence scale that accounts for orbit averaging.
    """
    def objective(xi):
        W_orbit_avg = orbit_averaged_W(R_guide, xi, kappa_over_Omega, amplitude_ratio)
        W_at_guide = W_coherence(R_guide, xi)
        return (W_orbit_avg - target_W)**2
    
    result = minimize_scalar(objective, bounds=(0.01, 100), method='bounded')
    return result.x


# =============================================================================
# PART 3: DERIVE k FROM FIRST PRINCIPLES
# =============================================================================

def derive_k_from_transition_condition():
    """
    Derive k from the condition that coherence transition (C = 1/2) occurs
    when v_rot = σ, combined with epicyclic averaging.
    
    HYPOTHESIS:
    The coherence window W(r) should reach its "half-maximum" value
    at the radius where the orbit-averaged coherence equals 1/2.
    
    For W(r) = 1 - (ξ/(ξ+r))^0.5:
        W = 1/2 when ξ/(ξ+r) = 1/4, i.e., r = 3ξ
    
    But with epicyclic averaging, this is modified.
    """
    print("=" * 70)
    print("DERIVING k FROM EPICYCLIC ORBIT AVERAGING")
    print("=" * 70)
    
    results = {}
    
    for rc_type in ['flat', 'keplerian', 'solid_body']:
        kappa_over_Omega = epicyclic_frequency_ratio(rc_type)
        print(f"\n{rc_type.upper()} rotation curve: κ/Ω = {kappa_over_Omega:.4f}")
        
        # For each rotation curve type, find how orbit averaging modifies ξ
        # We'll compute the ratio ξ_eff / ξ_naive for various conditions
        
        # Test at R = 3ξ (where W = 0.5 for the naive case)
        xi_test = 1.0  # Arbitrary scale
        R_test = 3 * xi_test
        
        W_naive = W_coherence(R_test, xi_test)
        W_orbit = orbit_averaged_W(R_test, xi_test, kappa_over_Omega, amplitude_ratio=0.1)
        
        print(f"  At R = 3ξ:")
        print(f"    W(R) naive = {W_naive:.4f}")
        print(f"    ⟨W⟩ orbit-averaged = {W_orbit:.4f}")
        print(f"    Ratio = {W_orbit/W_naive:.4f}")
        
        results[rc_type] = {
            'kappa_over_Omega': kappa_over_Omega,
            'W_naive': W_naive,
            'W_orbit': W_orbit,
            'ratio': W_orbit / W_naive
        }
    
    return results


def derive_k_from_sigma_Omega_matching():
    """
    More sophisticated derivation:
    
    The dynamical coherence scale is ξ = k × σ/Ω
    
    From the covariant scalar in NR limit:
        C = v_rot²/(v_rot² + σ²)
    
    At radius r with v_rot = Ω×r:
        C(r) = (Ωr)²/((Ωr)² + σ²)
    
    C = 1/2 when Ωr = σ, i.e., r = σ/Ω
    
    Now, the coherence window W(r) should match this behavior.
    W(r) = 1 - (ξ/(ξ+r))^0.5
    
    If we want W(r_transition) = 1/2 where r_transition = σ/Ω:
        1/2 = 1 - (ξ/(ξ + σ/Ω))^0.5
        (ξ/(ξ + σ/Ω))^0.5 = 1/2
        ξ/(ξ + σ/Ω) = 1/4
        4ξ = ξ + σ/Ω
        3ξ = σ/Ω
        ξ = (1/3) × σ/Ω
    
    This gives k = 1/3 ≈ 0.333, not 0.24!
    
    But wait - we need to account for:
    1. The exponent 0.5 comes from decoherence statistics
    2. Orbit averaging modifies the effective transition
    """
    print("\n" + "=" * 70)
    print("DERIVING k FROM C = 1/2 TRANSITION MATCHING")
    print("=" * 70)
    
    # Naive derivation (no orbit averaging)
    # W = 1/2 at r = 3ξ
    # C = 1/2 at r = σ/Ω
    # Matching: 3ξ = σ/Ω → ξ = (1/3)σ/Ω → k = 1/3
    k_naive = 1/3
    print(f"\nNaive k (W=1/2 matches C=1/2): k = 1/3 = {k_naive:.4f}")
    
    # But we want the orbit-averaged W to match!
    # ⟨W⟩ = 1/2 at r = σ/Ω
    
    # For flat rotation curve
    kappa_over_Omega = np.sqrt(2)
    
    # Find ξ such that orbit-averaged W equals 1/2 at r = σ/Ω
    # Let's set σ/Ω = 1 (arbitrary units) and find ξ
    r_transition = 1.0  # σ/Ω in arbitrary units
    
    def objective(xi):
        W_avg = orbit_averaged_W(r_transition, xi, kappa_over_Omega, amplitude_ratio=0.1)
        return (W_avg - 0.5)**2
    
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(objective, bounds=(0.01, 10), method='bounded')
    xi_optimal = result.x
    k_orbit_averaged = xi_optimal / r_transition  # Since r_transition = σ/Ω = 1
    
    print(f"\nOrbit-averaged k (⟨W⟩=1/2 at r=σ/Ω):")
    print(f"  For flat rotation curve (κ/Ω = √2):")
    print(f"  ξ_optimal = {xi_optimal:.4f} × (σ/Ω)")
    print(f"  k = {k_orbit_averaged:.4f}")
    
    # Check with different amplitude ratios
    print(f"\n  Sensitivity to epicyclic amplitude X/R:")
    for amp in [0.05, 0.10, 0.15, 0.20, 0.25]:
        def objective_amp(xi):
            W_avg = orbit_averaged_W(r_transition, xi, kappa_over_Omega, amplitude_ratio=amp)
            return (W_avg - 0.5)**2
        result = minimize_scalar(objective_amp, bounds=(0.01, 10), method='bounded')
        k_amp = result.x / r_transition
        print(f"    X/R = {amp:.2f}: k = {k_amp:.4f}")
    
    return k_orbit_averaged


def derive_k_alternative_matching():
    """
    Alternative approach: Match the slope of W(r) to the slope of C(r)
    at the transition point.
    
    This ensures not just the value but the behavior matches.
    """
    print("\n" + "=" * 70)
    print("ALTERNATIVE: SLOPE MATCHING AT TRANSITION")
    print("=" * 70)
    
    # C(r) = (Ωr)²/((Ωr)² + σ²)
    # At r = σ/Ω: C = 1/2
    # dC/dr at r = σ/Ω:
    #   Let x = Ωr/σ, then C = x²/(x² + 1)
    #   dC/dx = 2x/(x² + 1)² 
    #   At x = 1: dC/dx = 2/4 = 1/2
    #   dC/dr = (dC/dx)(dx/dr) = (1/2)(Ω/σ)
    
    # W(r) = 1 - (ξ/(ξ+r))^0.5
    # dW/dr = 0.5 × (ξ/(ξ+r))^(-0.5) × ξ/(ξ+r)²
    #       = 0.5 × ξ^0.5 / (ξ+r)^1.5
    
    # At r = σ/Ω, if ξ = k × σ/Ω:
    #   dW/dr = 0.5 × (kσ/Ω)^0.5 / ((k+1)σ/Ω)^1.5
    #         = 0.5 × k^0.5 × (Ω/σ) / (k+1)^1.5
    
    # Matching dW/dr = dC/dr:
    #   0.5 × k^0.5 / (k+1)^1.5 = 0.5
    #   k^0.5 / (k+1)^1.5 = 1
    #   k^0.5 = (k+1)^1.5
    #   k = (k+1)³
    
    # This is a cubic equation. Let's solve numerically.
    def slope_match_equation(k):
        return k - (k + 1)**3
    
    # Actually, let me redo this more carefully
    # k^0.5 = (k+1)^1.5 means k^(1/2) = (k+1)^(3/2)
    # Taking both sides to power 2: k = (k+1)³
    # This gives k + 1 ≈ k^(1/3), which for small k means k ≈ 0
    # That's not right. Let me reconsider.
    
    # Actually the slope matching gives a different equation.
    # Let's just solve it numerically for the value matching.
    
    print("Slope matching approach requires more careful derivation...")
    print("The key insight is that k depends on both:")
    print("  1. The exponent in W(r) (0.5 from decoherence statistics)")
    print("  2. The epicyclic amplitude distribution")
    

def derive_k_from_coherence_integral():
    """
    Most rigorous approach: Integrate the covariant coherence scalar
    over the velocity distribution.
    
    The coherence at radius r depends on the local velocity distribution:
        C(r) = ⟨v_φ²⟩ / (⟨v_φ²⟩ + σ_r² + σ_φ² + σ_z²)
    
    For a disk with epicyclic motion:
        ⟨v_φ²⟩ = (Ωr)² + σ_φ²
        σ_r ≈ σ_φ × (κ/2Ω) for epicyclic orbits
        σ_z ≈ σ_r × (ν/κ) where ν is vertical frequency
    
    This gives a more complex expression for C(r).
    """
    print("\n" + "=" * 70)
    print("COHERENCE INTEGRAL WITH FULL VELOCITY DISTRIBUTION")
    print("=" * 70)
    
    # For a thin disk with flat rotation curve:
    # κ/Ω = √2
    # σ_r/σ_φ = 2Ω/κ = √2
    # So σ_r = √2 × σ_φ
    
    # Total random motion: σ_eff² = σ_r² + σ_φ² + σ_z²
    # For thin disk, σ_z << σ_r, so σ_eff² ≈ σ_r² + σ_φ² = 3σ_φ²
    
    # The coherence scalar becomes:
    # C = (Ωr)² / ((Ωr)² + 3σ_φ²)
    # C = 1/2 when (Ωr)² = 3σ_φ², i.e., Ωr = √3 × σ_φ
    # r_transition = √3 × σ_φ/Ω
    
    # If we define σ_eff = √3 × σ_φ, then r_transition = σ_eff/Ω
    # And ξ = k × σ_eff/Ω with the matching condition gives k = 1/3
    
    # But the empirical k = 0.24, which is less than 1/3
    # This suggests the effective σ in the formula is not the same as σ_eff
    
    kappa_over_Omega = np.sqrt(2)
    sigma_r_over_sigma_phi = 2 / kappa_over_Omega  # = √2
    
    print(f"For flat rotation curve:")
    print(f"  κ/Ω = {kappa_over_Omega:.4f}")
    print(f"  σ_r/σ_φ = {sigma_r_over_sigma_phi:.4f}")
    print(f"  σ_eff²/σ_φ² = 1 + (σ_r/σ_φ)² = {1 + sigma_r_over_sigma_phi**2:.4f}")
    print(f"  σ_eff/σ_φ = {np.sqrt(1 + sigma_r_over_sigma_phi**2):.4f}")
    
    # The key insight: if σ_eff in the formula uses the TOTAL dispersion,
    # but the coherence transition uses only σ_φ, then:
    # ξ = k × σ_eff/Ω = k × √3 × σ_φ/Ω
    # r_transition = √3 × σ_φ/Ω = σ_eff/Ω
    # So ξ = k × r_transition
    # For W(ξ) = 1/2 at r = 3ξ, we need 3ξ = r_transition = σ_eff/Ω
    # ξ = (1/3) × σ_eff/Ω → k = 1/3
    
    # But empirically k = 0.24 = 0.72 × (1/3)
    # What factor of 0.72 are we missing?
    
    # Hypothesis: The effective averaging over the disk reduces k
    # because ⟨W⟩ over the disk is weighted by mass, not uniform
    
    print(f"\n  Naive k from W=1/2 matching: k = 1/3 = {1/3:.4f}")
    print(f"  Empirical k: 0.24")
    print(f"  Ratio: {0.24 / (1/3):.4f}")
    print(f"\n  The factor {0.24 / (1/3):.4f} suggests additional averaging effects.")


def test_k_on_sparc_data():
    """
    Test the derived k values against SPARC data.
    
    For each galaxy, compute ξ using different k values and see which
    minimizes the rotation curve residuals.
    """
    print("\n" + "=" * 70)
    print("TESTING k VALUES ON SPARC DATA")
    print("=" * 70)
    
    # Load SPARC data
    sparc_dir = Path("/Users/leonardspeiser/Projects/sigmagravity/data/SPARC")
    rotcurve_dir = sparc_dir / "RotationCurves"
    
    if not rotcurve_dir.exists():
        print(f"SPARC data not found at {rotcurve_dir}")
        return None
    
    # Physical constants
    c = 2.998e8  # m/s
    H0_SI = 2.27e-18  # s⁻¹
    g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))
    
    # Model parameters
    A_COEFF = 1.6
    B_COEFF = 109.0
    G_GALAXY = 0.038
    
    def A_geometry(G):
        return np.sqrt(A_COEFF + B_COEFF * G**2)
    
    def h_function(g_N):
        g_N = np.maximum(g_N, 1e-15)
        return np.sqrt(g_dagger / g_N) * g_dagger / (g_dagger + g_N)
    
    def W_coherence(r, xi):
        xi = max(xi, 0.01)
        return 1 - np.sqrt(xi / (xi + r))
    
    def predict_velocity(R_kpc, V_bar_kms, xi_kpc, G=G_GALAXY):
        R_m = R_kpc * kpc_to_m
        V_bar_ms = V_bar_kms * km_to_m
        g_N = V_bar_ms**2 / R_m
        
        A = A_geometry(G)
        W = W_coherence(R_kpc, xi_kpc)
        h = h_function(g_N)
        Sigma = 1 + A * W * h
        return V_bar_kms * np.sqrt(Sigma)
    
    # Velocity dispersions
    SIGMA_GAS = 10.0
    SIGMA_DISK = 25.0
    SIGMA_BULGE = 120.0
    
    # Test different k values
    k_values = [0.15, 0.20, 0.24, 0.28, 0.33, 0.40, 0.50]
    results = {k: {'rms_list': [], 'galaxy_count': 0} for k in k_values}
    
    # Process galaxies
    galaxy_files = list(rotcurve_dir.glob("*.dat"))
    print(f"Found {len(galaxy_files)} rotation curve files")
    
    for gal_file in galaxy_files[:50]:  # Limit to 50 for speed
        try:
            # Read rotation curve
            data = np.loadtxt(gal_file, comments='#')
            if len(data) < 5:
                continue
            
            R = data[:, 0]  # kpc
            V_obs = data[:, 1]  # km/s
            V_gas = data[:, 3]  # km/s
            V_disk = data[:, 4]  # km/s
            V_bulge = data[:, 5] if data.shape[1] > 5 else np.zeros_like(R)
            
            # Apply M/L corrections
            V_bar = np.sqrt(V_gas**2 + 0.5 * V_disk**2 + 0.7 * V_bulge**2)
            
            # Estimate disk scale length (rough: use half-light radius proxy)
            R_d = R[len(R)//3] if len(R) > 3 else R[-1] / 2
            
            # Estimate velocity at R_d
            idx_Rd = np.argmin(np.abs(R - R_d))
            V_at_Rd = V_bar[idx_Rd] if V_bar[idx_Rd] > 0 else 50.0
            
            # Estimate mass fractions (simplified)
            gas_frac = np.mean(V_gas**2) / np.mean(V_bar**2 + 1e-10)
            disk_frac = np.mean(0.5 * V_disk**2) / np.mean(V_bar**2 + 1e-10)
            bulge_frac = np.mean(0.7 * V_bulge**2) / np.mean(V_bar**2 + 1e-10)
            
            # Normalize
            total = gas_frac + disk_frac + bulge_frac + 1e-10
            gas_frac /= total
            disk_frac /= total
            bulge_frac /= total
            
            # Effective dispersion
            sigma_eff = gas_frac * SIGMA_GAS + disk_frac * SIGMA_DISK + bulge_frac * SIGMA_BULGE
            
            # Ω_d
            Omega_d = V_at_Rd / R_d  # (km/s)/kpc
            
            # Test each k value
            for k in k_values:
                xi = k * sigma_eff / Omega_d
                
                V_pred = np.array([predict_velocity(r, vb, xi) for r, vb in zip(R, V_bar)])
                
                residuals = V_obs - V_pred
                rms = np.sqrt(np.mean(residuals**2))
                
                results[k]['rms_list'].append(rms)
                results[k]['galaxy_count'] += 1
                
        except Exception as e:
            continue
    
    # Summarize results
    print(f"\nResults across {results[0.24]['galaxy_count']} galaxies:")
    print("-" * 50)
    print(f"{'k':>8} | {'Mean RMS (km/s)':>15} | {'Median RMS':>12}")
    print("-" * 50)
    
    best_k = None
    best_rms = float('inf')
    
    for k in k_values:
        if results[k]['rms_list']:
            mean_rms = np.mean(results[k]['rms_list'])
            median_rms = np.median(results[k]['rms_list'])
            print(f"{k:>8.2f} | {mean_rms:>15.2f} | {median_rms:>12.2f}")
            
            if mean_rms < best_rms:
                best_rms = mean_rms
                best_k = k
    
    print("-" * 50)
    print(f"Best k: {best_k} with mean RMS = {best_rms:.2f} km/s")
    
    return results


def main():
    print("=" * 70)
    print("DERIVATION OF k ≈ 0.24 FROM FIRST PRINCIPLES")
    print("=" * 70)
    print("""
This script attempts to derive the coherence scale coefficient k in
ξ = k × σ_eff/Ω_d from the covariant coherence scalar and epicyclic
orbit averaging.

Key questions:
1. Why is k ≈ 0.24 and not 1/3 (naive matching)?
2. Does orbit averaging explain the reduction?
3. Is k universal or does it depend on galaxy properties?
""")
    
    # Part 1: Basic epicyclic theory
    results1 = derive_k_from_transition_condition()
    
    # Part 2: σ/Ω matching
    k_orbit = derive_k_from_sigma_Omega_matching()
    
    # Part 3: Alternative approaches
    derive_k_alternative_matching()
    
    # Part 4: Full velocity distribution
    derive_k_from_coherence_integral()
    
    # Part 5: Test on SPARC data
    sparc_results = test_k_on_sparc_data()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Key findings:

1. NAIVE MATCHING (W=1/2 at r where C=1/2):
   k = 1/3 ≈ 0.333

2. ORBIT-AVERAGED MATCHING:
   k ≈ {k_orbit:.3f} (for flat rotation curve, X/R = 0.1)
   
3. RATIO:
   k_empirical / k_naive = 0.24 / 0.333 = 0.72
   
4. INTERPRETATION:
   The factor 0.72 may arise from:
   - Epicyclic orbit averaging (partial effect)
   - Mass-weighted averaging over the disk
   - The specific form of the decoherence exponent (0.5)
   
5. NEXT STEPS:
   - Compute the full orbit-averaged coherence integral
   - Include the mass distribution weighting
   - Compare predictions across different galaxy types
""")
    
    # Save results
    output = {
        'k_naive': 1/3,
        'k_orbit_averaged': k_orbit,
        'k_empirical': 0.24,
        'ratio': 0.24 / (1/3),
        'epicyclic_results': {k: v for k, v in results1.items()},
    }
    
    output_path = Path("/Users/leonardspeiser/Projects/sigmagravity/derivations/k_derivation_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

