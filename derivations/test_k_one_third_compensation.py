#!/usr/bin/env python3
"""
Test: What needs to change if k = 1/3 instead of k = 0.24?

If we use the theoretically motivated k = 1/3, the coherence scale ξ increases:
    ξ_new = (1/3) × σ/Ω = 1.39 × ξ_old

This means W(r) transitions more slowly (coherence builds up at larger radii).
To compensate, we could:

1. Increase A (amplitude) - more enhancement at each radius
2. Modify h(g) - different acceleration dependence  
3. Change σ_eff definition - use a smaller effective dispersion
4. Adjust G (geometry factor) - changes A(G)

Let's test each option on SPARC data.

Author: Leonard Speiser
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from pathlib import Path
import json

# Physical constants
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # s⁻¹
kpc_to_m = 3.086e19
km_to_m = 1000

# Current model parameters
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 9.60e-11 m/s²

# Current parameters
K_CURRENT = 0.24
A_COEFF_CURRENT = 1.6
B_COEFF_CURRENT = 109.0
G_GALAXY_CURRENT = 0.038

# Velocity dispersions
SIGMA_GAS = 10.0
SIGMA_DISK = 25.0
SIGMA_BULGE = 120.0


def A_geometry(G, a_coeff, b_coeff):
    return np.sqrt(a_coeff + b_coeff * G**2)


def h_function(g_N):
    g_N = np.maximum(g_N, 1e-15)
    return np.sqrt(g_dagger / g_N) * g_dagger / (g_dagger + g_N)


def W_coherence(r, xi):
    xi = max(xi, 0.01)
    return 1 - np.sqrt(xi / (xi + r))


def predict_velocity(R_kpc, V_bar_kms, xi_kpc, G, a_coeff, b_coeff):
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar_kms * km_to_m
    g_N = V_bar_ms**2 / R_m
    
    A = A_geometry(G, a_coeff, b_coeff)
    W = W_coherence(R_kpc, xi_kpc)
    h = h_function(g_N)
    Sigma = 1 + A * W * h
    return V_bar_kms * np.sqrt(Sigma)


def load_sparc_galaxy(filepath):
    """Load a single SPARC rotation curve."""
    try:
        data = np.loadtxt(filepath, comments='#')
        if len(data) < 5:
            return None
        
        result = {
            'R': data[:, 0],
            'V_obs': data[:, 1],
            'e_V': data[:, 2],
            'V_gas': data[:, 3],
            'V_disk': data[:, 4],
            'V_bulge': data[:, 5] if data.shape[1] > 5 else np.zeros_like(data[:, 0])
        }
        
        result['V_bar'] = np.sqrt(
            result['V_gas']**2 + 
            0.5 * result['V_disk']**2 + 
            0.7 * result['V_bulge']**2
        )
        
        if np.any(np.isnan(result['V_bar'])) or np.max(result['V_obs']) < 10:
            return None
        
        return result
    except:
        return None


def compute_rms(galaxies, k, G, a_coeff, b_coeff, sigma_scale=1.0):
    """Compute total RMS across all galaxies."""
    total_sq_error = 0
    total_points = 0
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        
        # Estimate R_d and V at R_d
        idx_third = len(R) // 3
        R_d = R[idx_third] if idx_third > 0 else R[-1] / 2
        V_at_Rd = V_bar[idx_third] if V_bar[idx_third] > 10 else 50.0
        
        # Estimate σ_eff
        V_gas_sq = gal['V_gas']**2
        V_disk_sq = 0.5 * gal['V_disk']**2
        V_bulge_sq = 0.7 * gal['V_bulge']**2
        V_bar_sq = V_bar**2 + 1e-10
        
        gas_frac = np.mean(V_gas_sq / V_bar_sq)
        disk_frac = np.mean(V_disk_sq / V_bar_sq)
        bulge_frac = np.mean(V_bulge_sq / V_bar_sq)
        
        total = gas_frac + disk_frac + bulge_frac + 1e-10
        gas_frac /= total
        disk_frac /= total
        bulge_frac /= total
        
        sigma_eff = (gas_frac * SIGMA_GAS + disk_frac * SIGMA_DISK + bulge_frac * SIGMA_BULGE)
        sigma_eff *= sigma_scale  # Apply scaling
        
        Omega_d = V_at_Rd / R_d
        xi = k * sigma_eff / Omega_d
        
        V_pred = np.array([predict_velocity(r, vb, xi, G, a_coeff, b_coeff) 
                          for r, vb in zip(R, V_bar)])
        
        residuals = V_obs - V_pred
        total_sq_error += np.sum(residuals**2)
        total_points += len(R)
    
    return np.sqrt(total_sq_error / total_points)


def main():
    print("=" * 70)
    print("TESTING k = 1/3 WITH COMPENSATING CHANGES")
    print("=" * 70)
    
    # Load SPARC data
    sparc_dir = Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG")
    galaxy_files = list(sparc_dir.glob("*.dat"))
    
    galaxies = []
    for gf in galaxy_files:
        gal = load_sparc_galaxy(gf)
        if gal is not None:
            galaxies.append(gal)
    
    print(f"\nLoaded {len(galaxies)} galaxies")
    
    # Baseline: current parameters
    k_current = 0.24
    k_new = 1/3
    
    rms_current = compute_rms(galaxies, k_current, G_GALAXY_CURRENT, 
                               A_COEFF_CURRENT, B_COEFF_CURRENT)
    
    rms_new_no_change = compute_rms(galaxies, k_new, G_GALAXY_CURRENT,
                                     A_COEFF_CURRENT, B_COEFF_CURRENT)
    
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON")
    print(f"{'='*70}")
    print(f"Current (k=0.24): RMS = {rms_current:.2f} km/s")
    print(f"k=1/3, no changes: RMS = {rms_new_no_change:.2f} km/s")
    print(f"Degradation: {100*(rms_new_no_change/rms_current - 1):.1f}%")
    
    # =========================================================================
    # OPTION 1: Adjust A (amplitude) by changing a_coeff or b_coeff
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 1: ADJUST AMPLITUDE A(G)")
    print(f"{'='*70}")
    
    # The ratio of ξ values:
    xi_ratio = k_new / k_current  # = 1.39
    print(f"ξ ratio (new/old) = {xi_ratio:.3f}")
    
    # W(r) = 1 - (ξ/(ξ+r))^0.5
    # With larger ξ, W is smaller at each r
    # To compensate, we need larger A
    
    # At a typical radius r = 5 kpc with ξ_old = 1 kpc:
    # W_old = 1 - (1/6)^0.5 = 0.59
    # W_new = 1 - (1.39/6.39)^0.5 = 0.53
    # Ratio = 0.53/0.59 = 0.90
    # So A needs to increase by 1/0.90 = 1.11
    
    # But this is radius-dependent. Let's optimize.
    
    def objective_A(params):
        a_coeff, b_coeff = params
        if a_coeff < 0 or b_coeff < 0:
            return 1e10
        return compute_rms(galaxies, k_new, G_GALAXY_CURRENT, a_coeff, b_coeff)
    
    from scipy.optimize import minimize
    result = minimize(objective_A, [A_COEFF_CURRENT, B_COEFF_CURRENT], 
                     method='Nelder-Mead')
    a_opt, b_opt = result.x
    rms_opt_A = result.fun
    
    print(f"\nOptimal A(G) parameters with k=1/3:")
    print(f"  a_coeff: {A_COEFF_CURRENT:.2f} → {a_opt:.2f}")
    print(f"  b_coeff: {B_COEFF_CURRENT:.2f} → {b_opt:.2f}")
    print(f"  A(G=0.038): {A_geometry(G_GALAXY_CURRENT, A_COEFF_CURRENT, B_COEFF_CURRENT):.3f} → {A_geometry(G_GALAXY_CURRENT, a_opt, b_opt):.3f}")
    print(f"  RMS: {rms_opt_A:.2f} km/s (vs {rms_current:.2f} current)")
    
    # =========================================================================
    # OPTION 2: Adjust G (geometry factor)
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 2: ADJUST GEOMETRY FACTOR G")
    print(f"{'='*70}")
    
    def objective_G(G):
        if G < 0 or G > 1:
            return 1e10
        return compute_rms(galaxies, k_new, G, A_COEFF_CURRENT, B_COEFF_CURRENT)
    
    result = minimize_scalar(objective_G, bounds=(0.01, 0.5), method='bounded')
    G_opt = result.x
    rms_opt_G = result.fun
    
    print(f"\nOptimal G with k=1/3:")
    print(f"  G: {G_GALAXY_CURRENT:.4f} → {G_opt:.4f}")
    print(f"  A(G): {A_geometry(G_GALAXY_CURRENT, A_COEFF_CURRENT, B_COEFF_CURRENT):.3f} → {A_geometry(G_opt, A_COEFF_CURRENT, B_COEFF_CURRENT):.3f}")
    print(f"  RMS: {rms_opt_G:.2f} km/s (vs {rms_current:.2f} current)")
    
    # =========================================================================
    # OPTION 3: Adjust σ_eff definition (scale factor)
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 3: ADJUST σ_eff DEFINITION")
    print(f"{'='*70}")
    
    # If we scale σ_eff by a factor s, then ξ = k × s × σ/Ω
    # To keep ξ the same with k=1/3 vs k=0.24:
    # (1/3) × s × σ = 0.24 × σ
    # s = 0.24 × 3 = 0.72
    
    sigma_scale_theory = k_current / k_new
    print(f"\nTheoretical σ_eff scale to maintain same ξ: {sigma_scale_theory:.3f}")
    
    def objective_sigma(scale):
        if scale < 0.1 or scale > 2:
            return 1e10
        return compute_rms(galaxies, k_new, G_GALAXY_CURRENT, 
                          A_COEFF_CURRENT, B_COEFF_CURRENT, sigma_scale=scale)
    
    result = minimize_scalar(objective_sigma, bounds=(0.3, 1.5), method='bounded')
    sigma_scale_opt = result.x
    rms_opt_sigma = result.fun
    
    print(f"\nOptimal σ_eff scale with k=1/3:")
    print(f"  σ_eff scale: 1.0 → {sigma_scale_opt:.3f}")
    print(f"  This means σ_eff should be {100*(1-sigma_scale_opt):.1f}% smaller")
    print(f"  RMS: {rms_opt_sigma:.2f} km/s (vs {rms_current:.2f} current)")
    
    # Physical interpretation
    print(f"\n  Physical interpretation:")
    print(f"  If σ_eff = {sigma_scale_opt:.2f} × (current σ_eff), this could mean:")
    print(f"  - Using only azimuthal dispersion σ_φ instead of total σ")
    print(f"  - σ_φ/σ_total ≈ {sigma_scale_opt:.2f} (compare to 1/√3 ≈ 0.58 for flat RC)")
    
    # =========================================================================
    # OPTION 4: Combined adjustment
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 4: COMBINED ADJUSTMENT (G + σ_scale)")
    print(f"{'='*70}")
    
    def objective_combined(params):
        G, sigma_scale = params
        if G < 0.01 or G > 0.5 or sigma_scale < 0.3 or sigma_scale > 1.5:
            return 1e10
        return compute_rms(galaxies, k_new, G, A_COEFF_CURRENT, B_COEFF_CURRENT,
                          sigma_scale=sigma_scale)
    
    result = minimize(objective_combined, [G_GALAXY_CURRENT, 1.0], method='Nelder-Mead')
    G_comb, sigma_comb = result.x
    rms_comb = result.fun
    
    print(f"\nOptimal combined parameters with k=1/3:")
    print(f"  G: {G_GALAXY_CURRENT:.4f} → {G_comb:.4f}")
    print(f"  σ_eff scale: 1.0 → {sigma_comb:.3f}")
    print(f"  A(G): {A_geometry(G_GALAXY_CURRENT, A_COEFF_CURRENT, B_COEFF_CURRENT):.3f} → {A_geometry(G_comb, A_COEFF_CURRENT, B_COEFF_CURRENT):.3f}")
    print(f"  RMS: {rms_comb:.2f} km/s (vs {rms_current:.2f} current)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: OPTIONS TO USE k = 1/3")
    print(f"{'='*70}")
    
    print(f"""
Current model (k=0.24): RMS = {rms_current:.2f} km/s

If we change k to 1/3 (theoretically motivated):

| Option | Change | New RMS | Degradation |
|--------|--------|---------|-------------|
| No compensation | — | {rms_new_no_change:.2f} km/s | {100*(rms_new_no_change/rms_current - 1):+.1f}% |
| Adjust A(G) | a={a_opt:.1f}, b={b_opt:.0f} | {rms_opt_A:.2f} km/s | {100*(rms_opt_A/rms_current - 1):+.1f}% |
| Adjust G | G={G_opt:.4f} | {rms_opt_G:.2f} km/s | {100*(rms_opt_G/rms_current - 1):+.1f}% |
| Adjust σ_eff | ×{sigma_scale_opt:.2f} | {rms_opt_sigma:.2f} km/s | {100*(rms_opt_sigma/rms_current - 1):+.1f}% |
| Combined (G + σ) | G={G_comb:.4f}, σ×{sigma_comb:.2f} | {rms_comb:.2f} km/s | {100*(rms_comb/rms_current - 1):+.1f}% |

RECOMMENDED: Adjust σ_eff definition
- Scale σ_eff by {sigma_scale_opt:.2f}
- Physical interpretation: Use σ_φ (azimuthal) instead of σ_total
- This is theoretically motivated: coherence depends on ordered vs random motion
- The ratio σ_φ/σ_total ≈ 1/√3 ≈ 0.58 for flat RC, close to {sigma_scale_opt:.2f}
""")
    
    # Save results
    results = {
        'k_current': k_current,
        'k_new': k_new,
        'rms_current': rms_current,
        'rms_new_no_change': rms_new_no_change,
        'option_A': {
            'a_coeff': a_opt,
            'b_coeff': b_opt,
            'rms': rms_opt_A
        },
        'option_G': {
            'G': G_opt,
            'rms': rms_opt_G
        },
        'option_sigma': {
            'sigma_scale': sigma_scale_opt,
            'rms': rms_opt_sigma
        },
        'option_combined': {
            'G': G_comb,
            'sigma_scale': sigma_comb,
            'rms': rms_comb
        }
    }
    
    output_path = Path("/Users/leonardspeiser/Projects/sigmagravity/derivations/k_one_third_compensation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

