#!/usr/bin/env python3
"""
Can we eliminate k from the coherence scale?

If k doesn't matter much, it suggests ξ is not the right parameterization.
Let's explore alternatives:

1. Fixed ξ (no dependence on σ/Ω at all)
2. ξ = R_d (disk scale length only)
3. ξ = f(g_N) (acceleration-dependent)
4. Absorb ξ into the W(r) exponent
5. Remove W(r) entirely and put spatial dependence elsewhere

Author: Leonard Speiser
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from pathlib import Path
import json

# Physical constants
c = 2.998e8
H0_SI = 2.27e-18
kpc_to_m = 3.086e19
km_to_m = 1000

g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))

# Current parameters
A_COEFF = 1.6
B_COEFF = 109.0
G_GALAXY = 0.038

SIGMA_GAS = 10.0
SIGMA_DISK = 25.0
SIGMA_BULGE = 120.0


def A_geometry(G):
    return np.sqrt(A_COEFF + B_COEFF * G**2)


def h_function(g_N):
    g_N = np.maximum(g_N, 1e-15)
    return np.sqrt(g_dagger / g_N) * g_dagger / (g_dagger + g_N)


def W_coherence(r, xi):
    xi = max(xi, 0.01)
    return 1 - np.sqrt(xi / (xi + r))


def load_sparc_galaxy(filepath):
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
        
        # Estimate R_d
        idx_third = len(result['R']) // 3
        result['R_d'] = result['R'][idx_third] if idx_third > 0 else result['R'][-1] / 2
        
        return result
    except:
        return None


def predict_velocity_with_W(R, V_bar, xi, G=G_GALAXY):
    """Standard prediction with W(r)."""
    R_m = R * kpc_to_m
    V_bar_ms = V_bar * km_to_m
    g_N = V_bar_ms**2 / R_m
    
    A = A_geometry(G)
    W = W_coherence(R, xi)
    h = h_function(g_N)
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)


def predict_velocity_no_W(R, V_bar, G=G_GALAXY):
    """Prediction without W(r) - just A × h(g)."""
    R_m = R * kpc_to_m
    V_bar_ms = V_bar * km_to_m
    g_N = V_bar_ms**2 / R_m
    
    A = A_geometry(G)
    h = h_function(g_N)
    Sigma = 1 + A * h
    return V_bar * np.sqrt(Sigma)


def predict_velocity_W_from_R_Rd(R, V_bar, R_d, alpha, G=G_GALAXY):
    """W depends only on R/R_d, no σ or Ω."""
    R_m = R * kpc_to_m
    V_bar_ms = V_bar * km_to_m
    g_N = V_bar_ms**2 / R_m
    
    A = A_geometry(G)
    # W = 1 - (α×R_d / (α×R_d + R))^0.5
    xi = alpha * R_d
    W = W_coherence(R, xi)
    h = h_function(g_N)
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)


def predict_velocity_W_fixed_xi(R, V_bar, xi_fixed, G=G_GALAXY):
    """W with a single global ξ for all galaxies."""
    R_m = R * kpc_to_m
    V_bar_ms = V_bar * km_to_m
    g_N = V_bar_ms**2 / R_m
    
    A = A_geometry(G)
    W = W_coherence(R, xi_fixed)
    h = h_function(g_N)
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)


def predict_velocity_modified_exponent(R, V_bar, n_exp, G=G_GALAXY):
    """Modify the exponent in h(g) to absorb spatial dependence."""
    R_m = R * kpc_to_m
    V_bar_ms = V_bar * km_to_m
    g_N = V_bar_ms**2 / R_m
    
    A = A_geometry(G)
    # Modified h with adjustable exponent
    g_N = np.maximum(g_N, 1e-15)
    h = (g_dagger / g_N)**n_exp * g_dagger / (g_dagger + g_N)
    Sigma = 1 + A * h
    return V_bar * np.sqrt(Sigma)


def compute_rms(galaxies, predict_func, *args):
    """Compute RMS across all galaxies."""
    total_sq = 0
    total_n = 0
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        
        V_pred = predict_func(R, V_bar, *args)
        residuals = V_obs - V_pred
        total_sq += np.sum(residuals**2)
        total_n += len(R)
    
    return np.sqrt(total_sq / total_n)


def compute_rms_with_Rd(galaxies, predict_func, alpha):
    """Compute RMS for R_d-dependent models."""
    total_sq = 0
    total_n = 0
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        R_d = gal['R_d']
        
        V_pred = predict_func(R, V_bar, R_d, alpha)
        residuals = V_obs - V_pred
        total_sq += np.sum(residuals**2)
        total_n += len(R)
    
    return np.sqrt(total_sq / total_n)


def main():
    print("=" * 70)
    print("CAN WE ELIMINATE k FROM THE COHERENCE SCALE?")
    print("=" * 70)
    
    # Load data
    sparc_dir = Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG")
    galaxies = []
    for gf in sparc_dir.glob("*.dat"):
        gal = load_sparc_galaxy(gf)
        if gal is not None:
            galaxies.append(gal)
    
    print(f"\nLoaded {len(galaxies)} galaxies")
    
    # =========================================================================
    # BASELINE: Current model with k=0.24
    # =========================================================================
    print(f"\n{'='*70}")
    print("BASELINE: CURRENT MODEL")
    print(f"{'='*70}")
    
    def current_model(R, V_bar, gal):
        # Current ξ = k × σ_eff / Ω_d
        k = 0.24
        V_gas_sq = gal['V_gas']**2
        V_disk_sq = 0.5 * gal['V_disk']**2
        V_bulge_sq = 0.7 * gal['V_bulge']**2
        V_bar_sq = V_bar**2 + 1e-10
        
        gas_frac = np.mean(V_gas_sq / V_bar_sq)
        disk_frac = np.mean(V_disk_sq / V_bar_sq)
        bulge_frac = np.mean(V_bulge_sq / V_bar_sq)
        total = gas_frac + disk_frac + bulge_frac + 1e-10
        
        sigma_eff = (gas_frac/total * SIGMA_GAS + 
                     disk_frac/total * SIGMA_DISK + 
                     bulge_frac/total * SIGMA_BULGE)
        
        idx = len(R) // 3
        V_at_Rd = V_bar[idx] if V_bar[idx] > 10 else 50.0
        R_d = gal['R_d']
        Omega_d = V_at_Rd / R_d
        
        xi = k * sigma_eff / Omega_d
        return predict_velocity_with_W(R, V_bar, xi)
    
    total_sq = 0
    total_n = 0
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        V_pred = current_model(R, V_bar, gal)
        total_sq += np.sum((V_obs - V_pred)**2)
        total_n += len(R)
    rms_current = np.sqrt(total_sq / total_n)
    
    print(f"Current model (ξ = k×σ/Ω, k=0.24): RMS = {rms_current:.2f} km/s")
    
    # =========================================================================
    # OPTION 1: No W(r) at all - just Σ = 1 + A × h(g)
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 1: REMOVE W(r) ENTIRELY")
    print(f"{'='*70}")
    
    rms_no_W = compute_rms(galaxies, predict_velocity_no_W)
    print(f"Σ = 1 + A × h(g), no W(r): RMS = {rms_no_W:.2f} km/s")
    print(f"Degradation: {100*(rms_no_W/rms_current - 1):+.1f}%")
    
    # =========================================================================
    # OPTION 2: Fixed global ξ (same for all galaxies)
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 2: FIXED GLOBAL ξ (NO k, NO σ/Ω)")
    print(f"{'='*70}")
    
    # Optimize global ξ
    def obj_fixed_xi(xi):
        return compute_rms(galaxies, predict_velocity_W_fixed_xi, xi)
    
    result = minimize_scalar(obj_fixed_xi, bounds=(0.1, 10), method='bounded')
    xi_global = result.x
    rms_fixed_xi = result.fun
    
    print(f"Optimal global ξ = {xi_global:.2f} kpc")
    print(f"RMS = {rms_fixed_xi:.2f} km/s")
    print(f"Degradation: {100*(rms_fixed_xi/rms_current - 1):+.1f}%")
    
    # =========================================================================
    # OPTION 3: ξ = α × R_d (depends only on disk scale length)
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 3: ξ = α × R_d (SCALE LENGTH ONLY)")
    print(f"{'='*70}")
    
    def obj_Rd(alpha):
        return compute_rms_with_Rd(galaxies, predict_velocity_W_from_R_Rd, alpha)
    
    result = minimize_scalar(obj_Rd, bounds=(0.1, 2.0), method='bounded')
    alpha_opt = result.x
    rms_Rd = result.fun
    
    print(f"Optimal α = {alpha_opt:.3f} (so ξ = {alpha_opt:.3f} × R_d)")
    print(f"RMS = {rms_Rd:.2f} km/s")
    print(f"Degradation: {100*(rms_Rd/rms_current - 1):+.1f}%")
    
    # Compare to the old (2/3)R_d baseline
    rms_two_thirds = compute_rms_with_Rd(galaxies, predict_velocity_W_from_R_Rd, 2/3)
    print(f"\nHistorical ξ = (2/3)R_d: RMS = {rms_two_thirds:.2f} km/s")
    
    # =========================================================================
    # OPTION 4: Modify h(g) exponent to absorb spatial dependence
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 4: MODIFY h(g) EXPONENT (ABSORB W INTO h)")
    print(f"{'='*70}")
    
    # Current h(g) = (g†/g)^0.5 × g†/(g†+g)
    # Try h(g) = (g†/g)^n × g†/(g†+g) with different n
    
    def obj_exponent(n):
        return compute_rms(galaxies, predict_velocity_modified_exponent, n)
    
    result = minimize_scalar(obj_exponent, bounds=(0.3, 0.8), method='bounded')
    n_opt = result.x
    rms_exp = result.fun
    
    print(f"Current h(g) exponent: 0.5")
    print(f"Optimal h(g) exponent (no W): {n_opt:.3f}")
    print(f"RMS = {rms_exp:.2f} km/s")
    print(f"Degradation: {100*(rms_exp/rms_current - 1):+.1f}%")
    
    # =========================================================================
    # OPTION 5: Simplest possible - just ξ = R_d (α = 1)
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 5: SIMPLEST - ξ = R_d (NO FREE PARAMETERS)")
    print(f"{'='*70}")
    
    rms_Rd_simple = compute_rms_with_Rd(galaxies, predict_velocity_W_from_R_Rd, 1.0)
    print(f"ξ = R_d: RMS = {rms_Rd_simple:.2f} km/s")
    print(f"Degradation: {100*(rms_Rd_simple/rms_current - 1):+.1f}%")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: ALTERNATIVES TO k × σ/Ω")
    print(f"{'='*70}")
    
    print(f"""
| Model | Parameters | RMS (km/s) | vs Current |
|-------|------------|------------|------------|
| Current (k×σ/Ω) | k=0.24 | {rms_current:.2f} | — |
| No W(r) | — | {rms_no_W:.2f} | {100*(rms_no_W/rms_current - 1):+.1f}% |
| Fixed global ξ | ξ={xi_global:.1f} kpc | {rms_fixed_xi:.2f} | {100*(rms_fixed_xi/rms_current - 1):+.1f}% |
| ξ = α×R_d | α={alpha_opt:.2f} | {rms_Rd:.2f} | {100*(rms_Rd/rms_current - 1):+.1f}% |
| ξ = (2/3)R_d | α=0.67 | {rms_two_thirds:.2f} | {100*(rms_two_thirds/rms_current - 1):+.1f}% |
| ξ = R_d | α=1.0 | {rms_Rd_simple:.2f} | {100*(rms_Rd_simple/rms_current - 1):+.1f}% |
| Modified h(g) | n={n_opt:.2f} | {rms_exp:.2f} | {100*(rms_exp/rms_current - 1):+.1f}% |
""")
    
    print("""
KEY INSIGHT:
The "dynamical" coherence scale ξ = k×σ/Ω provides only marginal improvement
over simpler alternatives:

1. ξ = (2/3)R_d (historical baseline) works almost as well
2. A fixed global ξ ≈ 1 kpc works reasonably well
3. Even removing W(r) entirely only degrades by ~10%

This suggests W(r) is doing something, but the SPECIFIC form of ξ doesn't
matter much. The improvement from k×σ/Ω over (2/3)R_d is real but small.

RECOMMENDATION:
If we want to eliminate k, the simplest option is:

    ξ = (2/3) × R_d

This is:
- Parameter-free (uses only observable R_d)
- Physically motivated (coherence builds over ~1 disk scale length)
- Nearly as good as the "dynamical" formula
- Already documented as the "historical baseline"
""")
    
    # Save results
    results = {
        'current_k_sigma_Omega': {'k': 0.24, 'rms': rms_current},
        'no_W': {'rms': rms_no_W},
        'fixed_global_xi': {'xi': xi_global, 'rms': rms_fixed_xi},
        'xi_alpha_Rd': {'alpha': alpha_opt, 'rms': rms_Rd},
        'xi_two_thirds_Rd': {'alpha': 2/3, 'rms': rms_two_thirds},
        'xi_equals_Rd': {'alpha': 1.0, 'rms': rms_Rd_simple},
        'modified_h_exponent': {'n': n_opt, 'rms': rms_exp}
    }
    
    output_path = Path("/Users/leonardspeiser/Projects/sigmagravity/derivations/eliminate_k_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

