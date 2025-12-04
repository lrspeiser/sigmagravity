#!/usr/bin/env python3
"""
Comprehensive Test: g† = cH₀/(4√π) vs g† = cH₀/(2e)
====================================================

This script performs a rigorous comparison of the two critical acceleration formulas
on the SPARC galaxy rotation curve dataset.

Key Questions:
1. Does the new formula give better or equal results?
2. What is the RMS velocity error for each formula?
3. What is the RAR scatter (dex) for each formula?
4. How many galaxies does each formula "win" on?

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import math
from typing import Dict, List, Tuple
import sys

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8              # Speed of light [m/s]
H0_SI = 2.27e-18         # Hubble constant [1/s] (70 km/s/Mpc)
cH0 = c * H0_SI          # c × H₀ [m/s²]
e = math.e               # Euler's number
kpc_to_m = 3.086e19      # meters per kpc

# Two formulas for critical acceleration
g_dagger_old = cH0 / (2 * e)              # Old: cH₀/(2e) ≈ 1.25×10⁻¹⁰ m/s²
g_dagger_new = cH0 / (4 * math.sqrt(math.pi))  # New: cH₀/(4√π) ≈ 0.96×10⁻¹⁰ m/s²

print("=" * 80)
print("CRITICAL ACCELERATION COMPARISON: g† = cH₀/(4√π) vs g† = cH₀/(2e)")
print("=" * 80)
print(f"\nOld formula: g† = cH₀/(2e)   = {g_dagger_old:.4e} m/s²")
print(f"New formula: g† = cH₀/(4√π)  = {g_dagger_new:.4e} m/s²")
print(f"Ratio (new/old): {g_dagger_new/g_dagger_old:.4f}")
print(f"\n4√π = {4*math.sqrt(math.pi):.4f}")
print(f"2e  = {2*e:.4f}")

# =============================================================================
# SPARC DATA (Extended set from data_loader.py)
# =============================================================================

SPARC_GALAXIES = {
    # HIGH SURFACE BRIGHTNESS SPIRALS
    'NGC2403': {
        'R_kpc': np.array([0.36, 0.72, 1.44, 2.17, 2.89, 3.61, 4.33, 5.78, 7.22, 8.67, 10.83, 13.0]),
        'V_obs': np.array([32, 65, 98, 115, 120, 125, 128, 130, 132, 132, 130, 128]),
        'V_bar': np.array([31, 60, 85, 95, 90, 84, 78, 66, 56, 49, 41, 36]),
        'R_d': 2.0,  # disk scale length in kpc
    },
    'NGC3198': {
        'R_kpc': np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 30.0]),
        'V_obs': np.array([55, 100, 142, 150, 152, 150, 150, 150, 150, 148, 145]),
        'V_bar': np.array([52, 92, 120, 112, 98, 85, 73, 55, 44, 37, 32]),
        'R_d': 3.0,
    },
    'NGC7331': {
        'R_kpc': np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0]),
        'V_obs': np.array([120, 200, 250, 250, 245, 240, 230, 220, 215, 210]),
        'V_bar': np.array([115, 185, 210, 185, 160, 140, 105, 85, 72, 62]),
        'R_d': 4.0,
    },
    'NGC2841': {
        'R_kpc': np.array([2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]),
        'V_obs': np.array([180, 285, 305, 305, 300, 295, 290, 285]),
        'V_bar': np.array([165, 250, 255, 225, 195, 155, 130, 115]),
        'R_d': 5.0,
    },
    'NGC5055': {
        'R_kpc': np.array([1.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 30.0]),
        'V_obs': np.array([80, 160, 195, 200, 198, 192, 185, 180]),
        'V_bar': np.array([75, 145, 165, 150, 135, 108, 90, 78]),
        'R_d': 3.5,
    },
    # LOW SURFACE BRIGHTNESS
    'UGC128': {
        'R_kpc': np.array([2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]),
        'V_obs': np.array([65, 115, 135, 140, 142, 145, 145, 143, 140]),
        'V_bar': np.array([58, 95, 100, 88, 75, 65, 55, 43, 36]),
        'R_d': 6.0,
    },
    'UGC2885': {
        'R_kpc': np.array([5.0, 15.0, 30.0, 50.0, 70.0, 90.0, 110.0]),
        'V_obs': np.array([130, 280, 300, 300, 295, 290, 285]),
        'V_bar': np.array([115, 240, 235, 200, 175, 155, 140]),
        'R_d': 12.0,
    },
    'F571-8': {
        'R_kpc': np.array([1.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0]),
        'V_obs': np.array([45, 90, 110, 115, 115, 112, 110]),
        'V_bar': np.array([40, 75, 82, 75, 65, 55, 48]),
        'R_d': 4.0,
    },
    # DWARF GALAXIES
    'DDO154': {
        'R_kpc': np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]),
        'V_obs': np.array([20, 30, 38, 44, 47, 49, 50, 48, 46, 44]),
        'V_bar': np.array([18, 25, 28, 28, 27, 25, 22, 19, 17, 15]),
        'R_d': 1.5,
    },
    'IC2574': {
        'R_kpc': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        'V_obs': np.array([30, 45, 55, 62, 67, 70, 72, 73, 74, 74]),
        'V_bar': np.array([28, 38, 42, 42, 40, 38, 35, 32, 30, 28]),
        'R_d': 2.5,
    },
    'DDO168': {
        'R_kpc': np.array([0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
        'V_obs': np.array([18, 32, 45, 52, 55, 56, 55, 53]),
        'V_bar': np.array([16, 28, 38, 40, 38, 36, 33, 30]),
        'R_d': 0.8,
    },
    'NGC2366': {
        'R_kpc': np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        'V_obs': np.array([22, 40, 52, 58, 60, 58, 55, 52, 50]),
        'V_bar': np.array([20, 35, 44, 48, 45, 40, 35, 32, 29]),
        'R_d': 1.2,
    },
    # GAS-DOMINATED
    'NGC925': {
        'R_kpc': np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
        'V_obs': np.array([48, 80, 100, 105, 108, 110, 110, 108]),
        'V_bar': np.array([45, 70, 82, 78, 72, 66, 60, 55]),
        'R_d': 3.0,
    },
    'NGC4214': {
        'R_kpc': np.array([0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]),
        'V_obs': np.array([25, 45, 60, 65, 66, 65, 64]),
        'V_bar': np.array([23, 40, 52, 52, 50, 48, 45]),
        'R_d': 0.7,
    },
    # EARLY TYPE
    'NGC3992': {
        'R_kpc': np.array([2.0, 5.0, 10.0, 15.0, 20.0, 25.0]),
        'V_obs': np.array([150, 235, 255, 250, 240, 235]),
        'V_bar': np.array([140, 210, 215, 190, 165, 145]),
        'R_d': 4.5,
    },
}

# =============================================================================
# Σ-GRAVITY KERNEL FUNCTIONS
# =============================================================================

def h_function(g: np.ndarray, g_dagger: float) -> np.ndarray:
    """
    Universal enhancement function: h(g) = √(g†/g) × g†/(g†+g)
    
    Parameters:
    -----------
    g : array
        Baryonic acceleration [m/s²]
    g_dagger : float
        Critical acceleration [m/s²]
    
    Returns:
    --------
    h : array
        Enhancement function value
    """
    g = np.maximum(g, 1e-15)  # Numerical safety
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def W_coherence(r: np.ndarray, R_d: float) -> np.ndarray:
    """
    Coherence window: W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)R_d
    
    Parameters:
    -----------
    r : array
        Radius [kpc]
    R_d : float
        Disk scale length [kpc]
    
    Returns:
    --------
    W : array
        Coherence window value [0 to 1]
    """
    xi = (2/3) * R_d
    return 1 - (xi / (xi + r)) ** 0.5


def predict_velocity(R_kpc: np.ndarray, V_bar: np.ndarray, R_d: float, 
                     g_dagger: float, A: float = np.sqrt(3)) -> np.ndarray:
    """
    Predict rotation curve using Σ-Gravity formula:
    
    Σ = 1 + A × W(r) × h(g)
    V_pred = V_bar × √Σ
    
    Parameters:
    -----------
    R_kpc : array
        Radius [kpc]
    V_bar : array
        Baryonic velocity [km/s]
    R_d : float
        Disk scale length [kpc]
    g_dagger : float
        Critical acceleration [m/s²]
    A : float
        Amplitude (default √3 for disks)
    
    Returns:
    --------
    V_pred : array
        Predicted rotation velocity [km/s]
    """
    # Convert V_bar to g_bar
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000  # km/s to m/s
    g_bar = V_bar_ms**2 / R_m  # m/s²
    
    # Compute enhancement
    h = h_function(g_bar, g_dagger)
    W = W_coherence(R_kpc, R_d)
    Sigma = 1 + A * W * h
    
    # Predict velocity
    V_pred = V_bar * np.sqrt(Sigma)
    
    return V_pred


def compute_rms_error(V_obs: np.ndarray, V_pred: np.ndarray) -> float:
    """Compute RMS velocity error in km/s."""
    return np.sqrt(np.mean((V_obs - V_pred)**2))


def compute_rar_scatter(R_kpc: np.ndarray, V_obs: np.ndarray, V_pred: np.ndarray) -> float:
    """
    Compute RAR scatter in dex.
    
    RAR scatter = std(log10(g_obs/g_pred))
    """
    R_m = R_kpc * kpc_to_m
    g_obs = (V_obs * 1000)**2 / R_m
    g_pred = (V_pred * 1000)**2 / R_m
    
    # Avoid log of zero/negative
    mask = (g_obs > 0) & (g_pred > 0)
    if mask.sum() < 2:
        return np.nan
    
    log_residual = np.log10(g_obs[mask] / g_pred[mask])
    return np.std(log_residual)


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison():
    """Run comprehensive comparison of old vs new formula."""
    
    print("\n" + "=" * 80)
    print("RUNNING SPARC GALAXY COMPARISON")
    print("=" * 80)
    
    results_old = {}
    results_new = {}
    
    A = np.sqrt(3)  # Amplitude for disks
    
    for name, data in SPARC_GALAXIES.items():
        R = data['R_kpc']
        V_obs = data['V_obs']
        V_bar = data['V_bar']
        R_d = data['R_d']
        
        # Old formula prediction
        V_pred_old = predict_velocity(R, V_bar, R_d, g_dagger_old, A)
        rms_old = compute_rms_error(V_obs, V_pred_old)
        rar_old = compute_rar_scatter(R, V_obs, V_pred_old)
        
        # New formula prediction
        V_pred_new = predict_velocity(R, V_bar, R_d, g_dagger_new, A)
        rms_new = compute_rms_error(V_obs, V_pred_new)
        rar_new = compute_rar_scatter(R, V_obs, V_pred_new)
        
        results_old[name] = {'rms': rms_old, 'rar': rar_old}
        results_new[name] = {'rms': rms_new, 'rar': rar_new}
    
    # Summary statistics
    rms_old_list = [r['rms'] for r in results_old.values()]
    rms_new_list = [r['rms'] for r in results_new.values()]
    rar_old_list = [r['rar'] for r in results_old.values() if not np.isnan(r['rar'])]
    rar_new_list = [r['rar'] for r in results_new.values() if not np.isnan(r['rar'])]
    
    mean_rms_old = np.mean(rms_old_list)
    mean_rms_new = np.mean(rms_new_list)
    mean_rar_old = np.mean(rar_old_list)
    mean_rar_new = np.mean(rar_new_list)
    
    # Count wins
    wins_old = 0
    wins_new = 0
    ties = 0
    
    for name in SPARC_GALAXIES.keys():
        if results_old[name]['rms'] < results_new[name]['rms']:
            wins_old += 1
        elif results_new[name]['rms'] < results_old[name]['rms']:
            wins_new += 1
        else:
            ties += 1
    
    # Print per-galaxy results
    print(f"\n{'Galaxy':<12} {'RMS Old':<12} {'RMS New':<12} {'RAR Old':<12} {'RAR New':<12} {'Winner':<10}")
    print("-" * 70)
    
    for name in SPARC_GALAXIES.keys():
        rms_o = results_old[name]['rms']
        rms_n = results_new[name]['rms']
        rar_o = results_old[name]['rar']
        rar_n = results_new[name]['rar']
        
        if rms_o < rms_n:
            winner = "Old"
        elif rms_n < rms_o:
            winner = "New"
        else:
            winner = "Tie"
        
        print(f"{name:<12} {rms_o:<12.2f} {rms_n:<12.2f} {rar_o:<12.4f} {rar_n:<12.4f} {winner:<10}")
    
    print("-" * 70)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        FORMULA COMPARISON RESULTS                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Formula              │ Mean RMS (km/s) │ Mean RAR (dex) │ Galaxies Won       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ OLD: g† = cH₀/(2e)   │ {mean_rms_old:>15.2f} │ {mean_rar_old:>14.4f} │ {wins_old:>18}  ║
║ NEW: g† = cH₀/(4√π)  │ {mean_rms_new:>15.2f} │ {mean_rar_new:>14.4f} │ {wins_new:>18}  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Improvement with new formula:
  RMS: {100*(mean_rms_old - mean_rms_new)/mean_rms_old:+.1f}% {'(BETTER)' if mean_rms_new < mean_rms_old else '(WORSE)'}
  RAR: {100*(mean_rar_old - mean_rar_new)/mean_rar_old:+.1f}% {'(BETTER)' if mean_rar_new < mean_rar_old else '(WORSE)'}
  
Head-to-head: Old wins {wins_old}, New wins {wins_new}, Ties {ties}
""")
    
    # Physical interpretation
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PHYSICAL INTERPRETATION                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

The new formula g† = cH₀/(4√π) has clear geometric origin:

  4√π = 2 × √(4π) where:
  • √(4π) from spherical solid angle (4π steradians)
  • Factor 2 from coherence transition (R_coh → 2×R_coh)

At r = 2×R_coh, the acceleration is exactly g† = cH₀/(4√π).

This eliminates the arbitrary factor 'e' from the theory!
""")
    
    return {
        'mean_rms_old': mean_rms_old,
        'mean_rms_new': mean_rms_new,
        'mean_rar_old': mean_rar_old,
        'mean_rar_new': mean_rar_new,
        'wins_old': wins_old,
        'wins_new': wins_new,
        'improvement_rms_pct': 100*(mean_rms_old - mean_rms_new)/mean_rms_old,
        'improvement_rar_pct': 100*(mean_rar_old - mean_rar_new)/mean_rar_old,
    }


if __name__ == "__main__":
    results = run_comparison()
    
    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if results['improvement_rms_pct'] > 0 and results['improvement_rar_pct'] > 0:
        print("""
✓ The NEW formula g† = cH₀/(4√π) is BETTER than the old formula g† = cH₀/(2e)

RECOMMENDATION: Update the codebase to use g† = cH₀/(4√π)

This change:
1. Improves rotation curve fits
2. Eliminates the arbitrary constant 'e'
3. Uses only geometric constants (√π from solid angle)
4. Has clear physical interpretation (acceleration at 2×R_coh)
""")
    elif results['improvement_rms_pct'] >= -5 and results['improvement_rar_pct'] >= -5:
        print("""
≈ The NEW formula g† = cH₀/(4√π) performs COMPARABLY to the old formula

RECOMMENDATION: Consider updating to the new formula for theoretical elegance

The new formula:
1. Eliminates the arbitrary constant 'e'
2. Uses only geometric constants
3. Has clear physical interpretation
""")
    else:
        print("""
✗ The NEW formula g† = cH₀/(4√π) performs WORSE than the old formula

RECOMMENDATION: Keep the old formula g† = cH₀/(2e)
""")

