#!/usr/bin/env python3
"""
Test if bulge-dominated galaxies need Σ < 1 (screening).

The residual analysis suggests bulges may need gravitational SCREENING
rather than just suppressed enhancement. This script tests:

1. What happens if we allow φ to push Σ below 1 for bulge points?
2. Is there an optimal bulge φ that minimizes RMS?
3. Do bulge points systematically overpredict velocities?

Author: Leonard Speiser
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_regression_extended import (
    load_sparc, predict_velocity_baseline, predict_mond,
    h_function, C_coherence, A_0, kpc_to_m, ML_DISK, ML_BULGE
)

def predict_with_bulge_phi(R_kpc, V_bar, R_d, f_bulge_r, phi_disk=1.0, phi_bulge=1.0, sigma_kms=20.0):
    """
    Predict rotation curve with separate φ for disk and bulge regions.
    
    Parameters:
    -----------
    phi_disk : float
        φ for disk-dominated points (f_bulge < 0.3)
    phi_bulge : float
        φ for bulge-dominated points (f_bulge >= 0.3)
        Can be < 0 for screening (Σ < 1)
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    h = h_function(g_bar)
    V = np.array(V_bar, dtype=float)
    
    # Per-point φ based on bulge fraction
    phi = np.where(f_bulge_r >= 0.3, phi_bulge, phi_disk)
    
    for _ in range(50):
        C = C_coherence(V, sigma_kms)
        Sigma = 1 + A_0 * phi * C * h
        # Allow Σ < 1 (screening) when φ is negative
        Sigma = np.maximum(Sigma, 0.01)  # Floor to prevent negative/zero
        V_new = V_bar * np.sqrt(Sigma)
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new
    return V, phi


def analyze_bulge_residuals(galaxies):
    """Analyze if bulge points systematically overpredict."""
    
    disk_residuals = []  # V_pred - V_obs for disk points
    bulge_residuals = []  # V_pred - V_obs for bulge points
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        R_d = gal['R_d']
        f_bulge_r = gal.get('f_bulge_r', np.zeros_like(R))
        
        # Baseline prediction
        V_pred = predict_velocity_baseline(R, V_bar, R_d)
        residual = V_pred - V_obs
        
        # Split by bulge fraction
        disk_mask = f_bulge_r < 0.3
        bulge_mask = f_bulge_r >= 0.3
        
        disk_residuals.extend(residual[disk_mask])
        bulge_residuals.extend(residual[bulge_mask])
    
    return np.array(disk_residuals), np.array(bulge_residuals)


def sweep_bulge_phi(galaxies, phi_bulge_range):
    """Sweep φ_bulge to find optimal value."""
    
    results = []
    
    for phi_bulge in phi_bulge_range:
        rms_list = []
        
        for gal in galaxies:
            R = gal['R']
            V_obs = gal['V_obs']
            V_bar = gal['V_bar']
            R_d = gal['R_d']
            f_bulge_r = gal.get('f_bulge_r', np.zeros_like(R))
            
            # Skip galaxies with no bulge points
            if not np.any(f_bulge_r >= 0.3):
                continue
            
            V_pred, _ = predict_with_bulge_phi(R, V_bar, R_d, f_bulge_r, 
                                               phi_disk=1.0, phi_bulge=phi_bulge)
            rms = np.sqrt(((V_pred - V_obs)**2).mean())
            rms_list.append(rms)
        
        if rms_list:
            results.append({
                'phi_bulge': phi_bulge,
                'mean_rms': np.mean(rms_list),
                'n_galaxies': len(rms_list)
            })
    
    return results


def main():
    print("=" * 80)
    print("BULGE SCREENING HYPOTHESIS TEST")
    print("=" * 80)
    print()
    print("Question: Do bulge-dominated regions need Sigma < 1 (screening)?")
    print()
    
    data_dir = Path(__file__).parent.parent / "data"
    galaxies = load_sparc(data_dir)
    
    print(f"Loaded {len(galaxies)} SPARC galaxies")
    
    # Count galaxies with bulge points
    n_with_bulge = sum(1 for g in galaxies 
                       if 'f_bulge_r' in g and np.any(g['f_bulge_r'] >= 0.3))
    print(f"Galaxies with bulge-dominated points (f_bulge >= 0.3): {n_with_bulge}")
    print()
    
    # 1. Analyze residuals
    print("-" * 80)
    print("1. RESIDUAL ANALYSIS: Do bulge points systematically overpredict?")
    print("-" * 80)
    
    disk_res, bulge_res = analyze_bulge_residuals(galaxies)
    
    print(f"Disk points (f_bulge < 0.3): N = {len(disk_res)}")
    print(f"  Mean residual (V_pred - V_obs): {np.mean(disk_res):.2f} km/s")
    print(f"  Std residual: {np.std(disk_res):.2f} km/s")
    print()
    print(f"Bulge points (f_bulge >= 0.3): N = {len(bulge_res)}")
    if len(bulge_res) > 0:
        print(f"  Mean residual (V_pred - V_obs): {np.mean(bulge_res):.2f} km/s")
        print(f"  Std residual: {np.std(bulge_res):.2f} km/s")
        print()
        if np.mean(bulge_res) > 0:
            print("  --> OVERPREDICTING: Baseline Sigma-Gravity gives too high V in bulges")
            print("  --> This suggests bulges may need SCREENING (Sigma < 1)")
        else:
            print("  --> UNDERPREDICTING: Baseline gives too low V in bulges")
    else:
        print("  No bulge-dominated points found with current threshold")
    print()
    
    # 2. Sweep phi_bulge
    print("-" * 80)
    print("2. PHI_BULGE SWEEP: What phi minimizes RMS for bulge galaxies?")
    print("-" * 80)
    
    phi_range = np.linspace(-1.0, 1.5, 26)  # -1 to 1.5 in steps of 0.1
    results = sweep_bulge_phi(galaxies, phi_range)
    
    if results:
        print(f"\n{'phi_bulge':>10} {'Mean RMS':>12} {'N galaxies':>12}")
        print("-" * 40)
        for r in results:
            marker = " <-- optimal" if r['mean_rms'] == min(x['mean_rms'] for x in results) else ""
            print(f"{r['phi_bulge']:>10.2f} {r['mean_rms']:>12.2f} {r['n_galaxies']:>12}{marker}")
        
        best = min(results, key=lambda x: x['mean_rms'])
        print()
        print(f"OPTIMAL phi_bulge = {best['phi_bulge']:.2f}")
        print(f"  At this phi, mean RMS = {best['mean_rms']:.2f} km/s")
        
        # Compare to phi=1 (no modification)
        baseline = next((r for r in results if abs(r['phi_bulge'] - 1.0) < 0.05), None)
        if baseline:
            improvement = (baseline['mean_rms'] - best['mean_rms']) / baseline['mean_rms'] * 100
            print(f"  Improvement over phi=1: {improvement:.1f}%")
        
        if best['phi_bulge'] < 0:
            print()
            print("  --> SCREENING CONFIRMED: Optimal phi_bulge < 0 means Sigma < 1 helps")
        elif best['phi_bulge'] < 1:
            print()
            print("  --> SUPPRESSION CONFIRMED: Optimal phi_bulge < 1 means less enhancement")
        else:
            print()
            print("  --> Enhancement still optimal for bulge points")
    else:
        print("No results - insufficient bulge data")
    
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)


if __name__ == "__main__":
    main()

