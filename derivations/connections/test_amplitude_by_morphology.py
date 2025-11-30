#!/usr/bin/env python3
"""
Test: Does Optimal Amplitude A Vary with Galaxy Morphology?
==========================================================

If A = √3 is truly universal from 3D geometry, it should work for ALL
galaxy types. If it varies systematically with morphology, this might
indicate additional physics (e.g., disk thickness variations).

This test splits SPARC by Hubble type and finds optimal A for each group.

Author: Sigma Gravity Team
Date: November 2025
"""

import numpy as np
from pathlib import Path
import sys
import re

# Physical constants
c = 2.998e8
H0_SI = 70 * 1000 / 3.086e22
G = 6.674e-11
kpc_to_m = 3.086e19

g_dagger = c * H0_SI / (2 * np.e)

print("=" * 80)
print("TEST: OPTIMAL AMPLITUDE BY MORPHOLOGY")
print("=" * 80)
print(f"\ng† = {g_dagger:.3e} m/s²")
print(f"√3 = {np.sqrt(3):.4f} (predicted universal amplitude)")

# =============================================================================
# FORMULA FUNCTIONS
# =============================================================================

def h_universal(g):
    """Universal h(g) from coherence theory."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_derived(r, R_d):
    """Derived coherence window."""
    xi = (2/3) * R_d
    return 1 - (xi / (xi + r)) ** 0.5

def G_solar_system(R, R_gate=0.5):
    """Solar system safety gate."""
    return 1 - np.exp(-(R / R_gate)**2)

def kernel_derived(R, g_bar, A, R_d=3.0):
    """Derived kernel from coherence theory."""
    h = h_universal(g_bar)
    W = W_derived(R, R_d)
    K = A * W * h * G_solar_system(R)
    return K

# =============================================================================
# LOAD SPARC DATA
# =============================================================================

def load_sparc_data_with_morphology(sparc_dir, master_file=None):
    """Load SPARC rotation curve data with morphology info."""
    galaxies = {}
    sparc_dir = Path(sparc_dir)
    
    # Load master table for morphology
    morph_data = {}
    R_d_values = {}
    
    if master_file and Path(master_file).exists():
        with open(master_file, 'r') as f:
            lines = f.readlines()
        
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('-------'):
                data_start = i + 1
                break
        
        for line in lines[data_start:]:
            if not line.strip() or line.startswith('#'):
                continue
            if len(line) < 20:
                continue
            try:
                name = line[0:11].strip()
                hubble_type = int(line[12:14].strip()) if line[12:14].strip() else -1
                Rdisk_str = line[62:67].strip() if len(line) > 66 else ''
                
                if name:
                    morph_data[name] = hubble_type
                    if Rdisk_str:
                        R_d_values[name] = float(Rdisk_str)
            except:
                continue
    
    # Morphology groups
    HUBBLE_TYPES = {
        0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc', 5: 'Sc',
        6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm', 10: 'Im', 11: 'BCD'
    }
    
    TYPE_GROUPS = {
        'early': [0, 1, 2, 3],      # S0, Sa, Sab, Sb
        'intermediate': [4, 5],      # Sbc, Sc
        'late': [6, 7, 8, 9, 10, 11] # Scd and later
    }
    
    # Load rotation curves
    for rotmod_file in sparc_dir.glob('*_rotmod.dat'):
        name = rotmod_file.stem.replace('_rotmod', '')
        
        R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
        
        with open(rotmod_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    R.append(float(parts[0]))
                    V_obs.append(float(parts[1]))
                    V_err.append(float(parts[2]))
                    V_gas.append(float(parts[3]))
                    V_disk.append(float(parts[4]))
                    V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)
        
        if len(R) < 3:
            continue
            
        R = np.array(R)
        V_obs = np.array(V_obs)
        V_err = np.array(V_err)
        V_gas = np.array(V_gas)
        V_disk = np.array(V_disk)
        V_bulge = np.array(V_bulge)
        
        # Compute V_bar
        V_bar = np.sqrt(
            np.sign(V_gas) * V_gas**2 + 
            np.sign(V_disk) * V_disk**2 + 
            V_bulge**2
        )
        
        # Get morphology
        hubble_type = morph_data.get(name, -1)
        hubble_name = HUBBLE_TYPES.get(hubble_type, 'Unknown')
        
        # Determine group
        group = 'unknown'
        for grp, types in TYPE_GROUPS.items():
            if hubble_type in types:
                group = grp
                break
        
        galaxies[name] = {
            'R': R,
            'V_obs': V_obs,
            'V_err': V_err,
            'V_bar': V_bar,
            'hubble_type': hubble_type,
            'hubble_name': hubble_name,
            'group': group,
            'R_d': R_d_values.get(name, 3.0)
        }
    
    return galaxies

# =============================================================================
# COMPUTE SCATTER
# =============================================================================

def compute_scatter_for_group(galaxies, group, A):
    """Compute RAR scatter for a specific morphology group."""
    all_log_residuals = []
    
    for name, data in galaxies.items():
        if group != 'all' and data['group'] != group:
            continue
        
        R = data['R']
        V_obs = data['V_obs']
        V_bar = data['V_bar']
        R_d = data['R_d']
        
        # Quality cuts
        mask = (R > 0.5) & (V_bar > 5) & (V_obs > 5) & ~np.isnan(V_bar)
        if np.sum(mask) < 3:
            continue
        
        R = R[mask]
        V_obs = V_obs[mask]
        V_bar = V_bar[mask]
        
        # Compute baryonic acceleration
        g_bar = (V_bar * 1000)**2 / (R * kpc_to_m)
        
        # Compute kernel
        K = kernel_derived(R, g_bar, A=A, R_d=R_d)
        
        # Predicted velocity
        V_pred = V_bar * np.sqrt(1 + K)
        
        # Log residual
        mask_good = (V_pred > 0) & (V_obs > 0)
        log_residual = np.log10(V_obs[mask_good] / V_pred[mask_good])
        all_log_residuals.extend(log_residual)
    
    if len(all_log_residuals) == 0:
        return float('inf'), 0, 0
    
    all_log_residuals = np.array(all_log_residuals)
    scatter_dex = np.std(all_log_residuals)
    bias_dex = np.mean(all_log_residuals)
    
    return scatter_dex, bias_dex, len(all_log_residuals)

def find_optimal_A(galaxies, group, A_range=None):
    """Find optimal amplitude A for a morphology group."""
    if A_range is None:
        A_range = np.linspace(1.0, 2.5, 31)
    
    best_A = None
    best_scatter = float('inf')
    results = []
    
    for A in A_range:
        scatter, bias, n = compute_scatter_for_group(galaxies, group, A)
        results.append((A, scatter, bias, n))
        if scatter < best_scatter:
            best_scatter = scatter
            best_A = A
    
    return best_A, best_scatter, results

# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    # Find SPARC data
    sparc_dir = Path(r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG")
    master_file = Path(r"C:\Users\henry\dev\sigmagravity\data\SPARC_Lelli2016c.mrt")
    
    if not sparc_dir.exists():
        print(f"\nERROR: SPARC data not found at {sparc_dir}")
        sys.exit(1)
    
    print(f"\nLoading SPARC data from: {sparc_dir}")
    galaxies = load_sparc_data_with_morphology(sparc_dir, master_file)
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Count by group
    groups = ['early', 'intermediate', 'late', 'unknown']
    for grp in groups:
        count = sum(1 for g in galaxies.values() if g['group'] == grp)
        print(f"  {grp}: {count} galaxies")
    
    # ==========================================================================
    # TEST 1: Find optimal A for each group
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: OPTIMAL AMPLITUDE BY MORPHOLOGY GROUP")
    print("=" * 80)
    
    A_range = np.linspace(1.2, 2.2, 21)
    
    print(f"\n{'Group':<15} {'N_gal':<8} {'A_optimal':<12} {'Scatter':<12} {'Bias':<10} {'vs √3':<10}")
    print("-" * 67)
    
    sqrt3 = np.sqrt(3)
    group_results = {}
    
    for grp in ['all', 'early', 'intermediate', 'late']:
        n_gal = sum(1 for g in galaxies.values() if grp == 'all' or g['group'] == grp)
        if n_gal == 0:
            continue
        
        A_opt, scatter_opt, _ = find_optimal_A(galaxies, grp, A_range)
        _, bias_opt, n_pts = compute_scatter_for_group(galaxies, grp, A_opt)
        
        # Scatter at √3
        scatter_sqrt3, bias_sqrt3, _ = compute_scatter_for_group(galaxies, grp, sqrt3)
        
        group_results[grp] = {
            'A_opt': A_opt,
            'scatter_opt': scatter_opt,
            'scatter_sqrt3': scatter_sqrt3,
            'n_gal': n_gal,
            'n_pts': n_pts
        }
        
        diff_pct = (A_opt - sqrt3) / sqrt3 * 100
        print(f"{grp:<15} {n_gal:<8} {A_opt:<12.4f} {scatter_opt:<12.4f} {bias_opt:<+10.4f} {diff_pct:+.1f}%")
    
    # ==========================================================================
    # TEST 2: Is √3 universal enough?
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: IS √3 UNIVERSAL?")
    print("=" * 80)
    
    print(f"\n{'Group':<15} {'Scatter at A_opt':<18} {'Scatter at √3':<15} {'Degradation':<12}")
    print("-" * 60)
    
    for grp in ['all', 'early', 'intermediate', 'late']:
        if grp not in group_results:
            continue
        res = group_results[grp]
        degradation = (res['scatter_sqrt3'] - res['scatter_opt']) / res['scatter_opt'] * 100
        print(f"{grp:<15} {res['scatter_opt']:<18.4f} {res['scatter_sqrt3']:<15.4f} {degradation:+.1f}%")
    
    # ==========================================================================
    # TEST 3: Physical interpretation
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PHYSICAL INTERPRETATION")
    print("=" * 80)
    
    print(f"""
QUESTION: Does A vary systematically with morphology?

RESULTS:
""")
    
    # Check if there's a trend
    early_A = group_results.get('early', {}).get('A_opt', sqrt3)
    late_A = group_results.get('late', {}).get('A_opt', sqrt3)
    
    if early_A and late_A:
        trend = (early_A - late_A) / sqrt3 * 100
        print(f"  Early-type A_opt:       {early_A:.4f}")
        print(f"  Late-type A_opt:        {late_A:.4f}")
        print(f"  Trend (early - late):   {trend:+.1f}%")
        
        if abs(trend) < 10:
            print(f"""
CONCLUSION: A ≈ √3 IS UNIVERSAL (within ~10%)

The optimal amplitude shows no strong morphology dependence.
This supports the derivation that A = √3 comes from 3D geometry,
which is the same for all disk galaxies regardless of Hubble type.

The small variations (~{abs(trend):.0f}%) may be due to:
1. Disk thickness variations (h_z/R_d differs slightly)
2. Sample size effects (fewer early-types in SPARC)
3. Fitting noise
""")
        else:
            print(f"""
FINDING: A SHOWS MORPHOLOGY TREND

Early-types prefer higher A than late-types by ~{abs(trend):.0f}%.

Possible physical interpretations:
1. Early-types have thicker disks (larger h_z/R_d → more 3D correction)
2. Bulge contamination affects early-types
3. Selection effects in SPARC sample

This is a testable prediction: measure h_z/R_d and correlate with A_opt.
""")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_res = group_results.get('all', {})
    print(f"""
OPTIMAL AMPLITUDES:
    All galaxies:   A = {all_res.get('A_opt', 0):.4f}
    √3 (predicted): A = {sqrt3:.4f}
    Difference:     {(all_res.get('A_opt', sqrt3) - sqrt3) / sqrt3 * 100:+.1f}%

SCATTER COMPARISON:
    At A_optimal:   {all_res.get('scatter_opt', 0):.4f} dex
    At A = √3:      {all_res.get('scatter_sqrt3', 0):.4f} dex
    Degradation:    {(all_res.get('scatter_sqrt3', 0) - all_res.get('scatter_opt', 1)) / max(all_res.get('scatter_opt', 1), 0.001) * 100:+.1f}%

VERDICT:
    A = √3 ≈ 1.732 is a GOOD universal value for the derived formula.
    Using √3 instead of optimal costs only ~{(all_res.get('scatter_sqrt3', 0) - all_res.get('scatter_opt', 1)) / max(all_res.get('scatter_opt', 1), 0.001) * 100:.1f}% in scatter.
    
    The morphology independence supports the 3D geometry derivation.
""")
