#!/usr/bin/env python3
"""
Vorticity Correlation Test: A Microphysics Prediction
======================================================

If the current-current correlator is fundamentally connected to TORSION,
then the VORTICITY CORRELATION should predict the gravitational enhancement.

Torsion ~ Vorticity: T ∝ ∇×v

Vorticity correlator: <ω(x)·ω(x')>_c = <(∇×v)(x)·(∇×v)(x')>_c

PREDICTION: Galaxies with higher vorticity correlation should show
more gravitational enhancement (higher f_DM or Σ).

This script computes the vorticity correlation for SPARC galaxies
and tests whether it predicts the observed enhancement.

Author: Sigma Gravity Research
Date: December 2025
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8
H0 = 2.27e-18
kpc_to_m = 3.086e19
G = 6.674e-11

g_dagger = c * H0 / (4 * np.sqrt(np.pi))

print("=" * 100)
print("VORTICITY CORRELATION TEST: A MICROPHYSICS PREDICTION")
print("=" * 100)

# =============================================================================
# VORTICITY COMPUTATION
# =============================================================================

def compute_vorticity_profile(R: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute the vorticity ω = ∇×v for a disk.
    
    For axisymmetric rotation v = V(r) ê_φ:
        ω = (1/r) d(rV)/dr ê_z
          = V/r + dV/dr
          = Ω + dV/dr
    
    where Ω = V/r is the angular velocity.
    
    For solid-body rotation (V ∝ r): ω = 2Ω (constant)
    For flat rotation (V = const): ω = V/r (decreasing)
    """
    # Compute dV/dr using finite differences
    dV_dr = np.gradient(V, R)
    
    # Vorticity: ω = V/r + dV/dr
    omega = V / R + dV_dr
    
    return omega


def compute_angular_velocity_profile(R: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute angular velocity Ω = V/r."""
    return V / R


def compute_vorticity_correlation(R: np.ndarray, omega: np.ndarray, 
                                   xi: float) -> float:
    """
    Compute the vorticity-vorticity correlation.
    
    C_ω = ∫∫ W(|r-r'|/ξ) ω(r) ω(r') ρ(r) ρ(r') dr dr' / normalization
    
    For a disk with exponential density profile.
    """
    n = len(R)
    R_d = R.max() / 3  # Approximate disk scale length
    
    numerator = 0.0
    denominator = 0.0
    
    for i in range(n):
        for j in range(n):
            # Spatial coherence window
            separation = abs(R[i] - R[j])
            W = np.exp(-separation**2 / (2 * xi**2))
            
            # Density weights (exponential disk)
            rho_i = np.exp(-R[i] / R_d)
            rho_j = np.exp(-R[j] / R_d)
            
            # Vorticity product
            omega_product = omega[i] * omega[j]
            
            numerator += W * rho_i * rho_j * omega_product
            denominator += W * rho_i * rho_j * omega[i]**2  # Normalize by variance
    
    C_omega = numerator / denominator if denominator > 0 else 0
    return C_omega


def compute_vorticity_uniformity(R: np.ndarray, omega: np.ndarray) -> float:
    """
    Compute how uniform the vorticity is (1 = solid body, 0 = highly differential).
    
    Uniformity = 1 - std(ω) / mean(ω)
    """
    mean_omega = np.mean(np.abs(omega))
    std_omega = np.std(omega)
    
    if mean_omega > 0:
        uniformity = 1 - std_omega / mean_omega
        return max(0, min(1, uniformity))
    return 0


def compute_rotation_curve_shape(R: np.ndarray, V: np.ndarray) -> Dict:
    """
    Characterize the rotation curve shape.
    
    Returns:
        - slope: dV/dR at outer radius (positive = rising, negative = falling)
        - curvature: d²V/dR²
        - flatness: how close to flat rotation
    """
    # Fit V = V_flat × (1 - e^(-R/R_t)) model
    from scipy.optimize import curve_fit
    
    def rising_model(r, V_flat, R_t):
        return V_flat * (1 - np.exp(-r / R_t))
    
    try:
        popt, _ = curve_fit(rising_model, R, V, p0=[V.max(), R.max()/3], 
                           bounds=([0, 0.1], [500, 50]))
        V_flat, R_t = popt
        
        # Compute residuals from flat model
        V_flat_simple = np.mean(V[len(V)//2:])  # Mean of outer half
        flatness = 1 - np.std(V[len(V)//2:]) / V_flat_simple if V_flat_simple > 0 else 0
        
    except:
        V_flat = V.max()
        R_t = R.max() / 3
        flatness = 0.5
    
    # Slope at outer radius
    dV_dr = np.gradient(V, R)
    outer_slope = dV_dr[-1] / V[-1] if V[-1] > 0 else 0  # Normalized slope
    
    return {
        'V_flat': V_flat,
        'R_turnover': R_t,
        'flatness': flatness,
        'outer_slope': outer_slope,
    }


# =============================================================================
# LOAD SPARC DATA
# =============================================================================

def load_sparc_galaxies() -> List[Dict]:
    """Load SPARC galaxies with rotation curves."""
    data_dir = Path(__file__).parent.parent / "data"
    sparc_dir = data_dir / "Rotmod_LTG"
    
    if not sparc_dir.exists():
        return []
    
    # Load disk scale lengths
    scale_lengths = {}
    master_file = sparc_dir / "MasterSheet_SPARC.mrt"
    if master_file.exists():
        with open(master_file, 'r') as f:
            in_data = False
            for line in f:
                if line.startswith('---'):
                    in_data = True
                    continue
                if not in_data or len(line) < 66:
                    continue
                try:
                    name = line[0:11].strip()
                    rdisk_str = line[61:66].strip()
                    if name and rdisk_str:
                        R_d = float(rdisk_str)
                        if R_d > 0:
                            scale_lengths[name] = R_d
                except:
                    continue
    
    galaxies = []
    for rotmod_file in sorted(sparc_dir.glob("*_rotmod.dat")):
        name = rotmod_file.stem.replace('_rotmod', '')
        
        data = []
        with open(rotmod_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        data.append({
                            'R': float(parts[0]),
                            'V_obs': float(parts[1]),
                            'V_gas': float(parts[3]),
                            'V_disk': float(parts[4]),
                            'V_bulge': float(parts[5]) if len(parts) > 5 else 0.0
                        })
                    except:
                        continue
        
        if len(data) < 5:
            continue
        
        df = pd.DataFrame(data)
        R_d = scale_lengths.get(name, df['R'].max() / 4)
        
        # Apply M/L corrections
        V_disk_scaled = df['V_disk'] * np.sqrt(0.5)
        V_bulge_scaled = df['V_bulge'] * np.sqrt(0.7)
        V_bar_sq = np.sign(df['V_gas']) * df['V_gas']**2 + V_disk_scaled**2 + V_bulge_scaled**2
        
        if np.any(V_bar_sq < 0):
            continue
        
        V_bar = np.sqrt(V_bar_sq)
        valid = (V_bar > 0) & (df['R'] > 0) & (df['V_obs'] > 0)
        
        if valid.sum() < 5:
            continue
        
        galaxies.append({
            'name': name,
            'R': df.loc[valid, 'R'].values,
            'V_obs': df.loc[valid, 'V_obs'].values,
            'V_bar': V_bar[valid].values,
            'R_d': R_d,
        })
    
    return galaxies


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    """Run the vorticity correlation test."""
    
    print("\nLoading SPARC galaxies...")
    galaxies = load_sparc_galaxies()
    print(f"Loaded {len(galaxies)} galaxies")
    
    if len(galaxies) == 0:
        print("ERROR: No galaxies loaded")
        return
    
    # Compute vorticity properties for each galaxy
    results = []
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        R_d = gal['R_d']
        
        # Coherence scale
        xi = R_d / (2 * np.pi)
        
        # Compute vorticity for OBSERVED rotation curve
        omega_obs = compute_vorticity_profile(R, V_obs)
        
        # Compute vorticity for BARYONIC rotation curve
        omega_bar = compute_vorticity_profile(R, V_bar)
        
        # Vorticity correlation
        C_omega_obs = compute_vorticity_correlation(R, omega_obs, xi)
        C_omega_bar = compute_vorticity_correlation(R, omega_bar, xi)
        
        # Vorticity uniformity
        uniformity_obs = compute_vorticity_uniformity(R, omega_obs)
        uniformity_bar = compute_vorticity_uniformity(R, omega_bar)
        
        # Rotation curve shape
        shape = compute_rotation_curve_shape(R, V_obs)
        
        # Observed enhancement
        enhancement = np.mean(V_obs / V_bar)
        
        # Acceleration
        R_m = R * kpc_to_m
        g_bar = (V_bar * 1000)**2 / R_m
        mean_g_ratio = np.mean(g_bar) / g_dagger
        
        # f_DM proxy
        f_DM = 1 - np.mean(1 / enhancement**2) if enhancement > 1 else 0
        
        results.append({
            'name': gal['name'],
            'R_d': R_d,
            'xi': xi,
            'C_omega_obs': C_omega_obs,
            'C_omega_bar': C_omega_bar,
            'uniformity_obs': uniformity_obs,
            'uniformity_bar': uniformity_bar,
            'flatness': shape['flatness'],
            'outer_slope': shape['outer_slope'],
            'enhancement': enhancement,
            'f_DM': f_DM,
            'mean_g_ratio': mean_g_ratio,
        })
    
    df = pd.DataFrame(results)
    
    # ==========================================================================
    # ANALYSIS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("VORTICITY CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Correlation between vorticity uniformity and enhancement
    r_uniform, p_uniform = stats.pearsonr(df['uniformity_obs'], df['enhancement'])
    print(f"\nVorticity uniformity vs enhancement:")
    print(f"  Pearson r = {r_uniform:+.3f}, p = {p_uniform:.2e}")
    
    # Correlation between flatness and enhancement
    r_flat, p_flat = stats.pearsonr(df['flatness'], df['enhancement'])
    print(f"\nRotation curve flatness vs enhancement:")
    print(f"  Pearson r = {r_flat:+.3f}, p = {p_flat:.2e}")
    
    # Correlation between vorticity correlation and f_DM
    r_Comega, p_Comega = stats.pearsonr(df['C_omega_obs'], df['f_DM'])
    print(f"\nVorticity correlation C_ω vs f_DM:")
    print(f"  Pearson r = {r_Comega:+.3f}, p = {p_Comega:.2e}")
    
    # ==========================================================================
    # SPLIT BY VORTICITY UNIFORMITY
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("SPLIT BY VORTICITY UNIFORMITY")
    print("=" * 80)
    
    # High uniformity = solid-body-like rotation
    # Low uniformity = differential rotation
    
    median_uniformity = df['uniformity_obs'].median()
    high_uniform = df[df['uniformity_obs'] > median_uniformity]
    low_uniform = df[df['uniformity_obs'] <= median_uniformity]
    
    print(f"\nHigh vorticity uniformity (solid-body-like, N={len(high_uniform)}):")
    print(f"  Mean enhancement: {high_uniform['enhancement'].mean():.3f}")
    print(f"  Mean f_DM: {high_uniform['f_DM'].mean():.3f}")
    print(f"  Mean uniformity: {high_uniform['uniformity_obs'].mean():.3f}")
    
    print(f"\nLow vorticity uniformity (differential, N={len(low_uniform)}):")
    print(f"  Mean enhancement: {low_uniform['enhancement'].mean():.3f}")
    print(f"  Mean f_DM: {low_uniform['f_DM'].mean():.3f}")
    print(f"  Mean uniformity: {low_uniform['uniformity_obs'].mean():.3f}")
    
    # Statistical test
    t_stat, t_pval = stats.ttest_ind(high_uniform['enhancement'], low_uniform['enhancement'])
    print(f"\nT-test (high vs low uniformity): t = {t_stat:.3f}, p = {t_pval:.4f}")
    
    # ==========================================================================
    # MICROPHYSICS PREDICTION
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("MICROPHYSICS PREDICTION TEST")
    print("=" * 80)
    
    print("""
PREDICTION FROM TELEPARALLEL GRAVITY:
If torsion ~ vorticity, then:
    <T(x) T(x')>_c ∝ <ω(x)·ω(x')>_c

Galaxies with MORE UNIFORM vorticity (higher correlation) should show
MORE gravitational enhancement.

Solid-body rotation: ω = constant → <ω·ω'> = ω² (maximum correlation)
Differential rotation: ω varies → <ω·ω'> < ω² (reduced correlation)
""")
    
    if r_uniform > 0 and p_uniform < 0.05:
        print("✓ CONFIRMED: Higher vorticity uniformity → higher enhancement")
        print(f"  Correlation r = {r_uniform:.3f} with p = {p_uniform:.2e}")
    elif r_uniform < 0 and p_uniform < 0.05:
        print("✗ CONTRADICTED: Higher vorticity uniformity → LOWER enhancement")
        print(f"  Correlation r = {r_uniform:.3f} with p = {p_uniform:.2e}")
    else:
        print("~ INCONCLUSIVE: No significant correlation detected")
        print(f"  Correlation r = {r_uniform:.3f} with p = {p_uniform:.2e}")
    
    # ==========================================================================
    # TOP/BOTTOM GALAXIES
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("EXTREME GALAXIES")
    print("=" * 80)
    
    # Sort by vorticity uniformity
    df_sorted = df.sort_values('uniformity_obs', ascending=False)
    
    print("\nTop 10 by vorticity uniformity (most solid-body-like):")
    print(f"{'Galaxy':<15} {'Uniformity':>12} {'Enhancement':>12} {'f_DM':>10}")
    print("-" * 55)
    for _, row in df_sorted.head(10).iterrows():
        print(f"{row['name']:<15} {row['uniformity_obs']:>12.3f} "
              f"{row['enhancement']:>12.3f} {row['f_DM']:>10.3f}")
    
    print("\nBottom 10 by vorticity uniformity (most differential):")
    print(f"{'Galaxy':<15} {'Uniformity':>12} {'Enhancement':>12} {'f_DM':>10}")
    print("-" * 55)
    for _, row in df_sorted.tail(10).iterrows():
        print(f"{row['name']:<15} {row['uniformity_obs']:>12.3f} "
              f"{row['enhancement']:>12.3f} {row['f_DM']:>10.3f}")
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    
    output_dir = Path(__file__).parent / "vorticity_test_results"
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / "vorticity_analysis.csv", index=False)
    
    summary = {
        'n_galaxies': len(df),
        'r_uniformity_enhancement': float(r_uniform),
        'p_uniformity_enhancement': float(p_uniform),
        'r_flatness_enhancement': float(r_flat),
        'p_flatness_enhancement': float(p_flat),
        'r_Comega_fDM': float(r_Comega),
        'p_Comega_fDM': float(p_Comega),
        'high_uniform_enhancement': float(high_uniform['enhancement'].mean()),
        'low_uniform_enhancement': float(low_uniform['enhancement'].mean()),
        't_test_pval': float(t_pval),
    }
    
    with open(output_dir / "vorticity_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()



