#!/usr/bin/env python3
"""
RESEARCH: Amplitude vs Path Length for Individual Systems
==========================================================

This script explores what happens when we plot individual galaxies and clusters
on an amplitude vs path length chart, rather than just category averages.

For each system, we:
1. Estimate the path length L through baryons
2. Compute the "effective amplitude" A_eff needed to match observations
3. Plot all systems and compare to the theoretical curve

This is RESEARCH - not for the paper yet.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize_scalar
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Physical constants
c = 2.998e8
H0_SI = 2.27e-18
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

# Σ-Gravity parameters
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))
A_0 = np.exp(1 / (2 * np.pi))
XI_SCALE = 1 / (2 * np.pi)
L_0 = 0.40  # Reference path length (kpc)
N_EXP = 0.27  # Path length exponent

def h_function(g: np.ndarray) -> np.ndarray:
    """h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r: np.ndarray, R_d: float) -> np.ndarray:
    """W(r) = r/(ξ+r) where ξ = R_d/(2π)"""
    xi = XI_SCALE * R_d
    xi = max(xi, 0.01)
    return r / (xi + r)

def theoretical_amplitude(L: float, D: float = 0) -> float:
    """A = A₀ × [1 - D + D × (L/L₀)^n]"""
    return A_0 * (1 - D + D * (L / L_0)**N_EXP)

def load_sparc_galaxies(data_dir: Path) -> List[Dict]:
    """Load SPARC galaxies with path length estimates."""
    sparc_dir = data_dir / "Rotmod_LTG"
    if not sparc_dir.exists():
        print(f"SPARC data not found at {sparc_dir}")
        return []
    
    galaxies = []
    for gf in sorted(sparc_dir.glob("*_rotmod.dat")):
        data = []
        with open(gf) as f:
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
                    except ValueError:
                        continue
        
        if len(data) < 5:
            continue
        
        df = pd.DataFrame(data)
        
        # Apply M/L scaling
        V_bar_sq = (np.sign(df['V_gas']) * df['V_gas']**2 + 
                    0.5 * df['V_disk']**2 + 0.7 * df['V_bulge']**2)
        df['V_bar'] = np.sqrt(np.abs(V_bar_sq))
        
        valid = (df['V_bar'] > 0) & (df['R'] > 0) & (df['V_obs'] > 0)
        df = df[valid]
        
        if len(df) >= 5:
            # Estimate R_d from where V_disk peaks
            idx = len(df) // 3
            R_d = df['R'].iloc[idx] if idx > 0 else df['R'].iloc[-1] / 2
            
            # Estimate disk thickness: h ≈ 0.1-0.15 × R_d
            h_disk = 0.12 * R_d  # kpc
            
            # Path length = 2 × disk thickness
            L = 2 * h_disk
            
            galaxies.append({
                'name': gf.stem.replace('_rotmod', ''),
                'R': df['R'].values,
                'V_obs': df['V_obs'].values,
                'V_bar': df['V_bar'].values,
                'R_d': R_d,
                'h_disk': h_disk,
                'path_length': L,
                'type': 'disk',
                'D': 0,  # Disk-dominated
            })
    
    return galaxies

def load_clusters(data_dir: Path) -> List[Dict]:
    """Load Fox+ 2022 clusters with mass-dependent path lengths.
    
    Path length estimation:
    - R500 scales with M500 as R500 ∝ M500^(1/3) (virial scaling)
    - For M500 = 5×10^14 M_sun, R500 ≈ 1 Mpc
    - Path length L ≈ 2 × R500 (diameter through cluster core)
    """
    cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
    if not cluster_file.exists():
        print(f"Cluster data not found at {cluster_file}")
        return []
    
    df = pd.read_csv(cluster_file)
    df_valid = df[
        df['M500_1e14Msun'].notna() & 
        df['MSL_200kpc_1e12Msun'].notna() &
        (df['spec_z_constraint'] == 'yes')
    ].copy()
    
    clusters = []
    f_baryon = 0.15
    
    # Reference values for R500 scaling
    M500_ref = 5e14  # M_sun
    R500_ref = 1000  # kpc (≈ 1 Mpc for a massive cluster)
    
    for idx, row in df_valid.iterrows():
        M500 = row['M500_1e14Msun'] * 1e14  # M_sun
        M_bar_200 = 0.4 * f_baryon * M500  # Baryonic mass within 200 kpc
        M_lens_200 = row['MSL_200kpc_1e12Msun'] * 1e12  # Lensing mass
        
        # Mass-dependent R500 using virial scaling: R ∝ M^(1/3)
        R500 = R500_ref * (M500 / M500_ref)**(1/3)
        
        # Path length = 2 × R500 (diameter)
        L = 2 * R500
        
        clusters.append({
            'name': row['cluster'],
            'M_bar': M_bar_200,
            'M_lens': M_lens_200,
            'r_kpc': 200,
            'R500': R500,
            'path_length': L,
            'type': 'cluster',
            'D': 1,  # Dispersion-dominated
        })
    
    return clusters

def compute_effective_amplitude_galaxy(gal: Dict) -> Tuple[float, float]:
    """
    Compute the effective amplitude A_eff that best fits this galaxy's rotation curve.
    
    Returns: (A_eff, uncertainty)
    """
    R = gal['R']
    V_obs = gal['V_obs']
    V_bar = gal['V_bar']
    R_d = gal['R_d']
    
    # Compute g_bar at each radius
    g_bar = (V_bar * 1000)**2 / (R * kpc_to_m)
    
    # Compute W and h at each radius
    W = W_coherence(R, R_d)
    h = h_function(g_bar)
    
    # For each point: Σ_obs = (V_obs/V_bar)²
    # Σ = 1 + A × W × h
    # So: A = (Σ_obs - 1) / (W × h)
    
    Sigma_obs = (V_obs / V_bar)**2
    
    # Avoid division by zero
    Wh = W * h
    valid = Wh > 0.01
    
    if np.sum(valid) < 3:
        return np.nan, np.nan
    
    A_eff_points = (Sigma_obs[valid] - 1) / Wh[valid]
    
    # Use median to be robust to outliers
    A_eff = np.median(A_eff_points)
    A_std = np.std(A_eff_points) / np.sqrt(np.sum(valid))
    
    # Clip to reasonable range
    if A_eff < 0.1 or A_eff > 20:
        return np.nan, np.nan
    
    return A_eff, A_std

def compute_effective_amplitude_cluster(cl: Dict) -> Tuple[float, float]:
    """
    Compute the effective amplitude A_eff for a cluster from lensing mass.
    
    Returns: (A_eff, uncertainty)
    """
    M_bar = cl['M_bar']
    M_lens = cl['M_lens']
    r_kpc = cl['r_kpc']
    
    # Σ = M_lens / M_bar
    Sigma = M_lens / M_bar
    
    # g_bar at r_kpc
    r_m = r_kpc * kpc_to_m
    g_bar = G_const * M_bar * M_sun / r_m**2
    
    # h(g)
    h = h_function(np.array([g_bar]))[0]
    
    # For clusters, W ≈ 1
    W = 1.0
    
    # A = (Σ - 1) / (W × h)
    A_eff = (Sigma - 1) / (W * h)
    
    # Assume 20% uncertainty
    A_std = 0.2 * A_eff
    
    if A_eff < 0.5 or A_eff > 50:
        return np.nan, np.nan
    
    return A_eff, A_std

def main():
    print("=" * 80)
    print("RESEARCH: Amplitude vs Path Length for Individual Systems")
    print("=" * 80)
    
    # Find data directory
    script_dir = Path(__file__).resolve().parent.parent
    data_dir = script_dir / "data"
    
    # Load data
    print("\nLoading data...")
    galaxies = load_sparc_galaxies(data_dir)
    clusters = load_clusters(data_dir)
    
    print(f"  Loaded {len(galaxies)} SPARC galaxies")
    print(f"  Loaded {len(clusters)} clusters")
    
    if len(galaxies) == 0 and len(clusters) == 0:
        print("\nNo data found! Make sure data files are in place.")
        return
    
    # Compute effective amplitudes
    print("\nComputing effective amplitudes...")
    
    galaxy_data = []
    for gal in galaxies:
        A_eff, A_std = compute_effective_amplitude_galaxy(gal)
        if not np.isnan(A_eff):
            galaxy_data.append({
                'name': gal['name'],
                'L': gal['path_length'],
                'A_eff': A_eff,
                'A_std': A_std,
                'type': 'galaxy',
                'D': 0,
            })
    
    cluster_data = []
    for cl in clusters:
        A_eff, A_std = compute_effective_amplitude_cluster(cl)
        if not np.isnan(A_eff):
            cluster_data.append({
                'name': cl['name'],
                'L': cl['path_length'],
                'A_eff': A_eff,
                'A_std': A_std,
                'type': 'cluster',
                'D': 1,
            })
    
    print(f"  Valid galaxy amplitudes: {len(galaxy_data)}")
    print(f"  Valid cluster amplitudes: {len(cluster_data)}")
    
    # Combine data
    all_data = galaxy_data + cluster_data
    df = pd.DataFrame(all_data)
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    if len(galaxy_data) > 0:
        gal_L = [d['L'] for d in galaxy_data]
        gal_A = [d['A_eff'] for d in galaxy_data]
        print(f"\nGalaxies:")
        print(f"  Path length L: {np.min(gal_L):.2f} - {np.max(gal_L):.2f} kpc (median: {np.median(gal_L):.2f})")
        print(f"  Effective A: {np.min(gal_A):.2f} - {np.max(gal_A):.2f} (median: {np.median(gal_A):.2f})")
        print(f"  Theoretical A (D=0): {A_0:.3f}")
    
    if len(cluster_data) > 0:
        cl_L = [d['L'] for d in cluster_data]
        cl_A = [d['A_eff'] for d in cluster_data]
        print(f"\nClusters:")
        print(f"  Path length L: {np.min(cl_L):.0f} - {np.max(cl_L):.0f} kpc (median: {np.median(cl_L):.0f})")
        print(f"  Effective A: {np.min(cl_A):.1f} - {np.max(cl_A):.1f} (median: {np.median(cl_A):.1f})")
        L_median = np.median(cl_L)
        print(f"  Theoretical A (D=1, L={L_median:.0f}): {theoretical_amplitude(L_median, D=1):.2f}")
    
    # Create figure
    print("\n" + "=" * 80)
    print("CREATING FIGURE")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Theoretical curves (extend to 10,000 kpc to cover clusters)
    L_range = np.logspace(-0.5, 4, 100)
    A_disp = A_0 * (L_range / L_0)**N_EXP  # D=1, Σ-Gravity prediction
    
    # Σ-Gravity prediction (path-length dependent)
    ax.loglog(L_range, A_disp, 'b-', lw=2.5,
              label=r'Σ-Gravity: $A = A_0 (L/L_0)^{0.27}$')
    
    # MOND prediction: constant A ≈ 1 at all scales (scale-independent)
    ax.axhline(y=1.0, color='red', ls='--', lw=2, alpha=0.8,
               label='MOND: A ≈ 1 (scale-independent)')
    
    # GR (no dark matter): A = 0 (no enhancement)
    ax.axhline(y=0.15, color='gray', ls=':', lw=2, alpha=0.6,
               label='GR (no DM): A → 0')
    ax.text(0.15, 0.12, 'GR: A = 0', fontsize=9, color='gray', alpha=0.8)
    
    # Plot individual galaxies
    if len(galaxy_data) > 0:
        gal_L = np.array([d['L'] for d in galaxy_data])
        gal_A = np.array([d['A_eff'] for d in galaxy_data])
        gal_err = np.array([d['A_std'] for d in galaxy_data])
        
        ax.errorbar(gal_L, gal_A, yerr=gal_err, fmt='o', ms=4, alpha=0.5,
                    color='green', ecolor='green', capsize=0,
                    label=f'SPARC galaxies (N={len(galaxy_data)})')
    
    # Plot individual clusters
    if len(cluster_data) > 0:
        cl_L = np.array([d['L'] for d in cluster_data])
        cl_A = np.array([d['A_eff'] for d in cluster_data])
        cl_err = np.array([d['A_std'] for d in cluster_data])
        
        ax.errorbar(cl_L, cl_A, yerr=cl_err, fmt='^', ms=8, alpha=0.7,
                    color='red', ecolor='red', capsize=2,
                    label=f'Clusters (N={len(cluster_data)})')
    
    # Fit a power law to all data
    if len(all_data) > 10:
        L_all = np.array([d['L'] for d in all_data])
        A_all = np.array([d['A_eff'] for d in all_data])
        
        # Log-log linear fit
        log_L = np.log10(L_all)
        log_A = np.log10(A_all)
        
        # Fit: log(A) = n * log(L) + log(A0_fit)
        coeffs = np.polyfit(log_L, log_A, 1)
        n_fit = coeffs[0]
        A0_fit = 10**coeffs[1]
        
        L_fit = np.logspace(np.log10(L_all.min()), np.log10(L_all.max()), 50)
        A_fit = A0_fit * (L_fit)**n_fit
        
        ax.loglog(L_fit, A_fit, 'k:', lw=2, 
                  label=f'Best fit: A = {A0_fit:.2f} × L^{n_fit:.3f}')
        
        print(f"\nBest fit power law:")
        print(f"  A = {A0_fit:.3f} × L^{n_fit:.3f}")
        print(f"  (Theory predicts n = 0.27 for D=1 systems)")
    
    ax.set_xlabel('Path length through baryons L [kpc]', fontsize=12)
    ax.set_ylabel('Effective amplitude A', fontsize=12)
    ax.set_title('Amplitude vs Path Length: Individual Systems', fontsize=14)
    ax.set_xlim(0.1, 5000)
    ax.set_ylim(0.1, 50)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save
    output_dir = script_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    outpath = output_dir / "research_amplitude_vs_path_length.png"
    plt.savefig(outpath, dpi=150)
    print(f"\nSaved: {outpath}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

if __name__ == '__main__':
    main()

