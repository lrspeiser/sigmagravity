#!/usr/bin/env python3
"""
Y-COLLAPSE DIAGNOSTIC TEST
===========================

Tests whether the separable unification ansatz is correct:

    Σ = 1 + A(D,L) × W(r) × h(g)

By defining:
    Y ≡ (Σ_obs - 1) / h(g_bar)

If the factorization is right, then:
    Y ≈ A(D,L) × W(r)

This script performs three "collapse" checks:
1. SPARC galaxies: Does Y vs r/ξ trace a saturating curve?
2. Gaia/MW: Does Y(r) follow the same W(r) trend?
3. Clusters: Does Y_cluster ≈ A_cluster correlate with depth L?

Author: Leonard Speiser
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # 1/s (H0 = 70 km/s/Mpc)
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

# Critical acceleration
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 9.60e-11 m/s²

# Model parameters
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173
XI_SCALE = 1 / (2 * np.pi)  # ξ = R_d/(2π)
ML_DISK = 0.5
ML_BULGE = 0.7

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def h_function(g: np.ndarray) -> np.ndarray:
    """RAR piece: h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def W_model(r: np.ndarray, xi) -> np.ndarray:
    """Coherence window: W(r) = r/(ξ+r)"""
    xi = np.maximum(xi, 0.01)
    return r / (xi + r)


def compute_Y(g_obs: np.ndarray, g_bar: np.ndarray) -> np.ndarray:
    """
    Compute Y ≡ (Σ_obs - 1) / h(g_bar)
    
    If separable factorization is correct: Y ≈ A × W(r)
    """
    Sigma_obs = g_obs / g_bar
    h = h_function(g_bar)
    # Avoid division by very small h
    h = np.maximum(h, 1e-10)
    Y = (Sigma_obs - 1) / h
    return Y


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_sparc(data_dir: Path):
    """Load SPARC galaxies with all radius points."""
    sparc_dir = data_dir / "Rotmod_LTG"
    if not sparc_dir.exists():
        return None
    
    all_points = []
    
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
        
        # Apply M/L corrections
        df['V_disk_scaled'] = df['V_disk'] * np.sqrt(ML_DISK)
        df['V_bulge_scaled'] = df['V_bulge'] * np.sqrt(ML_BULGE)
        V_bar_sq = (np.sign(df['V_gas']) * df['V_gas']**2 + 
                    df['V_disk_scaled']**2 + df['V_bulge_scaled']**2)
        df['V_bar'] = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)
        
        valid = (df['V_bar'] > 0) & (df['R'] > 0) & (df['V_obs'] > 0)
        df = df[valid]
        
        if len(df) < 5:
            continue
        
        # Estimate R_d
        idx = len(df) // 3
        R_d = df['R'].iloc[idx] if idx > 0 else df['R'].iloc[-1] / 2
        xi = XI_SCALE * R_d
        
        # Estimate bulge fraction for D_eff
        total_sq = np.sum(df['V_disk']**2 + df['V_bulge']**2 + df['V_gas']**2)
        f_bulge = np.sum(df['V_bulge']**2) / max(total_sq, 1e-10)
        
        # Compute accelerations
        R_m = df['R'].values * kpc_to_m
        V_obs_ms = df['V_obs'].values * 1000
        V_bar_ms = df['V_bar'].values * 1000
        
        g_obs = V_obs_ms**2 / R_m
        g_bar = V_bar_ms**2 / R_m
        
        for i in range(len(df)):
            all_points.append({
                'galaxy': gf.stem.replace('_rotmod', ''),
                'R': df['R'].iloc[i],
                'R_d': R_d,
                'xi': xi,
                'r_over_xi': df['R'].iloc[i] / xi,
                'g_obs': g_obs[i],
                'g_bar': g_bar[i],
                'V_obs': df['V_obs'].iloc[i],
                'V_bar': df['V_bar'].iloc[i],
                'f_bulge': f_bulge,
                'system': 'galaxy'
            })
    
    return pd.DataFrame(all_points)


def load_gaia(data_dir: Path):
    """Load Gaia/MW data."""
    gaia_file = data_dir / "gaia" / "eilers_apogee_6d_disk.csv"
    if not gaia_file.exists():
        return None
    
    df = pd.read_csv(gaia_file)
    df['v_phi_obs'] = -df['v_phi']  # Sign correction
    
    # McMillan 2017 baryonic model
    R = df['R_gal'].values
    MW_VBAR_SCALE = 1.16
    M_disk = 4.6e10 * MW_VBAR_SCALE**2
    M_bulge = 1.0e10 * MW_VBAR_SCALE**2
    M_gas = 1.0e10 * MW_VBAR_SCALE**2
    G_kpc = 4.302e-6
    
    v2_disk = G_kpc * M_disk * R**2 / (R**2 + 3.3**2)**1.5
    v2_bulge = G_kpc * M_bulge * R / (R + 0.5)**2
    v2_gas = G_kpc * M_gas * R**2 / (R**2 + 7.0**2)**1.5
    V_bar = np.sqrt(v2_disk + v2_bulge + v2_gas)
    
    # MW parameters
    R_d_mw = 2.6
    xi_mw = XI_SCALE * R_d_mw
    
    # Bin by radius
    R_bins = np.arange(4, 15, 0.5)
    points = []
    
    for i in range(len(R_bins) - 1):
        mask = (R >= R_bins[i]) & (R < R_bins[i + 1])
        if mask.sum() < 50:
            continue
        
        R_mean = R[mask].mean()
        V_obs_mean = df.loc[mask, 'v_phi_obs'].mean()
        V_bar_mean = V_bar[mask].mean()
        
        R_m = R_mean * kpc_to_m
        g_obs = (V_obs_mean * 1000)**2 / R_m
        g_bar = (V_bar_mean * 1000)**2 / R_m
        
        points.append({
            'R': R_mean,
            'R_d': R_d_mw,
            'xi': xi_mw,
            'r_over_xi': R_mean / xi_mw,
            'g_obs': g_obs,
            'g_bar': g_bar,
            'V_obs': V_obs_mean,
            'V_bar': V_bar_mean,
            'f_bulge': 0.1,  # MW is mostly disk
            'system': 'MW'
        })
    
    return pd.DataFrame(points)


def load_clusters(data_dir: Path):
    """Load cluster data."""
    cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
    if not cluster_file.exists():
        return None
    
    df = pd.read_csv(cluster_file)
    
    # Filter
    df_valid = df[
        df['M500_1e14Msun'].notna() & 
        df['MSL_200kpc_1e12Msun'].notna() &
        (df['spec_z_constraint'] == 'yes') &
        (df['M500_1e14Msun'] > 2.0)
    ].copy()
    
    points = []
    f_baryon = 0.15
    r_kpc = 200
    
    for idx, row in df_valid.iterrows():
        M500 = row['M500_1e14Msun'] * 1e14
        M_bar_200 = 0.4 * f_baryon * M500
        M_lens_200 = row['MSL_200kpc_1e12Msun'] * 1e12
        
        r_m = r_kpc * kpc_to_m
        g_bar = G_const * M_bar_200 * M_sun / r_m**2
        g_obs = G_const * M_lens_200 * M_sun / r_m**2
        
        # Estimate path length from M500 (proxy for size)
        # Larger clusters have longer path lengths
        L_cluster = 400 + 200 * (M500 / 5e14)  # Rough scaling
        
        # Cluster coherence scale (dispersion-dominated)
        xi_cluster = 20  # kpc, typical
        
        points.append({
            'cluster': row['cluster'],
            'R': r_kpc,
            'xi': xi_cluster,
            'r_over_xi': r_kpc / xi_cluster,
            'g_obs': g_obs,
            'g_bar': g_bar,
            'M500': M500,
            'M_lens': M_lens_200,
            'M_bar': M_bar_200,
            'L': L_cluster,
            'f_bulge': 1.0,  # Fully 3D
            'system': 'cluster'
        })
    
    return pd.DataFrame(points)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def test_sparc_collapse(df_sparc, output_dir):
    """Test 1: Do SPARC galaxies collapse onto W(r) when scaled by r/ξ?"""
    print("\n" + "="*70)
    print("TEST 1: SPARC Y-COLLAPSE")
    print("="*70)
    
    # Compute Y for all points
    df_sparc['Y'] = compute_Y(df_sparc['g_obs'].values, df_sparc['g_bar'].values)
    
    # Filter valid Y values
    valid = np.isfinite(df_sparc['Y']) & (df_sparc['Y'] > 0) & (df_sparc['Y'] < 100)
    df_valid = df_sparc[valid].copy()
    
    print(f"Valid points: {len(df_valid)} / {len(df_sparc)}")
    
    # Compute expected W(r)
    df_valid['W_expected'] = W_model(df_valid['R'].values, df_valid['xi'].values)
    
    # If Y ≈ A × W(r), then Y/W ≈ A (constant)
    df_valid['Y_over_W'] = df_valid['Y'] / np.maximum(df_valid['W_expected'], 0.01)
    
    # Statistics
    mean_A = df_valid['Y_over_W'].median()
    std_A = df_valid['Y_over_W'].std()
    
    print(f"\nIf Y = A × W(r), then Y/W should be constant:")
    print(f"  Median Y/W = {mean_A:.3f}")
    print(f"  Std Y/W = {std_A:.3f}")
    print(f"  Expected A_galaxy = {A_0:.3f}")
    print(f"  Ratio = {mean_A/A_0:.3f}")
    
    # Correlation between Y and W
    r_pearson, p_pearson = pearsonr(df_valid['W_expected'], df_valid['Y'])
    print(f"\nCorrelation Y vs W(r):")
    print(f"  Pearson r = {r_pearson:.3f}, p = {p_pearson:.2e}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Y vs r/ξ (should saturate)
    ax = axes[0, 0]
    
    # Bin by r/ξ for cleaner visualization
    r_xi_bins = np.logspace(-1, 2, 30)
    bin_centers = []
    bin_Y_median = []
    bin_Y_16 = []
    bin_Y_84 = []
    
    for i in range(len(r_xi_bins) - 1):
        mask = (df_valid['r_over_xi'] >= r_xi_bins[i]) & (df_valid['r_over_xi'] < r_xi_bins[i+1])
        if mask.sum() > 10:
            bin_centers.append(np.sqrt(r_xi_bins[i] * r_xi_bins[i+1]))
            Y_vals = df_valid.loc[mask, 'Y'].values
            bin_Y_median.append(np.median(Y_vals))
            bin_Y_16.append(np.percentile(Y_vals, 16))
            bin_Y_84.append(np.percentile(Y_vals, 84))
    
    ax.fill_between(bin_centers, bin_Y_16, bin_Y_84, alpha=0.3, color='blue')
    ax.plot(bin_centers, bin_Y_median, 'b-', linewidth=2, label='SPARC data (median)')
    
    # Expected curve: Y = A × W(r/ξ)
    r_xi_theory = np.logspace(-1, 2, 100)
    W_theory = r_xi_theory / (1 + r_xi_theory)
    Y_theory = A_0 * W_theory
    ax.plot(r_xi_theory, Y_theory, 'r--', linewidth=2, label=f'Theory: Y = {A_0:.2f} × W(r/ξ)')
    
    ax.set_xscale('log')
    ax.set_xlabel('r / ξ', fontsize=12)
    ax.set_ylabel('Y = (Σ_obs - 1) / h(g)', fontsize=12)
    ax.set_title('SPARC: Y vs r/ξ (should saturate at A)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)
    
    # Panel 2: Y/W histogram (should peak at A)
    ax = axes[0, 1]
    Y_over_W = df_valid['Y_over_W'].values
    Y_over_W = Y_over_W[(Y_over_W > 0) & (Y_over_W < 10)]
    ax.hist(Y_over_W, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(A_0, color='red', linestyle='--', linewidth=2, label=f'Expected A₀ = {A_0:.3f}')
    ax.axvline(np.median(Y_over_W), color='green', linestyle='-', linewidth=2, 
               label=f'Observed median = {np.median(Y_over_W):.3f}')
    ax.set_xlabel('Y / W(r)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('SPARC: Y/W distribution (should peak at A)', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 5)
    
    # Panel 3: Y vs W (should be linear through origin)
    ax = axes[1, 0]
    ax.scatter(df_valid['W_expected'], df_valid['Y'], alpha=0.1, s=5, c='blue')
    
    # Fit line through origin
    W_fit = df_valid['W_expected'].values
    Y_fit = df_valid['Y'].values
    valid_fit = np.isfinite(W_fit) & np.isfinite(Y_fit) & (W_fit > 0.01) & (Y_fit > 0)
    A_fitted = np.sum(W_fit[valid_fit] * Y_fit[valid_fit]) / np.sum(W_fit[valid_fit]**2)
    
    W_line = np.linspace(0, 1, 100)
    ax.plot(W_line, A_fitted * W_line, 'r-', linewidth=2, label=f'Fit: Y = {A_fitted:.3f} × W')
    ax.plot(W_line, A_0 * W_line, 'g--', linewidth=2, label=f'Theory: Y = {A_0:.3f} × W')
    
    ax.set_xlabel('W(r) = r/(ξ+r)', fontsize=12)
    ax.set_ylabel('Y = (Σ_obs - 1) / h(g)', fontsize=12)
    ax.set_title('SPARC: Y vs W (should be linear)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)
    
    # Panel 4: Residuals by galaxy type (bulge fraction)
    ax = axes[1, 1]
    
    # Color by bulge fraction
    scatter = ax.scatter(df_valid['r_over_xi'], df_valid['Y'] / np.maximum(df_valid['W_expected'], 0.01),
                        c=df_valid['f_bulge'], cmap='coolwarm', alpha=0.3, s=10)
    ax.axhline(A_0, color='red', linestyle='--', linewidth=2, label=f'Expected A₀ = {A_0:.3f}')
    
    ax.set_xscale('log')
    ax.set_xlabel('r / ξ', fontsize=12)
    ax.set_ylabel('Y / W(r) = effective A', fontsize=12)
    ax.set_title('SPARC: Effective A by bulge fraction', fontsize=14)
    ax.legend()
    ax.set_ylim(0, 5)
    plt.colorbar(scatter, ax=ax, label='Bulge fraction')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sparc_y_collapse.png', dpi=150)
    plt.close()
    
    print(f"\nPlot saved: {output_dir / 'sparc_y_collapse.png'}")
    
    return {
        'median_A': mean_A,
        'std_A': std_A,
        'fitted_A': A_fitted,
        'correlation': r_pearson,
        'n_points': len(df_valid)
    }


def test_mw_collapse(df_mw, output_dir):
    """Test 2: Does MW follow the same W(r) trend?"""
    print("\n" + "="*70)
    print("TEST 2: MILKY WAY Y-COLLAPSE")
    print("="*70)
    
    if df_mw is None or len(df_mw) == 0:
        print("No MW data available")
        return None
    
    # Compute Y
    df_mw['Y'] = compute_Y(df_mw['g_obs'].values, df_mw['g_bar'].values)
    df_mw['W_expected'] = W_model(df_mw['R'].values, df_mw['xi'].values)
    df_mw['Y_over_W'] = df_mw['Y'] / np.maximum(df_mw['W_expected'], 0.01)
    
    print(f"MW radial bins: {len(df_mw)}")
    print(f"\nY/W values by radius:")
    for _, row in df_mw.iterrows():
        print(f"  R={row['R']:.1f} kpc: Y={row['Y']:.3f}, W={row['W_expected']:.3f}, Y/W={row['Y_over_W']:.3f}")
    
    mean_A = df_mw['Y_over_W'].median()
    print(f"\nMedian Y/W = {mean_A:.3f} (expected A₀ = {A_0:.3f})")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.plot(df_mw['r_over_xi'], df_mw['Y'], 'bo-', markersize=8, linewidth=2, label='MW data')
    
    r_xi_theory = np.linspace(0, df_mw['r_over_xi'].max() * 1.2, 100)
    W_theory = r_xi_theory / (1 + r_xi_theory)
    ax.plot(r_xi_theory, A_0 * W_theory, 'r--', linewidth=2, label=f'Theory: Y = {A_0:.2f} × W(r/ξ)')
    
    ax.set_xlabel('r / ξ', fontsize=12)
    ax.set_ylabel('Y = (Σ_obs - 1) / h(g)', fontsize=12)
    ax.set_title('Milky Way: Y vs r/ξ', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(df_mw['R'], df_mw['Y_over_W'], 'go-', markersize=8, linewidth=2)
    ax.axhline(A_0, color='red', linestyle='--', linewidth=2, label=f'Expected A₀ = {A_0:.3f}')
    ax.axhline(mean_A, color='blue', linestyle='-', linewidth=2, label=f'Median = {mean_A:.3f}')
    
    ax.set_xlabel('R (kpc)', fontsize=12)
    ax.set_ylabel('Y / W(r) = effective A', fontsize=12)
    ax.set_title('Milky Way: Effective A vs radius', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mw_y_collapse.png', dpi=150)
    plt.close()
    
    print(f"\nPlot saved: {output_dir / 'mw_y_collapse.png'}")
    
    return {'median_A': mean_A, 'n_bins': len(df_mw)}


def test_cluster_amplitude(df_clusters, output_dir):
    """Test 3: Do clusters give Y ≈ A_cluster, correlating with depth L?"""
    print("\n" + "="*70)
    print("TEST 3: CLUSTER AMPLITUDE")
    print("="*70)
    
    if df_clusters is None or len(df_clusters) == 0:
        print("No cluster data available")
        return None
    
    # Compute Y for clusters
    df_clusters['Y'] = compute_Y(df_clusters['g_obs'].values, df_clusters['g_bar'].values)
    
    # For clusters at r >> ξ, W ≈ 1, so Y ≈ A_cluster directly
    df_clusters['W_expected'] = W_model(df_clusters['R'].values, df_clusters['xi'].values)
    
    print(f"Clusters: {len(df_clusters)}")
    print(f"\nAt r=200 kpc, W ≈ {df_clusters['W_expected'].mean():.3f} (should be ~1)")
    print(f"\nY values (should equal A_cluster ≈ 8.45):")
    print(f"  Mean Y = {df_clusters['Y'].mean():.2f}")
    print(f"  Median Y = {df_clusters['Y'].median():.2f}")
    print(f"  Std Y = {df_clusters['Y'].std():.2f}")
    
    # Correlation with path length proxy
    r_L, p_L = pearsonr(df_clusters['L'], df_clusters['Y'])
    print(f"\nCorrelation Y vs L (path length):")
    print(f"  Pearson r = {r_L:.3f}, p = {p_L:.3f}")
    
    # Correlation with M500 (another size proxy)
    r_M, p_M = pearsonr(df_clusters['M500'], df_clusters['Y'])
    print(f"\nCorrelation Y vs M500:")
    print(f"  Pearson r = {r_M:.3f}, p = {p_M:.3f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Y distribution
    ax = axes[0]
    ax.hist(df_clusters['Y'], bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(8.45, color='red', linestyle='--', linewidth=2, label='Expected A_cluster = 8.45')
    ax.axvline(df_clusters['Y'].median(), color='green', linestyle='-', linewidth=2,
               label=f'Median = {df_clusters["Y"].median():.2f}')
    ax.set_xlabel('Y = (Σ_obs - 1) / h(g)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Clusters: Y distribution (should peak at A_cluster)', fontsize=14)
    ax.legend()
    
    # Panel 2: Y vs L (path length)
    ax = axes[1]
    ax.scatter(df_clusters['L'], df_clusters['Y'], s=50, alpha=0.7, c='blue', edgecolor='black')
    
    # Expected scaling: A = A_0 × (L/L_0)^n
    L_theory = np.linspace(df_clusters['L'].min(), df_clusters['L'].max(), 100)
    L_0 = 0.40
    n = 0.27
    A_theory = A_0 * (L_theory / L_0)**n
    ax.plot(L_theory, A_theory, 'r--', linewidth=2, label=f'Theory: A = {A_0:.2f} × (L/0.4)^0.27')
    
    ax.set_xlabel('Path length L (kpc)', fontsize=12)
    ax.set_ylabel('Y ≈ A_cluster', fontsize=12)
    ax.set_title(f'Clusters: Y vs L (r = {r_L:.2f})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Y vs M500
    ax = axes[2]
    ax.scatter(df_clusters['M500'] / 1e14, df_clusters['Y'], s=50, alpha=0.7, c='blue', edgecolor='black')
    ax.set_xlabel('M500 (10¹⁴ M☉)', fontsize=12)
    ax.set_ylabel('Y ≈ A_cluster', fontsize=12)
    ax.set_title(f'Clusters: Y vs M500 (r = {r_M:.2f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_amplitude.png', dpi=150)
    plt.close()
    
    print(f"\nPlot saved: {output_dir / 'cluster_amplitude.png'}")
    
    return {
        'mean_Y': df_clusters['Y'].mean(),
        'median_Y': df_clusters['Y'].median(),
        'std_Y': df_clusters['Y'].std(),
        'r_L': r_L,
        'r_M': r_M,
        'n_clusters': len(df_clusters)
    }


def test_continuous_D(df_sparc, output_dir):
    """Test continuous D_eff instead of discrete D=0/1."""
    print("\n" + "="*70)
    print("TEST 4: CONTINUOUS DIMENSIONALITY")
    print("="*70)
    
    # Compute Y and effective A
    df_sparc['Y'] = compute_Y(df_sparc['g_obs'].values, df_sparc['g_bar'].values)
    df_sparc['W_expected'] = W_model(df_sparc['R'].values, df_sparc['xi'].values)
    df_sparc['A_eff'] = df_sparc['Y'] / np.maximum(df_sparc['W_expected'], 0.01)
    
    valid = np.isfinite(df_sparc['A_eff']) & (df_sparc['A_eff'] > 0) & (df_sparc['A_eff'] < 10)
    df_valid = df_sparc[valid].copy()
    
    # Use bulge fraction as proxy for D_eff
    df_valid['D_eff'] = df_valid['f_bulge']
    
    # Bin by D_eff
    D_bins = np.linspace(0, 0.5, 10)
    bin_centers = []
    bin_A_median = []
    bin_A_16 = []
    bin_A_84 = []
    
    for i in range(len(D_bins) - 1):
        mask = (df_valid['D_eff'] >= D_bins[i]) & (df_valid['D_eff'] < D_bins[i+1])
        if mask.sum() > 50:
            bin_centers.append((D_bins[i] + D_bins[i+1]) / 2)
            A_vals = df_valid.loc[mask, 'A_eff'].values
            bin_A_median.append(np.median(A_vals))
            bin_A_16.append(np.percentile(A_vals, 16))
            bin_A_84.append(np.percentile(A_vals, 84))
    
    print(f"Effective A by D_eff (bulge fraction):")
    for i, (D, A) in enumerate(zip(bin_centers, bin_A_median)):
        print(f"  D_eff = {D:.2f}: A_eff = {A:.3f}")
    
    # Correlation
    r_D, p_D = pearsonr(df_valid['D_eff'], df_valid['A_eff'])
    print(f"\nCorrelation A_eff vs D_eff: r = {r_D:.3f}, p = {p_D:.2e}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.fill_between(bin_centers, bin_A_16, bin_A_84, alpha=0.3, color='blue')
    ax.plot(bin_centers, bin_A_median, 'b-', linewidth=2, marker='o', label='SPARC data')
    ax.axhline(A_0, color='red', linestyle='--', linewidth=2, label=f'A₀ = {A_0:.3f}')
    
    ax.set_xlabel('D_eff = bulge fraction (σ²/(σ²+v²) proxy)', fontsize=12)
    ax.set_ylabel('Effective A = Y / W(r)', fontsize=12)
    ax.set_title('SPARC: Does A increase with "3D-ness"?', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.scatter(df_valid['D_eff'], df_valid['A_eff'], alpha=0.1, s=5, c='blue')
    ax.set_xlabel('D_eff = bulge fraction', fontsize=12)
    ax.set_ylabel('Effective A = Y / W(r)', fontsize=12)
    ax.set_title(f'SPARC: A vs D_eff (r = {r_D:.3f})', fontsize=14)
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'continuous_D.png', dpi=150)
    plt.close()
    
    print(f"\nPlot saved: {output_dir / 'continuous_D.png'}")
    
    return {'r_D': r_D, 'p_D': p_D}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("Y-COLLAPSE DIAGNOSTIC TEST")
    print("Testing separable unification: Y = (Σ-1)/h ≈ A × W(r)")
    print("="*70)
    
    # Setup
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent / "data"
    output_dir = script_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    df_sparc = load_sparc(data_dir)
    print(f"  SPARC: {len(df_sparc) if df_sparc is not None else 0} points")
    
    df_mw = load_gaia(data_dir)
    print(f"  MW: {len(df_mw) if df_mw is not None else 0} bins")
    
    df_clusters = load_clusters(data_dir)
    print(f"  Clusters: {len(df_clusters) if df_clusters is not None else 0}")
    
    # Run tests
    results = {}
    
    if df_sparc is not None and len(df_sparc) > 0:
        results['sparc'] = test_sparc_collapse(df_sparc, output_dir)
    
    if df_mw is not None and len(df_mw) > 0:
        results['mw'] = test_mw_collapse(df_mw, output_dir)
    
    if df_clusters is not None and len(df_clusters) > 0:
        results['clusters'] = test_cluster_amplitude(df_clusters, output_dir)
    
    if df_sparc is not None and len(df_sparc) > 0:
        results['continuous_D'] = test_continuous_D(df_sparc, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nKey question: Does Y = (Σ-1)/h factorize as A × W(r)?")
    print()
    
    if 'sparc' in results:
        r = results['sparc']
        print(f"SPARC galaxies:")
        print(f"  Fitted A from Y/W: {r['fitted_A']:.3f} (expected {A_0:.3f})")
        print(f"  Y vs W correlation: r = {r['correlation']:.3f}")
        print(f"  Verdict: {'✓ GOOD' if r['correlation'] > 0.5 else '✗ WEAK'} factorization")
    
    if 'mw' in results:
        r = results['mw']
        print(f"\nMilky Way:")
        print(f"  Median A from Y/W: {r['median_A']:.3f} (expected {A_0:.3f})")
    
    if 'clusters' in results:
        r = results['clusters']
        print(f"\nClusters:")
        print(f"  Mean Y (≈ A_cluster): {r['mean_Y']:.2f} (expected ~8.45)")
        print(f"  Y vs L correlation: r = {r['r_L']:.3f}")
        print(f"  Verdict: {'✓ Path length matters' if r['r_L'] > 0.3 else '✗ No L dependence'}")
    
    if 'continuous_D' in results:
        r = results['continuous_D']
        print(f"\nContinuous D_eff test:")
        print(f"  A vs D_eff correlation: r = {r['r_D']:.3f}")
        print(f"  Verdict: {'✓ A increases with 3D-ness' if r['r_D'] > 0.1 else '✗ No D dependence'}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
If the separable factorization Σ = 1 + A(D,L) × W(r) × h(g) is correct:

1. SPARC Y vs r/ξ should trace a saturating curve (W = r/(ξ+r))
   → This tests the "near vs far" spatial dependence

2. Y/W should be approximately constant = A for galaxies
   → This tests whether h(g) captures the full acceleration dependence

3. Cluster Y values (at r >> ξ where W ≈ 1) should equal A_cluster
   → This measures the amplitude directly

4. Cluster A should correlate with path length L
   → This tests the "geometry/depth" amplitude scaling

5. Galaxy A should increase with bulge fraction (D_eff)
   → This tests continuous transition from 2D to 3D
""")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

