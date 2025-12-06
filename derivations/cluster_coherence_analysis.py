#!/usr/bin/env python3
"""
Cluster Coherence Analysis

This script investigates whether the coherence-orbital timescale relationship
found in galaxies also applies to galaxy clusters.

For clusters, we analyze:
1. Per-cluster fitted coherence parameters (r0, A)
2. Correlations with cluster physical properties
3. Whether orbital/dynamical timescales predict coherence

Physical proxies for clusters:
- Total mass (M_lens)
- Baryonic mass (M_bar = M_gas + M_star)
- Gas fraction (f_gas)
- Stellar fraction (f_star)
- Characteristic radius (R_200, R_500)
- Velocity dispersion (σ_v)
- Dynamical time (t_dyn = R/σ)
- Crossing time (t_cross = R/v_circ)
- Relaxation state (if available)
- Redshift

Usage:
    python cluster_coherence_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import minimize_scalar, minimize
import json
import math
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # 1/s (70 km/s/Mpc)
kpc_to_m = 3.086e19
Mpc_to_m = 3.086e22
G_const = 6.674e-11
M_sun = 1.989e30

# Critical acceleration
g_dagger = c * H0_SI / (4 * math.sqrt(math.pi))

# Hubble time
t_Hubble_Gyr = 14.0  # Gyr

# Fixed global parameters
A_COEFF = 1.6
B_COEFF = 109.0
G_CLUSTER = 1.0  # Spherical geometry
A_CLUSTER_GLOBAL = np.sqrt(A_COEFF + B_COEFF * G_CLUSTER**2)
R0_GLOBAL = 5.0  # kpc


@dataclass
class ClusterPhysics:
    """Container for cluster data and derived physical properties."""
    name: str
    
    # Basic masses (in M_sun)
    M_gas: float = 0.0
    M_star: float = 0.0
    M_bar: float = 0.0
    M_lens: float = 0.0
    
    # Radii
    R_200: float = 200.0  # kpc (default)
    R_500: float = 0.0
    
    # Redshift
    z: float = 0.0
    
    # Derived properties
    gas_fraction: float = 0.0
    star_fraction: float = 0.0
    M_ratio: float = 0.0  # M_lens / M_bar
    
    # Dynamical properties
    v_circ: float = 0.0  # Circular velocity at R_200 (km/s)
    sigma_v: float = 0.0  # Velocity dispersion (km/s)
    t_dyn: float = 0.0  # Dynamical time (Gyr)
    t_cross: float = 0.0  # Crossing time (Gyr)
    orbital_periods: float = 0.0  # Number of orbits in Hubble time
    
    # Acceleration
    g_bar: float = 0.0  # Baryonic acceleration at R_200
    g_ratio: float = 0.0  # g_bar / g_dagger
    
    # Fit quality with global parameters
    M_pred_global: float = 0.0
    ratio_global: float = 0.0
    
    # Fitted per-cluster parameters
    r0_fitted: float = 0.0
    A_fitted: float = 0.0
    M_pred_fitted: float = 0.0
    ratio_fitted: float = 0.0
    improvement: float = 0.0


def h_function(g: float) -> float:
    """Enhancement function h(g)."""
    g = max(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def f_path(r: float, r0: float) -> float:
    """Path-length coherence factor f(r) = r/(r+r0)."""
    return r / (r + r0)


def predict_cluster_mass(M_bar: float, r_kpc: float, r0: float, A: float) -> float:
    """Predict cluster total mass using Σ-Gravity."""
    r_m = r_kpc * kpc_to_m
    g_bar = G_const * M_bar * M_sun / r_m**2
    
    h = h_function(g_bar)
    f = f_path(r_kpc, r0)
    
    Sigma = 1 + A * f * h
    return M_bar * Sigma


def load_cluster_data(data_dir: Path) -> List[ClusterPhysics]:
    """Load cluster data from Fox+ 2022 catalog."""
    cluster_file = data_dir / "clusters" / "fox2022_table1.dat"
    
    clusters = []
    
    if not cluster_file.exists():
        print(f"Warning: Cluster file not found at {cluster_file}")
        # Return synthetic data for testing
        return create_synthetic_clusters()
    
    with open(cluster_file) as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        
        parts = line.split()
        if len(parts) < 15:
            continue
        
        # Find where numeric data starts
        idx = 0
        for i, p in enumerate(parts):
            try:
                float(p)
                idx = i
                break
            except:
                continue
        
        try:
            # Extract cluster name
            name = '_'.join(parts[:idx])
            
            # Extract redshift (first numeric column)
            z = float(parts[idx])
            
            # Extract masses (columns vary, need to parse carefully)
            # Typical format: z, R_200, M_200, M_gas, M_gas_err, M_star, M_star_err, ..., M_lens
            
            # Try to extract key values
            M_gas_str = parts[idx + 3] if idx + 3 < len(parts) else '---'
            M_star_str = parts[idx + 6] if idx + 6 < len(parts) else '---'
            M_lens_str = parts[idx + 10] if idx + 10 < len(parts) else '---'
            
            if M_gas_str == '---' or M_star_str == '---' or M_lens_str == '---':
                continue
            
            M_gas = float(M_gas_str) * 1e12  # Convert to M_sun
            M_star = float(M_star_str) * 1e12
            M_lens = float(M_lens_str) * 1e12
            
            if M_lens <= 0:
                continue
            
            cluster = ClusterPhysics(
                name=name,
                M_gas=M_gas,
                M_star=M_star,
                M_bar=M_gas + M_star,
                M_lens=M_lens,
                z=z,
                R_200=200.0  # Default, could extract from data
            )
            
            clusters.append(cluster)
            
        except (ValueError, IndexError):
            continue
    
    if len(clusters) < 10:
        print(f"Warning: Only {len(clusters)} clusters loaded, adding synthetic data")
        clusters.extend(create_synthetic_clusters())
    
    return clusters


def create_synthetic_clusters() -> List[ClusterPhysics]:
    """Create synthetic cluster data for testing."""
    # Based on typical cluster properties from literature
    cluster_data = [
        # (name, M_gas, M_star, M_lens, z, R_200)
        ('Abell_2744', 1.5e12, 0.3e12, 2.0e14, 0.308, 200),
        ('Abell_370', 2.0e12, 0.4e12, 3.5e14, 0.375, 220),
        ('MACS_0416', 1.2e12, 0.25e12, 1.8e14, 0.396, 190),
        ('MACS_0717', 1.5e12, 0.35e12, 3.0e14, 0.545, 210),
        ('MACS_1149', 1.8e12, 0.4e12, 2.5e14, 0.544, 200),
        ('Abell_1689', 2.2e12, 0.5e12, 2.8e14, 0.183, 230),
        ('Abell_2218', 1.0e12, 0.2e12, 1.5e14, 0.171, 180),
        ('CL0024', 0.8e12, 0.15e12, 1.2e14, 0.395, 170),
        ('MS2137', 0.6e12, 0.12e12, 0.9e14, 0.313, 160),
        ('RXJ1347', 2.5e12, 0.6e12, 4.0e14, 0.451, 250),
        ('Abell_383', 0.9e12, 0.18e12, 1.3e14, 0.187, 175),
        ('Abell_611', 1.1e12, 0.22e12, 1.6e14, 0.288, 185),
        ('Abell_963', 1.3e12, 0.28e12, 1.9e14, 0.206, 195),
        ('Abell_1703', 1.4e12, 0.3e12, 2.1e14, 0.281, 200),
        ('Abell_2261', 1.7e12, 0.38e12, 2.4e14, 0.224, 210),
        # Add some low-mass and high-mass outliers
        ('Low_mass_1', 0.3e12, 0.05e12, 0.4e14, 0.2, 120),
        ('Low_mass_2', 0.4e12, 0.08e12, 0.5e14, 0.25, 130),
        ('High_mass_1', 3.0e12, 0.7e12, 5.0e14, 0.3, 280),
        ('High_mass_2', 3.5e12, 0.8e12, 6.0e14, 0.35, 300),
        # Add some high-z clusters
        ('High_z_1', 1.0e12, 0.2e12, 1.4e14, 0.8, 180),
        ('High_z_2', 1.2e12, 0.25e12, 1.7e14, 1.0, 190),
        # Add some gas-rich and gas-poor
        ('Gas_rich_1', 2.0e12, 0.1e12, 2.2e14, 0.3, 200),
        ('Gas_poor_1', 0.5e12, 0.5e12, 1.5e14, 0.25, 180),
    ]
    
    clusters = []
    for name, M_gas, M_star, M_lens, z, R_200 in cluster_data:
        cluster = ClusterPhysics(
            name=name,
            M_gas=M_gas,
            M_star=M_star,
            M_bar=M_gas + M_star,
            M_lens=M_lens,
            z=z,
            R_200=R_200
        )
        clusters.append(cluster)
    
    return clusters


def compute_cluster_physics(cluster: ClusterPhysics) -> None:
    """Compute derived physical properties for a cluster."""
    
    # Fractions
    if cluster.M_bar > 0:
        cluster.gas_fraction = cluster.M_gas / cluster.M_bar
        cluster.star_fraction = cluster.M_star / cluster.M_bar
    
    cluster.M_ratio = cluster.M_lens / cluster.M_bar if cluster.M_bar > 0 else 0
    
    # Circular velocity at R_200
    # v_circ = sqrt(G * M_lens / R)
    R_m = cluster.R_200 * kpc_to_m
    cluster.v_circ = np.sqrt(G_const * cluster.M_lens * M_sun / R_m) / 1000  # km/s
    
    # Velocity dispersion (approximate from virial theorem: σ ≈ v_circ / sqrt(2))
    cluster.sigma_v = cluster.v_circ / np.sqrt(2)
    
    # Dynamical time: t_dyn = R / σ
    # In Gyr: t = R(kpc) * kpc_to_m / (σ(km/s) * 1000) / (1e9 * 3.15e7)
    cluster.t_dyn = cluster.R_200 * kpc_to_m / (cluster.sigma_v * 1000) / (1e9 * 3.15e7)
    
    # Crossing time: t_cross = 2R / v_circ
    cluster.t_cross = 2 * cluster.R_200 * kpc_to_m / (cluster.v_circ * 1000) / (1e9 * 3.15e7)
    
    # Number of orbital periods in Hubble time
    T_orbit = 2 * np.pi * cluster.R_200 * kpc_to_m / (cluster.v_circ * 1000) / (1e9 * 3.15e7)  # Gyr
    cluster.orbital_periods = t_Hubble_Gyr / T_orbit if T_orbit > 0 else 0
    
    # Baryonic acceleration at R_200
    cluster.g_bar = G_const * cluster.M_bar * M_sun / R_m**2
    cluster.g_ratio = cluster.g_bar / g_dagger


def fit_cluster_global(cluster: ClusterPhysics) -> None:
    """Compute prediction with global parameters."""
    cluster.M_pred_global = predict_cluster_mass(
        cluster.M_bar, cluster.R_200, R0_GLOBAL, A_CLUSTER_GLOBAL
    )
    cluster.ratio_global = cluster.M_pred_global / cluster.M_lens if cluster.M_lens > 0 else 0


def fit_cluster_per_cluster(cluster: ClusterPhysics) -> None:
    """Fit optimal r0 and A for a single cluster."""
    
    def objective(params):
        r0, A = params
        if r0 <= 0.1 or A <= 0.1:
            return 1e10
        M_pred = predict_cluster_mass(cluster.M_bar, cluster.R_200, r0, A)
        ratio = M_pred / cluster.M_lens if cluster.M_lens > 0 else 0
        return abs(ratio - 1.0)  # Target ratio = 1.0
    
    # Grid search for initial guess
    best_r0, best_A = R0_GLOBAL, A_CLUSTER_GLOBAL
    best_obj = objective([best_r0, best_A])
    
    for r0 in [1, 2, 5, 10, 20, 50, 100]:
        for A in [5, 10, 15, 20, 30, 50]:
            obj = objective([r0, A])
            if obj < best_obj:
                best_obj = obj
                best_r0, best_A = r0, A
    
    # Refine with optimizer
    from scipy.optimize import minimize
    result = minimize(
        objective,
        [best_r0, best_A],
        bounds=[(0.5, 200), (1, 100)],
        method='L-BFGS-B'
    )
    
    cluster.r0_fitted = result.x[0]
    cluster.A_fitted = result.x[1]
    cluster.M_pred_fitted = predict_cluster_mass(
        cluster.M_bar, cluster.R_200, cluster.r0_fitted, cluster.A_fitted
    )
    cluster.ratio_fitted = cluster.M_pred_fitted / cluster.M_lens if cluster.M_lens > 0 else 0
    
    # Improvement
    error_global = abs(cluster.ratio_global - 1.0)
    error_fitted = abs(cluster.ratio_fitted - 1.0)
    cluster.improvement = (error_global - error_fitted) / error_global * 100 if error_global > 0 else 0


def compute_correlations(clusters: List[ClusterPhysics]) -> Dict[str, Dict[str, float]]:
    """Compute correlations between fitted parameters and physical properties."""
    
    # Collect arrays
    r0_fitted = np.array([c.r0_fitted for c in clusters])
    A_fitted = np.array([c.A_fitted for c in clusters])
    
    properties = {
        'M_bar': np.array([c.M_bar for c in clusters]),
        'M_lens': np.array([c.M_lens for c in clusters]),
        'M_ratio': np.array([c.M_ratio for c in clusters]),
        'R_200': np.array([c.R_200 for c in clusters]),
        'gas_fraction': np.array([c.gas_fraction for c in clusters]),
        'star_fraction': np.array([c.star_fraction for c in clusters]),
        'v_circ': np.array([c.v_circ for c in clusters]),
        'sigma_v': np.array([c.sigma_v for c in clusters]),
        't_dyn': np.array([c.t_dyn for c in clusters]),
        't_cross': np.array([c.t_cross for c in clusters]),
        'orbital_periods': np.array([c.orbital_periods for c in clusters]),
        'g_bar': np.array([c.g_bar for c in clusters]),
        'g_ratio': np.array([c.g_ratio for c in clusters]),
        'z': np.array([c.z for c in clusters]),
    }
    
    correlations = {'r0': {}, 'A': {}}
    
    for name, values in properties.items():
        valid = np.isfinite(r0_fitted) & np.isfinite(values) & (values > 0)
        if valid.sum() > 5:
            # Correlation with r0
            corr_r0 = np.corrcoef(r0_fitted[valid], values[valid])[0, 1]
            correlations['r0'][name] = corr_r0
            
            # Correlation with A
            corr_A = np.corrcoef(A_fitted[valid], values[valid])[0, 1]
            correlations['A'][name] = corr_A
        else:
            correlations['r0'][name] = np.nan
            correlations['A'][name] = np.nan
    
    return correlations


def quartile_analysis(clusters: List[ClusterPhysics], 
                      property_name: str, 
                      get_value: callable) -> Dict:
    """Analyze performance by quartiles of a property."""
    values = np.array([get_value(c) for c in clusters])
    valid = np.isfinite(values) & (values > 0)
    
    if valid.sum() < 8:
        return {'error': 'insufficient data'}
    
    # Compute quartiles
    valid_values = values[valid]
    q25, q50, q75 = np.percentile(valid_values, [25, 50, 75])
    
    quartiles = {
        'Q1 (lowest)': (values <= q25) & valid,
        'Q2': (values > q25) & (values <= q50) & valid,
        'Q3': (values > q50) & (values <= q75) & valid,
        'Q4 (highest)': (values > q75) & valid
    }
    
    results = {'property': property_name, 'quartiles': {}}
    
    for q_name, mask in quartiles.items():
        q_clusters = [c for c, m in zip(clusters, mask) if m]
        
        if len(q_clusters) < 2:
            continue
        
        ratios_global = [c.ratio_global for c in q_clusters]
        ratios_fitted = [c.ratio_fitted for c in q_clusters]
        r0_fitted = [c.r0_fitted for c in q_clusters]
        A_fitted = [c.A_fitted for c in q_clusters]
        
        results['quartiles'][q_name] = {
            'count': len(q_clusters),
            'property_range': f"{min([get_value(c) for c in q_clusters]):.2e} - {max([get_value(c) for c in q_clusters]):.2e}",
            'ratio_global_mean': np.mean(ratios_global),
            'ratio_global_std': np.std(ratios_global),
            'ratio_fitted_mean': np.mean(ratios_fitted),
            'r0_fitted_mean': np.mean(r0_fitted),
            'r0_fitted_std': np.std(r0_fitted),
            'A_fitted_mean': np.mean(A_fitted),
            'A_fitted_std': np.std(A_fitted),
            'clusters': [c.name for c in q_clusters]
        }
    
    return results


def compare_with_galaxies(clusters: List[ClusterPhysics]) -> Dict:
    """Compare cluster coherence properties with galaxy findings."""
    
    # From galaxy analysis:
    # - r0 correlates with coherence_time (r = +0.43)
    # - r0 correlates with sigma_v_ratio (r = +0.30)
    
    # For clusters, compute similar metrics
    r0_fitted = np.array([c.r0_fitted for c in clusters])
    t_dyn = np.array([c.t_dyn for c in clusters])
    t_cross = np.array([c.t_cross for c in clusters])
    orbital_periods = np.array([c.orbital_periods for c in clusters])
    sigma_v = np.array([c.sigma_v for c in clusters])
    v_circ = np.array([c.v_circ for c in clusters])
    
    # Compute sigma/v ratio (analogous to galaxy sigma_v_ratio)
    sigma_v_ratio = sigma_v / v_circ
    
    results = {
        'galaxy_comparison': {
            'galaxy_r0_vs_coherence_time': 0.43,  # From galaxy analysis
            'galaxy_r0_vs_sigma_v_ratio': 0.30,   # From galaxy analysis
        },
        'cluster_correlations': {}
    }
    
    # Compute cluster correlations
    valid = np.isfinite(r0_fitted) & np.isfinite(t_dyn)
    if valid.sum() > 5:
        results['cluster_correlations']['r0_vs_t_dyn'] = float(np.corrcoef(r0_fitted[valid], t_dyn[valid])[0, 1])
    
    valid = np.isfinite(r0_fitted) & np.isfinite(t_cross)
    if valid.sum() > 5:
        results['cluster_correlations']['r0_vs_t_cross'] = float(np.corrcoef(r0_fitted[valid], t_cross[valid])[0, 1])
    
    valid = np.isfinite(r0_fitted) & np.isfinite(orbital_periods)
    if valid.sum() > 5:
        results['cluster_correlations']['r0_vs_orbital_periods'] = float(np.corrcoef(r0_fitted[valid], orbital_periods[valid])[0, 1])
    
    valid = np.isfinite(r0_fitted) & np.isfinite(sigma_v_ratio)
    if valid.sum() > 5:
        results['cluster_correlations']['r0_vs_sigma_v_ratio'] = float(np.corrcoef(r0_fitted[valid], sigma_v_ratio[valid])[0, 1])
    
    # Interpretation
    results['interpretation'] = []
    
    cluster_t_corr = results['cluster_correlations'].get('r0_vs_t_dyn', 0)
    galaxy_t_corr = results['galaxy_comparison']['galaxy_r0_vs_coherence_time']
    
    if abs(cluster_t_corr) > 0.3 and np.sign(cluster_t_corr) == np.sign(galaxy_t_corr):
        results['interpretation'].append(
            f"CONSISTENT: Both galaxies (r={galaxy_t_corr:.2f}) and clusters (r={cluster_t_corr:.2f}) "
            "show r₀ correlating with dynamical timescale"
        )
    elif abs(cluster_t_corr) > 0.3:
        results['interpretation'].append(
            f"DIFFERENT: Clusters show opposite sign for r₀ vs timescale (r={cluster_t_corr:.2f})"
        )
    else:
        results['interpretation'].append(
            f"WEAK: Cluster r₀ vs timescale correlation is weak (r={cluster_t_corr:.2f})"
        )
    
    return results


def print_report(clusters: List[ClusterPhysics],
                 correlations: Dict,
                 quartile_results: List[Dict],
                 comparison: Dict) -> None:
    """Print comprehensive analysis report."""
    
    print("=" * 100)
    print("CLUSTER COHERENCE ANALYSIS")
    print("=" * 100)
    
    # =========================================================================
    # 1. Overview
    # =========================================================================
    print(f"\n{'='*100}")
    print("1. CLUSTER SAMPLE OVERVIEW")
    print("=" * 100)
    
    print(f"\nTotal clusters: {len(clusters)}")
    
    M_bar = [c.M_bar for c in clusters]
    M_lens = [c.M_lens for c in clusters]
    M_ratio = [c.M_ratio for c in clusters]
    
    print(f"\nMass ranges:")
    print(f"  M_bar: {min(M_bar):.2e} - {max(M_bar):.2e} M☉")
    print(f"  M_lens: {min(M_lens):.2e} - {max(M_lens):.2e} M☉")
    print(f"  M_lens/M_bar ratio: {min(M_ratio):.1f} - {max(M_ratio):.1f}")
    
    z_vals = [c.z for c in clusters]
    print(f"\nRedshift range: {min(z_vals):.2f} - {max(z_vals):.2f}")
    
    # =========================================================================
    # 2. Global fit performance
    # =========================================================================
    print(f"\n{'='*100}")
    print("2. GLOBAL PARAMETER PERFORMANCE")
    print("=" * 100)
    
    print(f"\nGlobal parameters: r₀ = {R0_GLOBAL} kpc, A = {A_CLUSTER_GLOBAL:.2f}")
    
    ratios_global = [c.ratio_global for c in clusters]
    print(f"\nM_pred/M_lens ratios:")
    print(f"  Mean: {np.mean(ratios_global):.3f}")
    print(f"  Median: {np.median(ratios_global):.3f}")
    print(f"  Std: {np.std(ratios_global):.3f}")
    print(f"  Range: {min(ratios_global):.3f} - {max(ratios_global):.3f}")
    
    # =========================================================================
    # 3. Per-cluster fitting
    # =========================================================================
    print(f"\n{'='*100}")
    print("3. PER-CLUSTER PARAMETER FITTING")
    print("=" * 100)
    
    r0_fitted = [c.r0_fitted for c in clusters]
    A_fitted = [c.A_fitted for c in clusters]
    ratios_fitted = [c.ratio_fitted for c in clusters]
    
    print(f"\nFitted r₀ distribution:")
    print(f"  Mean: {np.mean(r0_fitted):.2f} kpc")
    print(f"  Median: {np.median(r0_fitted):.2f} kpc")
    print(f"  Std: {np.std(r0_fitted):.2f} kpc")
    print(f"  Range: {min(r0_fitted):.2f} - {max(r0_fitted):.2f} kpc")
    
    print(f"\nFitted A distribution:")
    print(f"  Mean: {np.mean(A_fitted):.2f}")
    print(f"  Median: {np.median(A_fitted):.2f}")
    print(f"  Std: {np.std(A_fitted):.2f}")
    print(f"  Range: {min(A_fitted):.2f} - {max(A_fitted):.2f}")
    
    print(f"\nFitted M_pred/M_lens ratios:")
    print(f"  Mean: {np.mean(ratios_fitted):.3f}")
    print(f"  Std: {np.std(ratios_fitted):.3f}")
    
    # =========================================================================
    # 4. Correlations
    # =========================================================================
    print(f"\n{'='*100}")
    print("4. CORRELATIONS: FITTED PARAMETERS vs PHYSICAL PROPERTIES")
    print("=" * 100)
    
    print("\n--- Correlations with fitted r₀ ---")
    print(f"{'Property':<20} {'Correlation':>12} {'Interpretation':<50}")
    print("-" * 85)
    
    interpretations = {
        'M_bar': 'More massive clusters need different r₀',
        'M_lens': 'Total mass affects coherence scale',
        'M_ratio': 'Enhancement ratio affects r₀',
        'R_200': 'Larger clusters need larger r₀',
        'gas_fraction': 'Gas-rich clusters differ',
        'star_fraction': 'Star-rich clusters differ',
        'v_circ': 'Faster clusters need different r₀',
        'sigma_v': 'Velocity dispersion affects r₀',
        't_dyn': 'Dynamical time affects coherence',
        't_cross': 'Crossing time affects coherence',
        'orbital_periods': 'More orbits → different r₀',
        'g_bar': 'Baryonic acceleration affects r₀',
        'g_ratio': 'g/g† ratio affects r₀',
        'z': 'Redshift affects coherence',
    }
    
    sorted_r0 = sorted(correlations['r0'].items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
    
    for prop, corr in sorted_r0:
        if np.isnan(corr):
            continue
        interp = interpretations.get(prop, '')
        print(f"{prop:<20} {corr:>12.3f} {interp:<50}")
    
    print("\n--- Correlations with fitted A ---")
    print(f"{'Property':<20} {'Correlation':>12}")
    print("-" * 35)
    
    sorted_A = sorted(correlations['A'].items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
    
    for prop, corr in sorted_A:
        if np.isnan(corr):
            continue
        print(f"{prop:<20} {corr:>12.3f}")
    
    # Highlight strongest
    print("\n" + "-" * 85)
    print("STRONGEST CORRELATIONS (|r| > 0.3):")
    
    for prop, corr in sorted_r0:
        if not np.isnan(corr) and abs(corr) > 0.3:
            direction = "increases" if corr > 0 else "decreases"
            print(f"  • r₀ {direction} with {prop} (r = {corr:.3f})")
    
    for prop, corr in sorted_A:
        if not np.isnan(corr) and abs(corr) > 0.3:
            direction = "increases" if corr > 0 else "decreases"
            print(f"  • A {direction} with {prop} (r = {corr:.3f})")
    
    # =========================================================================
    # 5. Quartile analysis
    # =========================================================================
    print(f"\n{'='*100}")
    print("5. QUARTILE ANALYSIS BY PHYSICAL PROPERTY")
    print("=" * 100)
    
    for qr in quartile_results:
        if 'error' in qr:
            continue
        
        print(f"\n--- {qr['property']} ---")
        print(f"{'Quartile':<15} {'N':>4} {'Range':<25} {'Ratio(global)':>14} {'r₀ fitted':>12} {'A fitted':>10}")
        print("-" * 90)
        
        for q_name, stats in qr['quartiles'].items():
            print(f"{q_name:<15} {stats['count']:>4} {stats['property_range']:<25} "
                  f"{stats['ratio_global_mean']:>14.3f} "
                  f"{stats['r0_fitted_mean']:>12.2f} {stats['A_fitted_mean']:>10.2f}")
    
    # =========================================================================
    # 6. Comparison with galaxies
    # =========================================================================
    print(f"\n{'='*100}")
    print("6. COMPARISON WITH GALAXY COHERENCE ANALYSIS")
    print("=" * 100)
    
    print("\nGalaxy findings (from coherence_root_cause_analysis.py):")
    print(f"  • r₀ vs coherence_time: r = +0.43")
    print(f"  • r₀ vs sigma_v_ratio: r = +0.30")
    print(f"  • Interpretation: Coherence scale tracks orbital timescale")
    
    print("\nCluster findings:")
    for name, corr in comparison['cluster_correlations'].items():
        print(f"  • {name}: r = {corr:+.3f}")
    
    print("\nInterpretation:")
    for interp in comparison['interpretation']:
        print(f"  • {interp}")
    
    # =========================================================================
    # 7. Key insights
    # =========================================================================
    print(f"\n{'='*100}")
    print("7. KEY INSIGHTS: CLUSTER COHERENCE")
    print("=" * 100)
    
    # Check if timescale correlation holds
    t_dyn_corr = correlations['r0'].get('t_dyn', 0)
    t_cross_corr = correlations['r0'].get('t_cross', 0)
    orbital_corr = correlations['r0'].get('orbital_periods', 0)
    
    print("\nDoes orbital timescale drive coherence in clusters?")
    if abs(t_dyn_corr) > 0.3 or abs(t_cross_corr) > 0.3:
        print(f"  YES: r₀ correlates with dynamical time (r = {t_dyn_corr:.3f}) / crossing time (r = {t_cross_corr:.3f})")
    else:
        print(f"  WEAK: r₀ vs t_dyn (r = {t_dyn_corr:.3f}), r₀ vs t_cross (r = {t_cross_corr:.3f})")
    
    # Check mass dependence
    M_bar_corr = correlations['r0'].get('M_bar', 0)
    M_lens_corr = correlations['r0'].get('M_lens', 0)
    
    print("\nDoes mass drive coherence?")
    if abs(M_bar_corr) > 0.3 or abs(M_lens_corr) > 0.3:
        print(f"  YES: r₀ correlates with M_bar (r = {M_bar_corr:.3f}) / M_lens (r = {M_lens_corr:.3f})")
    else:
        print(f"  WEAK: r₀ vs M_bar (r = {M_bar_corr:.3f}), r₀ vs M_lens (r = {M_lens_corr:.3f})")
    
    # Check gas fraction
    gas_corr = correlations['r0'].get('gas_fraction', 0)
    print("\nDoes gas fraction matter?")
    if abs(gas_corr) > 0.3:
        direction = "more" if gas_corr > 0 else "less"
        print(f"  YES: Gas-rich clusters need {direction} coherence (r = {gas_corr:.3f})")
    else:
        print(f"  WEAK: r₀ vs gas_fraction (r = {gas_corr:.3f})")
    
    # Summary
    print("\n" + "-" * 100)
    print("SUMMARY: GALAXY vs CLUSTER COHERENCE")
    print("-" * 100)
    
    galaxy_t_corr = 0.43  # From galaxy analysis
    cluster_t_corr = t_dyn_corr
    
    if abs(cluster_t_corr) > 0.3 and np.sign(cluster_t_corr) == np.sign(galaxy_t_corr):
        print("\n✓ UNIVERSAL PRINCIPLE: Coherence scale tracks dynamical timescale")
        print("  in BOTH galaxies and clusters")
        print(f"  • Galaxies: r₀ ∝ T_orbit (r = {galaxy_t_corr:.2f})")
        print(f"  • Clusters: r₀ ∝ T_dyn (r = {cluster_t_corr:.2f})")
        print("\n  Physical interpretation: Coherence builds over dynamical timescales")
        print("  regardless of system type (rotating disk or virialized sphere)")
    elif abs(cluster_t_corr) < 0.3:
        print("\n? INCONCLUSIVE: Cluster timescale correlation is weak")
        print(f"  • Galaxies: r₀ ∝ T_orbit (r = {galaxy_t_corr:.2f})")
        print(f"  • Clusters: r₀ vs T_dyn (r = {cluster_t_corr:.2f}) - not significant")
        print("\n  Possible explanations:")
        print("  1. Cluster sample too small/homogeneous")
        print("  2. Different physics in virialized vs rotating systems")
        print("  3. Coherence mechanism differs between galaxies and clusters")
    else:
        print("\n✗ DIFFERENT PHYSICS: Opposite correlation in clusters")
        print(f"  • Galaxies: r₀ ∝ T_orbit (r = {galaxy_t_corr:.2f})")
        print(f"  • Clusters: r₀ vs T_dyn (r = {cluster_t_corr:.2f}) - opposite sign!")
        print("\n  This suggests different coherence mechanisms")


def main():
    print("=" * 100)
    print("CLUSTER COHERENCE ANALYSIS")
    print("Investigating whether orbital timescale relationship applies to clusters")
    print("=" * 100)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    
    print("\nLoading cluster data...")
    clusters = load_cluster_data(data_dir)
    print(f"Loaded {len(clusters)} clusters")
    
    # Compute physics
    print("\nComputing cluster physical properties...")
    for cluster in clusters:
        compute_cluster_physics(cluster)
    
    # Global fit
    print("\nComputing predictions with global parameters...")
    for cluster in clusters:
        fit_cluster_global(cluster)
    
    # Per-cluster fit
    print("\nFitting per-cluster parameters...")
    for cluster in clusters:
        fit_cluster_per_cluster(cluster)
    
    # Correlations
    print("\nComputing correlations...")
    correlations = compute_correlations(clusters)
    
    # Quartile analysis
    print("\nPerforming quartile analysis...")
    properties_to_analyze = [
        ('M_bar (M☉)', lambda c: c.M_bar),
        ('M_lens (M☉)', lambda c: c.M_lens),
        ('M_ratio', lambda c: c.M_ratio),
        ('R_200 (kpc)', lambda c: c.R_200),
        ('gas_fraction', lambda c: c.gas_fraction),
        ('t_dyn (Gyr)', lambda c: c.t_dyn),
        ('orbital_periods', lambda c: c.orbital_periods),
        ('g_ratio (g/g†)', lambda c: c.g_ratio),
        ('z', lambda c: c.z),
    ]
    
    quartile_results = []
    for prop_name, get_value in properties_to_analyze:
        result = quartile_analysis(clusters, prop_name, get_value)
        quartile_results.append(result)
    
    # Compare with galaxies
    print("\nComparing with galaxy findings...")
    comparison = compare_with_galaxies(clusters)
    
    # Print report
    print_report(clusters, correlations, quartile_results, comparison)
    
    # Save results
    output_dir = Path(__file__).parent / "cluster_coherence_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save correlations
    with open(output_dir / "correlations.json", 'w') as f:
        json.dump({
            'r0': {k: float(v) if not np.isnan(v) else None for k, v in correlations['r0'].items()},
            'A': {k: float(v) if not np.isnan(v) else None for k, v in correlations['A'].items()}
        }, f, indent=2)
    
    # Save cluster data
    cluster_data = []
    for c in clusters:
        cluster_data.append({
            'name': c.name,
            'M_bar': float(c.M_bar),
            'M_lens': float(c.M_lens),
            'M_ratio': float(c.M_ratio),
            'R_200': float(c.R_200),
            'gas_fraction': float(c.gas_fraction),
            'z': float(c.z),
            'v_circ': float(c.v_circ),
            'sigma_v': float(c.sigma_v),
            't_dyn': float(c.t_dyn),
            't_cross': float(c.t_cross),
            'orbital_periods': float(c.orbital_periods),
            'g_bar': float(c.g_bar),
            'g_ratio': float(c.g_ratio),
            'ratio_global': float(c.ratio_global),
            'r0_fitted': float(c.r0_fitted),
            'A_fitted': float(c.A_fitted),
            'ratio_fitted': float(c.ratio_fitted),
        })
    
    with open(output_dir / "cluster_physics.json", 'w') as f:
        json.dump(cluster_data, f, indent=2)
    
    # Save comparison
    with open(output_dir / "galaxy_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

