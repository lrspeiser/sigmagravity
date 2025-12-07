#!/usr/bin/env python3
"""
Test Path Length Model Predictions

This script tests the prediction that A ∝ L^(1/4) for different system types:

1. Disk galaxies (SPARC): A ≈ √3, L ≈ 0.6 kpc
2. Elliptical galaxies (MaNGA): A ≈ 2.5-3.5?, L ≈ 2-5 kpc
3. Galaxy clusters (Fox+): A ≈ 8.0, L ≈ 400 kpc

Also explores whether ξ should scale with system type.

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.table import Table
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8
H0_SI = 2.27e-18
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))

# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def h_function(g: np.ndarray) -> np.ndarray:
    """Enhancement function."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r: np.ndarray, xi: float) -> np.ndarray:
    """Coherence window."""
    xi = max(xi, 0.01)
    return 1 - np.power(xi / (xi + r), 0.5)

def path_length_amplitude(L_kpc: float, A0: float = 1.9, exponent: float = 0.25) -> float:
    """Compute amplitude from path length model."""
    return A0 * (L_kpc ** exponent)

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_manga_ellipticals(data_dir: Path) -> List[Dict]:
    """Load elliptical galaxies from MaNGA DynPop."""
    manga_file = data_dir / "manga_dynpop" / "SDSSDR17_MaNGA_JAM.fits"
    if not manga_file.exists():
        return []
    
    with fits.open(manga_file) as hdul:
        # HDU 1: Basic properties
        basic = Table(hdul[1].data)
        # HDU 4: JAM results with NFW dark matter
        jam_nfw = Table(hdul[4].data)
    
    ellipticals = []
    
    for i in range(len(basic)):
        try:
            # Select early-type galaxies using Lambda_Re and Sersic n
            # Lambda_Re < 0.2 = slow rotator (elliptical-like)
            # Sersic n > 2.5 = early-type morphology
            lambda_re = float(basic['Lambda_Re'][i])
            sersic_n = float(basic['nsa_sersic_n'][i])
            
            # Skip if not early-type
            if lambda_re > 0.2 or sersic_n < 2.5 or sersic_n < 0:
                continue
            
            # Get stellar mass and effective radius
            # nsa_sersic_mass is already log10(M/M_sun)
            log_mstar = float(basic['nsa_sersic_mass'][i])
                
            Re_arcsec = float(basic['Re_arcsec_MGE'][i])
            z = float(basic['z'][i])
            
            if not (9.0 < log_mstar < 12.0 and 0.01 < z < 0.15):
                continue
            
            if Re_arcsec <= 0:
                continue
            
            # Convert Re to kpc using angular diameter distance
            D_A = float(basic['DA'][i])  # Already in Mpc
            Re_kpc = Re_arcsec * D_A * 1000 / 206265  # kpc
            
            if not (0.5 < Re_kpc < 30):
                continue
            
            # Get dark matter fraction
            fdm = float(jam_nfw['fdm_Re'][i])
            
            if not (np.isfinite(fdm) and 0 <= fdm <= 1):
                continue
            
            # Get velocity dispersion
            sigma = float(basic['Sigma_Re'][i]) if 'Sigma_Re' in basic.colnames else None
            
            ellipticals.append({
                'mangaid': str(basic['mangaid'][i]),
                'log_mstar': log_mstar,
                'Re_kpc': Re_kpc,
                'sigma_e': sigma,
                'fdm_Re': fdm,
                'z': z,
                'lambda_re': lambda_re,
                'sersic_n': sersic_n,
            })
            
        except (ValueError, IndexError, KeyError) as e:
            continue
    
    return ellipticals

def load_sparc(data_dir: Path) -> List[Dict]:
    """Load SPARC disk galaxies."""
    sparc_dir = data_dir / "Rotmod_LTG"
    if not sparc_dir.exists():
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
        df['V_disk_scaled'] = df['V_disk'] * np.sqrt(0.5)
        df['V_bulge_scaled'] = df['V_bulge'] * np.sqrt(0.7)
        V_bar_sq = (np.sign(df['V_gas']) * df['V_gas']**2 + 
                    df['V_disk_scaled']**2 + df['V_bulge_scaled']**2)
        df['V_bar'] = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)
        
        valid = (df['V_bar'] > 0) & (df['R'] > 0) & (df['V_obs'] > 0)
        df = df[valid]
        
        if len(df) >= 5:
            idx = len(df) // 3
            R_d = df['R'].iloc[idx] if idx > 0 else df['R'].iloc[-1] / 2
            
            # Estimate disk thickness: h ≈ 0.1-0.2 × R_d
            h_disk = 0.15 * R_d  # kpc
            
            galaxies.append({
                'name': gf.stem.replace('_rotmod', ''),
                'R': df['R'].values,
                'V_obs': df['V_obs'].values,
                'V_bar': df['V_bar'].values,
                'R_d': R_d,
                'h_disk': h_disk,
                'path_length': 2 * h_disk,  # L = 2h
            })
    
    return galaxies

def load_clusters(data_dir: Path) -> List[Dict]:
    """Load Fox+ 2022 clusters."""
    cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
    if not cluster_file.exists():
        return []
    
    df = pd.read_csv(cluster_file)
    df_valid = df[
        df['M500_1e14Msun'].notna() & 
        df['MSL_200kpc_1e12Msun'].notna() &
        (df['spec_z_constraint'] == 'yes')
    ].copy()
    df_valid = df_valid[df_valid['M500_1e14Msun'] > 2.0].copy()
    
    clusters = []
    f_baryon = 0.15
    
    for idx, row in df_valid.iterrows():
        M500 = row['M500_1e14Msun'] * 1e14
        M_bar_200 = 0.4 * f_baryon * M500
        M_lens_200 = row['MSL_200kpc_1e12Msun'] * 1e12
        
        clusters.append({
            'name': row['cluster'],
            'M_bar': M_bar_200,
            'M_lens': M_lens_200,
            'r_kpc': 200,
            'path_length': 400,  # L = 2 × R_lens
        })
    
    return clusters

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_sparc_with_path_model(galaxies: List[Dict], A0: float, exponent: float, 
                                xi_model: str = "fixed") -> Dict:
    """Test SPARC galaxies with path length amplitude model."""
    rms_list = []
    
    for gal in galaxies:
        R = gal['R']
        V_bar = gal['V_bar']
        V_obs = gal['V_obs']
        R_d = gal['R_d']
        L = gal['path_length']
        
        # Compute amplitude from path length
        A = path_length_amplitude(L, A0, exponent)
        
        # Compute ξ based on model
        if xi_model == "fixed":
            xi = (2/3) * R_d
        elif xi_model == "path_scaled":
            # ξ scales with path length
            xi = L / 2  # ξ = L/2
        elif xi_model == "Re_scaled":
            # ξ scales with effective radius
            xi = R_d  # ξ = R_d
        else:
            xi = (2/3) * R_d
        
        # Predict velocity
        R_m = R * kpc_to_m
        V_bar_ms = V_bar * 1000
        g_bar = V_bar_ms**2 / R_m
        
        h = h_function(g_bar)
        W = W_coherence(R, xi)
        
        Sigma = 1 + A * W * h
        V_pred = V_bar * np.sqrt(Sigma)
        
        rms = np.sqrt(((V_obs - V_pred)**2).mean())
        rms_list.append(rms)
    
    return {
        'mean_rms': np.mean(rms_list),
        'median_rms': np.median(rms_list),
        'n_galaxies': len(galaxies)
    }

def test_clusters_with_path_model(clusters: List[Dict], A0: float, exponent: float) -> Dict:
    """Test clusters with path length amplitude model."""
    ratios = []
    
    for cl in clusters:
        L = cl['path_length']
        A = path_length_amplitude(L, A0, exponent)
        
        r_m = cl['r_kpc'] * kpc_to_m
        g_bar = G_const * cl['M_bar'] * M_sun / r_m**2
        
        h = h_function(np.array([g_bar]))[0]
        Sigma = 1 + A * h  # W ≈ 1 for clusters
        
        M_pred = cl['M_bar'] * Sigma
        ratio = M_pred / cl['M_lens']
        if np.isfinite(ratio) and ratio > 0:
            ratios.append(ratio)
    
    return {
        'median_ratio': np.median(ratios),
        'scatter_dex': np.std(np.log10(ratios)),
        'n_clusters': len(ratios)
    }

def predict_elliptical_fdm(ellipticals: List[Dict], A0: float, exponent: float,
                           xi_model: str = "Re_scaled") -> List[Dict]:
    """Predict f_DM for elliptical galaxies."""
    predictions = []
    
    for ell in ellipticals:
        Re = ell['Re_kpc']
        log_mstar = ell['log_mstar']
        
        # Path length for elliptical: L ≈ 2 × Re (diameter)
        L = 2 * Re
        
        # Predict amplitude
        A = path_length_amplitude(L, A0, exponent)
        
        # Coherence scale for elliptical
        if xi_model == "Re_scaled":
            xi = Re
        elif xi_model == "path_scaled":
            xi = L / 2
        else:
            xi = Re
        
        # Estimate g_bar at Re
        M_star = 10**log_mstar * M_sun
        r_m = Re * kpc_to_m
        g_bar = G_const * M_star / r_m**2
        
        # Compute enhancement at Re
        h = h_function(np.array([g_bar]))[0]
        W = W_coherence(np.array([Re]), xi)[0]
        
        Sigma = 1 + A * W * h
        
        # f_DM = (M_total - M_bar) / M_total = 1 - 1/Σ
        fdm_pred = 1 - 1/Sigma
        
        predictions.append({
            'mangaid': ell['mangaid'],
            'Re_kpc': Re,
            'log_mstar': log_mstar,
            'path_length': L,
            'A_predicted': A,
            'fdm_observed': ell['fdm_Re'],
            'fdm_predicted': fdm_pred,
            'Sigma': Sigma,
        })
    
    return predictions

# =============================================================================
# JOINT A AND ξ EXPLORATION
# =============================================================================

def explore_joint_A_xi(galaxies: List[Dict], clusters: List[Dict]) -> Dict:
    """Explore joint variation of A and ξ scaling."""
    
    results = []
    
    # Different ξ models
    xi_models = [
        ("fixed_2/3", lambda R_d, L: (2/3) * R_d),
        ("fixed_1/2", lambda R_d, L: 0.5 * R_d),
        ("fixed_1", lambda R_d, L: R_d),
        ("path_scaled", lambda R_d, L: L / 2),
        ("path_quarter", lambda R_d, L: L / 4),
        ("geometric_mean", lambda R_d, L: np.sqrt(R_d * L)),
    ]
    
    # Different A models
    A_models = [
        ("fixed_sqrt3", lambda L: np.sqrt(3)),
        ("path_0.20", lambda L: 1.9 * L**0.20),
        ("path_0.235", lambda L: 1.9 * L**0.235),
        ("path_0.25", lambda L: 1.9 * L**0.25),
        ("path_0.30", lambda L: 1.9 * L**0.30),
    ]
    
    for xi_name, xi_func in xi_models:
        for A_name, A_func in A_models:
            # Test on SPARC
            rms_list = []
            for gal in galaxies:
                R = gal['R']
                V_bar = gal['V_bar']
                V_obs = gal['V_obs']
                R_d = gal['R_d']
                L = gal['path_length']
                
                A = A_func(L)
                xi = xi_func(R_d, L)
                
                R_m = R * kpc_to_m
                V_bar_ms = V_bar * 1000
                g_bar = V_bar_ms**2 / R_m
                
                h = h_function(g_bar)
                W = W_coherence(R, xi)
                
                Sigma = 1 + A * W * h
                V_pred = V_bar * np.sqrt(Sigma)
                
                rms = np.sqrt(((V_obs - V_pred)**2).mean())
                rms_list.append(rms)
            
            galaxy_rms = np.mean(rms_list)
            
            # Test on clusters (A from path model, W=1)
            ratios = []
            for cl in clusters:
                L = cl['path_length']
                A = A_func(L)
                
                r_m = cl['r_kpc'] * kpc_to_m
                g_bar = G_const * cl['M_bar'] * M_sun / r_m**2
                
                h = h_function(np.array([g_bar]))[0]
                Sigma = 1 + A * h
                
                M_pred = cl['M_bar'] * Sigma
                ratio = M_pred / cl['M_lens']
                if np.isfinite(ratio) and ratio > 0:
                    ratios.append(ratio)
            
            cluster_ratio = np.median(ratios) if ratios else np.nan
            
            results.append({
                'xi_model': xi_name,
                'A_model': A_name,
                'galaxy_rms': galaxy_rms,
                'cluster_ratio': cluster_ratio,
                'combined_score': galaxy_rms + 10 * abs(np.log10(cluster_ratio)) if cluster_ratio > 0 else 999
            })
    
    return results

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("TESTING PATH LENGTH MODEL PREDICTIONS")
    print("=" * 80)
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load data
    print("\nLoading data...")
    galaxies = load_sparc(data_dir)
    clusters = load_clusters(data_dir)
    ellipticals = load_manga_ellipticals(data_dir)
    
    print(f"  SPARC disk galaxies: {len(galaxies)}")
    print(f"  Fox+ clusters: {len(clusters)}")
    print(f"  MaNGA ellipticals: {len(ellipticals)}")
    
    # Path length statistics
    if galaxies:
        L_disk = np.mean([g['path_length'] for g in galaxies])
        print(f"\n  Mean disk path length: {L_disk:.2f} kpc")
    
    if clusters:
        L_cluster = np.mean([c['path_length'] for c in clusters])
        print(f"  Mean cluster path length: {L_cluster:.0f} kpc")
    
    if ellipticals:
        L_ellip = np.mean([2 * e['Re_kpc'] for e in ellipticals])
        print(f"  Mean elliptical path length: {L_ellip:.2f} kpc")
    
    # Test path length model
    print("\n" + "=" * 80)
    print("PATH LENGTH MODEL: A = A₀ × L^(1/4)")
    print("=" * 80)
    
    A0 = 1.9
    exponent = 0.25
    
    # SPARC
    if galaxies:
        result = test_sparc_with_path_model(galaxies, A0, exponent)
        print(f"\nSPARC (path model): RMS = {result['mean_rms']:.2f} km/s")
        
        # Compare to fixed A = √3
        result_fixed = test_sparc_with_path_model(galaxies, np.sqrt(3), 0)  # A = √3 constant
        print(f"SPARC (fixed A=√3): RMS = {result_fixed['mean_rms']:.2f} km/s")
    
    # Clusters
    if clusters:
        result = test_clusters_with_path_model(clusters, A0, exponent)
        print(f"\nClusters (path model): Median ratio = {result['median_ratio']:.3f}")
        
        # What A does the model predict for clusters?
        A_cluster_pred = path_length_amplitude(400, A0, exponent)
        print(f"  Predicted A_cluster = {A_cluster_pred:.2f}")
    
    # Ellipticals prediction
    if ellipticals:
        print("\n" + "=" * 80)
        print("ELLIPTICAL GALAXY PREDICTIONS")
        print("=" * 80)
        
        predictions = predict_elliptical_fdm(ellipticals, A0, exponent)
        
        # Summary statistics
        A_values = [p['A_predicted'] for p in predictions]
        fdm_obs = [p['fdm_observed'] for p in predictions]
        fdm_pred = [p['fdm_predicted'] for p in predictions]
        
        print(f"\nPredicted amplitude range: {np.min(A_values):.2f} - {np.max(A_values):.2f}")
        print(f"Mean predicted A: {np.mean(A_values):.2f}")
        
        print(f"\nObserved f_DM: mean = {np.mean(fdm_obs):.3f}, median = {np.median(fdm_obs):.3f}")
        print(f"Predicted f_DM: mean = {np.mean(fdm_pred):.3f}, median = {np.median(fdm_pred):.3f}")
        
        # Correlation
        valid = [(o, p) for o, p in zip(fdm_obs, fdm_pred) if np.isfinite(o) and np.isfinite(p)]
        if len(valid) > 10:
            obs, pred = zip(*valid)
            corr = np.corrcoef(obs, pred)[0, 1]
            print(f"Correlation (obs vs pred): r = {corr:.3f}")
            
            # RMS error
            rms = np.sqrt(np.mean([(o - p)**2 for o, p in valid]))
            print(f"RMS error in f_DM: {rms:.3f}")
    
    # Joint A and ξ exploration
    print("\n" + "=" * 80)
    print("JOINT A AND ξ EXPLORATION")
    print("=" * 80)
    
    if galaxies and clusters:
        results = explore_joint_A_xi(galaxies, clusters)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'])
        
        print(f"\n{'ξ Model':<20} {'A Model':<15} {'Galaxy RMS':>12} {'Cluster Ratio':>14}")
        print("-" * 65)
        
        for r in results[:15]:
            print(f"{r['xi_model']:<20} {r['A_model']:<15} {r['galaxy_rms']:>12.2f} {r['cluster_ratio']:>14.3f}")
        
        # Best result
        best = results[0]
        print(f"\nBest combination:")
        print(f"  ξ model: {best['xi_model']}")
        print(f"  A model: {best['A_model']}")
        print(f"  Galaxy RMS: {best['galaxy_rms']:.2f} km/s")
        print(f"  Cluster ratio: {best['cluster_ratio']:.3f}")
    
    # Save results
    output_dir = Path(__file__).parent / "path_length_results"
    output_dir.mkdir(exist_ok=True)
    
    summary = {
        'n_galaxies': len(galaxies),
        'n_clusters': len(clusters),
        'n_ellipticals': len(ellipticals),
        'A0': A0,
        'exponent': exponent,
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()

