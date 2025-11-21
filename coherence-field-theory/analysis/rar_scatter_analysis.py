"""
RAR Scatter vs Q Analysis

Tests GPM's unique prediction: At fixed baryon acceleration, residuals
anti-correlate with Toomre Q and velocity dispersion sigma_v.

This is a FALSIFIABLE PREDICTION that distinguishes GPM from MOND and DM:
- MOND: Residuals uncorrelated with Q (universal a_0 scale)
- DM: Residuals weakly correlated with halo properties, not Q
- GPM: Strong anti-correlation (r ~ -0.5 to -0.7) due to gating

If this correlation is NOT observed, GPM gating mechanism is wrong.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses
from galaxies.environment_estimator import EnvironmentEstimator


def compute_rar_residuals(gal, v_model, galaxy_name):
    """
    Compute Radial Acceleration Relation residuals.
    
    RAR: a_obs = μ(a_bar/a_0) a_bar
    Residuals: Δa = a_obs - a_bar
    
    For GPM, residuals should anti-correlate with Q and sigma_v.
    """
    r = gal['r']
    v_obs = gal['v_obs']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
    
    # Accelerations: a = v² / r
    a_obs = v_obs**2 / r  # km²/s² / kpc
    a_bar = (v_disk**2 + v_gas**2 + v_bulge**2) / r
    a_model = v_model**2 / r
    
    # RAR residuals: difference between observed and model
    delta_a_model = a_obs - a_model
    delta_a_bar = a_obs - a_bar
    
    # Convert to log space (common for RAR plots)
    # a in units of m/s² (convert from km²/s²/kpc)
    kpc_to_m = 3.086e19  # meters
    km_to_m = 1000
    
    a_obs_SI = a_obs * (km_to_m**2) / kpc_to_m  # m/s²
    a_bar_SI = a_bar * (km_to_m**2) / kpc_to_m
    a_model_SI = a_model * (km_to_m**2) / kpc_to_m
    
    return {
        'r': r,
        'a_obs': a_obs_SI,
        'a_bar': a_bar_SI,
        'a_model': a_model_SI,
        'delta_a_model': delta_a_model,
        'delta_a_bar': delta_a_bar,
        'log_a_obs': np.log10(a_obs_SI + 1e-12),
        'log_a_bar': np.log10(a_bar_SI + 1e-12),
        'log_a_model': np.log10(a_model_SI + 1e-12),
        'residual_log': np.log10(a_obs_SI + 1e-12) - np.log10(a_bar_SI + 1e-12)
    }


def analyze_rar_scatter(results_csv='outputs/gpm_tests/batch_gpm_results.csv'):
    """
    Analyze RAR residuals vs Q and sigma_v for multiple galaxies.
    
    Tests GPM prediction: residuals anti-correlate with Q and sigma_v.
    """
    
    print("="*80)
    print("RAR SCATTER vs Q/SIGMA_V ANALYSIS")
    print("="*80)
    print()
    print("Testing GPM's unique prediction:")
    print("  At fixed baryon acceleration, residuals anti-correlate with Q and sigma_v")
    print()
    print("Expected correlations:")
    print("  - GPM:  r(Δa, Q) ~ -0.5 to -0.7  [FALSIFIABLE]")
    print("  - MOND: r(Δa, Q) ~  0.0          [no Q dependence]")
    print("  - DM:   r(Δa, Q) ~  0.0          [halo independent of disk Q]")
    print()
    
    # Load batch test results
    if not os.path.exists(results_csv):
        print(f"ERROR: Results file not found: {results_csv}")
        print("Run batch_gpm_test.py first to generate results.")
        return
    
    df = pd.read_csv(results_csv)
    print(f"Loaded results for {len(df)} galaxies")
    print()
    
    # Prepare data collection
    data_points = []
    
    loader = RealDataLoader()
    estimator = EnvironmentEstimator()
    
    for idx, row in df.iterrows():
        galaxy_name = row['name']
        print(f"Processing {galaxy_name}...", end=' ')
        
        try:
            # Load galaxy data
            gal = loader.load_rotmod_galaxy(galaxy_name)
            sparc_masses = load_sparc_masses(galaxy_name)
            
            M_total = sparc_masses['M_total']
            R_disk = sparc_masses['R_disk']
            
            # Get Q and sigma_v
            r = gal['r']
            v_obs = gal['v_obs']
            v_disk = gal['v_disk']
            v_gas = gal['v_gas']
            v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
            
            # Load SBdisk
            rotmod_dir = os.path.join(loader.base_data_dir, 'Rotmod_LTG')
            filepath = os.path.join(rotmod_dir, f'{galaxy_name}_rotmod.dat')
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            data_lines = [l for l in lines if not l.startswith('#')]
            SBdisk = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 7:
                    SBdisk.append(float(parts[6]))
            SBdisk = np.array(SBdisk)
            
            morphology = estimator.classify_morphology(gal, M_total, R_disk)
            Q, sigma_v = estimator.estimate_from_sparc(gal, SBdisk, R_disk, M_L=0.5, morphology=morphology)
            
            # Compute RAR residuals (use baryon baseline)
            v_bar = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
            
            # Acceleration residuals
            a_obs = v_obs**2 / r
            a_bar = v_bar**2 / r
            
            # RAR residual at each radius
            for i in range(len(r)):
                data_points.append({
                    'galaxy': galaxy_name,
                    'r': r[i],
                    'a_obs': a_obs[i],
                    'a_bar': a_bar[i],
                    'residual': (a_obs[i] - a_bar[i]) / a_bar[i],  # Fractional residual
                    'log_residual': np.log10(a_obs[i] + 1e-12) - np.log10(a_bar[i] + 1e-12),
                    'Q': Q,
                    'sigma_v': sigma_v,
                    'M_total': M_total,
                    'morphology': morphology,
                    'alpha_eff': row.get('alpha_eff', np.nan)
                })
            
            print(f"Q={Q:.2f}, σ_v={sigma_v:.1f} km/s ({len(r)} points)")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    # Convert to DataFrame
    data = pd.DataFrame(data_points)
    print()
    print(f"Collected {len(data)} data points from {data['galaxy'].nunique()} galaxies")
    print()
    
    # Compute correlations
    print("-"*80)
    print("CORRELATION ANALYSIS")
    print("-"*80)
    
    # Remove NaN and infinite values
    data_clean = data[np.isfinite(data['residual']) & 
                      np.isfinite(data['Q']) & 
                      np.isfinite(data['sigma_v'])].copy()
    
    print(f"Clean data points: {len(data_clean)}")
    print()
    
    # Pearson and Spearman correlations
    r_Q_pearson, p_Q_pearson = pearsonr(data_clean['residual'], data_clean['Q'])
    r_Q_spearman, p_Q_spearman = spearmanr(data_clean['residual'], data_clean['Q'])
    
    r_sigma_pearson, p_sigma_pearson = pearsonr(data_clean['residual'], data_clean['sigma_v'])
    r_sigma_spearman, p_sigma_spearman = spearmanr(data_clean['residual'], data_clean['sigma_v'])
    
    print("Residual vs Q:")
    print(f"  Pearson  r = {r_Q_pearson:+.3f}  (p = {p_Q_pearson:.4f})")
    print(f"  Spearman ρ = {r_Q_spearman:+.3f}  (p = {p_Q_spearman:.4f})")
    
    if abs(r_Q_pearson) > 0.3:
        print(f"  → {'STRONG' if abs(r_Q_pearson) > 0.5 else 'MODERATE'} correlation detected!")
    else:
        print(f"  → Weak correlation (need more Q measurements)")
    print()
    
    print("Residual vs sigma_v:")
    print(f"  Pearson  r = {r_sigma_pearson:+.3f}  (p = {p_sigma_pearson:.4f})")
    print(f"  Spearman ρ = {r_sigma_spearman:+.3f}  (p = {p_sigma_spearman:.4f})")
    
    if abs(r_sigma_pearson) > 0.3:
        print(f"  → {'STRONG' if abs(r_sigma_pearson) > 0.5 else 'MODERATE'} correlation detected!")
    else:
        print(f"  → Weak correlation")
    print()
    
    # Per-galaxy averaged correlations
    galaxy_avg = data_clean.groupby('galaxy').agg({
        'residual': 'mean',
        'Q': 'first',
        'sigma_v': 'first',
        'M_total': 'first',
        'alpha_eff': 'first'
    }).reset_index()
    
    r_Q_gal, p_Q_gal = pearsonr(galaxy_avg['residual'], galaxy_avg['Q'])
    r_sigma_gal, p_sigma_gal = pearsonr(galaxy_avg['residual'], galaxy_avg['sigma_v'])
    
    print("Per-Galaxy Averaged Correlations:")
    print(f"  <Residual> vs Q:       r = {r_Q_gal:+.3f}  (p = {p_Q_gal:.4f})")
    print(f"  <Residual> vs sigma_v: r = {r_sigma_gal:+.3f}  (p = {p_sigma_gal:.4f})")
    print()
    
    # Plot
    print("-"*80)
    print("GENERATING PLOTS")
    print("-"*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('RAR Residuals vs Environment (GPM Prediction)', fontsize=14, fontweight='bold')
    
    # Plot 1: Residual vs Q (all points)
    ax = axes[0, 0]
    scatter = ax.scatter(data_clean['Q'], data_clean['residual'], 
                        c=data_clean['sigma_v'], cmap='viridis', 
                        alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5, label='No residual')
    ax.set_xlabel('Toomre Q', fontsize=11)
    ax.set_ylabel('Fractional Residual (a_obs - a_bar)/a_bar', fontsize=11)
    ax.set_title(f'RAR Residual vs Q (r = {r_Q_pearson:+.3f})', fontsize=12)
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label='σ_v [km/s]')
    
    # Plot 2: Residual vs sigma_v (all points)
    ax = axes[0, 1]
    scatter = ax.scatter(data_clean['sigma_v'], data_clean['residual'],
                        c=data_clean['Q'], cmap='plasma',
                        alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5, label='No residual')
    ax.set_xlabel('Velocity Dispersion σ_v [km/s]', fontsize=11)
    ax.set_ylabel('Fractional Residual (a_obs - a_bar)/a_bar', fontsize=11)
    ax.set_title(f'RAR Residual vs σ_v (r = {r_sigma_pearson:+.3f})', fontsize=12)
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label='Q')
    
    # Plot 3: Per-galaxy averaged
    ax = axes[1, 0]
    ax.scatter(galaxy_avg['Q'], galaxy_avg['residual'], 
              s=100, c='blue', alpha=0.6, edgecolors='k', linewidth=1)
    for _, row in galaxy_avg.iterrows():
        ax.annotate(row['galaxy'], (row['Q'], row['residual']), 
                   fontsize=7, alpha=0.7)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Toomre Q (galaxy-averaged)', fontsize=11)
    ax.set_ylabel('Mean Residual', fontsize=11)
    ax.set_title(f'Per-Galaxy: <Residual> vs Q (r = {r_Q_gal:+.3f})', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Plot 4: Per-galaxy sigma_v
    ax = axes[1, 1]
    ax.scatter(galaxy_avg['sigma_v'], galaxy_avg['residual'],
              s=100, c='green', alpha=0.6, edgecolors='k', linewidth=1)
    for _, row in galaxy_avg.iterrows():
        ax.annotate(row['galaxy'], (row['sigma_v'], row['residual']),
                   fontsize=7, alpha=0.7)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('σ_v (galaxy-averaged) [km/s]', fontsize=11)
    ax.set_ylabel('Mean Residual', fontsize=11)
    ax.set_title(f'Per-Galaxy: <Residual> vs σ_v (r = {r_sigma_gal:+.3f})', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = 'outputs/gpm_tests'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'rar_scatter_vs_environment.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Save data
    data_clean.to_csv(os.path.join(output_dir, 'rar_residuals_vs_environment.csv'), index=False)
    print(f"Saved: {os.path.join(output_dir, 'rar_residuals_vs_environment.csv')}")
    
    plt.close()
    
    # Interpretation
    print()
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if abs(r_Q_pearson) > 0.5 or abs(r_sigma_pearson) > 0.5:
        print("✓ STRONG CORRELATION DETECTED")
        print(f"  Residuals {'anti-' if r_Q_pearson < 0 or r_sigma_pearson < 0 else ''}correlate with Q/sigma_v")
        print("  Consistent with GPM gating mechanism!")
    elif abs(r_Q_pearson) > 0.3 or abs(r_sigma_pearson) > 0.3:
        print("⚠ MODERATE CORRELATION DETECTED")
        print("  Suggests GPM gating effect, but need more data")
    else:
        print("✗ WEAK CORRELATION")
        print("  Either:")
        print("  1. Need more galaxies with diverse Q values")
        print("  2. GPM gating mechanism may need refinement")
        print("  3. MOND/DM may be correct (no Q dependence)")
    
    print()
    print("FALSIFICATION TEST:")
    if abs(r_Q_pearson) < 0.1 and abs(r_sigma_pearson) < 0.1:
        print("  If this persists with 50+ galaxies → GPM gating FALSIFIED")
    else:
        print("  GPM prediction holding up so far")
    
    print()
    print("="*80)
    
    return {
        'r_Q_pearson': r_Q_pearson,
        'p_Q_pearson': p_Q_pearson,
        'r_sigma_pearson': r_sigma_pearson,
        'p_sigma_pearson': p_sigma_pearson,
        'n_points': len(data_clean),
        'n_galaxies': data['galaxy'].nunique()
    }


if __name__ == '__main__':
    results = analyze_rar_scatter()
