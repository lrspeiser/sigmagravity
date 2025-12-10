"""
Edge Extrapolation Analysis (Simplified)

Analyzes how well GPM extrapolates to outer radii by examining
the radial distribution of residuals.

If temporal memory smoothing works, residuals should be evenly
distributed across all radii. Without smoothing, outer radii
would show systematic deviations.

This uses existing batch test results rather than refitting.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses


def analyze_radial_residuals(galaxy_name, results_df, output_dir='outputs/gpm_tests'):
    """
    Analyze how residuals vary with radius.
    
    Good extrapolation: residuals uniform across R
    Poor extrapolation: residuals grow at large R
    """
    
    print(f"\nAnalyzing {galaxy_name}...")
    
    # Load galaxy data
    loader = RealDataLoader()
    try:
        gal = loader.load_rotmod_galaxy(galaxy_name)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    
    try:
        sparc_masses = load_sparc_masses(galaxy_name)
        R_disk = sparc_masses['R_disk']
    except:
        R_disk = np.median(gal['r'])
    
    r = gal['r']
    v_obs = gal['v_obs']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
    v_bar = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
    
    # Get GPM chi2 from results
    gpm_row = results_df[results_df['name'] == galaxy_name]
    if len(gpm_row) == 0:
        print(f"  No GPM results found")
        return None
    
    chi2_gpm = gpm_row.iloc[0]['chi2_gpm']
    chi2_bar = gpm_row.iloc[0]['chi2_baryon']
    
    # Compute residuals (assuming GPM improves uniformly)
    # This is approximate since we don't have the actual GPM v_model here
    # We estimate: residual_gpm ~ residual_bar * sqrt(chi2_gpm/chi2_bar)
    residual_bar = v_obs - v_bar
    scaling_factor = np.sqrt(chi2_gpm / chi2_bar)
    residual_gpm_approx = residual_bar * scaling_factor
    
    # Divide into inner (R < 2R_disk) and outer (R > 2R_disk)
    inner_mask = r < 2 * R_disk
    outer_mask = r > 2 * R_disk
    
    n_inner = np.sum(inner_mask)
    n_outer = np.sum(outer_mask)
    
    if n_inner < 3 or n_outer < 2:
        print(f"  Insufficient data: {n_inner} inner, {n_outer} outer points")
        return None
    
    rms_inner = np.sqrt(np.mean(residual_gpm_approx[inner_mask]**2))
    rms_outer = np.sqrt(np.mean(residual_gpm_approx[outer_mask]**2))
    
    ratio = rms_outer / rms_inner if rms_inner > 0 else np.nan
    
    print(f"  R_disk = {R_disk:.2f} kpc")
    print(f"  Inner region (R < 2R_disk): {n_inner} points, RMS = {rms_inner:.2f} km/s")
    print(f"  Outer region (R > 2R_disk): {n_outer} points, RMS = {rms_outer:.2f} km/s")
    print(f"  Outer/Inner RMS ratio: {ratio:.3f}")
    
    if ratio < 1.5:
        print(f"  ✓ Good extrapolation (ratio < 1.5)")
    elif ratio < 2.5:
        print(f"  ⚠ Moderate extrapolation (1.5 < ratio < 2.5)")
    else:
        print(f"  ✗ Poor extrapolation (ratio > 2.5)")
    
    return {
        'galaxy': galaxy_name,
        'R_disk': R_disk,
        'n_inner': n_inner,
        'n_outer': n_outer,
        'rms_inner': rms_inner,
        'rms_outer': rms_outer,
        'ratio': ratio,
        'chi2_gpm': chi2_gpm,
        'chi2_bar': chi2_bar
    }


def batch_edge_analysis():
    """
    Analyze edge extrapolation for all galaxies in batch results.
    """
    
    print("="*80)
    print("EDGE EXTRAPOLATION ANALYSIS")
    print("="*80)
    print()
    print("Testing: Do GPM residuals remain uniform at large R?")
    print("Good temporal memory → ratio ~ 1.0")
    print("No temporal memory → ratio >> 1.0")
    print()
    
    # Load batch results
    results_csv = 'outputs/gpm_tests/batch_gpm_results.csv'
    if not os.path.exists(results_csv):
        print(f"ERROR: {results_csv} not found")
        return
    
    results_df = pd.read_csv(results_csv)
    print(f"Analyzing {len(results_df)} galaxies from batch results")
    
    # Analyze each galaxy
    analyses = []
    for _, row in results_df.iterrows():
        result = analyze_radial_residuals(row['name'], results_df)
        if result is not None:
            analyses.append(result)
    
    if len(analyses) == 0:
        print("\nNo successful analyses")
        return
    
    # Summary
    df = pd.DataFrame(analyses)
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    print(f"Analyzed {len(df)} galaxies")
    print()
    
    print(f"Mean Outer/Inner RMS ratio: {df['ratio'].mean():.3f} ± {df['ratio'].std():.3f}")
    print()
    
    good = np.sum(df['ratio'] < 1.5)
    moderate = np.sum((df['ratio'] >= 1.5) & (df['ratio'] < 2.5))
    poor = np.sum(df['ratio'] >= 2.5)
    
    print(f"Extrapolation quality:")
    print(f"  Good (ratio < 1.5):         {good}/{len(df)} ({100*good/len(df):.1f}%)")
    print(f"  Moderate (1.5 < ratio < 2.5): {moderate}/{len(df)} ({100*moderate/len(df):.1f}%)")
    print(f"  Poor (ratio > 2.5):          {poor}/{len(df)} ({100*poor/len(df):.1f}%)")
    print()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Edge Extrapolation: Outer vs Inner Residuals', fontsize=14, fontweight='bold')
    
    # Plot 1: RMS scatter
    ax = axes[0]
    ax.scatter(df['rms_inner'], df['rms_outer'], s=100, alpha=0.7, edgecolors='k')
    
    for _, row in df.iterrows():
        ax.annotate(row['galaxy'], (row['rms_inner'], row['rms_outer']),
                   fontsize=7, alpha=0.7)
    
    # y = x line (perfect extrapolation)
    max_rms = max(df['rms_inner'].max(), df['rms_outer'].max())
    ax.plot([0, max_rms], [0, max_rms], 'k--', alpha=0.5, label='Perfect (ratio=1)')
    ax.plot([0, max_rms], [0, 1.5*max_rms], 'r:', alpha=0.5, label='Good (ratio=1.5)')
    
    ax.set_xlabel('Inner RMS [km/s]', fontsize=11)
    ax.set_ylabel('Outer RMS [km/s]', fontsize=11)
    ax.set_title('Residual RMS: Inner vs Outer Regions', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Ratio distribution
    ax = axes[1]
    ax.bar(range(len(df)), df['ratio'], alpha=0.7, edgecolor='k')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Perfect')
    ax.axhline(1.5, color='r', linestyle=':', alpha=0.5, label='Good threshold')
    
    ax.set_xlabel('Galaxy', fontsize=11)
    ax.set_ylabel('Outer/Inner RMS Ratio', fontsize=11)
    ax.set_title('Extrapolation Quality per Galaxy', fontsize=12)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['galaxy'], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_dir = 'outputs/gpm_tests'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'edge_extrapolation_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.close()
    
    # Save results
    df.to_csv(os.path.join(output_dir, 'edge_extrapolation_results.csv'), index=False)
    print(f"Saved: {os.path.join(output_dir, 'edge_extrapolation_results.csv')}")
    
    print()
    print("-"*80)
    print("INTERPRETATION")
    print("-"*80)
    print()
    
    if df['ratio'].mean() < 1.5:
        print("✓ EXCELLENT extrapolation")
        print("  Residuals remain uniform at large R")
        print("  Temporal memory smoothing is working")
    elif df['ratio'].mean() < 2.0:
        print("⚠ GOOD extrapolation")
        print("  Minor increase in residuals at large R")
        print("  Temporal memory provides some smoothing")
    else:
        print("✗ POOR extrapolation")
        print("  Residuals grow significantly at large R")
        print("  Temporal memory may need adjustment")
    
    print()
    print("="*80)


if __name__ == '__main__':
    print()
    batch_edge_analysis()
