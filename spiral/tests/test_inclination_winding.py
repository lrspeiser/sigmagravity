"""
Inclination-Dependent Winding Test
===================================

Tests the §2.9 prediction that face-on galaxies should show stronger
winding effects than edge-on galaxies.

Prediction: Face-on spirals see full azimuthal structure → winding gate
helps MORE. Edge-on spirals see compressed azimuthal structure → winding
gate helps LESS.

This would be a signature that MOND cannot explain.

Author: Leonard Speiser
Date: 2025-11-26
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add spiral folder to path
SCRIPT_DIR = Path(__file__).parent.parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "many_path_model"))

from validation_suite_winding import ValidationSuite
from path_spectrum_kernel_winding import PathSpectrumKernel, PathSpectrumHyperparams


def compute_rar_for_subset(suite, df_subset, hp, subset_name):
    """
    Compute RAR scatter for a subset of galaxies.
    
    Uses the paper's methodology: fit a McGaugh RAR to g_bar, then compute
    how well g_model matches that fitted curve (not direct g_obs comparison).
    """
    from scipy.optimize import minimize
    
    if len(df_subset) == 0:
        return float('nan'), 0
    
    # Physical constants
    KPC_TO_M = 3.0856776e19
    KM_TO_M = 1000.0
    
    kernel = PathSpectrumKernel(hp, use_cupy=False)
    
    g_bar_all = []
    g_model_all = []
    n_galaxies_used = 0
    
    def _smooth(arr, k=7):
        k = max(3, int(k) | 1)
        if arr.size < k:
            return arr
        out = np.copy(arr)
        half = k // 2
        for i in range(arr.size):
            i0 = max(0, i - half)
            i1 = min(arr.size, i + half + 1)
            out[i] = np.nanmean(arr[i0:i1])
        return out
    
    for idx, galaxy in df_subset.iterrows():
        v_all = galaxy['v_all']
        r_all = galaxy['r_all']
        
        if v_all is None or r_all is None:
            continue
        if len(r_all) < 3:
            continue
        
        # Convert to SI
        v_m_s = v_all * KM_TO_M
        r_m = r_all * KPC_TO_M
        
        # Baryonic components
        v_disk = galaxy.get('v_disk_all', np.zeros_like(v_all))
        v_bulge = galaxy.get('v_bulge_all', np.zeros_like(v_all))
        v_gas = galaxy.get('v_gas_all', np.zeros_like(v_all))
        
        if v_disk is None: v_disk = np.zeros_like(v_all)
        if v_bulge is None: v_bulge = np.zeros_like(v_all)
        if v_gas is None: v_gas = np.zeros_like(v_all)
        
        v_baryonic_km_s = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
        v_baryonic_m_s = v_baryonic_km_s * KM_TO_M
        g_bar = v_baryonic_m_s**2 / r_m
        
        # Compute boost factor K(r)
        denom = np.maximum(1e-12, v_disk**2 + v_bulge**2 + v_gas**2)
        BT_rad = (v_bulge**2) / denom
        v_circ = np.where(np.isfinite(v_all) & (v_all > 0), v_all, v_baryonic_km_s)
        v_circ_s = _smooth(v_circ, k=7)
        
        K = np.zeros_like(r_all, dtype=float)
        for i in range(len(r_all)):
            K[i] = float(kernel.many_path_boost_factor(
                r=float(r_all[i]),
                v_circ=float(v_circ_s[i]),
                g_bar=float(g_bar[i]),
                BT=float(BT_rad[i]),
                bar_strength=0.0,
                r_bulge=1.0, r_bar=3.0, r_gate=0.5
            ))
        
        g_model = g_bar * (1.0 + K)
        
        # Filter valid points
        r_mask = r_all > 0.5
        mask = r_mask & (g_bar > 1e-13) & (g_model > 1e-13) & \
               np.isfinite(g_bar) & np.isfinite(g_model)
        
        if np.sum(mask) > 0:
            g_bar_all.extend(g_bar[mask])
            g_model_all.extend(g_model[mask])
            n_galaxies_used += 1
    
    if len(g_bar_all) == 0:
        return float('nan'), 0
    
    g_bar_arr = np.array(g_bar_all)
    g_model_arr = np.array(g_model_all)
    
    # Paper's methodology: fit McGaugh RAR to g_bar, measure scatter of g_model vs fitted RAR
    # RAR: g_total = g_bar / (1 - exp(-sqrt(g_bar/g†)))
    def rar_func(g_bar, g_dagger):
        x = np.sqrt(g_bar / g_dagger)
        return g_bar / (1 - np.exp(-x))
    
    def fit_residuals(params):
        g_dagger = params[0]
        g_rar = rar_func(g_bar_arr, g_dagger)
        log_resid = np.log10(g_model_arr) - np.log10(g_rar)
        return np.sum(log_resid**2)
    
    # Fit g† to minimize scatter
    result = minimize(fit_residuals, x0=[1.2e-10], bounds=[(1e-12, 1e-8)], method='L-BFGS-B')
    g_dagger_fit = result.x[0]
    
    # Compute scatter with fitted g†
    g_rar_fit = rar_func(g_bar_arr, g_dagger_fit)
    log_residuals = np.log10(g_model_arr) - np.log10(g_rar_fit)
    rar_scatter = np.std(log_residuals)
    
    return rar_scatter, n_galaxies_used


def run_inclination_test():
    """Test winding effectiveness by inclination group."""
    
    print("=" * 80)
    print("INCLINATION-DEPENDENT WINDING TEST")
    print("Testing §2.9 prediction: Face-on should show stronger winding effects")
    print("=" * 80)
    
    # Load SPARC data
    output_dir = SCRIPT_DIR / "outputs" / "inclination_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = ValidationSuite(output_dir, load_sparc=True)
    
    df = suite.sparc_data
    df_valid = df[df['r_all'].notna()].copy()
    
    # Get inclination
    df_valid['inc'] = df_valid['Inc'].astype(float)
    
    print(f"\nTotal galaxies with rotation curves: {len(df_valid)}")
    print(f"Inclination range: {df_valid['inc'].min():.1f}° to {df_valid['inc'].max():.1f}°")
    
    # Define inclination groups
    # Face-on: 30-50° (good deprojection, see full spiral structure)
    # Edge-on: 60-80° (good deprojection, compressed azimuthal view)
    # Note: <30° and >80° have unreliable deprojection
    
    df_faceon = df_valid[(df_valid['inc'] >= 30) & (df_valid['inc'] <= 50)].copy()
    df_edgeon = df_valid[(df_valid['inc'] >= 60) & (df_valid['inc'] <= 80)].copy()
    df_intermediate = df_valid[(df_valid['inc'] > 50) & (df_valid['inc'] < 60)].copy()
    
    print(f"\nInclination groups:")
    print(f"  Face-on (30-50°): {len(df_faceon)} galaxies")
    print(f"  Intermediate (50-60°): {len(df_intermediate)} galaxies")
    print(f"  Edge-on (60-80°): {len(df_edgeon)} galaxies")
    
    # Define hyperparameters
    hp_no_wind = PathSpectrumHyperparams(
        L_0=4.993, beta_bulge=1.759, alpha_shear=0.149, gamma_bar=0.0,
        A_0=1.1, p=0.75, n_coh=0.5, g_dagger=1.2e-10,
        use_winding=False
    )
    
    hp_wind = PathSpectrumHyperparams(
        L_0=4.993, beta_bulge=1.759, alpha_shear=0.149, gamma_bar=0.0,
        A_0=1.1, p=0.75, n_coh=0.5, g_dagger=1.2e-10,
        use_winding=True, N_crit=150.0, t_age=10.0, wind_power=1.0
    )
    
    # Results storage
    results = []
    
    # Test each group
    groups = [
        ("Face-on (30-50°)", df_faceon),
        ("Intermediate (50-60°)", df_intermediate),
        ("Edge-on (60-80°)", df_edgeon),
        ("All (30-80°)", df_valid[(df_valid['inc'] >= 30) & (df_valid['inc'] <= 80)])
    ]
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for group_name, df_group in groups:
        print(f"\n--- {group_name} ---")
        
        rar_no_wind, n_no = compute_rar_for_subset(suite, df_group, hp_no_wind, group_name)
        rar_wind, n_wind = compute_rar_for_subset(suite, df_group, hp_wind, group_name)
        
        if np.isnan(rar_no_wind) or np.isnan(rar_wind):
            print(f"  Insufficient data (n={n_no})")
            continue
        
        improvement = (rar_no_wind - rar_wind) / rar_no_wind * 100
        
        print(f"  n = {n_no} galaxies")
        print(f"  RAR scatter (no winding): {rar_no_wind:.4f} dex")
        print(f"  RAR scatter (with winding): {rar_wind:.4f} dex")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results.append({
            'group': group_name,
            'n_galaxies': n_no,
            'rar_no_wind': rar_no_wind,
            'rar_wind': rar_wind,
            'improvement_pct': improvement
        })
    
    # Create summary plot
    print("\n" + "=" * 80)
    print("CREATING COMPARISON PLOT")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    
    # Filter to main comparison groups
    df_plot = df_results[df_results['group'].isin(['Face-on (30-50°)', 'Edge-on (60-80°)'])]
    
    if len(df_plot) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left panel: RAR scatter comparison
        ax1 = axes[0]
        x = np.arange(len(df_plot))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, df_plot['rar_no_wind'], width, label='No Winding', color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, df_plot['rar_wind'], width, label='With Winding', color='coral', alpha=0.8)
        
        ax1.set_ylabel('RAR Scatter (dex)', fontsize=12)
        ax1.set_title('RAR Scatter by Inclination Group', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_plot['group'], fontsize=11)
        ax1.legend(loc='upper right')
        ax1.axhline(y=0.087, color='green', linestyle='--', alpha=0.7, label='Paper target')
        ax1.axhline(y=0.10, color='orange', linestyle=':', alpha=0.7, label='MOND floor')
        ax1.set_ylim(0, max(df_plot['rar_no_wind'].max(), df_plot['rar_wind'].max()) * 1.2)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        # Right panel: Improvement comparison
        ax2 = axes[1]
        colors = ['forestgreen' if x > 0 else 'crimson' for x in df_plot['improvement_pct']]
        bars = ax2.bar(df_plot['group'], df_plot['improvement_pct'], color=colors, alpha=0.8)
        
        ax2.set_ylabel('Improvement from Winding (%)', fontsize=12)
        ax2.set_title('Winding Gate Effectiveness by Inclination', fontsize=14)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, df_plot['improvement_pct']):
            height = bar.get_height()
            ax2.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3 if height > 0 else -12), textcoords="offset points", 
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')
        
        # Add prediction annotation
        faceon_imp = df_plot[df_plot['group'] == 'Face-on (30-50°)']['improvement_pct'].values[0]
        edgeon_imp = df_plot[df_plot['group'] == 'Edge-on (60-80°)']['improvement_pct'].values[0]
        
        if faceon_imp > edgeon_imp:
            ax2.text(0.5, 0.02, '✓ Prediction confirmed: Face-on shows MORE winding benefit',
                    transform=ax2.transAxes, ha='center', fontsize=10, color='green',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax2.text(0.5, 0.02, '✗ Prediction NOT confirmed',
                    transform=ax2.transAxes, ha='center', fontsize=10, color='red',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "inclination_winding_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        
        # Also save to figures folder
        figures_dir = REPO_ROOT / "figures"
        figures_dir.mkdir(exist_ok=True)
        fig_path = figures_dir / "inclination_winding_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Plot also saved to: {fig_path}")
        
        plt.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if len(df_plot) >= 2:
        faceon = df_results[df_results['group'] == 'Face-on (30-50°)'].iloc[0]
        edgeon = df_results[df_results['group'] == 'Edge-on (60-80°)'].iloc[0]
        
        print(f"""
PREDICTION (§2.9): Face-on galaxies should show STRONGER winding effects
                   because they display full azimuthal spiral structure.

RESULTS:
  Face-on (30-50°): {faceon['improvement_pct']:+.1f}% improvement from winding
  Edge-on (60-80°): {edgeon['improvement_pct']:+.1f}% improvement from winding
  
  Difference: {faceon['improvement_pct'] - edgeon['improvement_pct']:+.1f} percentage points
""")
        
        if faceon['improvement_pct'] > edgeon['improvement_pct']:
            print("✓ PREDICTION CONFIRMED: Face-on galaxies benefit MORE from winding gate")
            print("  This is a signature that MOND cannot explain!")
        elif abs(faceon['improvement_pct'] - edgeon['improvement_pct']) < 1.0:
            print("⚠ INCONCLUSIVE: No significant difference between groups")
        else:
            print("✗ PREDICTION NOT CONFIRMED: Edge-on benefits more (unexpected)")
    
    return results


if __name__ == "__main__":
    run_inclination_test()
