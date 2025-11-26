"""
Comprehensive Gate Prediction Tests
====================================

Tests the gate predictions from §6.3:
1. G_bar: Strongly barred galaxies should show more suppression than unbarred
2. G_bulge: High bulge-to-disk ratio galaxies should benefit more from bulge gate
3. G_shear: High shear galaxies should show more suppression

These tests, combined with the already-confirmed inclination/winding test,
establish that the gates make successful a priori predictions rather than
being post-hoc epicycles.

Author: Leonard Speiser
Date: 2025-11-26
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Add spiral folder to path
SCRIPT_DIR = Path(__file__).parent.parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "many_path_model"))

from validation_suite_winding import ValidationSuite
from path_spectrum_kernel_winding import PathSpectrumKernel, PathSpectrumHyperparams


def compute_rar_scatter(suite, df_subset, hp, subset_name):
    """
    Compute RAR scatter for a subset of galaxies.
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
        
        # Get bar strength for this galaxy
        bar_strength = galaxy.get('bar_strength', 0.0)
        if bar_strength is None or np.isnan(bar_strength):
            bar_strength = 0.0
        
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
                bar_strength=float(bar_strength),
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
    
    # RAR fit
    def rar_func(g_bar, g_dagger):
        x = np.sqrt(g_bar / g_dagger)
        return g_bar / (1 - np.exp(-x))
    
    def fit_residuals(params):
        g_dagger = params[0]
        g_rar = rar_func(g_bar_arr, g_dagger)
        log_resid = np.log10(g_model_arr) - np.log10(g_rar)
        return np.sum(log_resid**2)
    
    result = minimize(fit_residuals, x0=[1.2e-10], bounds=[(1e-12, 1e-8)], method='L-BFGS-B')
    g_dagger_fit = result.x[0]
    
    g_rar_fit = rar_func(g_bar_arr, g_dagger_fit)
    log_residuals = np.log10(g_model_arr) - np.log10(g_rar_fit)
    rar_scatter = np.std(log_residuals)
    
    return rar_scatter, n_galaxies_used


def test_bar_gate():
    """
    Test 1: Barred vs Unbarred Galaxies
    
    Prediction: The bar gate (G_bar) should help more for strongly barred galaxies
    because bars disrupt coherent orbital phases through non-axisymmetric perturbations.
    """
    print("\n" + "=" * 80)
    print("TEST 1: BARRED vs UNBARRED GALAXIES")
    print("Prediction: Bar gate should help MORE for strongly barred galaxies")
    print("=" * 80)
    
    # Load data
    output_dir = SCRIPT_DIR / "outputs" / "gate_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = ValidationSuite(output_dir, load_sparc=True)
    
    df = suite.sparc_data
    df_valid = df[df['r_all'].notna()].copy()
    
    # Load bar classification
    bar_csv = REPO_ROOT / "many_path_model" / "bt_law" / "sparc_bar_classification.csv"
    if bar_csv.exists():
        bar_df = pd.read_csv(bar_csv)
        df_valid = df_valid.merge(bar_df[['name', 'bar_class', 'bar_strength']], 
                                   left_on='Galaxy', right_on='name', how='left')
        df_valid['bar_strength'] = df_valid['bar_strength'].fillna(0.2)
    else:
        print("WARNING: Bar classification file not found!")
        return None
    
    # Split by bar classification
    df_barred = df_valid[df_valid['bar_class'] == 'SB'].copy()  # Strongly barred
    df_unbarred = df_valid[df_valid['bar_class'] == 'S'].copy()  # Unbarred
    df_intermediate = df_valid[df_valid['bar_class'] == 'SAB'].copy()  # Intermediate
    
    print(f"\nSample sizes:")
    print(f"  Strongly barred (SB): {len(df_barred)} galaxies")
    print(f"  Unbarred (S): {len(df_unbarred)} galaxies")
    print(f"  Intermediate (SAB): {len(df_intermediate)} galaxies")
    
    # Define hyperparameters - without bar gate
    hp_no_bar = PathSpectrumHyperparams(
        L_0=4.993, beta_bulge=1.759, alpha_shear=0.149, gamma_bar=0.0,
        A_0=1.1, p=0.75, n_coh=0.5, g_dagger=1.2e-10,
        use_winding=True, N_crit=150.0, t_age=10.0, wind_power=1.0
    )
    
    # With bar gate (gamma_bar > 0 means bar suppresses coherence)
    hp_with_bar = PathSpectrumHyperparams(
        L_0=4.993, beta_bulge=1.759, alpha_shear=0.149, gamma_bar=0.5,
        A_0=1.1, p=0.75, n_coh=0.5, g_dagger=1.2e-10,
        use_winding=True, N_crit=150.0, t_age=10.0, wind_power=1.0
    )
    
    results = []
    
    for group_name, df_group in [("Strongly Barred (SB)", df_barred), 
                                   ("Unbarred (S)", df_unbarred)]:
        if len(df_group) < 3:
            continue
            
        print(f"\n--- {group_name} ---")
        
        rar_no_bar, n1 = compute_rar_scatter(suite, df_group, hp_no_bar, group_name)
        rar_with_bar, n2 = compute_rar_scatter(suite, df_group, hp_with_bar, group_name)
        
        if np.isnan(rar_no_bar) or np.isnan(rar_with_bar):
            print(f"  Insufficient data")
            continue
        
        improvement = (rar_no_bar - rar_with_bar) / rar_no_bar * 100
        
        print(f"  n = {n1} galaxies")
        print(f"  RAR scatter (no bar gate): {rar_no_bar:.4f} dex")
        print(f"  RAR scatter (with bar gate): {rar_with_bar:.4f} dex")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results.append({
            'test': 'bar_gate',
            'group': group_name,
            'n_galaxies': n1,
            'rar_without': rar_no_bar,
            'rar_with': rar_with_bar,
            'improvement_pct': improvement
        })
    
    return results


def test_bulge_gate():
    """
    Test 2: High vs Low Bulge-to-Disk Ratio
    
    Prediction: The bulge gate (G_bulge) should help more for bulge-dominated galaxies
    because bulges are pressure-supported, disrupting coherent orbital phases.
    """
    print("\n" + "=" * 80)
    print("TEST 2: HIGH vs LOW BULGE-TO-DISK RATIO")
    print("Prediction: Bulge gate should help MORE for bulge-dominated galaxies")
    print("=" * 80)
    
    output_dir = SCRIPT_DIR / "outputs" / "gate_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = ValidationSuite(output_dir, load_sparc=True)
    
    df = suite.sparc_data
    df_valid = df[df['r_all'].notna()].copy()
    
    # Load bulge fractions from sparc_combined.csv
    sparc_csv = REPO_ROOT / "data" / "sparc" / "sparc_combined.csv"
    if sparc_csv.exists():
        sparc_df = pd.read_csv(sparc_csv)
        df_valid = df_valid.merge(sparc_df[['galaxy_name', 'bulge_frac']], 
                                   left_on='Galaxy', right_on='galaxy_name', how='left')
        df_valid['bulge_frac'] = df_valid['bulge_frac'].fillna(0.0)
    else:
        print("WARNING: sparc_combined.csv not found!")
        return None
    
    # Split by bulge fraction
    # High B/D: bulge_frac > 0.15 (significant bulge)
    # Low B/D: bulge_frac < 0.05 (disk-dominated)
    df_high_bulge = df_valid[df_valid['bulge_frac'] > 0.15].copy()
    df_low_bulge = df_valid[df_valid['bulge_frac'] < 0.05].copy()
    
    print(f"\nSample sizes:")
    print(f"  High B/D (>0.15): {len(df_high_bulge)} galaxies")
    print(f"  Low B/D (<0.05): {len(df_low_bulge)} galaxies")
    
    # Without bulge gate
    hp_no_bulge = PathSpectrumHyperparams(
        L_0=4.993, beta_bulge=0.0, alpha_shear=0.149, gamma_bar=0.0,
        A_0=1.1, p=0.75, n_coh=0.5, g_dagger=1.2e-10,
        use_winding=True, N_crit=150.0, t_age=10.0, wind_power=1.0
    )
    
    # With bulge gate
    hp_with_bulge = PathSpectrumHyperparams(
        L_0=4.993, beta_bulge=1.759, alpha_shear=0.149, gamma_bar=0.0,
        A_0=1.1, p=0.75, n_coh=0.5, g_dagger=1.2e-10,
        use_winding=True, N_crit=150.0, t_age=10.0, wind_power=1.0
    )
    
    results = []
    
    for group_name, df_group in [("High B/D (>0.15)", df_high_bulge), 
                                   ("Low B/D (<0.05)", df_low_bulge)]:
        if len(df_group) < 3:
            print(f"\n--- {group_name} ---")
            print(f"  Insufficient galaxies ({len(df_group)})")
            continue
            
        print(f"\n--- {group_name} ---")
        
        rar_no_bulge, n1 = compute_rar_scatter(suite, df_group, hp_no_bulge, group_name)
        rar_with_bulge, n2 = compute_rar_scatter(suite, df_group, hp_with_bulge, group_name)
        
        if np.isnan(rar_no_bulge) or np.isnan(rar_with_bulge):
            print(f"  Insufficient data")
            continue
        
        improvement = (rar_no_bulge - rar_with_bulge) / rar_no_bulge * 100
        
        print(f"  n = {n1} galaxies")
        print(f"  RAR scatter (no bulge gate): {rar_no_bulge:.4f} dex")
        print(f"  RAR scatter (with bulge gate): {rar_with_bulge:.4f} dex")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results.append({
            'test': 'bulge_gate',
            'group': group_name,
            'n_galaxies': n1,
            'rar_without': rar_no_bulge,
            'rar_with': rar_with_bulge,
            'improvement_pct': improvement
        })
    
    return results


def test_shear_gate():
    """
    Test 3: High vs Low Velocity Shear
    
    Prediction: The shear gate (G_shear) should help more for high-shear galaxies
    because high dv/dR leads to rapid phase mixing and decoherence.
    """
    print("\n" + "=" * 80)
    print("TEST 3: HIGH vs LOW VELOCITY SHEAR")
    print("Prediction: Shear gate should help MORE for high-shear galaxies")
    print("=" * 80)
    
    output_dir = SCRIPT_DIR / "outputs" / "gate_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = ValidationSuite(output_dir, load_sparc=True)
    
    df = suite.sparc_data
    df_valid = df[df['r_all'].notna()].copy()
    
    # Compute mean shear |dv/dR| for each galaxy
    shear_values = []
    for idx, galaxy in df_valid.iterrows():
        v_all = galaxy['v_all']
        r_all = galaxy['r_all']
        
        if v_all is None or r_all is None or len(r_all) < 4:
            shear_values.append(np.nan)
            continue
        
        # Compute dv/dR using gradient
        dr = np.diff(r_all)
        dv = np.diff(v_all)
        
        # Avoid division by zero
        valid = dr > 0.1
        if np.sum(valid) < 2:
            shear_values.append(np.nan)
            continue
        
        dvdr = np.abs(dv[valid] / dr[valid])
        mean_shear = np.nanmean(dvdr)
        shear_values.append(mean_shear)
    
    df_valid['mean_shear'] = shear_values
    df_valid = df_valid[df_valid['mean_shear'].notna()].copy()
    
    # Split by shear (median split)
    median_shear = df_valid['mean_shear'].median()
    df_high_shear = df_valid[df_valid['mean_shear'] > median_shear * 1.5].copy()
    df_low_shear = df_valid[df_valid['mean_shear'] < median_shear * 0.67].copy()
    
    print(f"\nMedian shear: {median_shear:.2f} km/s/kpc")
    print(f"Sample sizes:")
    print(f"  High shear (>1.5×median): {len(df_high_shear)} galaxies")
    print(f"  Low shear (<0.67×median): {len(df_low_shear)} galaxies")
    
    # Without shear gate
    hp_no_shear = PathSpectrumHyperparams(
        L_0=4.993, beta_bulge=1.759, alpha_shear=0.0, gamma_bar=0.0,
        A_0=1.1, p=0.75, n_coh=0.5, g_dagger=1.2e-10,
        use_winding=True, N_crit=150.0, t_age=10.0, wind_power=1.0
    )
    
    # With shear gate
    hp_with_shear = PathSpectrumHyperparams(
        L_0=4.993, beta_bulge=1.759, alpha_shear=0.149, gamma_bar=0.0,
        A_0=1.1, p=0.75, n_coh=0.5, g_dagger=1.2e-10,
        use_winding=True, N_crit=150.0, t_age=10.0, wind_power=1.0
    )
    
    results = []
    
    for group_name, df_group in [("High Shear", df_high_shear), 
                                   ("Low Shear", df_low_shear)]:
        if len(df_group) < 3:
            print(f"\n--- {group_name} ---")
            print(f"  Insufficient galaxies ({len(df_group)})")
            continue
            
        print(f"\n--- {group_name} ---")
        
        rar_no_shear, n1 = compute_rar_scatter(suite, df_group, hp_no_shear, group_name)
        rar_with_shear, n2 = compute_rar_scatter(suite, df_group, hp_with_shear, group_name)
        
        if np.isnan(rar_no_shear) or np.isnan(rar_with_shear):
            print(f"  Insufficient data")
            continue
        
        improvement = (rar_no_shear - rar_with_shear) / rar_no_shear * 100
        
        print(f"  n = {n1} galaxies")
        print(f"  Mean shear: {df_group['mean_shear'].mean():.2f} km/s/kpc")
        print(f"  RAR scatter (no shear gate): {rar_no_shear:.4f} dex")
        print(f"  RAR scatter (with shear gate): {rar_with_shear:.4f} dex")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results.append({
            'test': 'shear_gate',
            'group': group_name,
            'n_galaxies': n1,
            'rar_without': rar_no_shear,
            'rar_with': rar_with_shear,
            'improvement_pct': improvement
        })
    
    return results


def create_summary_plot(all_results, output_dir):
    """Create a summary plot of all gate tests."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    tests = [
        ('bar_gate', 'Bar Gate Test', axes[0]),
        ('bulge_gate', 'Bulge Gate Test', axes[1]),
        ('shear_gate', 'Shear Gate Test', axes[2])
    ]
    
    for test_name, title, ax in tests:
        test_results = [r for r in all_results if r['test'] == test_name]
        if len(test_results) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        groups = [r['group'].split('(')[0].strip() for r in test_results]
        improvements = [r['improvement_pct'] for r in test_results]
        colors = ['forestgreen' if x > 0 else 'crimson' for x in improvements]
        
        bars = ax.bar(groups, improvements, color=colors, alpha=0.8)
        ax.set_ylabel('Improvement from Gate (%)')
        ax.set_title(title)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            ax.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3 if height > 0 else -12), textcoords="offset points", 
                       ha='center', va='bottom' if height > 0 else 'top', 
                       fontsize=10, fontweight='bold')
        
        # Check prediction
        if len(test_results) >= 2:
            # First group should benefit MORE for bar/bulge/shear tests
            if test_results[0]['improvement_pct'] > test_results[1]['improvement_pct']:
                status = '✓ Prediction confirmed'
                color = 'green'
            else:
                status = '✗ Not confirmed'
                color = 'red'
            ax.text(0.5, -0.15, status, transform=ax.transAxes, ha='center', 
                   fontsize=9, color=color)
    
    plt.tight_layout()
    
    plot_path = output_dir / "gate_prediction_tests.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSummary plot saved to: {plot_path}")
    
    # Also save to figures folder
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / "gate_prediction_tests.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Plot also saved to: {fig_path}")
    
    plt.close()


def main():
    """Run all gate prediction tests."""
    
    print("=" * 80)
    print("COMPREHENSIVE GATE PREDICTION TESTS")
    print("Testing §6.3 predictions about gate mechanisms")
    print("=" * 80)
    
    output_dir = SCRIPT_DIR / "outputs" / "gate_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Test 1: Bar gate
    bar_results = test_bar_gate()
    if bar_results:
        all_results.extend(bar_results)
    
    # Test 2: Bulge gate
    bulge_results = test_bulge_gate()
    if bulge_results:
        all_results.extend(bulge_results)
    
    # Test 3: Shear gate
    shear_results = test_shear_gate()
    if shear_results:
        all_results.extend(shear_results)
    
    # Save results
    results_path = output_dir / "gate_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Create summary plot
    if len(all_results) >= 2:
        create_summary_plot(all_results, output_dir)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    confirmed = 0
    total = 0
    
    for test_name in ['bar_gate', 'bulge_gate', 'shear_gate']:
        test_results = [r for r in all_results if r['test'] == test_name]
        if len(test_results) >= 2:
            total += 1
            # Check if high-impact group benefits more
            if test_results[0]['improvement_pct'] > test_results[1]['improvement_pct']:
                confirmed += 1
                print(f"✓ {test_name}: Prediction CONFIRMED")
                print(f"    {test_results[0]['group']}: {test_results[0]['improvement_pct']:+.1f}%")
                print(f"    {test_results[1]['group']}: {test_results[1]['improvement_pct']:+.1f}%")
            else:
                print(f"✗ {test_name}: Prediction NOT confirmed")
                print(f"    {test_results[0]['group']}: {test_results[0]['improvement_pct']:+.1f}%")
                print(f"    {test_results[1]['group']}: {test_results[1]['improvement_pct']:+.1f}%")
    
    print(f"\nOverall: {confirmed}/{total} predictions confirmed")
    
    if confirmed > 0:
        print("\n*** At least one gate prediction confirmed - these are NOT epicycles! ***")
    
    return all_results


if __name__ == "__main__":
    main()
