"""
p-Morphology Correlation Test for Σ-Gravity
=============================================

Tests whether the decoherence exponent p correlates with galaxy morphology,
as predicted by the interaction network interpretation.

Physical Prediction:
- p = interaction dimension d_I
- Smooth systems (ellipticals) → p ≈ 2 (area-like)
- Clumpy systems (late spirals, irregulars) → p < 1 (fractal)

Expected ordering:
p(Early) > p(Intermediate) > p(Late) > p(Irregular)

The fitted global value p ≈ 0.757 suggests sub-linear accumulation consistent
with sparse, clustered interaction networks.

Author: Leonard Speiser
Date: 2025-11-26
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import spearmanr, pearsonr
import json
import warnings
warnings.filterwarnings('ignore')

# Add paths
SCRIPT_DIR = Path(__file__).parent.parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "many_path_model"))

from validation_suite_winding import ValidationSuite


# Physical constants
KPC_TO_M = 3.0856776e19
KM_TO_M = 1000.0
G_DAGGER = 1.2e-10  # m/s^2


def classify_morphology(hubble_type_code):
    """
    Classify SPARC Hubble type code into morphological groups.
    
    SPARC uses numerical T-types:
    0 = S0, 1 = Sa, 2 = Sab, 3 = Sb, 4 = Sbc, 5 = Sc,
    6 = Scd, 7 = Sd, 8 = Sdm, 9 = Sm, 10 = Im, 11 = BCD
    
    Returns: (group_name, group_code)
    """
    try:
        T = int(hubble_type_code)
    except (ValueError, TypeError):
        return 'Unknown', -1
    
    if T <= 1:  # S0, Sa
        return 'Early (S0-Sa)', 4
    elif T <= 3:  # Sab, Sb
        return 'Early-Spiral (Sab-Sb)', 3
    elif T <= 5:  # Sbc, Sc
        return 'Intermediate (Sbc-Sc)', 2
    elif T <= 8:  # Scd, Sd, Sdm
        return 'Late-Spiral (Scd-Sdm)', 1
    elif T <= 11:  # Sm, Im, BCD
        return 'Irregular (Sm-BCD)', 0
    else:
        return 'Unknown', -1


def burr_xii_coherence(R, ell_0, p, n_coh):
    """Burr-XII coherence window."""
    ratio = (R / ell_0) ** p
    return 1 - (1 + ratio) ** (-n_coh)


def sigma_gravity_boost(R, g_bar, A0, ell_0, p, n_coh=0.5):
    """
    Simplified Σ-Gravity boost factor.
    
    K(R) = A0 * (g†/g_bar)^0.5 * C(R; ell_0, p, n_coh)
    """
    C = burr_xii_coherence(R, ell_0, p, n_coh)
    # Use 0.5 power for acceleration ratio (simplified)
    acc_factor = np.where(g_bar > 1e-14, (G_DAGGER / g_bar) ** 0.5, 0)
    return A0 * acc_factor * C


def fit_galaxy_p(r_kpc, v_obs, v_bar, v_err, 
                 A0_fixed=0.591, ell_0_fixed=4.993, n_coh_fixed=0.5,
                 fit_mode='p_only'):
    """
    Fit a single galaxy with p as a free parameter.
    
    fit_mode:
        'p_only' - only fit p (A0, ell_0 fixed)
        'p_and_A0' - fit p and A0
        'all' - fit p, A0, ell_0
    
    Returns: dict with fitted params and chi2
    """
    # Convert to accelerations
    r_m = r_kpc * KPC_TO_M
    v_obs_m = v_obs * KM_TO_M
    v_bar_m = v_bar * KM_TO_M
    v_err_m = v_err * KM_TO_M
    
    g_obs = v_obs_m**2 / r_m
    g_bar = v_bar_m**2 / r_m
    g_err = 2 * v_obs_m * v_err_m / r_m
    
    # Filter valid points
    mask = (r_kpc > 0.5) & (g_bar > 1e-14) & (g_err > 0) & np.isfinite(g_obs)
    if np.sum(mask) < 5:
        return None
    
    r_fit = r_kpc[mask]
    g_obs_fit = g_obs[mask]
    g_bar_fit = g_bar[mask]
    g_err_fit = g_err[mask]
    
    def chi_squared(params):
        if fit_mode == 'p_only':
            p = params[0]
            A0, ell_0 = A0_fixed, ell_0_fixed
        elif fit_mode == 'p_and_A0':
            p, A0 = params
            ell_0 = ell_0_fixed
        else:  # all
            p, A0, ell_0 = params
        
        # Bounds check
        if p <= 0.1 or p > 3.0:
            return 1e10
        if A0 < 0.01 or A0 > 10:
            return 1e10
        if ell_0 < 0.5 or ell_0 > 50:
            return 1e10
        
        K = sigma_gravity_boost(r_fit, g_bar_fit, A0, ell_0, p, n_coh_fixed)
        g_pred = g_bar_fit * (1 + K)
        
        chi2 = np.sum(((g_obs_fit - g_pred) / g_err_fit) ** 2)
        return chi2
    
    # Initial guesses and bounds
    if fit_mode == 'p_only':
        x0 = [0.75]
        bounds = [(0.1, 3.0)]
    elif fit_mode == 'p_and_A0':
        x0 = [0.75, 0.6]
        bounds = [(0.1, 3.0), (0.01, 5.0)]
    else:
        x0 = [0.75, 0.6, 5.0]
        bounds = [(0.1, 3.0), (0.01, 5.0), (0.5, 30.0)]
    
    # Optimize with differential evolution for robustness
    try:
        result = differential_evolution(chi_squared, bounds, seed=42, 
                                        maxiter=500, tol=1e-6, polish=True)
        params_fit = result.x
        chi2_min = result.fun
    except Exception as e:
        return None
    
    # Extract fitted parameters
    if fit_mode == 'p_only':
        p_fit = params_fit[0]
        A0_fit, ell_0_fit = A0_fixed, ell_0_fixed
    elif fit_mode == 'p_and_A0':
        p_fit, A0_fit = params_fit
        ell_0_fit = ell_0_fixed
    else:
        p_fit, A0_fit, ell_0_fit = params_fit
    
    # Estimate uncertainty on p via finite difference
    delta_p = 0.02
    chi2_plus = chi_squared([p_fit + delta_p] if fit_mode == 'p_only' 
                           else [p_fit + delta_p, A0_fit] if fit_mode == 'p_and_A0'
                           else [p_fit + delta_p, A0_fit, ell_0_fit])
    chi2_minus = chi_squared([p_fit - delta_p] if fit_mode == 'p_only'
                            else [p_fit - delta_p, A0_fit] if fit_mode == 'p_and_A0'
                            else [p_fit - delta_p, A0_fit, ell_0_fit])
    
    d2chi2_dp2 = (chi2_plus + chi2_minus - 2 * chi2_min) / (delta_p ** 2)
    
    if d2chi2_dp2 > 0:
        p_err = np.sqrt(2 / d2chi2_dp2)  # 1-sigma from delta-chi^2 = 1
    else:
        p_err = np.nan
    
    # Reduced chi-squared
    dof = np.sum(mask) - len(x0)
    chi2_red = chi2_min / max(1, dof)
    
    return {
        'p_fit': p_fit,
        'p_err': min(p_err, 2.0) if np.isfinite(p_err) else np.nan,
        'A0_fit': A0_fit,
        'ell_0_fit': ell_0_fit,
        'chi2': chi2_min,
        'chi2_red': chi2_red,
        'n_points': np.sum(mask),
        'dof': dof
    }


def run_p_morphology_test(fit_mode='p_only'):
    """
    Main analysis: fit p for each galaxy, test correlation with morphology.
    """
    print("=" * 80)
    print("p-MORPHOLOGY CORRELATION TEST")
    print("Testing whether decoherence exponent p correlates with galaxy structure")
    print("=" * 80)
    
    # Load SPARC data
    output_dir = SCRIPT_DIR / "outputs" / "p_morphology"
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = ValidationSuite(output_dir, load_sparc=True)
    
    df = suite.sparc_data
    df_valid = df[df['r_all'].notna()].copy()
    
    print(f"\nLoaded {len(df_valid)} galaxies with rotation curves")
    print(f"Fit mode: {fit_mode}")
    
    # Get Hubble types
    if 'T' in df_valid.columns:
        df_valid['hubble_T'] = df_valid['T']
    elif 'HubbleType' in df_valid.columns:
        df_valid['hubble_T'] = df_valid['HubbleType']
    else:
        print("Looking for Hubble type column...")
        print(f"Available columns: {df_valid.columns.tolist()}")
        # Try to find it
        for col in df_valid.columns:
            if 'type' in col.lower() or col == 'T':
                print(f"Using column: {col}")
                df_valid['hubble_T'] = df_valid[col]
                break
    
    # Fit each galaxy
    print("\nFitting p for each galaxy...")
    results = []
    n_success = 0
    n_failed = 0
    
    for idx, (row_idx, galaxy) in enumerate(df_valid.iterrows()):
        if idx % 20 == 0:
            print(f"  Processing {idx+1}/{len(df_valid)}...")
        
        v_all = galaxy['v_all']
        r_all = galaxy['r_all']
        
        if v_all is None or r_all is None or len(r_all) < 5:
            n_failed += 1
            continue
        
        # Get baryonic velocity
        v_disk = galaxy.get('v_disk_all', np.zeros_like(v_all))
        v_bulge = galaxy.get('v_bulge_all', np.zeros_like(v_all))
        v_gas = galaxy.get('v_gas_all', np.zeros_like(v_all))
        
        if v_disk is None: v_disk = np.zeros_like(v_all)
        if v_bulge is None: v_bulge = np.zeros_like(v_all)
        if v_gas is None: v_gas = np.zeros_like(v_all)
        
        v_bar = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
        
        # Estimate velocity error (use 10% if not available)
        v_err = galaxy.get('e_v_all', v_all * 0.1)
        if v_err is None:
            v_err = v_all * 0.1
        
        # Fit
        fit_result = fit_galaxy_p(r_all, v_all, v_bar, v_err, fit_mode=fit_mode)
        
        if fit_result is None:
            n_failed += 1
            continue
        
        # Get morphology
        hubble_T = galaxy.get('hubble_T', galaxy.get('T', -999))
        morph_group, morph_code = classify_morphology(hubble_T)
        
        results.append({
            'name': galaxy.get('Galaxy', galaxy.get('name', f'Galaxy_{idx}')),
            'hubble_T': hubble_T,
            'morph_group': morph_group,
            'morph_code': morph_code,
            **fit_result
        })
        n_success += 1
    
    print(f"\nFitting complete: {n_success} successful, {n_failed} failed")
    
    results_df = pd.DataFrame(results)
    
    # Save raw results
    results_df.to_csv(output_dir / "p_fits_by_galaxy.csv", index=False)
    print(f"Raw results saved to: {output_dir / 'p_fits_by_galaxy.csv'}")
    
    return results_df, output_dir


def analyze_p_morphology_correlation(results_df, output_dir):
    """Analyze the p-morphology correlation."""
    
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Filter valid fits
    valid = results_df[
        (results_df['p_fit'] > 0.1) &
        (results_df['p_fit'] < 3.0) &
        (results_df['p_err'] < 1.5) &
        (results_df['chi2_red'] < 50) &
        (results_df['morph_code'] >= 0)
    ].copy()
    
    print(f"\nValid fits: {len(valid)} / {len(results_df)}")
    
    if len(valid) < 10:
        print("WARNING: Too few valid fits for meaningful analysis")
        return valid, None
    
    # Overall statistics
    print(f"\n=== Overall p Distribution ===")
    print(f"Mean p: {valid['p_fit'].mean():.3f} ± {valid['p_fit'].std():.3f}")
    print(f"Median p: {valid['p_fit'].median():.3f}")
    print(f"Range: [{valid['p_fit'].min():.3f}, {valid['p_fit'].max():.3f}]")
    print(f"Global fit value: p = 0.757")
    
    # Group statistics
    print("\n=== p by Morphological Group ===")
    group_stats = valid.groupby('morph_group').agg({
        'p_fit': ['mean', 'std', 'median', 'count'],
        'chi2_red': 'mean'
    }).round(3)
    print(group_stats)
    
    # Correlation tests
    print("\n=== Correlation Tests ===")
    
    # Spearman (rank correlation - more robust)
    rho_spearman, p_spearman = spearmanr(valid['morph_code'], valid['p_fit'])
    print(f"Spearman ρ = {rho_spearman:.3f}, p-value = {p_spearman:.4f}")
    
    # Pearson (linear correlation)
    rho_pearson, p_pearson = pearsonr(valid['morph_code'], valid['p_fit'])
    print(f"Pearson r = {rho_pearson:.3f}, p-value = {p_pearson:.4f}")
    
    # Test the ordering prediction
    print("\n=== Testing p_Early > p_Intermediate > p_Late > p_Irregular ===")
    
    groups_ordered = ['Early (S0-Sa)', 'Early-Spiral (Sab-Sb)', 
                      'Intermediate (Sbc-Sc)', 'Late-Spiral (Scd-Sdm)', 
                      'Irregular (Sm-BCD)']
    
    means = {}
    for g in groups_ordered:
        subset = valid[valid['morph_group'] == g]
        if len(subset) > 0:
            means[g] = (subset['p_fit'].mean(), subset['p_fit'].std(), len(subset))
            print(f"  {g}: p = {means[g][0]:.3f} ± {means[g][1]:.3f} (n={means[g][2]})")
    
    # Check monotonicity
    ordered_means = [means[g][0] for g in groups_ordered if g in means]
    
    if len(ordered_means) >= 3:
        # Count monotonic pairs
        n_monotonic = sum(1 for i in range(len(ordered_means)-1) 
                        if ordered_means[i] >= ordered_means[i+1])
        total_pairs = len(ordered_means) - 1
        monotonicity_score = n_monotonic / total_pairs
        
        print(f"\nMonotonicity score: {monotonicity_score:.2f} ({n_monotonic}/{total_pairs} pairs)")
        
        if monotonicity_score >= 0.75:
            print("✓ PREDICTION SUPPORTED: p decreases with later morphology")
        elif monotonicity_score >= 0.5:
            print("⚠ PARTIAL SUPPORT: Some trend visible")
        else:
            print("✗ PREDICTION NOT SUPPORTED: No clear trend")
    
    # Interpretation
    print("\n=== Physical Interpretation ===")
    mean_p = valid['p_fit'].mean()
    
    if mean_p < 1:
        print(f"Mean p = {mean_p:.3f} < 1 indicates sub-linear decoherence accumulation")
        print("→ Consistent with sparse, fractal interaction network")
        print("→ Gravitational 'measurements' happen at discrete mass concentrations")
    elif mean_p < 2:
        print(f"Mean p = {mean_p:.3f} indicates between linear and area-like accumulation")
        print("→ Mixed interaction structure")
    else:
        print(f"Mean p = {mean_p:.3f} > 2 indicates super-quadratic accumulation")
        print("→ Highly connected, dense interaction network")
    
    # Save analysis results
    analysis = {
        'n_valid': int(len(valid)),
        'n_total': int(len(results_df)),
        'mean_p': float(valid['p_fit'].mean()),
        'std_p': float(valid['p_fit'].std()),
        'spearman_rho': float(rho_spearman),
        'spearman_p': float(p_spearman),
        'pearson_r': float(rho_pearson),
        'pearson_p': float(p_pearson),
        'group_means': {g: {'mean': float(m[0]), 'std': float(m[1]), 'n': int(m[2])} 
                       for g, m in means.items()},
        'prediction_supported': bool(rho_spearman > 0 and p_spearman < 0.05)
    }
    
    with open(output_dir / "p_morphology_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    return valid, analysis


def create_p_morphology_plots(valid_df, analysis, output_dir):
    """Create visualization of p vs morphology."""
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Color scheme by morphology
    colors = {
        'Early (S0-Sa)': 'red',
        'Early-Spiral (Sab-Sb)': 'orange',
        'Intermediate (Sbc-Sc)': 'gold',
        'Late-Spiral (Scd-Sdm)': 'green',
        'Irregular (Sm-BCD)': 'blue'
    }
    
    # Panel 1: Box plot by morphology
    ax1 = axes[0, 0]
    groups_ordered = ['Early (S0-Sa)', 'Early-Spiral (Sab-Sb)', 
                      'Intermediate (Sbc-Sc)', 'Late-Spiral (Scd-Sdm)', 
                      'Irregular (Sm-BCD)']
    
    group_data = []
    group_labels = []
    group_colors = []
    for g in groups_ordered:
        subset = valid_df[valid_df['morph_group'] == g]['p_fit'].values
        if len(subset) > 0:
            group_data.append(subset)
            group_labels.append(g.split('(')[0].strip())
            group_colors.append(colors[g])
    
    bp = ax1.boxplot(group_data, labels=group_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.axhline(0.757, color='purple', linestyle='--', linewidth=2, 
                label='Global fit (p=0.757)')
    ax1.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='p=1 (linear)')
    ax1.axhline(2.0, color='gray', linestyle=':', alpha=0.5, label='p=2 (area-like)')
    
    ax1.set_ylabel('Fitted p (decoherence exponent)', fontsize=12)
    ax1.set_xlabel('Morphological Type', fontsize=12)
    ax1.set_title('Decoherence Exponent by Galaxy Morphology', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.tick_params(axis='x', rotation=15)
    
    # Panel 2: Scatter plot with error bars
    ax2 = axes[0, 1]
    for morph_group in valid_df['morph_group'].unique():
        subset = valid_df[valid_df['morph_group'] == morph_group]
        jitter = np.random.normal(0, 0.1, len(subset))
        ax2.errorbar(subset['morph_code'] + jitter, subset['p_fit'], 
                    yerr=subset['p_err'].clip(upper=1.0),
                    fmt='o', label=morph_group, alpha=0.6, markersize=5,
                    color=colors.get(morph_group, 'gray'))
    
    ax2.axhline(0.757, color='purple', linestyle='--', linewidth=2)
    ax2.set_xlabel('Morphology Code (Early → Irregular)', fontsize=12)
    ax2.set_ylabel('Fitted p', fontsize=12)
    ax2.set_title('Individual Galaxy Fits', fontsize=14)
    ax2.legend(loc='upper right', fontsize=8)
    
    # Add trend line
    z = np.polyfit(valid_df['morph_code'], valid_df['p_fit'], 1)
    trend_x = np.array([0, 4])
    trend_y = z[0] * trend_x + z[1]
    ax2.plot(trend_x, trend_y, 'k-', alpha=0.3, linewidth=2, label='Linear trend')
    
    # Panel 3: Histogram of p values
    ax3 = axes[1, 0]
    ax3.hist(valid_df['p_fit'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(0.757, color='purple', linestyle='--', linewidth=2, 
                label=f'Global fit (p=0.757)')
    ax3.axvline(valid_df['p_fit'].mean(), color='red', linestyle='-', linewidth=2,
                label=f'Mean (p={valid_df["p_fit"].mean():.3f})')
    ax3.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
    ax3.axvline(2.0, color='gray', linestyle=':', alpha=0.7)
    
    ax3.set_xlabel('Fitted p', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Distribution of Fitted p Values', fontsize=14)
    ax3.legend(loc='upper right')
    
    # Panel 4: Group means with error bars
    ax4 = axes[1, 1]
    
    group_means = []
    group_stds = []
    group_ns = []
    group_labels_plot = []
    
    for g in groups_ordered:
        subset = valid_df[valid_df['morph_group'] == g]
        if len(subset) > 0:
            group_means.append(subset['p_fit'].mean())
            group_stds.append(subset['p_fit'].std() / np.sqrt(len(subset)))  # SEM
            group_ns.append(len(subset))
            group_labels_plot.append(g.split('(')[0].strip())
    
    x_pos = np.arange(len(group_means))
    bar_colors = [colors.get(g, 'gray') for g in groups_ordered[:len(group_means)]]
    bars = ax4.bar(x_pos, group_means, yerr=group_stds, 
                   color=bar_colors,
                   alpha=0.7, capsize=5, edgecolor='black')
    
    # Color bars
    for i, bar in enumerate(bars):
        bar.set_facecolor(list(colors.values())[i] if i < len(colors) else 'gray')
    
    ax4.axhline(0.757, color='purple', linestyle='--', linewidth=2)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(group_labels_plot, rotation=15)
    ax4.set_ylabel('Mean p ± SEM', fontsize=12)
    ax4.set_xlabel('Morphological Type', fontsize=12)
    ax4.set_title('Group Mean p Values', fontsize=14)
    
    # Add sample sizes
    for i, (x, y, n) in enumerate(zip(x_pos, group_means, group_ns)):
        ax4.annotate(f'n={n}', (x, y + group_stds[i] + 0.05), 
                    ha='center', fontsize=9)
    
    # Add prediction arrow
    if len(group_means) >= 2:
        ax4.annotate('', xy=(len(group_means)-0.5, min(group_means)-0.1),
                    xytext=(0.5, max(group_means)+0.15),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.5))
        ax4.text(len(group_means)/2, max(group_means)+0.2, 
                'Predicted trend', ha='center', color='green', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / "p_morphology_correlation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Also save to figures folder
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / "p_morphology_correlation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Plot also saved to: {fig_path}")
    
    plt.close()
    
    return fig


def main():
    """Run the complete p-morphology correlation test."""
    
    print("=" * 80)
    print("p-MORPHOLOGY CORRELATION TEST FOR Σ-GRAVITY")
    print("Testing the interaction network interpretation of decoherence")
    print("=" * 80)
    
    # Run fits
    results_df, output_dir = run_p_morphology_test(fit_mode='p_only')
    
    # Analyze correlation
    valid_df, analysis = analyze_p_morphology_correlation(results_df, output_dir)
    
    if analysis is None:
        print("\nAnalysis could not be completed - insufficient data")
        return
    
    # Create plots
    create_p_morphology_plots(valid_df, analysis, output_dir)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"""
Physical Prediction:
  p = interaction dimension d_I
  Smooth systems → p ≈ 2 (area-like interactions)
  Clumpy systems → p < 1 (fractal/sparse interactions)
  
Expected: p(Early) > p(Late) > p(Irregular)

Results:
  Mean fitted p: {analysis['mean_p']:.3f} ± {analysis['std_p']:.3f}
  Global fit p: 0.757
  
Correlation with morphology:
  Spearman ρ = {analysis['spearman_rho']:.3f} (p = {analysis['spearman_p']:.4f})
  Pearson r = {analysis['pearson_r']:.3f} (p = {analysis['pearson_p']:.4f})
""")
    
    if analysis['spearman_rho'] > 0 and analysis['spearman_p'] < 0.05:
        print("✓ SIGNIFICANT POSITIVE CORRELATION FOUND")
        print("  → Supports interaction network interpretation")
        print("  → p reflects structural complexity of mass distribution")
    elif analysis['spearman_rho'] > 0 and analysis['spearman_p'] < 0.1:
        print("⚠ MARGINAL POSITIVE CORRELATION")
        print("  → Suggestive but not conclusive")
        print("  → Larger sample may clarify")
    elif abs(analysis['spearman_rho']) < 0.1:
        print("✗ NO SIGNIFICANT CORRELATION")
        print("  → p appears universal across morphologies")
        print("  → May indicate more fundamental quantum gravity origin")
    else:
        print("? UNEXPECTED NEGATIVE CORRELATION")
        print("  → Requires theoretical reinterpretation")
    
    print(f"\nFull results saved to: {output_dir}")
    
    return analysis


if __name__ == "__main__":
    main()
