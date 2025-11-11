#!/usr/bin/env python3
"""
Compare Sigma-Gravity model results against PCA structure.

Tests:
1. Are residuals uncorrelated with PC1? (Model captures dominant mode)
2. Do parameters align with empirical axes? (Physical grounding)
3. Bootstrap confidence intervals for robustness

Generates:
- Correlation table (residuals vs PCs)
- Scatter plots (residual vs PC1, parameters in PC space)
- Summary statistics
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

def bootstrap_correlation(x, y, n_boot=1000, seed=42):
    """Bootstrap confidence interval for Spearman correlation"""
    np.random.seed(seed)
    n = len(x)
    boot_rhos = []
    
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        rho, _ = spearmanr(x[idx], y[idx])
        boot_rhos.append(rho)
    
    boot_rhos = np.array(boot_rhos)
    ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
    
    return ci_low, ci_high

def main():
    parser = argparse.ArgumentParser(description='Compare Sigma-Gravity to PCA')
    parser.add_argument('--pca_npz', default='pca/outputs/pca_results_curve_only.npz',
                       help='PCA results file')
    parser.add_argument('--model_csv', default='pca/outputs/sigmagravity_fits/sparc_sigmagravity_fits.csv',
                       help='Sigma-Gravity fit results CSV')
    parser.add_argument('--out_dir', default='pca/outputs/model_comparison',
                       help='Output directory for plots and tables')
    args = parser.parse_args()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SIGMA-GRAVITY vs PCA COMPARISON")
    print("=" * 70)
    
    # Load PCA results
    print("\nLoading PCA results...")
    pca = np.load(args.pca_npz, allow_pickle=True)
    names = pca['names']
    scores = pca['scores']  # [N, n_components]
    evr = pca['evr']  # explained variance ratio
    
    print(f"  PCA: {len(names)} galaxies, {scores.shape[1]} components")
    print(f"  PC1-3 explain: {evr[:3].sum()*100:.1f}% of variance")
    
    # Load model results
    print(f"\nLoading Sigma-Gravity results...")
    model = pd.read_csv(args.model_csv)
    print(f"  Model: {len(model)} galaxies")
    
    # Create PC dataframe
    pc_df = pd.DataFrame({
        'name': names,
        'PC1': scores[:, 0],
        'PC2': scores[:, 1],
        'PC3': scores[:, 2]
    })
    
    # Merge
    merged = model.merge(pc_df, on='name', how='inner')
    print(f"  Matched: {len(merged)} galaxies")
    
    if len(merged) < 10:
        print("\nError: Too few matched galaxies for analysis!")
        return
    
    # ==================================================================
    # TEST A: RESIDUAL ALIGNMENT (CRITICAL TEST)
    # ==================================================================
    
    print("\n" + "=" * 70)
    print("TEST A: RESIDUAL vs PC CORRELATIONS (Critical Test)")
    print("=" * 70)
    
    print("\n{:<10s} {:<10s} {:<10s} {:<12s} {:<20s}".format(
        "Variable", "Pearson", "p-value", "Spearman", "95% CI"))
    print("-" * 70)
    
    residual_tests = {}
    
    for i in range(3):
        pc = f'PC{i+1}'
        
        # Filter valid data
        mask = np.isfinite(merged['residual_rms']) & np.isfinite(merged[pc])
        x = merged.loc[mask, 'residual_rms'].values
        y = merged.loc[mask, pc].values
        
        if len(x) < 10:
            continue
        
        # Pearson
        r_pearson, p_pearson = pearsonr(x, y)
        
        # Spearman
        rho_spearman, p_spearman = spearmanr(x, y)
        
        # Bootstrap CI
        ci_low, ci_high = bootstrap_correlation(x, y, n_boot=1000)
        
        residual_tests[pc] = {
            'pearson': r_pearson,
            'p_pearson': p_pearson,
            'spearman': rho_spearman,
            'p_spearman': p_spearman,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n': len(x)
        }
        
        print(f"residual_rms vs {pc:4s}: "
              f"{r_pearson:+.3f}  {p_pearson:.3e}  "
              f"{rho_spearman:+.3f}  [{ci_low:+.3f}, {ci_high:+.3f}]")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("Interpretation:")
    
    pc1_rho = residual_tests['PC1']['spearman']
    pc1_p = residual_tests['PC1']['p_spearman']
    
    if abs(pc1_rho) < 0.2 and pc1_p > 0.05:
        print("  [PASS] Residuals uncorrelated with PC1 (dominant 79.9% mode)")
        print(f"         |rho| = {abs(pc1_rho):.3f} < 0.2, p = {pc1_p:.3f} > 0.05")
        print("         -> Model successfully captures dominant empirical structure")
    else:
        print(f"  [FAIL] Residuals correlate with PC1: rho = {pc1_rho:+.3f}, p = {pc1_p:.3e}")
        print("         -> Model misses dominant mass-velocity mode")
    
    # Check PC2/PC3
    for i in [2, 3]:
        pc = f'PC{i}'
        if pc not in residual_tests:
            continue
        rho = residual_tests[pc]['spearman']
        p_val = residual_tests[pc]['p_spearman']
        
        if abs(rho) > 0.3 and p_val < 0.01:
            mode_name = "scale-length" if i == 2 else "density"
            print(f"  [WARNING] Significant {pc} correlation (rho = {rho:+.3f})")
            print(f"            -> Model may need {mode_name}-dependent refinement")
    
    # ==================================================================
    # TEST B: PARAMETER ALIGNMENT
    # ==================================================================
    
    print("\n" + "=" * 70)
    print("TEST B: PARAMETER vs PC CORRELATIONS")
    print("=" * 70)
    
    # Parameters to test (if they vary - currently fixed but structure ready)
    param_cols = ['A', 'l0']  # Fixed in current fits, but structure ready
    
    print("\n{:<15s} {:<8s} {:<12s} {:<10s}".format(
        "Parameter vs PC", "Spearman", "p-value", "Interpret"))
    print("-" * 70)
    
    param_tests = {}
    
    for param in param_cols:
        if param not in merged.columns:
            continue
        
        # Check if parameter actually varies
        if merged[param].std() < 1e-6:
            print(f"{param:15s}: (fixed parameter, no variation to test)")
            continue
        
        param_tests[param] = {}
        
        for i in range(3):
            pc = f'PC{i+1}'
            
            mask = np.isfinite(merged[param]) & np.isfinite(merged[pc])
            x = merged.loc[mask, param].values
            y = merged.loc[mask, pc].values
            
            if len(x) < 10:
                continue
            
            rho, p_val = spearmanr(x, y)
            param_tests[param][pc] = {'rho': rho, 'p': p_val, 'n': len(x)}
            
            # Interpret
            interp = ""
            if abs(rho) > 0.5 and p_val < 0.01:
                if param == 'A' and i == 0:
                    interp = "(expected: amplitude vs mass-velocity)"
                elif param == 'l0' and i == 1:
                    interp = "(expected: scale vs disk size)"
                else:
                    interp = "(unexpected alignment)"
            
            print(f"{param} vs {pc:4s}:    {rho:+.3f}    {p_val:.3e}  {interp}")
    
    # Note about fixed parameters
    if not param_tests:
        print("\nNote: All parameters fixed in current fits (A=0.6, l0=5.0).")
        print("      This test becomes meaningful with per-galaxy parameter fitting.")
    
    # ==================================================================
    # VISUALIZATION
    # ==================================================================
    
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    # Figure 1: Residual vs PC1
    fig, ax = plt.subplots(figsize=(10, 7))
    
    mask = np.isfinite(merged['residual_rms']) & np.isfinite(merged['PC1'])
    x = merged.loc[mask, 'PC1'].values
    y = merged.loc[mask, 'residual_rms'].values
    
    ax.scatter(x, y, alpha=0.6, s=60, edgecolors='k', linewidths=0.5)
    ax.set_xlabel('PC1 Score (79.9% variance, mass-velocity mode)', fontsize=13)
    ax.set_ylabel('Sigma-Gravity RMS Residual (km/s)', fontsize=13)
    ax.set_title('Model Residuals vs Dominant Empirical Mode', fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add correlation text
    rho = residual_tests['PC1']['spearman']
    p_val = residual_tests['PC1']['p_spearman']
    ci_low = residual_tests['PC1']['ci_low']
    ci_high = residual_tests['PC1']['ci_high']
    
    textstr = f'Spearman ρ = {rho:+.3f}\n'
    textstr += f'p-value = {p_val:.3e}\n'
    textstr += f'95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]\n'
    textstr += f'n = {residual_tests["PC1"]["n"]}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Add pass/fail indicator
    if abs(rho) < 0.2 and p_val > 0.05:
        status_text = "PASS: Model captures PC1"
        status_color = 'green'
    else:
        status_text = "FAIL: Systematic trend with PC1"
        status_color = 'red'
    
    ax.text(0.95, 0.05, status_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3),
            fontweight='bold')
    
    plt.tight_layout()
    fig_file = out_dir / 'residual_vs_PC1.png'
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {fig_file}")
    plt.close()
    
    # Figure 2: PC1 vs PC2 scatter colored by residual
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = (np.isfinite(merged['PC1']) & np.isfinite(merged['PC2']) & 
            np.isfinite(merged['residual_rms']))
    x = merged.loc[mask, 'PC1'].values
    y = merged.loc[mask, 'PC2'].values
    c = merged.loc[mask, 'residual_rms'].values
    
    scatter = ax.scatter(x, y, c=c, cmap='viridis', s=80, alpha=0.8, 
                        edgecolors='k', linewidths=0.5)
    ax.set_xlabel('PC1 (79.9% variance, mass-velocity)', fontsize=13)
    ax.set_ylabel('PC2 (11.2% variance, scale-length)', fontsize=13)
    ax.set_title('Model Residuals in PCA Space', fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('RMS Residual (km/s)', fontsize=12)
    
    plt.tight_layout()
    fig_file = out_dir / 'residuals_in_PC_space.png'
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()
    
    # ==================================================================
    # SAVE SUMMARY
    # ==================================================================
    
    summary_file = out_dir / 'comparison_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("SIGMA-GRAVITY vs PCA COMPARISON SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("TEST A: Residual vs PC Correlations\n")
        f.write("-" * 70 + "\n")
        for pc, results in residual_tests.items():
            f.write(f"\n{pc}:\n")
            f.write(f"  Spearman rho = {results['spearman']:+.4f}\n")
            f.write(f"  p-value = {results['p_spearman']:.4e}\n")
            f.write(f"  95% CI: [{results['ci_low']:+.4f}, {results['ci_high']:+.4f}]\n")
            f.write(f"  n = {results['n']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERPRETATION:\n\n")
        
        pc1_rho = residual_tests['PC1']['spearman']
        pc1_p = residual_tests['PC1']['p_spearman']
        
        if abs(pc1_rho) < 0.2 and pc1_p > 0.05:
            f.write("RESULT: PASS\n")
            f.write(f"|rho(residual, PC1)| = {abs(pc1_rho):.3f} < 0.2\n")
            f.write(f"p-value = {pc1_p:.3f} > 0.05\n\n")
            f.write("CONCLUSION: Sigma-Gravity successfully captures the dominant\n")
            f.write("            empirical mode (79.9% of rotation curve variance).\n")
        else:
            f.write("RESULT: FAIL\n")
            f.write(f"rho(residual, PC1) = {pc1_rho:+.3f}\n")
            f.write(f"p-value = {pc1_p:.3e}\n\n")
            f.write("CONCLUSION: Residuals systematically correlate with PC1.\n")
            f.write("            Model does not capture the dominant mass-velocity mode.\n")
    
    print(f"\n  Saved: {summary_file}")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {out_dir}")
    
    # Final verdict
    print("\n" + "=" * 70)
    if abs(pc1_rho) < 0.2 and pc1_p > 0.05:
        print("VERDICT: Sigma-Gravity PASSES empirical structure test")
        print("         Model captures the dominant 79.9% mode")
    else:
        print("VERDICT: Sigma-Gravity FAILS empirical structure test")
        print("         Systematic residuals indicate missing physics")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()
