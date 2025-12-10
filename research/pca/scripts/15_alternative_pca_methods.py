#!/usr/bin/env python3
"""
Test alternative PCA methodologies to check robustness and gain new insights.

Tests:
1. Unweighted PCA (vs current weighted)
2. Inner/Outer/Transition region PCA (spatial localization)
3. Acceleration-space PCA (g(R) instead of V(R))
4. Model-residual PCA (what Σ-Gravity misses)
5. Mass-stratified PCA (dwarfs vs giants)

Each test computes principal angles vs baseline to assess similarity.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def principal_angles(A, B):
    """Compute principal angles between subspaces"""
    Qa, _ = np.linalg.qr(A[:, :3], mode='reduced')
    Qb, _ = np.linalg.qr(B[:, :3], mode='reduced')
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0, 1)
    return np.arccos(s)

def run_pca_variant(X, weights=None, name="Variant"):
    """Run PCA with optional weighting"""
    # Handle NaNs
    X_filled = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        col_mean = np.nanmean(col)
        X_filled[:, j] = np.where(np.isfinite(col), col, col_mean)
    
    if weights is not None:
        # Weighted standardization
        W = weights
        Wsum = np.sum(W, axis=0) + 1e-12
        mu = np.sum(W * X_filled, axis=0) / Wsum
        var = np.sum(W * (X_filled - mu)**2, axis=0) / Wsum
        sd = np.sqrt(np.maximum(var, 1e-12))
        X_std = (X_filled - mu) / sd
        
        # Sample weights
        w = np.median(W, axis=1)
        w = w / (np.mean(w) + 1e-12)
        
        # Weighted PCA via sqrt(w) scaling
        sw = np.sqrt(w)[:, None]
        X_weighted = X_std * sw
        
        pca = sklearnPCA(n_components=10)
        scores = pca.fit_transform(X_weighted)
        components = pca.components_
        evr = pca.explained_variance_ratio_
    else:
        # Standard unweighted PCA
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_filled)
        
        pca = sklearnPCA(n_components=10)
        scores = pca.fit_transform(X_std)
        components = pca.components_
        evr = pca.explained_variance_ratio_
    
    print(f"\n{name}:")
    print(f"  PC1-3 explain: {evr[:3].sum()*100:.1f}% variance")
    print(f"  PC1: {evr[0]*100:.1f}%, PC2: {evr[1]*100:.1f}%, PC3: {evr[2]*100:.1f}%")
    
    return scores, components, evr

def main():
    print("=" * 70)
    print("ALTERNATIVE PCA METHODOLOGIES TEST")
    print("=" * 70)
    
    # Load data
    repo_root = Path(__file__).parent.parent.parent
    curve_data = np.load(repo_root / 'pca' / 'data' / 'processed' / 'sparc_curvematrix.npz', allow_pickle=True)
    pca_baseline = np.load(repo_root / 'pca' / 'outputs' / 'pca_results_curve_only.npz', allow_pickle=True)
    
    curve_mat = curve_data['curve_mat']  # [N, K]
    weight_mat = curve_data['weight_mat']
    x_grid = curve_data['x_grid']
    names = curve_data['names']
    
    scores_baseline = pca_baseline['scores']
    
    print(f"\nData: {curve_mat.shape[0]} galaxies x {curve_mat.shape[1]} radial points")
    print(f"Radial grid: {x_grid.min():.2f} - {x_grid.max():.2f} (R/Rd)")
    
    # Load metadata
    meta = pd.read_csv(repo_root / 'pca' / 'data' / 'raw' / 'metadata' / 'sparc_meta.csv')
    meta_dict = {row['name']: row for _, row in meta.iterrows()}
    
    results = {}
    
    # ==================================================================
    # TEST 1: UNWEIGHTED PCA
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 1: UNWEIGHTED PCA (vs weighted baseline)")
    print("=" * 70)
    
    scores_unweighted, comps_unweighted, evr_unweighted = run_pca_variant(
        curve_mat, weights=None, name="Unweighted PCA"
    )
    
    angles_unweighted = principal_angles(scores_baseline, scores_unweighted)
    print(f"\nPrincipal angles vs baseline: {np.degrees(angles_unweighted)}")
    
    if angles_unweighted[0] < np.radians(10):
        print("  -> PC1 is ROBUST to weighting choice (angle < 10 deg)")
    else:
        print(f"  -> PC1 differs significantly (angle = {np.degrees(angles_unweighted[0]):.1f} deg)")
    
    results['unweighted'] = {
        'scores': scores_unweighted,
        'evr': evr_unweighted,
        'angles': angles_unweighted
    }
    
    # ==================================================================
    # TEST 2: RADIAL REGION PCA
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 2: RADIAL REGION PCA (inner/outer/transition)")
    print("=" * 70)
    
    # Inner region: R/Rd < 1.5
    inner_mask = x_grid < 1.5
    scores_inner, _, evr_inner = run_pca_variant(
        curve_mat[:, inner_mask], 
        weights=weight_mat[:, inner_mask] if weight_mat is not None else None,
        name="Inner Region (R/Rd < 1.5)"
    )
    
    # Transition: 1.5 < R/Rd < 3
    trans_mask = (x_grid >= 1.5) & (x_grid <= 3.0)
    scores_trans, _, evr_trans = run_pca_variant(
        curve_mat[:, trans_mask],
        weights=weight_mat[:, trans_mask] if weight_mat is not None else None,
        name="Transition (1.5 < R/Rd < 3)"
    )
    
    # Outer: R/Rd > 3
    outer_mask = x_grid > 3.0
    scores_outer, _, evr_outer = run_pca_variant(
        curve_mat[:, outer_mask],
        weights=weight_mat[:, outer_mask] if weight_mat is not None else None,
        name="Outer Region (R/Rd > 3)"
    )
    
    print("\nRegion comparison:")
    print(f"  Inner PC1 explains: {evr_inner[0]*100:.1f}%")
    print(f"  Transition PC1 explains: {evr_trans[0]*100:.1f}%")
    print(f"  Outer PC1 explains: {evr_outer[0]*100:.1f}%")
    
    # ==================================================================
    # TEST 3: ACCELERATION-SPACE PCA
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 3: ACCELERATION-SPACE PCA (g instead of V)")
    print("=" * 70)
    
    # Compute g = V²/R for each point
    g_mat = np.zeros_like(curve_mat)
    for i in range(curve_mat.shape[0]):
        V = curve_mat[i, :]
        R = x_grid * meta_dict.get(names[i], {'Rd': 2.0})['Rd']  # Convert back to kpc
        g_mat[i, :] = V**2 / np.maximum(R, 0.1)
    
    scores_accel, _, evr_accel = run_pca_variant(
        g_mat,
        weights=weight_mat,
        name="Acceleration Space"
    )
    
    angles_accel = principal_angles(scores_baseline, scores_accel)
    print(f"\nPrincipal angles vs baseline: {np.degrees(angles_accel)}")
    
    # ==================================================================
    # TEST 4: MASS-STRATIFIED PCA
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 4: MASS-STRATIFIED PCA (dwarfs vs giants)")
    print("=" * 70)
    
    # Get Mbar for each galaxy
    Mbar_arr = np.array([meta_dict.get(name, {'Mbar': np.nan})['Mbar'] for name in names])
    Mbar_median = np.nanmedian(Mbar_arr)
    
    dwarf_mask = (np.isfinite(Mbar_arr)) & (Mbar_arr < Mbar_median)
    giant_mask = (np.isfinite(Mbar_arr)) & (Mbar_arr >= Mbar_median)
    
    print(f"\nDwarfs: {dwarf_mask.sum()} galaxies (Mbar < {Mbar_median:.1f} x 10^9)")
    print(f"Giants: {giant_mask.sum()} galaxies (Mbar >= {Mbar_median:.1f} x 10^9)")
    
    if dwarf_mask.sum() > 10 and giant_mask.sum() > 10:
        scores_dwarf, _, evr_dwarf = run_pca_variant(
            curve_mat[dwarf_mask, :],
            weights=weight_mat[dwarf_mask, :] if weight_mat is not None else None,
            name="Dwarfs Only"
        )
        
        scores_giant, _, evr_giant = run_pca_variant(
            curve_mat[giant_mask, :],
            weights=weight_mat[giant_mask, :] if weight_mat is not None else None,
            name="Giants Only"
        )
        
        # Compare dwarf vs giant PC subspaces
        angles_mass = principal_angles(scores_dwarf, scores_giant)
        print(f"\nDwarf vs Giant principal angles: {np.degrees(angles_mass)}")
        
        if angles_mass[0] < np.radians(15):
            print("  → Dwarfs and giants share similar PC1 structure")
        else:
            print(f"  → Dwarfs and giants have DIFFERENT PC1 (angle = {np.degrees(angles_mass[0]):.1f}°)")
            print("     This confirms mass-dependent shape physics!")
    
    # ==================================================================
    # TEST 5: MODEL RESIDUAL PCA
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 5: PCA ON MODEL RESIDUALS (what Σ-Gravity misses)")
    print("=" * 70)
    
    # Load best model (local density)
    fits_file = repo_root / 'pca' / 'outputs' / 'sigmagravity_fits' / 'sparc_sigmagravity_local_density_fits.csv'
    
    if fits_file.exists():
        # For simplicity, use the fact that residuals correlate with curves
        # True residual PCA would need per-point residuals, not per-galaxy RMS
        print("\nNote: Full residual PCA requires per-point model predictions")
        print("      (not implemented - would need curve-level predictions)")
        print("      Current: Testing if residual magnitudes create structure")
        
        fits = pd.read_csv(fits_file)
        
        # Match to PCA galaxies
        residual_scores = []
        for name in names:
            fit_row = fits[fits['name'] == name]
            if len(fit_row) > 0:
                residual_scores.append(fit_row.iloc[0]['residual_rms'])
            else:
                residual_scores.append(np.nan)
        
        residual_arr = np.array(residual_scores)
        
        # Check correlation with PCs
        print(f"\nResidual RMS vs baseline PCs:")
        for i in range(3):
            mask = np.isfinite(residual_arr)
            rho, p = spearmanr(residual_arr[mask], scores_baseline[mask, i])
            print(f"  vs PC{i+1}: rho = {rho:+.3f}, p = {p:.3e}")
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: ROBUSTNESS ACROSS METHODS")
    print("=" * 70)
    
    print("\nKey findings:")
    print(f"  1. Weighted vs unweighted: PC1 angle = {np.degrees(angles_unweighted[0]):.1f}°")
    if angles_unweighted[0] < np.radians(10):
        print("     → Findings ROBUST to weighting")
    
    print(f"\n  2. Radial regions:")
    print(f"     Inner PC1: {evr_inner[0]*100:.1f}% variance")
    print(f"     Transition PC1: {evr_trans[0]*100:.1f}% variance")
    print(f"     Outer PC1: {evr_outer[0]*100:.1f}% variance")
    
    if evr_outer[0] > 0.7:
        print("     → Outer region dominates variance")
    elif evr_inner[0] > 0.7:
        print("     → Inner region dominates variance")
    else:
        print("     → Variance distributed across radial range")
    
    print(f"\n  3. Acceleration space: PC1 angle = {np.degrees(angles_accel[0]):.1f}°")
    if angles_accel[0] < np.radians(15):
        print("     → V(R) and g(R) give similar structure")
    
    if dwarf_mask.sum() > 10 and giant_mask.sum() > 10:
        print(f"\n  4. Mass stratification: Dwarf-Giant angle = {np.degrees(angles_mass[0]):.1f}°")
        if angles_mass[0] > np.radians(20):
            print("     → CONFIRMS mass-dependent PC structure!")
            print("        Dwarfs and giants have fundamentally different dominant modes")
    
    # Save results
    output_dir = repo_root / 'pca' / 'outputs' / 'alternative_methods'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_dir / 'methodology_comparison.txt'
    with open(summary_file, 'w') as f:
        f.write("ALTERNATIVE PCA METHODOLOGIES COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Test 1: Unweighted PCA\n")
        f.write(f"  PC1-3 variance: {evr_unweighted[:3].sum()*100:.1f}%\n")
        f.write(f"  PC1 angle vs baseline: {np.degrees(angles_unweighted[0]):.2f}°\n")
        f.write(f"  Status: {'ROBUST' if angles_unweighted[0] < np.radians(10) else 'DIFFERENT'}\n")
        
        f.write("\nTest 2: Radial Regions\n")
        f.write(f"  Inner (<1.5 Rd): PC1 = {evr_inner[0]*100:.1f}%\n")
        f.write(f"  Transition (1.5-3 Rd): PC1 = {evr_trans[0]*100:.1f}%\n")
        f.write(f"  Outer (>3 Rd): PC1 = {evr_outer[0]*100:.1f}%\n")
        
        f.write("\nTest 3: Acceleration Space\n")
        f.write(f"  PC1-3 variance: {evr_accel[:3].sum()*100:.1f}%\n")
        f.write(f"  PC1 angle vs baseline: {np.degrees(angles_accel[0]):.2f}°\n")
        
        if dwarf_mask.sum() > 10 and giant_mask.sum() > 10:
            f.write("\nTest 4: Mass Stratification\n")
            f.write(f"  Dwarf PC1: {evr_dwarf[0]*100:.1f}%\n")
            f.write(f"  Giant PC1: {evr_giant[0]*100:.1f}%\n")
            f.write(f"  Dwarf-Giant angle: {np.degrees(angles_mass[0]):.2f}°\n")
            if angles_mass[0] > np.radians(20):
                f.write("  Finding: Mass-dependent structure CONFIRMED\n")
    
    print(f"\nResults saved to: {summary_file}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Variance comparison
    ax = axes[0, 0]
    methods = ['Baseline\n(weighted)', 'Unweighted', 'Inner\nRegion', 'Outer\nRegion', 'Acceleration']
    pc1_vars = [pca_baseline['evr'][0], evr_unweighted[0], evr_inner[0], evr_outer[0], evr_accel[0]]
    
    ax.bar(methods, np.array(pc1_vars) * 100, color=['blue', 'orange', 'green', 'red', 'purple'], alpha=0.7)
    ax.set_ylabel('PC1 Variance Explained (%)', fontsize=11)
    ax.set_title('PC1 Variance Across Methods', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Plot 2: Principal angles
    ax = axes[0, 1]
    test_names = ['Unweighted', 'Acceleration']
    angles_list = [angles_unweighted, angles_accel]
    
    x_pos = np.arange(len(test_names))
    width = 0.25
    
    ax.bar(x_pos - width, [np.degrees(a[0]) for a in angles_list], width, label='PC1', alpha=0.8)
    ax.bar(x_pos, [np.degrees(a[1]) for a in angles_list], width, label='PC2', alpha=0.8)
    ax.bar(x_pos + width, [np.degrees(a[2]) for a in angles_list], width, label='PC3', alpha=0.8)
    
    ax.axhline(10, color='r', ls='--', lw=1, alpha=0.5, label='10° threshold')
    ax.axhline(20, color='orange', ls='--', lw=1, alpha=0.5, label='20° threshold')
    
    ax.set_ylabel('Principal Angle (degrees)', fontsize=11)
    ax.set_title('Subspace Similarity to Baseline', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticks_labels(test_names)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Radial variance profile
    ax = axes[1, 0]
    ax.plot(x_grid, np.var(curve_mat, axis=0), 'o-', lw=2, ms=4, label='Variance across galaxies')
    ax.set_xlabel('R / Rd', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Where Does Variance Come From?', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Mass stratification (if available)
    ax = axes[1, 1]
    if dwarf_mask.sum() > 10 and giant_mask.sum() > 10:
        # Plot mean curves
        ax.plot(x_grid, np.nanmean(curve_mat[dwarf_mask, :], axis=0), 
                'b-', lw=2, label=f'Dwarfs (n={dwarf_mask.sum()})')
        ax.fill_between(x_grid,
                        np.nanmean(curve_mat[dwarf_mask, :], axis=0) - np.nanstd(curve_mat[dwarf_mask, :], axis=0),
                        np.nanmean(curve_mat[dwarf_mask, :], axis=0) + np.nanstd(curve_mat[dwarf_mask, :], axis=0),
                        alpha=0.2, color='b')
        
        ax.plot(x_grid, np.nanmean(curve_mat[giant_mask, :], axis=0),
                'r-', lw=2, label=f'Giants (n={giant_mask.sum()})')
        ax.fill_between(x_grid,
                        np.nanmean(curve_mat[giant_mask, :], axis=0) - np.nanstd(curve_mat[giant_mask, :], axis=0),
                        np.nanmean(curve_mat[giant_mask, :], axis=0) + np.nanstd(curve_mat[giant_mask, :], axis=0),
                        alpha=0.2, color='r')
        
        ax.set_xlabel('R / Rd', fontsize=11)
        ax.set_ylabel('V / Vf (normalized)', fontsize=11)
        ax.set_title(f'Dwarfs vs Giants (Angle={np.degrees(angles_mass[0]):.1f}°)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Mass stratification\nrequires more data', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    fig_file = output_dir / 'methodology_comparison.png'
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {fig_file}")
    
    print("\n" + "=" * 70)
    print("ALTERNATIVE METHODS TESTING COMPLETE")
    print("=" * 70)
    print(f"\nSee: {summary_file}")
    print(f"     {fig_file}")

if __name__ == '__main__':
    main()

