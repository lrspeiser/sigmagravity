#!/usr/bin/env python3
"""
Parameter-Space PCA: Analyze which Sigma-Gravity features drive performance.

This is complementary to curve-shape PCA:
- Curve PCA: What shape modes exist? (96.8% in 3 dims)
- Parameter PCA: What model features explain outcome variance?

Features per galaxy:
- Structural: Rd, Mbar, Sigma0, Vf, T-type, Inc
- Kernel: K at key radii (2, 5, 10 kpc)
- Baryonic fields: g_bar at key radii
- Outcomes: RAR bias/scatter, BTFR residual
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def coherence_burr_xii(R, l0=5.0, p=2.0, n_coh=1.5):
    """Burr-XII coherence"""
    x = (R / l0)**p
    return 1.0 - (1.0 + x)**(-n_coh)

def extract_features_for_galaxy(curve_data, meta_row, name):
    """Extract all parameter-space features for one galaxy"""
    R = curve_data['R_kpc'].values
    V_obs = curve_data['V_obs'].values
    eV_obs = curve_data['eV_obs'].values
    
    V_disk = curve_data.get('V_disk', np.zeros_like(R)).values
    V_gas = curve_data.get('V_gas', np.zeros_like(R)).values
    V_bul = curve_data.get('V_bul', np.zeros_like(R)).values if 'V_bul' in curve_data else np.zeros_like(R)
    V_bar = np.sqrt(V_disk**2 + V_gas**2 + V_bul**2)
    
    # Metadata
    Rd = meta_row.get('Rd', np.nan)
    Mbar = meta_row.get('Mbar', np.nan)  # In 10^9 Msun
    Sigma0 = meta_row.get('Sigma0', np.nan)
    Vf = meta_row.get('Vf', np.nan)
    T_type = meta_row.get('T', np.nan)
    Inc = meta_row.get('Inc', np.nan)
    
    # Compute g_bar at key radii
    g_bar_full = V_bar**2 / np.maximum(R, 0.1)  # (km/s)^2 / kpc
    
    # Helper to get value at radius
    def value_at_R(R_target, R_arr, V_arr):
        if R_target < R_arr.min() or R_target > R_arr.max():
            return np.nan
        return np.interp(R_target, R_arr, V_arr)
    
    gbar_2kpc = value_at_R(2.0, R, g_bar_full) if len(R) > 0 else np.nan
    gbar_5kpc = value_at_R(5.0, R, g_bar_full) if len(R) > 0 else np.nan
    gbar_10kpc = value_at_R(10.0, R, g_bar_full) if len(R) > 0 else np.nan
    
    # Compute K at key radii (using calibrated parameters)
    A_gal = 0.6
    l0_gal = 5.0
    p_gal = 2.0
    n_coh_gal = 1.5
    
    K_2kpc = A_gal * coherence_burr_xii(2.0, l0_gal, p_gal, n_coh_gal)
    K_5kpc = A_gal * coherence_burr_xii(5.0, l0_gal, p_gal, n_coh_gal)
    K_10kpc = A_gal * coherence_burr_xii(10.0, l0_gal, p_gal, n_coh_gal)
    
    # RAR metrics
    mask = (V_bar > 1.0) & (V_obs > 1.0) & np.isfinite(V_obs) & np.isfinite(V_bar)
    if mask.sum() >= 3:
        log_gobs = np.log10(V_obs[mask]**2)
        log_gbar = np.log10(V_bar[mask]**2)
        log_ratio = log_gobs - log_gbar
        rar_bias = np.mean(log_ratio)
        rar_scatter = np.std(log_ratio, ddof=1)
    else:
        rar_bias = np.nan
        rar_scatter = np.nan
    
    # BTFR residual
    if np.isfinite(Mbar) and np.isfinite(Vf) and Mbar > 0 and Vf > 0:
        log_Mbar = np.log10(Mbar * 1e9)  # M_sun
        log_Vf = np.log10(Vf)
        btfr_expected = 3.5 * log_Vf - 2.5
        btfr_residual = log_Mbar - btfr_expected
    else:
        btfr_residual = np.nan
    
    # Mean K in inner region (R < 3 kpc)
    inner_mask = R < 3.0
    if inner_mask.sum() > 0:
        K_inner_mean = np.mean([A_gal * coherence_burr_xii(r, l0_gal, p_gal, n_coh_gal) for r in R[inner_mask]])
    else:
        K_inner_mean = np.nan
    
    # Mean K in outer region (R > 5 kpc)
    outer_mask = R > 5.0
    if outer_mask.sum() > 0:
        K_outer_mean = np.mean([A_gal * coherence_burr_xii(r, l0_gal, p_gal, n_coh_gal) for r in R[outer_mask]])
    else:
        K_outer_mean = np.nan
    
    return {
        'galaxy_id': name,
        't_type': T_type,
        'incl_deg': Inc,
        'r_d_kpc': Rd,
        'mbar_1e9Msun': Mbar,
        'sigma0_Msun_pc2': Sigma0,
        'v_flat_kms': Vf,
        'log_mbar': np.log10(Mbar) if np.isfinite(Mbar) and Mbar > 0 else np.nan,
        'log_sigma0': np.log10(Sigma0) if np.isfinite(Sigma0) and Sigma0 > 0 else np.nan,
        'log_rd': np.log10(Rd) if np.isfinite(Rd) and Rd > 0 else np.nan,
        'log_vf': np.log10(Vf) if np.isfinite(Vf) and Vf > 0 else np.nan,
        'gbar_2kpc': gbar_2kpc,
        'gbar_5kpc': gbar_5kpc,
        'gbar_10kpc': gbar_10kpc,
        'K_2kpc': K_2kpc,
        'K_5kpc': K_5kpc,
        'K_10kpc': K_10kpc,
        'K_inner_mean': K_inner_mean,
        'K_outer_mean': K_outer_mean,
        'rar_bias_dex': rar_bias,
        'rar_scatter_dex': rar_scatter,
        'btfr_residual_dex': btfr_residual,
        'n_datapoints': len(R)
    }

def main():
    print("=" * 70)
    print("PARAMETER-SPACE PCA: Sigma-Gravity Feature Analysis")
    print("=" * 70)
    
    # Paths
    repo_root = Path(__file__).parent.parent.parent
    curves_dir = repo_root / 'pca' / 'data' / 'raw' / 'sparc_curves'
    meta_file = repo_root / 'pca' / 'data' / 'raw' / 'metadata' / 'sparc_meta.csv'
    output_dir = repo_root / 'pca' / 'outputs' / 'parameter_space'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    meta = pd.read_csv(meta_file)
    
    print(f"\nExtracting parameter-space features for {len(meta)} galaxies...")
    
    # Extract features
    features_list = []
    for _, row in meta.iterrows():
        name = row['name']
        curve_file = curves_dir / f"{name}.csv"
        
        if not curve_file.exists():
            continue
        
        try:
            curve = pd.read_csv(curve_file)
            features = extract_features_for_galaxy(curve, row, name)
            features_list.append(features)
        except:
            continue
    
    features_df = pd.DataFrame(features_list)
    
    print(f"Extracted features for {len(features_df)} galaxies")
    
    # Save
    features_file = output_dir / 'sparc_parameter_features.csv'
    features_df.to_csv(features_file, index=False)
    print(f"Saved to: {features_file}")
    
    # Select numeric features for PCA
    exclude_cols = ['galaxy_id']
    numeric_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    # Drop columns with too many NaNs
    valid_cols = []
    for col in numeric_cols:
        if features_df[col].notna().sum() > 0.7 * len(features_df):
            valid_cols.append(col)
    
    print(f"\nUsing {len(valid_cols)} features for PCA")
    
    # Prepare data
    X = features_df[valid_cols].copy()
    
    # Impute NaNs with column mean
    for col in X.columns:
        X[col] = X[col].fillna(X[col].mean())
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run PCA
    pca = PCA(n_components=min(10, len(valid_cols)))
    scores = pca.fit_transform(X_scaled)
    components = pca.components_
    evr = pca.explained_variance_ratio_
    
    print("\n" + "=" * 70)
    print("PARAMETER-SPACE PCA RESULTS")
    print("=" * 70)
    
    print(f"\nVariance Explained:")
    for i in range(min(5, len(evr))):
        cum_var = evr[:i+1].sum()
        print(f"  PC{i+1}: {evr[i]*100:.1f}% (cumulative: {cum_var*100:.1f}%)")
    
    # Interpret loadings
    print("\n" + "=" * 70)
    print("PC1 LOADINGS (Top features)")
    print("=" * 70)
    
    loadings_pc1 = components[0, :]
    loading_df = pd.DataFrame({
        'feature': valid_cols,
        'loading': loadings_pc1
    }).sort_values('loading', key=abs, ascending=False)
    
    print("\nTop 10 features (by |loading|):")
    print(loading_df.head(10).to_string(index=False))
    
    # Check correlations with outcomes
    print("\n" + "=" * 70)
    print("PC CORRELATIONS WITH OUTCOMES")
    print("=" * 70)
    
    outcome_cols = ['rar_bias_dex', 'rar_scatter_dex', 'btfr_residual_dex']
    
    for outcome in outcome_cols:
        if outcome in features_df.columns:
            print(f"\n{outcome}:")
            for i in range(min(3, scores.shape[1])):
                mask = np.isfinite(features_df[outcome])
                if mask.sum() > 10:
                    rho, p = spearmanr(scores[mask, i], features_df.loc[mask, outcome])
                    print(f"  PC{i+1}: rho = {rho:+.3f}, p = {p:.3e}")
    
    # Save results
    scores_df = pd.DataFrame(scores[:, :5], columns=[f'PC{i+1}' for i in range(min(5, scores.shape[1]))])
    scores_df.insert(0, 'galaxy_id', features_df['galaxy_id'])
    scores_df.to_csv(output_dir / 'parameter_pca_scores.csv', index=False)
    
    loadings_df = pd.DataFrame(components[:5, :].T, index=valid_cols, 
                                columns=[f'PC{i+1}' for i in range(min(5, components.shape[0]))])
    loadings_df.to_csv(output_dir / 'parameter_pca_loadings.csv')
    
    evr_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(evr))],
        'explained_variance_ratio': evr
    })
    evr_df.to_csv(output_dir / 'parameter_pca_explained_variance.csv', index=False)
    
    # Visualizations
    # 1. Scree plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(evr)+1), evr * 100, 'o-', lw=2, ms=8)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_title('Parameter-Space PCA: Scree Plot', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_pca_scree.png', dpi=150)
    plt.close()
    
    # 2. Biplot (PC1 vs PC2)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter of galaxies
    ax.scatter(scores[:, 0], scores[:, 1], alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    
    # Loading vectors (scaled)
    scale = 3.0
    for i, feat in enumerate(valid_cols):
        # Only show strongest loadings for clarity
        if abs(loadings_pc1[i]) > 0.15:  # Threshold
            ax.arrow(0, 0, components[0, i]*scale, components[1, i]*scale,
                    head_width=0.15, head_length=0.2, fc='red', ec='red', alpha=0.6)
            ax.text(components[0, i]*scale*1.1, components[1, i]*scale*1.1, 
                   feat, fontsize=8, ha='center')
    
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('Parameter-Space PCA: Biplot', fontsize=14, fontweight='bold')
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    ax.axvline(0, color='k', ls='--', alpha=0.3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_pca_biplot.png', dpi=150)
    plt.close()
    
    # 3. PC1 vs outcomes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    outcomes = [
        ('rar_scatter_dex', 'RAR Scatter (dex)'),
        ('btfr_residual_dex', 'BTFR Residual (dex)'),
        ('log_mbar', 'log(Mbar)')
    ]
    
    for ax, (col, label) in zip(axes, outcomes):
        if col in features_df.columns:
            mask = np.isfinite(features_df[col]) & np.isfinite(scores[:, 0])
            if mask.sum() > 10:
                ax.scatter(scores[mask, 0], features_df.loc[mask, col], alpha=0.6, s=50)
                
                # Correlation
                rho, p = spearmanr(scores[mask, 0], features_df.loc[mask, col])
                ax.text(0.05, 0.95, f'ρ = {rho:+.3f}\np = {p:.2e}',
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', fc='white', alpha=0.8))
                
                ax.set_xlabel('PC1 Score', fontsize=11)
                ax.set_ylabel(label, fontsize=11)
                ax.grid(alpha=0.3)
    
    plt.suptitle('Parameter-Space PC1 vs Outcomes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_pc1_vs_outcomes.png', dpi=150)
    plt.close()
    
    print(f"\nSaved results to: {output_dir}/")
    
    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    
    print("\nPC1 represents:")
    top_3_features = loading_df.head(3)
    for _, row in top_3_features.iterrows():
        sign = "positively" if row['loading'] > 0 else "negatively"
        print(f"  - {row['feature']:20s} ({sign}, |load| = {abs(row['loading']):.3f})")
    
    print("\nWhat this means:")
    print("  Galaxies with high PC1 scores have:")
    for _, row in top_3_features.iterrows():
        direction = "high" if row['loading'] > 0 else "low"
        print(f"    - {direction} {row['feature']}")
    
    # Compare to curve-shape PCA
    print("\n" + "=" * 70)
    print("COMPARISON TO CURVE-SHAPE PCA")
    print("=" * 70)
    
    # Load curve-shape PC scores
    pca_curve = np.load(repo_root / 'pca' / 'outputs' / 'pca_results_curve_only.npz', allow_pickle=True)
    names_curve = pca_curve['names']
    scores_curve = pca_curve['scores']
    
    # Merge
    curve_pc_df = pd.DataFrame({
        'galaxy_id': names_curve,
        'Curve_PC1': scores_curve[:, 0],
        'Curve_PC2': scores_curve[:, 1]
    })
    
    param_pc_df = pd.DataFrame({
        'galaxy_id': features_df['galaxy_id'],
        'Param_PC1': scores[:, 0],
        'Param_PC2': scores[:, 1]
    })
    
    merged = curve_pc_df.merge(param_pc_df, on='galaxy_id', how='inner')
    
    print(f"\nCorrelations between curve-PCA and parameter-PCA:")
    print(f"  Curve-PC1 vs Param-PC1: rho = {spearmanr(merged['Curve_PC1'], merged['Param_PC1'])[0]:+.3f}")
    print(f"  Curve-PC1 vs Param-PC2: rho = {spearmanr(merged['Curve_PC1'], merged['Param_PC2'])[0]:+.3f}")
    print(f"  Curve-PC2 vs Param-PC1: rho = {spearmanr(merged['Curve_PC2'], merged['Param_PC1'])[0]:+.3f}")
    print(f"  Curve-PC2 vs Param-PC2: rho = {spearmanr(merged['Curve_PC2'], merged['Param_PC2'])[0]:+.3f}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nParameter-space PCA reveals which MODEL FEATURES drive variance.")
    print("Curve-shape PCA reveals which EMPIRICAL MODES exist.")
    print("\nBoth perspectives are complementary and valuable!")

if __name__ == '__main__':
    main()








