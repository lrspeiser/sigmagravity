#!/usr/bin/env python3
"""
Quick exploration script for PCA results.
Run this to load and explore the PCA analysis interactively.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load PCA results
pca_data = np.load('pca/outputs/pca_results_curve_only.npz')
components = pca_data['components']  # [n_components, n_features]
scores = pca_data['scores']  # [n_galaxies, n_components]
explained_variance = pca_data['evr']  # explained variance ratio

# Load curve matrix
curve_data = np.load('pca/data/processed/sparc_curvematrix.npz', allow_pickle=True)
curve_mat = curve_data['curve_mat']
weight_mat = curve_data['weight_mat']
names = curve_data['names']
x_grid = curve_data['x_grid']

# Load clusters
clusters = pd.read_csv('pca/outputs/clusters.csv')

# Load metadata
metadata = pd.read_csv('pca/data/raw/metadata/sparc_meta.csv')

print("=" * 70)
print("PCA ANALYSIS RESULTS")
print("=" * 70)
print(f"\nNumber of galaxies: {len(names)}")
print(f"Number of radial points: {len(x_grid)}")
print(f"Radial grid range: {x_grid.min():.2f} - {x_grid.max():.2f} (R/Rd)")
print(f"\nPC1 explains {explained_variance[0]*100:.1f}% of variance")
print(f"PC1-3 explain {explained_variance[:3].sum()*100:.1f}% of variance")
print(f"PC1-5 explain {explained_variance[:5].sum()*100:.1f}% of variance")

print("\n" + "=" * 70)
print("CLUSTER STATISTICS")
print("=" * 70)
cluster_counts = clusters['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"  Cluster {cluster_id}: {count} galaxies")

print("\n" + "=" * 70)
print("EXAMPLE: Top 10 galaxies by PC1 score")
print("=" * 70)
top10_idx = np.argsort(scores[:, 0])[-10:]
for idx in reversed(top10_idx):
    name = names[idx]
    pc1_score = scores[idx, 0]
    cluster_id = clusters.loc[clusters['name'] == name, 'cluster'].values[0] if name in clusters['name'].values else -1
    print(f"  {name:12s}  PC1={pc1_score:7.2f}  Cluster={cluster_id}")

print("\n" + "=" * 70)
print("INTERACTIVE PLOTTING")
print("=" * 70)
print("\nTo plot rotation curves for a specific cluster:")
print("  >>> plot_cluster_curves(cluster_id=5)")
print("\nTo plot PC loadings:")
print("  >>> plot_pc_loadings(pc_idx=0)")
print("\nTo compare a galaxy to the mean:")
print("  >>> plot_galaxy_vs_mean('NGC3198')")


def plot_cluster_curves(cluster_id=5, max_curves=20):
    """Plot rotation curves for galaxies in a specific cluster"""
    cluster_galaxies = clusters[clusters['cluster'] == cluster_id]['name'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot up to max_curves randomly selected
    n_plot = min(len(cluster_galaxies), max_curves)
    selected = np.random.choice(cluster_galaxies, n_plot, replace=False)
    
    for name in selected:
        idx = np.where(names == name)[0]
        if len(idx) > 0:
            idx = idx[0]
            curve = curve_mat[idx, :]
            ax.plot(x_grid, curve, alpha=0.5, lw=1)
    
    ax.set_xlabel('R / Rd', fontsize=12)
    ax.set_ylabel('V / Vf (normalized)', fontsize=12)
    ax.set_title(f'Cluster {cluster_id} Rotation Curves (n={len(cluster_galaxies)})', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_pc_loadings(pc_idx=0):
    """Plot the radial loading profile for a principal component"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    loading = components[pc_idx, :]
    ax.plot(x_grid, loading, 'o-', lw=2, ms=4)
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('R / Rd', fontsize=12)
    ax.set_ylabel(f'PC{pc_idx+1} Loading', fontsize=12)
    ax.set_title(f'PC{pc_idx+1} Radial Loading Profile ({explained_variance[pc_idx]*100:.1f}% variance)', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_galaxy_vs_mean(galaxy_name):
    """Plot a specific galaxy's curve vs the mean curve"""
    idx = np.where(names == galaxy_name)[0]
    if len(idx) == 0:
        print(f"Galaxy '{galaxy_name}' not found!")
        print(f"Available galaxies: {', '.join(names[:10])}...")
        return None
    
    idx = idx[0]
    curve = curve_mat[idx, :]
    mean_curve = np.mean(curve_mat, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot curves
    ax1.plot(x_grid, mean_curve, 'k--', lw=2, label='Mean')
    ax1.plot(x_grid, curve, 'r-', lw=2, label=galaxy_name)
    ax1.set_ylabel('V / Vf', fontsize=12)
    ax1.set_title(f'{galaxy_name} vs Mean Rotation Curve', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot residual
    residual = curve - mean_curve
    ax2.plot(x_grid, residual, 'b-', lw=2)
    ax2.axhline(0, color='k', ls='--', alpha=0.3)
    ax2.set_xlabel('R / Rd', fontsize=12)
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.set_title('Residual from Mean', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print PC scores
    print(f"\n{galaxy_name} PC scores:")
    for i in range(min(5, len(scores[idx]))):
        print(f"  PC{i+1}: {scores[idx, i]:7.3f}")
    
    return fig


def plot_pc_scatter(pc_x=0, pc_y=1):
    """Plot scatter of galaxies in PC space"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by cluster
    for cluster_id in clusters['cluster'].unique():
        mask = clusters['cluster'] == cluster_id
        cluster_names = clusters[mask]['name'].values
        indices = [np.where(names == name)[0][0] for name in cluster_names if name in names]
        
        if len(indices) > 0:
            ax.scatter(scores[indices, pc_x], scores[indices, pc_y], 
                      label=f'Cluster {cluster_id}', alpha=0.7, s=50)
    
    ax.set_xlabel(f'PC{pc_x+1} ({explained_variance[pc_x]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC{pc_y+1} ({explained_variance[pc_y]*100:.1f}%)', fontsize=12)
    ax.set_title('Galaxy Distribution in PC Space', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def correlate_pcs_with_physics(max_pc=3):
    """Compute correlations between PC scores and physical parameters"""
    from scipy.stats import spearmanr, pearsonr
    
    # Merge names→PC scores
    pc_df = pd.DataFrame(scores[:, :max_pc], columns=[f'PC{i+1}' for i in range(max_pc)])
    pc_df['name'] = names
    meta_df = pd.read_csv('pca/data/raw/metadata/sparc_meta.csv')
    df = pc_df.merge(meta_df, left_on='name', right_on='name', how='inner')
    
    # Build log-scaled columns
    cols = [('log10_Mbar','Mbar'), ('log10_Sigma0','Sigma0'),
            ('log10_Rd','Rd'), ('log10_Vf','Vf')]
    for new, src in cols:
        if new not in df.columns and src in df.columns:
            df[new] = np.log10(df[src].replace(0, np.nan))
    
    print("\n" + "=" * 70)
    print("PC-PHYSICS CORRELATIONS")
    print("=" * 70)
    
    for j in range(max_pc):
        pc = f'PC{j+1}'
        print(f'\n{pc} correlations:')
        for new, _ in cols:
            if new in df.columns:
                m = np.isfinite(df[pc]) & np.isfinite(df[new])
                if m.sum() > 10:
                    r, _ = pearsonr(df.loc[m, pc], df.loc[m, new])
                    rho, _ = spearmanr(df.loc[m, pc], df.loc[m, new])
                    print(f'  {new:>12s}: Pearson {r:+.3f}, Spearman {rho:+.3f}, n={m.sum()}')
    
    return df


def principal_angles_between_subspaces(A, B):
    """
    Compute principal angles between k-dimensional subspaces spanned by A and B.
    
    Parameters:
    -----------
    A, B : ndarray [n_samples, k]
        Score matrices for the same galaxies (must be aligned)
        
    Returns:
    --------
    angles : ndarray
        Principal angles in radians, ascending order
    """
    # Orthonormalize both bases
    Qa, _ = np.linalg.qr(A, mode='reduced')
    Qb, _ = np.linalg.qr(B, mode='reduced')
    
    # SVD of overlap matrix
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0, 1)  # numerical stability
    
    return np.arccos(s)  # radians


def partial_spearman(x, y, controls, df):
    """
    Compute partial Spearman correlation between x and y, controlling for other variables.
    
    Uses rank-transformation followed by linear regression of residuals.
    
    Parameters:
    -----------
    x, y : str
        Column names for variables of interest
    controls : list of str
        Column names for control variables
    df : DataFrame
        Data containing all columns
        
    Returns:
    --------
    rho, pval : float
        Partial Spearman correlation and p-value
    """
    from scipy.stats import spearmanr
    
    # Rank-transform all variables
    cols = [x, y] + controls
    Z = df[cols].rank()
    
    # Regress out controls from x and y using OLS on ranks
    Xc = np.c_[np.ones(len(Z)), Z[controls].values]
    
    # OLS coefficients and residuals
    bx = np.linalg.lstsq(Xc, Z[x].values, rcond=None)[0]
    by = np.linalg.lstsq(Xc, Z[y].values, rcond=None)[0]
    
    rx = Z[x].values - Xc @ bx
    ry = Z[y].values - Xc @ by
    
    # Spearman on residuals
    return spearmanr(rx, ry)


def compute_partial_correlations(max_pc=3):
    """Compute partial correlations controlling for other physical parameters"""
    from scipy.stats import spearmanr
    
    # Merge data
    pc_df = pd.DataFrame(scores[:, :max_pc], columns=[f'PC{i+1}' for i in range(max_pc)])
    pc_df['name'] = names
    meta_df = pd.read_csv('pca/data/raw/metadata/sparc_meta.csv')
    df = pc_df.merge(meta_df, on='name', how='inner')
    
    # Build log columns
    cols = [('log10_Mbar','Mbar'), ('log10_Sigma0','Sigma0'),
            ('log10_Rd','Rd'), ('log10_Vf','Vf')]
    for new, src in cols:
        if new not in df.columns and src in df.columns:
            df[new] = np.log10(df[src].replace(0, np.nan))
    
    print("\n" + "=" * 70)
    print("PARTIAL CORRELATIONS (controlling for other physics)")
    print("=" * 70)
    
    phys_cols = ['log10_Mbar', 'log10_Sigma0', 'log10_Rd', 'log10_Vf']
    
    for j in range(max_pc):
        pc = f'PC{j+1}'
        print(f'\n{pc}:')
        for col in phys_cols:
            if col in df.columns:
                # Controls = all other physics columns
                controls = [c for c in phys_cols if c != col and c in df.columns]
                
                # Regular Spearman
                m = np.isfinite(df[pc]) & np.isfinite(df[col])
                for c in controls:
                    m &= np.isfinite(df[c])
                
                if m.sum() > 20:
                    rho_raw, _ = spearmanr(df.loc[m, pc], df.loc[m, col])
                    rho_partial, p_partial = partial_spearman(pc, col, controls, df.loc[m])
                    print(f'  {col:>14s}: ρ={rho_raw:+.3f} → ρ_partial={rho_partial:+.3f} (p={p_partial:.3e})')
    
    return df


def reconstruction_error_budget():
    """Compute reconstruction quality with PC1-3 vs observational errors"""
    # Load curve data
    curve_data = np.load('pca/data/processed/sparc_curvematrix.npz', allow_pickle=True)
    curve_mat = curve_data['curve_mat']
    weight_mat = curve_data['weight_mat']
    
    # Reconstruct with PC1-3
    # scores [N, 10], components [10, D]
    # Original (centered, z-scored) = scores @ components
    # We need to access the mean and std from PCA
    pca_data = np.load('pca/outputs/pca_results_curve_only.npz')
    mu = pca_data['mu']
    sd = pca_data['sd']
    
    # Reconstruct with 3 PCs
    X_recon_3pc = scores[:, :3] @ components[:3, :]
    X_recon_3pc = X_recon_3pc * sd + mu  # un-standardize
    
    # Compute weighted RMSE per galaxy
    residuals_3pc = curve_mat - X_recon_3pc
    weighted_rmse = np.sqrt(np.sum(weight_mat * residuals_3pc**2, axis=1) / np.sum(weight_mat, axis=1))
    
    # Also compute for all 10 PCs (should be near-perfect)
    X_recon_10pc = scores @ components
    X_recon_10pc = X_recon_10pc * sd + mu
    residuals_10pc = curve_mat - X_recon_10pc
    weighted_rmse_10pc = np.sqrt(np.sum(weight_mat * residuals_10pc**2, axis=1) / np.sum(weight_mat, axis=1))
    
    print("\n" + "=" * 70)
    print("RECONSTRUCTION ERROR BUDGET")
    print("=" * 70)
    print(f"\nWeighted RMSE with 3 PCs:")
    print(f"  Mean: {weighted_rmse.mean():.4f}")
    print(f"  Median: {np.median(weighted_rmse):.4f}")
    print(f"  Std: {weighted_rmse.std():.4f}")
    print(f"\nWeighted RMSE with 10 PCs (all):")
    print(f"  Mean: {weighted_rmse_10pc.mean():.6f}")
    print(f"  Median: {np.median(weighted_rmse_10pc):.6f}")
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(weighted_rmse, bins=30, alpha=0.7, label='3 PCs', edgecolor='black')
    ax.axvline(np.median(weighted_rmse), color='r', ls='--', lw=2, label=f'Median = {np.median(weighted_rmse):.3f}')
    ax.set_xlabel('Weighted RMSE', fontsize=12)
    ax.set_ylabel('Number of Galaxies', fontsize=12)
    ax.set_title('Reconstruction Error with PC1-3', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return weighted_rmse, weighted_rmse_10pc


def list_outliers_by_cluster(min_size=5):
    """List galaxies in small clusters (potential outliers)"""
    small = clusters['cluster'].value_counts()
    small = small[small < min_size].index.tolist()
    meta_df = pd.read_csv('pca/data/raw/metadata/sparc_meta.csv')
    
    out = []
    for c in small:
        names_c = clusters[clusters['cluster']==c]['name']
        for n in names_c:
            idx = np.where(names==n)[0]
            if idx.size:
                rec = meta_df.loc[meta_df['name']==n]
                if len(rec) > 0:
                    rec = rec.iloc[0]
                    # Get PC scores
                    pc1 = scores[idx[0], 0] if idx.size else np.nan
                    pc2 = scores[idx[0], 1] if idx.size else np.nan
                    out.append(dict(
                        cluster=c, 
                        name=n,
                        PC1=pc1,
                        PC2=pc2,
                        Vf=rec.get('Vf', np.nan),
                        Rd=rec.get('Rd', np.nan),
                        Sigma0=rec.get('Sigma0', np.nan),
                        Mbar=rec.get('Mbar', np.nan)
                    ))
    
    result = pd.DataFrame(out).sort_values(['cluster','name'])
    print("\n" + "=" * 70)
    print(f"OUTLIER GALAXIES (clusters with < {min_size} members)")
    print("=" * 70)
    print(result.to_string(index=False))
    return result


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Ready for interactive exploration!")
    print("=" * 70)
    print("\nExample commands:")
    print("  plot_cluster_curves(cluster_id=5)")
    print("  plot_pc_loadings(pc_idx=0)")
    print("  plot_galaxy_vs_mean('NGC3198')")
    print("  plot_pc_scatter(pc_x=0, pc_y=1)")
    print("  correlate_pcs_with_physics()       # Physics correlations")
    print("  compute_partial_correlations()     # NEW: Partial correlations")
    print("  reconstruction_error_budget()      # NEW: Error budget analysis")
    print("  list_outliers_by_cluster()         # Outlier diagnostics")
    print("\nTo enter interactive mode, run: python -i pca/explore_results.py")

