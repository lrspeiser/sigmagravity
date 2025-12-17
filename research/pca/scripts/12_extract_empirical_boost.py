#!/usr/bin/env python3
"""
Extract EMPIRICAL boost function K(R) from SPARC data using PCA.

Strategy:
1. For each galaxy: K_empirical = (V_obs^2 / V_bar^2) - 1
2. Normalize K by peak value to get shape
3. Run PCA on K(R/Rd) profiles
4. Fit Sigma-Gravity functional form to empirical PC1

This reveals what K(R) SHOULD look like to match data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit, minimize
from scipy.stats import spearmanr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_boost_from_curve(curve_data, Rd):
    """
    Extract empirical boost function from rotation curve.
    
    K_empirical = (V_obs^2 / V_bar^2) - 1
    
    Returns normalized radius and boost.
    """
    R = curve_data['R_kpc'].values
    V_obs = curve_data['V_obs'].values
    
    # Baryonic velocity
    V_disk = curve_data.get('V_disk', np.zeros_like(R)).values
    V_gas = curve_data.get('V_gas', np.zeros_like(R)).values
    V_bul = curve_data.get('V_bul', np.zeros_like(R)).values if 'V_bul' in curve_data else np.zeros_like(R)
    V_bar = np.sqrt(V_disk**2 + V_gas**2 + V_bul**2)
    
    # Normalized radius
    x = R / Rd
    
    # Empirical boost
    V_bar_safe = np.maximum(V_bar, 1.0)  # Avoid division by zero
    K_emp = (V_obs**2 / V_bar_safe**2) - 1.0
    
    # Filter reasonable values
    mask = (x > 0.1) & (x < 10) & (K_emp > -0.9) & (K_emp < 10)
    
    return x[mask], K_emp[mask]

def resample_to_grid(x_data, K_data, x_grid):
    """Resample boost profile to common grid"""
    # Simple linear interpolation
    K_resampled = np.interp(x_grid, x_data, K_data, left=np.nan, right=np.nan)
    return K_resampled

def coherence_burr_xii(R, l0=5.0, p=2.0, n_coh=1.5):
    """Burr-XII coherence function"""
    x = (R / l0)**p
    return 1.0 - (1.0 + x)**(-n_coh)

def fit_coherence_to_shape(x_grid, K_empirical, A_est=1.0):
    """
    Fit Burr-XII parameters to empirical boost shape.
    
    K_empirical ≈ A * C(R/l0; p, n_coh)
    
    We normalize K by an estimated amplitude first.
    """
    # Normalize by peak
    K_peak = np.nanmax(K_empirical)
    if K_peak < 0.1:
        return None
    
    K_norm = K_empirical / K_peak
    
    # Fit coherence function shape
    def model(x, l0, p, n_coh):
        return coherence_burr_xii(x, l0, p, n_coh)
    
    try:
        # Initial guess
        p0 = [3.0, 2.0, 1.5]
        bounds = ([0.5, 0.5, 0.5], [10.0, 5.0, 3.0])
        
        # Only use finite points
        mask = np.isfinite(K_norm) & np.isfinite(x_grid)
        if mask.sum() < 10:
            return None
        
        popt, _ = curve_fit(model, x_grid[mask], K_norm[mask], 
                            p0=p0, bounds=bounds, maxfev=5000)
        
        return {'l0': popt[0], 'p': popt[1], 'n_coh': popt[2], 'A': K_peak}
    except:
        return None

def main():
    print("=" * 70)
    print("EXTRACTING EMPIRICAL BOOST FUNCTION FROM SPARC DATA")
    print("=" * 70)
    
    # Paths
    repo_root = Path(__file__).parent.parent.parent
    curves_dir = repo_root / 'pca' / 'data' / 'raw' / 'sparc_curves'
    meta_file = repo_root / 'pca' / 'data' / 'raw' / 'metadata' / 'sparc_meta.csv'
    output_dir = repo_root / 'pca' / 'outputs' / 'empirical_boost'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    meta = pd.read_csv(meta_file)
    
    # Setup grid
    x_grid = np.linspace(0.2, 6.0, 50)
    
    # Extract boost profiles
    print(f"\nExtracting empirical boost K(R) for each galaxy...")
    
    K_matrix = []
    names_list = []
    fit_params = []
    
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Extracting"):
        name = row['name']
        Rd = row['Rd']
        
        if not np.isfinite(Rd) or Rd <= 0:
            continue
        
        curve_file = curves_dir / f"{name}.csv"
        if not curve_file.exists():
            continue
        
        try:
            curve = pd.read_csv(curve_file)
            
            # Extract empirical boost
            x_data, K_data = extract_boost_from_curve(curve, Rd)
            
            if len(x_data) < 5:
                continue
            
            # Resample to grid
            K_resampled = resample_to_grid(x_data, K_data, x_grid)
            
            # Fit parameters to this galaxy's boost
            params = fit_coherence_to_shape(x_grid, K_resampled)
            
            if params is not None:
                K_matrix.append(K_resampled)
                names_list.append(name)
                fit_params.append({
                    'name': name,
                    'A_empirical': params['A'],
                    'l0_empirical': params['l0'],
                    'p_empirical': params['p'],
                    'n_coh_empirical': params['n_coh'],
                    'Rd': Rd,
                    'Vf': row['Vf'],
                    'Mbar': row['Mbar']
                })
        except Exception as e:
            continue
    
    K_matrix = np.array(K_matrix)
    n_galaxies = len(names_list)
    
    print(f"\nExtracted boost profiles for {n_galaxies} galaxies")
    
    if n_galaxies < 10:
        print("Error: Too few galaxies with valid boost profiles")
        return
    
    # Run PCA on empirical boost
    print("\nRunning PCA on empirical boost functions...")
    
    # Standardize (handle NaNs)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA as sklearnPCA
    
    # Fill NaNs with column mean
    K_filled = K_matrix.copy()
    for j in range(K_matrix.shape[1]):
        col = K_matrix[:, j]
        col_mean = np.nanmean(col)
        K_filled[:, j] = np.where(np.isfinite(col), col, col_mean)
    
    # PCA
    pca_boost = sklearnPCA(n_components=5)
    scores_boost = pca_boost.fit_transform(K_filled)
    components_boost = pca_boost.components_
    evr_boost = pca_boost.explained_variance_ratio_
    
    print(f"\nPCA on Boost Functions:")
    print(f"  PC1: {evr_boost[0]*100:.1f}% of variance")
    print(f"  PC2: {evr_boost[1]*100:.1f}% of variance")
    print(f"  PC3: {evr_boost[2]*100:.1f}% of variance")
    print(f"  PC1-3: {evr_boost[:3].sum()*100:.1f}% total")
    
    # Plot empirical PC1 of boost
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top: Mean boost and PC1 mode
    K_mean = np.nanmean(K_filled, axis=0)
    K_pc1_mode = K_mean + 2 * components_boost[0, :]  # +2 std along PC1
    
    ax1.plot(x_grid, K_mean, 'k-', lw=3, label='Mean K(R/Rd)', alpha=0.8)
    ax1.fill_between(x_grid, 
                      K_mean - np.nanstd(K_filled, axis=0),
                      K_mean + np.nanstd(K_filled, axis=0),
                      alpha=0.2, color='gray', label='±1 std')
    ax1.plot(x_grid, K_pc1_mode, 'r--', lw=2, label=f'PC1 mode (+2σ, {evr_boost[0]*100:.1f}% var)')
    ax1.axhline(0, color='k', ls=':', alpha=0.5)
    ax1.set_xlabel('R / Rd', fontsize=12)
    ax1.set_ylabel('Empirical Boost K(R)', fontsize=12)
    ax1.set_title('Empirical Boost Function from SPARC Data', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Bottom: PC1 loading (what varies most)
    ax2.plot(x_grid, components_boost[0, :], 'o-', lw=2, ms=4, color='darkblue')
    ax2.axhline(0, color='k', ls='--', alpha=0.5)
    ax2.set_xlabel('R / Rd', fontsize=12)
    ax2.set_ylabel('PC1 Loading (Boost Space)', fontsize=12)
    ax2.set_title(f'Boost PC1: Dominant Mode of Variation ({evr_boost[0]*100:.1f}% variance)', fontsize=13)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    fig_file = output_dir / 'empirical_boost_pca.png'
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure: {fig_file}")
    plt.close()
    
    # Save empirical parameters
    params_df = pd.DataFrame(fit_params)
    params_file = output_dir / 'empirical_boost_params.csv'
    params_df.to_csv(params_file, index=False)
    print(f"Saved parameters: {params_file}")
    
    # Check correlations
    print("\n" + "=" * 70)
    print("EMPIRICAL PARAMETER CORRELATIONS")
    print("=" * 70)
    
    print("\nHow empirical A varies with galaxy properties:")
    for col in ['Vf', 'Mbar', 'Rd']:
        mask = np.isfinite(params_df['A_empirical']) & np.isfinite(params_df[col]) & (params_df[col] > 0)
        if mask.sum() > 10:
            rho, p = spearmanr(np.log10(params_df.loc[mask, col]), params_df.loc[mask, 'A_empirical'])
            print(f"  A_empirical vs log10({col}): rho = {rho:+.3f}, p = {p:.3e}")
    
    print("\nHow empirical l0 varies with galaxy properties:")
    for col in ['Vf', 'Mbar', 'Rd']:
        mask = np.isfinite(params_df['l0_empirical']) & np.isfinite(params_df[col]) & (params_df[col] > 0)
        if mask.sum() > 10:
            rho, p = spearmanr(np.log10(params_df.loc[mask, col]), params_df.loc[mask, 'l0_empirical'])
            print(f"  l0_empirical vs log10({col}): rho = {rho:+.3f}, p = {p:.3e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nExtracted empirical boost K(R) for {n_galaxies} galaxies")
    print(f"\nBoost PCA Results:")
    print(f"  PC1 explains {evr_boost[0]*100:.1f}% of boost variation")
    print(f"  This reveals the DOMINANT SHAPE that K(R) should have")
    
    print(f"\nEmpirical Parameter Statistics:")
    print(f"  A_empirical:  mean = {params_df['A_empirical'].mean():.3f}, std = {params_df['A_empirical'].std():.3f}")
    print(f"  l0_empirical: mean = {params_df['l0_empirical'].mean():.2f} kpc, std = {params_df['l0_empirical'].std():.2f} kpc")
    print(f"  p_empirical:  mean = {params_df['p_empirical'].mean():.2f}, std = {params_df['p_empirical'].std():.2f}")
    
    print(f"\nNext: Compare Burr-XII form to empirical PC1 shape")
    print(f"      to see if functional form is correct.")

if __name__ == '__main__':
    main()












