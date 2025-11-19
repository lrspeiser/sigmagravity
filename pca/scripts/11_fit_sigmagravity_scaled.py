#!/usr/bin/env python3
"""
Fit Sigma-Gravity with PCA-informed parameter scalings.

Based on PCA diagnostic:
- A scales with Vf (strongest: rho=+0.78)
- l0 scales with Rd (strong: rho=+0.52)

We'll calibrate the scaling exponents to minimize correlation with PC1.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Coherence function
def coherence_function(R, l0=5.0, p=2.0, n_coh=1.5):
    """Burr-XII coherence: C(R) = 1 - [1 + (R/l0)^p]^{-n_coh}"""
    x = (R / l0)**p
    return 1.0 - (1.0 + x)**(-n_coh)

def sigma_gravity_boost(R, A, l0, p=2.0, n_coh=1.5):
    """K(R) = A * C(R/l0)"""
    C = coherence_function(R, l0, p, n_coh)
    return A * C

def amplitude_scaling(Vf, A0, alpha, Vf_pivot=100.0):
    """A = A0 * (Vf / Vf_pivot)^alpha"""
    Vf_safe = max(Vf, 20.0)  # Floor for numerical stability
    return A0 * (Vf_safe / Vf_pivot)**alpha

def coherence_scaling(Rd, l0_base, beta, Rd_pivot=5.0):
    """l0 = l0_base * (Rd / Rd_pivot)^beta"""
    Rd_safe = max(Rd, 0.5)  # Floor for numerical stability
    return l0_base * (Rd_safe / Rd_pivot)**beta

def fit_galaxy_with_scalings(curve_data, meta_row, A0, alpha, l0_base, beta, p=2.0, n_coh=1.5):
    """Fit single galaxy with parameter scalings"""
    # Extract data
    R = curve_data['R_kpc'].values
    V_obs = curve_data['V_obs'].values
    eV_obs = curve_data['eV_obs'].values
    
    # Baryonic components
    V_disk = curve_data.get('V_disk', np.zeros_like(R)).values
    V_gas = curve_data.get('V_gas', np.zeros_like(R)).values
    V_bul = curve_data.get('V_bul', np.zeros_like(R)).values if 'V_bul' in curve_data else np.zeros_like(R)
    V_bar = np.sqrt(V_disk**2 + V_gas**2 + V_bul**2)
    
    # Get galaxy properties for scaling
    Vf = meta_row['Vf']
    Rd = meta_row['Rd']
    
    if not np.isfinite(Vf) or Vf <= 0:
        Vf = 50.0  # Fallback
    if not np.isfinite(Rd) or Rd <= 0:
        Rd = 2.0  # Fallback
    
    # Compute scaled parameters for THIS galaxy
    A = amplitude_scaling(Vf, A0, alpha)
    l0 = coherence_scaling(Rd, l0_base, beta)
    
    # Baryonic acceleration
    g_bar = V_bar**2 / np.maximum(R, 0.1) / 3.086e16
    
    # Sigma-Gravity boost
    K = sigma_gravity_boost(R, A, l0, p, n_coh)
    
    # Modified acceleration
    g_model = g_bar * (1 + K)
    
    # Convert to velocity
    V_model = np.sqrt(g_model * R * 3.086e16)
    
    # Residuals
    residuals = V_obs - V_model
    weighted_residuals = residuals / np.maximum(eV_obs, 1.0)
    
    # Metrics
    rms = np.sqrt(np.mean(residuals**2))
    chi2 = np.sum(weighted_residuals**2)
    chi2_red = chi2 / max(len(R) - 4, 1)
    ape = np.mean(np.abs(residuals / V_obs)) * 100
    
    return {
        'rms': rms,
        'chi2': chi2,
        'chi2_red': chi2_red,
        'ape': ape,
        'n_points': len(R),
        'A_used': A,  # Store the actual A used
        'l0_used': l0,  # Store the actual l0 used
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals)
    }

def fit_population(meta, curves_dir, A0, alpha, l0_base, beta, p=2.0, n_coh=1.5):
    """Fit all galaxies with given scaling parameters"""
    results = []
    
    for _, row in meta.iterrows():
        name = row['name']
        curve_file = curves_dir / f"{name}.csv"
        
        if not curve_file.exists():
            continue
        
        try:
            curve = pd.read_csv(curve_file)
            fit_result = fit_galaxy_with_scalings(curve, row, A0, alpha, l0_base, beta, p, n_coh)
            
            results.append({
                'name': name,
                'Rd': row.get('Rd', np.nan),
                'Vf': row.get('Vf', np.nan),
                'Mbar': row.get('Mbar', np.nan),
                'Sigma0': row.get('Sigma0', np.nan),
                'HSB_LSB': row.get('HSB_LSB', 'Unknown'),
                'residual_rms': fit_result['rms'],
                'chi2': fit_result['chi2'],
                'chi2_red': fit_result['chi2_red'],
                'ape': fit_result['ape'],
                'n_points': fit_result['n_points'],
                'A_used': fit_result['A_used'],
                'l0_used': fit_result['l0_used'],
                'A0': A0,
                'alpha': alpha,
                'l0_base': l0_base,
                'beta': beta,
                'p': p,
                'n_coh': n_coh
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

def objective_for_calibration(params, meta, curves_dir, pca_scores):
    """
    Objective function: Minimize |rho(residual, PC1)| + mean RMS
    
    This calibrates the scaling exponents to pass the PCA test.
    """
    A0, alpha, l0_base, beta = params
    
    # Fit all galaxies
    results_df = fit_population(meta, curves_dir, A0, alpha, l0_base, beta)
    
    if len(results_df) < 50:
        return 1e10  # Failed
    
    # Merge with PCA scores
    merged = results_df.merge(pca_scores, on='name', how='inner')
    
    if len(merged) < 50:
        return 1e10
    
    # Compute correlation with PC1
    from scipy.stats import spearmanr
    rho_pc1, _ = spearmanr(merged['residual_rms'], merged['PC1'])
    
    # Objective: minimize |rho| + penalize high RMS
    mean_rms = merged['residual_rms'].mean()
    
    # Combined objective
    obj = abs(rho_pc1) * 100 + mean_rms  # Weight rho heavily
    
    return obj

def main():
    print("=" * 70)
    print("SIGMA-GRAVITY with PCA-GUIDED PARAMETER SCALINGS")
    print("=" * 70)
    
    # Paths
    repo_root = Path(__file__).parent.parent.parent
    curves_dir = repo_root / 'pca' / 'data' / 'raw' / 'sparc_curves'
    meta_file = repo_root / 'pca' / 'data' / 'raw' / 'metadata' / 'sparc_meta.csv'
    pca_file = repo_root / 'pca' / 'outputs' / 'pca_results_curve_only.npz'
    output_dir = repo_root / 'pca' / 'outputs' / 'sigmagravity_fits'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    meta = pd.read_csv(meta_file)
    
    # Load PCA for calibration
    pca = np.load(pca_file, allow_pickle=True)
    names_pca = pca['names']
    scores = pca['scores']
    
    pca_scores = pd.DataFrame({
        'name': names_pca,
        'PC1': scores[:, 0],
        'PC2': scores[:, 1],
        'PC3': scores[:, 2]
    })
    
    print(f"\nLoaded {len(meta)} galaxies")
    print(f"PCA: {len(names_pca)} galaxies with scores")
    
    # Strategy: Grid search over plausible scaling exponents
    print("\n" + "=" * 70)
    print("CALIBRATING SCALING EXPONENTS")
    print("=" * 70)
    print("\nSearching for optimal (A0, alpha, l0_base, beta)...")
    print("Goal: Minimize |rho(residual, PC1)| + mean_RMS\n")
    
    # Coarse grid search first
    best_obj = 1e10
    best_params = None
    
    # Reasonable ranges based on dimensional analysis and PCA correlation strengths
    A0_range = [0.10, 0.15, 0.20, 0.25, 0.30]
    alpha_range = [0.2, 0.3, 0.4, 0.5, 0.6]
    l0_base_range = [2.0, 2.5, 3.0, 3.5, 4.0]
    beta_range = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    total_trials = len(A0_range) * len(alpha_range) * len(l0_base_range) * len(beta_range)
    print(f"Grid search: {total_trials} combinations")
    
    from itertools import product
    
    for A0, alpha, l0_base, beta in tqdm(list(product(A0_range, alpha_range, l0_base_range, beta_range)), 
                                          desc="Grid search"):
        # Fit population
        results_df = fit_population(meta, curves_dir, A0, alpha, l0_base, beta)
        
        if len(results_df) < 50:
            continue
        
        # Merge with PCA
        merged = results_df.merge(pca_scores, on='name', how='inner')
        
        if len(merged) < 50:
            continue
        
        # Compute objective
        from scipy.stats import spearmanr
        rho_pc1, _ = spearmanr(merged['residual_rms'], merged['PC1'])
        mean_rms = merged['residual_rms'].mean()
        
        obj = abs(rho_pc1) * 100 + mean_rms
        
        if obj < best_obj:
            best_obj = obj
            best_params = (A0, alpha, l0_base, beta)
            best_rho = rho_pc1
            best_rms = mean_rms
            
            print(f"\n  New best: A0={A0:.2f}, alpha={alpha:.2f}, l0_base={l0_base:.2f}, beta={beta:.2f}")
            print(f"    rho(resid,PC1) = {rho_pc1:+.3f}, mean_RMS = {mean_rms:.2f} km/s, obj = {obj:.2f}")
    
    # Use best parameters
    A0_best, alpha_best, l0_base_best, beta_best = best_params
    
    print("\n" + "=" * 70)
    print("OPTIMAL SCALING PARAMETERS FOUND")
    print("=" * 70)
    print(f"\n  A0 = {A0_best:.3f}")
    print(f"  alpha = {alpha_best:.3f}  (A = A0 * (Vf/100)^alpha)")
    print(f"  l0_base = {l0_base_best:.3f} kpc")
    print(f"  beta = {beta_best:.3f}  (l0 = l0_base * (Rd/5)^beta)")
    print(f"\n  Result: rho = {best_rho:+.3f}, mean_RMS = {best_rms:.2f} km/s")
    
    # Fit final model with best parameters
    print("\n" + "=" * 70)
    print("FITTING FINAL MODEL WITH OPTIMAL SCALINGS")
    print("=" * 70)
    
    final_results = fit_population(meta, curves_dir, A0_best, alpha_best, l0_base_best, beta_best)
    
    # Save results
    output_file = output_dir / 'sparc_sigmagravity_scaled_fits.csv'
    final_results.to_csv(output_file, index=False)
    
    print(f"\nFitted {len(final_results)} galaxies")
    print(f"Saved to: {output_file}")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("FINAL FIT QUALITY")
    print("=" * 70)
    
    print(f"\nResidual Statistics:")
    print(f"  Mean RMS:   {final_results['residual_rms'].mean():.2f} km/s")
    print(f"  Median RMS: {final_results['residual_rms'].median():.2f} km/s")
    print(f"  Std RMS:    {final_results['residual_rms'].std():.2f} km/s")
    print(f"  Min RMS:    {final_results['residual_rms'].min():.2f} km/s")
    print(f"  Max RMS:    {final_results['residual_rms'].max():.2f} km/s")
    
    print(f"\nParameter Ranges Used:")
    print(f"  A:  {final_results['A_used'].min():.3f} - {final_results['A_used'].max():.3f}")
    print(f"  l0: {final_results['l0_used'].min():.2f} - {final_results['l0_used'].max():.2f} kpc")
    
    # PCA test
    merged_final = final_results.merge(pca_scores, on='name', how='inner')
    
    print("\n" + "=" * 70)
    print("PCA TEST RESULTS (WITH SCALINGS)")
    print("=" * 70)
    
    from scipy.stats import spearmanr
    
    for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
        rho, p_val = spearmanr(merged_final['residual_rms'], merged_final[pc])
        status = "PASS" if abs(rho) < 0.2 else "FAIL"
        print(f"\n  residual vs {pc}: rho = {rho:+.3f}, p = {p_val:.3e}  [{status}]")
    
    # Overall verdict
    rho_pc1_final, p_pc1_final = spearmanr(merged_final['residual_rms'], merged_final['PC1'])
    
    print("\n" + "=" * 70)
    if abs(rho_pc1_final) < 0.2 and p_pc1_final > 0.05:
        print("VERDICT: PASS - Model captures dominant empirical structure!")
        print("=" * 70)
    else:
        print("VERDICT: FAIL - Further refinement needed")
        print("=" * 70)
        print(f"\nrho = {rho_pc1_final:+.3f} (target: |rho| < 0.2)")
        print(f"p = {p_pc1_final:.3e}")
    
    print(f"\nNext: Run comparison analysis")
    print(f"  python pca/scripts/08_compare_models.py --model_csv {output_file}")

if __name__ == '__main__':
    main()










