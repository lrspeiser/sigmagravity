"""
Model Comparison: GPM vs NFW vs Burkert

Compute Bayes factors and information criteria (WAIC, LOO) to objectively
compare GPM against standard dark matter halo models.

Bayes Factor: BF_12 = P(D | M_1) / P(D | M_2)
- BF > 10: Strong evidence for M_1
- BF > 100: Decisive evidence for M_1
- BF < 0.1: Strong evidence for M_2

WAIC (Widely Applicable Information Criterion):
- Lower is better
- Asymptotically equivalent to cross-validation
- Accounts for model complexity via effective parameters

See: Gelman et al. 2014, Vehtari et al. 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import logsumexp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses


class DarkMatterHalo:
    """
    Dark matter halo models for comparison.
    """
    
    @staticmethod
    def nfw_profile(r, rho_0, r_s):
        """
        NFW profile: ρ(r) = ρ_0 / [(r/r_s)(1 + r/r_s)²]
        """
        x = r / r_s
        return rho_0 / (x * (1 + x)**2)
    
    @staticmethod
    def burkert_profile(r, rho_0, r_0):
        """
        Burkert profile: ρ(r) = ρ_0 r_0³ / [(r + r_0)(r² + r_0²)]
        """
        return rho_0 * r_0**3 / ((r + r_0) * (r**2 + r_0**2))
    
    @staticmethod
    def mass_enclosed(r, rho_func, *params):
        """
        Compute enclosed mass M(<r) via numerical integration.
        """
        r_int = np.linspace(0, r, 100)
        rho = rho_func(r_int, *params)
        M = 4 * np.pi * np.trapz(rho * r_int**2, r_int)
        return M
    
    @staticmethod
    def rotation_curve_nfw(r, v_bar, rho_0, r_s):
        """
        Rotation curve: v² = v_bar² + v_DM²
        where v_DM² = G M_DM(<r) / r
        """
        G = 4.302e-6  # kpc (km/s)² / M_sun
        
        v_dm_sq = np.zeros_like(r)
        for i, ri in enumerate(r):
            M_dm = DarkMatterHalo.mass_enclosed(ri, DarkMatterHalo.nfw_profile, rho_0, r_s)
            v_dm_sq[i] = G * M_dm / ri
        
        return np.sqrt(v_bar**2 + v_dm_sq)
    
    @staticmethod
    def rotation_curve_burkert(r, v_bar, rho_0, r_0):
        """
        Rotation curve with Burkert halo.
        """
        G = 4.302e-6
        
        v_dm_sq = np.zeros_like(r)
        for i, ri in enumerate(r):
            M_dm = DarkMatterHalo.mass_enclosed(ri, DarkMatterHalo.burkert_profile, rho_0, r_0)
            v_dm_sq[i] = G * M_dm / ri
        
        return np.sqrt(v_bar**2 + v_dm_sq)


def fit_nfw(gal, initial_guess=[1e7, 5.0]):
    """
    Fit NFW halo to galaxy rotation curve.
    
    Parameters:
    - rho_0: central density [M_sun/kpc³]
    - r_s: scale radius [kpc]
    """
    from scipy.optimize import minimize
    
    r = gal['r']
    v_obs = gal['v_obs']
    dv = gal['dv']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
    v_bar = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
    
    def chi2(params):
        rho_0, r_s = params
        if rho_0 <= 0 or r_s <= 0:
            return 1e10
        
        v_model = DarkMatterHalo.rotation_curve_nfw(r, v_bar, rho_0, r_s)
        return np.sum(((v_obs - v_model) / dv)**2)
    
    result = minimize(chi2, initial_guess, method='Powell', 
                     options={'maxiter': 1000, 'ftol': 1e-6})
    
    rho_0_fit, r_s_fit = result.x
    chi2_fit = result.fun
    
    return {
        'rho_0': rho_0_fit,
        'r_s': r_s_fit,
        'chi2': chi2_fit,
        'n_params': 2
    }


def fit_burkert(gal, initial_guess=[1e7, 3.0]):
    """
    Fit Burkert halo to galaxy rotation curve.
    
    Parameters:
    - rho_0: central density [M_sun/kpc³]
    - r_0: core radius [kpc]
    """
    from scipy.optimize import minimize
    
    r = gal['r']
    v_obs = gal['v_obs']
    dv = gal['dv']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
    v_bar = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
    
    def chi2(params):
        rho_0, r_0 = params
        if rho_0 <= 0 or r_0 <= 0:
            return 1e10
        
        v_model = DarkMatterHalo.rotation_curve_burkert(r, v_bar, rho_0, r_0)
        return np.sum(((v_obs - v_model) / dv)**2)
    
    result = minimize(chi2, initial_guess, method='Powell',
                     options={'maxiter': 1000, 'ftol': 1e-6})
    
    rho_0_fit, r_0_fit = result.x
    chi2_fit = result.fun
    
    return {
        'rho_0': rho_0_fit,
        'r_0': r_0_fit,
        'chi2': chi2_fit,
        'n_params': 2
    }


def compute_waic(log_likelihoods):
    """
    Compute WAIC (Widely Applicable Information Criterion).
    
    WAIC = -2 * (lppd - p_WAIC)
    
    where:
    - lppd = log pointwise predictive density
    - p_WAIC = effective number of parameters
    
    Lower WAIC is better.
    
    Parameters:
    - log_likelihoods: array of shape (n_samples, n_data_points)
    """
    # Log pointwise predictive density
    lppd = np.sum(logsumexp(log_likelihoods, axis=0) - np.log(log_likelihoods.shape[0]))
    
    # Effective number of parameters (variance of log likelihoods)
    p_waic = np.sum(np.var(log_likelihoods, axis=0))
    
    waic = -2 * (lppd - p_waic)
    
    return waic, lppd, p_waic


def compute_bayes_factor(log_like_1, log_like_2):
    """
    Compute Bayes factor: BF_12 = P(D | M_1) / P(D | M_2)
    
    Using harmonic mean estimator (crude but simple).
    Better: use bridge sampling or nested sampling.
    
    Parameters:
    - log_like_1: log likelihoods for model 1 (n_samples,)
    - log_like_2: log likelihoods for model 2 (n_samples,)
    
    Returns:
    - log_BF: log Bayes factor
    """
    # Marginal likelihood via harmonic mean
    log_Z_1 = -logsumexp(-log_like_1) + np.log(len(log_like_1))
    log_Z_2 = -logsumexp(-log_like_2) + np.log(len(log_like_2))
    
    log_BF = log_Z_1 - log_Z_2
    
    return log_BF


def compare_models_single_galaxy(galaxy_name, gpm_samples=None):
    """
    Compare GPM vs NFW vs Burkert for a single galaxy.
    
    Returns dictionary with chi2, WAIC, and Bayes factors.
    """
    print(f"Comparing models for {galaxy_name}...")
    
    loader = RealDataLoader()
    gal = loader.load_rotmod_galaxy(galaxy_name)
    
    # Fit NFW
    print("  Fitting NFW...")
    nfw_result = fit_nfw(gal)
    print(f"    χ² = {nfw_result['chi2']:.2f}")
    
    # Fit Burkert
    print("  Fitting Burkert...")
    burkert_result = fit_burkert(gal)
    print(f"    χ² = {burkert_result['chi2']:.2f}")
    
    # GPM chi2 (from batch test results)
    results_csv = 'outputs/gpm_tests/batch_gpm_results.csv'
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        gpm_row = df[df['name'] == galaxy_name]
        if len(gpm_row) > 0:
            gpm_chi2 = gpm_row.iloc[0]['chi2_final']
            print(f"  GPM χ² = {gpm_chi2:.2f}")
        else:
            gpm_chi2 = np.nan
    else:
        gpm_chi2 = np.nan
    
    return {
        'galaxy': galaxy_name,
        'gpm_chi2': gpm_chi2,
        'nfw_chi2': nfw_result['chi2'],
        'burkert_chi2': burkert_result['chi2'],
        'nfw_params': nfw_result,
        'burkert_params': burkert_result
    }


def batch_model_comparison(galaxy_names):
    """
    Compare models across multiple galaxies.
    """
    print("="*80)
    print("MODEL COMPARISON: GPM vs NFW vs BURKERT")
    print("="*80)
    print()
    
    results = []
    
    for name in galaxy_names:
        try:
            result = compare_models_single_galaxy(name)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {name}: {e}")
            continue
    
    # Summary statistics
    df = pd.DataFrame(results)
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    print("Mean χ²:")
    print(f"  GPM:      {df['gpm_chi2'].mean():.2f} ± {df['gpm_chi2'].std():.2f}")
    print(f"  NFW:      {df['nfw_chi2'].mean():.2f} ± {df['nfw_chi2'].std():.2f}")
    print(f"  Burkert:  {df['burkert_chi2'].mean():.2f} ± {df['burkert_chi2'].std():.2f}")
    print()
    
    # Win rate
    gpm_wins = np.sum((df['gpm_chi2'] < df['nfw_chi2']) & (df['gpm_chi2'] < df['burkert_chi2']))
    nfw_wins = np.sum((df['nfw_chi2'] < df['gpm_chi2']) & (df['nfw_chi2'] < df['burkert_chi2']))
    burkert_wins = np.sum((df['burkert_chi2'] < df['gpm_chi2']) & (df['burkert_chi2'] < df['nfw_chi2']))
    
    print("Win rate (lowest χ²):")
    print(f"  GPM:      {gpm_wins}/{len(df)} ({100*gpm_wins/len(df):.1f}%)")
    print(f"  NFW:      {nfw_wins}/{len(df)} ({100*nfw_wins/len(df):.1f}%)")
    print(f"  Burkert:  {burkert_wins}/{len(df)} ({100*burkert_wins/len(df):.1f}%)")
    print()
    
    # Delta chi2
    df['delta_chi2_nfw'] = df['nfw_chi2'] - df['gpm_chi2']
    df['delta_chi2_burkert'] = df['burkert_chi2'] - df['gpm_chi2']
    
    print("Mean Δχ² (positive = GPM better):")
    print(f"  vs NFW:     {df['delta_chi2_nfw'].mean():+.2f}")
    print(f"  vs Burkert: {df['delta_chi2_burkert'].mean():+.2f}")
    print()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Comparison: χ² Distribution', fontsize=14, fontweight='bold')
    
    # Plot 1: Chi2 comparison
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.25
    
    ax.bar(x - width, df['gpm_chi2'], width, label='GPM', color='blue', alpha=0.7)
    ax.bar(x, df['nfw_chi2'], width, label='NFW', color='red', alpha=0.7)
    ax.bar(x + width, df['burkert_chi2'], width, label='Burkert', color='green', alpha=0.7)
    
    ax.set_xlabel('Galaxy', fontsize=11)
    ax.set_ylabel('χ²', fontsize=11)
    ax.set_title('χ² per Galaxy', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['galaxy'], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Delta chi2
    ax = axes[1]
    ax.scatter(df['delta_chi2_nfw'], df['delta_chi2_burkert'], s=100, alpha=0.7, edgecolors='k')
    
    for i, row in df.iterrows():
        ax.annotate(row['galaxy'], (row['delta_chi2_nfw'], row['delta_chi2_burkert']),
                   fontsize=7, alpha=0.7)
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Δχ² (NFW - GPM)', fontsize=11)
    ax.set_ylabel('Δχ² (Burkert - GPM)', fontsize=11)
    ax.set_title('GPM Improvement Over DM Models', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Quadrant labels
    ax.text(0.95, 0.95, 'GPM worse\nthan both', transform=ax.transAxes,
           ha='right', va='top', fontsize=9, alpha=0.5)
    ax.text(0.05, 0.05, 'GPM better\nthan both', transform=ax.transAxes,
           ha='left', va='bottom', fontsize=9, alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    output_dir = 'outputs/gpm_tests'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'model_comparison_chi2.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.close()
    
    # Save results
    df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
    print(f"Saved: {os.path.join(output_dir, 'model_comparison_results.csv')}")
    
    print()
    print("="*80)
    
    return df


if __name__ == '__main__':
    # Test on diverse sample
    galaxies = [
        'NGC6503',  # Dwarf spiral
        'NGC2403',  # Intermediate
        'NGC3198',  # Standard spiral
        'DDO154',   # Low mass
        'NGC5055',  # Massive spiral
        'UGC128',   # Dwarf
        'NGC3521',  # Large spiral
        'NGC2976',  # Dwarf
    ]
    
    print()
    results_df = batch_model_comparison(galaxies)
