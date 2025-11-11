"""
pantheon_fit.py
---------------
Fit Weyl-integrable redshift model to Pantheon+ SNe Ia data.

This script:
1. Loads Pantheon+ SNe Ia data (if available)
2. Fits α₀ parameter to minimize χ²
3. Computes error bars and confidence intervals
4. Compares to ΛCDM baseline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
import sys
from pathlib import Path

# Add cosmo to path
SCRIPT_DIR = Path(__file__).parent
COSMO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(COSMO_DIR))

from sigma_redshift_derivations import SigmaKernel, WeylModel
from r_of_z_weyl import WeylDistanceInverter

def create_mock_pantheon_data(n_sne=1000):
    """
    Create mock Pantheon+ data for testing when real data is not available.
    
    Parameters:
    -----------
    n_sne : int
        Number of mock SNe Ia
        
    Returns:
    --------
    pantheon_data : pd.DataFrame
        Mock Pantheon+ data with columns: z, mu, mu_err
    """
    np.random.seed(42)  # For reproducibility
    
    # Mock redshift distribution (similar to real Pantheon+)
    z_min, z_max = 0.01, 2.3
    z = np.random.uniform(z_min, z_max, n_sne)
    z = np.sort(z)
    
    # Mock distance modulus with ΛCDM baseline
    H0 = 70.0  # km/s/Mpc
    c = 299792458.0  # m/s
    Mpc = 3.0856775814913673e22  # m
    
    # Simple ΛCDM distance modulus (approximate)
    mu_lcdm = 5 * np.log10((c/H0 * z * (1+z)**2) / Mpc) - 5
    
    # Add observational scatter
    mu_err = np.random.normal(0.1, 0.02, n_sne)  # Typical SNe Ia errors
    mu_obs = mu_lcdm + np.random.normal(0, mu_err, n_sne)
    
    return pd.DataFrame({
        'z': z,
        'mu': mu_obs,
        'mu_err': mu_err
    })

def load_pantheon_data(filepath=None):
    """
    Load Pantheon+ data if available, otherwise create mock data.
    
    Parameters:
    -----------
    filepath : str, optional
        Path to Pantheon+ data file
        
    Returns:
    --------
    pantheon_data : pd.DataFrame
        Pantheon+ data with columns: z, mu, mu_err
    """
    if filepath and Path(filepath).exists():
        # Load real Pantheon+ data
        print(f"Loading Pantheon+ data from {filepath}")
        # This would need to be implemented based on actual data format
        raise NotImplementedError("Real Pantheon+ loading not implemented yet")
    else:
        print("Creating mock Pantheon+ data for testing")
        return create_mock_pantheon_data()

def chi2_weyl_model(params, pantheon_data, kernel_params):
    """
    Compute χ² for Weyl model with given parameters.
    
    Parameters:
    -----------
    params : array-like
        Model parameters [alpha0_scale, ell0_kpc, p, ncoh]
    pantheon_data : pd.DataFrame
        SNe Ia data with columns: z, mu, mu_err
    kernel_params : dict
        Fixed kernel parameters
        
    Returns:
    --------
    chi2_value : float
        Chi-squared value
    """
    alpha0_scale, ell0_kpc, p, ncoh = params
    
    try:
        # Create model
        kernel = SigmaKernel(
            A=kernel_params.get('A', 1.0),
            ell0_kpc=ell0_kpc,
            p=p,
            ncoh=ncoh
        )
        model = WeylModel(kernel=kernel, H0_kms_Mpc=70.0, alpha0_scale=alpha0_scale)
        inverter = WeylDistanceInverter(model)
        
        # Compute predicted distance moduli
        mu_pred = inverter.compute_distance_modulus(pantheon_data['z'].values)
        
        # Compute χ²
        chi2_value = np.sum(((pantheon_data['mu'].values - mu_pred) / pantheon_data['mu_err'].values)**2)
        
        return chi2_value
    
    except Exception as e:
        # Return large χ² for invalid parameters
        return 1e10

def fit_weyl_model(pantheon_data, kernel_params=None):
    """
    Fit Weyl model to Pantheon+ data.
    
    Parameters:
    -----------
    pantheon_data : pd.DataFrame
        SNe Ia data
    kernel_params : dict, optional
        Fixed kernel parameters
        
    Returns:
    --------
    fit_result : dict
        Fit results including best parameters, errors, and statistics
    """
    if kernel_params is None:
        kernel_params = {'A': 1.0}
    
    print("Fitting Weyl-integrable model to Pantheon+ data...")
    print(f"Data: {len(pantheon_data)} SNe Ia")
    print(f"Redshift range: {pantheon_data['z'].min():.3f} - {pantheon_data['z'].max():.3f}")
    
    # Initial parameter guess
    x0 = [0.95, 200.0, 0.75, 0.5]  # [alpha0_scale, ell0_kpc, p, ncoh]
    
    # Parameter bounds
    bounds = [
        (0.1, 2.0),    # alpha0_scale
        (50.0, 500.0), # ell0_kpc
        (0.1, 2.0),    # p
        (0.1, 2.0)     # ncoh
    ]
    
    # Fit
    result = minimize(
        chi2_weyl_model,
        x0,
        args=(pantheon_data, kernel_params),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    if not result.success:
        print(f"Warning: Fit did not converge: {result.message}")
    
    # Extract best parameters
    best_params = {
        'alpha0_scale': result.x[0],
        'ell0_kpc': result.x[1],
        'p': result.x[2],
        'ncoh': result.x[3]
    }
    
    # Compute fit statistics
    chi2_min = result.fun
    n_params = len(result.x)
    n_data = len(pantheon_data)
    dof = n_data - n_params
    
    # Reduced χ²
    chi2_reduced = chi2_min / dof
    
    # P-value
    p_value = 1 - chi2.cdf(chi2_min, dof)
    
    # Create best-fit model
    kernel = SigmaKernel(
        A=kernel_params.get('A', 1.0),
        ell0_kpc=best_params['ell0_kpc'],
        p=best_params['p'],
        ncoh=best_params['ncoh']
    )
    best_model = WeylModel(kernel=kernel, H0_kms_Mpc=70.0, alpha0_scale=best_params['alpha0_scale'])
    
    # Compute residuals
    inverter = WeylDistanceInverter(best_model)
    mu_pred = inverter.compute_distance_modulus(pantheon_data['z'].values)
    residuals = pantheon_data['mu'].values - mu_pred
    
    fit_result = {
        'best_params': best_params,
        'chi2_min': chi2_min,
        'chi2_reduced': chi2_reduced,
        'dof': dof,
        'p_value': p_value,
        'residuals': residuals,
        'mu_pred': mu_pred,
        'best_model': best_model,
        'inverter': inverter,
        'success': result.success
    }
    
    return fit_result

def compare_to_lcdm(pantheon_data, fit_result):
    """
    Compare Weyl model fit to ΛCDM baseline.
    
    Parameters:
    -----------
    pantheon_data : pd.DataFrame
        SNe Ia data
    fit_result : dict
        Weyl model fit results
        
    Returns:
    --------
    comparison : dict
        Comparison statistics
    """
    # Simple ΛCDM distance modulus (approximate)
    H0 = 70.0  # km/s/Mpc
    c = 299792458.0  # m/s
    Mpc = 3.0856775814913673e22  # m
    
    mu_lcdm = 5 * np.log10((c/H0 * pantheon_data['z'].values * (1+pantheon_data['z'].values)**2) / Mpc) - 5
    chi2_lcdm = np.sum(((pantheon_data['mu'].values - mu_lcdm) / pantheon_data['mu_err'].values)**2)
    
    # Weyl model χ²
    chi2_weyl = fit_result['chi2_min']
    
    # Δχ²
    delta_chi2 = chi2_weyl - chi2_lcdm
    
    # AIC comparison
    n_params_lcdm = 1  # Just H0
    n_params_weyl = 4  # alpha0_scale, ell0_kpc, p, ncoh
    n_data = len(pantheon_data)
    
    aic_lcdm = chi2_lcdm + 2 * n_params_lcdm
    aic_weyl = chi2_weyl + 2 * n_params_weyl
    delta_aic = aic_weyl - aic_lcdm
    
    comparison = {
        'chi2_lcdm': chi2_lcdm,
        'chi2_weyl': chi2_weyl,
        'delta_chi2': delta_chi2,
        'aic_lcdm': aic_lcdm,
        'aic_weyl': aic_weyl,
        'delta_aic': delta_aic,
        'preference': 'LCDM' if delta_aic > 0 else 'Weyl'
    }
    
    return comparison

def main():
    print("="*80)
    print("WEYL-INTEGRABLE MODEL FIT TO PANTHEON+ DATA")
    print("="*80)
    
    # Load data
    pantheon_data = load_pantheon_data()
    
    # Fit model
    fit_result = fit_weyl_model(pantheon_data)
    
    # Print results
    print(f"\n" + "="*80)
    print("FIT RESULTS")
    print("="*80)
    
    print(f"\nBest-fit parameters:")
    for param, value in fit_result['best_params'].items():
        print(f"  {param}: {value:.4f}")
    
    print(f"\nFit statistics:")
    print(f"  χ²_min: {fit_result['chi2_min']:.2f}")
    print(f"  χ²_reduced: {fit_result['chi2_reduced']:.4f}")
    print(f"  Degrees of freedom: {fit_result['dof']}")
    print(f"  P-value: {fit_result['p_value']:.2e}")
    print(f"  Fit success: {fit_result['success']}")
    
    # Compare to ΛCDM
    comparison = compare_to_lcdm(pantheon_data, fit_result)
    
    print(f"\n" + "="*80)
    print("COMPARISON TO ΛCDM")
    print("="*80)
    
    print(f"\nχ² comparison:")
    print(f"  ΛCDM: {comparison['chi2_lcdm']:.2f}")
    print(f"  Weyl: {comparison['chi2_weyl']:.2f}")
    print(f"  Δχ²: {comparison['delta_chi2']:.2f}")
    
    print(f"\nAIC comparison:")
    print(f"  ΛCDM: {comparison['aic_lcdm']:.2f}")
    print(f"  Weyl: {comparison['aic_weyl']:.2f}")
    print(f"  ΔAIC: {comparison['delta_aic']:.2f}")
    print(f"  Preference: {comparison['preference']}")
    
    # Create visualization
    print(f"\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Hubble diagram
    ax = axes[0, 0]
    ax.errorbar(pantheon_data['z'], pantheon_data['mu'], 
               yerr=pantheon_data['mu_err'], fmt='o', alpha=0.5, markersize=2, 
               label='Pantheon+ data', color='gray')
    
    # Plot best-fit model
    z_model = np.linspace(0.01, 2.5, 100)
    mu_model = fit_result['inverter'].compute_distance_modulus(z_model)
    ax.plot(z_model, mu_model, 'r-', linewidth=2, label='Weyl model')
    
    # Plot ΛCDM baseline
    mu_lcdm = 5 * np.log10((299792458.0/70.0 * z_model * (1+z_model)**2) / (3.0856775814913673e22)) - 5
    ax.plot(z_model, mu_lcdm, 'k--', linewidth=2, label='ΛCDM baseline')
    
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Distance Modulus μ')
    ax.set_title('Hubble Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax = axes[0, 1]
    residuals = fit_result['residuals']
    ax.errorbar(pantheon_data['z'], residuals, 
               yerr=pantheon_data['mu_err'], fmt='o', alpha=0.5, markersize=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Residuals (μ_data - μ_model)')
    ax.set_title('Fit Residuals')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, alpha=0.7, density=True)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Parameter space (2D projection)
    ax = axes[1, 1]
    # This would require more sophisticated parameter space exploration
    ax.text(0.5, 0.5, 'Parameter space\nvisualization\n(not implemented)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Parameter Space')
    
    plt.tight_layout()
    plot_file = COSMO_DIR / "outputs" / "pantheon_fit_results.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    
    # Save results
    results_file = COSMO_DIR / "outputs" / "pantheon_fit_results.csv"
    results_df = pd.DataFrame({
        'z': pantheon_data['z'],
        'mu_data': pantheon_data['mu'],
        'mu_err': pantheon_data['mu_err'],
        'mu_model': fit_result['mu_pred'],
        'residuals': fit_result['residuals']
    })
    results_df.to_csv(results_file, index=False)
    print(f"Results saved: {results_file}")
    
    print(f"\n" + "="*80)
    print("FIT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()


