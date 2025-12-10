"""
Fit real SPARC galaxy rotation curve with coherence field halo.

Uses actual data from Rotmod_LTG directory.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from galaxies.rotation_curves import GalaxyRotationCurve
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


def fit_real_galaxy(galaxy_name, plot=True, savefig=None):
    """
    Fit a real galaxy rotation curve.
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name from Rotmod_LTG
    plot : bool
        Whether to plot results
    savefig : str
        Save figure filename
        
    Returns:
    --------
    result : dict
        Best-fit parameters and chi-squared
    """
    print("=" * 70)
    print(f"FITTING REAL GALAXY: {galaxy_name}")
    print("=" * 70)
    
    # Load data
    loader = RealDataLoader()
    data = loader.load_rotmod_galaxy(galaxy_name)
    
    r_obs = data['r']
    v_obs = data['v_obs']
    v_err = data['v_err']
    
    # Estimate baryonic velocity from components
    v_disk = data['v_disk']
    v_gas = data['v_gas']
    v_bulge = data['v_bulge']
    v_baryon = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
    
    # Estimate disk mass and scale from components
    # Use outermost point where baryons dominate
    G = 4.30091e-6  # (km/s)^2 kpc / M_sun
    
    # Rough estimate: M ~ v^2 * r / G at mid-radius
    mid_idx = len(r_obs) // 2
    M_disk_est = v_baryon[mid_idx]**2 * r_obs[mid_idx] / G
    R_disk_est = np.median(r_obs) / 2.5
    
    print(f"\nInitial estimates:")
    print(f"  M_disk ~ {M_disk_est:.2e} M_sun")
    print(f"  R_disk ~ {R_disk_est:.2f} kpc")
    
    # Define fitting function
    def chi_squared(params):
        """Chi-squared to minimize."""
        M_disk, R_disk, rho_c0, R_c = params
        
        if M_disk <= 0 or R_disk <= 0 or rho_c0 < 0 or R_c <= 0:
            return 1e10
        
        # Create model
        galaxy = GalaxyRotationCurve(G=G)
        galaxy.set_baryon_profile(M_disk=M_disk, R_disk=R_disk)
        
        # Set coherence halo (rho_c0 is dimensionless, convert to physical)
        rho_c0_physical = rho_c0 * M_disk / (R_disk**3)
        galaxy.set_coherence_halo_simple(rho_c0=rho_c0_physical, R_c=R_c)
        
        # Compute model
        v_model = galaxy.circular_velocity(r_obs)
        
        # Chi-squared
        chi2 = np.sum(((v_obs - v_model) / v_err)**2)
        
        return chi2
    
    # Optimize
    print(f"\nOptimizing parameters...")
    bounds = [
        (M_disk_est*0.1, M_disk_est*10),  # M_disk
        (R_disk_est*0.3, R_disk_est*3),   # R_disk
        (0.001, 50.0),                     # rho_c0 (dimensionless)
        (0.5, 50.0)                        # R_c (kpc)
    ]
    
    result = differential_evolution(chi_squared, bounds, 
                                   maxiter=200, seed=42, 
                                   workers=1, disp=True, polish=True)
    
    best_params = result.x
    chi2_min = result.fun
    dof = len(r_obs) - len(best_params)
    chi2_red = chi2_min / dof
    
    print(f"\n{'=' * 70}")
    print(f"BEST-FIT RESULTS:")
    print(f"{'=' * 70}")
    print(f"  M_disk = {best_params[0]:.2e} M_sun")
    print(f"  R_disk = {best_params[1]:.2f} kpc")
    print(f"  rho_c0 = {best_params[2]:.4f} (dimensionless)")
    print(f"  R_c    = {best_params[3]:.2f} kpc")
    print(f"\nGoodness of fit:")
    print(f"  chi^2     = {chi2_min:.2f}")
    print(f"  DOF       = {dof}")
    print(f"  chi^2_red = {chi2_red:.3f}")
    print(f"{'=' * 70}")
    
    # Create best-fit model for plotting
    M_disk, R_disk, rho_c0_dim, R_c = best_params
    rho_c0_phys = rho_c0_dim * M_disk / (R_disk**3)
    
    galaxy = GalaxyRotationCurve(G=G)
    galaxy.set_baryon_profile(M_disk=M_disk, R_disk=R_disk)
    galaxy.set_coherence_halo_simple(rho_c0=rho_c0_phys, R_c=R_c)
    
    # Compute components
    r_plot = np.linspace(r_obs[0], r_obs[-1]*1.2, 200)
    v_total = galaxy.circular_velocity(r_plot)
    
    # Baryons only
    v_baryon_plot = np.zeros_like(r_plot)
    for i, r in enumerate(r_plot):
        M_b = galaxy.mass_enclosed(r, galaxy.rho_disk)
        v_baryon_plot[i] = np.sqrt(G * M_b / r) if r > 0 else 0
    
    # Plot
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Top panel: Rotation curve
        ax = axes[0]
        ax.errorbar(r_obs, v_obs, yerr=v_err, fmt='o', color='black', 
                   label='Observed', markersize=6, capsize=3, alpha=0.7)
        ax.plot(r_plot, v_baryon_plot, '--', label='Baryons only', 
               linewidth=2, alpha=0.7, color='C1')
        ax.plot(r_plot, v_total, '-', label='Baryons + Coherence field', 
               linewidth=2.5, color='C0')
        
        ax.set_xlabel('Radius (kpc)', fontsize=12)
        ax.set_ylabel('Circular Velocity (km/s)', fontsize=12)
        ax.set_title(f'{galaxy_name} - Rotation Curve Fit', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        # Add chi-squared annotation
        ax.text(0.05, 0.95, f'$\\chi^2_{{\\rm red}}$ = {chi2_red:.2f}',
               transform=ax.transAxes, fontsize=12, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Bottom panel: Residuals
        ax = axes[1]
        v_model_obs = galaxy.circular_velocity(r_obs)
        residuals = (v_obs - v_model_obs) / v_err
        
        ax.errorbar(r_obs, residuals, yerr=np.ones_like(r_obs), fmt='o', 
                   color='black', markersize=6, capsize=3, alpha=0.7)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.axhline(2, color='gray', linestyle=':', alpha=0.3)
        ax.axhline(-2, color='gray', linestyle=':', alpha=0.3)
        
        ax.set_xlabel('Radius (kpc)', fontsize=12)
        ax.set_ylabel('Residuals (σ)', fontsize=12)
        ax.set_title('Fit Residuals', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"\nSaved figure: {savefig}")
        
        plt.show()
    
    return {
        'galaxy': galaxy_name,
        'params': best_params,
        'chi2': chi2_min,
        'chi2_reduced': chi2_red,
        'dof': dof,
        'data': data
    }


def fit_multiple_galaxies(galaxy_names, output_dir='../outputs'):
    """
    Fit multiple galaxies and create summary.
    
    Parameters:
    -----------
    galaxy_names : list
        List of galaxy names
    output_dir : str
        Directory for output plots
        
    Returns:
    --------
    results : list
        List of fit results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for galaxy_name in galaxy_names:
        try:
            savefig = os.path.join(output_dir, f'{galaxy_name}_fit.png')
            result = fit_real_galaxy(galaxy_name, plot=True, savefig=savefig)
            results.append(result)
        except Exception as e:
            print(f"\nERROR fitting {galaxy_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF FITS")
    print("=" * 70)
    print(f"{'Galaxy':<15} {'chi^2':>8} {'chi^2_red':>10} {'M_disk':>12} {'R_c':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['galaxy']:<15} {r['chi2']:>8.2f} {r['chi2_reduced']:>10.3f} "
              f"{r['params'][0]:>12.2e} {r['params'][3]:>8.2f}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    # Example: Fit a few galaxies
    print("=" * 70)
    print("COHERENCE FIELD THEORY - REAL GALAXY FITTING")
    print("=" * 70)
    
    # Get list of available galaxies
    loader = RealDataLoader()
    galaxies = loader.list_available_galaxies()
    
    # Select interesting examples (diverse sizes)
    example_galaxies = ['DDO154', 'NGC2403', 'NGC6946', 'UGC02885']
    
    # Filter to only those available
    available_examples = [g for g in example_galaxies if g in galaxies]
    
    if not available_examples:
        # Use first few available
        available_examples = galaxies[:3]
    
    print(f"\nFitting {len(available_examples)} galaxies: {available_examples}")
    
    # Fit them
    results = fit_multiple_galaxies(available_examples)
    
    print(f"\n{'=' * 70}")
    print(f"Completed {len(results)} fits")
    print(f"Check outputs/ for plots")
    print(f"{'=' * 70}")


