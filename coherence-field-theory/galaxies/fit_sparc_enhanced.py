"""
Enhanced SPARC galaxy rotation curve fitting with coherence field halos.

Features:
- Uses real Rotmod_LTG data (175 galaxies)
- Improved parameter bounds and priors
- NFW dark matter baseline comparison
- Chi-squared comparison: coherence vs DM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from galaxies.rotation_curves import GalaxyRotationCurve
from data_integration.load_real_data import RealDataLoader


class EnhancedSPARCFitter:
    """
    Enhanced SPARC fitter with real data and NFW comparison.
    """
    
    def __init__(self):
        """Initialize fitter."""
        self.G = 4.30091e-6  # (km/s)^2 kpc / M_sun
        self.data_loader = RealDataLoader()
        
    def load_galaxy(self, galaxy_name):
        """
        Load galaxy from Rotmod_LTG.
        
        Parameters:
        -----------
        galaxy_name : str
            Galaxy name (e.g., 'DDO154', 'NGC2403')
            
        Returns:
        --------
        data : dict
            Galaxy data dictionary
        """
        try:
            data = self.data_loader.load_rotmod_galaxy(galaxy_name)
            
            # Compute baryonic velocity from components
            v_disk = data['v_disk']
            v_gas = data['v_gas']
            v_bulge = data['v_bulge']
            v_baryon = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
            
            return {
                'name': galaxy_name,
                'r': data['r'],
                'v_obs': data['v_obs'],
                'v_err': data['v_err'],
                'v_disk': v_disk,
                'v_gas': v_gas,
                'v_bulge': v_bulge,
                'v_baryon': v_baryon,
                'distance_Mpc': data.get('distance_Mpc', None)
            }
        except Exception as e:
            print(f"Error loading {galaxy_name}: {e}")
            raise
    
    def fit_coherence_halo(self, data, method='global', bounds_mult=1.0):
        """
        Fit coherence halo with improved bounds.
        
        Parameters:
        -----------
        data : dict
            Galaxy data
        method : str
            'local' or 'global'
        bounds_mult : float
            Multiplier for parameter bounds (1.0 = standard)
            
        Returns:
        --------
        result : dict
            Best-fit parameters and diagnostics
        """
        print(f"\n{'=' * 70}")
        print(f"FITTING COHERENCE HALO: {data['name']}")
        print(f"{'=' * 70}")
        
        r_obs = data['r']
        v_obs = data['v_obs']
        v_err = data['v_err']
        v_baryon = data['v_baryon']
        
        # Estimate disk mass and scale from baryonic velocity
        mid_idx = len(r_obs) // 2
        M_disk_est = v_baryon[mid_idx]**2 * r_obs[mid_idx] / self.G
        R_disk_est = np.median(r_obs) / 2.5
        
        print(f"\nEstimated disk parameters:")
        print(f"  M_disk ~ {M_disk_est:.2e} M_sun")
        print(f"  R_disk ~ {R_disk_est:.2f} kpc")
        
        def chi_squared(params):
            """Chi-squared objective function."""
            M_disk, R_disk, rho_c0, R_c = params
            
            # Penalties for unphysical values
            if M_disk <= 0 or R_disk <= 0 or rho_c0 < 0 or R_c <= 0:
                return 1e10
            
            # Prevent extreme values
            if R_c > 50.0 or R_c < 0.1:  # Reasonable halo sizes
                return 1e10
            if rho_c0 > 100.0:  # Reasonable density
                return 1e10
            
            # Create model
            galaxy = GalaxyRotationCurve(G=self.G)
            galaxy.set_baryon_profile(M_disk=M_disk, R_disk=R_disk)
            
            # Set coherence halo
            rho_c0_physical = rho_c0 * M_disk / (R_disk**3)
            galaxy.set_coherence_halo_simple(rho_c0=rho_c0_physical, R_c=R_c)
            
            # Compute model velocities
            v_model = galaxy.circular_velocity(r_obs)
            
            # Chi-squared
            chi2 = np.sum(((v_obs - v_model) / v_err)**2)
            
            return chi2
        
        # Improved bounds with reasonable priors
        bounds = [
            (M_disk_est*0.1 * bounds_mult, M_disk_est*10 * bounds_mult),  # M_disk
            (R_disk_est*0.3 * bounds_mult, R_disk_est*3 * bounds_mult),   # R_disk
            (0.01, 50.0),  # rho_c0 (dimensionless) - tighter than before
            (0.5, 50.0)    # R_c (kpc) - reasonable halo sizes
        ]
        
        # Initial guess
        p0 = [M_disk_est, R_disk_est, 0.5, 5.0]
        
        if method == 'local':
            result = minimize(chi_squared, p0, method='L-BFGS-B', bounds=bounds)
            best_params = result.x
            chi2_min = result.fun
        else:
            result = differential_evolution(chi_squared, bounds, 
                                          maxiter=200, seed=42, 
                                          workers=1, disp=False, polish=True)
            best_params = result.x
            chi2_min = result.fun
        
        # Compute reduced chi-squared
        dof = len(r_obs) - len(best_params)
        chi2_reduced = chi2_min / dof if dof > 0 else chi2_min
        
        print(f"\nBest-fit parameters:")
        print(f"  M_disk = {best_params[0]:.2e} M_sun")
        print(f"  R_disk = {best_params[1]:.2f} kpc")
        print(f"  rho_c0 = {best_params[2]:.4f} (dimensionless)")
        print(f"  R_c    = {best_params[3]:.2f} kpc")
        print(f"\nGoodness of fit:")
        print(f"  chi^2 = {chi2_min:.2f}")
        print(f"  DOF = {dof}")
        print(f"  chi^2_red = {chi2_reduced:.3f}")
        
        return {
            'params': best_params,
            'chi2': chi2_min,
            'chi2_reduced': chi2_reduced,
            'dof': dof,
            'data': data
        }
    
    def fit_NFW_dark_matter(self, data, method='global'):
        """
        Fit NFW dark matter halo as baseline.
        
        Parameters:
        -----------
        data : dict
            Galaxy data
        method : str
            'local' or 'global'
            
        Returns:
        --------
        result : dict
            Best-fit parameters and diagnostics
        """
        print(f"\n{'=' * 70}")
        print(f"FITTING NFW DARK MATTER: {data['name']}")
        print(f"{'=' * 70}")
        
        r_obs = data['r']
        v_obs = data['v_obs']
        v_err = data['v_err']
        v_baryon = data['v_baryon']
        
        # Estimate disk mass
        mid_idx = len(r_obs) // 2
        M_disk_est = v_baryon[mid_idx]**2 * r_obs[mid_idx] / self.G
        R_disk_est = np.median(r_obs) / 2.5
        
        def chi_squared(params):
            """Chi-squared for NFW + baryons."""
            M_disk, R_disk, M200, c = params
            
            if M_disk <= 0 or R_disk <= 0 or M200 <= 0 or c <= 0:
                return 1e10
            
            # Baryonic component (direct computation)
            rho_disk_0 = M_disk / (4 * np.pi * R_disk**3)
            v_baryon_model = np.zeros_like(r_obs)
            for i, r in enumerate(r_obs):
                if r > 0:
                    # Enclosed mass for exponential disk
                    M_enc = M_disk * (1 - np.exp(-r / R_disk) * (1 + r / R_disk))
                    v_baryon_model[i] = np.sqrt(self.G * M_enc / r)
            
            # NFW halo
            # Use virial radius estimate: r_200 = (3*M200/(4*pi*rho_crit))^(1/3)
            # For typical galaxy: rho_crit ~ 140 M_sun/kpc^3
            # Simplified: use optical radius approximation
            r_max = np.max(r_obs) * 2.0  # Use 2× max radius as virial estimate
            r_s = r_max / c  # More realistic r_s estimate
            rho_s = M200 / (4 * np.pi * r_s**3 * (np.log(1 + c) - c / (1 + c))) if c > 0 else 0
            
            v_NFW = np.zeros_like(r_obs)
            for i, r in enumerate(r_obs):
                x = r / r_s
                if x < 0.01:
                    v_NFW[i] = 0.0
                else:
                    # NFW mass enclosed: M(<r) = 4πρ_s r_s³ [ln(1+x) - x/(1+x)]
                    M_enc = 4 * np.pi * rho_s * r_s**3 * (np.log(1 + x) - x / (1 + x))
                    v_NFW[i] = np.sqrt(self.G * M_enc / r)
            
            # Total velocity
            v_model = np.sqrt(v_baryon_model**2 + v_NFW**2)
            
            # Chi-squared
            chi2 = np.sum(((v_obs - v_model) / v_err)**2)
            
            return chi2
        
        # Bounds for NFW fit (fair comparison with coherence)
        bounds = [
            (M_disk_est*0.1, M_disk_est*10),  # M_disk
            (R_disk_est*0.3, R_disk_est*3),   # R_disk
            (M_disk_est*0.1, M_disk_est*100), # M200 (halo mass)
            (1.0, 30.0)                       # c (concentration) - wider range
        ]
        
        p0 = [M_disk_est, R_disk_est, M_disk_est*10, 10.0]
        
        if method == 'local':
            result = minimize(chi_squared, p0, method='L-BFGS-B', bounds=bounds)
            best_params = result.x
            chi2_min = result.fun
        else:
            result = differential_evolution(chi_squared, bounds, 
                                          maxiter=200, seed=42, 
                                          workers=1, disp=False, polish=True)
            best_params = result.x
            chi2_min = result.fun
        
        dof = len(r_obs) - len(best_params)
        chi2_reduced = chi2_min / dof if dof > 0 else chi2_min
        
        print(f"\nBest-fit parameters:")
        print(f"  M_disk = {best_params[0]:.2e} M_sun")
        print(f"  R_disk = {best_params[1]:.2f} kpc")
        print(f"  M200   = {best_params[2]:.2e} M_sun")
        print(f"  c      = {best_params[3]:.2f}")
        print(f"\nGoodness of fit:")
        print(f"  chi^2 = {chi2_min:.2f}")
        print(f"  DOF = {dof}")
        print(f"  chi^2_red = {chi2_reduced:.3f}")
        
        return {
            'params': best_params,
            'chi2': chi2_min,
            'chi2_reduced': chi2_reduced,
            'dof': dof,
            'data': data,
            'model': 'NFW'
        }
    
    def compare_fits(self, coherence_result, nfw_result, savefig=None):
        """
        Compare coherence vs NFW fits.
        
        Parameters:
        -----------
        coherence_result : dict
            Coherence fit result
        nfw_result : dict
            NFW fit result
        savefig : str
            Save figure filename
        """
        data = coherence_result['data']
        r_obs = data['r']
        v_obs = data['v_obs']
        v_err = data['v_err']
        v_baryon = data['v_baryon']
        
        # Coherence model
        params_co = coherence_result['params']
        galaxy_co = GalaxyRotationCurve(G=self.G)
        galaxy_co.set_baryon_profile(M_disk=params_co[0], R_disk=params_co[1])
        rho_c0_phys = params_co[2] * params_co[0] / (params_co[1]**3)
        galaxy_co.set_coherence_halo_simple(rho_c0=rho_c0_phys, R_c=params_co[3])
        r_plot = np.linspace(r_obs[0], r_obs[-1]*1.2, 200)
        v_co = galaxy_co.circular_velocity(r_plot)
        
        # Baryons only for coherence
        v_baryon_co = np.zeros_like(r_plot)
        for i, r in enumerate(r_plot):
            M_b = galaxy_co.mass_enclosed(r, galaxy_co.rho_disk)
            v_baryon_co[i] = np.sqrt(self.G * M_b / r) if r > 0 else 0
        
        # NFW model
        params_nfw = nfw_result['params']
        M_disk_nfw = params_nfw[0]
        R_disk_nfw = params_nfw[1]
        
        # Baryonic velocity (direct computation)
        v_baryon_nfw = np.zeros_like(r_plot)
        for i, r in enumerate(r_plot):
            if r > 0:
                M_enc = M_disk_nfw * (1 - np.exp(-r / R_disk_nfw) * (1 + r / R_disk_nfw))
                v_baryon_nfw[i] = np.sqrt(self.G * M_enc / r)
        
        # NFW halo
        M200 = params_nfw[2]
        c = params_nfw[3]
        r_max = np.max(r_plot) * 2.0  # Use 2× max radius as virial estimate
        r_s = r_max / c
        rho_s = M200 / (4 * np.pi * r_s**3 * (np.log(1 + c) - c / (1 + c))) if c > 0 else 0
        
        v_NFW = np.zeros_like(r_plot)
        for i, r in enumerate(r_plot):
            x = r / r_s
            if x > 0.01:
                M_enc = 4 * np.pi * rho_s * r_s**3 * (np.log(1 + x) - x / (1 + x))
                v_NFW[i] = np.sqrt(self.G * M_enc / r)
        
        v_nfw = np.sqrt(v_baryon_nfw**2 + v_NFW**2)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top: Rotation curves
        ax = axes[0]
        ax.errorbar(r_obs, v_obs, yerr=v_err, fmt='o', color='black', 
                   label='Observed', markersize=6, capsize=3, alpha=0.7, zorder=10)
        ax.plot(r_plot, v_baryon_co, '--', label='Baryons only', 
               linewidth=2, alpha=0.6, color='C1')
        ax.plot(r_plot, v_co, '-', label=f'Coherence (chi^2_red={coherence_result["chi2_reduced"]:.2f})', 
               linewidth=2.5, color='C0')
        ax.plot(r_plot, v_nfw, '--', label=f'NFW DM (chi^2_red={nfw_result["chi2_reduced"]:.2f})', 
               linewidth=2.5, color='C2', alpha=0.7)
        
        ax.set_xlabel('Radius (kpc)', fontsize=12)
        ax.set_ylabel('Circular Velocity (km/s)', fontsize=12)
        ax.set_title(f'{data["name"]} - Coherence vs NFW Dark Matter', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3)
        
        # Bottom: Residuals
        ax = axes[1]
        v_co_obs = galaxy_co.circular_velocity(r_obs)
        v_nfw_obs = np.sqrt(v_baryon_nfw[r_plot <= r_obs[-1]]**2 + 
                          v_NFW[r_plot <= r_obs[-1]]**2)
        v_nfw_obs = np.interp(r_obs, r_plot, v_nfw)
        
        residuals_co = (v_obs - v_co_obs) / v_err
        residuals_nfw = (v_obs - v_nfw_obs) / v_err
        
        ax.errorbar(r_obs, residuals_co, yerr=np.ones_like(r_obs), fmt='o', 
                   label='Coherence', color='C0', markersize=5, capsize=2, alpha=0.7)
        ax.errorbar(r_obs, residuals_nfw, yerr=np.ones_like(r_obs), fmt='s', 
                   label='NFW DM', color='C2', markersize=5, capsize=2, alpha=0.7)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(2, color='gray', linestyle=':', alpha=0.3)
        ax.axhline(-2, color='gray', linestyle=':', alpha=0.3)
        
        ax.set_xlabel('Radius (kpc)', fontsize=12)
        ax.set_ylabel('Residuals (σ)', fontsize=12)
        ax.set_title('Fit Residuals Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"\nSaved comparison plot: {savefig}")
        
        plt.show()
        
        # Print comparison summary
        print(f"\n{'=' * 70}")
        print("FIT COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"\nCoherence Field Model:")
        print(f"  chi^2 = {coherence_result['chi2']:.2f}")
        print(f"  chi^2_red = {coherence_result['chi2_reduced']:.3f}")
        print(f"  DOF = {coherence_result['dof']}")
        print(f"  Free parameters: 4 (M_disk, R_disk, rho_c0, R_c)")
        
        print(f"\nNFW Dark Matter Model:")
        print(f"  chi^2 = {nfw_result['chi2']:.2f}")
        print(f"  chi^2_red = {nfw_result['chi2_reduced']:.3f}")
        print(f"  DOF = {nfw_result['dof']}")
        print(f"  Free parameters: 4 (M_disk, R_disk, M200, c)")
        
        ratio = coherence_result['chi2_reduced'] / nfw_result['chi2_reduced']
        if ratio < 1.0:
            print(f"\n{'=' * 70}")
            print(f"[SUCCESS] Coherence model fits BETTER than NFW DM!")
            print(f"  Ratio: {ratio:.3f} (chi^2_red_co / chi^2_red_nfw)")
        elif ratio < 1.1:
            print(f"\n{'=' * 70}")
            print(f"[TIE] Coherence model fits comparably to NFW DM")
            print(f"  Ratio: {ratio:.3f} (chi^2_red_co / chi^2_red_nfw)")
        else:
            print(f"\n{'=' * 70}")
            print(f"[NEEDS TUNING] NFW DM fits better")
            print(f"  Ratio: {ratio:.3f} (chi^2_red_co / chi^2_red_nfw)")
        print(f"{'=' * 70}")


def fit_galaxy_with_comparison(galaxy_name, output_dir='../outputs'):
    """
    Fit a galaxy with both coherence and NFW models.
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    output_dir : str
        Output directory for plots
        
    Returns:
    --------
    results : dict
        Both fit results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fitter = EnhancedSPARCFitter()
    
    # Load data
    print(f"\n{'=' * 70}")
    print(f"LOADING GALAXY: {galaxy_name}")
    print(f"{'=' * 70}")
    data = fitter.load_galaxy(galaxy_name)
    
    # Fit coherence halo
    coherence_result = fitter.fit_coherence_halo(data, method='global')
    
    # Fit NFW dark matter
    nfw_result = fitter.fit_NFW_dark_matter(data, method='global')
    
    # Compare fits
    savefig = os.path.join(output_dir, f'{galaxy_name}_coherence_vs_NFW.png')
    fitter.compare_fits(coherence_result, nfw_result, savefig=savefig)
    
    return {
        'galaxy': galaxy_name,
        'coherence': coherence_result,
        'nfw': nfw_result
    }


def fit_multiple_galaxies(galaxy_names, output_dir='../outputs'):
    """
    Fit multiple galaxies and create summary.
    
    Parameters:
    -----------
    galaxy_names : list
        List of galaxy names
    output_dir : str
        Output directory
        
    Returns:
    --------
    results : list
        List of fit results
    """
    results = []
    
    for galaxy_name in galaxy_names:
        try:
            result = fit_galaxy_with_comparison(galaxy_name, output_dir)
            results.append(result)
        except Exception as e:
            print(f"\nERROR fitting {galaxy_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY OF FITS")
    print(f"{'=' * 80}")
    print(f"{'Galaxy':<15} {'Coherence chi^2_red':<20} {'NFW chi^2_red':<18} {'Ratio':<10} {'Winner'}")
    print("-" * 80)
    
    for r in results:
        co_chi2 = r['coherence']['chi2_reduced']
        nfw_chi2 = r['nfw']['chi2_reduced']
        ratio = co_chi2 / nfw_chi2
        
        if ratio < 1.0:
            winner = "Coherence"
        elif ratio < 1.1:
            winner = "Tie"
        else:
            winner = "NFW"
        
        print(f"{r['galaxy']:<15} {co_chi2:>18.3f}  {nfw_chi2:>16.3f}  {ratio:>8.3f}  {winner}")
    
    print(f"{'=' * 80}")
    
    # Count wins
    coherence_wins = sum(1 for r in results if r['coherence']['chi2_reduced'] < r['nfw']['chi2_reduced'])
    nfw_wins = sum(1 for r in results if r['coherence']['chi2_reduced'] > r['nfw']['chi2_reduced'])
    ties = len(results) - coherence_wins - nfw_wins
    
    print(f"\nWin counts:")
    print(f"  Coherence: {coherence_wins}")
    print(f"  NFW DM: {nfw_wins}")
    print(f"  Ties: {ties}")
    
    # Export to CSV
    csv_file = os.path.join(output_dir, 'sparc_fit_summary.csv')
    export_results_to_csv(results, csv_file)
    print(f"\nSaved results to: {csv_file}")
    
    return results


def export_results_to_csv(results, csv_file):
    """
    Export fit results to CSV.
    
    Parameters:
    -----------
    results : list
        List of fit results from fit_multiple_galaxies
    csv_file : str
        Output CSV filename
    """
    rows = []
    
    for r in results:
        galaxy = r['galaxy']
        co = r['coherence']
        nfw = r['nfw']
        
        co_params = co['params']
        nfw_params = nfw['params']
        
        ratio = co['chi2_reduced'] / nfw['chi2_reduced'] if nfw['chi2_reduced'] > 0 else np.inf
        
        rows.append({
            'galaxy': galaxy,
            'chi2_red_coherence': co['chi2_reduced'],
            'chi2_red_nfw': nfw['chi2_reduced'],
            'ratio': ratio,
            'winner': 'Coherence' if ratio < 1.0 else ('NFW' if ratio > 1.1 else 'Tie'),
            'M_disk_co': co_params[0],
            'R_disk_co': co_params[1],
            'rho_c0': co_params[2],
            'R_c': co_params[3],
            'M_disk_nfw': nfw_params[0],
            'R_disk_nfw': nfw_params[1],
            'M200': nfw_params[2],
            'c': nfw_params[3],
            'n_points': len(co['data']['r']),
            'dof_coherence': co['dof'],
            'dof_nfw': nfw['dof']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"\nExported {len(rows)} galaxy fits to CSV")


if __name__ == '__main__':
    print("=" * 80)
    print("ENHANCED SPARC FITTING WITH NFW COMPARISON")
    print("=" * 80)
    
    # Example galaxies (diverse sizes)
    example_galaxies = ['DDO154', 'NGC2403', 'NGC6946']
    
    # Get available galaxies
    loader = RealDataLoader()
    available = loader.list_available_galaxies()
    
    # Filter to available
    test_galaxies = [g for g in example_galaxies if g in available[:10]]
    
    if not test_galaxies:
        test_galaxies = available[:3]
    
    print(f"\nFitting {len(test_galaxies)} galaxies: {test_galaxies}")
    
    # Fit with comparison
    results = fit_multiple_galaxies(test_galaxies)
    
    print(f"\n{'=' * 80}")
    print(f"Completed {len(results)} fits")
    print(f"Check ../outputs/ for plots")
    print(f"{'=' * 80}")

