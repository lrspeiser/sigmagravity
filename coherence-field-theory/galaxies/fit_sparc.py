"""
Fit SPARC galaxy rotation curves with coherence field model.

Uses existing SPARC data to constrain coherence halo parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from rotation_curves import GalaxyRotationCurve
import os


class SPARCFitter:
    """
    Fit SPARC rotation curves with coherence field halos.
    """
    
    def __init__(self, data_dir='../data/sparc'):
        """
        Parameters:
        -----------
        data_dir : str
            Directory containing SPARC data files
        """
        self.data_dir = data_dir
        self.G = 4.30091e-6  # G in (km/s)^2 kpc / M_sun
        
    def load_sparc_galaxy(self, galaxy_name):
        """
        Load SPARC data for a galaxy.
        
        Parameters:
        -----------
        galaxy_name : str
            Galaxy identifier
            
        Returns:
        --------
        data : dict
            Dictionary with keys: r (kpc), v_obs (km/s), v_err, v_gas, v_disk, v_bulge
        """
        # This is a placeholder - actual implementation depends on SPARC data format
        # User has SPARC data in ../data/sparc/
        sparc_file = os.path.join(self.data_dir, f'{galaxy_name}.csv')
        
        if not os.path.exists(sparc_file):
            print(f"Warning: {sparc_file} not found. Using synthetic data for demo.")
            return self._generate_synthetic_data()
        
        # Load actual SPARC data
        try:
            df = pd.read_csv(sparc_file)
            data = {
                'name': galaxy_name,
                'r': df['Rad'].values,  # Radius in kpc
                'v_obs': df['Vobs'].values,  # Observed velocity
                'v_err': df['errV'].values if 'errV' in df else np.ones_like(df['Vobs'].values) * 5.0,
                'v_disk': df['Vdisk'].values if 'Vdisk' in df else None,
                'v_gas': df['Vgas'].values if 'Vgas' in df else None,
                'v_bulge': df['Vbul'].values if 'Vbul' in df else None,
            }
            return data
        except Exception as e:
            print(f"Error loading {sparc_file}: {e}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic galaxy data for testing."""
        r = np.linspace(0.5, 25.0, 40)
        
        # Synthetic baryonic curve (declining)
        v_baryon = 150 * np.sqrt(r / (r + 3.0))
        
        # Add "dark matter" component to simulate flat curve
        v_obs = np.sqrt(v_baryon**2 + (120)**2 * (1 - np.exp(-r/5.0)))
        v_err = np.ones_like(r) * 5.0
        
        # Add noise
        v_obs += np.random.normal(0, v_err)
        
        return {
            'name': 'Synthetic',
            'r': r,
            'v_obs': v_obs,
            'v_err': v_err,
            'v_disk': v_baryon,
            'v_gas': v_baryon * 0.3,
            'v_bulge': None
        }
    
    def fit_coherence_halo(self, data, method='global'):
        """
        Fit coherence halo parameters to match observed rotation curve.
        
        Parameters:
        -----------
        data : dict
            Galaxy data dictionary
        method : str
            'local' for local optimization, 'global' for differential evolution
            
        Returns:
        --------
        result : dict
            Best-fit parameters and diagnostics
        """
        print(f"\nFitting coherence halo for galaxy: {data['name']}")
        print("=" * 60)
        
        # Extract data
        r_obs = data['r']
        v_obs = data['v_obs']
        v_err = data['v_err']
        
        # Estimate baryonic mass from v_disk and v_gas if available
        if data['v_disk'] is not None:
            v_baryon = np.sqrt(data['v_disk']**2 + 
                             (data['v_gas']**2 if data['v_gas'] is not None else 0) +
                             (data['v_bulge']**2 if data['v_bulge'] is not None else 0))
            M_disk_est = np.max(v_baryon)**2 * np.max(r_obs) / self.G
        else:
            # Rough estimate from inner velocity
            M_disk_est = v_obs[0]**2 * r_obs[0] / self.G
        
        R_disk_est = np.median(r_obs) / 3.0
        
        print(f"Estimated disk parameters:")
        print(f"  M_disk ~ {M_disk_est:.2e} M_sun")
        print(f"  R_disk ~ {R_disk_est:.2f} kpc")
        
        def chi_squared(params):
            """Chi-squared objective function."""
            M_disk, R_disk, rho_c0, R_c = params
            
            # Penalties for unphysical values
            if M_disk <= 0 or R_disk <= 0 or rho_c0 < 0 or R_c <= 0:
                return 1e10
            
            # Create model
            galaxy = GalaxyRotationCurve(G=self.G)
            galaxy.set_baryon_profile(M_disk=M_disk, R_disk=R_disk)
            
            # Convert rho_c0 to mass units (requires dimensionful conversion)
            # Here rho_c0 is in units such that the integral gives reasonable masses
            rho_c0_physical = rho_c0 * M_disk / (R_disk**3)
            galaxy.set_coherence_halo_simple(rho_c0=rho_c0_physical, R_c=R_c)
            
            # Compute model velocities
            v_model = galaxy.circular_velocity(r_obs)
            
            # Chi-squared
            chi2 = np.sum(((v_obs - v_model) / v_err)**2)
            
            return chi2
        
        # Initial guess
        p0 = [M_disk_est, R_disk_est, 0.1, 5.0]
        
        if method == 'local':
            # Local optimization
            bounds = [(M_disk_est*0.1, M_disk_est*10), 
                     (R_disk_est*0.5, R_disk_est*3),
                     (0.001, 10.0), 
                     (1.0, 50.0)]
            
            result = minimize(chi_squared, p0, method='L-BFGS-B', bounds=bounds)
            best_params = result.x
            chi2_min = result.fun
            
        else:
            # Global optimization
            bounds = [(M_disk_est*0.1, M_disk_est*10), 
                     (R_disk_est*0.5, R_disk_est*3),
                     (0.001, 10.0), 
                     (1.0, 50.0)]
            
            result = differential_evolution(chi_squared, bounds, 
                                          maxiter=300, seed=42, 
                                          workers=1, disp=True)
            best_params = result.x
            chi2_min = result.fun
        
        # Compute reduced chi-squared
        dof = len(r_obs) - len(best_params)
        chi2_reduced = chi2_min / dof
        
        print(f"\nBest-fit parameters:")
        print(f"  M_disk = {best_params[0]:.2e} M_sun")
        print(f"  R_disk = {best_params[1]:.2f} kpc")
        print(f"  rho_c0 = {best_params[2]:.4f} (dimensionless)")
        print(f"  R_c    = {best_params[3]:.2f} kpc")
        print(f"\nGoodness of fit:")
        print(f"  χ² = {chi2_min:.2f}")
        print(f"  χ²_red = {chi2_reduced:.2f}")
        
        return {
            'params': best_params,
            'chi2': chi2_min,
            'chi2_reduced': chi2_reduced,
            'data': data
        }
    
    def plot_fit(self, fit_result, savefig=None):
        """
        Plot fitted rotation curve.
        
        Parameters:
        -----------
        fit_result : dict
            Result from fit_coherence_halo
        savefig : str
            Filename to save figure
        """
        params = fit_result['params']
        data = fit_result['data']
        
        M_disk, R_disk, rho_c0_dim, R_c = params
        rho_c0_phys = rho_c0_dim * M_disk / (R_disk**3)
        
        # Create model
        galaxy = GalaxyRotationCurve(G=self.G)
        galaxy.set_baryon_profile(M_disk=M_disk, R_disk=R_disk)
        galaxy.set_coherence_halo_simple(rho_c0=rho_c0_phys, R_c=R_c)
        
        # Plot
        galaxy.plot_rotation_curve(
            r_max=np.max(data['r']) * 1.2,
            observed_r=data['r'],
            observed_v=data['v_obs'],
            observed_err=data['v_err'],
            savefig=savefig
        )
        
        # Add chi-squared to plot
        plt.gcf().text(0.15, 0.85, 
                      f"$\\chi^2_{{\\rm red}}$ = {fit_result['chi2_reduced']:.2f}",
                      fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')


def main():
    """Main fitting routine."""
    print("=" * 70)
    print("COHERENCE FIELD THEORY - SPARC GALAXY FITTING")
    print("=" * 70)
    
    fitter = SPARCFitter()
    
    # Try to load real SPARC data, otherwise use synthetic
    print("\nLoading galaxy data...")
    data = fitter.load_sparc_galaxy('NGC2403')  # Example galaxy
    
    # Fit coherence halo
    fit_result = fitter.fit_coherence_halo(data, method='global')
    
    # Plot result
    print("\nGenerating plot...")
    fitter.plot_fit(fit_result, savefig='sparc_fit_example.png')
    
    print("\n" + "=" * 70)
    print("Fitting complete! Check sparc_fit_example.png")
    print("=" * 70)


if __name__ == '__main__':
    main()

