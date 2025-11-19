"""
Parameter optimization framework for coherence field theory.

Simultaneous fitting across multiple scales:
- Cosmological expansion (Pantheon SNe)
- Galaxy rotation curves (SPARC)
- Cluster lensing
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import emcee
import corner
from multiprocessing import Pool
import os
import sys
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cosmology.background_evolution import CoherenceCosmology
from galaxies.rotation_curves import GalaxyRotationCurve
from clusters.lensing_profiles import ClusterLensing


class MultiScaleFitter:
    """
    Fit coherence field parameters across multiple scales simultaneously.
    """
    
    def __init__(self):
        """Initialize fitter with data containers."""
        self.cosmo_data = None
        self.galaxy_data = []
        self.cluster_data = []
        
        self.param_names = ['V0', 'lambda', 'rho_c0_gal', 'R_c_gal', 
                           'rho_c0_cluster', 'R_c_cluster']
        
    def add_cosmology_data(self, z, dL_obs, dL_err):
        """
        Add cosmological distance data (e.g., from Pantheon).
        
        Parameters:
        -----------
        z : array
            Redshifts
        dL_obs : array
            Observed luminosity distances
        dL_err : array
            Distance uncertainties
        """
        self.cosmo_data = {
            'z': z,
            'dL_obs': dL_obs,
            'dL_err': dL_err
        }
        print(f"Added cosmology data: {len(z)} SNe")
    
    def add_galaxy_data(self, name, r, v_obs, v_err, M_disk_est, R_disk_est):
        """
        Add galaxy rotation curve data.
        
        Parameters:
        -----------
        name : str
            Galaxy name
        r : array
            Radii (kpc)
        v_obs : array
            Observed velocities (km/s)
        v_err : array
            Velocity uncertainties
        M_disk_est, R_disk_est : float
            Estimated disk parameters
        """
        self.galaxy_data.append({
            'name': name,
            'r': r,
            'v_obs': v_obs,
            'v_err': v_err,
            'M_disk_est': M_disk_est,
            'R_disk_est': R_disk_est
        })
        print(f"Added galaxy: {name} ({len(r)} data points)")
    
    def add_cluster_data(self, name, R, Sigma_obs, Sigma_err, M200_est, c_est):
        """
        Add cluster lensing data.
        
        Parameters:
        -----------
        name : str
            Cluster name
        R : array
            Projected radii (kpc)
        Sigma_obs : array
            Observed surface density
        Sigma_err : array
            Uncertainties
        M200_est, c_est : float
            Estimated NFW parameters
        """
        self.cluster_data.append({
            'name': name,
            'R': R,
            'Sigma_obs': Sigma_obs,
            'Sigma_err': Sigma_err,
            'M200_est': M200_est,
            'c_est': c_est
        })
        print(f"Added cluster: {name} ({len(R)} data points)")
    
    def compute_chi_squared(self, params):
        """
        Total chi-squared across all datasets.
        
        Parameters:
        -----------
        params : array
            [V0, lambda, rho_c0_gal, R_c_gal, rho_c0_cluster, R_c_cluster]
            
        Returns:
        --------
        chi2_total : float
            Total chi-squared
        """
        V0, lam, rho_c0_gal, R_c_gal, rho_c0_cluster, R_c_cluster = params
        
        # Check for unphysical values
        if V0 <= 0 or lam <= 0 or rho_c0_gal < 0 or R_c_gal <= 0:
            return 1e10
        if rho_c0_cluster < 0 or R_c_cluster <= 0:
            return 1e10
        
        chi2_total = 0.0
        
        # 1. Cosmology chi-squared
        if self.cosmo_data is not None:
            try:
                cosmo = CoherenceCosmology(V0=V0, lambda_param=lam)
                cosmo.evolve()
                
                dL_model = cosmo.compute_dL(self.cosmo_data['z'])
                residuals = (self.cosmo_data['dL_obs'] - dL_model) / self.cosmo_data['dL_err']
                chi2_cosmo = np.sum(residuals**2)
                chi2_total += chi2_cosmo
                
            except Exception as e:
                print(f"Cosmology error: {e}")
                return 1e10
        
        # 2. Galaxy rotation curves chi-squared
        G = 4.30091e-6
        for gal_data in self.galaxy_data:
            try:
                # Use estimated disk params (could also fit these)
                M_disk = gal_data['M_disk_est']
                R_disk = gal_data['R_disk_est']
                
                galaxy = GalaxyRotationCurve(G=G)
                galaxy.set_baryon_profile(M_disk=M_disk, R_disk=R_disk)
                
                rho_c0_phys = rho_c0_gal * M_disk / (R_disk**3)
                galaxy.set_coherence_halo_simple(rho_c0=rho_c0_phys, R_c=R_c_gal)
                
                v_model = galaxy.circular_velocity(gal_data['r'])
                residuals = (gal_data['v_obs'] - v_model) / gal_data['v_err']
                chi2_gal = np.sum(residuals**2)
                chi2_total += chi2_gal
                
            except Exception as e:
                print(f"Galaxy {gal_data['name']} error: {e}")
                return 1e10
        
        # 3. Cluster lensing chi-squared
        for cluster_data in self.cluster_data:
            try:
                lens = ClusterLensing(z_lens=0.3, z_source=1.0)
                
                M200 = cluster_data['M200_est']
                c = cluster_data['c_est']
                r_vir = 2000  # kpc, rough estimate
                
                lens.set_baryonic_profile_NFW(M200, c, r_vir)
                lens.set_coherence_profile_simple(rho_c0_cluster, R_c_cluster)
                
                # Compute model surface densities
                Sigma_model = np.array([
                    lens.surface_density(R, lens.rho_NFW) + 
                    lens.surface_density(R, lens.rho_coherence)
                    for R in cluster_data['R']
                ])
                
                residuals = (cluster_data['Sigma_obs'] - Sigma_model) / cluster_data['Sigma_err']
                chi2_cluster = np.sum(residuals**2)
                chi2_total += chi2_cluster
                
            except Exception as e:
                print(f"Cluster {cluster_data['name']} error: {e}")
                return 1e10
        
        return chi2_total
    
    def fit_maximum_likelihood(self, p0=None, method='global'):
        """
        Maximum likelihood fit.
        
        Parameters:
        -----------
        p0 : array
            Initial guess
        method : str
            'local' or 'global'
            
        Returns:
        --------
        result : dict
            Best-fit parameters and diagnostics
        """
        print("\n" + "=" * 70)
        print("Maximum Likelihood Fitting")
        print("=" * 70)
        
        if p0 is None:
            p0 = [1e-6, 1.0, 0.1, 5.0, 1e8, 500.0]
        
        bounds = [
            (1e-8, 1e-4),   # V0
            (0.1, 5.0),     # lambda
            (1e-3, 10.0),   # rho_c0_gal (dimensionless)
            (1.0, 100.0),   # R_c_gal (kpc)
            (1e6, 1e10),    # rho_c0_cluster (M_sun/kpc^3)
            (50.0, 2000.0)  # R_c_cluster (kpc)
        ]
        
        if method == 'local':
            print("\nUsing local optimization (L-BFGS-B)...")
            result = minimize(self.compute_chi_squared, p0, 
                            method='L-BFGS-B', bounds=bounds)
            best_params = result.x
            chi2_min = result.fun
            
        else:
            print("\nUsing global optimization (differential evolution)...")
            result = differential_evolution(self.compute_chi_squared, bounds,
                                          maxiter=500, seed=42, 
                                          workers=1, disp=True, polish=True)
            best_params = result.x
            chi2_min = result.fun
        
        # Compute total DOF
        n_data = 0
        if self.cosmo_data is not None:
            n_data += len(self.cosmo_data['z'])
        for gal in self.galaxy_data:
            n_data += len(gal['r'])
        for cluster in self.cluster_data:
            n_data += len(cluster['R'])
        
        n_params = len(best_params)
        dof = n_data - n_params
        chi2_reduced = chi2_min / dof if dof > 0 else chi2_min
        
        print(f"\n{'=' * 70}")
        print("Best-fit parameters:")
        for i, name in enumerate(self.param_names):
            print(f"  {name:20s} = {best_params[i]:.4e}")
        
        print(f"\nGoodness of fit:")
        print(f"  χ² = {chi2_min:.2f}")
        print(f"  DOF = {dof}")
        print(f"  χ²_red = {chi2_reduced:.3f}")
        print("=" * 70)
        
        return {
            'params': best_params,
            'param_names': self.param_names,
            'chi2': chi2_min,
            'chi2_reduced': chi2_reduced,
            'dof': dof
        }
    
    def fit_mcmc(self, p0, n_walkers=32, n_steps=5000, burn_in=1000):
        """
        MCMC sampling with emcee.
        
        Parameters:
        -----------
        p0 : array
            Initial parameter guess
        n_walkers : int
            Number of MCMC walkers
        n_steps : int
            Number of steps per walker
        burn_in : int
            Burn-in steps to discard
            
        Returns:
        --------
        result : dict
            MCMC chains and statistics
        """
        print("\n" + "=" * 70)
        print("MCMC Sampling with emcee")
        print("=" * 70)
        
        ndim = len(p0)
        
        # Initialize walkers around p0
        pos = p0 + 1e-4 * np.random.randn(n_walkers, ndim)
        
        # Define log probability
        def log_prob(params):
            chi2 = self.compute_chi_squared(params)
            if chi2 > 1e9:
                return -np.inf
            return -0.5 * chi2
        
        # Run MCMC
        print(f"\nRunning {n_walkers} walkers for {n_steps} steps...")
        
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
        sampler.run_mcmc(pos, n_steps, progress=True)
        
        # Discard burn-in
        samples = sampler.get_chain(discard=burn_in, flat=True)
        
        # Compute statistics
        param_means = np.mean(samples, axis=0)
        param_stds = np.std(samples, axis=0)
        
        print(f"\n{'=' * 70}")
        print("MCMC Results (mean ± std):")
        for i, name in enumerate(self.param_names):
            print(f"  {name:20s} = {param_means[i]:.4e} ± {param_stds[i]:.4e}")
        print("=" * 70)
        
        return {
            'samples': samples,
            'param_names': self.param_names,
            'means': param_means,
            'stds': param_stds,
            'sampler': sampler
        }
    
    def plot_corner(self, mcmc_result, savefig='corner_plot.png'):
        """
        Create corner plot from MCMC samples.
        
        Parameters:
        -----------
        mcmc_result : dict
            Output from fit_mcmc
        savefig : str
            Filename to save figure
        """
        samples = mcmc_result['samples']
        labels = mcmc_result['param_names']
        
        fig = corner.corner(samples, labels=labels,
                          quantiles=[0.16, 0.5, 0.84],
                          show_titles=True, title_fmt='.3e')
        
        fig.savefig(savefig, dpi=300)
        print(f"\nCorner plot saved: {savefig}")
        plt.show()


def example_fit():
    """Example multi-scale fit."""
    print("=" * 70)
    print("MULTI-SCALE PARAMETER FITTING EXAMPLE")
    print("=" * 70)
    
    fitter = MultiScaleFitter()
    
    # Add synthetic cosmology data
    print("\nGenerating synthetic data...")
    z_cosmo = np.linspace(0.1, 1.5, 30)
    # Use ΛCDM as "truth"
    Omega_m = 0.3
    Omega_L = 0.7
    dL_true = []
    for z in z_cosmo:
        zs = np.linspace(0, z, 500)
        a_s = 1.0 / (1.0 + zs)
        H_s = np.sqrt(Omega_m * a_s**-3 + Omega_L)
        chi = trapz(1.0 / H_s, zs)
        dL_true.append((1 + z) * chi)
    dL_true = np.array(dL_true)
    dL_obs = dL_true + np.random.normal(0, 0.02 * dL_true)
    dL_err = 0.02 * dL_true
    
    fitter.add_cosmology_data(z_cosmo, dL_obs, dL_err)
    
    # Add synthetic galaxy data
    r_gal = np.linspace(1, 20, 25)
    v_gal = 150 * np.ones_like(r_gal) + np.random.normal(0, 5, len(r_gal))
    v_err = np.ones_like(r_gal) * 5.0
    fitter.add_galaxy_data('Synthetic1', r_gal, v_gal, v_err, 
                          M_disk_est=1e11, R_disk_est=3.0)
    
    # Fit
    result = fitter.fit_maximum_likelihood(method='global')
    
    print("\n" + "=" * 70)
    print("Example fit complete!")
    print("=" * 70)


if __name__ == '__main__':
    example_fit()

