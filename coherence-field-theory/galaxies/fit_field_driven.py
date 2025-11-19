"""
Fit galaxy rotation curves using field-driven halos.

Instead of free halo parameters (ρ_c0, R_c), uses scalar field solver
to derive halos from field theory parameters (V₀, λ, β).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from galaxies.rotation_curves import GalaxyRotationCurve
from galaxies.halo_field_profile import HaloFieldSolver
from data_integration.load_real_data import RealDataLoader


class FieldDrivenSPARCFitter:
    """
    Fit SPARC galaxies with field-driven halos.
    
    Uses scalar field solver to derive halos from:
    - Global field parameters: V₀, λ, β (shared across galaxies)
    - Per-galaxy baryonic parameters: M_disk, R_disk
    """
    
    def __init__(self):
        """Initialize fitter."""
        self.G = 4.30091e-6  # (km/s)^2 kpc / M_sun
        self.data_loader = RealDataLoader()
        
    def load_galaxy(self, galaxy_name):
        """Load galaxy from Rotmod_LTG."""
        try:
            data = self.data_loader.load_rotmod_galaxy(galaxy_name)
            
            # Compute baryonic velocity
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
    
    def rho_baryon_profile(self, r, M_disk, R_disk):
        """Baryon density profile (exponential disk)."""
        # Exponential disk: ρ(r) = (M_disk / (2π R_disk²)) exp(-r/R_disk)
        rho_0 = M_disk / (2 * np.pi * R_disk**2)
        return rho_0 * np.exp(-r / R_disk)
    
    def fit_field_driven_halo(self, data, V0, lambda_param, beta, 
                              method='global', phi_inf=None, M4=None):
        """
        Fit galaxy with field-driven halo.
        
        Parameters:
        -----------
        data : dict
            Galaxy data
        V0 : float
            Potential scale (cosmology units)
        lambda_param : float
            Exponential slope
        beta : float
            Matter coupling
        method : str
            'local' or 'global'
        phi_inf : float, optional
            Cosmological field value (if None, use 0)
        M4 : float, optional
            Chameleon mass scale (if None, pure exponential potential)
            
        Returns:
        --------
        result : dict
            Fit results
        """
        print(f"\n{'=' * 70}")
        print(f"FITTING FIELD-DRIVEN HALO: {data['name']}")
        print(f"{'=' * 70}")
        M4_str = f", M4={M4:.2e}" if M4 is not None else ""
        print(f"Field parameters: V0={V0:.2e}, lambda={lambda_param:.2f}, beta={beta:.2f}{M4_str}")
        
        r_obs = data['r']
        v_obs = data['v_obs']
        v_err = data['v_err']
        
        # Estimate disk mass
        mid_idx = len(r_obs) // 2
        v_baryon_mid = data['v_baryon'][mid_idx]
        M_disk_est = v_baryon_mid**2 * r_obs[mid_idx] / self.G
        R_disk_est = np.median(r_obs) / 2.5
        
        # Create field solver
        solver = HaloFieldSolver(V0, lambda_param, beta, M4=M4, phi_inf=phi_inf)
        
        # Radial grid for field solver (wider than data)
        r_min = min(0.1, r_obs[0] * 0.1)
        r_max = max(r_obs[-1] * 2.0, 50.0)
        r_grid = np.logspace(np.log10(r_min), np.log10(r_max), 200)
        
        def chi_squared(params):
            """Chi-squared for field-driven fit."""
            M_disk, R_disk = params
            
            if M_disk <= 0 or R_disk <= 0:
                return 1e10
            
            # Baryon profile
            def rho_b(r):
                return self.rho_baryon_profile(r, M_disk, R_disk)
            
            # Solve scalar field
            try:
                phi, dphi_dr = solver.solve(rho_b, r_grid, method='shooting')
                
                # Effective density
                rho_phi = solver.effective_density(phi, dphi_dr, convert_to_mass_density=True)
                
                # Interpolate to observation radii
                from scipy.interpolate import interp1d
                rho_phi_interp = interp1d(r_grid, rho_phi, kind='linear', 
                                         bounds_error=False, fill_value=(rho_phi[0], 0.0))
                
                # Create rotation curve
                galaxy = GalaxyRotationCurve(G=self.G)
                galaxy.set_baryon_profile(M_disk=M_disk, R_disk=R_disk)
                
                # Set field-driven halo
                phi_interp = interp1d(r_grid, phi, kind='linear', 
                                     bounds_error=False, fill_value=(phi[0], phi[-1]))
                galaxy.set_coherence_halo_field(phi_interp, solver.V_func)
                
                # Compute model velocities
                v_model = galaxy.circular_velocity(r_obs)
                
                # Chi-squared
                chi2 = np.sum(((v_obs - v_model) / v_err)**2)
                
                return chi2
                
            except Exception as e:
                # If solver fails, return large chi-squared
                return 1e10
        
        # Bounds
        bounds = [
            (M_disk_est*0.1, M_disk_est*10),  # M_disk
            (R_disk_est*0.3, R_disk_est*3),   # R_disk
        ]
        
        p0 = [M_disk_est, R_disk_est]
        
        if method == 'local':
            result = minimize(chi_squared, p0, method='L-BFGS-B', bounds=bounds)
            best_params = result.x
            chi2_min = result.fun
        else:
            result = differential_evolution(chi_squared, bounds, 
                                          maxiter=100, seed=42, 
                                          workers=1, disp=False, polish=True)
            best_params = result.x
            chi2_min = result.fun
        
        # Compute final solution for best parameters
        M_disk_best, R_disk_best = best_params
        rho_b = lambda r: self.rho_baryon_profile(r, M_disk_best, R_disk_best)
        phi_final, dphi_dr_final = solver.solve(rho_b, r_grid, method='shooting')
        rho_phi_final = solver.effective_density(phi_final, dphi_dr_final, 
                                                convert_to_mass_density=True)
        
        # Debug: print actual density values
        print(f"\nField solution density range:")
        print(f"  min(rho_phi) = {np.min(rho_phi_final):.2e} M_sun/kpc^3")
        print(f"  max(rho_phi) = {np.max(rho_phi_final):.2e} M_sun/kpc^3")
        print(f"  median(rho_phi) = {np.median(rho_phi_final):.2e} M_sun/kpc^3")
        
        # Compute effective mass and theoretical core radius
        R_c_theory, m_eff = solver.compute_effective_core_radius(
            phi_final, rho_b, r_grid, R_disk_best
        )
        print(f"\nEffective mass analysis:")
        print(f"  m_eff at r=2*R_disk = {m_eff:.6e} kpc^-1")
        print(f"  R_c^(theory) = 1/m_eff = {R_c_theory:.2f} kpc")
        
        # Fit halo parameters for comparison (with wider bounds if needed)
        rho_c0, R_c, chi2_fit = solver.fit_halo_parameters(rho_phi_final, r_grid)
        
        print(f"  R_c^(fitted) = {R_c:.2f} kpc")
        print(f"  Ratio: R_c^(fitted) / R_c^(theory) = {R_c / R_c_theory:.2f}")
        
        # If fit hit bounds, try with wider range
        if rho_c0 <= 1e3 + 1e-6:  # Hit lower bound
            print(f"\n[WARNING] Fit hit lower bound (rho_c0 = {rho_c0:.2e})")
            print(f"  Actual density is too low - may need stronger coupling or different V0")
        
        dof = len(r_obs) - len(best_params)
        chi2_reduced = chi2_min / dof if dof > 0 else chi2_min
        
        print(f"\nBest-fit parameters:")
        print(f"  M_disk = {M_disk_best:.2e} M_sun")
        print(f"  R_disk = {R_disk_best:.2f} kpc")
        print(f"\nEffective halo parameters (from field):")
        print(f"  rho_c0 = {rho_c0:.2e} M_sun/kpc^3")
        print(f"  R_c    = {R_c:.2f} kpc")
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
            'field_params': {'V0': V0, 'lambda': lambda_param, 'beta': beta},
            'effective_halo': {'rho_c0': rho_c0, 'R_c': R_c},
            'phi_profile': phi_final,
            'r_grid': r_grid
        }


def test_field_driven_fit():
    """Test field-driven fitting on a real galaxy."""
    print("=" * 80)
    print("TESTING FIELD-DRIVEN GALAXY FITTING")
    print("=" * 80)
    
    fitter = FieldDrivenSPARCFitter()
    
    # Test on a good-fitting dwarf galaxy
    galaxy_name = 'CamB'
    
    try:
        # Load galaxy
        data = fitter.load_galaxy(galaxy_name)
        print(f"\nLoaded {galaxy_name}: {len(data['r'])} data points")
        
        # Field parameters (same as cosmology for consistency)
        V0 = 1e-6
        lambda_param = 1.0
        beta = 0.1
        
        # Fit
        result = fitter.fit_field_driven_halo(data, V0, lambda_param, beta, 
                                              method='global')
        
        print(f"\n{'=' * 70}")
        print("FIELD-DRIVEN FIT COMPLETE")
        print(f"{'=' * 70}")
        print(f"Galaxy: {galaxy_name}")
        print(f"Field parameters: V0={V0:.2e}, lambda={lambda_param:.2f}, beta={beta:.2f}")
        print(f"Effective halo: rho_c0={result['effective_halo']['rho_c0']:.2e}, "
              f"R_c={result['effective_halo']['R_c']:.2f}")
        print(f"Chi^2_red: {result['chi2_reduced']:.3f}")
        
        # Compare with fitted halo from sparc_fit_summary.csv
        try:
            df = pd.read_csv('../outputs/sparc_fit_summary.csv')
            fitted = df[df['galaxy'] == galaxy_name]
            if len(fitted) > 0:
                fitted_rho_c0 = fitted['rho_c0'].values[0]
                fitted_R_c = fitted['R_c'].values[0]
                fitted_chi2 = fitted['chi2_red_coherence'].values[0]
                
                print(f"\nComparison with phenomenological fit:")
                print(f"  Phenomenological: rho_c0={fitted_rho_c0:.2e}, "
                      f"R_c={fitted_R_c:.2f}, chi^2_red={fitted_chi2:.3f}")
                print(f"  Field-driven: rho_c0={result['effective_halo']['rho_c0']:.2e}, "
                      f"R_c={result['effective_halo']['R_c']:.2f}, "
                      f"chi^2_red={result['chi2_reduced']:.3f}")
                
                ratio_rho = result['effective_halo']['rho_c0'] / fitted_rho_c0
                ratio_R = result['effective_halo']['R_c'] / fitted_R_c
                print(f"  Ratios: rho_c0_field/rho_c0_fit={ratio_rho:.2f}, "
                      f"R_c_field/R_c_fit={ratio_R:.2f}")
        except Exception as e:
            print(f"\nCould not load fitted parameters: {e}")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_field_driven_fit()

