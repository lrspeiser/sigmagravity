"""
Gate Fitting Tool

Fit gate parameters to rotation curve data and validate solar system safety.

Usage:
    python gate_fitting_tool.py
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_core import (
    G_distance, G_acceleration, G_bulge_exponential,
    G_unified, K_sigma_gravity, C_burr_XII
)


class GateFitter:
    """
    Fit gate parameters to rotation curve data
    """
    
    def __init__(self, coherence_length_l0=5.0, p=0.75, n_coh=0.5):
        """
        Parameters
        ----------
        coherence_length_l0 : float
            Coherence length from paper (kpc)
        p, n_coh : float
            Coherence window parameters from paper
        """
        self.l0 = coherence_length_l0
        self.p = p
        self.n_coh = n_coh
        self.AU_in_kpc = 4.848136811e-9
    
    def fit_to_rotation_curve(self, R_data, v_obs, v_bar, gate_type='exponential',
                              bounds=None, method='differential_evolution'):
        """
        Fit gate parameters to observed rotation curve
        
        Parameters
        ----------
        R_data : array_like
            Radial positions (kpc)
        v_obs : array_like
            Observed circular velocities (km/s)
        v_bar : array_like
            Baryonic model velocities (km/s)
        gate_type : str
            'distance', 'exponential', 'unified'
        bounds : dict, optional
            Parameter bounds for optimization
        method : str
            'differential_evolution' (global) or 'minimize' (local)
        
        Returns
        -------
        result : dict
            Best-fit parameters, chi-squared, and residuals
        """
        # Convert velocities to accelerations
        g_obs = (v_obs * 1000)**2 / (R_data * 3.086e19)  # m/s²
        g_bar = (v_bar * 1000)**2 / (R_data * 3.086e19)
        
        # Set default bounds
        if bounds is None:
            bounds = self._get_default_bounds(gate_type)
        
        # Objective function
        def objective(params):
            A = params[0]
            gate_params = self._unpack_gate_params(params[1:], gate_type, bounds)
            
            # Compute kernel
            try:
                K = K_sigma_gravity(R_data, g_bar, A, self.l0, self.p, self.n_coh,
                                   gate_type, gate_params)
                g_model = g_bar * (1 + K)
                
                # Chi-squared
                residuals = (g_model - g_obs) / g_obs
                chi2 = np.sum(residuals**2)
                
                # Penalty for violating PPN
                K_1AU = K_sigma_gravity(self.AU_in_kpc, 5.9e-3, A, self.l0, 
                                       self.p, self.n_coh, gate_type, gate_params)
                if K_1AU > 1e-14:
                    chi2 += 1e6 * (K_1AU - 1e-14)  # Large penalty
                
                return chi2
            except:
                return 1e10  # Invalid parameters
        
        # Optimize
        param_bounds = self._get_scipy_bounds(bounds)
        
        if method == 'differential_evolution':
            result_opt = differential_evolution(objective, param_bounds, 
                                               seed=42, maxiter=300, atol=1e-4, tol=1e-4)
        else:
            x0 = [(b[0] + b[1])/2 for b in param_bounds]
            result_opt = minimize(objective, x0, bounds=param_bounds, method='L-BFGS-B')
        
        # Extract results
        A_fit = result_opt.x[0]
        gate_params_fit = self._unpack_gate_params(result_opt.x[1:], gate_type, bounds)
        
        # Compute final model
        K_fit = K_sigma_gravity(R_data, g_bar, A_fit, self.l0, self.p, self.n_coh,
                               gate_type, gate_params_fit)
        g_model_fit = g_bar * (1 + K_fit)
        v_model_fit = np.sqrt(g_model_fit * R_data * 3.086e19) / 1000  # km/s
        
        # Diagnostics
        residuals = v_model_fit - v_obs
        chi2 = np.sum((residuals / v_obs)**2)
        chi2_reduced = chi2 / (len(R_data) - len(result_opt.x))
        
        return {
            'A': A_fit,
            'gate_params': gate_params_fit,
            'gate_type': gate_type,
            'v_model': v_model_fit,
            'residuals': residuals,
            'chi2': chi2,
            'chi2_reduced': chi2_reduced,
            'success': result_opt.success,
            'message': result_opt.message if hasattr(result_opt, 'message') else 'OK'
        }
    
    def _get_default_bounds(self, gate_type):
        """Default parameter bounds"""
        if gate_type == 'distance':
            return {
                'A': (0.1, 2.0),
                'R_min': (0.5, 3.0),
                'alpha': (1.5, 3.0),
                'beta': (0.5, 2.0)
            }
        elif gate_type == 'exponential':
            return {
                'A': (0.1, 2.0),
                'R_bulge': (0.5, 5.0),
                'alpha': (1.5, 3.0),
                'beta': (0.5, 2.0)
            }
        elif gate_type == 'unified':
            return {
                'A': (0.1, 2.0),
                'R_min': (0.5, 3.0),
                'g_crit': (1e-11, 1e-9),
                'alpha_R': (1.5, 3.0),
                'beta_R': (0.5, 2.0),
                'alpha_g': (1.5, 3.0),
                'beta_g': (0.5, 2.0)
            }
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
    
    def _get_scipy_bounds(self, bounds_dict):
        """Convert bounds dict to scipy format"""
        return [bounds_dict[key] for key in sorted(bounds_dict.keys())]
    
    def _unpack_gate_params(self, params, gate_type, bounds_dict):
        """Convert parameter array to gate_params dict"""
        keys = sorted([k for k in bounds_dict.keys() if k != 'A'])
        return {k: v for k, v in zip(keys, params)}
    
    def check_solar_system_safety(self, gate_params, gate_type, A=0.6):
        """
        Verify that fitted gates satisfy PPN constraints
        
        Parameters
        ----------
        gate_params : dict
            Fitted gate parameters
        gate_type : str
            Gate type
        A : float
            Amplitude
        
        Returns
        -------
        safety : dict
            G and K values at AU scales
        """
        AU_scales = [1.0, 10.0, 100.0, 1000.0, 10000.0]  # AU
        R_AU = np.array(AU_scales) * self.AU_in_kpc
        g_solar = 5.9e-3  # Earth acceleration (m/s²)
        
        K_vals = K_sigma_gravity(R_AU, g_solar, A, self.l0, self.p, self.n_coh,
                                gate_type, gate_params)
        
        return {
            'AU_scales': AU_scales,
            'K_values': K_vals,
            'G_at_1AU': K_vals[0],
            'is_safe': K_vals[0] < 1e-14,
            'margin': 1e-14 / K_vals[0] if K_vals[0] > 0 else np.inf
        }
    
    def plot_fit_results(self, R_data, v_obs, v_bar, result, 
                        save_path='outputs/gate_fit_example.png'):
        """
        Visualize fit results
        
        Parameters
        ----------
        R_data, v_obs, v_bar : array_like
            Data arrays
        result : dict
            Output from fit_to_rotation_curve
        save_path : str
            Where to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel 1: Rotation curve
        ax = axes[0, 0]
        ax.plot(R_data, v_obs, 'ko', label='Observed', markersize=5)
        ax.plot(R_data, v_bar, 'b--', label='Baryons only', linewidth=2)
        ax.plot(R_data, result['v_model'], 'r-', label='Σ-Gravity', linewidth=2)
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('v_circ (km/s)')
        ax.set_title('Rotation Curve Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Residuals
        ax = axes[0, 1]
        ax.plot(R_data, result['residuals'], 'ko', markersize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('Residual (km/s)')
        ax.set_title(f"Residuals (χ²_red = {result['chi2_reduced']:.2f})")
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Gate function
        ax = axes[1, 0]
        R_fine = np.logspace(np.log10(R_data.min()), np.log10(R_data.max()), 200)
        g_bar_fine = (v_bar[0] * 1000)**2 / (R_data[0] * 3.086e19)  # Approximate
        K_fine = K_sigma_gravity(R_fine, g_bar_fine, result['A'], self.l0, 
                                self.p, self.n_coh, result['gate_type'], 
                                result['gate_params'])
        
        ax.plot(R_fine, K_fine, 'r-', linewidth=2, label='K(R)')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('K(R)')
        ax.set_title('Σ-Gravity Kernel')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, None])
        
        # Panel 4: Parameter summary
        ax = axes[1, 1]
        ax.axis('off')
        
        text = f"Fit Results\n{'='*40}\n\n"
        text += f"Gate type: {result['gate_type']}\n"
        text += f"A = {result['A']:.3f}\n"
        text += f"χ²_reduced = {result['chi2_reduced']:.3f}\n\n"
        text += "Gate parameters:\n"
        for k, v in result['gate_params'].items():
            if 'crit' in k:
                text += f"  {k} = {v:.2e}\n"
            else:
                text += f"  {k} = {v:.3f}\n"
        
        # Check safety
        safety = self.check_solar_system_safety(result['gate_params'], 
                                                result['gate_type'], result['A'])
        text += f"\nSolar System Safety:\n"
        text += f"  K(1 AU) = {safety['G_at_1AU']:.2e}\n"
        text += f"  PPN safe? {safety['is_safe']}\n"
        if safety['is_safe']:
            text += f"  Safety margin: {safety['margin']:.0e}×\n"
        
        ax.text(0.1, 0.5, text, fontsize=10, family='monospace',
               verticalalignment='center')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def generate_toy_rotation_curve(R, M_disk=1e10, R_disk=3.0, M_bulge=1e9, R_bulge=0.5):
    """
    Generate a toy rotation curve for testing
    
    Parameters
    ----------
    R : array_like
        Radial positions (kpc)
    M_disk, R_disk : float
        Disk mass (M_sun) and scale length (kpc)
    M_bulge, R_bulge : float
        Bulge mass (M_sun) and scale radius (kpc)
    
    Returns
    -------
    v_bar : array_like
        Baryonic circular velocity (km/s)
    v_obs : array_like
        "Observed" velocity with added boost
    """
    G_newton = 4.302e-6  # kpc (km/s)^2 / M_sun
    
    # Disk contribution (exponential)
    y = R / (2 * R_disk)
    from scipy.special import i0, i1, k0, k1
    v_disk_sq = G_newton * M_disk / R * y**2 * (i0(y)*k0(y) - i1(y)*k1(y))
    v_disk_sq = np.maximum(v_disk_sq, 0)
    
    # Bulge contribution (Hernquist)
    v_bulge_sq = G_newton * M_bulge * R / (R + R_bulge)**2
    
    # Total baryonic
    v_bar = np.sqrt(v_disk_sq + v_bulge_sq)
    
    # Add "dark matter" boost (simulated)
    boost = 1 + 0.6 * (1 - np.exp(-R/5.0))
    v_obs = v_bar * np.sqrt(boost)
    
    # Add noise
    noise = np.random.normal(0, 5, len(R))  # 5 km/s scatter
    v_obs += noise
    
    return v_bar, v_obs


def main():
    """Demo of gate fitting"""
    
    print("Gate Fitting Tool - Demo")
    print("=" * 60)
    
    # Generate toy data
    np.random.seed(42)
    R_data = np.linspace(1, 20, 30)
    v_bar, v_obs = generate_toy_rotation_curve(R_data)
    
    print("\nGenerated toy rotation curve")
    print(f"  R range: {R_data.min():.1f} - {R_data.max():.1f} kpc")
    print(f"  v_obs range: {v_obs.min():.1f} - {v_obs.max():.1f} km/s")
    
    # Fit with different gate types
    gate_types = ['distance', 'exponential']
    
    for gate_type in gate_types:
        print(f"\n{'='*60}")
        print(f"Fitting with {gate_type} gate...")
        print('='*60)
        
        fitter = GateFitter(coherence_length_l0=5.0, p=0.75, n_coh=0.5)
        result = fitter.fit_to_rotation_curve(R_data, v_obs, v_bar, 
                                             gate_type=gate_type,
                                             method='differential_evolution')
        
        print(f"\nResults:")
        print(f"  A = {result['A']:.3f}")
        print(f"  chi2_reduced = {result['chi2_reduced']:.3f}")
        print(f"  Success: {result['success']}")
        
        print(f"\nGate parameters:")
        for k, v in result['gate_params'].items():
            if 'crit' in k:
                print(f"  {k} = {v:.2e}")
            else:
                print(f"  {k} = {v:.3f}")
        
        # Check safety
        safety = fitter.check_solar_system_safety(result['gate_params'], 
                                                  gate_type, result['A'])
        print(f"\nSolar System Safety:")
        print(f"  K(1 AU) = {safety['G_at_1AU']:.2e}")
        print(f"  PPN safe? {safety['is_safe']}")
        if safety['is_safe']:
            print(f"  Safety margin: {safety['margin']:.0e}×")
        
        # Plot
        save_path = f"outputs/gate_fit_{gate_type}_example.png"
        fitter.plot_fit_results(R_data, v_obs, v_bar, result, save_path)
    
    print("\n" + "="*60)
    print("[OK] Demo complete!")
    print("\nNext steps:")
    print("1. Adapt this to your real SPARC data")
    print("2. Loop over multiple galaxies")
    print("3. Check population consistency")


if __name__ == '__main__':
    main()

