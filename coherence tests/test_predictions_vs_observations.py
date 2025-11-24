"""
Direct Prediction vs Observation Comparison
============================================

Tests coherence models from coherence_models.py against example data:
- Galaxy rotation velocities
- Cluster-like multi-source scenarios

Clear visual comparisons showing model success/failure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import existing coherence models
sys.path.insert(0, str(Path(__file__).parent))
from coherence_models import MODEL_REGISTRY, apply_coherence_model

# Constants
G = 6.674e-11  # m^3 kg^-1 s^-2
M_sun = 1.989e30  # kg
kpc = 3.086e19  # m


class CoherenceModelTester:
    """
    Test coherence models with synthetic and real data
    """
    
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / 'results'
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_galaxy_example(self):
        """
        Generate example galaxy rotation curve data
        """
        print("\n" + "="*70)
        print("GENERATING EXAMPLE GALAXY DATA")
        print("="*70)
        
        # Typical dwarf galaxy parameters
        radii_kpc = np.linspace(0.5, 15, 30)  # kpc
        M_baryon = 1e9  # M_sun
        R_galaxy = 10.0  # kpc
        sigma_v = 25.0  # km/s
        
        # Calculate baryonic contribution (simple exponential disk)
        v_baryon = []
        for r in radii_kpc:
            # Enclosed mass in exponential disk
            if r > 0:
                M_enc = M_baryon * (1 - np.exp(-r/3) * (1 + r/3))
                v_b = np.sqrt(G * M_enc * M_sun / (r * kpc)) / 1000  # km/s
            else:
                v_b = 0
            v_baryon.append(v_b)
        v_baryon = np.array(v_baryon)
        
        # Add "dark matter" halo contribution to create observations
        v_dm = 100 * np.sqrt(radii_kpc / (radii_kpc + 5))  # km/s, NFW-like
        v_obs = np.sqrt(v_baryon**2 + v_dm**2)
        
        # Add noise
        v_obs += np.random.normal(0, 5, len(v_obs))
        
        # Calculate baryonic acceleration
        g_bar = (v_baryon * 1000)**2 / (radii_kpc * kpc)  # m/s^2
        
        # Wavelength (orbital)
        lambda_gw = 2 * np.pi * radii_kpc
        
        print(f"Created galaxy with:")
        print(f"  M_baryon: {M_baryon:.2e} M_sun")
        print(f"  R_galaxy: {R_galaxy} kpc")
        print(f"  sigma_v: {sigma_v} km/s")
        print(f"  {len(radii_kpc)} data points")
        
        return {
            'name': 'Example_Galaxy',
            'radii': radii_kpc,
            'v_obs': v_obs,
            'v_baryon': v_baryon,
            'g_bar': g_bar,
            'lambda_gw': lambda_gw,
            'sigma_v': sigma_v,
            'M_baryon': M_baryon,
            'R_galaxy': R_galaxy
        }
    
    def test_model_on_galaxy(self, galaxy_data, model_name, params):
        """
        Test a specific coherence model on galaxy data
        
        Parameters:
        -----------
        galaxy_data : dict
            Galaxy data from generate_galaxy_example or loaded data
        model_name : str
            Name of model from MODEL_REGISTRY
        params : dict
            Model parameters
        
        Returns:
        --------
        results : dict
            Prediction results and metrics
        """
        print(f"\n{'='*70}")
        print(f"TESTING MODEL: {model_name}")
        print(f"{'='*70}")
        
        # Get coherence multiplier
        f_multiplier = apply_coherence_model(
            model_name,
            galaxy_data['radii'],
            galaxy_data['g_bar'],
            galaxy_data['lambda_gw'],
            galaxy_data['sigma_v'],
            params
        )
        
        # Calculate predicted velocities
        g_eff = galaxy_data['g_bar'] * f_multiplier
        v_pred = np.sqrt(np.clip(g_eff * galaxy_data['radii'] * kpc, 0, None)) / 1000  # km/s
        
        # Calculate residuals
        residuals = v_pred - galaxy_data['v_obs']
        rms_residual = np.sqrt(np.mean(residuals**2))
        mean_frac_error = np.mean(np.abs(residuals / galaxy_data['v_obs'])) * 100
        
        print(f"\nParameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")
        
        print(f"\nPrediction quality:")
        print(f"  RMS residual: {rms_residual:.2f} km/s")
        print(f"  Mean fractional error: {mean_frac_error:.1f}%")
        print(f"  Mean amplification: {np.mean(f_multiplier):.3f}×")
        print(f"  Max amplification: {np.max(f_multiplier):.3f}×")
        
        return {
            'model_name': model_name,
            'galaxy': galaxy_data['name'],
            'v_pred': v_pred,
            'f_multiplier': f_multiplier,
            'residuals': residuals,
            'rms_residual': rms_residual,
            'mean_frac_error': mean_frac_error,
            'params': params
        }
    
    def plot_galaxy_results(self, galaxy_data, results_list):
        """
        Plot comparison of multiple models against observations
        
        Parameters:
        -----------
        galaxy_data : dict
            Galaxy data
        results_list : list of dict
            List of results from test_model_on_galaxy
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Plot 1: Rotation curves
        ax = axes[0, 0]
        ax.plot(galaxy_data['radii'], galaxy_data['v_obs'], 
                'ko', markersize=6, label='Observed', alpha=0.7, zorder=10)
        ax.plot(galaxy_data['radii'], galaxy_data['v_baryon'], 
                'k--', linewidth=1.5, label='Baryonic only', alpha=0.5)
        
        for i, result in enumerate(results_list):
            ax.plot(galaxy_data['radii'], result['v_pred'], 
                   color=colors[i % len(colors)], linewidth=2, 
                   label=f"{result['model_name']}", alpha=0.8)
        
        ax.set_xlabel('Radius (kpc)', fontsize=12)
        ax.set_ylabel('Rotation Velocity (km/s)', fontsize=12)
        ax.set_title(f"{galaxy_data['name']}\nObserved vs Coherence Models", 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residuals comparison
        ax = axes[0, 1]
        for i, result in enumerate(results_list):
            ax.plot(galaxy_data['radii'], result['residuals'], 
                   color=colors[i % len(colors)], linewidth=2, 
                   label=f"{result['model_name']}: {result['rms_residual']:.1f} km/s", 
                   alpha=0.7, marker='o', markersize=4)
        
        ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
        ax.fill_between(galaxy_data['radii'], -20, 20, color='gray', alpha=0.2, 
                       label='±20 km/s')
        ax.set_xlabel('Radius (kpc)', fontsize=12)
        ax.set_ylabel('Residual (Pred - Obs) [km/s]', fontsize=12)
        ax.set_title('Model Residuals', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Amplification factors
        ax = axes[1, 0]
        for i, result in enumerate(results_list):
            ax.plot(galaxy_data['radii'], result['f_multiplier'], 
                   color=colors[i % len(colors)], linewidth=2.5, 
                   label=result['model_name'], alpha=0.8)
        
        ax.axhline(y=1, color='k', linestyle='--', linewidth=1.5, label='No amplification')
        ax.set_xlabel('Radius (kpc)', fontsize=12)
        ax.set_ylabel('Coherence Multiplier f', fontsize=12)
        ax.set_title('Amplification Profiles', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"Galaxy: {galaxy_data['name']}\n"
        summary_text += f"M_baryon: {galaxy_data['M_baryon']:.2e} M_sun\n"
        summary_text += f"σ_v: {galaxy_data['sigma_v']:.1f} km/s\n"
        summary_text += f"N_points: {len(galaxy_data['radii'])}\n\n"
        summary_text += "Model Performance:\n"
        summary_text += "-" * 50 + "\n"
        
        for result in results_list:
            summary_text += f"\n{result['model_name']}:\n"
            summary_text += f"  RMS residual: {result['rms_residual']:.2f} km/s\n"
            summary_text += f"  Mean frac. error: {result['mean_frac_error']:.1f}%\n"
            summary_text += f"  Mean amplification: {np.mean(result['f_multiplier']):.3f}×\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        save_path = self.output_dir / f"{galaxy_data['name']}_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to {save_path}")
        plt.close()
    
    def run_all_models_test(self):
        """
        Test all available coherence models on example data
        """
        print("\n" + "="*70)
        print("COHERENCE MODEL COMPARISON TEST")
        print("="*70)
        
        # Generate example galaxy
        galaxy_data = self.generate_galaxy_example()
        
        # Define parameters for each model (tuned for reasonable amplification)
        model_params = {
            'path_interference': {
                'A': 0.5,
                'ell0': 5.0,
                'p': 0.8,
                'n_coh': 1.0,
                'beta_sigma': 0.4,
                'sigma_ref': 30.0
            },
            'metric_resonance': {
                'A': 0.6,
                'ell0': 5.0,
                'p': 0.75,
                'n_coh': 1.0,
                'lambda_m0': 10.0,
                'log_width': 0.5,
                'beta_sigma': 0.0
            },
            'entanglement': {
                'A': 0.7,
                'ell0': 6.0,
                'p': 0.8,
                'n_coh': 0.8,
                'sigma0': 30.0
            },
            'vacuum_condensation': {
                'A': 0.8,
                'ell0': 5.5,
                'p': 0.85,
                'n_coh': 0.9,
                'sigma_c': 40.0,
                'alpha': 2.0,
                'beta': 1.0
            },
            'graviton_pairing': {
                'A': 0.6,
                'xi0': 5.0,
                'p': 0.75,
                'n_coh': 0.8,
                'sigma0': 30.0,
                'gamma_xi': 0.3
            }
        }
        
        # Test each model
        results_list = []
        for model_name, params in model_params.items():
            try:
                result = self.test_model_on_galaxy(galaxy_data, model_name, params)
                results_list.append(result)
            except Exception as e:
                print(f"ERROR testing {model_name}: {e}")
        
        # Plot comparison
        if results_list:
            self.plot_galaxy_results(galaxy_data, results_list)
        
        # Print summary
        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        print(f"\nTested {len(results_list)} models on {galaxy_data['name']}")
        print(f"Results saved to: {self.output_dir}")
        
        return galaxy_data, results_list


def main():
    """
    Main test routine
    """
    tester = CoherenceModelTester()
    galaxy_data, results = tester.run_all_models_test()
    
    print("\n" + "="*70)
    print("SUMMARY RANKINGS (by RMS residual)")
    print("="*70)
    
    # Sort by performance
    results_sorted = sorted(results, key=lambda x: x['rms_residual'])
    
    for i, result in enumerate(results_sorted, 1):
        print(f"{i}. {result['model_name']:25s} RMS = {result['rms_residual']:6.2f} km/s")
    
    return tester, galaxy_data, results


if __name__ == "__main__":
    tester, galaxy_data, results = main()
