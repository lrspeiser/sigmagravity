"""
Tune field parameters (β, V₀, λ) to match phenomenological fits.

Test different parameter combinations and find optimal values.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from galaxies.fit_field_driven import FieldDrivenSPARCFitter
from cosmology.background_evolution import CoherenceCosmology
import pandas as pd

def test_parameter_combination(galaxy_name, V0, lambda_param, beta, phi_inf):
    """Test a parameter combination on a galaxy."""
    fitter = FieldDrivenSPARCFitter()
    
    try:
        data = fitter.load_galaxy(galaxy_name)
        result = fitter.fit_field_driven_halo(
            data,
            V0=V0,
            lambda_param=lambda_param,
            beta=beta,
            phi_inf=phi_inf,
            method='global'
        )
        
        return {
            'success': True,
            'rho_c0': result['effective_halo']['rho_c0'],
            'R_c': result['effective_halo']['R_c'],
            'chi2_red': result['chi2_reduced'],
            'M_disk': result['params'][0],
            'R_disk': result['params'][1]
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def compare_with_phenomenological(galaxy_name, field_result):
    """Compare field-driven result with phenomenological fit."""
    try:
        df = pd.read_csv('outputs/sparc_fit_summary.csv')
        fitted = df[df['galaxy'] == galaxy_name]
        
        if len(fitted) > 0:
            fitted_rho_c0_dim = fitted['rho_c0'].values[0]
            fitted_M_disk = fitted['M_disk_co'].values[0]
            fitted_R_disk = fitted['R_disk_co'].values[0]
            fitted_R_c = fitted['R_c'].values[0]
            fitted_chi2 = fitted['chi2_red_coherence'].values[0]
            
            fitted_rho_c0_phys = fitted_rho_c0_dim * fitted_M_disk / (fitted_R_disk**3)
            
            return {
                'fitted_rho_c0': fitted_rho_c0_phys,
                'fitted_R_c': fitted_R_c,
                'fitted_chi2': fitted_chi2,
                'ratio_rho': field_result['rho_c0'] / fitted_rho_c0_phys,
                'ratio_R': field_result['R_c'] / fitted_R_c,
                'ratio_chi2': field_result['chi2_red'] / fitted_chi2
            }
    except:
        return None

def main():
    """Tune parameters on CamB galaxy."""
    print("=" * 80)
    print("PARAMETER TUNING FOR FIELD-DRIVEN HALOS")
    print("=" * 80)
    
    # Cosmology parameters (fixed)
    V0 = 1e-6
    lambda_param = 1.0
    
    # Evolve cosmology to get phi_inf
    cosmo = CoherenceCosmology(V0=V0, lambda_param=lambda_param)
    cosmo.evolve()
    phi_inf = cosmo.get_phi_0()
    
    print(f"\nCosmology parameters:")
    print(f"  V0 = {V0:.2e}")
    print(f"  lambda = {lambda_param:.2f}")
    print(f"  phi_inf = {phi_inf:.6f}")
    
    # Test galaxy
    galaxy_name = 'CamB'
    
    # Beta values to test
    beta_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    print(f"\n{'=' * 80}")
    print(f"Testing on {galaxy_name}")
    print(f"{'=' * 80}")
    
    results = []
    
    for beta in beta_values:
        print(f"\n--- Testing beta = {beta:.3f} ---")
        
        field_result = test_parameter_combination(galaxy_name, V0, lambda_param, beta, phi_inf)
        
        if field_result['success']:
            comparison = compare_with_phenomenological(galaxy_name, field_result)
            
            print(f"  Field-driven:")
            print(f"    rho_c0 = {field_result['rho_c0']:.2e} M_sun/kpc^3")
            print(f"    R_c = {field_result['R_c']:.2f} kpc")
            print(f"    chi^2_red = {field_result['chi2_red']:.3f}")
            
            if comparison:
                print(f"  Comparison:")
                print(f"    rho_c0 ratio: {comparison['ratio_rho']:.2f}x")
                print(f"    R_c ratio: {comparison['ratio_R']:.2f}x")
                print(f"    chi^2 ratio: {comparison['ratio_chi2']:.2f}x")
                
                # Score: closer to 1.0 is better
                score = 1.0 / (1.0 + abs(comparison['ratio_rho'] - 1.0) + 
                              abs(comparison['ratio_R'] - 1.0))
                
                results.append({
                    'beta': beta,
                    'rho_c0': field_result['rho_c0'],
                    'R_c': field_result['R_c'],
                    'chi2_red': field_result['chi2_red'],
                    'ratio_rho': comparison['ratio_rho'],
                    'ratio_R': comparison['ratio_R'],
                    'ratio_chi2': comparison['ratio_chi2'],
                    'score': score
                })
        else:
            print(f"  [ERROR] {field_result.get('error', 'Unknown error')}")
    
    # Find best beta
    if results:
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        
        print(f"\n{'beta':<10} {'rho_c0 ratio':<15} {'R_c ratio':<15} {'chi^2 ratio':<15} {'score':<10}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['beta']:<10.3f} {r['ratio_rho']:<15.2f} {r['ratio_R']:<15.2f} "
                  f"{r['ratio_chi2']:<15.2f} {r['score']:<10.3f}")
        
        best = max(results, key=lambda x: x['score'])
        print(f"\nBest beta: {best['beta']:.3f}")
        print(f"  rho_c0 ratio: {best['ratio_rho']:.2f}x")
        print(f"  R_c ratio: {best['ratio_R']:.2f}x")
        print(f"  chi^2 ratio: {best['ratio_chi2']:.2f}x")
        print(f"  score: {best['score']:.3f}")

if __name__ == '__main__':
    main()

