"""
Tune field parameters with R_c penalty term.

Adds penalty for oversized halos to force field solution to match
phenomenological R_c values.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from galaxies.fit_field_driven import FieldDrivenSPARCFitter
from cosmology.background_evolution import CoherenceCosmology
import pandas as pd

def test_parameter_combination_with_penalty(galaxy_name, V0, lambda_param, beta, phi_inf, w_R=1.0):
    """Test a parameter combination with R_c penalty."""
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
        
        # Get phenomenological R_c for comparison
        try:
            df = pd.read_csv('../outputs/sparc_fit_summary.csv')
            fitted = df[df['galaxy'] == galaxy_name]
            if len(fitted) > 0:
                R_c_phen = fitted['R_c'].values[0]
            else:
                R_c_phen = None
        except:
            R_c_phen = None
        
        R_c_field = result['effective_halo']['R_c']
        chi2_rot = result['chi2_reduced']
        
        # Add R_c penalty
        if R_c_phen is not None and R_c_phen > 0:
            log_ratio = np.log(R_c_field / R_c_phen)
            chi2_eff = chi2_rot + w_R * log_ratio**2
        else:
            chi2_eff = chi2_rot
            log_ratio = np.nan
        
        return {
            'success': True,
            'rho_c0': result['effective_halo']['rho_c0'],
            'R_c': R_c_field,
            'R_c_phen': R_c_phen,
            'chi2_red': chi2_rot,
            'chi2_eff': chi2_eff,
            'log_ratio': log_ratio,
            'M_disk': result['params'][0],
            'R_disk': result['params'][1]
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Tune parameters with R_c penalty."""
    print("=" * 80)
    print("PARAMETER TUNING WITH R_C PENALTY")
    print("=" * 80)
    
    # Cosmology parameters (fixed)
    V0 = 1e-6
    lambda_param = 1.0
    
    # Evolve cosmology
    cosmo = CoherenceCosmology(V0=V0, lambda_param=lambda_param)
    cosmo.evolve()
    phi_inf = cosmo.get_phi_0()
    
    print(f"\nCosmology parameters:")
    print(f"  V0 = {V0:.2e}")
    print(f"  lambda = {lambda_param:.2f}")
    print(f"  phi_inf = {phi_inf:.6f}")
    
    # Test galaxies (the 3 that win)
    test_galaxies = ['DDO154', 'DDO168', 'DDO064']
    
    # Beta values to test
    beta_values = [0.005, 0.01, 0.02, 0.05]
    
    # Lambda values to test
    lambda_values = [0.8, 1.0, 1.2]
    
    print(f"\n{'=' * 80}")
    print(f"Testing on {len(test_galaxies)} galaxies")
    print(f"{'=' * 80}")
    
    results = []
    w_R = 1.0  # Penalty weight
    
    for lambda_param in lambda_values:
        for beta in beta_values:
            print(f"\n--- Testing lambda={lambda_param:.1f}, beta={beta:.3f} ---")
            
            # Re-evolve cosmology with new lambda
            cosmo = CoherenceCosmology(V0=V0, lambda_param=lambda_param)
            cosmo.evolve()
            phi_inf = cosmo.get_phi_0()
            
            galaxy_results = []
            
            for galaxy_name in test_galaxies:
                result = test_parameter_combination_with_penalty(
                    galaxy_name, V0, lambda_param, beta, phi_inf, w_R
                )
                
                if result['success']:
                    galaxy_results.append(result)
                    print(f"  {galaxy_name}: R_c={result['R_c']:.2f} kpc, "
                          f"R_c_phen={result['R_c_phen']:.2f} kpc, "
                          f"chi2_eff={result['chi2_eff']:.3f}")
            
            if galaxy_results:
                # Aggregate statistics
                chi2_ratios = [r['chi2_red'] for r in galaxy_results]
                R_c_ratios = [r['R_c'] / r['R_c_phen'] 
                             for r in galaxy_results if r['R_c_phen'] is not None and r['R_c_phen'] > 0]
                chi2_eff_vals = [r['chi2_eff'] for r in galaxy_results]
                
                if R_c_ratios:
                    results.append({
                        'lambda': lambda_param,
                        'beta': beta,
                        'median_chi2_red': np.median(chi2_ratios),
                        'median_Rc_ratio': np.median(R_c_ratios),
                        'median_chi2_eff': np.median(chi2_eff_vals),
                        'mean_Rc_ratio': np.mean(R_c_ratios)
                    })
    
    # Find best parameters
    if results:
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        
        print(f"\n{'lambda':<10} {'beta':<10} {'median chi2':<15} {'median Rc ratio':<15} {'median chi2_eff':<15}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['lambda']:<10.1f} {r['beta']:<10.3f} {r['median_chi2_red']:<15.3f} "
                  f"{r['median_Rc_ratio']:<15.2f} {r['median_chi2_eff']:<15.3f}")
        
        # Score: minimize chi2_eff and Rc_ratio near 1
        for r in results:
            r['score'] = r['median_chi2_eff'] / (1.0 + abs(r['median_Rc_ratio'] - 1.0))
        
        best = min(results, key=lambda x: x['score'])
        print(f"\nBest parameters:")
        print(f"  lambda = {best['lambda']:.1f}")
        print(f"  beta = {best['beta']:.3f}")
        print(f"  median chi2_red = {best['median_chi2_red']:.3f}")
        print(f"  median Rc ratio = {best['median_Rc_ratio']:.2f}")
        print(f"  median chi2_eff = {best['median_chi2_eff']:.3f}")
        print(f"  score = {best['score']:.3f}")

if __name__ == '__main__':
    main()

