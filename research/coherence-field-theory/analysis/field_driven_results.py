"""
Generate comprehensive results report for field-driven galaxy fitting.

Tests field-driven fits on multiple galaxies and compares with phenomenological fits.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from galaxies.fit_field_driven import FieldDrivenSPARCFitter
from cosmology.background_evolution import CoherenceCosmology

def test_galaxy(galaxy_name, V0, lambda_param, beta, phi_inf):
    """Test field-driven fit on a galaxy."""
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
            'R_disk': result['params'][1],
            'chi2': result['chi2_reduced'] * (len(data['r']) - 2)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def compare_with_phenomenological(galaxy_name, field_result):
    """Compare field-driven result with phenomenological fit."""
    try:
        df = pd.read_csv('../outputs/sparc_fit_summary.csv')
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
                'fitted_M_disk': fitted_M_disk,
                'fitted_R_disk': fitted_R_disk,
                'ratio_rho': field_result['rho_c0'] / fitted_rho_c0_phys if fitted_rho_c0_phys > 0 else np.nan,
                'ratio_R': field_result['R_c'] / fitted_R_c if fitted_R_c > 0 else np.nan,
                'ratio_chi2': field_result['chi2_red'] / fitted_chi2 if fitted_chi2 > 0 else np.nan
            }
    except:
        return None

def main():
    """Generate comprehensive results report."""
    print("=" * 80)
    print("FIELD-DRIVEN GALAXY FITTING - COMPREHENSIVE RESULTS")
    print("=" * 80)
    
    # Field parameters
    V0 = 1e-6
    lambda_param = 1.0
    beta = 0.01  # Best from tuning
    
    # Evolve cosmology
    cosmo = CoherenceCosmology(V0=V0, lambda_param=lambda_param)
    cosmo.evolve()
    phi_inf = cosmo.get_phi_0()
    
    print(f"\nField Parameters:")
    print(f"  V0 = {V0:.2e}")
    print(f"  lambda = {lambda_param:.2f}")
    print(f"  beta = {beta:.3f}")
    print(f"  phi_inf = {phi_inf:.6f}")
    print(f"  Omega_m0 = {cosmo.results['Omega_m0']:.4f}")
    print(f"  Omega_phi0 = {cosmo.results['Omega_phi0']:.4f}")
    
    # Test galaxies
    test_galaxies = ['CamB', 'DDO154', 'DDO168', 'NGC2403', 'DDO064']
    
    print(f"\n{'=' * 80}")
    print(f"Testing on {len(test_galaxies)} galaxies")
    print(f"{'=' * 80}")
    
    results = []
    
    for galaxy_name in test_galaxies:
        print(f"\n--- {galaxy_name} ---")
        
        field_result = test_galaxy(galaxy_name, V0, lambda_param, beta, phi_inf)
        
        if field_result['success']:
            comparison = compare_with_phenomenological(galaxy_name, field_result)
            
            print(f"  Field-driven:")
            print(f"    M_disk = {field_result['M_disk']:.2e} M_sun")
            print(f"    R_disk = {field_result['R_disk']:.2f} kpc")
            print(f"    rho_c0 = {field_result['rho_c0']:.2e} M_sun/kpc^3")
            print(f"    R_c = {field_result['R_c']:.2f} kpc")
            print(f"    chi^2_red = {field_result['chi2_red']:.3f}")
            
            if comparison:
                print(f"  Phenomenological:")
                print(f"    M_disk = {comparison['fitted_M_disk']:.2e} M_sun")
                print(f"    R_disk = {comparison['fitted_R_disk']:.2f} kpc")
                print(f"    rho_c0 = {comparison['fitted_rho_c0']:.2e} M_sun/kpc^3")
                print(f"    R_c = {comparison['fitted_R_c']:.2f} kpc")
                print(f"    chi^2_red = {comparison['fitted_chi2']:.3f}")
                
                print(f"  Ratios (field / phenomenological):")
                print(f"    rho_c0: {comparison['ratio_rho']:.2f}x")
                print(f"    R_c: {comparison['ratio_R']:.2f}x")
                print(f"    chi^2: {comparison['ratio_chi2']:.2f}x")
                
                results.append({
                    'galaxy': galaxy_name,
                    'field_M_disk': field_result['M_disk'],
                    'field_R_disk': field_result['R_disk'],
                    'field_rho_c0': field_result['rho_c0'],
                    'field_R_c': field_result['R_c'],
                    'field_chi2_red': field_result['chi2_red'],
                    'phenom_M_disk': comparison['fitted_M_disk'],
                    'phenom_R_disk': comparison['fitted_R_disk'],
                    'phenom_rho_c0': comparison['fitted_rho_c0'],
                    'phenom_R_c': comparison['fitted_R_c'],
                    'phenom_chi2_red': comparison['fitted_chi2'],
                    'ratio_rho': comparison['ratio_rho'],
                    'ratio_R': comparison['ratio_R'],
                    'ratio_chi2': comparison['ratio_chi2']
                })
        else:
            print(f"  [ERROR] {field_result.get('error', 'Unknown error')}")
            results.append({
                'galaxy': galaxy_name,
                'error': field_result.get('error', 'Unknown error')
            })
    
    # Summary statistics
    if results:
        valid_results = [r for r in results if 'error' not in r]
        
        if valid_results:
            print(f"\n{'=' * 80}")
            print("SUMMARY STATISTICS")
            print(f"{'=' * 80}")
            
            ratios_rho = [r['ratio_rho'] for r in valid_results]
            ratios_R = [r['ratio_R'] for r in valid_results]
            ratios_chi2 = [r['ratio_chi2'] for r in valid_results]
            
            print(f"\nDensity ratio (field / phenomenological):")
            print(f"  Mean: {np.mean(ratios_rho):.2f}x")
            print(f"  Median: {np.median(ratios_rho):.2f}x")
            print(f"  Range: {np.min(ratios_rho):.2f}x - {np.max(ratios_rho):.2f}x")
            
            print(f"\nRadius ratio (field / phenomenological):")
            print(f"  Mean: {np.mean(ratios_R):.2f}x")
            print(f"  Median: {np.median(ratios_R):.2f}x")
            print(f"  Range: {np.min(ratios_R):.2f}x - {np.max(ratios_R):.2f}x")
            
            print(f"\nChi^2 ratio (field / phenomenological):")
            print(f"  Mean: {np.mean(ratios_chi2):.2f}x")
            print(f"  Median: {np.median(ratios_chi2):.2f}x")
            print(f"  Range: {np.min(ratios_chi2):.2f}x - {np.max(ratios_chi2):.2f}x")
            
            # Count wins
            chi2_wins = sum(1 for r in ratios_chi2 if r < 1.0)
            print(f"\nField-driven wins (chi^2 < 1.0): {chi2_wins}/{len(valid_results)}")
            
            # Save results
            df_results = pd.DataFrame(valid_results)
            df_results.to_csv('../outputs/field_driven_results.csv', index=False)
            print(f"\nResults saved to: outputs/field_driven_results.csv")
    
    return results

if __name__ == '__main__':
    main()

