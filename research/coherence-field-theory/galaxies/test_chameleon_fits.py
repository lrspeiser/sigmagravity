"""
Test chameleon parameters in actual galaxy fits.

Uses best parameters from m_eff scan: beta=0.1, M4=5e-2
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from galaxies.fit_field_driven import FieldDrivenSPARCFitter
from cosmology.background_evolution import CoherenceCosmology
import pandas as pd

def main():
    """Test chameleon parameters on test galaxies."""
    print("=" * 80)
    print("TESTING CHAMELEON PARAMETERS IN GALAXY FITS")
    print("=" * 80)
    
    # Best parameters from scan
    V0 = 1e-6
    lambda_param = 1.0
    beta = 0.1
    M4 = 5e-2  # Chameleon parameter
    
    print(f"\nField Parameters:")
    print(f"  V0 = {V0:.2e}")
    print(f"  lambda = {lambda_param:.2f}")
    print(f"  beta = {beta:.3f}")
    print(f"  M4 = {M4:.2e} (chameleon)")
    
    # Evolve cosmology (without chameleon for now - would need to add M4)
    cosmo = CoherenceCosmology(V0=V0, lambda_param=lambda_param)
    cosmo.evolve()
    phi_inf = cosmo.get_phi_0()
    
    print(f"  phi_inf = {phi_inf:.6f}")
    
    # Test galaxies
    test_galaxies = ['DDO154', 'DDO168', 'DDO064']
    
    print(f"\n{'=' * 80}")
    print(f"Testing on {len(test_galaxies)} galaxies")
    print(f"{'=' * 80}")
    
    results = []
    
    for galaxy_name in test_galaxies:
        print(f"\n--- {galaxy_name} ---")
        
        fitter = FieldDrivenSPARCFitter()
        
        try:
            data = fitter.load_galaxy(galaxy_name)
            result = fitter.fit_field_driven_halo(
                data,
                V0=V0,
                lambda_param=lambda_param,
                beta=beta,
                phi_inf=phi_inf,
                method='global',
                M4=M4
            )
            
            print(f"  Field-driven fit (with chameleon):")
            print(f"    M_disk = {result['params'][0]:.2e} M_sun")
            print(f"    R_disk = {result['params'][1]:.2f} kpc")
            print(f"    rho_c0 = {result['effective_halo']['rho_c0']:.2e} M_sun/kpc^3")
            print(f"    R_c = {result['effective_halo']['R_c']:.2f} kpc")
            print(f"    chi^2_red = {result['chi2_reduced']:.3f}")
            
            # Compare with phenomenological
            try:
                df = pd.read_csv('../outputs/sparc_fit_summary.csv')
                fitted = df[df['galaxy'] == galaxy_name]
                if len(fitted) > 0:
                    fitted_R_c = fitted['R_c'].values[0]
                    fitted_chi2 = fitted['chi2_red_coherence'].values[0]
                    
                    print(f"  Phenomenological:")
                    print(f"    R_c = {fitted_R_c:.2f} kpc")
                    print(f"    chi^2_red = {fitted_chi2:.3f}")
                    print(f"  Ratio: R_c_field/R_c_phen = {result['effective_halo']['R_c'] / fitted_R_c:.2f}")
            except:
                pass
            
            results.append({
                'galaxy': galaxy_name,
                'R_c': result['effective_halo']['R_c'],
                'chi2_red': result['chi2_reduced']
            })
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print("NOTE: Chameleon parameter M4 not yet integrated into fit_field_driven_halo")
    print("      Need to modify fit_field_driven.py to pass M4 to HaloFieldSolver")
    print(f"{'=' * 80}")

if __name__ == '__main__':
    main()

