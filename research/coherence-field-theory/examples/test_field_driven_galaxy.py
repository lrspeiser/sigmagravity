"""
Test field-driven galaxy fitting on a real SPARC galaxy.

This script:
1. Loads cosmology to get φ(∞) and field parameters
2. Fits a real galaxy (CamB) with field-driven halo
3. Compares with phenomenological fit
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from galaxies.fit_field_driven import FieldDrivenSPARCFitter
from cosmology.background_evolution import CoherenceCosmology
import pandas as pd


def main():
    """Test field-driven fitting on CamB galaxy."""
    print("=" * 80)
    print("FIELD-DRIVEN GALAXY FITTING TEST")
    print("=" * 80)
    
    # Step 1: Get field parameters from cosmology
    print("\n" + "=" * 80)
    print("STEP 1: Get field parameters from cosmology")
    print("=" * 80)
    
    # Use same parameters as cosmology
    V0 = 1e-6
    lambda_param = 1.0
    
    # Evolve cosmology to get today's field value
    cosmo = CoherenceCosmology(V0=V0, lambda_param=lambda_param)
    cosmo.evolve()
    phi_inf = cosmo.get_phi_0()
    
    print(f"Cosmology parameters:")
    print(f"  V0 = {V0:.2e}")
    print(f"  lambda = {lambda_param:.2f}")
    print(f"  phi_0 (today) = {phi_inf:.6f}")
    print(f"  Omega_m0 = {cosmo.results['Omega_m0']:.4f}")
    print(f"  Omega_phi0 = {cosmo.results['Omega_phi0']:.4f}")
    
    # Step 2: Fit galaxy with field-driven halo
    print("\n" + "=" * 80)
    print("STEP 2: Fit galaxy with field-driven halo")
    print("=" * 80)
    
    fitter = FieldDrivenSPARCFitter()
    galaxy_name = 'CamB'  # Good-fitting dwarf galaxy
    
    try:
        # Load galaxy
        data = fitter.load_galaxy(galaxy_name)
        print(f"\nLoaded {galaxy_name}: {len(data['r'])} data points")
        print(f"  Radius range: {data['r'][0]:.2f} - {data['r'][-1]:.2f} kpc")
        print(f"  Velocity range: {data['v_obs'].min():.1f} - {data['v_obs'].max():.1f} km/s")
        
        # Field coupling (try different values)
        # With corrected A(φ) = e^(βφ) coupling, need smaller β
        beta = 0.01  # Small coupling (test with corrected form)
        # beta = 0.1  # Medium coupling
        # beta = 1.0  # Strong (may cause numerical issues)
        
        # Fit with field-driven halo
        result = fitter.fit_field_driven_halo(
            data, 
            V0=V0, 
            lambda_param=lambda_param, 
            beta=beta,
            phi_inf=phi_inf,
            method='global'
        )
        
        # Step 3: Compare with phenomenological fit
        print("\n" + "=" * 80)
        print("STEP 3: Compare with phenomenological fit")
        print("=" * 80)
        
        try:
            df = pd.read_csv('outputs/sparc_fit_summary.csv')
            fitted = df[df['galaxy'] == galaxy_name]
            
            if len(fitted) > 0:
                # Phenomenological fit uses dimensionless rho_c0
                fitted_rho_c0_dim = fitted['rho_c0'].values[0]
                fitted_M_disk = fitted['M_disk_co'].values[0]
                fitted_R_disk = fitted['R_disk_co'].values[0]
                fitted_R_c = fitted['R_c'].values[0]
                fitted_chi2 = fitted['chi2_red_coherence'].values[0]
                
                # Convert to physical density
                fitted_rho_c0_phys = fitted_rho_c0_dim * fitted_M_disk / (fitted_R_disk**3)
                
                field_rho_c0 = result['effective_halo']['rho_c0']
                field_R_c = result['effective_halo']['R_c']
                field_chi2 = result['chi2_reduced']
                field_M_disk = result['params'][0]
                field_R_disk = result['params'][1]
                
                print(f"\nPhenomenological fit (free halo parameters):")
                print(f"  M_disk = {fitted_M_disk:.2e} M_sun")
                print(f"  R_disk = {fitted_R_disk:.2f} kpc")
                print(f"  rho_c0 (dim) = {fitted_rho_c0_dim:.4f}")
                print(f"  rho_c0 (phys) = {fitted_rho_c0_phys:.2e} M_sun/kpc^3")
                print(f"  R_c    = {fitted_R_c:.2f} kpc")
                print(f"  chi^2_red = {fitted_chi2:.3f}")
                
                print(f"\nField-driven fit (halo from field theory):")
                print(f"  M_disk = {field_M_disk:.2e} M_sun")
                print(f"  R_disk = {field_R_disk:.2f} kpc")
                print(f"  rho_c0 (phys) = {field_rho_c0:.2e} M_sun/kpc^3")
                print(f"  R_c    = {field_R_c:.2f} kpc")
                print(f"  chi^2_red = {field_chi2:.3f}")
                
                ratio_rho = field_rho_c0 / fitted_rho_c0_phys
                ratio_R = field_R_c / fitted_R_c
                ratio_chi2 = field_chi2 / fitted_chi2
                
                print(f"\nRatios (field / phenomenological):")
                print(f"  rho_c0: {ratio_rho:.2f}x")
                print(f"  R_c:    {ratio_R:.2f}x")
                print(f"  chi^2:  {ratio_chi2:.2f}x")
                
                if ratio_rho < 2.0 and ratio_rho > 0.5 and ratio_R < 2.0 and ratio_R > 0.5:
                    print(f"\n[SUCCESS] Field-driven halo matches phenomenological fit!")
                    print(f"  Parameters within factor of 2")
                elif ratio_chi2 < 2.0:
                    print(f"\n[OK] Field-driven fit quality is reasonable")
                    print(f"  Chi^2 within factor of 2, but parameters differ")
                else:
                    print(f"\n[NEEDS TUNING] Field-driven fit needs refinement")
                    print(f"  Try adjusting beta or field parameters")
            else:
                print(f"\nNo phenomenological fit found for {galaxy_name}")
                print(f"Field-driven results:")
                print(f"  rho_c0 = {result['effective_halo']['rho_c0']:.2e} M_sun/kpc^3")
                print(f"  R_c    = {result['effective_halo']['R_c']:.2f} kpc")
                print(f"  chi^2_red = {result['chi2_reduced']:.3f}")
                
        except Exception as e:
            print(f"\nCould not load phenomenological fit: {e}")
            print(f"Field-driven results:")
            print(f"  rho_c0 = {result['effective_halo']['rho_c0']:.2e} M_sun/kpc^3")
            print(f"  R_c    = {result['effective_halo']['R_c']:.2f} kpc")
            print(f"  chi^2_red = {result['chi2_reduced']:.3f}")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

