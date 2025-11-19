"""
Test smaller M4 values to find cosmology-compatible parameters.

Goal: Find M4 that gives reasonable R_c in galaxies without breaking cosmology.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cosmology.background_evolution import CoherenceCosmology
from galaxies.scan_meff_vs_density import compute_meff_and_Rc

def main():
    """Test smaller M4 values."""
    print("=" * 80)
    print("TESTING SMALLER M4 VALUES FOR COSMOLOGY COMPATIBILITY")
    print("=" * 80)
    
    # Fixed parameters
    V0 = 1e-6
    lambda_param = 1.0
    beta = 0.1
    
    # Test densities
    rho_cosmic = 1.45e3  # M_sun/kpc^3
    rho_dwarf = 1e7
    rho_spiral = 1e8
    
    # Test smaller M4 values - focus on cosmologically viable ones
    # From diagnosis: M4 < 1e-3 gives V_cham/V_exp < 1e-8 (negligible)
    M4_values = [None, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    
    print(f"\nParameters:")
    print(f"  V0 = {V0:.2e}")
    print(f"  lambda = {lambda_param:.2f}")
    print(f"  beta = {beta:.3f}")
    
    print(f"\n{'=' * 80}")
    print("COSMOLOGY TEST")
    print(f"{'=' * 80}")
    
    print(f"\n{'M4':<12} {'Omega_m0':<12} {'Omega_phi0':<12} {'phi_0':<15} {'Cosmo OK':<10}")
    print("-" * 80)
    
    cosmo_results = []
    
    for M4 in M4_values:
        cosmo = CoherenceCosmology(V0=V0, lambda_param=lambda_param, 
                                   rho_m0_guess=1e-6, M4=M4)
        cosmo.evolve()
        
        Omega_m0 = cosmo.results['Omega_m0']
        Omega_phi0 = cosmo.results['Omega_phi0']
        phi_0 = cosmo.get_phi_0()
        
        # Check if cosmology is reasonable
        cosmo_ok = (abs(Omega_m0 - 0.3) < 0.2 and 
                   abs(Omega_phi0 - 0.7) < 0.2 and 
                   phi_0 > 0)
        
        M4_str = "None" if M4 is None else f"{M4:.2e}"
        ok_str = "YES" if cosmo_ok else "NO"
        
        print(f"{M4_str:<12} {Omega_m0:<12.4f} {Omega_phi0:<12.4f} {phi_0:<15.6f} {ok_str:<10}")
        
        cosmo_results.append({
            'M4': M4,
            'Omega_m0': Omega_m0,
            'Omega_phi0': Omega_phi0,
            'phi_0': phi_0,
            'cosmo_ok': cosmo_ok
        })
    
    print(f"\n{'=' * 80}")
    print("GALAXY R_C TEST")
    print(f"{'=' * 80}")
    
    print(f"\n{'M4':<12} {'R_c^cosmic':<15} {'R_c^dwarf':<15} {'R_c^spiral':<15} {'Galaxy OK':<12}")
    print("-" * 80)
    
    galaxy_results = []
    
    for M4 in M4_values:
        R_c_cosmic, _, _ = compute_meff_and_Rc(V0, lambda_param, beta, M4, 
                                               rho_cosmic, phi_fixed=None)
        R_c_dwarf, _, _ = compute_meff_and_Rc(V0, lambda_param, beta, M4, 
                                              rho_dwarf, phi_fixed=None)
        R_c_spiral, _, _ = compute_meff_and_Rc(V0, lambda_param, beta, M4, 
                                               rho_spiral, phi_fixed=None)
        
        # Check if galaxies have reasonable R_c
        # Relaxed: R_c < 500 kpc (much better than ~1.5e6)
        galaxy_ok = (R_c_cosmic > 1e5 and R_c_dwarf < 500.0 and R_c_spiral < 500.0)
        
        M4_str = "None" if M4 is None else f"{M4:.2e}"
        ok_str = "YES" if galaxy_ok else "NO"
        
        print(f"{M4_str:<12} {R_c_cosmic:<15.2e} {R_c_dwarf:<15.2f} {R_c_spiral:<15.2f} {ok_str:<12}")
        
        galaxy_results.append({
            'M4': M4,
            'R_c_cosmic': R_c_cosmic,
            'R_c_dwarf': R_c_dwarf,
            'R_c_spiral': R_c_spiral,
            'galaxy_ok': galaxy_ok
        })
    
    print(f"\n{'=' * 80}")
    print("COMPATIBLE PARAMETERS")
    print(f"{'=' * 80}")
    
    compatible = []
    
    for i, cosmo_result in enumerate(cosmo_results):
        galaxy_result = galaxy_results[i]
        if cosmo_result['cosmo_ok'] and galaxy_result['galaxy_ok']:
            compatible.append({
                'M4': cosmo_result['M4'],
                'Omega_m0': cosmo_result['Omega_m0'],
                'Omega_phi0': cosmo_result['Omega_phi0'],
                'R_c_dwarf': galaxy_result['R_c_dwarf'],
                'R_c_spiral': galaxy_result['R_c_spiral']
            })
    
    if compatible:
        print(f"\nFound {len(compatible)} compatible parameter(s):\n")
        for p in compatible:
            M4_str = "None" if p['M4'] is None else f"{p['M4']:.2e}"
            print(f"M4 = {M4_str}:")
            print(f"  Omega_m0 = {p['Omega_m0']:.4f}, Omega_phi0 = {p['Omega_phi0']:.4f}")
            print(f"  R_c^dwarf = {p['R_c_dwarf']:.2f} kpc")
            print(f"  R_c^spiral = {p['R_c_spiral']:.2f} kpc\n")
    else:
        print(f"\nNo parameters found that satisfy both cosmology and galaxy constraints!")
        print(f"May need to:")
        print(f"  1. Try even smaller M4 values")
        print(f"  2. Adjust V0 or lambda to compensate")
        print(f"  3. Make M4 density-dependent")
        print(f"  4. Consider alternative screening mechanism")

if __name__ == '__main__':
    main()

