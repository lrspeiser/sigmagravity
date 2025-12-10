"""
Diagnose chameleon cosmology issue.

Check why chameleon breaks cosmology and test potential fixes.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cosmology.background_evolution import CoherenceCosmology
from galaxies.halo_field_profile import HaloFieldSolver

def main():
    """Diagnose chameleon cosmology issue."""
    print("=" * 80)
    print("CHAMELEON COSMOLOGY DIAGNOSIS")
    print("=" * 80)
    
    V0 = 1e-6
    lambda_param = 1.0
    M4 = 5e-2
    
    print(f"\nTest parameters: V0={V0:.2e}, lambda={lambda_param:.2f}, M4={M4:.2e}")
    
    # Test 1: Check potential values at typical field values
    print(f"\n{'=' * 80}")
    print("Test 1: Potential Values at Different phi")
    print(f"{'=' * 80}")
    
    phi_values = [0.05, 0.5, 1.0, 5.0, 10.0]
    
    print(f"\n{'phi':<10} {'V_exp':<15} {'V_cham_term':<15} {'V_total':<15} {'dV_dphi':<15}")
    print("-" * 80)
    
    for phi in phi_values:
        # Pure exponential
        V_exp = V0 * np.exp(-lambda_param * phi)
        # Chameleon term
        V_cham = M4**5 / phi
        V_total = V_exp + V_cham
        dV_dphi = -lambda_param * V_exp - M4**5 / (phi**2)
        
        print(f"{phi:<10.2f} {V_exp:<15.6e} {V_cham:<15.6e} {V_total:<15.6e} {dV_dphi:<15.6e}")
    
    # Test 2: Check field evolution without chameleon
    print(f"\n{'=' * 80}")
    print("Test 2: Field Evolution (Pure Exponential)")
    print(f"{'=' * 80}")
    
    cosmo_exp = CoherenceCosmology(V0=V0, lambda_param=lambda_param, M4=None)
    cosmo_exp.evolve()
    
    phi_array = cosmo_exp.results['phi']
    a_array = cosmo_exp.results['a']
    
    print(f"\nField evolution:")
    print(f"  phi(a=1.0) = {cosmo_exp.get_phi_0():.6f}")
    print(f"  phi range: {np.min(phi_array):.6f} to {np.max(phi_array):.6f}")
    print(f"  Omega_m0 = {cosmo_exp.results['Omega_m0']:.4f}")
    print(f"  Omega_phi0 = {cosmo_exp.results['Omega_phi0']:.4f}")
    
    # Test 3: Check field evolution with chameleon
    print(f"\n{'=' * 80}")
    print("Test 3: Field Evolution (With Chameleon M4=5e-2)")
    print(f"{'=' * 80}")
    
    cosmo_cham = CoherenceCosmology(V0=V0, lambda_param=lambda_param, M4=M4)
    cosmo_cham.evolve()
    
    phi_array_cham = cosmo_cham.results['phi']
    a_array_cham = cosmo_cham.results['a']
    
    print(f"\nField evolution:")
    print(f"  phi(a=1.0) = {cosmo_cham.get_phi_0():.6f}")
    print(f"  phi range: {np.min(phi_array_cham):.6f} to {np.max(phi_array_cham):.6f}")
    print(f"  Omega_m0 = {cosmo_cham.results['Omega_m0']:.4f}")
    print(f"  Omega_phi0 = {cosmo_cham.results['Omega_phi0']:.4f}")
    
    # Test 4: Check what happens with smaller M4
    print(f"\n{'=' * 80}")
    print("Test 4: Smaller M4 Values")
    print(f"{'=' * 80}")
    
    print(f"\n{'M4':<12} {'phi_0':<15} {'Omega_m0':<12} {'Omega_phi0':<12} {'V_cham/V_exp':<15}")
    print("-" * 80)
    
    for M4_test in [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
        cosmo = CoherenceCosmology(V0=V0, lambda_param=lambda_param, M4=M4_test)
        cosmo.evolve()
        phi_0 = cosmo.get_phi_0()
        
        # Compare V_cham to V_exp at phi_0
        V_exp_0 = V0 * np.exp(-lambda_param * phi_0)
        V_cham_0 = M4_test**5 / phi_0 if phi_0 > 0 else np.inf
        ratio = V_cham_0 / V_exp_0 if V_exp_0 > 0 else np.inf
        
        print(f"{M4_test:<12.2e} {phi_0:<15.6f} {cosmo.results['Omega_m0']:<12.4f} "
              f"{cosmo.results['Omega_phi0']:<12.4f} {ratio:<15.6e}")
    
    # Test 5: Try adjusting V0
    print(f"\n{'=' * 80}")
    print("Test 5: Adjusting V0 to Compensate for Chameleon")
    print(f"{'=' * 80}")
    
    M4_fixed = 5e-2
    V0_values = [1e-6, 1e-5, 1e-4, 1e-3]
    
    print(f"\n{'V0':<12} {'phi_0':<15} {'Omega_m0':<12} {'Omega_phi0':<12} {'Close?':<10}")
    print("-" * 80)
    
    for V0_test in V0_values:
        cosmo = CoherenceCosmology(V0=V0_test, lambda_param=lambda_param, M4=M4_fixed)
        cosmo.evolve()
        phi_0 = cosmo.get_phi_0()
        
        Omega_m0 = cosmo.results['Omega_m0']
        Omega_phi0 = cosmo.results['Omega_phi0']
        close = (abs(Omega_m0 - 0.3) < 0.2 and abs(Omega_phi0 - 0.7) < 0.2)
        close_str = "YES" if close else "NO"
        
        print(f"{V0_test:<12.2e} {phi_0:<15.6f} {Omega_m0:<12.4f} "
              f"{Omega_phi0:<12.4f} {close_str:<10}")

if __name__ == '__main__':
    main()

