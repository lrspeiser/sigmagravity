"""
Test cosmology with chameleon parameters.

Verify that best chameleon parameters (β=0.1, M4=5e-2) are cosmologically viable.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cosmology.background_evolution import CoherenceCosmology
import matplotlib.pyplot as plt

def main():
    """Test cosmology with chameleon parameters."""
    print("=" * 80)
    print("COSMOLOGY COMPATIBILITY TEST - CHAMELEON PARAMETERS")
    print("=" * 80)
    
    # Best parameters from scan
    V0 = 1e-6
    lambda_param = 1.0
    M4 = 5e-2  # Chameleon parameter
    
    print(f"\nField Parameters:")
    print(f"  V0 = {V0:.2e}")
    print(f"  lambda = {lambda_param:.2f}")
    print(f"  M4 = {M4:.2e} (chameleon)")
    
    # Test both without and with chameleon
    print(f"\n{'=' * 80}")
    print("Test 1: Pure Exponential (M4 = None)")
    print(f"{'=' * 80}")
    
    cosmo_exp = CoherenceCosmology(V0=V0, lambda_param=lambda_param, rho_m0_guess=1e-6, M4=None)
    cosmo_exp.evolve()
    
    print(f"\nResults:")
    print(f"  Omega_m0 = {cosmo_exp.results['Omega_m0']:.4f}")
    print(f"  Omega_phi0 = {cosmo_exp.results['Omega_phi0']:.4f}")
    print(f"  phi_0 (today) = {cosmo_exp.get_phi_0():.6f}")
    print(f"  H_0 = 1.0 (normalized)")
    
    print(f"\n{'=' * 80}")
    print("Test 2: With Chameleon (M4 = 5e-2)")
    print(f"{'=' * 80}")
    
    cosmo_cham = CoherenceCosmology(V0=V0, lambda_param=lambda_param, rho_m0_guess=1e-6, M4=M4)
    cosmo_cham.evolve()
    
    print(f"\nResults:")
    print(f"  Omega_m0 = {cosmo_cham.results['Omega_m0']:.4f}")
    print(f"  Omega_phi0 = {cosmo_cham.results['Omega_phi0']:.4f}")
    print(f"  phi_0 (today) = {cosmo_cham.get_phi_0():.6f}")
    print(f"  H_0 = 1.0 (normalized)")
    
    # Compare with LCDM expectations (Omega_m ~ 0.3, Omega_phi ~ 0.7)
    print(f"\n{'=' * 80}")
    print("Comparison with LCDM")
    print(f"{'=' * 80}")
    
    print(f"\nPure Exponential:")
    print(f"  Omega_m = {cosmo_exp.results['Omega_m0']:.4f} (target: ~0.3)")
    print(f"  Omega_phi = {cosmo_exp.results['Omega_phi0']:.4f} (target: ~0.7)")
    print(f"  Delta_Omega_m = {cosmo_exp.results['Omega_m0'] - 0.3:.4f}")
    print(f"  Delta_Omega_phi = {cosmo_exp.results['Omega_phi0'] - 0.7:.4f}")
    
    print(f"\nWith Chameleon:")
    print(f"  Omega_m = {cosmo_cham.results['Omega_m0']:.4f} (target: ~0.3)")
    print(f"  Omega_phi = {cosmo_cham.results['Omega_phi0']:.4f} (target: ~0.7)")
    print(f"  Delta_Omega_m = {cosmo_cham.results['Omega_m0'] - 0.3:.4f}")
    print(f"  Delta_Omega_phi = {cosmo_cham.results['Omega_phi0'] - 0.7:.4f}")
    
    # Check if chameleon is cosmologically viable
    print(f"\n{'=' * 80}")
    print("Cosmological Viability")
    print(f"{'=' * 80}")
    
    cham_viable = True
    issues = []
    
    if abs(cosmo_cham.results['Omega_m0'] - 0.3) > 0.2:
        cham_viable = False
        issues.append(f"Omega_m too far from 0.3 (Delta = {cosmo_cham.results['Omega_m0'] - 0.3:.2f})")
    
    if abs(cosmo_cham.results['Omega_phi0'] - 0.7) > 0.2:
        cham_viable = False
        issues.append(f"Omega_phi too far from 0.7 (Delta = {cosmo_cham.results['Omega_phi0'] - 0.7:.2f})")
    
    if cosmo_cham.get_phi_0() <= 0:
        cham_viable = False
        issues.append("phi_0 is negative or zero")
    
    if cham_viable:
        print(f"\n[SUCCESS] Chameleon parameters are cosmologically viable!")
        print(f"  Omega_m and Omega_phi are reasonable")
        print(f"  phi_0 = {cosmo_cham.get_phi_0():.6f} is positive")
    else:
        print(f"\n[WARNING] Chameleon parameters have cosmological issues:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\n  May need to adjust V0 or lambda to compensate")
    
    # Compute H(z) and d_L(z) for comparison
    print(f"\n{'=' * 80}")
    print("H(z) and d_L(z) Evolution")
    print(f"{'=' * 80}")
    
    z_array = np.linspace(0, 1, 11)
    H_exp = cosmo_exp.compute_H_of_z(z_array)
    H_cham = cosmo_cham.compute_H_of_z(z_array)
    dL_exp = cosmo_exp.compute_dL(z_array)
    dL_cham = cosmo_cham.compute_dL(z_array)
    
    print(f"\n{'z':<10} {'H_exp':<15} {'H_cham':<15} {'Delta_H/H_exp':<15}")
    print("-" * 60)
    for i, z in enumerate(z_array):
        if H_exp[i] > 0:
            rel_diff = abs(H_cham[i] - H_exp[i]) / H_exp[i]
            print(f"{z:<10.2f} {H_exp[i]:<15.4f} {H_cham[i]:<15.4f} {rel_diff:<15.4f}")
    
    print(f"\n{'z':<10} {'dL_exp':<15} {'dL_cham':<15} {'Delta_dL/dL_exp':<15}")
    print("-" * 60)
    for i, z in enumerate(z_array):
        if dL_exp[i] > 0:
            rel_diff = abs(dL_cham[i] - dL_exp[i]) / dL_exp[i]
            print(f"{z:<10.2f} {dL_exp[i]:<15.4f} {dL_cham[i]:<15.4f} {rel_diff:<15.4f}")
    
    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print(f"{'=' * 80}")
    
    if cham_viable:
        print(f"\n[SUCCESS] Chameleon parameters (beta=0.1, M4=5e-2) are cosmologically viable")
        print(f"   Can proceed to PPN tests and galaxy fitting refinement")
    else:
        print(f"\n[WARNING] Chameleon parameters need cosmological adjustment")
        print(f"   May need to tune V0 or lambda to match Omega_m ~ 0.3, Omega_phi ~ 0.7")
        print(f"   OR: Chameleon should only affect galaxy scales, not cosmology")

if __name__ == '__main__':
    main()

