"""
Scan effective mass vs density for chameleon potential.

Tests if chameleon term can give required dynamic range:
- m_eff^cosmic ~ H0 (light in voids)
- m_eff^galaxy ~ (1-3 kpc)^-1 (heavy in galaxies)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from galaxies.halo_field_profile import HaloFieldSolver

# Physical constants
G = 4.30091e-6  # (km/s)^2 kpc / M_sun
H0_kms_kpc = 70.0 / 306.6  # km/s/kpc
H0_squared = H0_kms_kpc**2
rho_crit = 3 * H0_squared / (8 * np.pi * G)  # M_sun/kpc^3
c_km_s = 2.998e5  # km/s
H0_kpc_inv = H0_kms_kpc / c_km_s  # kpc^-1

def compute_meff_and_Rc(V0, lambda_param, beta, M4, phi, rho_b):
    """Compute m_eff and R_c for given parameters."""
    # HaloFieldSolver uses M4 parameter, where M4^5 gives the M^5 term in chameleon potential
    solver = HaloFieldSolver(V0, lambda_param, beta, M4=M4, phi_inf=phi)
    
    # Compute effective mass squared
    m_eff_sq = solver.effective_mass_squared(phi, rho_b)
    
    # Convert to physical units
    if m_eff_sq > 0:
        m_eff = np.sqrt(m_eff_sq) * H0_kpc_inv  # kpc^-1
        R_c = 1.0 / m_eff  # kpc
    else:
        m_eff = 0.0
        R_c = np.inf
    
    return m_eff, R_c, m_eff_sq

def main():
    """Scan m_eff vs density for different parameter combinations."""
    print("=" * 80)
    print("SCANNING EFFECTIVE MASS VS DENSITY")
    print("=" * 80)
    
    # Fixed cosmology parameters
    V0 = 1e-6
    lambda_param = 1.0
    
    # Test densities (in M_sun/kpc^3)
    rho_cosmic = 1.45e3  # Critical density
    rho_dwarf = 1e7  # Dwarf galaxy mid-disk (~10^7 M_sun/kpc^3)
    rho_spiral = 1e8  # Spiral galaxy mid-disk (~10^8 M_sun/kpc^3)
    
    # Field values (typical cosmological value)
    phi_cosmic = 0.05  # From cosmology evolution
    
    print(f"\nTest densities:")
    print(f"  Cosmic: {rho_cosmic:.2e} M_sun/kpc^3")
    print(f"  Dwarf: {rho_dwarf:.2e} M_sun/kpc^3")
    print(f"  Spiral: {rho_spiral:.2e} M_sun/kpc^3")
    
    # Scan parameters
    beta_values = [0.01, 0.05, 0.1]
    M4_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]  # M4 parameter (M4^5 gives M^5 term)
    
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    
    print(f"\n{'beta':<10} {'M4':<12} {'m_eff^cosmic':<15} {'R_c^cosmic':<15} "
          f"{'m_eff^dwarf':<15} {'R_c^dwarf':<15} {'m_eff^spiral':<15} {'R_c^spiral':<15}")
    print("-" * 120)
    
    viable_params = []
    
    for beta in beta_values:
        for M4 in M4_values:
            # Compute at different densities
            m_eff_cosmic, R_c_cosmic, _ = compute_meff_and_Rc(
                V0, lambda_param, beta, M4, phi_cosmic, rho_cosmic
            )
            m_eff_dwarf, R_c_dwarf, _ = compute_meff_and_Rc(
                V0, lambda_param, beta, M4, phi_cosmic, rho_dwarf
            )
            m_eff_spiral, R_c_spiral, _ = compute_meff_and_Rc(
                V0, lambda_param, beta, M4, phi_cosmic, rho_spiral
            )
            
            print(f"{beta:<10.3f} {M4:<12.2e} {m_eff_cosmic:<15.6e} {R_c_cosmic:<15.2e} "
                  f"{m_eff_dwarf:<15.6e} {R_c_dwarf:<15.2f} {m_eff_spiral:<15.6e} {R_c_spiral:<15.2f}")
            
            # Check if viable: R_c^cosmic >> R_c^galaxy
            # Target: R_c^cosmic ~ 10^4 Mpc ~ 3e9 kpc, R_c^galaxy ~ 1-5 kpc
            if (R_c_cosmic > 1e6 and R_c_dwarf < 10.0 and R_c_spiral < 10.0):
                viable_params.append({
                    'beta': beta,
                    'M4': M4,
                    'R_c_cosmic': R_c_cosmic,
                    'R_c_dwarf': R_c_dwarf,
                    'R_c_spiral': R_c_spiral
                })
    
    if viable_params:
        print(f"\n{'=' * 80}")
        print("VIABLE PARAMETER REGIONS")
        print(f"{'=' * 80}")
        
        for p in viable_params:
            print(f"\nbeta = {p['beta']:.3f}, M4 = {p['M4']:.2e}:")
            print(f"  R_c^cosmic = {p['R_c_cosmic']:.2e} kpc (~{p['R_c_cosmic']/3e9:.1f} Mpc)")
            print(f"  R_c^dwarf = {p['R_c_dwarf']:.2f} kpc")
            print(f"  R_c^spiral = {p['R_c_spiral']:.2f} kpc")
    else:
        print(f"\n{'=' * 80}")
        print("NO VIABLE PARAMETER REGIONS FOUND")
        print(f"{'=' * 80}")
        print("May need to adjust M5 range or test different potential forms.")

if __name__ == '__main__':
    main()

