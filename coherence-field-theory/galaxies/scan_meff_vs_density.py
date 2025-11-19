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

def compute_meff_and_Rc(V0, lambda_param, beta, M4, rho_b, phi_fixed=None):
    """
    Compute m_eff and R_c for given parameters.
    
    If phi_fixed is None, solves for phi_min that minimizes V_eff.
    """
    # HaloFieldSolver uses M4 parameter, where M4^5 gives the M^5 term in chameleon potential
    # Use a default phi_inf for initialization
    phi_inf_default = 0.05 if phi_fixed is None else phi_fixed
    solver = HaloFieldSolver(V0, lambda_param, beta, M4=M4, phi_inf=phi_inf_default)
    
    # Find phi that minimizes V_eff for this density
    if phi_fixed is None and M4 is not None:
        # Chameleon: solve for phi_min
        phi = solver.find_phi_min(rho_b)
    else:
        # No chameleon or fixed phi: use provided/default value
        phi = phi_fixed if phi_fixed is not None else phi_inf_default
    
    # Compute effective mass squared at this phi
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
    
    print(f"\nTest densities:")
    print(f"  Cosmic: {rho_cosmic:.2e} M_sun/kpc^3")
    print(f"  Dwarf: {rho_dwarf:.2e} M_sun/kpc^3")
    print(f"  Spiral: {rho_spiral:.2e} M_sun/kpc^3")
    print(f"\nNote: For chameleon (M4 != None), solving for phi_min at each density.")
    print(f"      For pure exponential (M4 = None), using fixed phi = 0.05")
    
    # Scan parameters
    beta_values = [0.01, 0.05, 0.1]
    # Test wider M4 range - chameleon needs to be strong enough
    M4_values = [None, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # None = pure exponential, else chameleon
    
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    
    print(f"\n{'beta':<10} {'M4':<12} {'phi^cosmic':<12} {'phi^dwarf':<12} {'phi^spiral':<12} "
          f"{'R_c^cosmic':<15} {'R_c^dwarf':<15} {'R_c^spiral':<15}")
    print("-" * 120)
    print("Note: R_c in kpc. Target: R_c^cosmic >> 1e6 kpc, R_c^galaxy < 10 kpc")
    
    viable_params = []
    
    for beta in beta_values:
        for M4 in M4_values:
            # Compute at different densities
            # For chameleon, phi_min will be solved automatically
            solver_cosmic = HaloFieldSolver(V0, lambda_param, beta, M4=M4, phi_inf=0.05)
            solver_dwarf = HaloFieldSolver(V0, lambda_param, beta, M4=M4, phi_inf=0.05)
            solver_spiral = HaloFieldSolver(V0, lambda_param, beta, M4=M4, phi_inf=0.05)
            
            # Get phi_min values
            if M4 is not None:
                phi_cosmic = solver_cosmic.find_phi_min(rho_cosmic)
                phi_dwarf = solver_dwarf.find_phi_min(rho_dwarf)
                phi_spiral = solver_spiral.find_phi_min(rho_spiral)
            else:
                phi_cosmic = phi_dwarf = phi_spiral = 0.05
            
            m_eff_cosmic, R_c_cosmic, _ = compute_meff_and_Rc(
                V0, lambda_param, beta, M4, rho_cosmic, phi_fixed=phi_cosmic
            )
            m_eff_dwarf, R_c_dwarf, _ = compute_meff_and_Rc(
                V0, lambda_param, beta, M4, rho_dwarf, phi_fixed=phi_dwarf
            )
            m_eff_spiral, R_c_spiral, _ = compute_meff_and_Rc(
                V0, lambda_param, beta, M4, rho_spiral, phi_fixed=phi_spiral
            )
            
            M4_str = "None" if M4 is None else f"{M4:.2e}"
            print(f"{beta:<10.3f} {M4_str:<12} {phi_cosmic:<12.6f} {phi_dwarf:<12.6f} {phi_spiral:<12.6f} "
                  f"{R_c_cosmic:<15.2e} {R_c_dwarf:<15.2f} {R_c_spiral:<15.2f}")
            
            # Check if viable: R_c^cosmic >> R_c^galaxy
            # Target: R_c^cosmic ~ 10^4 Mpc ~ 3e9 kpc, R_c^galaxy ~ 1-5 kpc
            # Relaxed: R_c^cosmic > 1e5 kpc, R_c^galaxy < 50 kpc (still much better than current)
            if (R_c_cosmic > 1e5 and R_c_dwarf < 50.0 and R_c_spiral < 50.0):
                viable_params.append({
                    'beta': beta,
                    'M4': M4,
                    'phi_cosmic': phi_cosmic,
                    'phi_dwarf': phi_dwarf,
                    'phi_spiral': phi_spiral,
                    'R_c_cosmic': R_c_cosmic,
                    'R_c_dwarf': R_c_dwarf,
                    'R_c_spiral': R_c_spiral
                })
    
    if viable_params:
        print(f"\n{'=' * 80}")
        print("VIABLE PARAMETER REGIONS")
        print(f"{'=' * 80}")
        
        for p in viable_params:
            M4_str = "None" if p['M4'] is None else f"{p['M4']:.2e}"
            print(f"\nbeta = {p['beta']:.3f}, M4 = {M4_str}:")
            print(f"  phi^cosmic = {p['phi_cosmic']:.6f}, phi^dwarf = {p['phi_dwarf']:.6f}, phi^spiral = {p['phi_spiral']:.6f}")
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

