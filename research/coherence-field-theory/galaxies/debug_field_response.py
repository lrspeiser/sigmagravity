"""
Debug field response to baryons.

Check if coupling term is working and field responds to baryon density.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from galaxies.halo_field_profile import HaloFieldSolver

# Test parameters
V0 = 1e-6
lambda_param = 1.0
beta_values = [0.1, 1.0, 10.0, 100.0]

# Test baryon profile (exponential disk)
M_disk = 1e8  # M_sun
R_disk = 1.0  # kpc

def rho_disk(r):
    """Exponential disk density."""
    rho_0 = M_disk / (2 * np.pi * R_disk**2)
    return rho_0 * np.exp(-r / R_disk)

# Radial grid
r_grid = np.logspace(-1, 1, 100)  # 0.1 to 10 kpc

print("=" * 80)
print("DEBUGGING FIELD RESPONSE TO BARYONS")
print("=" * 80)

print(f"\nTest parameters:")
print(f"  V0 = {V0:.2e}")
print(f"  lambda = {lambda_param:.2f}")
print(f"  M_disk = {M_disk:.2e} M_sun")
print(f"  R_disk = {R_disk:.2f} kpc")

# Check baryon density
rho_b_vals = np.array([rho_disk(r) for r in r_grid])
print(f"\nBaryon density:")
print(f"  max(rho_b) = {np.max(rho_b_vals):.2e} M_sun/kpc^3")
print(f"  min(rho_b) = {np.min(rho_b_vals):.2e} M_sun/kpc^3")
print(f"  rho_b(0) = {rho_b_vals[0]:.2e} M_sun/kpc^3")
print(f"  rho_b(10 kpc) = {rho_b_vals[-1]:.2e} M_sun/kpc^3")

# Check V(φ) scale
solver_test = HaloFieldSolver(V0, lambda_param, 1.0, M4=None, phi_inf=0.0)
V_scale = solver_test.V_func(0.0)
print(f"\nPotential scale:")
print(f"  V(0) = {V_scale:.6e} (cosmology units)")

# Critical density for conversion
G = 4.30091e-6
H0_kms_kpc = 70.0 / 306.6
H0_squared = H0_kms_kpc**2
rho_crit = 3 * H0_squared / (8 * np.pi * G)
print(f"  rho_crit = {rho_crit:.2e} M_sun/kpc^3")
print(f"  V(0) * rho_crit = {V_scale * rho_crit:.2e} M_sun/kpc^3")

# Check coupling term scale
rho_b_max = np.max(rho_b_vals)
rho_b_cosm_max = rho_b_max / rho_crit
print(f"\nCoupling term scale (for beta=1):")
print(f"  rho_b_max / rho_crit = {rho_b_cosm_max:.2e}")
print(f"  beta * (rho_b / rho_crit) = {1.0 * rho_b_cosm_max:.6e} (cosmology units)")
print(f"  Ratio: coupling / V(0) = {rho_b_cosm_max / V_scale:.2e}")

print(f"\n{'=' * 80}")
print("Testing different beta values:")
print(f"{'=' * 80}")

results = []

for beta in beta_values:
    print(f"\n--- beta = {beta:.1f} ---")
    
    solver = HaloFieldSolver(V0, lambda_param, beta, M4=None, phi_inf=0.0)
    
    # Solve
    try:
        phi, dphi_dr = solver.solve(rho_disk, r_grid, method='shooting')
        
        # Check field variation
        phi_range = np.max(phi) - np.min(phi)
        dphi_dr_max = np.max(np.abs(dphi_dr))
        
        print(f"  Field solution:")
        print(f"    phi(0) = {phi[0]:.6f}")
        print(f"    phi(inf) = {phi[-1]:.6f}")
        print(f"    phi range = {phi_range:.6e}")
        print(f"    max(|dphi/dr|) = {dphi_dr_max:.6e}")
        
        # Effective density
        rho_phi = solver.effective_density(phi, dphi_dr, convert_to_mass_density=True)
        rho_phi_max = np.max(rho_phi)
        rho_phi_min = np.min(rho_phi)
        rho_phi_range = rho_phi_max - rho_phi_min
        
        print(f"  Effective density:")
        print(f"    min(ρ_φ) = {rho_phi_min:.2e} M_sun/kpc^3")
        print(f"    max(ρ_φ) = {rho_phi_max:.2e} M_sun/kpc^3")
        print(f"    range = {rho_phi_range:.2e} M_sun/kpc^3")
        print(f"    variation = {rho_phi_range / rho_phi_max * 100:.1f}%")
        
        # Check Veff at different radii
        r_test = [0.1, 1.0, 10.0]
        print(f"  Veff at different radii:")
        for r in r_test:
            rho_b_r = rho_disk(r)
            phi_r = np.interp(r, r_grid, phi)
            Veff_r = solver.Veff(phi_r, rho_b_r)
            V_r = solver.V_func(phi_r)
            coupling_r = beta * (rho_b_r / rho_crit)
            print(f"    r = {r:.1f} kpc:")
            print(f"      rho_b = {rho_b_r:.2e} M_sun/kpc^3")
            print(f"      V(phi) = {V_r:.6e}")
            print(f"      coupling = {coupling_r:.6e}")
            print(f"      Veff = {Veff_r:.6e}")
            print(f"      coupling/V = {coupling_r / V_r:.2f}")
        
        results.append({
            'beta': beta,
            'phi_range': phi_range,
            'dphi_dr_max': dphi_dr_max,
            'rho_phi_max': rho_phi_max,
            'rho_phi_min': rho_phi_min,
            'rho_phi_range': rho_phi_range
        })
        
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        results.append({
            'beta': beta,
            'error': str(e)
        })

print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")

print(f"\n{'beta':<10} {'phi range':<15} {'max(dphi/dr)':<15} {'rho_phi max':<15} {'rho_phi range':<15}")
print("-" * 80)

for r in results:
    if 'error' not in r:
        print(f"{r['beta']:<10.1f} {r['phi_range']:<15.6e} {r['dphi_dr_max']:<15.6e} "
              f"{r['rho_phi_max']:<15.2e} {r['rho_phi_range']:<15.2e}")
    else:
        print(f"{r['beta']:<10.1f} ERROR: {r['error']}")

print(f"\n{'=' * 80}")

# Find best β
if len(results) > 0:
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['rho_phi_range'])
        print(f"\nBest beta for field response: {best['beta']:.1f}")
        print(f"  Produces rho_phi range: {best['rho_phi_range']:.2e} M_sun/kpc^3")

