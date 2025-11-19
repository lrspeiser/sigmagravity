"""
Debug unit conversion in halo solver.

Check what values we're getting and if conversion is correct.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from galaxies.halo_field_profile import HaloFieldSolver

# Physical constants
G = 4.30091e-6  # (km/s)^2 kpc / M_sun

# Test parameters
V0 = 1e-6
lambda_param = 1.0
beta = 0.1

# Test baryon profile (simple exponential disk)
M_disk = 1e8  # M_sun
R_disk = 1.0  # kpc

def rho_disk(r):
    """Exponential disk density."""
    rho_0 = M_disk / (2 * np.pi * R_disk**2)
    return rho_0 * np.exp(-r / R_disk)

# Radial grid
r_grid = np.logspace(-1, 1, 100)  # 0.1 to 10 kpc

print("=" * 80)
print("DEBUGGING UNIT CONVERSION")
print("=" * 80)

print(f"\nTest parameters:")
print(f"  V0 = {V0:.2e}")
print(f"  lambda = {lambda_param:.2f}")
print(f"  beta = {beta:.2f}")
print(f"  M_disk = {M_disk:.2e} M_sun")
print(f"  R_disk = {R_disk:.2f} kpc")

# Create solver
solver = HaloFieldSolver(V0, lambda_param, beta, M4=None, phi_inf=0.0)

# Solve
print(f"\nSolving scalar field...")
phi, dphi_dr = solver.solve(rho_disk, r_grid, method='shooting')

print(f"\nField solution:")
print(f"  phi(0) = {phi[0]:.6f}")
print(f"  phi(inf) = {phi[-1]:.6f}")
print(f"  max(|dphi/dr|) = {np.max(np.abs(dphi_dr)):.6e}")

# Check V(φ) values
V_vals = solver.V_func(phi)
print(f"\nPotential values:")
print(f"  V(phi(0)) = {V_vals[0]:.6e}")
print(f"  V(phi(inf)) = {V_vals[-1]:.6e}")
print(f"  max(V) = {np.max(V_vals):.6e}")

# Effective density WITHOUT conversion
rho_phi_raw = solver.effective_density(phi, dphi_dr, convert_to_mass_density=False)
print(f"\nEffective density (raw, no conversion):")
print(f"  rho_phi(0) = {rho_phi_raw[0]:.6e}")
print(f"  rho_phi(inf) = {rho_phi_raw[-1]:.6e}")
print(f"  max(rho_phi) = {np.max(rho_phi_raw):.6e}")

# Effective density WITH conversion
rho_phi_conv = solver.effective_density(phi, dphi_dr, convert_to_mass_density=True)
print(f"\nEffective density (with conversion):")
print(f"  rho_phi(0) = {rho_phi_conv[0]:.2e} M_sun/kpc^3")
print(f"  rho_phi(inf) = {rho_phi_conv[-1]:.2e} M_sun/kpc^3")
print(f"  max(rho_phi) = {np.max(rho_phi_conv):.2e} M_sun/kpc^3")

# Check conversion factor
H0_kms_kpc = 70.0 / 306.6
H0_squared = H0_kms_kpc**2
rho_crit = 3 * H0_squared / (8 * np.pi * G)
print(f"\nConversion factor:")
print(f"  H0 = {H0_kms_kpc:.6f} km/s/kpc")
print(f"  H0² = {H0_squared:.6e} (km/s)²/kpc²")
print(f"  rho_crit = {rho_crit:.2e} M_sun/kpc^3")
print(f"  V0 * rho_crit = {V0 * rho_crit:.2e} M_sun/kpc^3")

# Compare with typical halo density
print(f"\nTypical halo density (for comparison):")
print(f"  Expected rho_c0 ~ 1e6 - 1e8 M_sun/kpc^3 for dwarf galaxies")
print(f"  CamB fitted: rho_c0 = 1.05 M_sun/kpc^3 (very low!)")
print(f"  Our converted: rho_phi(0) = {rho_phi_conv[0]:.2e} M_sun/kpc^3")

# Check if we need different scaling
if rho_phi_conv[0] > 1e6:
    scale_factor = 1e6 / rho_phi_conv[0]
    print(f"\n[ISSUE] Density is {rho_phi_conv[0]/1e6:.1f}x too high")
    print(f"  Would need to scale by {scale_factor:.2e} to get ~1e6 M_sun/kpc^3")
elif rho_phi_conv[0] < 1e3:
    scale_factor = 1e6 / rho_phi_conv[0]
    print(f"\n[ISSUE] Density is {1e6/rho_phi_conv[0]:.1f}x too low")
    print(f"  Would need to scale by {scale_factor:.2e} to get ~1e6 M_sun/kpc^3")

print("\n" + "=" * 80)

