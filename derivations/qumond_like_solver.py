#!/usr/bin/env python3
"""
QUMOND-Like Solver for Σ-Gravity
================================

This implements the REVIEWER-PROOF formulation where:
- Matter couples MINIMALLY to the metric (standard geodesics, NO fifth force)
- The modification lives in the FIELD EQUATIONS via a "phantom density"
- g_eff = g_N × Σ emerges as the field solution, not as an extra force

This resolves the fifth-force problem: there IS no c²∇ln(Σ) force on particles
because matter doesn't couple to Σ in the particle action.

The QUMOND-style approach:
1. Solve ∇²Φ_N = 4πG ρ_bary (standard Newtonian from baryons)
2. Compute g_N = |∇Φ_N| and ν(g_N, r) = Σ_eff
3. Compute phantom density: ρ_phantom = (1/4πG) ∇·[(ν-1) ∇Φ_N]
4. Solve ∇²Φ = 4πG (ρ_bary + ρ_phantom)
5. Particles follow geodesics of Φ (no extra fifth force!)

Author: Leonard Speiser
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# Physical constants
G = 6.67430e-11  # m³/kg/s²
c = 2.998e8      # m/s
H0_SI = 2.27e-18  # s⁻¹ (70 km/s/Mpc)
kpc_to_m = 3.086e19
Msun = 1.989e30

# Σ-Gravity parameters
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 9.60e-11 m/s²


def poisson_fft_2d(rho: np.ndarray, dx: float) -> np.ndarray:
    """
    Solve ∇²Φ = 4πG ρ on a periodic 2D grid using FFT.
    
    Parameters
    ----------
    rho : 2D array
        Density field [kg/m³]
    dx : float
        Grid spacing [m]
    
    Returns
    -------
    phi : 2D array
        Gravitational potential [m²/s²], mean-subtracted
    """
    ny, nx = rho.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX**2 + KY**2
    
    rho_k = np.fft.fft2(rho)
    phi_k = np.zeros_like(rho_k, dtype=np.complex128)
    
    mask = k2 > 0
    phi_k[mask] = -4 * np.pi * G * rho_k[mask] / k2[mask]
    
    phi = np.real(np.fft.ifft2(phi_k))
    return phi - phi.mean()


def gradient_2d(phi: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute gradient using central differences."""
    dphidx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
    dphidy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)
    return dphidx, dphidy


def divergence_2d(fx: np.ndarray, fy: np.ndarray, dx: float) -> np.ndarray:
    """Compute divergence using central differences."""
    dfxdx = (np.roll(fx, -1, axis=1) - np.roll(fx, 1, axis=1)) / (2 * dx)
    dfydy = (np.roll(fy, -1, axis=0) - np.roll(fy, 1, axis=0)) / (2 * dx)
    return dfxdx + dfydy


def h_function(g_N: np.ndarray, g_dag: float = g_dagger) -> np.ndarray:
    """
    Universal enhancement function h(g_N).
    
    h(g) = √(g†/g) × g†/(g† + g)
    """
    g_safe = np.maximum(g_N, 1e-30)
    return np.sqrt(g_dag / g_safe) * (g_dag / (g_dag + g_safe))


def W_coherence(r: np.ndarray, xi: float) -> np.ndarray:
    """
    Coherence window W(r) = 1 - √(ξ/(ξ+r))
    
    Parameters
    ----------
    r : array
        Radius [m]
    xi : float
        Coherence scale [m], typically (2/3) × R_d
    """
    r_safe = np.maximum(r, 1e-30)
    return 1.0 - np.sqrt(xi / (xi + r_safe))


def nu_sigma_eff(g_N: np.ndarray, r: np.ndarray, 
                 A_eff: float = np.sqrt(3), 
                 xi: float = 2.0 * kpc_to_m,
                 g_dag: float = g_dagger) -> np.ndarray:
    """
    Compute ν(g_N, r) = Σ_eff = 1 + A_eff × W(r) × h(g_N)
    
    This is the QUMOND-style interpolation function.
    """
    W = W_coherence(r, xi)
    h = h_function(g_N, g_dag)
    return 1.0 + A_eff * W * h


def qumond_total_potential(rho_bary_2d: np.ndarray, 
                           dx: float,
                           A_eff: float = np.sqrt(3),
                           R_d: float = 3.0 * kpc_to_m) -> Dict:
    """
    Solve the QUMOND-like field equations for Σ-Gravity.
    
    This is the CLEAN formulation with NO fifth force:
    - Matter couples minimally (particles follow geodesics of Φ)
    - Enhancement comes from phantom density in field equations
    
    Parameters
    ----------
    rho_bary_2d : 2D array
        Baryonic density [kg/m³]
    dx : float
        Grid spacing [m]
    A_eff : float
        Enhancement amplitude (default √3)
    R_d : float
        Disk scale length [m]
    
    Returns
    -------
    dict with:
        phi_N : Newtonian potential from baryons
        phi_total : Total potential (what particles actually feel)
        rho_phantom : Phantom density that produces enhancement
        g_N : Newtonian acceleration magnitude
        nu : Enhancement factor ν(g_N, r)
        g_eff : Effective acceleration (from phi_total)
    """
    ny, nx = rho_bary_2d.shape
    xi = (2.0 / 3.0) * R_d
    
    # Build coordinate grid
    y = (np.arange(ny) - ny // 2) * dx
    x = (np.arange(nx) - nx // 2) * dx
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    
    # Step 1: Newtonian potential from baryons only
    phi_N = poisson_fft_2d(rho_bary_2d, dx)
    
    # Step 2: Compute g_N = -∇Φ_N (magnitude)
    dphi_dx, dphi_dy = gradient_2d(phi_N, dx)
    g_N_x, g_N_y = -dphi_dx, -dphi_dy
    g_N = np.sqrt(g_N_x**2 + g_N_y**2)
    
    # Step 3: Compute ν(g_N, r) = Σ_eff
    nu = nu_sigma_eff(g_N, r, A_eff=A_eff, xi=xi)
    
    # Step 4: Phantom density: ρ_phantom = (1/4πG) ∇·[(ν-1) g_N]
    # where g_N = -∇Φ_N
    phantom_flux_x = (nu - 1.0) * g_N_x
    phantom_flux_y = (nu - 1.0) * g_N_y
    rho_phantom = divergence_2d(phantom_flux_x, phantom_flux_y, dx) / (4 * np.pi * G)
    
    # Step 5: Total potential from ρ_bary + ρ_phantom
    rho_total = rho_bary_2d + rho_phantom
    phi_total = poisson_fft_2d(rho_total, dx)
    
    # Compute effective acceleration from total potential
    dphi_tot_dx, dphi_tot_dy = gradient_2d(phi_total, dx)
    g_eff_x, g_eff_y = -dphi_tot_dx, -dphi_tot_dy
    g_eff = np.sqrt(g_eff_x**2 + g_eff_y**2)
    
    return {
        'phi_N': phi_N,
        'phi_total': phi_total,
        'rho_phantom': rho_phantom,
        'rho_total': rho_total,
        'g_N': g_N,
        'g_N_x': g_N_x,
        'g_N_y': g_N_y,
        'nu': nu,
        'g_eff': g_eff,
        'g_eff_x': g_eff_x,
        'g_eff_y': g_eff_y,
        'r': r,
        'X': X,
        'Y': Y,
    }


def verify_no_fifth_force(result: Dict) -> Dict:
    """
    Verify that the QUMOND formulation has NO fifth force.
    
    In this formulation:
    - Particles follow geodesics of phi_total
    - There is NO additional a_fifth = -c²∇ln(Σ) term
    - The enhancement is ALREADY in phi_total via phantom density
    
    Compare:
    - g_eff from phi_total (what particles feel)
    - g_N × ν (the "naive" Σ-Gravity formula)
    
    They should match (no double-counting).
    """
    g_N = result['g_N']
    nu = result['nu']
    g_eff = result['g_eff']
    
    # The naive formula: g_eff = g_N × ν
    g_naive = g_N * nu
    
    # Relative difference
    mask = g_N > 1e-15  # Avoid division by zero at center
    ratio = np.ones_like(g_eff)
    ratio[mask] = g_eff[mask] / g_naive[mask]
    
    # Also compute what the "fifth force" WOULD be if we had it
    # a_fifth = c² × |∇ln(Σ)| ≈ c² × |∇ν/ν|
    # This is what we're AVOIDING by using QUMOND formulation
    
    return {
        'g_naive': g_naive,
        'ratio_eff_to_naive': ratio,
        'mean_ratio': np.mean(ratio[mask]),
        'std_ratio': np.std(ratio[mask]),
        'max_deviation': np.max(np.abs(ratio[mask] - 1.0)),
    }


def create_exponential_disk(nx: int, ny: int, dx: float,
                            M_disk: float = 5e10 * Msun,
                            R_d: float = 3.0 * kpc_to_m,
                            z_0: float = 0.3 * kpc_to_m) -> np.ndarray:
    """
    Create a 2D projection of an exponential disk.
    
    Surface density: Σ(R) = (M_disk / 2πR_d²) × exp(-R/R_d)
    """
    y = (np.arange(ny) - ny // 2) * dx
    x = (np.arange(nx) - nx // 2) * dx
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Surface density
    Sigma_0 = M_disk / (2 * np.pi * R_d**2)
    Sigma = Sigma_0 * np.exp(-R / R_d)
    
    # Convert to volume density (spread over z_0)
    rho = Sigma / (2 * z_0)
    
    return rho


def test_qumond_formulation():
    """
    Test the QUMOND-like formulation and verify no fifth force.
    """
    print("=" * 70)
    print("QUMOND-LIKE FORMULATION TEST")
    print("Testing that g_eff from field equations matches g_N × Σ")
    print("(This proves there's NO separate fifth force)")
    print("=" * 70)
    
    # Grid setup
    nx, ny = 256, 256
    L = 40 * kpc_to_m  # 40 kpc box
    dx = L / nx
    
    # Create exponential disk
    M_disk = 5e10 * Msun
    R_d = 3.0 * kpc_to_m
    rho_bary = create_exponential_disk(nx, ny, dx, M_disk=M_disk, R_d=R_d)
    
    print(f"\nGrid: {nx}×{ny}, box size: {L/kpc_to_m:.1f} kpc")
    print(f"Disk: M = {M_disk/Msun:.2e} M☉, R_d = {R_d/kpc_to_m:.1f} kpc")
    
    # Solve QUMOND-like equations
    result = qumond_total_potential(rho_bary, dx, A_eff=np.sqrt(3), R_d=R_d)
    
    # Verify no fifth force
    verification = verify_no_fifth_force(result)
    
    print(f"\n--- VERIFICATION ---")
    print(f"Mean(g_eff / g_naive): {verification['mean_ratio']:.6f}")
    print(f"Std(g_eff / g_naive):  {verification['std_ratio']:.6f}")
    print(f"Max deviation from 1:  {verification['max_deviation']:.6f}")
    
    if verification['max_deviation'] < 0.1:
        print("\n✓ SUCCESS: g_eff ≈ g_N × ν (no separate fifth force)")
        print("  The QUMOND formulation correctly produces enhanced gravity")
        print("  without any additional fifth force term.")
    else:
        print("\n⚠ WARNING: Significant deviation detected")
        print("  Check numerical resolution or boundary conditions")
    
    # Extract radial profile along x-axis
    mid_y = ny // 2
    r_profile = result['r'][mid_y, nx//2:]
    g_N_profile = result['g_N'][mid_y, nx//2:]
    g_eff_profile = result['g_eff'][mid_y, nx//2:]
    nu_profile = result['nu'][mid_y, nx//2:]
    
    # Convert to physical units
    r_kpc = r_profile / kpc_to_m
    
    # What the fifth force WOULD be (if we had the wrong formulation)
    # a_fifth = c² |∇ln(ν)| ≈ c² × (1/ν) × |∂ν/∂r|
    dnu_dr = np.gradient(nu_profile, r_profile)
    a_fifth_would_be = c**2 * np.abs(dnu_dr / nu_profile)
    
    print(f"\n--- FIFTH FORCE ANALYSIS ---")
    print(f"If we had the WRONG formulation (f(φ)L_m coupling):")
    at_5kpc = np.argmin(np.abs(r_kpc - 5.0))
    at_10kpc = np.argmin(np.abs(r_kpc - 10.0))
    print(f"  a_fifth at 5 kpc:  {a_fifth_would_be[at_5kpc]:.2e} m/s²")
    print(f"  a_fifth at 10 kpc: {a_fifth_would_be[at_10kpc]:.2e} m/s²")
    print(f"  g_N at 10 kpc:     {g_N_profile[at_10kpc]:.2e} m/s²")
    print(f"  Ratio a_fifth/g_N: {a_fifth_would_be[at_10kpc]/g_N_profile[at_10kpc]:.1f}")
    print(f"\nIn the QUMOND formulation, this fifth force DOES NOT EXIST.")
    print("The enhancement is built into the field solution, not the particle action.")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Density
    ax = axes[0, 0]
    im = ax.imshow(np.log10(rho_bary + 1e-30), extent=[-L/2/kpc_to_m, L/2/kpc_to_m,
                                                        -L/2/kpc_to_m, L/2/kpc_to_m],
                   origin='lower', cmap='viridis')
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_title('log₁₀(ρ_bary)')
    plt.colorbar(im, ax=ax)
    
    # Plot 2: Enhancement factor ν
    ax = axes[0, 1]
    im = ax.imshow(result['nu'], extent=[-L/2/kpc_to_m, L/2/kpc_to_m,
                                          -L/2/kpc_to_m, L/2/kpc_to_m],
                   origin='lower', cmap='plasma', vmin=1, vmax=3)
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_title('ν = Σ_eff(g_N, r)')
    plt.colorbar(im, ax=ax)
    
    # Plot 3: Phantom density
    ax = axes[0, 2]
    rho_ph = result['rho_phantom']
    vmax = np.percentile(np.abs(rho_ph), 99)
    im = ax.imshow(rho_ph, extent=[-L/2/kpc_to_m, L/2/kpc_to_m,
                                    -L/2/kpc_to_m, L/2/kpc_to_m],
                   origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_title('ρ_phantom (QUMOND source)')
    plt.colorbar(im, ax=ax)
    
    # Plot 4: Radial profiles
    ax = axes[1, 0]
    valid = r_kpc > 0.5
    ax.loglog(r_kpc[valid], g_N_profile[valid], 'b-', label='g_N (Newtonian)', lw=2)
    ax.loglog(r_kpc[valid], g_eff_profile[valid], 'r--', label='g_eff (total)', lw=2)
    ax.loglog(r_kpc[valid], (g_N_profile * nu_profile)[valid], 'g:', 
              label='g_N × ν (should match g_eff)', lw=2)
    ax.axhline(g_dagger, color='gray', ls=':', label=f'g† = {g_dagger:.2e}')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('g [m/s²]')
    ax.set_title('Acceleration Profiles')
    ax.legend()
    ax.set_xlim(0.5, 20)
    
    # Plot 5: Enhancement ratio
    ax = axes[1, 1]
    ax.semilogx(r_kpc[valid], nu_profile[valid], 'b-', lw=2, label='ν = g_eff/g_N')
    ax.semilogx(r_kpc[valid], g_eff_profile[valid]/g_N_profile[valid], 'r--', 
                lw=2, label='Actual g_eff/g_N')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('Enhancement factor')
    ax.set_title('Enhancement: ν vs actual ratio')
    ax.legend()
    ax.set_xlim(0.5, 20)
    ax.set_ylim(1, 3)
    
    # Plot 6: Fifth force comparison
    ax = axes[1, 2]
    ax.loglog(r_kpc[valid], g_eff_profile[valid], 'b-', lw=2, 
              label='g_eff (what particles feel)')
    ax.loglog(r_kpc[valid], a_fifth_would_be[valid], 'r--', lw=2,
              label='a_fifth (if WRONG formulation)')
    ax.axhline(g_dagger, color='gray', ls=':', label='g†')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('Acceleration [m/s²]')
    ax.set_title('Fifth Force: AVOIDED in QUMOND')
    ax.legend()
    ax.set_xlim(0.5, 20)
    
    plt.tight_layout()
    plt.savefig('derivations/qumond_formulation_test.png', dpi=150)
    print(f"\nFigure saved: derivations/qumond_formulation_test.png")
    plt.close()
    
    return result, verification


if __name__ == "__main__":
    result, verification = test_qumond_formulation()

