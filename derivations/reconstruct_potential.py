#!/usr/bin/env python3
"""
Reconstruct V(φ_C) from Target Σ Profile
========================================

If you insist on keeping the f(φ_C)L_m formulation, you MUST specify V(φ_C).

This script INVERTS the static field equation to reconstruct V(φ) that
produces your target Σ(r) profile. This forces you to confront whether
V(φ) is physically sensible (bounded below, stable, etc.).

The field equation (quasi-static limit):
    ∇²φ - V'(φ) = (2φ/M²) ρ c²

Given:
- Target Σ(r) from your phenomenological formula
- Coupling f(φ) = 1 + φ²/M² = Σ  →  φ = M√(Σ-1)

We can compute:
- φ(r) from Σ(r)
- ∇²φ from φ(r)
- V'(φ) = ∇²φ - (2φ/M²) ρ c²
- V(φ) by integration

Author: Leonard Speiser
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

# Physical constants
G = 6.67430e-11  # m³/kg/s²
c = 2.998e8      # m/s
H0_SI = 2.27e-18  # s⁻¹
kpc_to_m = 3.086e19
Msun = 1.989e30

# Σ-Gravity parameters
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))


def laplacian_spherical(phi: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Compute ∇²φ in spherical symmetry.
    
    ∇²φ = (1/r²) d/dr(r² dφ/dr) = d²φ/dr² + (2/r) dφ/dr
    """
    # First derivative
    dphi_dr = np.gradient(phi, r)
    
    # Second derivative
    d2phi_dr2 = np.gradient(dphi_dr, r)
    
    # Laplacian in spherical coords
    r_safe = np.maximum(r, r[1])  # Avoid division by zero
    lap = d2phi_dr2 + 2 * dphi_dr / r_safe
    
    return lap


def mass_enclosed_spherical(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Compute M(<r) for spherical density profile."""
    dr = np.gradient(r)
    dM = 4 * np.pi * r**2 * rho * dr
    return np.cumsum(dM)


def g_N_spherical(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Compute Newtonian acceleration from spherical density."""
    M_enc = mass_enclosed_spherical(r, rho)
    r_safe = np.maximum(r, r[1])
    return G * M_enc / r_safe**2


def h_function(g_N: np.ndarray) -> np.ndarray:
    """h(g) = √(g†/g) × g†/(g† + g)"""
    g_safe = np.maximum(g_N, 1e-30)
    return np.sqrt(g_dagger / g_safe) * (g_dagger / (g_dagger + g_safe))


def W_coherence(r: np.ndarray, xi: float) -> np.ndarray:
    """W(r) = 1 - √(ξ/(ξ+r))"""
    r_safe = np.maximum(r, 1e-30)
    return 1.0 - np.sqrt(xi / (xi + r_safe))


def sigma_target(g_N: np.ndarray, r: np.ndarray, 
                 A: float = np.sqrt(3), 
                 xi: float = 2.0 * kpc_to_m) -> np.ndarray:
    """
    Target Σ(g_N, r) = 1 + A × W(r) × h(g_N)
    """
    W = W_coherence(r, xi)
    h = h_function(g_N)
    return 1.0 + A * W * h


def reconstruct_V_from_target(r: np.ndarray, 
                              rho: np.ndarray, 
                              M_scale: float,
                              A: float = np.sqrt(3),
                              R_d: float = 3.0 * kpc_to_m) -> Dict:
    """
    Reconstruct V(φ) from target Σ(r) profile.
    
    Parameters
    ----------
    r : array
        Radial grid [m]
    rho : array
        Density profile [kg/m³]
    M_scale : float
        Coupling mass scale in the relation f = 1 + φ²/M²
    A : float
        Enhancement amplitude
    R_d : float
        Disk scale length [m]
    
    Returns
    -------
    dict with reconstructed potential and diagnostics
    """
    xi = (2.0 / 3.0) * R_d
    
    # Step 1: Compute g_N from density
    g_N = g_N_spherical(r, rho)
    
    # Step 2: Target Σ(r)
    Sigma = sigma_target(g_N, r, A=A, xi=xi)
    
    # Step 3: φ(r) from Σ = 1 + φ²/M²
    phi_r = M_scale * np.sqrt(np.maximum(Sigma - 1.0, 0.0))
    
    # Step 4: Compute ∇²φ
    lap_phi = laplacian_spherical(phi_r, r)
    
    # Step 5: Source term from matter coupling
    # Field equation: ∇²φ = V'(φ) + (2φ/M²) ρ c²
    source = (2.0 * phi_r / M_scale**2) * rho * c**2
    
    # Step 6: Reconstructed V'(φ)
    Vprime_r = lap_phi - source
    
    # Step 7: Convert to V'(φ) as function of φ (not r)
    # Sort by φ value to get single-valued function
    idx = np.argsort(phi_r)
    phi_sorted = phi_r[idx]
    Vprime_sorted = Vprime_r[idx]
    
    # Remove duplicate φ values (average V' at same φ)
    phi_unique, unique_idx = np.unique(phi_sorted, return_index=True)
    Vprime_unique = Vprime_sorted[unique_idx]
    
    # Step 8: Integrate to get V(φ)
    # V(φ) = ∫ V'(φ) dφ
    V_unique = np.zeros_like(phi_unique)
    for i in range(1, len(phi_unique)):
        dphi = phi_unique[i] - phi_unique[i-1]
        V_unique[i] = V_unique[i-1] + 0.5 * (Vprime_unique[i] + Vprime_unique[i-1]) * dphi
    
    # Step 9: Compute V''(φ) for stability analysis
    Vdoubleprime = np.gradient(Vprime_unique, phi_unique)
    
    return {
        # Spatial profiles
        'r': r,
        'rho': rho,
        'g_N': g_N,
        'Sigma': Sigma,
        'phi_r': phi_r,
        'lap_phi': lap_phi,
        'source': source,
        'Vprime_r': Vprime_r,
        
        # As function of φ
        'phi': phi_unique,
        'Vprime': Vprime_unique,
        'V': V_unique,
        'Vdoubleprime': Vdoubleprime,
        
        # Parameters
        'M_scale': M_scale,
        'A': A,
        'xi': xi,
    }


def analyze_potential_stability(result: Dict) -> Dict:
    """
    Analyze whether the reconstructed V(φ) is physically sensible.
    
    Checks:
    1. Is V bounded from below?
    2. Is V'' > 0 (stable minimum)?
    3. What is the effective mass m_eff² = V''?
    """
    phi = result['phi']
    V = result['V']
    Vprime = result['Vprime']
    Vpp = result['Vdoubleprime']
    
    # Check 1: Bounded from below
    V_min = np.min(V)
    is_bounded = V_min > -np.inf
    
    # Check 2: Stability (V'' > 0 at minimum)
    min_idx = np.argmin(V)
    Vpp_at_min = Vpp[min_idx]
    is_stable = Vpp_at_min > 0
    
    # Check 3: Effective mass
    # m_eff² = V''(φ_min) in natural units
    m_eff_sq = Vpp_at_min
    
    # Check 4: Does V have a sensible shape?
    # Look for pathologies like unbounded growth, oscillations, etc.
    V_range = np.max(V) - np.min(V)
    phi_range = np.max(phi) - np.min(phi)
    
    # Typical potential forms for comparison
    # Mexican hat: V = λ(φ² - v²)²
    # Quadratic: V = m²φ²/2
    
    return {
        'V_min': V_min,
        'phi_at_V_min': phi[min_idx],
        'is_bounded': is_bounded,
        'Vpp_at_min': Vpp_at_min,
        'is_stable': is_stable,
        'm_eff_sq': m_eff_sq,
        'V_range': V_range,
        'phi_range': phi_range,
    }


def create_hernquist_profile(r: np.ndarray, 
                             M_total: float = 5e10 * Msun,
                             a: float = 3.0 * kpc_to_m) -> np.ndarray:
    """
    Hernquist density profile (good approximation for bulge+disk).
    
    ρ(r) = M a / (2π r (r+a)³)
    """
    r_safe = np.maximum(r, r[1])
    return M_total * a / (2 * np.pi * r_safe * (r_safe + a)**3)


def create_exponential_disk_spherical(r: np.ndarray,
                                      M_disk: float = 5e10 * Msun,
                                      R_d: float = 3.0 * kpc_to_m,
                                      z_0: float = 0.3 * kpc_to_m) -> np.ndarray:
    """
    Spherically-averaged exponential disk density.
    
    This is an approximation: we spread the disk mass in a shell.
    """
    # Surface density: Σ(R) = (M/2πR_d²) exp(-R/R_d)
    # Spherically averaged: ρ(r) ≈ Σ(r) / (2z_0) × exp(-r/R_d)
    Sigma_0 = M_disk / (2 * np.pi * R_d**2)
    return Sigma_0 * np.exp(-r / R_d) / (2 * z_0)


def test_potential_reconstruction():
    """
    Test the potential reconstruction and analyze results.
    """
    print("=" * 70)
    print("POTENTIAL RECONSTRUCTION TEST")
    print("Inverting field equation to find V(φ) that produces target Σ(r)")
    print("=" * 70)
    
    # Create radial grid
    r_min = 0.1 * kpc_to_m
    r_max = 30 * kpc_to_m
    n_r = 500
    r = np.linspace(r_min, r_max, n_r)
    
    # Create density profile
    M_disk = 5e10 * Msun
    R_d = 3.0 * kpc_to_m
    rho = create_exponential_disk_spherical(r, M_disk=M_disk, R_d=R_d)
    
    # Coupling mass scale (this is a free parameter)
    # Choose M_scale so that φ ~ O(1) in dimensionless units
    M_scale = 1.0  # Natural units where φ is dimensionless
    
    print(f"\nDensity profile: Exponential disk")
    print(f"  M_disk = {M_disk/Msun:.2e} M☉")
    print(f"  R_d = {R_d/kpc_to_m:.1f} kpc")
    print(f"  M_scale = {M_scale}")
    
    # Reconstruct potential
    result = reconstruct_V_from_target(r, rho, M_scale, A=np.sqrt(3), R_d=R_d)
    
    # Analyze stability
    stability = analyze_potential_stability(result)
    
    print(f"\n--- RECONSTRUCTED POTENTIAL ANALYSIS ---")
    print(f"V(φ) range: [{np.min(result['V']):.2e}, {np.max(result['V']):.2e}]")
    print(f"φ range: [{np.min(result['phi']):.4f}, {np.max(result['phi']):.4f}]")
    print(f"V_min = {stability['V_min']:.2e} at φ = {stability['phi_at_V_min']:.4f}")
    print(f"V''(φ_min) = {stability['Vpp_at_min']:.2e}")
    print(f"Is bounded from below: {stability['is_bounded']}")
    print(f"Is stable (V'' > 0): {stability['is_stable']}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Density and Σ profiles
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.semilogy(r / kpc_to_m, rho, 'b-', lw=2, label='ρ(r)')
    ax2.plot(r / kpc_to_m, result['Sigma'], 'r-', lw=2, label='Σ(r)')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('ρ [kg/m³]', color='b')
    ax2.set_ylabel('Σ', color='r')
    ax.set_title('Density and Enhancement')
    ax.legend(loc='upper right')
    ax2.legend(loc='center right')
    
    # Plot 2: φ(r) profile
    ax = axes[0, 1]
    ax.plot(r / kpc_to_m, result['phi_r'], 'b-', lw=2)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('φ(r)')
    ax.set_title('Field profile φ(r) = M√(Σ-1)')
    
    # Plot 3: Components of field equation
    ax = axes[0, 2]
    ax.plot(r / kpc_to_m, result['lap_phi'], 'b-', lw=2, label='∇²φ')
    ax.plot(r / kpc_to_m, result['source'], 'r--', lw=2, label='(2φ/M²)ρc²')
    ax.plot(r / kpc_to_m, result['Vprime_r'], 'g:', lw=2, label="V'(φ)")
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('Value')
    ax.set_title('Field equation components')
    ax.legend()
    ax.set_yscale('symlog', linthresh=1e10)
    
    # Plot 4: V(φ) reconstructed
    ax = axes[1, 0]
    ax.plot(result['phi'], result['V'], 'b-', lw=2)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('φ')
    ax.set_ylabel('V(φ)')
    ax.set_title('Reconstructed potential V(φ)')
    
    # Plot 5: V'(φ)
    ax = axes[1, 1]
    ax.plot(result['phi'], result['Vprime'], 'b-', lw=2)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('φ')
    ax.set_ylabel("V'(φ)")
    ax.set_title("Potential derivative V'(φ)")
    
    # Plot 6: V''(φ) - stability
    ax = axes[1, 2]
    ax.plot(result['phi'], result['Vdoubleprime'], 'b-', lw=2)
    ax.axhline(0, color='r', ls='--', alpha=0.5, label='V\'\'=0 (marginal)')
    ax.set_xlabel('φ')
    ax.set_ylabel("V''(φ)")
    ax.set_title("Stability: V''(φ) > 0 required")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('derivations/potential_reconstruction_test.png', dpi=150)
    print(f"\nFigure saved: derivations/potential_reconstruction_test.png")
    plt.close()
    
    # Summary assessment
    print("\n" + "=" * 70)
    print("SUMMARY: Is the reconstructed V(φ) physically sensible?")
    print("=" * 70)
    
    if stability['is_bounded'] and stability['is_stable']:
        print("✓ V(φ) is bounded from below and has a stable minimum")
        print("  This suggests the f(φ)L_m formulation COULD be consistent")
    else:
        issues = []
        if not stability['is_bounded']:
            issues.append("V(φ) is NOT bounded from below")
        if not stability['is_stable']:
            issues.append("V(φ) does NOT have a stable minimum (V'' < 0)")
        print("⚠ PROBLEMS DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n  This suggests the f(φ)L_m formulation may be INCONSISTENT")
        print("  Consider switching to the QUMOND-like formulation instead")
    
    return result, stability


if __name__ == "__main__":
    result, stability = test_potential_reconstruction()

