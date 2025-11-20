"""
Symmetron / Landau-Ginzburg Potential for Coherence Gravity
============================================================

This is Approach C: environment-dependent spontaneous symmetry breaking.

Physical Picture:
-----------------
The coherence field φ has a double-well potential that couples to
local matter density:

    V_eff(φ, ρ) = V₀ + (1/2)[ρ/M² - μ²]φ² + (λ/4)φ⁴

Key regimes:

1. HIGH DENSITY (Solar System, galaxy centers):
   ρ > ρ_crit = μ²M² → effective mass² > 0
   → minimum at φ = 0 (SCREENED)
   → no extra gravity, passes PPN

2. LOW DENSITY (galaxy outskirts, cosmic voids):
   ρ < ρ_crit → effective mass² < 0
   → minima at φ = ±φ₀ where φ₀ = μ√(2/λ) (UNSCREENED)
   → coherence builds, extra gravity kicks in

3. COSMOLOGY:
   φ evolves with ρ_cosmic(a) → can drive late-time acceleration

This is the SAME structure as symmetron dark energy models, but here
φ represents gravitational wave coherence order parameter.

Connection to your phenomenology:
---------------------------------
Your K(R) kernel emerges as:

    K(R) ≈ (β/M_Pl²) φ²(R)

where φ(R) transitions from 0 (center) to φ₀ (edge) on scale R_c.

The screening radius R_c emerges naturally from solving the field
equation, not imposed by hand.
"""

import numpy as np
from typing import Tuple, Callable
from dataclasses import dataclass


@dataclass
class SymmetronParams:
    """
    Parameters for symmetron/Landau-Ginzburg potential.
    
    Physical meanings:
    ------------------
    mu : Bare mass scale [eV or normalized]
        Sets the VEV φ₀ = μ/√λ in vacuum
        
    lambda_self : Self-interaction coupling (dimensionless)
        Controls steepness of quartic term
        
    M : Coupling scale to matter [M_Pl or normalized]
        Ratio ρ/M² determines effective mass
        
    V0 : Vacuum energy offset
        Can tune this for cosmology (like Λ)
        
    beta : Coupling to metric
        Effective gravity: g_eff = g_N(1 + β φ²/M_Pl²)
    """
    mu: float = 2.4e-3  # Bare mass [eV], typical for dark energy
    lambda_self: float = 1.0  # Self-coupling
    M: float = 2.4e-3  # Matter coupling [M_Pl units]
    V0: float = 0.0  # Vacuum energy offset
    beta: float = 1.0  # Metric coupling strength


def potential(phi: np.ndarray, params: SymmetronParams) -> np.ndarray:
    """
    Bare potential (no density coupling).
    
    V(φ) = V₀ - (1/2)μ² φ² + (λ/4) φ⁴
    
    Note the NEGATIVE quadratic term → unstable at φ=0 in vacuum.
    This drives spontaneous symmetry breaking.
    """
    return (params.V0 
            - 0.5 * params.mu**2 * phi**2 
            + 0.25 * params.lambda_self * phi**4)


def potential_eff(phi: np.ndarray, rho: np.ndarray, 
                  params: SymmetronParams) -> np.ndarray:
    """
    Effective potential including matter coupling.
    
    V_eff(φ, ρ) = V₀ + (1/2)[ρ/M² - μ²]φ² + (λ/4)φ⁴
    
    When ρ > ρ_crit = μ²M²:
        Effective mass² = ρ/M² - μ² > 0
        → minimum at φ = 0 (screened)
        
    When ρ < ρ_crit:
        Effective mass² < 0
        → minima at φ = ±√[(μ² - ρ/M²)/(λ/2)]
    """
    rho = np.atleast_1d(rho)
    phi = np.atleast_1d(phi)
    
    m_eff_sq = rho / params.M**2 - params.mu**2
    
    return (params.V0 
            + 0.5 * m_eff_sq * phi**2 
            + 0.25 * params.lambda_self * phi**4)


def dV_dphi(phi: np.ndarray, rho: np.ndarray, 
            params: SymmetronParams) -> np.ndarray:
    """
    First derivative of effective potential.
    
    dV_eff/dφ = [ρ/M² - μ²]φ + λφ³
    
    This is the "force" term in Klein-Gordon equation.
    """
    rho = np.atleast_1d(rho)
    phi = np.atleast_1d(phi)
    
    m_eff_sq = rho / params.M**2 - params.mu**2
    
    return m_eff_sq * phi + params.lambda_self * phi**3


def d2V_dphi2(phi: np.ndarray, rho: np.ndarray,
              params: SymmetronParams) -> np.ndarray:
    """
    Second derivative (effective mass squared).
    
    d²V_eff/dφ² = ρ/M² - μ² + 3λφ²
    
    This determines:
    - Screening: large m_eff² → short range
    - Stability: negative → tachyonic
    """
    rho = np.atleast_1d(rho)
    phi = np.atleast_1d(phi)
    
    m_eff_sq = rho / params.M**2 - params.mu**2
    
    return m_eff_sq + 3 * params.lambda_self * phi**2


def phi_minimum(rho: np.ndarray, params: SymmetronParams) -> np.ndarray:
    """
    Minimum of effective potential at given density.
    
    For ρ > ρ_crit = μ²M²:
        φ_min = 0 (screened)
        
    For ρ < ρ_crit:
        φ_min = √[(μ² - ρ/M²) / (λ/2)]
        
    This is the "equilibrium coherence" at each density.
    """
    rho = np.atleast_1d(rho)
    rho_crit = params.mu**2 * params.M**2
    
    # Screened regime (high density)
    phi_min = np.zeros_like(rho)
    
    # Unscreened regime (low density)
    mask = rho < rho_crit
    if np.any(mask):
        numerator = params.mu**2 - rho[mask] / params.M**2
        phi_min[mask] = np.sqrt(numerator / (0.5 * params.lambda_self))
    
    return phi_min


def critical_density(params: SymmetronParams) -> float:
    """
    Critical density for screening transition.
    
    ρ_crit = μ² M²
    
    Above this density: φ = 0 (screened)
    Below this density: φ ≠ 0 (coherence builds)
    """
    return params.mu**2 * params.M**2


def screening_radius_estimate(rho_center: float, rho_edge: float,
                               R_scale: float, params: SymmetronParams) -> float:
    """
    Estimate radius where screening turns off.
    
    Rough approximation: R_c where ρ(R_c) ≈ ρ_crit
    
    For exponential profile: ρ(r) = ρ_c exp(-r/R_d)
    → R_c ≈ R_d ln(ρ_c/ρ_crit)
    """
    rho_crit = critical_density(params)
    
    if rho_center < rho_crit:
        # Already unscreened at center
        return 0.0
    
    if rho_edge > rho_crit:
        # Still screened at edge
        return np.inf
    
    # Exponential decay approximation
    return R_scale * np.log(rho_center / rho_crit)


def effective_gravity_boost(phi: np.ndarray, params: SymmetronParams,
                            M_Pl: float = 1.0) -> np.ndarray:
    """
    Boost to effective gravity from coherence field.
    
    g_eff / g_N = 1 + β φ² / M_Pl²
    
    So: K(R) = β φ²(R) / M_Pl²
    
    This is your multiplicative kernel K(R)!
    """
    phi = np.atleast_1d(phi)
    return params.beta * phi**2 / M_Pl**2


# ==============================================================================
# COSMOLOGICAL EVOLUTION SPECIFIC FUNCTIONS
# ==============================================================================

def phi_cosmology(a: np.ndarray, rho_m0: float, Omega_m0: float,
                  params: SymmetronParams) -> np.ndarray:
    """
    Cosmological evolution of φ (approximate, assuming φ tracks minimum).
    
    ρ_m(a) = ρ_m0 / a³
    
    Then φ(a) ≈ φ_min(ρ_m(a))
    
    This is the "tracking solution" where field adiabatically follows
    the changing minimum as universe expands.
    
    More accurate: solve full Friedmann + KG equations.
    """
    a = np.atleast_1d(a)
    rho_m = rho_m0 / a**3
    
    return phi_minimum(rho_m, params)


def dark_energy_fraction(a: np.ndarray, rho_m0: float, params: SymmetronParams,
                        H0: float = 70.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Ω_φ(a) assuming tracking solution.
    
    Returns:
        Omega_m(a), Omega_phi(a)
    """
    a = np.atleast_1d(a)
    
    # Matter density
    rho_m = rho_m0 / a**3
    
    # Field value (tracking)
    phi_a = phi_minimum(rho_m, params)
    
    # Field energy density
    # Kinetic: neglected in tracking (adiabatic)
    # Potential: V_eff(φ, ρ_m)
    rho_phi = potential_eff(phi_a, rho_m, params)
    
    # Total
    rho_total = rho_m + rho_phi
    
    Omega_m = rho_m / rho_total
    Omega_phi = rho_phi / rho_total
    
    return Omega_m, Omega_phi


# ==============================================================================
# EXAMPLE: TYPICAL PARAMETER VALUES FOR COHERENCE GRAVITY
# ==============================================================================

def default_params_for_coherence_gravity() -> SymmetronParams:
    """
    Educated guess for parameters that might work.
    
    Goals:
    - ρ_crit ~ 10⁻²¹ kg/m³ (galaxy transition)
    - φ₀ ~ 1 in natural units
    - Cosmologically relevant at late times
    
    These are starting points for viability scan.
    """
    # If we want ρ_crit ~ 10⁻²¹ kg/m³ and μ ~ M ~ 10⁻³ eV:
    # ρ_crit = μ² M² ~ (10⁻³ eV)⁴ ~ 10⁻²¹ kg/m³ (rough match!)
    
    return SymmetronParams(
        mu=2.4e-3,       # [eV] ~ H₀ scale
        lambda_self=1.0,  # Order unity
        M=2.4e-3,        # [M_Pl units] ~ μ for simplicity
        V0=0.0,          # Tune later for cosmology
        beta=1.0         # Coupling strength
    )


if __name__ == "__main__":
    """
    Quick test: plot potential shapes at different densities.
    """
    import matplotlib.pyplot as plt
    
    params = default_params_for_coherence_gravity()
    
    print("="*80)
    print("SYMMETRON POTENTIAL TEST")
    print("="*80)
    print(f"\nParameters:")
    print(f"  μ = {params.mu:.2e}")
    print(f"  λ = {params.lambda_self:.2f}")
    print(f"  M = {params.M:.2e}")
    print(f"  β = {params.beta:.2f}")
    
    rho_crit = critical_density(params)
    print(f"\nCritical density: ρ_crit = {rho_crit:.2e}")
    
    # Test densities
    rho_solar = 1e-15  # kg/m³ (very high density)
    rho_galaxy_center = 1e-20
    rho_galaxy_edge = 1e-22
    rho_cosmic = 1e-26
    
    densities = [
        ("Solar System", rho_solar),
        ("Galaxy center", rho_galaxy_center),
        ("Galaxy edge", rho_galaxy_edge),
        ("Cosmic mean", rho_cosmic)
    ]
    
    phi_range = np.linspace(-3, 3, 500)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (label, rho) in enumerate(densities):
        ax = axes[i]
        
        # Compute potential
        V = potential_eff(phi_range, rho, params)
        V_norm = V - np.min(V)  # Shift minimum to 0
        
        # Find minimum
        phi_min = phi_minimum(rho, params)
        phi_min_val = phi_min if np.isscalar(phi_min) else phi_min.item() if getattr(phi_min, 'size', 1) == 1 else phi_min[0]
        
        # Plot
        ax.plot(phi_range, V_norm, 'b-', lw=2)
        ax.axvline(phi_min_val, color='r', ls='--', label=f'φ_min = {phi_min_val:.2f}')
        ax.axvline(-phi_min_val, color='r', ls='--')
        ax.axhline(0, color='k', ls='-', lw=0.5)
        
        ax.set_xlabel('Field φ')
        ax.set_ylabel('V_eff - V_min')
        ax.set_title(f'{label}\nρ/ρ_crit = {rho/rho_crit:.2e}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 2)
    
    plt.tight_layout()
    
    import os
    os.makedirs('coherence-field-theory/outputs', exist_ok=True)
    outpath = 'coherence-field-theory/outputs/symmetron_potential_shapes.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {outpath}")
    
    # Test cosmological evolution
    print("\n" + "="*80)
    print("COSMOLOGICAL EVOLUTION TEST")
    print("="*80)
    
    a_arr = np.logspace(-1, 0, 100)  # a = 0.1 to 1.0
    rho_m0 = 2.5e-27  # kg/m³ (roughly Ω_m ~ 0.3 today)
    
    phi_a = phi_cosmology(a_arr, rho_m0, 0.3, params)
    Omega_m, Omega_phi = dark_energy_fraction(a_arr, rho_m0, params)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Field evolution
    ax = axes[0]
    ax.plot(a_arr, phi_a, 'b-', lw=2)
    ax.set_xlabel('Scale factor a')
    ax.set_ylabel('Field φ(a)')
    ax.set_title('Coherence Field Evolution')
    ax.grid(True, alpha=0.3)
    
    # Omega evolution
    ax = axes[1]
    ax.plot(a_arr, Omega_m, 'r-', lw=2, label='Ω_m')
    ax.plot(a_arr, Omega_phi, 'b-', lw=2, label='Ω_φ')
    ax.axhline(0.3, color='r', ls='--', alpha=0.5)
    ax.axhline(0.7, color='b', ls='--', alpha=0.5)
    ax.set_xlabel('Scale factor a')
    ax.set_ylabel('Density fraction')
    ax.set_title('Energy Budget')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    outpath = 'coherence-field-theory/outputs/symmetron_cosmology.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {outpath}")
    
    print(f"\nToday (a=1):")
    print(f"  φ(a=1) = {phi_a[-1]:.3f}")
    print(f"  Ω_m(a=1) = {Omega_m[-1]:.3f}")
    print(f"  Ω_φ(a=1) = {Omega_phi[-1]:.3f}")
    
    print("\n" + "="*80)
