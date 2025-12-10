"""
Σ-Gravity with Spiral Winding Gate
===================================

Integrates the original Σ-Gravity coherence kernel with a morphology-dependent
spiral winding gate.

Original Σ-Gravity:
    K(R) = A × C(R; ℓ₀, p, n_coh) × Π_j G_j

Extended Σ-Gravity:
    K(R) = A × C(R; ℓ₀, p, n_coh) × Π_j G_j × G_winding(R, v_c)

where:
    G_winding = 1 / (1 + (N_orbits/N_crit)²)
    N_orbits = t_age × v_c / (2πR × 0.978)
    N_crit ~ 10 (derived from v_c/σ_v ~ 200/20)

Author: Leonard Speiser
Date: 2025-11-25
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import os
import sys

# Import winding gate from local module
from winding_gate import compute_winding_gate, compute_N_orbits


@dataclass
class SigmaGravityParams:
    """Parameters for Σ-Gravity with winding."""
    # Amplitude
    A: float = 0.6
    
    # Burr-XII coherence window
    ell_0: float = 4.993  # kpc (from paper - fitted value)
    p: float = 0.75       # shape exponent
    n_coh: float = 0.5    # coherence exponent
    
    # Winding gate
    t_age: float = 10.0   # Gyr
    N_crit: float = 10.0  # critical winding number
    wind_power: float = 2.0
    use_winding: bool = True
    
    # Other gates (from original Σ-Gravity)
    R_min: float = 0.1    # kpc - inner distance gate
    alpha_R: float = 2.0  # distance gate steepness
    beta_R: float = 1.0   # distance gate strength


def C_burr_XII(R: np.ndarray, ell_0: float, p: float, n_coh: float) -> np.ndarray:
    """
    Burr-XII coherence window from Σ-Gravity paper Section 2.3.
    
    C(R) = 1 - [1 + (R/ℓ₀)^p]^(-n_coh)
    
    Properties:
    - C(R → 0) → 0  (no coherence at small scales)
    - C(R → ∞) → 1  (full coherence at large scales)
    - C(ℓ₀) ≈ 0.5  (transition at coherence length)
    """
    R_safe = np.maximum(R, 1e-10)
    return 1.0 - (1.0 + (R_safe / ell_0)**p)**(-n_coh)


def G_distance(R: np.ndarray, R_min: float, alpha: float, beta: float) -> np.ndarray:
    """
    Distance-based gate for Solar System safety.
    
    G(R) = [1 + (R_min/R)^α]^(-β)
    
    Properties:
    - G(R → 0) → 0 (suppressed at small scales)
    - G(R → ∞) → 1 (no suppression at large scales)
    """
    R_safe = np.maximum(R, 1e-10)
    return (1.0 + (R_min / R_safe)**alpha)**(-beta)


def sigma_gravity_kernel(
    R: np.ndarray,
    v_c: np.ndarray,
    params: SigmaGravityParams,
    diagnostics: Optional[Dict] = None
) -> np.ndarray:
    """
    Compute full Σ-Gravity kernel with winding gate.
    
    K(R) = A × C(R) × G_distance(R) × G_winding(R, v_c)
    
    Parameters
    ----------
    R : array
        Galactocentric radius [kpc]
    v_c : array
        Circular velocity [km/s] - used for winding calculation
    params : SigmaGravityParams
        Model parameters
    diagnostics : dict, optional
        If provided, filled with intermediate values
    
    Returns
    -------
    K : array
        Kernel value (dimensionless boost factor)
    """
    # 1. Coherence window
    C = C_burr_XII(R, params.ell_0, params.p, params.n_coh)
    
    # 2. Distance gate (Solar System safety)
    G_dist = G_distance(R, params.R_min, params.alpha_R, params.beta_R)
    
    # 3. Winding gate (morphology-dependent suppression)
    if params.use_winding:
        G_wind = compute_winding_gate(R, v_c, params.t_age, params.N_crit, params.wind_power)
        N_orbits = compute_N_orbits(R, v_c, params.t_age)
    else:
        G_wind = np.ones_like(R)
        N_orbits = np.zeros_like(R)
    
    # 4. Full kernel
    K = params.A * C * G_dist * G_wind
    
    # Store diagnostics
    if diagnostics is not None:
        diagnostics['C'] = C
        diagnostics['G_dist'] = G_dist
        diagnostics['G_wind'] = G_wind
        diagnostics['N_orbits'] = N_orbits
        diagnostics['K'] = K
    
    return K


def sigma_gravity_velocity(
    R: np.ndarray,
    v_bary: np.ndarray,
    params: SigmaGravityParams,
    diagnostics: Optional[Dict] = None
) -> np.ndarray:
    """
    Predict rotation velocity using Σ-Gravity with winding.
    
    g_total = g_bar × (1 + K)
    v_pred = v_bary × √(1 + K)
    
    Parameters
    ----------
    R : array
        Galactocentric radius [kpc]
    v_bary : array
        Baryonic circular velocity [km/s]
    params : SigmaGravityParams
        Model parameters
    diagnostics : dict, optional
        If provided, filled with intermediate values
    
    Returns
    -------
    v_pred : array
        Predicted circular velocity [km/s]
    """
    # Use baryonic velocity as proxy for circular velocity in winding calculation
    K = sigma_gravity_kernel(R, v_bary, params, diagnostics)
    
    # Enhancement: g_total = g_bar × (1 + K)
    # Velocity: v² = g × R → v = v_bary × √(1 + K)
    v_pred = v_bary * np.sqrt(1.0 + K)
    
    if diagnostics is not None:
        diagnostics['v_pred'] = v_pred
        diagnostics['F'] = 1.0 + K  # Enhancement factor for comparison
    
    return v_pred


def load_sparc_galaxy(filepath: str) -> Dict:
    """Load SPARC rotation curve data."""
    data = np.loadtxt(filepath, comments='#')
    
    if data.shape[1] < 6:
        raise ValueError(f"Expected 6 columns, got {data.shape[1]}")
    
    name = os.path.basename(filepath).replace('_rotmod.dat', '')
    
    return {
        'name': name,
        'R': data[:, 0],           # kpc
        'v_obs': data[:, 1],       # km/s
        'v_err': data[:, 2],       # km/s
        'v_gas': data[:, 3],       # km/s
        'v_disk': data[:, 4],      # km/s
        'v_bul': data[:, 5],       # km/s
    }


def fit_galaxy(data: Dict, params: SigmaGravityParams) -> Dict:
    """
    Fit Σ-Gravity with winding to a galaxy.
    
    Returns fitting results with RMS comparison.
    """
    R = data['R']
    v_obs = data['v_obs']
    v_gas = np.abs(data['v_gas'])
    v_disk = np.abs(data['v_disk'])
    v_bul = np.abs(data['v_bul'])
    
    # Baryonic velocity (quadrature sum)
    v_bary = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
    
    # Compute prediction
    diag = {}
    v_pred = sigma_gravity_velocity(R, v_bary, params, diag)
    
    # RMS comparison
    rms_bary = np.sqrt(np.mean((v_bary - v_obs)**2))
    rms_pred = np.sqrt(np.mean((v_pred - v_obs)**2))
    delta_rms = rms_pred - rms_bary
    improved = rms_pred < rms_bary
    
    # Flat velocity (outer disk)
    v_flat = np.mean(v_obs[-3:]) if len(v_obs) >= 3 else v_obs[-1]
    
    return {
        'name': data['name'],
        'n_points': len(R),
        'v_flat': v_flat,
        'rms_bary': rms_bary,
        'rms_pred': rms_pred,
        'delta_rms': delta_rms,
        'improved': improved,
        'R': R,
        'v_obs': v_obs,
        'v_bary': v_bary,
        'v_pred': v_pred,
        'K': diag['K'],
        'F': diag['F'],
        'C': diag['C'],
        'G_dist': diag['G_dist'],
        'G_wind': diag['G_wind'],
        'N_orbits': diag['N_orbits'],
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Σ-GRAVITY WITH SPIRAL WINDING")
    print("=" * 70)
    
    # Test on synthetic profiles
    R_test = np.linspace(1, 25, 25)  # kpc
    v_flat = np.array([60, 150, 220])  # dwarf, intermediate, massive
    
    params = SigmaGravityParams()
    
    print(f"\nParameters:")
    print(f"  A = {params.A}")
    print(f"  ℓ₀ = {params.ell_0} kpc")
    print(f"  p = {params.p}, n_coh = {params.n_coh}")
    print(f"  N_crit = {params.N_crit}")
    print(f"  t_age = {params.t_age} Gyr")
    
    print("\n" + "-" * 70)
    print(f"{'Galaxy Type':<20} {'v_flat':<10} {'Mean K':<12} {'Mean G_wind':<12} {'Mean N_orb':<12}")
    print("-" * 70)
    
    for name, v in zip(['Dwarf', 'Intermediate', 'Massive'], v_flat):
        v_c = np.full_like(R_test, v)
        diag = {}
        K = sigma_gravity_kernel(R_test, v_c, params, diag)
        
        print(f"{name:<20} {v:<10.0f} {np.mean(K):<12.4f} {np.mean(diag['G_wind']):<12.3f} {np.mean(diag['N_orbits']):<12.1f}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
Massive spirals have more orbits → tighter winding → lower G_wind → lower K
Dwarfs have fewer orbits → looser winding → higher G_wind → higher K

This provides MORPHOLOGY-DEPENDENT enhancement without ad-hoc classification!
""")
