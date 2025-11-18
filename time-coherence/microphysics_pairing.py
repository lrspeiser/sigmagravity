"""
Microphysics Model 2: Graviton Pairing / Superfluid-like Condensate

This is your best bet for the "missing 90%" piece: something like a Bose-condensed 
gravitational degree of freedom (scalar or tensor) that makes gravity stronger in 
cold, dilute, extended systems and shuts off in hot, compact ones.

Toy field equations:
- Add a complex scalar condensate field ψ with action involving coupling to baryonic trace
- Variation gives modified Einstein equations with extra T^(ψ)_μν
- Gross-Pitaevskii-like equation for ψ

In static, weak-field, spherically symmetric limit:
    ∇²Φ - μ²Φ = 4πG_eff(ρ,σ_v) ρ

with G_eff(ρ,σ_v) = G[1 + K_pair(ρ,σ_v)]

Because the condensate is destroyed by velocity dispersion:
    K_pair ∝ (σ_c / σ_v)^γ  for σ_v < σ_c

The Newtonian-limit prescription:
    g_eff(R) = g_GR(R) * [1 + K_pair(R, σ_v(R))]
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PairingParams:
    """Parameters for the graviton pairing/superfluid condensate model."""
    A_pair: float = 1.0          # overall amplitude
    sigma_c: float = 25.0        # critical dispersion [km/s] (superfluid transition)
    gamma_sigma: float = 2.0     # how sharply pairing dies with sigma_v
    ell_pair_kpc: float = 5.0    # coherence length for radial falloff
    p: float = 1.0               # shape exponent for radial envelope


def K_pairing(R_kpc, sigma_v_kms, params: PairingParams):
    """
    Toy graviton-pairing enhancement kernel.
    
    - Strong in cold disks (sigma_v << sigma_c)
    - Suppressed in hot systems (sigma_v >> sigma_c)
    - Has a finite radial envelope set by ell_pair_kpc
    
    K_pair(R, σ_v) = A_pair * G_sigma(σ_v) * C_R(R)
    
    where:
        G_sigma ~ 1 for cold systems, falls as (sigma_c/sigma_v)^gamma otherwise
        C_R is a radial envelope (can swap for Burr-XII later)
    
    Parameters
    ----------
    R_kpc : array_like
        Radii in kpc
    sigma_v_kms : array_like
        Velocity dispersion in km/s (can be array or scalar)
    params : PairingParams
        Model parameters
        
    Returns
    -------
    K_pair : ndarray
        Enhancement kernel
    """
    R = np.asarray(R_kpc, dtype=float)
    sigma_v = np.asarray(sigma_v_kms, dtype=float)
    
    # Broadcast sigma_v if scalar
    if sigma_v.ndim == 0 or (sigma_v.ndim == 1 and len(sigma_v) == 1):
        sigma_v = np.full_like(R, float(sigma_v))
    
    # Sigma gate: ~1 for cold systems, falls as (sigma_c/sigma_v)^gamma otherwise
    # Use a smooth gate that transitions around sigma_c
    x = np.maximum(params.sigma_c / np.maximum(sigma_v, 1e-3), 1e-3)
    G_sigma = x**params.gamma_sigma / (1.0 + x**params.gamma_sigma)
    
    # Simple radial envelope (can swap for Burr-XII later)
    # C(R) = 1 - exp(-(R/ℓ_pair)^p)
    # This gives C → 0 as R → 0 (Solar System safe) and C → 1 for R ≫ ℓ_pair
    C_R = 1.0 - np.exp(-(R / np.maximum(params.ell_pair_kpc, 1e-3))**params.p)
    
    return params.A_pair * G_sigma * C_R


def apply_pairing_boost(g_gr, R_kpc, sigma_v_kms, params: PairingParams):
    """
    Apply pairing boost: g_eff = g_GR * (1 + K_pair(R, σ_v)).
    
    Parameters
    ----------
    g_gr : array_like
        GR acceleration in km/s²
    R_kpc : array_like
        Radii in kpc
    sigma_v_kms : array_like
        Velocity dispersion in km/s
    params : PairingParams
        Model parameters
        
    Returns
    -------
    g_eff : ndarray
        Enhanced acceleration in km/s²
    """
    K_pair = K_pairing(R_kpc, sigma_v_kms, params)
    g_eff = np.asarray(g_gr, dtype=float) * (1.0 + K_pair)
    return g_eff

