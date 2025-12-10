"""
Pairing model with Burr-XII radial envelope.

Replaces the exponential radial envelope:
  C_R = 1 - exp(-(R/ℓ)^p)

with Burr-XII envelope (matching empirical Σ-Gravity kernel):
  C_R = 1 - [1 + (R/ℓ)^p]^(-q)

This should better match the empirical kernel shape.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PairingBurrParams:
    """Parameters for pairing model with Burr-XII radial envelope."""
    A_pair: float = 5.0          # overall amplitude
    sigma_c: float = 15.0        # critical dispersion [km/s]
    gamma_sigma: float = 3.0     # velocity gate sharpness
    ell_pair_kpc: float = 20.0   # coherence length scale
    p: float = 0.757             # Burr-XII shape parameter (from empirical)
    q: float = 0.5               # Burr-XII shape parameter (from empirical)


def K_pairing_burr(R_kpc, sigma_v_kms, params: PairingBurrParams):
    """
    Pairing enhancement with Burr-XII radial envelope.
    
    K_pair(R, σ_v) = A_pair × G_sigma(σ_v) × C_R(R)
    
    where:
        G_sigma = (σ_c/σ_v)^γ / (1 + (σ_c/σ_v)^γ)  (superfluid gate)
        C_R = 1 - [1 + (R/ℓ)^p]^(-q)                (Burr-XII envelope)
    
    Parameters
    ----------
    R_kpc : array_like
        Radii in kpc
    sigma_v_kms : array_like
        Velocity dispersion in km/s
    params : PairingBurrParams
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
    
    # Sigma gate: ~1 for cold systems, 0 for hot systems
    x = np.maximum(params.sigma_c / np.maximum(sigma_v, 1e-3), 1e-3)
    G_sigma = x**params.gamma_sigma / (1.0 + x**params.gamma_sigma)
    
    # Burr-XII radial envelope
    # C(R) = 1 - [1 + (R/ℓ)^p]^(-q)
    x_R = R / np.maximum(params.ell_pair_kpc, 1e-3)
    C_R = 1.0 - (1.0 + x_R**params.p) ** (-params.q)
    
    return params.A_pair * G_sigma * C_R


def apply_pairing_burr_boost(g_gr, R_kpc, sigma_v_kms, params: PairingBurrParams):
    """
    Apply pairing boost with Burr-XII envelope.
    
    Parameters
    ----------
    g_gr : array_like
        GR acceleration in km/s²
    R_kpc : array_like
        Radii in kpc
    sigma_v_kms : array_like
        Velocity dispersion in km/s
    params : PairingBurrParams
        Model parameters
        
    Returns
    -------
    g_eff : ndarray
        Enhanced acceleration in km/s²
    """
    K_pair = K_pairing_burr(R_kpc, sigma_v_kms, params)
    g_eff = np.asarray(g_gr, dtype=float) * (1.0 + K_pair)
    return g_eff


def check_solar_system_safety_burr(params: PairingBurrParams):
    """
    Check Solar System safety with Burr-XII envelope.
    
    Returns
    -------
    K_solar : float
        Enhancement at 1 AU
    is_safe : bool
        Whether K < 10^-10
    """
    R_au = 5e-9  # 1 AU in kpc
    sigma_v_solar = 10.0  # km/s
    
    K_solar = K_pairing_burr(np.array([R_au]), sigma_v_solar, params)[0]
    
    return K_solar, K_solar < 1e-10

