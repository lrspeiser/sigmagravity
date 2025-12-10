"""
Combined Microphysics Model: Roughness + Pairing

Combines two mechanisms:
1. Time-coherence / roughness: System-level enhancement from stochastic fluctuations
2. Graviton pairing: R-dependent, σ_v-gated enhancement from superfluid condensate

Two combination strategies:
- Additive: g_eff = g_GR × (1 + K_rough + K_pair)
- Multiplicative: g_eff = g_GR × (1 + K_rough) × (1 + K_pair)
"""

from dataclasses import dataclass
import numpy as np

from microphysics_roughness import (
    RoughnessParams,
    system_level_exposure,
    K_rough_from_Xi,
)
from microphysics_pairing import (
    PairingParams,
    K_pairing,
)


@dataclass
class CombinedParams:
    """Parameters for combined roughness + pairing model."""
    roughness: RoughnessParams
    pairing: PairingParams
    combination: str = "additive"  # "additive" or "multiplicative"


def combined_boost(
    g_gr,
    R_kpc,
    v_circ_kms,
    sigma_v_kms,
    params: CombinedParams,
):
    """
    Apply combined roughness + pairing boost.
    
    Two strategies:
    - Additive: g_eff = g_GR × (1 + K_rough + K_pair)
    - Multiplicative: g_eff = g_GR × (1 + K_rough) × (1 + K_pair)
    
    Parameters
    ----------
    g_gr : array_like
        GR acceleration in km/s²
    R_kpc : array_like
        Radii in kpc
    v_circ_kms : array_like
        Circular velocity in km/s
    sigma_v_kms : array_like
        Velocity dispersion in km/s
    params : CombinedParams
        Combined model parameters
        
    Returns
    -------
    g_eff : ndarray
        Enhanced acceleration in km/s²
    K_rough : float
        Roughness enhancement (system-level)
    K_pair : ndarray
        Pairing enhancement (R-dependent)
    """
    R = np.asarray(R_kpc, dtype=float)
    v_circ = np.asarray(v_circ_kms, dtype=float)
    sigma_v = np.asarray(sigma_v_kms, dtype=float)
    g = np.asarray(g_gr, dtype=float)
    
    # Compute roughness enhancement (system-level)
    Xi_sys = system_level_exposure(R, v_circ, sigma_v, params.roughness)
    K_rough = K_rough_from_Xi(Xi_sys, params.roughness)
    
    # Compute pairing enhancement (R-dependent)
    K_pair = K_pairing(R, sigma_v, params.pairing)
    
    # Combine
    if params.combination == "additive":
        # g_eff = g_GR × (1 + K_rough + K_pair)
        g_eff = g * (1.0 + K_rough + K_pair)
    elif params.combination == "multiplicative":
        # g_eff = g_GR × (1 + K_rough) × (1 + K_pair)
        g_eff = g * (1.0 + K_rough) * (1.0 + K_pair)
    else:
        raise ValueError(f"Unknown combination: {params.combination}")
    
    return g_eff, K_rough, K_pair


def apply_combined_boost(
    g_gr,
    R_kpc,
    v_circ_kms,
    sigma_v_kms,
    params: CombinedParams,
):
    """
    Apply combined boost (convenience wrapper).
    
    Returns only g_eff, not the individual components.
    """
    g_eff, _, _ = combined_boost(g_gr, R_kpc, v_circ_kms, sigma_v_kms, params)
    return g_eff

