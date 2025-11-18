"""
Mass-coherence model for F_missing (multiplicative ratio).

F_missing = K_total / K_rough, i.e. how much larger the total kernel
should be compared to the rough component.

This is a RATIO, not a kernel by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MassCoherenceParams:
    """Parameters for mass-coherence F_missing prediction."""
    sigma_ref: float = 14.8  # km/s
    Rd_ref: float = 12.3     # kpc
    A_F: float = 10.02       # best-fit amplitude for F_missing
    beta_sigma: float = 0.10
    beta_Rd: float = 0.31
    F_min: float = 1.0       # lower clamp: no reduction below roughness
    F_max: float = 5.0       # upper clamp: prevents crazy boosts


def predict_F_missing(
    galaxy_props: dict,
    params: MassCoherenceParams | None = None,
) -> float:
    """
    Predict F_missing = K_total / K_rough, i.e. how much larger the
    total kernel should be compared to the rough component.
    
    This is a RATIO, not a kernel by itself.
    
    Parameters
    ----------
    galaxy_props : dict
        Dictionary with 'sigma_v' (km/s) and 'R_d' or 'R_disk' (kpc)
    params : MassCoherenceParams, optional
        Model parameters. If None, uses defaults.
        
    Returns
    -------
    float
        F_missing ratio (clamped between F_min and F_max)
    """
    if params is None:
        params = MassCoherenceParams()
    
    # Extract galaxy properties
    sigma_v = float(galaxy_props.get("sigma_v", galaxy_props.get("sigma_velocity", 20.0)))
    R_d = float(galaxy_props.get("R_d", galaxy_props.get("R_disk", 5.0)))
    
    # Guardrails
    sigma_v = max(sigma_v, 1e-3)
    R_d = max(R_d, 1e-3)
    
    # Functional form from fit
    F = (
        params.A_F
        * (params.sigma_ref / sigma_v) ** params.beta_sigma
        * (params.Rd_ref / R_d) ** params.beta_Rd
    )
    
    # Clamp to physically reasonable range
    F = max(params.F_min, min(F, params.F_max))
    
    return float(F)


def predict_F_missing_mass_model(
    M_baryon_msun: np.ndarray,
    R_d_kpc: np.ndarray,
    ell0_kpc: np.ndarray,
    *,
    R_eff_factor: float = 1.33,
    K_max: float = 19.58,
    psi0: float = 7.34e-8,
    gamma: float = 0.136,
) -> np.ndarray:
    """
    Legacy function for compatibility.
    
    This was fitted to F_missing values, but should now use predict_F_missing
    with the functional form parameters instead.
    
    Kept for backward compatibility but deprecated.
    """
    from mass_coherence_model import K_missing_from_mass
    
    M_b = np.asarray(M_baryon_msun, dtype=float)
    R_d = np.asarray(R_d_kpc, dtype=float)
    ell0 = np.asarray(ell0_kpc, dtype=float)
    R_eff = R_eff_factor * R_d
    
    F_pred = np.empty_like(M_b, dtype=float)
    for i in range(M_b.size):
        # This returns K_missing, but we want F_missing ratio
        # For now, use the old model but note it's deprecated
        K_missing = K_missing_from_mass(
            M_baryon_msun=M_b[i],
            R_eff_kpc=R_eff[i],
            ell0_kpc=ell0[i],
            K_max=K_max,
            psi0=psi0,
            gamma=gamma,
        )
        # Convert K_missing to F_missing approximation
        # This is a rough conversion - prefer predict_F_missing instead
        F_pred[i] = 1.0 + K_missing / 10.0  # Rough scaling
    
    return F_pred
