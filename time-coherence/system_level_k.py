"""
System-level roughness enhancement from exposure factor.

This module provides K_rough(Ξ) based on Phase-2 fit results.
"""

import numpy as np


def system_level_K(
    Xi_mean: float,
    A0: float = 0.774,
    gamma: float = 0.1,
) -> float:
    """
    System-level roughness enhancement from Phase-2 fit.
    
    K_rough = A0 * Xi_mean^gamma
    
    where Xi_mean is the mean exposure factor (τ_coh / T_orb) for the system.
    
    Parameters
    ----------
    Xi_mean : float
        Mean exposure factor for the system
    A0 : float
        Amplitude parameter from Phase-2 fit (default 0.774)
    gamma : float
        Power-law index from Phase-2 fit (default 0.1)
        
    Returns
    -------
    K_rough : float
        System-level roughness enhancement factor
    """
    Xi_clipped = max(Xi_mean, 0.0)
    return A0 * (Xi_clipped ** gamma)


def compute_Xi_mean(
    R_kpc: np.ndarray,
    g_bar_kms2: np.ndarray,
    tau_coh_sec: np.ndarray,
) -> float:
    """
    Compute mean exposure factor for a system.
    
    Parameters
    ----------
    R_kpc : np.ndarray
        Radii in kpc
    g_bar_kms2 : np.ndarray
        Baryonic acceleration in km/s²
    tau_coh_sec : np.ndarray
        Coherence time in seconds
        
    Returns
    -------
    Xi_mean : float
        Mean exposure factor
    """
    from coherence_time_kernel import compute_exposure_factor
    
    Xi = compute_exposure_factor(R_kpc, g_bar_kms2, tau_coh_sec)
    return float(np.mean(Xi))

