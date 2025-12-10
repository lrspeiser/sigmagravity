"""
Burr-XII shape function (unit amplitude) for Σ-Gravity kernel.

This module provides the radial shape C(R) normalized to [0, 1],
which will be multiplied by system-level K_rough to get total enhancement.
"""

import numpy as np


def burr_xii_shape(
    R_kpc: np.ndarray,
    ell0_kpc: float,
    p: float = 0.757,
    n_coh: float = 0.5,
) -> np.ndarray:
    """
    Burr-XII coherence window: C(R/ℓ₀) in [0, 1].
    
    This is the dimensionless radial shape, normalized to unit amplitude.
    The actual enhancement is K_total(R) = K_rough(Ξ) * C(R/ℓ₀).
    
    Parameters
    ----------
    R_kpc : np.ndarray
        Radii in kpc
    ell0_kpc : float
        Characteristic coherence length in kpc
    p : float
        Burr-XII shape parameter (default 0.757 from empirical fits)
    n_coh : float
        Burr-XII shape parameter (default 0.5 from empirical fits)
        
    Returns
    -------
    C : np.ndarray
        Coherence window in [0, 1]
    """
    R = np.asarray(R_kpc, dtype=np.float64)
    ell0 = max(ell0_kpc, 1e-6)  # Avoid division by zero
    
    x = R / ell0
    C = 1.0 - (1.0 + x**p) ** (-n_coh)
    
    return np.clip(C, 0.0, 1.0)


def compute_total_kernel(
    R_kpc: np.ndarray,
    K_rough: float,
    ell0_kpc: float,
    p: float = 0.757,
    n_coh: float = 0.5,
) -> np.ndarray:
    """
    Compute total enhancement kernel: K_total(R) = K_rough * C(R/ℓ₀).
    
    Parameters
    ----------
    R_kpc : np.ndarray
        Radii in kpc
    K_rough : float
        System-level roughness enhancement (from K(Ξ) relation)
    ell0_kpc : float
        Characteristic coherence length in kpc
    p : float
        Burr-XII shape parameter
    n_coh : float
        Burr-XII shape parameter
        
    Returns
    -------
    K_total : np.ndarray
        Total enhancement kernel K_total(R)
    """
    C = burr_xii_shape(R_kpc, ell0_kpc, p=p, n_coh=n_coh)
    return K_rough * C

