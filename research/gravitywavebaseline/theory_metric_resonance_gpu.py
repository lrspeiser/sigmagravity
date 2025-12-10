"""
GPU-accelerated version of theory_metric_resonance using CuPy.

Falls back to NumPy if CuPy is not available.
"""

from __future__ import annotations

import numpy as np

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("[GPU] CuPy available - GPU acceleration enabled")
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
    print("[CPU] CuPy not available - using NumPy")


def get_array_module(use_gpu: bool | None = None) -> type:
    """
    Get the appropriate array module (CuPy or NumPy).
    
    Parameters
    ----------
    use_gpu : bool | None
        If True, force GPU. If False, force CPU. If None, use GPU if available.
    
    Returns
    -------
    type
        Either cupy or numpy module
    """
    if use_gpu is False:
        return np
    if use_gpu is True:
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available but GPU requested")
        return cp
    # Auto-detect
    return cp if CUPY_AVAILABLE else np


def compute_theory_kernel_gpu(
    R_kpc: np.ndarray,
    sigma_v_kms: float,
    *,
    alpha: float = 3.5,
    lam_min_kpc: float = 0.1,
    lam_max_kpc: float = 500.0,
    lam_coh_kpc: float = 5.0,
    lam_cut_kpc: float = 300.0,
    Q_ref: float = 1.0,
    A_global: float = 1.0,
    n_lambda: int = 400,
    v_circ_ref_kms: float = 200.0,
    burr_ell0_kpc: float | None = None,
    burr_p: float = 0.757,
    burr_n: float = 0.5,
    use_gpu: bool | None = None,
    return_cpu: bool = True,
) -> np.ndarray:
    """
    GPU-accelerated version of compute_theory_kernel.
    
    Parameters
    ----------
    R_kpc : np.ndarray
        Radii in kpc
    sigma_v_kms : float
        Velocity dispersion in km/s
    use_gpu : bool | None
        Force GPU/CPU usage. None = auto-detect.
    return_cpu : bool
        If True, convert result back to CPU numpy array.
    
    Returns
    -------
    np.ndarray
        Theory kernel K(R)
    """
    xp = get_array_module(use_gpu)
    
    # Convert input to GPU array if using GPU
    R = xp.asarray(R_kpc, dtype=xp.float64)
    lam_min = max(lam_min_kpc, 1e-6)
    lam_max = max(lam_max_kpc, lam_min * 1.01)
    lam_grid = xp.logspace(xp.log10(lam_min), xp.log10(lam_max), n_lambda)

    lam_ref = lam_min
    P_lambda = (lam_ref / lam_grid) ** alpha * xp.exp(-lam_grid / max(lam_cut_kpc, 1e-6))

    lam_matter = 2.0 * xp.pi * R[:, None]
    sigma_v = max(float(sigma_v_kms), 1e-6)
    Q = (v_circ_ref_kms / sigma_v) / max(Q_ref, 1e-6)

    ratio = lam_grid[None, :] / xp.maximum(lam_matter, 1e-12)
    inv_ratio = xp.maximum(lam_matter, 1e-12) / lam_grid[None, :]
    denom = Q**2 + (ratio - inv_ratio) ** 2
    C_res = (Q**2) / denom

    lam_p = (lam_grid / max(lam_coh_kpc, 1e-6)) ** 2.0
    C_coh = xp.exp(-lam_p)

    lam_over_R = lam_grid[None, :] / xp.maximum(R[:, None], 1e-6)
    W_geom = xp.where(lam_over_R < 1.0, 1.0, lam_over_R**-2)

    integrand = (
        P_lambda[None, :]
        * C_res
        * C_coh[None, :]
        * W_geom
        / lam_grid[None, :]
    )
    
    # Use GPU-accelerated trapz if available
    if xp is cp:
        K0 = cp.trapz(integrand, lam_grid, axis=1)
    else:
        K0 = np.trapz(integrand, lam_grid, axis=1)
    
    K0_max = xp.max(xp.abs(K0)) or 1.0
    K = A_global * (K0 / K0_max)

    if burr_ell0_kpc is not None:
        ell0 = max(burr_ell0_kpc, 1e-6)
        shaped = 1.0 - (1.0 + (xp.maximum(R, 0.0) / ell0) ** max(burr_p, 1e-6)) ** (
            -max(burr_n, 1e-6)
        )
        K = K * shaped

    # Convert back to CPU numpy array if requested
    if return_cpu and xp is cp:
        return cp.asnumpy(K)
    return K


