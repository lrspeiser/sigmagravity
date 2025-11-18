"""
Unified kernel combining roughness and mass-coherence.

K_total(R) = K_rough(Ξ_mean) × C(R/ℓ₀) + K_missing(Ψ)

Where:
- K_rough: System-level roughness from time-coherence
- C(R/ℓ₀): Burr-XII radial shape
- K_missing: Mass-coherence enhancement
"""

from __future__ import annotations

import numpy as np
from burr_xii_shape import burr_xii_shape
from mass_coherence_model import K_missing_from_mass
from system_level_k import system_level_K, compute_Xi_mean
from coherence_time_kernel import (
    compute_tau_geom,
    compute_tau_noise,
    compute_tau_coh,
)


def compute_unified_kernel(
    R_kpc: np.ndarray,
    g_bar_kms2: np.ndarray,
    sigma_v_kms: float,
    rho_bar_msun_pc3: np.ndarray,
    M_baryon_msun: float,
    R_disk_kpc: float,
    ell0_kpc: float,
    *,
    # Roughness parameters
    A0: float = 0.774,
    gamma_rough: float = 0.1,
    p: float = 0.757,
    n_coh: float = 0.5,
    tau_geom_method: str = "tidal",
    alpha_geom: float = 1.0,
    beta_sigma: float = 1.5,
    # Mass-coherence parameters
    K_max: float = 19.58,
    psi0: float = 7.34e-8,
    gamma_mass: float = 0.136,
    R_eff_factor: float = 1.33,
) -> tuple[np.ndarray, dict]:
    """
    Compute unified kernel K_total(R) = K_rough × C(R) + K_missing.

    Parameters
    ----------
    R_kpc : array_like
        Radii in kpc.
    g_bar_kms2 : array_like
        Baryonic acceleration in km/s^2.
    sigma_v_kms : float
        Velocity dispersion in km/s.
    rho_bar_msun_pc3 : array_like
        Baryonic density in Msun/pc^3.
    M_baryon_msun : float
        Total baryonic mass in Msun.
    R_disk_kpc : float
        Disk scale length in kpc.
    ell0_kpc : float
        Coherence scale in kpc.
    A0, gamma_rough, p, n_coh : floats
        Roughness parameters.
    K_max, psi0, gamma_mass, R_eff_factor : floats
        Mass-coherence parameters.

    Returns
    -------
    K_total : ndarray
        Total enhancement kernel K_total(R).
    info : dict
        Dictionary with intermediate values (K_rough, K_missing, Xi_mean, etc.).
    """
    R = np.asarray(R_kpc, dtype=float)
    g_bar = np.asarray(g_bar_kms2, dtype=float)
    rho_bar = np.asarray(rho_bar_msun_pc3, dtype=float)

    # 1. Compute roughness component
    tau_geom = compute_tau_geom(
        R,
        g_bar,
        rho_bar,
        method=tau_geom_method,
        alpha_geom=alpha_geom,
    )
    tau_noise = compute_tau_noise(
        R,
        sigma_v_kms,  # scalar, not array
        method="galaxy",
        beta_sigma=beta_sigma,
    )
    tau_coh = compute_tau_coh(tau_geom, tau_noise)

    # Compute mean exposure factor
    from coherence_time_kernel import compute_exposure_factor

    Xi = compute_exposure_factor(R, g_bar, tau_coh)
    Xi_mean = float(np.mean(Xi)) if len(Xi) > 0 else 0.0

    # System-level roughness
    K_rough = system_level_K(Xi_mean, A0=A0, gamma=gamma_rough)

    # Radial shape (unit amplitude)
    C_R = burr_xii_shape(R, ell0_kpc, p=p, n_coh=n_coh)

    # Roughness contribution (radial)
    K_rough_radial = K_rough * C_R

    # 2. Compute mass-coherence component (system-level)
    R_eff = R_eff_factor * R_disk_kpc
    K_missing = K_missing_from_mass(
        M_baryon_msun=M_baryon_msun,
        R_eff_kpc=R_eff,
        ell0_kpc=ell0_kpc,
        K_max=K_max,
        psi0=psi0,
        gamma=gamma_mass,
    )

    # Total enhancement
    # Note: K_missing is system-level, so we apply it uniformly
    # Alternatively, could make it radial if we compute Ψ(R)
    K_total = K_rough_radial + K_missing

    info = {
        "K_rough": float(K_rough),
        "K_missing": float(K_missing),
        "Xi_mean": float(Xi_mean),
        "C_mean": float(np.mean(C_R)),
        "K_rough_radial_mean": float(np.mean(K_rough_radial)),
        "K_total_mean": float(np.mean(K_total)),
    }

    return K_total, info

