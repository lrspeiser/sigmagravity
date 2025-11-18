"""
Time-coherence based Σ-Gravity kernel.

Core idea: Enhancement is controlled by coherence time τ_coh(R), set by competition
between gravitational time-dilation-driven phase alignment (τ_geom) and 
environment-driven decoherence (τ_noise from σ_v, turbulence, temperature).

The coherence length ℓ_coh = c · τ_coh then feeds the Burr-XII kernel.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# Speed of light in km/s
C_LIGHT_KMS = 299792.458

# Gravitational constant in Msun-kpc-km²/s² units
G_MSUN_KPC_KM2_S2 = 4.302e-6


def compute_tau_geom(
    R_kpc: np.ndarray,
    g_bar_kms2: np.ndarray,
    rho_bar_msun_pc3: Optional[np.ndarray] = None,
    *,
    delta_R_kpc: float = 0.1,
    method: str = "tidal",
    alpha_geom: float = 1.0,
) -> np.ndarray:
    """
    Compute geometry-driven dephasing time τ_geom(R).
    
    Two nearby geodesics separated by ΔR accumulate different proper times
    due to gravitational time dilation. τ_geom is the characteristic time
    for ~2π phase drift.
    
    Parameters:
    -----------
    R_kpc : np.ndarray
        Radii in kpc
    g_bar_kms2 : np.ndarray
        Baryonic acceleration g_bar(R) in km/s²
    rho_bar_msun_pc3 : np.ndarray, optional
        Baryonic density profile (for tidal method)
    delta_R_kpc : float
        Characteristic separation scale for nearby geodesics
    method : str
        "tidal" or "simple" - method for computing ΔΦ
    alpha_geom : float
        Prefactor for geometric timescale (default 1.0, can be tuned)
        
    Returns:
    --------
    tau_geom : np.ndarray
        Geometric dephasing time in seconds
    """
    R = np.asarray(R_kpc, dtype=np.float64)
    g_bar = np.asarray(g_bar_kms2, dtype=np.float64)
    
    if method == "simple":
        # Simple scaling: ΔΦ ~ g_bar * ΔR
        # τ_geom ~ (c² / ΔΦ) * T_orb
        # T_orb ~ 2πR / v_circ, v_circ ~ sqrt(g_bar * R)
        v_circ_kms = np.sqrt(np.clip(g_bar * R * 1e3, 1e-6, None))  # km/s
        T_orb_sec = 2 * np.pi * R * 3.086e16 / (v_circ_kms * 1e3)  # seconds
        
        delta_phi_c2 = g_bar * delta_R_kpc * 1e3 / (C_LIGHT_KMS**2)  # dimensionless
        delta_phi_c2 = np.clip(delta_phi_c2, 1e-10, None)
        
        # Time for 2π phase drift
        tau_geom = (2 * np.pi / delta_phi_c2) * T_orb_sec
        tau_geom = alpha_geom * tau_geom  # Apply prefactor
        tau_geom = np.clip(tau_geom, 1e6, 1e20)  # Reasonable bounds
        
    elif method == "tidal":
        # More sophisticated: use tidal field
        if rho_bar_msun_pc3 is None:
            # Fallback to simple method
            return compute_tau_geom(R_kpc, g_bar_kms2, method="simple")
        
        rho_bar = np.asarray(rho_bar_msun_pc3, dtype=np.float64)
        
        # Tidal acceleration ~ G * rho * ΔR
        # ΔΦ ~ tidal * ΔR ~ G * rho * ΔR²
        G_msun_kpc_km2_s2 = 4.302e-6  # G in Msun^-1 kpc^3 km^2 s^-2
        delta_phi_c2 = (
            G_msun_kpc_km2_s2 * rho_bar * (delta_R_kpc**2) / (C_LIGHT_KMS**2)
        )
        delta_phi_c2 = np.clip(delta_phi_c2, 1e-10, None)
        
        # Orbital period
        v_circ_kms = np.sqrt(np.clip(g_bar * R * 1e3, 1e-6, None))
        T_orb_sec = 2 * np.pi * R * 3.086e16 / (v_circ_kms * 1e3)
        
        tau_geom = (2 * np.pi / delta_phi_c2) * T_orb_sec
        tau_geom = alpha_geom * tau_geom  # Apply prefactor
        tau_geom = np.clip(tau_geom, 1e6, 1e20)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return tau_geom


def compute_tau_noise(
    R_kpc: np.ndarray,
    sigma_v_kms: float,
    *,
    method: str = "galaxy",
    v_turb_kms: Optional[float] = None,
    L_turb_kpc: Optional[float] = None,
    beta_sigma: float = 1.5,
) -> np.ndarray:
    """
    Compute noise-driven decoherence time τ_noise(R).
    
    For galaxies: τ_noise ~ R / σ_v
    For clusters: τ_noise ~ L_turb / v_turb (ICM turbulence)
    
    Parameters:
    -----------
    R_kpc : np.ndarray
        Radii in kpc
    sigma_v_kms : float
        Velocity dispersion in km/s (for galaxies)
    method : str
        "galaxy" or "cluster"
    v_turb_kms : float, optional
        Turbulent velocity for clusters (km/s)
    L_turb_kpc : float, optional
        Turbulent length scale for clusters (kpc)
        
    Returns:
    --------
    tau_noise : np.ndarray
        Noise decoherence time in seconds
    """
    R = np.asarray(R_kpc, dtype=np.float64)
    
    if method == "galaxy":
        # τ_noise ~ R / σ_v^β (stronger σ_v dependence)
        sigma_v = max(sigma_v_kms, 1e-3)  # Avoid division by zero
        sigma_v_power = sigma_v ** beta_sigma
        tau_noise_sec = (R * 3.086e16) / (sigma_v_power * 1e3)  # seconds
        tau_noise_sec = np.clip(tau_noise_sec, 1e6, 1e20)
        
    elif method == "cluster":
        # τ_noise ~ L_turb / v_turb
        if v_turb_kms is None or L_turb_kpc is None:
            # Fallback: use σ_v scaling
            sigma_v = max(sigma_v_kms, 1e-3)
            tau_noise_sec = (R * 3.086e16) / (sigma_v * 1e3)
        else:
            v_turb = max(v_turb_kms, 1e-3)
            L_turb = max(L_turb_kpc, 0.1)
            # Use characteristic scale, but allow R-dependence
            tau_noise_sec = (L_turb * 3.086e16) / (v_turb * 1e3)
            # Could make it R-dependent: tau_noise(R) ~ L_turb(R) / v_turb(R)
            tau_noise_sec = np.full_like(R, tau_noise_sec)
        
        tau_noise_sec = np.clip(tau_noise_sec, 1e6, 1e20)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return tau_noise_sec


def compute_tau_coh(
    tau_geom: np.ndarray,
    tau_noise: np.ndarray,
) -> np.ndarray:
    """
    Combine geometric and noise timescales to get coherence time.
    
    τ_coh⁻¹ = τ_geom⁻¹ + τ_noise⁻¹
    
    Parameters:
    -----------
    tau_geom : np.ndarray
        Geometric dephasing time (seconds)
    tau_noise : np.ndarray
        Noise decoherence time (seconds)
        
    Returns:
    --------
    tau_coh : np.ndarray
        Coherence time in seconds
    """
    tau_geom = np.asarray(tau_geom, dtype=np.float64)
    tau_noise = np.asarray(tau_noise, dtype=np.float64)
    
    # Harmonic mean: 1/τ_coh = 1/τ_geom + 1/τ_noise
    inv_tau_coh = 1.0 / tau_geom + 1.0 / tau_noise
    tau_coh = 1.0 / np.clip(inv_tau_coh, 1e-20, None)
    
    return tau_coh


def compute_coherence_length(
    tau_coh_sec: np.ndarray,
    *,
    alpha: float = 0.037,
) -> np.ndarray:
    """
    Convert coherence time to coherence length.
    
    ℓ_coh = α · c · τ_coh
    
    Parameters:
    -----------
    tau_coh_sec : np.ndarray
        Coherence time in seconds
    alpha : float
        Prefactor to scale coherence length (default ~0.037 gives ℓ_coh ~ 5 kpc for MW)
        
    Returns:
    --------
    ell_coh_kpc : np.ndarray
        Coherence length in kpc
    """
    tau_coh = np.asarray(tau_coh_sec, dtype=np.float64)
    ell_coh_kpc = alpha * C_LIGHT_KMS * tau_coh / (3.086e16)  # Convert to kpc
    return ell_coh_kpc


def burr_xii_coherence_window(
    R_kpc: np.ndarray,
    ell_coh_kpc: np.ndarray,
    *,
    p: float = 0.757,
    n_coh: float = 0.5,
) -> np.ndarray:
    """
    Burr-XII coherence window: C(R/ℓ_coh).
    
    C(x) = 1 - [1 + x^p]^(-n_coh)
    
    Parameters:
    -----------
    R_kpc : np.ndarray
        Radii in kpc
    ell_coh_kpc : np.ndarray
        Coherence length profile ℓ_coh(R) in kpc
    p : float
        Burr-XII shape parameter
    n_coh : float
        Burr-XII shape parameter
        
    Returns:
    --------
    C : np.ndarray
        Coherence window (0 to 1)
    """
    R = np.asarray(R_kpc, dtype=np.float64)
    ell_coh = np.asarray(ell_coh_kpc, dtype=np.float64)
    
    # Avoid division by zero
    ell_coh = np.clip(ell_coh, 1e-6, None)
    x = R / ell_coh
    
    C = 1.0 - (1.0 + x**p) ** (-n_coh)
    return np.clip(C, 0.0, 1.0)


def compute_coherence_kernel(
    R_kpc: np.ndarray,
    g_bar_kms2: np.ndarray,
    sigma_v_kms: float,
    *,
    A_global: float = 1.0,
    p: float = 0.757,
    n_coh: float = 0.5,
    rho_bar_msun_pc3: Optional[np.ndarray] = None,
    method: str = "galaxy",
    v_turb_kms: Optional[float] = None,
    L_turb_kpc: Optional[float] = None,
    delta_R_kpc: float = 0.1,
    tau_geom_method: str = "tidal",
    alpha_length: float = 0.037,
    beta_sigma: float = 1.5,
    alpha_geom: float = 1.0,
    backreaction_cap: Optional[float] = None,
) -> np.ndarray:
    """
    Compute time-coherence based enhancement kernel K(R).
    
    K(R) = A_global · C(R / ℓ_coh(R))
    
    where ℓ_coh(R) = c · τ_coh(R) and τ_coh combines geometric and noise timescales.
    
    Parameters:
    -----------
    R_kpc : np.ndarray
        Radii in kpc
    g_bar_kms2 : np.ndarray
        Baryonic acceleration g_bar(R) in km/s²
    sigma_v_kms : float
        Velocity dispersion in km/s
    A_global : float
        Global amplitude
    p, n_coh : float
        Burr-XII shape parameters
    rho_bar_msun_pc3 : np.ndarray, optional
        Baryonic density profile (for tidal τ_geom)
    method : str
        "galaxy" or "cluster" (affects τ_noise)
    v_turb_kms, L_turb_kpc : float, optional
        Turbulence parameters for clusters
    delta_R_kpc : float
        Geodesic separation scale for τ_geom
    tau_geom_method : str
        "tidal" or "simple" for computing τ_geom
    backreaction_cap : float, optional
        Maximum allowed kernel value (universal back-reaction limit).
        If None, no cap is applied.
        
    Returns:
    --------
    K : np.ndarray
        Enhancement kernel K(R)
    """
    R = np.asarray(R_kpc, dtype=np.float64)
    g_bar = np.asarray(g_bar_kms2, dtype=np.float64)
    
    # Compute timescales
    tau_geom = compute_tau_geom(
        R, g_bar, rho_bar_msun_pc3, delta_R_kpc=delta_R_kpc, method=tau_geom_method, alpha_geom=alpha_geom
    )
    
    tau_noise = compute_tau_noise(
        R, sigma_v_kms, method=method, v_turb_kms=v_turb_kms, L_turb_kpc=L_turb_kpc, beta_sigma=beta_sigma
    )
    
    # Combine to get coherence time
    tau_coh = compute_tau_coh(tau_geom, tau_noise)
    
    # Convert to coherence length
    ell_coh = compute_coherence_length(tau_coh, alpha=alpha_length)
    
    # Compute coherence window
    C = burr_xii_coherence_window(R, ell_coh, p=p, n_coh=n_coh)
    
    # Raw kernel
    K = A_global * C
    
    # Small-scale suppression for Solar System safety
    # K → 0 as R → 0 to ensure no enhancement at Solar System scales
    # Use stronger suppression: (R/R_suppress)^4 for R < R_suppress
    R_suppress_kpc = 0.01  # 10 pc - suppression scale
    suppression = np.where(R < R_suppress_kpc, (R / R_suppress_kpc)**4, 1.0)
    K = K * suppression
    
    # Universal back-reaction limit: metric fluctuations decohere
    # when enhancement becomes too large
    if backreaction_cap is not None:
        K = np.minimum(K, backreaction_cap)
    
    return K


def compute_exposure_factor(
    R_kpc: np.ndarray,
    g_bar_kms2: np.ndarray,
    tau_coh_sec: np.ndarray,
) -> np.ndarray:
    """
    Compute 'exposure factor' Xi(R) = tau_coh / T_orb, where T_orb is the
    GR orbital period inferred from g_bar.
    
    This measures how much "extra proper time in the gravitational field"
    a test particle experiences relative to one GR orbital period.
    
    Parameters
    ----------
    R_kpc : array_like
        Radii in kpc.
    g_bar_kms2 : array_like
        Baryonic acceleration in km/s^2 (GR).
    tau_coh_sec : array_like
        Coherence time tau_coh in seconds.
        
    Returns
    -------
    Xi : ndarray
        Dimensionless exposure factor tau_coh / T_orb.
    """
    R_kpc = np.asarray(R_kpc, dtype=float)
    g_bar = np.asarray(g_bar_kms2, dtype=float)
    tau_coh = np.asarray(tau_coh_sec, dtype=float)
    
    # v_circ^2 = g_bar * R (in km^2/s^2, with R in kpc → convert to km)
    v_circ_kms = np.sqrt(np.clip(g_bar * R_kpc * 1e3, 1e-12, None))
    # Orbital period in seconds: T = 2πR / v_circ
    R_km = R_kpc * 3.086e16
    T_orb_sec = 2.0 * np.pi * R_km / (v_circ_kms * 1e3)
    T_orb_sec = np.clip(T_orb_sec, 1e10, None)  # avoid division by tiny T
    
    Xi = tau_coh / T_orb_sec
    return Xi

