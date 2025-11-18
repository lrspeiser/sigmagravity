"""
Microphysics Model 1: Roughness / Time-Coherence (Path-Integral Decoherence)

Field-equation picture:
Start with GR + small stochastic metric fluctuations:
    g_μν = ḡ_μν + h_μν,  ⟨h_μν⟩ = 0, ⟨h h⟩ ≠ 0

After coarse-graining over fast fluctuations:
    G_μν[ḡ] = 8πG(T_μν + T^(rough)_μν[⟨h h⟩])

In weak-field, quasi-static limit:
    ∇²Φ = 4πG ρ + δS[ρ, σ_v, τ_coh]

The effective Newtonian law:
    g_eff(R) = g_GR(R) * [1 + K_rough(Ξ_system)]

where Ξ is the system-level exposure factor.

This is a "minimally invasive" first-principles story: you don't touch the field 
equations locally, you renormalize the effective coupling by integrating out fast 
stochastic modes.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RoughnessParams:
    """Parameters for the roughness/time-coherence microphysics model."""
    alpha_length: float = 0.037      # coherence length scale factor (already tuned)
    beta_sigma: float = 1.5          # velocity dispersion exponent (already tuned)
    K0: float = 0.774                # amplitude from K(Ξ) fit
    gamma: float = 0.1               # exposure exponent from K(Ξ) fit
    alpha_geom: float = 1.0          # geometric timescale prefactor
    tau_geom_method: str = "tidal"   # method for computing τ_geom


def compute_tau_geom(R_kpc, v_circ_kms):
    """
    Compute geometric dephasing time τ_geom ~ R / v_circ.
    
    Two nearby geodesics accumulate different proper times due to 
    gravitational time dilation. This is the simpler scaling.
    
    Parameters
    ----------
    R_kpc : array_like
        Radii in kpc
    v_circ_kms : array_like
        Circular velocity in km/s
        
    Returns
    -------
    tau_geom_sec : ndarray
        Geometric dephasing time in seconds
    """
    R = np.asarray(R_kpc, dtype=float)
    v_circ = np.asarray(v_circ_kms, dtype=float)
    
    # τ_geom ~ T_orb ~ 2πR / v_circ
    # (simplified version; full version uses tidal field)
    R_km = R * 3.086e16  # kpc to km
    tau_geom_sec = 2.0 * np.pi * R_km / np.maximum(v_circ * 1e3, 1e-3)
    return tau_geom_sec


def compute_tau_noise(R_kpc, sigma_v_kms, beta_sigma):
    """
    Compute noise-driven decoherence time τ_noise ~ R / σ_v^β.
    
    Parameters
    ----------
    R_kpc : array_like
        Radii in kpc
    sigma_v_kms : array_like
        Velocity dispersion in km/s
    beta_sigma : float
        Exponent for velocity dispersion scaling
        
    Returns
    -------
    tau_noise_sec : ndarray
        Noise decoherence time in seconds
    """
    R = np.asarray(R_kpc, dtype=float)
    sigma_v = np.asarray(sigma_v_kms, dtype=float)
    
    # τ_noise ~ R / σ_v^β
    R_km = R * 3.086e16  # kpc to km
    sigma_v_power = np.maximum(sigma_v, 1e-3) ** beta_sigma
    tau_noise_sec = R_km / (sigma_v_power * 1e3)
    return tau_noise_sec


def compute_tau_coh(tau_geom, tau_noise):
    """
    Combine timescales: τ_coh^(-1) = τ_geom^(-1) + τ_noise^(-1).
    
    Parameters
    ----------
    tau_geom : array_like
        Geometric dephasing time (seconds)
    tau_noise : array_like
        Noise decoherence time (seconds)
        
    Returns
    -------
    tau_coh : ndarray
        Coherence time in seconds
    """
    tau_geom = np.asarray(tau_geom, dtype=float)
    tau_noise = np.asarray(tau_noise, dtype=float)
    
    inv_tau_coh = 1.0 / np.maximum(tau_geom, 1e-20) + 1.0 / np.maximum(tau_noise, 1e-20)
    tau_coh = 1.0 / np.maximum(inv_tau_coh, 1e-20)
    return tau_coh


def compute_orbital_period(R_kpc, v_circ_kms):
    """
    Compute orbital period T_orb = 2πR / v_circ.
    
    Parameters
    ----------
    R_kpc : array_like
        Radii in kpc
    v_circ_kms : array_like
        Circular velocity in km/s
        
    Returns
    -------
    T_orb_sec : ndarray
        Orbital period in seconds
    """
    R = np.asarray(R_kpc, dtype=float)
    v_circ = np.asarray(v_circ_kms, dtype=float)
    
    R_km = R * 3.086e16  # kpc to km
    T_orb_sec = 2.0 * np.pi * R_km / np.maximum(v_circ * 1e3, 1e-3)
    return T_orb_sec


def system_level_exposure(R_kpc, v_circ_kms, sigma_v_kms, params: RoughnessParams):
    """
    Compute system-level exposure factor Ξ from arrays of R, v_circ, σ_v.
    
    Ξ = τ_coh / T_orb
    
    We take a robust average (median) over the radial range where the RC is measured.
    
    Parameters
    ----------
    R_kpc : array_like
        Radii in kpc
    v_circ_kms : array_like
        Circular velocity in km/s
    sigma_v_kms : array_like
        Velocity dispersion in km/s (can be array or scalar)
    params : RoughnessParams
        Model parameters
        
    Returns
    -------
    Xi_sys : float
        System-level exposure factor
    """
    R = np.asarray(R_kpc, dtype=float)
    v_circ = np.asarray(v_circ_kms, dtype=float)
    sigma_v = np.asarray(sigma_v_kms, dtype=float)
    
    # Broadcast sigma_v if scalar
    if sigma_v.ndim == 0 or len(sigma_v) == 1:
        sigma_v = np.full_like(R, float(sigma_v))
    
    tau_geom = compute_tau_geom(R, v_circ)
    tau_noise = compute_tau_noise(R, sigma_v, params.beta_sigma)
    tau_coh = compute_tau_coh(tau_geom, tau_noise)
    T_orb = compute_orbital_period(R, v_circ)
    
    Xi = tau_coh / np.maximum(T_orb, 1e-12)
    
    # System-level statistic: median (robust to outliers)
    Xi_sys = float(np.median(Xi[np.isfinite(Xi)]))
    return Xi_sys


def K_rough_from_Xi(Xi, params: RoughnessParams):
    """
    Microphysics kernel: K_rough(Ξ) = K0 * Ξ^γ.
    
    This is the empirically determined law from Phase-2 fits.
    
    Parameters
    ----------
    Xi : float
        System-level exposure factor
    params : RoughnessParams
        Model parameters
        
    Returns
    -------
    K_rough : float
        Enhancement factor
    """
    Xi_safe = max(float(Xi), 1e-6)
    return params.K0 * Xi_safe ** params.gamma


def apply_roughness_boost(g_gr, Xi, params: RoughnessParams):
    """
    Apply roughness boost: g_eff = g_GR * (1 + K_rough(Ξ)).
    
    Note that K_rough is constant per system, not R-dependent.
    This is a *feature*, not a bug: system-level properties
    set the effective coupling.
    
    Parameters
    ----------
    g_gr : array_like
        GR acceleration in km/s²
    Xi : float
        System-level exposure factor
    params : RoughnessParams
        Model parameters
        
    Returns
    -------
    g_eff : ndarray
        Enhanced acceleration in km/s²
    """
    K_rough = K_rough_from_Xi(Xi, params)
    g_eff = np.asarray(g_gr, dtype=float) * (1.0 + K_rough)
    return g_eff

