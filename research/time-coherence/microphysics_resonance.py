"""
Microphysics Model 3: Metric Resonance with Specific Fluctuation Spectrum

Treat gravity as responding to a spectrum of metric fluctuations (gravitational waves,
quantum foam, whatever your preferred microscopic picture) with a resonant coupling 
to orbital scales.

Effective theory:
- Start from GR + stochastic background of metric fluctuations with power spectrum P(λ)
- In Fourier space: ⟨h h⟩(λ) ∝ P(λ) ∝ λ^(-α) exp(-λ/λ_cut)
- Matter with orbital wavelength λ_orb(R) responds most strongly to modes that match:
  
  C(λ,R) ~ Q(R)² / [Q(R)² + (λ/λ_orb - λ_orb/λ)²]
  
  with quality factor Q ~ v_circ/σ_v

The effective enhancement:
    K_res(R) = A_res ∫ d ln λ P(λ) C(λ,R) W(R,λ)

where W is a geometric weight (W ≈ 1 for λ ≲ R).

In practice this integral gives something quite close to the metric-resonance kernel 
you already coded: a Burr-like radial window times a log-normal peak in λ.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ResonanceParams:
    """Parameters for the metric resonance microphysics model."""
    A_res: float = 1.0            # overall amplitude
    alpha: float = 2.0            # spectral slope P(λ) ~ λ^(-α)
    lam_coh_kpc: float = 10.0     # coherence scale (characteristic wavelength)
    lam_cut_kpc: float = 2000.0   # large-scale cutoff
    Q0: float = 5.0               # reference quality factor
    beta_Q: float = 1.0           # Q ~ (v_circ/sigma_v)^beta_Q


def fluctuation_spectrum(lam_kpc, params: ResonanceParams):
    """
    Metric fluctuation power spectrum P(λ).
    
    P(λ) = λ^(-α) * exp(-λ/λ_cut)
    
    Parameters
    ----------
    lam_kpc : array_like
        Wavelength in kpc
    params : ResonanceParams
        Model parameters
        
    Returns
    -------
    P : ndarray
        Power spectrum (arbitrary units)
    """
    lam = np.asarray(lam_kpc, dtype=float)
    lam = np.maximum(lam, 1e-6)
    return lam**(-params.alpha) * np.exp(-lam / params.lam_cut_kpc)


def resonance_filter(lam_kpc, lam_orb_kpc, Q):
    """
    Resonance filter: C(λ, λ_orb, Q).
    
    C = Q² / [Q² + (λ/λ_orb - λ_orb/λ)²]
    
    This peaks when λ = λ_orb (resonance) and falls off for detuned modes.
    The width is controlled by quality factor Q.
    
    Parameters
    ----------
    lam_kpc : array_like
        Wavelength in kpc
    lam_orb_kpc : float or array_like
        Orbital wavelength in kpc
    Q : float or array_like
        Quality factor (v_circ / sigma_v)
        
    Returns
    -------
    C : ndarray
        Resonance filter (0 to 1)
    """
    lam = np.asarray(lam_kpc, dtype=float)
    lam_orb = np.asarray(lam_orb_kpc, dtype=float)
    lam = np.maximum(lam, 1e-6)
    lam_orb = np.maximum(lam_orb, 1e-6)
    
    # Detuning parameter: x = λ/λ_orb - λ_orb/λ
    x = lam/lam_orb - lam_orb/lam
    
    # Resonance filter
    Q_arr = np.asarray(Q, dtype=float)
    C = Q_arr**2 / (Q_arr**2 + x**2)
    return C


def K_resonance_profile(R_kpc, v_circ_kms, sigma_v_kms, params: ResonanceParams):
    """
    Compute metric resonance enhancement kernel K_res(R).
    
    K_res(R) = A_res ∫ d ln λ P(λ) C(λ, λ_orb(R), Q(R)) W(R, λ)
    
    where:
        - P(λ) is the fluctuation spectrum
        - C is the resonance filter
        - W is a geometric weight
        - λ_orb = 2πR (orbital wavelength)
        - Q = (v_circ / σ_v)^β_Q (quality factor)
    
    Parameters
    ----------
    R_kpc : array_like
        Radii in kpc
    v_circ_kms : array_like
        Circular velocity in km/s
    sigma_v_kms : array_like
        Velocity dispersion in km/s (can be array or scalar)
    params : ResonanceParams
        Model parameters
        
    Returns
    -------
    K_res : ndarray
        Enhancement kernel
    """
    R = np.asarray(R_kpc, dtype=float)
    v_circ = np.asarray(v_circ_kms, dtype=float)
    sigma_v = np.asarray(sigma_v_kms, dtype=float)
    
    # Broadcast sigma_v if scalar
    if sigma_v.ndim == 0 or (sigma_v.ndim == 1 and len(sigma_v) == 1):
        sigma_v = np.full_like(R, float(sigma_v))
    
    # Orbital wavelength: λ_orb = 2πR
    lam_orb = 2.0 * np.pi * R
    
    # Quality factor: Q = (v_circ / σ_v)^β_Q
    Q = (v_circ / np.maximum(sigma_v, 1e-3)) ** params.beta_Q
    
    # Integrate over lambda on a log grid
    # Cover wavelengths from 0.1 kpc to 10^4 kpc
    lam_grid = np.logspace(-1, 4, 256)  # 256 points
    dlnlam = np.log(lam_grid[1] / lam_grid[0])
    
    # Fluctuation spectrum
    P = fluctuation_spectrum(lam_grid, params)
    
    # Compute K for each radius by integrating
    K_list = []
    for R_i, lam_orb_i, Q_i in zip(R, lam_orb, Q):
        # Resonance filter for this radius
        C = resonance_filter(lam_grid, lam_orb_i, Q_i)
        
        # Geometric weight: suppress modes much larger than R
        # W(λ) = 1 / (1 + (λ/R)²)
        W = 1.0 / (1.0 + (lam_grid / max(R_i, 1e-3))**2)
        
        # Integrand
        integrand = P * C * W
        
        # Integrate (simple rectangular rule)
        K_i = params.A_res * np.sum(integrand) * dlnlam
        K_list.append(K_i)
    
    return np.array(K_list)


def apply_resonance_boost(g_gr, R_kpc, v_circ_kms, sigma_v_kms, params: ResonanceParams):
    """
    Apply resonance boost: g_eff = g_GR * (1 + K_res(R)).
    
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
    params : ResonanceParams
        Model parameters
        
    Returns
    -------
    g_eff : ndarray
        Enhanced acceleration in km/s²
    """
    K_res = K_resonance_profile(R_kpc, v_circ_kms, sigma_v_kms, params)
    g_eff = np.asarray(g_gr, dtype=float) * (1.0 + K_res)
    return g_eff

