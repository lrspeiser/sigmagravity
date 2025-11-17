"""
First-principles-inspired metric-resonance kernel utilities.

Implements:
  - fluctuation_spectrum_lambda: P(λ)
  - resonance_filter_lambda: C(λ, λ_m, Q)
  - geometric_weight: W(R, λ)
  - integrate_K_lambda: numerical integral over ln λ
  - theory_metric_resonance_K: vectorized K(R)
  - theory_metric_resonance_multiplier: f_res = 1 + K(R)
"""

from __future__ import annotations

import numpy as np


def fluctuation_spectrum_lambda(
    lam_kpc: np.ndarray,
    lam_p_kpc: float = 1e-9,
    alpha: float = 3.5,
    lam_cut_kpc: float = 500.0,
    A_P: float = 1.0,
) -> np.ndarray:
    """
    Toy metric-fluctuation spectrum:
        P(λ) = A_P * (λ_p / λ)^α * exp(-λ / λ_cut)
    """

    lam = np.asarray(lam_kpc, dtype=float)
    lam = np.maximum(lam, 1e-12)
    lam_p = max(lam_p_kpc, 1e-12)
    lam_cut = max(lam_cut_kpc, 1e-6)

    power_law = (lam_p / lam) ** alpha
    cutoff = np.exp(-lam / lam_cut)
    return A_P * power_law * cutoff


def resonance_filter_lambda(
    lam_kpc: np.ndarray,
    lam_m_kpc: float,
    Q: float,
    lam_coh_kpc: float,
    p_coh: float = 2.0,
) -> np.ndarray:
    """
    Lorentzian in ln(λ/λ_m) with width ~ 1/Q, times a coherence cutoff.
    """

    lam = np.asarray(lam_kpc, dtype=float)
    lam = np.maximum(lam, 1e-12)
    lam_m = max(lam_m_kpc, 1e-12)
    lam_coh = max(lam_coh_kpc, 1e-6)

    x = np.log(lam / lam_m)
    Q_eff = max(Q, 1e-3)
    lorentz = (Q_eff**2) / (Q_eff**2 + x * x)

    cutoff = np.exp(- (lam / lam_coh) ** p_coh)
    return lorentz * cutoff


def geometric_weight(R_kpc: float, lam_kpc: np.ndarray) -> np.ndarray:
    """
    Geometric weight: unity for λ <= R, else (R/λ)^2 dilution.
    """

    lam = np.asarray(lam_kpc, dtype=float)
    R = max(float(R_kpc), 1e-6)
    weight = np.ones_like(lam)
    mask = lam > R
    weight[mask] = (R / lam[mask]) ** 2
    return weight


def integrate_K_lambda(
    R_kpc: float,
    lam_m_kpc: float,
    Q: float,
    *,
    lam_min_kpc: float = 0.1,
    lam_max_kpc: float = 1000.0,
    n_lambda: int = 256,
    spectrum_params: dict | None = None,
    resonance_params: dict | None = None,
) -> float:
    """
    Numerically integrate K(R) = ∫ dln λ P(λ) C(λ,R) W(R,λ).
    """

    if spectrum_params is None:
        spectrum_params = {}
    if resonance_params is None:
        resonance_params = {}

    lam_min = max(lam_min_kpc, 1e-3)
    lam_max = max(lam_max_kpc, lam_min * 1.01)

    log_lam = np.linspace(np.log(lam_min), np.log(lam_max), n_lambda)
    lam = np.exp(log_lam)
    dloglam = log_lam[1] - log_lam[0]

    P_lam = fluctuation_spectrum_lambda(lam, **spectrum_params)
    C_lam = resonance_filter_lambda(lam, lam_m_kpc, Q, **resonance_params)
    W_lam = geometric_weight(R_kpc, lam)

    integrand = P_lam * C_lam * W_lam
    K_R = np.sum(integrand) * dloglam
    return float(K_R)


def theory_metric_resonance_K(
    R_kpc: np.ndarray,
    v_circ_kms: np.ndarray,
    sigma_v_kms: float,
    *,
    lam_matter_mode: str = "orbital_circumference",
    lam_coh_kpc: float = 20.0,
    p_coh: float = 2.0,
    spectrum_params: dict | None = None,
    resonance_extra: dict | None = None,
) -> np.ndarray:
    """
    Vectorized computation of K(R) for arrays of radii/velocities.
    """

    R = np.asarray(R_kpc, dtype=float)
    v_circ = np.asarray(v_circ_kms, dtype=float)
    sigma_v = max(float(sigma_v_kms), 1e-3)

    if spectrum_params is None:
        spectrum_params = {}
    if resonance_extra is None:
        resonance_extra = {}

    if lam_matter_mode == "orbital_circumference":
        lam_matter = 2.0 * np.pi * R
    elif lam_matter_mode == "R":
        lam_matter = R.copy()
    else:
        raise ValueError(f"Unsupported lam_matter_mode={lam_matter_mode!r}")

    Q = v_circ / sigma_v

    Ks = np.empty_like(R, dtype=float)
    for i in range(R.size):
        Ks[i] = integrate_K_lambda(
            float(R[i]),
            float(lam_matter[i]),
            float(Q[i]),
            lam_min_kpc=0.1,
            lam_max_kpc=1000.0,
            n_lambda=256,
            spectrum_params=spectrum_params,
            resonance_params=dict(
                lam_coh_kpc=lam_coh_kpc,
                p_coh=p_coh,
                **resonance_extra,
            ),
        )
    return Ks


def theory_metric_resonance_multiplier(
    R_kpc: np.ndarray,
    v_circ_kms: np.ndarray,
    sigma_v_kms: float,
    *,
    A_global: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Convenience hook: f_res^theory = 1 + A_global * K_theory(R).
    """

    K_R = theory_metric_resonance_K(R_kpc, v_circ_kms, sigma_v_kms, **kwargs)
    return 1.0 + A_global * K_R


def compute_theory_kernel(
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
) -> np.ndarray:
    """
    Standalone λ-integral kernel: returns K_theory(R) so that g_eff = g_GR * (1 + K).
    """

    R = np.asarray(R_kpc, dtype=float)
    lam_min = max(lam_min_kpc, 1e-6)
    lam_max = max(lam_max_kpc, lam_min * 1.01)
    lam_grid = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lambda)

    lam_ref = lam_min
    P_lambda = (lam_ref / lam_grid) ** alpha * np.exp(-lam_grid / max(lam_cut_kpc, 1e-6))

    lam_matter = 2.0 * np.pi * R[:, None]
    sigma_v = max(float(sigma_v_kms), 1e-6)
    Q = (v_circ_ref_kms / sigma_v) / max(Q_ref, 1e-6)

    ratio = lam_grid[None, :] / np.maximum(lam_matter, 1e-12)
    inv_ratio = np.maximum(lam_matter, 1e-12) / lam_grid[None, :]
    denom = Q**2 + (ratio - inv_ratio) ** 2
    C_res = (Q**2) / denom

    lam_p = (lam_grid / max(lam_coh_kpc, 1e-6)) ** 2.0
    C_coh = np.exp(-lam_p)

    lam_over_R = lam_grid[None, :] / np.maximum(R[:, None], 1e-6)
    W_geom = np.where(lam_over_R < 1.0, 1.0, lam_over_R**-2)

    integrand = (
        P_lambda[None, :]
        * C_res
        * C_coh[None, :]
        * W_geom
        / lam_grid[None, :]
    )
    K0 = np.trapz(integrand, lam_grid, axis=1)
    K0_max = np.max(np.abs(K0)) or 1.0
    return A_global * (K0 / K0_max)


