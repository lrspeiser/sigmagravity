"""
Metric-resonance multiplier for Sigma-Gravity Phase 1 tests.
"""

import numpy as np


def burr_xii_coherence_window(R_kpc, ell0_kpc, p, n_coh):
    R = np.asarray(R_kpc, dtype=float)
    ell0 = float(max(ell0_kpc, 1e-6))
    x = (R / ell0) ** p
    return 1.0 - (1.0 + x) ** (-n_coh)


def lognormal_resonance(lambda_orb_kpc, lambda_peak_kpc, sigma_ln_lambda):
    lam = np.asarray(lambda_orb_kpc, dtype=float)
    lam = np.maximum(lam, 1e-9)
    log_lam = np.log(lam)
    log_peak = np.log(max(lambda_peak_kpc, 1e-9))
    width = max(sigma_ln_lambda, 1e-6)
    z = (log_lam - log_peak) / width
    return np.exp(-0.5 * z * z)


def metric_resonance_multiplier(
    R_kpc,
    lambda_orb_kpc,
    *,
    A=1.5,
    ell0_kpc=5.0,
    p=0.75,
    n_coh=0.5,
    lambda_peak_kpc=15.0,
    sigma_ln_lambda=1.0,
):
    C_R = burr_xii_coherence_window(R_kpc, ell0_kpc, p, n_coh)
    W_lam = lognormal_resonance(lambda_orb_kpc, lambda_peak_kpc, sigma_ln_lambda)
    return 1.0 + A * C_R * W_lam


def sigma_gate_amplitude(
    sigma_v_kms,
    A_base,
    sigma_ref_kms=30.0,
    beta_sigma=0.4,
    clamp=True,
):
    """
    Sigma-velocity gate for the resonance amplitude.

    A_eff(sigma) = A_base * min(1, (sigma_ref / sigma)**beta)   if clamp=True
                 = A_base * (sigma_ref / sigma)**beta          if clamp=False

    Parameters
    ----------
    sigma_v_kms : float
        Galaxy-level velocity dispersion [km/s].
    A_base : float
        Baseline MW amplitude (Phase-1 fit).
    sigma_ref_kms : float, optional
        Reference dispersion for a "cold" disk (default 30 km/s).
    beta_sigma : float, optional
        Gating exponent; larger => stronger damping for hot systems.
    clamp : bool, optional
        If True (recommended), never let A_eff exceed A_base.
    """
    sigma = max(float(sigma_v_kms), 1e-3)
    gate = (sigma_ref_kms / sigma) ** beta_sigma
    if clamp:
        gate = min(1.0, gate)
    return A_base * gate


def metric_resonance_multiplier_sigma(
    R_kpc,
    lambda_orb_kpc,
    *,
    sigma_v_kms,
    A_base,
    sigma_ref_kms=30.0,
    beta_sigma=0.4,
    ell0_kpc=5.0,
    p=0.75,
    n_coh=0.5,
    lambda_peak_kpc=15.0,
    sigma_ln_lambda=1.0,
    clamp=True,
):
    """
    Sigma-gated version of the metric-resonance multiplier.

    f_res_sigma(R, lambda; sigma_v) = 1 + A_eff(sigma_v) * C(R) * W(lambda),
    where A_eff is computed with sigma_gate_amplitude.

    Parameters
    ----------
    R_kpc, lambda_orb_kpc : array_like
        Radii and orbital wavelengths [kpc].
    sigma_v_kms : float
        Galaxy-level velocity dispersion [km/s].
    A_base : float
        Baseline MW amplitude from Phase-1 fit.
    sigma_ref_kms, beta_sigma : floats
        Sigma-gating hyperparameters.
    ell0_kpc, p, n_coh, lambda_peak_kpc, sigma_ln_lambda : floats
        Kernel-shape parameters as before.
    clamp : bool, optional
        If True, cap A_eff <= A_base.
    """
    A_eff = sigma_gate_amplitude(
        sigma_v_kms=sigma_v_kms,
        A_base=A_base,
        sigma_ref_kms=sigma_ref_kms,
        beta_sigma=beta_sigma,
        clamp=clamp,
    )
    return metric_resonance_multiplier(
        R_kpc,
        lambda_orb_kpc,
        A=A_eff,
        ell0_kpc=ell0_kpc,
        p=p,
        n_coh=n_coh,
        lambda_peak_kpc=lambda_peak_kpc,
        sigma_ln_lambda=sigma_ln_lambda,
    )


__all__ = [
    "burr_xii_coherence_window",
    "lognormal_resonance",
    "metric_resonance_multiplier",
    "sigma_gate_amplitude",
    "metric_resonance_multiplier_sigma",
]
