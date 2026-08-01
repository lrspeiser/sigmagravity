from __future__ import annotations

import numpy as np


def required_response(gbar_m_s2, observed_g_m_s2) -> dict[str, np.ndarray]:
    """Invert observed accelerations into constitutive-response targets.

    The pointwise acceleration-scale inverses are defined only where the
    observed acceleration exceeds the baryonic acceleration. Invalid entries
    are retained as NaN so callers cannot silently clip or delete them.
    """
    gbar = np.asarray(gbar_m_s2, dtype=float)
    observed = np.asarray(observed_g_m_s2, dtype=float)
    if gbar.shape != observed.shape:
        raise ValueError("gbar and observed acceleration must have matching shapes")
    if np.any(~np.isfinite(gbar)) or np.any(~np.isfinite(observed)):
        raise ValueError("accelerations must be finite")
    if np.any(gbar <= 0.0) or np.any(observed <= 0.0):
        raise ValueError("accelerations must be positive")

    nu = observed / gbar
    mu = 1.0 / nu
    valid = mu < 1.0
    rar_scale = np.full_like(gbar, np.nan)
    simple_scale = np.full_like(gbar, np.nan)
    standard_scale = np.full_like(gbar, np.nan)

    valid_mu = mu[valid]
    rar_root = -np.log1p(-valid_mu)
    rar_scale[valid] = gbar[valid] / np.square(rar_root)
    simple_scale[valid] = observed[valid] * (1.0 - valid_mu) / valid_mu
    standard_scale[valid] = (
        observed[valid] * np.sqrt(1.0 - np.square(valid_mu)) / valid_mu
    )
    return {
        "nu_required": nu,
        "mu_required": mu,
        "extra_g_m_s2": observed - gbar,
        "inverse_valid": valid,
        "rar_a_eff_m_s2": rar_scale,
        "simple_a_x_m_s2": simple_scale,
        "standard_a_x_m_s2": standard_scale,
    }


def simple_mu_acceleration(gbar_m_s2, a_x_m_s2) -> np.ndarray:
    """Solve gbar = [x/(1+x)] g with x=g/a_x for g."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    a_x = np.asarray(a_x_m_s2, dtype=float)
    if np.any(gbar <= 0.0) or np.any(a_x <= 0.0):
        raise ValueError("gbar and a_x must be positive")
    return 0.5 * (gbar + np.sqrt(np.square(gbar) + 4.0 * gbar * a_x))


def standard_mu_acceleration(gbar_m_s2, a_x_m_s2) -> np.ndarray:
    """Solve gbar = [x/sqrt(1+x^2)] g with x=g/a_x for g."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    a_x = np.asarray(a_x_m_s2, dtype=float)
    if np.any(gbar <= 0.0) or np.any(a_x <= 0.0):
        raise ValueError("gbar and a_x must be positive")
    gbar_squared = np.square(gbar)
    g_squared = 0.5 * (
        gbar_squared
        + np.sqrt(np.square(gbar_squared) + 4.0 * gbar_squared * np.square(a_x))
    )
    return np.sqrt(g_squared)
