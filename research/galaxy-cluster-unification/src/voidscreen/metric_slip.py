"""One-parameter weak-field metric slip tied to a screened matter response."""

from __future__ import annotations

import math

import numpy as np


def metric_slip_eta(gbar_m_s2, gdyn_m_s2, slip_s: float) -> np.ndarray:
    """Return ``eta=Psi/Phi`` for the declared weak-field potential split.

    ``Phi=Phi_N+phi`` and ``Psi=Phi_N+(1+s)phi`` imply
    ``eta=1+s*(g_dyn-g_bar)/g_dyn`` for aligned spherical gradients.
    """
    gbar = np.asarray(gbar_m_s2, dtype=float)
    gdyn = np.asarray(gdyn_m_s2, dtype=float)
    gbar, gdyn = np.broadcast_arrays(gbar, gdyn)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(gdyn)) or np.any(gdyn < gbar):
        raise ValueError("gdyn must be finite and at least gbar")
    if not math.isfinite(slip_s):
        raise ValueError("slip_s must be finite")
    return 1.0 + float(slip_s) * (gdyn - gbar) / gdyn


def metric_slip_lensing_acceleration(gbar_m_s2, gdyn_m_s2, slip_s: float) -> np.ndarray:
    """Return the Weyl/lensing acceleration ``(grad Phi + grad Psi)/2``."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    gdyn = np.asarray(gdyn_m_s2, dtype=float)
    gbar, gdyn = np.broadcast_arrays(gbar, gdyn)
    eta = metric_slip_eta(gbar, gdyn, slip_s)
    lensing = gbar + (1.0 + 0.5 * float(slip_s)) * (gdyn - gbar)
    if np.any(~np.isfinite(lensing)) or np.any(lensing <= 0.0):
        raise ValueError("lensing acceleration must be finite and positive")
    if np.any(~np.isfinite(eta)):
        raise ValueError("metric slip must be finite")
    return lensing


def extra_force_lensing_ratio(slip_s: float) -> float:
    """Return the lensing-to-dynamics ratio for the non-Newtonian extra force."""
    if not math.isfinite(slip_s):
        raise ValueError("slip_s must be finite")
    return 1.0 + 0.5 * float(slip_s)
