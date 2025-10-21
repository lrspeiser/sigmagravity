# SPDX-License-Identifier: MIT
# Local gravitational redshift utilities for Σ‑Gravity experiments
# This module is research-only and self-contained under redshift/.
# Units: SI (meters, seconds). Use kpc/Mpc shorthands only in callers.

from __future__ import annotations
import logging
from typing import Callable, Tuple
import numpy as np

# Physical constant (SI)
c: float = 299_792_458.0

# Types
Vec = np.ndarray
GeffFn = Callable[[Vec], Vec]  # geff(x) -> 3-vector [m/s^2]
SigmaFn = Callable[[Vec, float], float] | None
PhiFn = Callable[[Vec], float] | None

logger = logging.getLogger("redshift")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(levelname)s] %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


def _as_vec3(x: Vec) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 3:
        raise ValueError("position must be length-3 vector")
    return x


def line_integral_potential(
    x: Vec,
    geff: GeffFn,
    r_max: float = 5.0e22,
    n_steps: int = 2000,
) -> float:
    """
    Compute Ψ_eff(x) up to an additive constant by integrating along the straight
    radial path from R_max (treated as ~infinity) down to x.

    Ψ_eff(x) := -∫ g_eff · dl, with dl = r̂ dr along the radial line ⇒
    Ψ_eff(x) = -∫_{R_max}^{|x|} (g_eff(x(r))·r̂) dr, anchored at Ψ_eff(R_max)=0.

    Parameters
    ----------
    x : array-like, shape (3,), position in meters
    geff : callable returning g_eff(x) as a 3-vector [m/s^2]
    r_max : float, outer anchor radius [m] (choose >> baryon extent)
    n_steps : int, radial samples for the trapezoid integration (>=10)

    Returns
    -------
    float : Ψ_eff(x) in [m^2/s^2]
    """
    x = _as_vec3(x)
    r = float(np.linalg.norm(x))
    if r == 0.0:
        return 0.0
    if n_steps < 10:
        raise ValueError("n_steps must be >= 10 for stable integration")

    R0 = float(r_max)
    if not np.isfinite(R0) or R0 <= 0:
        raise ValueError("r_max must be positive and finite")
    if R0 <= r:
        logger.warning("r_max (%.3e m) <= |x| (%.3e m); increasing anchor to 1.1*|x|", R0, r)
        R0 = 1.1 * r

    xhat = x / r
    radii = np.linspace(R0, r, int(n_steps) + 1, dtype=float)

    # Evaluate radial component g_r = g_eff(x(r))·r̂ at each sample
    g_r = np.empty_like(radii)
    for i, R in enumerate(radii):
        xi = R * xhat
        gi = geff(xi)
        gi = np.asarray(gi, dtype=float).reshape(-1)
        if gi.size != 3 or not np.all(np.isfinite(gi)):
            raise ValueError("geff must return finite 3-vector")
        g_r[i] = float(np.dot(gi, xhat))

    # Ψ = -∫ g_r dr from R0 to r
    psi = -float(np.trapz(g_r, radii))
    return psi


def gravitational_redshift_endpoint(
    x_emit: Vec,
    x_obs: Vec,
    geff: GeffFn,
    r_max: float = 5.0e22,
    n_steps: int = 2000,
) -> float:
    """
    Endpoint gravitational redshift for a static field:
        z_end ≈ [Ψ_eff(obs) − Ψ_eff(emit)] / c^2.
    """
    psi_e = line_integral_potential(x_emit, geff, r_max=r_max, n_steps=n_steps)
    psi_o = line_integral_potential(x_obs, geff, r_max=r_max, n_steps=n_steps)
    return (psi_o - psi_e) / (c ** 2)


def los_redshift_isw_like(
    x_emit: Vec,
    x_obs: Vec,
    t_emit: float,
    t_obs: float,
    geff: GeffFn,
    sigma: SigmaFn = None,
    phi_bar: PhiFn = None,
    n_steps: int = 2000,
) -> Tuple[float, float]:
    """
    ISW-like (line-of-sight) redshift from time variation.

    If either `sigma` (dimensionless, time-varying field) is None or `phi_bar` is None,
    returns (0.0, 0.0).

    We implement Δν/ν = -(1/c^2) ∫ ∂_t[(1+σ)(Ψ_bar+Φ_bar)] dλ.
    In weak field and Φ≈Ψ, approximate with A(t) = (1+σ) * φ_bar and dλ ≈ c dt along
    the unperturbed path (Born approximation):
        z_los_numeric   = -(2/c^2) ∫ dA/dt dt
        z_los_endpoint  = -(2/c^2) [A(t_obs) - A(t_emit)]
    """
    if sigma is None or phi_bar is None:
        return 0.0, 0.0

    x_e = _as_vec3(x_emit)
    x_o = _as_vec3(x_obs)
    if not (np.isfinite(t_emit) and np.isfinite(t_obs)):
        raise ValueError("t_emit and t_obs must be finite")
    if n_steps < 10:
        raise ValueError("n_steps must be >= 10 for stable integration")

    Lvec = x_o - x_e
    s = np.linspace(0.0, 1.0, int(n_steps) + 1)
    t = t_emit + (t_obs - t_emit) * s

    A = np.empty_like(s)
    for i, si in enumerate(s):
        xi = x_e + si * Lvec
        sig = float(sigma(xi, t[i]))
        ph  = float(phi_bar(xi))
        if not (np.isfinite(sig) and np.isfinite(ph)):
            raise ValueError("sigma/phi_bar must return finite scalars")
        A[i] = (1.0 + sig) * ph

    dAdt = np.gradient(A, t)
    z_los_numeric  = - (2.0 / (c ** 2)) * float(np.trapz(dAdt, t))
    z_los_endpoint = - (2.0 / (c ** 2)) * float(A[-1] - A[0])
    return z_los_numeric, z_los_endpoint


__all__ = [
    "c",
    "line_integral_potential",
    "gravitational_redshift_endpoint",
    "los_redshift_isw_like",
]
