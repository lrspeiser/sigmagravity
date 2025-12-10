# SPDX-License-Identifier: MIT
# geff adapters: convenience factories and hooks
# Units: SI throughout (meters, seconds, kg). Choose kernel metric explicitly.

from __future__ import annotations
from typing import Callable, Optional
import importlib
import os
import numpy as np

Vec = np.ndarray
GeffFn = Callable[[Vec], Vec]

G_SI = 6.67430e-11  # m^3 kg^-1 s^-2


def coherence_kernel(R: np.ndarray | float, ell0: float, p: float, ncoh: float) -> np.ndarray | float:
    """
    C(R) = 1 - [1 + (R/ell0)^p]^(-ncoh)
    All inputs SI; R and ell0 in meters.
    """
    R = np.asarray(R, dtype=float)
    x = np.power(np.maximum(R, 0.0) / max(ell0, 1e-30), p)
    C = 1.0 - np.power(1.0 + x, -ncoh)
    return C


def geff_hernquist_factory(
    M: float,
    a: float,
    ell0: float,
    p: float,
    ncoh: float,
    *,
    A: float = 1.0,
    G: float = G_SI,
    kernel_metric: str = "spherical",  # "spherical" (strictly curl-safe) or "cylindrical"
) -> GeffFn:
    """
    Build geff(x) for a spherical Hernquist baryon model with a Σ-kernel multiplier.
    - g_bar = -GM/(r+a)^2 r̂  (handles r→0)
    - K = A * C(R_metric), where R_metric is r (spherical) or sqrt(x^2+y^2) (cylindrical)
    - geff = (1 + K) * g_bar = (1 + A*C) * g_bar
    """
    M = float(M); a = float(a); ell0 = float(ell0); p = float(p); ncoh = float(ncoh); A = float(A); G = float(G)
    use_cyl = (str(kernel_metric).lower().strip() == "cylindrical")

    def geff(x: Vec) -> Vec:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != 3:
            raise ValueError("x must be a 3-vector")
        r = float(np.linalg.norm(x))
        if r == 0.0:
            return np.zeros(3, dtype=float)
        xhat = x / r
        gbar_mag = - G * M / (r + max(a, 0.0)) ** 2
        gbar = gbar_mag * xhat  # [m/s^2]
        R_metric = r if not use_cyl else float(np.hypot(x[0], x[1]))
        K = float(A) * float(coherence_kernel(R_metric, ell0, p, ncoh))
        return (1.0 + K) * gbar

    return geff


def geff_point_masses_factory(
    masses: np.ndarray,
    positions: np.ndarray,
    ell0: float,
    p: float,
    ncoh: float,
    *,
    softening: float = 0.0,
    A: float = 1.0,
    G: float = G_SI,
    kernel_metric: str = "spherical",
) -> GeffFn:
    """
    Build geff(x) from discrete point masses, multiplied by a Σ-kernel.
    Note: For strict curl-free behavior, prefer kernel_metric="spherical" and
    use symmetric mass models. Research-only convenience.
    geff = (1 + A*C) * g_bar, with C = coherence_kernel(R_metric, ...).
    """
    masses = np.asarray(masses, dtype=float).reshape(-1)
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be (N,3)")
    if positions.shape[0] != masses.size:
        raise ValueError("masses and positions length mismatch")
    eps = float(max(softening, 0.0))
    use_cyl = (str(kernel_metric).lower().strip() == "cylindrical")
    A = float(A); G = float(G)

    def geff(x: Vec) -> Vec:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != 3:
            raise ValueError("x must be a 3-vector")
        acc = np.zeros(3, dtype=float)
        dx = x - positions  # (N,3)
        r2 = np.einsum('ij,ij->i', dx, dx) + eps * eps
        inv_r3 = 1.0 / np.power(r2, 1.5)
        # g_bar = -G Σ m_i r_i / |r_i|^3
        acc = -G * (dx * (masses * inv_r3)[:, None]).sum(axis=0)
        R_metric = float(np.linalg.norm(x)) if not use_cyl else float(np.hypot(x[0], x[1]))
        K = float(A) * float(coherence_kernel(R_metric, ell0, p, ncoh))
        return (1.0 + K) * acc

    return geff


def geff_from_env(var: str = "SIGMA_GEFF") -> GeffFn:
    """
    Import a user-provided geff(x)->3-vector from environment var:
        SIGMA_GEFF="module.submodule:function_name"
    Returns a wrapper that validates output shape.
    """
    spec = os.environ.get(var, "").strip()
    if not spec or ":" not in spec:
        raise RuntimeError(f"Env var {var} not set to 'module:function' spec")
    mod_name, func_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)

    def geff(x: Vec) -> Vec:
        v = np.asarray(fn(np.asarray(x, dtype=float)), dtype=float).reshape(-1)
        if v.size != 3 or not np.all(np.isfinite(v)):
            raise ValueError("Imported geff must return finite 3-vector")
        return v

    return geff


__all__ = [
    "coherence_kernel",
    "geff_hernquist_factory",
    "geff_point_masses_factory",
    "geff_from_env",
]
