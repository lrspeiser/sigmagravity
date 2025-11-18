"""
Mass-coherence model for F_missing (multiplicative ratio) with σ_v gating.

F_missing = K_total / K_rough, i.e. how much larger the total kernel
should be compared to the rough component.

This is a RATIO, not a kernel by itself.

Now includes σ_v-based gating to shut down F_missing for high-dispersion systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FMissingParams:
    """
    Hyperparameters for the mass–coherence 'missing' component.

    These came from your functional fit:
        F_missing ≈ 10.02 * (14.8 / σ_v)^0.10 * (12.3 / R_d)^0.31

    but we now:
      * allow exponents to vary,
      * add σ_v gating (gamma_sigma),
      * and clamp at F_max.
    """
    A0: float = 10.02          # overall amplitude of the ratio
    sigma_ref: float = 14.8    # km/s, reference dispersion
    R_ref: float = 12.3        # kpc, reference disk scale length

    a_sigma: float = 0.10      # exponent on (sigma_ref / sigma_v)
    a_Rd: float = 0.31         # exponent on (R_ref / R_d)

    F_max: float = 5.0         # hard cap on F (pre-gating)

    # NEW: σ_v–based coherence gate
    use_sigma_gate: bool = True
    sigma_gate_ref: float = 25.0  # km/s; where gate ≈ 0.5 when gamma_sigma ~ 1
    gamma_sigma: float = 1.0      # controls how fast F_missing shuts off at high σ_v

    # Optional floor so very hot systems still keep some fraction
    gate_floor: float = 0.1       # min gate value (0–1)


# Backward compatibility alias
MassCoherenceParams = FMissingParams


def _base_F_missing(
    sigma_v: np.ndarray,
    R_d: np.ndarray,
    params: FMissingParams,
) -> np.ndarray:
    """
    Raw F_missing from the mass–coherence fit, *before* σ gating and clamping.

    F_raw = A0 * (sigma_ref / σ_v)^a_sigma * (R_ref / R_d)^a_Rd
    """
    sigma_v = np.asarray(sigma_v, dtype=float)
    R_d = np.asarray(R_d, dtype=float)

    sigma_safe = np.clip(sigma_v, 1e-3, None)
    Rd_safe = np.clip(R_d, 1e-3, None)

    F_raw = (
        params.A0
        * (params.sigma_ref / sigma_safe) ** params.a_sigma
        * (params.R_ref / Rd_safe) ** params.a_Rd
    )
    # Clamp at F_max
    if params.F_max is not None and params.F_max > 1.0:
        F_raw = 1.0 + np.minimum(F_raw - 1.0, params.F_max - 1.0)
    return F_raw


def _sigma_gate(
    sigma_v: np.ndarray,
    params: FMissingParams,
) -> np.ndarray:
    """
    σ_v gate: returns a factor in [gate_floor, 1] that damps F_missing for hot systems.

    We use a simple logistic-like form:

        x = (sigma_gate_ref / σ_v)^gamma_sigma
        G_sigma = x / (1 + x)

    - For σ_v << sigma_gate_ref: x >> 1 → G_sigma → 1 (cold disks, full effect)
    - For σ_v >> sigma_gate_ref: x << 1 → G_sigma → x (hot systems, suppressed)
    """
    sigma_v = np.asarray(sigma_v, dtype=float)

    if not params.use_sigma_gate or params.gamma_sigma <= 0.0:
        return np.ones_like(sigma_v)

    sigma_safe = np.clip(sigma_v, 1e-3, None)
    x = (params.sigma_gate_ref / sigma_safe) ** params.gamma_sigma
    G = x / (1.0 + x)

    if params.gate_floor is not None and params.gate_floor > 0.0:
        G = np.maximum(G, params.gate_floor)

    return G


def compute_F_missing(
    sigma_v: np.ndarray,
    R_d: np.ndarray,
    params: FMissingParams,
) -> np.ndarray:
    """
    Full mass–coherence ratio F_missing(σ_v, R_d).

    Returns a *ratio*:
        F_missing = 1  → no extra effect
        F_missing > 1  → more enhancement demanded by mass–coherence arguments.

    Implementation:
        F_raw = mass–coherence power law (with F_max clamp)
        G_sigma = σ_v gate in [gate_floor, 1]
        F_eff = 1 + (F_raw - 1) * G_sigma
    """
    F_raw = _base_F_missing(sigma_v, R_d, params)
    G_sigma = _sigma_gate(sigma_v, params)
    return 1.0 + (F_raw - 1.0) * G_sigma


def predict_F_missing(
    galaxy_props: dict,
    params: FMissingParams | None = None,
) -> float:
    """
    Predict F_missing = K_total / K_rough for a single galaxy.

    This is a convenience wrapper around compute_F_missing for single values.

    Parameters
    ----------
    galaxy_props : dict
        Dictionary with 'sigma_v' (km/s) and 'R_d' or 'R_disk' (kpc)
    params : FMissingParams, optional
        Model parameters. If None, uses defaults.

    Returns
    -------
    float
        F_missing ratio
    """
    if params is None:
        params = FMissingParams()

    # Extract galaxy properties
    sigma_v = float(galaxy_props.get("sigma_v", galaxy_props.get("sigma_velocity", 20.0)))
    R_d = float(galaxy_props.get("R_d", galaxy_props.get("R_disk", 5.0)))

    # Compute F_missing
    F_array = compute_F_missing(
        sigma_v=np.array([sigma_v]),
        R_d=np.array([R_d]),
        params=params,
    )
    return float(F_array[0])
