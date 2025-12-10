"""
Unified kernel combining roughness and mass-coherence.

K_total(R) = K_rough(Ξ_mean) × C(R/ℓ₀) × [1 + f_amp × extra_amp × (F_missing - 1)]

Where:
- K_rough: System-level roughness from time-coherence
- C(R/ℓ₀): Burr-XII radial shape
- F_missing: Multiplicative ratio (not additive!) with σ_v gating
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Optional

import numpy as np
from burr_xii_shape import burr_xii_shape
from f_missing_mass_model import FMissingParams, compute_F_missing
from system_level_k import system_level_K, compute_Xi_mean
from coherence_time_kernel import (
    compute_tau_geom,
    compute_tau_noise,
    compute_tau_coh,
    compute_exposure_factor,
)


@dataclass
class UnifiedKernelParams:
    """
    Hyperparameters for the unified kernel K_total.

    Roughness piece:
        K_rough = K_rough(R, Xi; time-coherence params)

    Mass–coherence piece:
        F_missing = F_missing(σ_v, R_d; FMissingParams)

    Combined:
        K_total = K_rough * [1 + f_amp * extra_amp * (F_missing - 1)] * M_gate
    """
    # ---- roughness controls ----
    use_rough: bool = True   # turn off to test F_missing alone
    A0: float = 0.774
    gamma_rough: float = 0.1
    p: float = 0.757
    n_coh: float = 0.5
    ell0_kpc: float = 5.0
    tau_geom_method: str = "tidal"
    alpha_geom: float = 1.0
    beta_sigma: float = 1.5

    # ---- F_missing controls ----
    use_f_missing: bool = True
    f_amp: float = 1.0       # global knob for the 90% piece
    extra_amp: float = 0.25  # "lever arm" multiplier in unified kernel

    f_missing: FMissingParams = None  # Will be initialized to default if None

    # ---- morphology gate ----
    use_morph_gate: bool = False  # Default off for now
    morph_floor: float = 0.2  # min morphology factor

    def __post_init__(self):
        if self.f_missing is None:
            self.f_missing = FMissingParams()

    @classmethod
    def from_json(cls, path: str | Path) -> "UnifiedKernelParams":
        """Load parameters from JSON file."""
        raw = json.loads(Path(path).read_text())
        # Handle nested FMissingParams explicitly
        f_raw = raw.pop("f_missing", {})
        f_params = FMissingParams(**f_raw)
        return cls(f_missing=f_params, **raw)

    def to_json(self, path: str | Path) -> None:
        """Save parameters to JSON file."""
        raw = asdict(self)
        raw["f_missing"] = asdict(self.f_missing)
        Path(path).write_text(json.dumps(raw, indent=2))


def compute_unified_kernel(
    R_kpc: np.ndarray,
    g_bar_kms2: np.ndarray,
    sigma_v_kms: float,
    rho_bar_msun_pc3: np.ndarray,
    galaxy_props: dict,
    *,
    params: UnifiedKernelParams | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Compute unified kernel K_total(R) = K_rough × C(R) × [1 + f_amp × extra_amp × (F_missing - 1)].

    Parameters
    ----------
    R_kpc : array_like
        Radii in kpc.
    g_bar_kms2 : array_like
        Baryonic acceleration in km/s^2.
    sigma_v_kms : float
        Velocity dispersion in km/s.
    rho_bar_msun_pc3 : array_like
        Baryonic density in Msun/pc^3.
    galaxy_props : dict
        Dictionary with galaxy properties (sigma_v, R_disk, etc.)
    params : UnifiedKernelParams, optional
        Kernel parameters. If None, uses defaults.

    Returns
    -------
    K_total : ndarray
        Total enhancement kernel K_total(R).
    info : dict
        Dictionary with intermediate values (K_rough, F_missing, Xi_mean, etc.).
    """
    if params is None:
        params = UnifiedKernelParams()

    R = np.asarray(R_kpc, dtype=float)
    g_bar = np.asarray(g_bar_kms2, dtype=float)
    rho_bar = np.asarray(rho_bar_msun_pc3, dtype=float)

    # 1. Compute roughness component
    if params.use_rough:
        tau_geom = compute_tau_geom(
            R,
            g_bar,
            rho_bar,
            method=params.tau_geom_method,
            alpha_geom=params.alpha_geom,
        )
        tau_noise = compute_tau_noise(
            R,
            sigma_v_kms,  # scalar
            method="galaxy",
            beta_sigma=params.beta_sigma,
        )
        tau_coh = compute_tau_coh(tau_geom, tau_noise)

        # Compute mean exposure factor
        Xi = compute_exposure_factor(R, g_bar, tau_coh)
        Xi_mean = float(np.mean(Xi)) if len(Xi) > 0 else 0.0

        # System-level roughness
        K_rough = system_level_K(Xi_mean, A0=params.A0, gamma=params.gamma_rough)

        # Radial shape (unit amplitude)
        C_R = burr_xii_shape(R, params.ell0_kpc, p=params.p, n_coh=params.n_coh)

        # Roughness contribution (radial)
        K_rough_radial = K_rough * C_R
    else:
        K_rough_radial = np.zeros_like(R)
        Xi_mean = 0.0
        K_rough = 0.0
        C_R = np.zeros_like(R)

    # 2. Compute F_missing ratio (multiplicative, not additive!)
    if params.use_f_missing:
        # Ensure galaxy_props has required fields
        galaxy_props_complete = {
            "sigma_v": sigma_v_kms,
            "R_disk": galaxy_props.get("R_disk", galaxy_props.get("R_d", 5.0)),
            **galaxy_props,
        }

        F_missing = compute_F_missing(
            sigma_v=np.array([sigma_v_kms]),
            R_d=np.array([galaxy_props_complete["R_disk"]]),
            params=params.f_missing,
        )[0]

        # Apply F_missing as multiplicative scaling
        # f_amp=0 → K_total = K_rough
        # f_amp=1 → K_total = K_rough * [1 + extra_amp * (F_missing - 1)]
        raw_scale = F_missing - 1.0
        scale = 1.0 + params.extra_amp * params.f_amp * raw_scale
    else:
        F_missing = 1.0
        scale = 1.0

    K_total = K_rough_radial * scale

    # 3. Apply morphology gate if requested
    if params.use_morph_gate:
        from morphology_gate import morphology_gate
        morph_class = galaxy_props.get("morphology", galaxy_props.get("morph_class", None))
        if morph_class is not None:
            morph_meta = {"morphology": morph_class}
            M = morphology_gate(morph_meta)
            M = max(M, params.morph_floor)  # Apply floor
            K_total = K_total * M
        else:
            M = 1.0
    else:
        M = 1.0

    info = {
        "K_rough": float(K_rough),
        "F_missing": float(F_missing),
        "scale": float(scale),
        "Xi_mean": float(Xi_mean),
        "C_mean": float(np.mean(C_R)) if params.use_rough else 0.0,
        "K_rough_radial_mean": float(np.mean(K_rough_radial)) if params.use_rough else 0.0,
        "K_total_mean": float(np.mean(K_total)),
        "morph_factor": float(M),
    }

    return K_total, info
