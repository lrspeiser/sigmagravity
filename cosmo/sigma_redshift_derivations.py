"""
sigma_redshift_derivations.py
-----------------------------
Four alternative *non-expanding* Σ-Gravity redshift derivations, each producing
the transport law d ln nu / dl = - alpha0 * C(l) with a distinct physical meaning.

Derivations / models implemented:
  A) Weyl-integrable (non-metricity) model
  B) Effective-medium (eikonal, adiabatic, non-dispersive) model
  C) Open quantum system (Lindblad energy-drift) model
  D) Semiclassical path-sum (many-path stationary-phase) model

All models share a common interface and use the *same* Σ-coherence window C(R):
    C(R) = 1 - [ 1 + (R/ell0)^p ]^(-ncoh)
with optional import from your repo's redshift/geff_adapters.py (coherence_kernel).

Usage:
    from sigma_redshift_derivations import (
        SigmaKernel, WeylModel, EikonalModel, LindbladModel, PathSumModel,
        small_z_alpha0_for_H0
    )
    ker = SigmaKernel(A=1.0, ell0_kpc=200.0, p=0.75, ncoh=0.5)
    model = WeylModel(kernel=ker, alpha0_scale=1.0, H0_kms_Mpc=70.0)
    z = model.z_of_distance_Mpc(1000.0)

Author: you (generated scaffold)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

# -------- constants --------
c   = 299_792_458.0
Mpc = 3.085_677_581_491_3673e22

# -------- try to import your canonical coherence_kernel; else fallback --------
def _import_repo_coherence() -> Optional[Callable[[np.ndarray, float, float, float], np.ndarray]]:
    try:
        # preferred: from your repo
        from redshift.geff_adapters import coherence_kernel as repo_C
        return repo_C  # signature: C(R, ell0, p, ncoh) (dimensionless, no amplitude)
    except Exception:
        return None

_repo_C = _import_repo_coherence()

def burrXII_coherence(R: np.ndarray, ell0: float, p: float, ncoh: float) -> np.ndarray:
    x = (np.asarray(R, float) / float(ell0))**float(p)
    return 1.0 - (1.0 + x)**(-float(ncoh))

def coherence_C(R: np.ndarray, ell0: float, p: float, ncoh: float) -> np.ndarray:
    if _repo_C is not None:
        return _repo_C(R, ell0, p, ncoh)
    return burrXII_coherence(R, ell0, p, ncoh)

# -------- kernel wrapper (includes amplitude A; uses SI units) --------
@dataclass
class SigmaKernel:
    A: float = 1.0
    ell0_kpc: float = 200.0
    p: float = 0.75
    ncoh: float = 0.5
    metric: str = "spherical"  # 'spherical' => R=l ; 'cylindrical' => R=sqrt(x^2+y^2)
    def SI(self) -> Dict[str, Any]:
        return dict(A=self.A, ell0=self.ell0_kpc * (Mpc/1000.0), p=self.p, ncoh=self.ncoh)

    def C_along_line(self, l: np.ndarray) -> np.ndarray:
        """Default LOS: R=l (spherical). Override with metric choice if you pass (x,y,z)."""
        pars = self.SI()
        if self.metric == "spherical":
            R = l
        else:
            R = l  # placeholder; user can supply custom R(l)
        return coherence_C(R, pars["ell0"], pars["p"], pars["ncoh"])

# -------- shared helpers --------
def integrate_alpha0_C(L: float, kernel: SigmaKernel, alpha0_per_m: float, n_steps: int = 20000) -> float:
    l = np.linspace(0.0, L, int(n_steps)+1)
    C = kernel.C_along_line(l)
    I = np.trapz(C, l)
    return np.expm1(alpha0_per_m * I)

def small_z_alpha0_for_H0(H0_kms_Mpc: float) -> float:
    """alpha0 that reproduces small-z slope z ≈ (H0/c) D when C→1"""
    H0 = (H0_kms_Mpc * 1000.0) / Mpc  # s^-1
    return H0 / c  # [1/m]

# -------- base model --------
@dataclass
class BaseSigmaRedshift:
    kernel: SigmaKernel
    H0_kms_Mpc: float = 70.0
    alpha0_scale: float = 1.0

    def alpha0_per_m(self) -> float:
        return self.alpha0_scale * small_z_alpha0_for_H0(self.H0_kms_Mpc)

    def z_of_distance_Mpc(self, D_Mpc: float, n_steps: int = 20000) -> float:
        L = D_Mpc * Mpc
        return integrate_alpha0_C(L, self.kernel, self.alpha0_per_m(), n_steps=n_steps)

    def time_dilation(self, z: float) -> float:
        """Predicted SN time-stretch factor; model-specific override if needed."""
        return 1.0 + z

    def tolman_dimming(self, z: float) -> float:
        """Surface-brightness dimming factor: I_obs/I_em."""
        return 1.0 / (1.0 + z)**4

# -------- A) Weyl-integrable model --------
@dataclass
class WeylModel(BaseSigmaRedshift):
    """
    Q_mu k^mu = alpha0 * C => d ln nu / dl = - alpha0 * C
    SN time dilation and Tolman dimming follow from Weyl rescaling.
    """
    def time_dilation(self, z: float) -> float:
        return 1.0 + z

# -------- B) Eikonal effective-medium model --------
@dataclass
class EikonalModel(BaseSigmaRedshift):
    """
    n(x,t) = 1 + eps S,   d ln nu / dl = -(1/c) d ln n / dt ≈ - alpha0 C
    Adiabatic + non-dispersive => conserve I_nu/nu^3.
    """
    eps: float = 1.0
    def time_dilation(self, z: float) -> float:
        return 1.0 + z

# -------- C) Open quantum system (Lindblad) --------
@dataclass
class LindbladModel(BaseSigmaRedshift):
    """
    Energy relaxation rate Gamma_E = gammaE * C,  alpha0 = Gamma_E / v_g.
    Also allows phase diffusion Gamma_phi (line broadening, no shift).
    """
    gammaE_scale: float = 1.0  # scales alpha0_per_m internally
    gammaPhi_scale: float = 0.0

    def alpha0_per_m(self) -> float:
        # treat gammaE_scale * (H0/c) as effective alpha0
        return self.gammaE_scale * small_z_alpha0_for_H0(self.H0_kms_Mpc)

    def line_broadening_sigma(self, D_Mpc: float, n_steps: int = 20000) -> float:
        # crude: sigma_phase ~ sqrt(∫ Gamma_phi dl); here Gamma_phi ∝ C
        L = D_Mpc * Mpc
        l = np.linspace(0.0, L, int(n_steps)+1)
        C = self.kernel.C_along_line(l)
        # scale phi rate in units of alpha0 (dimensionally consistent for a prototype)
        phi_rate = self.gammaPhi_scale * small_z_alpha0_for_H0(self.H0_kms_Mpc) * c
        return np.sqrt(np.trapz(phi_rate * C, l))

    def time_dilation(self, z: float) -> float:
        # If wave-action conserved and arrival rate scales with ν, this gives 1+z.
        return 1.0 + z

# -------- D) Semiclassical path-sum model --------
@dataclass
class PathSumModel(BaseSigmaRedshift):
    """
    Effective drift from many-path stationary-phase ensemble:
        d ln nu / dl = - (beta/Lc) * C  => alpha0 = (beta/Lc)
    """
    beta: float = 1.0
    Lc_Mpc: float = 1000.0

    def alpha0_per_m(self) -> float:
        return (self.beta / (self.Lc_Mpc * Mpc))

# -------- utilities --------
def z_curve(model: BaseSigmaRedshift, D_Mpc_array: np.ndarray) -> np.ndarray:
    return np.array([model.z_of_distance_Mpc(float(D)) for D in D_Mpc_array], dtype=float)

def bundle_models(kernel: SigmaKernel, H0_kms_Mpc: float = 70.0) -> dict:
    return dict(
        weyl=WeylModel(kernel=kernel, H0_kms_Mpc=H0_kms_Mpc, alpha0_scale=1.0),
        eikonal=EikonalModel(kernel=kernel, H0_kms_Mpc=H0_kms_Mpc, alpha0_scale=1.0),
        lindblad=LindbladModel(kernel=kernel, H0_kms_Mpc=H0_kms_Mpc, gammaE_scale=1.0, gammaPhi_scale=0.0),
        pathsum=PathSumModel(kernel=kernel, H0_kms_Mpc=H0_kms_Mpc, beta=1.0, Lc_Mpc=1000.0),
    )










