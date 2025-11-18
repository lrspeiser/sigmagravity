"""
Wrapper for mass-coherence model to predict F_missing.

This provides vectorized predictions for F_missing based on
mass per coherence volume.
"""

from __future__ import annotations

import numpy as np
from mass_coherence_model import K_missing_from_mass


def predict_F_missing_mass_model(
    M_baryon_msun: np.ndarray,
    R_d_kpc: np.ndarray,
    ell0_kpc: np.ndarray,
    *,
    R_eff_factor: float = 2.2,
    K_max: float = 0.9,
    psi0: float = 1e-7,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Vectorized prediction of F_missing for a set of galaxies.

    Parameters
    ----------
    M_baryon_msun : array_like
        Total baryonic masses for each galaxy (Msun).
    R_d_kpc : array_like
        Disk scale lengths (kpc) for each galaxy.
    ell0_kpc : array_like
        Coherence scales per galaxy (kpc).
    R_eff_factor : float, optional
        Factor to convert R_d to an effective radius: R_eff = R_eff_factor * R_d.
    K_max, psi0, gamma : floats, optional
        Microphysics parameters for the mass-coherence model.

    Returns
    -------
    ndarray
        Predicted F_missing per galaxy.
    """
    M_b = np.asarray(M_baryon_msun, dtype=float)
    R_d = np.asarray(R_d_kpc, dtype=float)
    ell0 = np.asarray(ell0_kpc, dtype=float)

    R_eff = R_eff_factor * R_d
    F_pred = np.empty_like(M_b, dtype=float)

    for i in range(M_b.size):
        F_pred[i] = K_missing_from_mass(
            M_baryon_msun=M_b[i],
            R_eff_kpc=R_eff[i],
            ell0_kpc=ell0[i],
            K_max=K_max,
            psi0=psi0,
            gamma=gamma,
        )

    return F_pred


def predict_F_missing_velocity_model(
    v_flat_kms: np.ndarray,
    *,
    F_max: float = 10.0,
    psi0: float = 0.1,
    delta: float = 1.0,
    v_ref_kms: float = 200.0,
) -> np.ndarray:
    """
    Alternative: predict F_missing from circular velocity only.

    Parameters
    ----------
    v_flat_kms : array_like
        Characteristic circular speeds (km/s) for each galaxy.
    F_max, psi0, delta, v_ref_kms : floats, optional
        Model parameters.

    Returns
    -------
    ndarray
        Predicted F_missing per galaxy.
    """
    from mass_coherence_model import K_missing_from_velocity

    v_flat = np.asarray(v_flat_kms, dtype=float)
    F_pred = np.empty_like(v_flat, dtype=float)

    for i in range(v_flat.size):
        F_pred[i] = K_missing_from_velocity(
            v_flat_kms=v_flat[i],
            F_max=F_max,
            psi0=psi0,
            delta=delta,
            v_ref_kms=v_ref_kms,
        )

    return F_pred

