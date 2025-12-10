"""
Mass-per-coherence-volume model for F_missing.

Theory: F_missing comes from resonant gravitational cavity modes.
The number of modes depends on potential depth per coherence volume:
    Ψ = Φ_coh/c² = (G M_b / c²) * (ℓ₀² / R_eff³)

Then: K_missing = K_max * [1 - exp(-(Ψ/Ψ₀)^γ)]
"""

from __future__ import annotations

import numpy as np

# Physical constants
G_MSUN_KPC_KM2_S2 = 4.30091e-6  # G in (kpc / Msun) (km/s)^2
C_LIGHT_KMS = 299792.458  # Speed of light in km/s


def dimensionless_potential_depth(
    M_baryon_msun: float,
    R_eff_kpc: float,
    ell0_kpc: float,
) -> float:
    """
    Compute Ψ = Φ_coh/c² for one system.

    We approximate:
        M_coh ~ M_b * (ell0 / R_eff)^3
        Φ_coh ~ G M_coh / ell0
              ~ G M_b * ell0^2 / R_eff^3

    Parameters
    ----------
    M_baryon_msun : float
        Total baryonic mass of the system in Msun.
    R_eff_kpc : float
        Effective size of the system in kpc (e.g. 2.2 R_d for disks).
    ell0_kpc : float
        Coherence scale (Burr-XII ℓ0 or time-coherence ℓ_coh) in kpc.

    Returns
    -------
    float
        Dimensionless potential depth Ψ = Φ_coh / c².
    """
    M_b = float(M_baryon_msun)
    R_eff = float(max(R_eff_kpc, 1e-4))
    ell0 = float(max(ell0_kpc, 1e-4))

    phi_coh = G_MSUN_KPC_KM2_S2 * M_b * (ell0**2) / (R_eff**3)  # (km/s)^2
    psi = phi_coh / (C_LIGHT_KMS**2)
    return psi


def K_missing_from_mass(
    M_baryon_msun: float,
    R_eff_kpc: float,
    ell0_kpc: float,
    *,
    K_max: float = 0.9,
    psi0: float = 1e-7,
    gamma: float = 1.0,
) -> float:
    """
    Mass-per-coherence-volume model for the missing enhancement.

    K_missing = K_max * (1 - exp(-(Ψ/Ψ0)^gamma)).

    Parameters
    ----------
    M_baryon_msun : float
        Total baryonic mass in Msun.
    R_eff_kpc : float
        Effective radius in kpc (e.g. 2.2 * R_d for disks).
    ell0_kpc : float
        Coherence scale in kpc (use your fitted ℓ0 or time-coherence ℓ_coh).
    K_max : float, optional
        Saturation amplitude for this channel.
    psi0 : float, optional
        Depth scale Ψ0 where the effect becomes order-unity.
    gamma : float, optional
        Sharpness of the turn-on.

    Returns
    -------
    float
        Dimensionless K_missing for this system.
    """
    psi = dimensionless_potential_depth(M_baryon_msun, R_eff_kpc, ell0_kpc)
    x = (psi / max(psi0, 1e-12)) ** gamma
    return float(K_max * (1.0 - np.exp(-x)))


def K_total_system(
    K_rough: float,
    M_baryon_msun: float,
    R_eff_kpc: float,
    ell0_kpc: float,
    *,
    K_max: float = 0.9,
    psi0: float = 1e-7,
    gamma: float = 1.0,
) -> float:
    """
    Combine roughness and mass-coherence contributions at the system level.

    K_total = K_rough + K_missing

    Parameters
    ----------
    K_rough : float
        Roughness enhancement from time-coherence (K_rough(Ξ)).
    M_baryon_msun : float
        Total baryonic mass in Msun.
    R_eff_kpc : float
        Effective radius in kpc.
    ell0_kpc : float
        Coherence scale in kpc.
    K_max, psi0, gamma : floats, optional
        Parameters for mass-coherence model.

    Returns
    -------
    float
        Total system-level enhancement K_total.
    """
    K_missing = K_missing_from_mass(
        M_baryon_msun, R_eff_kpc, ell0_kpc, K_max=K_max, psi0=psi0, gamma=gamma
    )
    return K_rough + K_missing


def dimensionless_potential_from_velocity(
    v_flat_kms: float,
    v_ref_kms: float = 200.0,
) -> float:
    """
    Alternative: compute Ψ from circular velocity instead of mass.

    Ψ_pot = (v_flat / v_ref)^2

    Parameters
    ----------
    v_flat_kms : float
        Characteristic circular speed (e.g. median V_obs in flat part) in km/s.
    v_ref_kms : float, optional
        Reference velocity scale (default 200 km/s).

    Returns
    -------
    float
        Dimensionless potential depth Ψ_pot.
    """
    return (v_flat_kms / v_ref_kms) ** 2


def K_missing_from_velocity(
    v_flat_kms: float,
    *,
    F_max: float = 10.0,
    psi0: float = 0.1,
    delta: float = 1.0,
    v_ref_kms: float = 200.0,
) -> float:
    """
    Pure potential-depth model using circular velocity.

    F_missing = F_max * [1 - exp(-(Ψ_pot/Ψ₀)^δ)]

    Parameters
    ----------
    v_flat_kms : float
        Characteristic circular speed in km/s.
    F_max : float, optional
        Maximum F_missing value.
    psi0 : float, optional
        Depth scale where effect turns on.
    delta : float, optional
        Sharpness of turn-on.
    v_ref_kms : float, optional
        Reference velocity scale.

    Returns
    -------
    float
        Predicted F_missing.
    """
    psi_pot = dimensionless_potential_from_velocity(v_flat_kms, v_ref_kms=v_ref_kms)
    x = (psi_pot / max(psi0, 1e-12)) ** delta
    return float(F_max * (1.0 - np.exp(-x)))

