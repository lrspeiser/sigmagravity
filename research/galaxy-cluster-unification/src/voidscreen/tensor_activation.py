"""Baryon-only activation fields for the projected tensor-AQUAL candidate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voidscreen.geometric_transport import (
    PathGeometry,
    ThinSheetField,
    component_angle_mismatch,
    high_acceleration_screen,
    streamline_incoherence,
    thin_sheet_newtonian_field,
)
from voidscreen.tensor_aqual import simple_mu


@dataclass(frozen=True)
class TensorActivation2D:
    """Exact coefficient maps used by the P0659 constitutive tensor."""

    sigma: np.ndarray
    transverse_mismatch: np.ndarray
    survival: np.ndarray
    high_acceleration_screen: np.ndarray
    transport_direction_x: np.ndarray
    transport_direction_y: np.ndarray
    mu_newtonian_proxy: np.ndarray
    minimum_eigenvalue_proxy: np.ndarray
    maximum_eigenvalue_proxy: np.ndarray
    total_field: ThinSheetField
    path: PathGeometry


def _unit_component_difference(
    stars: ThinSheetField,
    gas: ThinSheetField,
    fallback_x: np.ndarray,
    fallback_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tiny = np.finfo(float).tiny
    star_norm = np.maximum(stars.magnitude_m_s2, tiny)
    gas_norm = np.maximum(gas.magnitude_m_s2, tiny)
    difference_x = gas.acceleration_x_m_s2 / gas_norm - stars.acceleration_x_m_s2 / star_norm
    difference_y = gas.acceleration_y_m_s2 / gas_norm - stars.acceleration_y_m_s2 / star_norm
    norm = np.hypot(difference_x, difference_y)
    active = norm > 1e-12
    direction_x = np.where(active, difference_x / np.maximum(norm, 1e-12), fallback_x)
    direction_y = np.where(active, difference_y / np.maximum(norm, 1e-12), fallback_y)
    fallback_norm = np.hypot(direction_x, direction_y)
    valid = fallback_norm > 1e-12
    direction_x = np.where(valid, direction_x / np.maximum(fallback_norm, 1e-12), 1.0)
    direction_y = np.where(valid, direction_y / np.maximum(fallback_norm, 1e-12), 0.0)
    return direction_x, direction_y


def exact_tensor_activation(
    stellar_surface_density_msun_kpc2: np.ndarray,
    gas_surface_density_msun_kpc2: np.ndarray,
    cell_kpc: float,
    *,
    a0_m_s2: float = 1.2e-10,
    coherence_length_kpc: float = 10.0,
    coherence_power: float = 1.0,
    mu_floor: float = 1e-6,
) -> TensorActivation2D:
    """Construct ``sigma`` and its direction from stellar and gas maps.

    This routine does not read any velocity or lensing target.  Its AQUAL
    eigenvalues use the baryon-only Newtonian field only as a conditioning
    proxy; the nonlinear solver recomputes ``mu`` from its solved potential.
    """

    stars = np.maximum(np.asarray(stellar_surface_density_msun_kpc2, dtype=float), 0.0)
    gas = np.maximum(np.asarray(gas_surface_density_msun_kpc2, dtype=float), 0.0)
    if stars.ndim != 2 or stars.shape != gas.shape or min(stars.shape) < 9:
        raise ValueError("stellar and gas inputs must be matching 2D maps")
    if not np.all(np.isfinite(stars)) or not np.all(np.isfinite(gas)):
        raise ValueError("surface-density maps must be finite")
    if cell_kpc <= 0.0 or a0_m_s2 <= 0.0 or coherence_length_kpc <= 0.0:
        raise ValueError("physical scales must be positive")
    if coherence_power <= 0.0 or mu_floor <= 0.0:
        raise ValueError("coherence power and mu floor must be positive")

    star_field = thin_sheet_newtonian_field(stars, cell_kpc)
    gas_field = thin_sheet_newtonian_field(gas, cell_kpc)
    total_field = thin_sheet_newtonian_field(stars + gas, cell_kpc)
    path = streamline_incoherence(total_field, cell_kpc)
    mismatch = component_angle_mismatch(
        star_field,
        gas_field,
        mode="transverse_tensor_mix",
    )
    length_ratio = np.maximum(path.trace_length_kpc / float(coherence_length_kpc), 0.0)
    survival = 1.0 - np.exp(-np.power(length_ratio, float(coherence_power)))
    screen = high_acceleration_screen(total_field.magnitude_m_s2, a0_m_s2)
    sigma = np.clip(screen * mismatch * survival, 0.0, 1.0)
    direction_x, direction_y = _unit_component_difference(
        star_field,
        gas_field,
        path.mean_direction_x,
        path.mean_direction_y,
    )
    mu_proxy = np.maximum(
        simple_mu(total_field.magnitude_m_s2 / float(a0_m_s2)),
        float(mu_floor),
    )
    minimum_eigenvalue = mu_proxy * (1.0 - sigma)
    return TensorActivation2D(
        sigma=sigma,
        transverse_mismatch=mismatch,
        survival=survival,
        high_acceleration_screen=screen,
        transport_direction_x=direction_x,
        transport_direction_y=direction_y,
        mu_newtonian_proxy=mu_proxy,
        minimum_eigenvalue_proxy=minimum_eigenvalue,
        maximum_eigenvalue_proxy=mu_proxy,
        total_field=total_field,
        path=path,
    )


def constitutive_tensor_components(
    sigma: np.ndarray,
    transport_direction_x: np.ndarray,
    transport_direction_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the unique components of ``I-sigma h h`` for invariance audits."""

    anisotropy = np.asarray(sigma, dtype=float)
    h_x = np.asarray(transport_direction_x, dtype=float)
    h_y = np.asarray(transport_direction_y, dtype=float)
    if not (anisotropy.shape == h_x.shape == h_y.shape):
        raise ValueError("tensor coefficient maps must have matching shapes")
    return (
        1.0 - anisotropy * h_x * h_x,
        -anisotropy * h_x * h_y,
        1.0 - anisotropy * h_y * h_y,
    )
