"""Distance-scaling utilities for photon-propagation hypotheses."""

from __future__ import annotations

import numpy as np

KPC_M = 3.085677581491367e19


def baryonic_speed(
    gas_km_s,
    disk_unit_ml_km_s,
    bulge_unit_ml_km_s,
    *,
    disk_mass_to_light: float,
    bulge_mass_to_light: float,
) -> np.ndarray:
    """Combine signed gas and stellar circular-speed contributions."""
    gas = np.asarray(gas_km_s, dtype=float)
    disk = np.asarray(disk_unit_ml_km_s, dtype=float)
    bulge = np.asarray(bulge_unit_ml_km_s, dtype=float)
    squared = (
        np.sign(gas) * np.square(gas)
        + disk_mass_to_light * np.square(disk)
        + bulge_mass_to_light * np.square(bulge)
    )
    return np.sqrt(np.maximum(squared, 0.0))


def rar_speed(
    radius_kpc,
    baryonic_speed_km_s,
    *,
    acceleration_scale_m_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-RAR speed and baryonic acceleration."""
    radius = np.asarray(radius_kpc, dtype=float)
    speed = np.asarray(baryonic_speed_km_s, dtype=float)
    acceleration = np.square(speed * 1000.0) / (radius * KPC_M)
    root = np.sqrt(np.maximum(acceleration / acceleration_scale_m_s2, 0.0))
    denominator = -np.expm1(-root)
    boost = np.divide(
        1.0,
        denominator,
        out=np.full_like(denominator, np.inf),
        where=denominator > 0.0,
    )
    predicted = np.zeros_like(speed)
    valid = speed > 0.0
    predicted[valid] = speed[valid] * np.sqrt(boost[valid])
    return predicted, acceleration


def path_feature(distance_mpc, *, kind: str, scale: float | None = None) -> np.ndarray:
    """Return a propagation-distance feature."""
    distance = np.asarray(distance_mpc, dtype=float)
    if np.any(distance <= 0.0):
        raise ValueError("distance must be positive")
    if kind == "log":
        return np.log(distance / 10.0)
    if kind == "power":
        if scale is None:
            raise ValueError("power feature requires an exponent")
        return np.power(distance / 10.0, scale) - 1.0
    if kind == "saturating":
        if scale is None or scale <= 0.0:
            raise ValueError("saturating feature requires a positive length")
        return 1.0 - np.exp(-distance / scale)
    raise ValueError(f"unknown path feature: {kind}")


def weighted_fit(feature, residual, sigma) -> tuple[np.ndarray, np.ndarray]:
    """Fit an intercept and one distance feature by weighted least squares."""
    x = np.asarray(feature, dtype=float)
    y = np.asarray(residual, dtype=float)
    error = np.asarray(sigma, dtype=float)
    design = np.column_stack((np.ones(len(x)), x))
    whitened = design / error[:, np.newaxis]
    normal = whitened.T @ whitened
    covariance = np.linalg.inv(normal)
    values = covariance @ whitened.T @ (y / error)
    return values, covariance
