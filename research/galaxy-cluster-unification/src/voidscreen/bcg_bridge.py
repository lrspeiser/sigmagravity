from __future__ import annotations

import numpy as np
from scipy.special import gammainc, gammaincinv

from .data import KPC_M
from .unified import G_SI, M_SUN_KG

ARCSEC_PER_RADIAN = 206_264.80624709636


def physical_radius_kpc(distance_mpc, radius_arcsec) -> np.ndarray:
    distance = np.asarray(distance_mpc, dtype=float)
    radius = np.asarray(radius_arcsec, dtype=float)
    if np.any(distance <= 0.0) or np.any(radius <= 0.0):
        raise ValueError("distance and angular radius must be positive")
    return distance * 1000.0 * radius / ARCSEC_PER_RADIAN


def acceleration_from_log_mass(log10_mass_msun, radius_kpc) -> np.ndarray:
    log_mass = np.asarray(log10_mass_msun, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    if np.any(radius <= 0.0):
        raise ValueError("radius must be positive")
    return G_SI * np.power(10.0, log_mass) * M_SUN_KG / np.square(radius * KPC_M)


def sersic_projected_fraction(radius_over_re, sersic_n) -> np.ndarray:
    ratio = np.asarray(radius_over_re, dtype=float)
    index = np.asarray(sersic_n, dtype=float)
    if np.any(ratio <= 0.0) or np.any(index <= 0.0):
        raise ValueError("radius ratio and Sersic index must be positive")
    b_n = gammaincinv(2.0 * index, 0.5)
    return gammainc(2.0 * index, b_n * np.power(ratio, 1.0 / index))


def nsa_sersic_log_mass_within_radius(
    log10_mass_h_minus_2_msun,
    radius_kpc,
    sersic_re_kpc,
    sersic_n,
    *,
    hubble_h: float = 0.677,
) -> np.ndarray:
    if hubble_h <= 0.0:
        raise ValueError("hubble_h must be positive")
    total = np.asarray(log10_mass_h_minus_2_msun, dtype=float) - 2.0 * np.log10(
        hubble_h
    )
    fraction = sersic_projected_fraction(
        np.asarray(radius_kpc, dtype=float) / np.asarray(sersic_re_kpc, dtype=float),
        sersic_n,
    )
    return total + np.log10(fraction)


def mfl_log_acceleration_at_radius(
    log10_mass_re_msun,
    re_kpc,
    target_radius_kpc,
    density_slope,
) -> np.ndarray:
    re = np.asarray(re_kpc, dtype=float)
    target = np.asarray(target_radius_kpc, dtype=float)
    slope = np.asarray(density_slope, dtype=float)
    acceleration_re = acceleration_from_log_mass(log10_mass_re_msun, re)
    acceleration = acceleration_re * np.power(target / re, 1.0 + slope)
    return np.log10(acceleration)


def calibrate_log_offset(raw_log_values, reference_log_values) -> dict[str, float | int]:
    raw = np.asarray(raw_log_values, dtype=float)
    reference = np.asarray(reference_log_values, dtype=float)
    valid = np.isfinite(raw) & np.isfinite(reference)
    if not np.any(valid):
        raise ValueError("calibration requires at least one finite pair")
    raw = raw[valid]
    reference = reference[valid]
    offset = float(np.median(reference - raw))
    residual = raw + offset - reference
    return {
        "systems": len(residual),
        "offset_dex": offset,
        "mean_residual_dex": float(np.mean(residual)),
        "rms_residual_dex": float(np.sqrt(np.mean(np.square(residual)))),
        "median_abs_residual_dex": float(np.median(np.abs(residual))),
        "correlation": float(np.corrcoef(raw, reference)[0, 1]),
    }
