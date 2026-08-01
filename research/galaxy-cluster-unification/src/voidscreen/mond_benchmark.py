"""Published SPARC RAR/MOND benchmark utilities.

The primary benchmark is Li et al. (2018), A&A 615 A3.  Their algebraic
relation is an empirical RAR and a MOND circular-orbit interpolation law.  It
is not a numerical solution of a particular modified-gravity field equation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import RotationCurve
from .synthetic_universe import KPC_M

A0_PUBLISHED_M_S2 = 1.2e-10


@dataclass(frozen=True)
class Li2018Fit:
    sparc_id: int
    galaxy: str
    log_luminosity_lsun: float
    disk_mass_to_light: float
    disk_mass_to_light_error: float
    bulge_mass_to_light: float | None
    bulge_mass_to_light_error: float | None
    distance_mpc: float
    distance_error_mpc: float
    distance_ratio: float
    inclination_deg: float
    inclination_error_deg: float
    inclination_ratio: float
    reduced_chi_square: float


_TABLE_ROW = re.compile(
    r"^\s*(\d{3})\s*&\s*([^&]+?)\s*&\s*([\d.]+)\s*&\s*"
    r"([\d.]+)\s*\$\\pm\$\s*([\d.]+)\s*&\s*"
    r"(?:(\\dots)|([\d.]+)\s*\$\\pm\$\s*([\d.]+))\s*&\s*"
    r"([\d.]+)\s*\$\\pm\$\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*"
    r"([\d.]+)\s*\$\\pm\$\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)"
)


def parse_li2018_table(path: Path) -> dict[str, Li2018Fit]:
    fits: dict[str, Li2018Fit] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            match = _TABLE_ROW.search(line)
            if match is None:
                continue
            values = match.groups()
            galaxy = values[1].strip()
            no_bulge = values[5] is not None
            fit = Li2018Fit(
                sparc_id=int(values[0]),
                galaxy=galaxy,
                log_luminosity_lsun=float(values[2]),
                disk_mass_to_light=float(values[3]),
                disk_mass_to_light_error=float(values[4]),
                bulge_mass_to_light=None if no_bulge else float(values[6]),
                bulge_mass_to_light_error=None if no_bulge else float(values[7]),
                distance_mpc=float(values[8]),
                distance_error_mpc=float(values[9]),
                distance_ratio=float(values[10]),
                inclination_deg=float(values[11]),
                inclination_error_deg=float(values[12]),
                inclination_ratio=float(values[13]),
                reduced_chi_square=float(values[14]),
            )
            if galaxy in fits:
                raise ValueError(f"Duplicate Li et al. table row for {galaxy}")
            fits[galaxy] = fit
    if len(fits) != 175:
        raise ValueError(f"Expected 175 Li et al. fits, found {len(fits)}")
    if {fit.sparc_id for fit in fits.values()} != set(range(1, 176)):
        raise ValueError("Li et al. SPARC IDs are incomplete")
    return fits


def li2018_rar_mond_acceleration(
    gbar_m_s2, a0_m_s2: float = A0_PUBLISHED_M_S2
) -> np.ndarray:
    """Equation 3 of Li et al. (2018)."""
    gbar = np.asarray(gbar_m_s2, dtype=np.float64)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("baryonic acceleration must be finite and positive")
    if not np.isfinite(a0_m_s2) or a0_m_s2 <= 0.0:
        raise ValueError("a0 must be finite and positive")
    denominator = 1.0 - np.exp(-np.sqrt(gbar / a0_m_s2))
    return gbar / np.maximum(denominator, 1.0e-15)


def simple_mond_acceleration(
    gbar_m_s2, a0_m_s2: float = A0_PUBLISHED_M_S2
) -> np.ndarray:
    """Algebraic MOND for mu(x)=x/(1+x)."""
    gbar = np.asarray(gbar_m_s2, dtype=np.float64)
    return 0.5 * (gbar + np.sqrt(gbar**2 + 4.0 * a0_m_s2 * gbar))


def standard_mond_acceleration(
    gbar_m_s2, a0_m_s2: float = A0_PUBLISHED_M_S2
) -> np.ndarray:
    """Algebraic MOND for mu(x)=x/sqrt(1+x^2)."""
    gbar = np.asarray(gbar_m_s2, dtype=np.float64)
    y = gbar / a0_m_s2
    nu = np.sqrt(0.5 + 0.5 * np.sqrt(1.0 + 4.0 / np.square(y)))
    return nu * gbar


def predict_acceleration(law: str, gbar_m_s2) -> np.ndarray:
    gbar = np.asarray(gbar_m_s2, dtype=np.float64)
    if law == "baryons":
        return gbar
    if law == "li2018_rar_mond":
        return li2018_rar_mond_acceleration(gbar)
    if law == "simple_mond":
        return simple_mond_acceleration(gbar)
    if law == "standard_mond":
        return standard_mond_acceleration(gbar)
    raise ValueError(f"Unknown MOND benchmark law {law!r}")


def signed_baryonic_velocity_squared(
    curve: RotationCurve,
    disk_mass_to_light: float,
    bulge_mass_to_light: float,
) -> np.ndarray:
    return (
        np.sign(curve.velocity_gas_kms) * np.square(curve.velocity_gas_kms)
        + disk_mass_to_light * np.square(curve.velocity_disk_unit_ml_kms)
        + bulge_mass_to_light * np.square(curve.velocity_bulge_unit_ml_kms)
    )


def evaluated_curve(
    curve: RotationCurve,
    *,
    disk_mass_to_light: float,
    bulge_mass_to_light: float,
    distance_mpc: float,
    inclination_deg: float,
    law: str,
) -> dict[str, np.ndarray]:
    """Apply the published distance/inclination transformations and one law."""
    if disk_mass_to_light <= 0.0 or bulge_mass_to_light < 0.0:
        raise ValueError("mass-to-light ratios are outside the physical range")
    if distance_mpc <= 0.0 or not 0.0 < inclination_deg <= 90.0:
        raise ValueError("distance or inclination is outside the physical range")
    distance_ratio = distance_mpc / curve.metadata.distance_mpc
    inclination_ratio = np.sin(np.radians(curve.metadata.inclination_deg)) / np.sin(
        np.radians(inclination_deg)
    )
    radius = curve.radius_kpc * distance_ratio
    observed_velocity = curve.velocity_observed_kms * inclination_ratio
    velocity_error = curve.velocity_error_kms * inclination_ratio
    baryonic_v2 = signed_baryonic_velocity_squared(
        curve, disk_mass_to_light, bulge_mass_to_light
    )
    gbar = baryonic_v2 * 1.0e6 / (curve.radius_kpc * KPC_M)
    gobs = np.square(observed_velocity) * 1.0e6 / (radius * KPC_M)
    gobs_error = (
        2.0 * observed_velocity * velocity_error * 1.0e6 / (radius * KPC_M)
    )
    valid = (
        np.isfinite(radius)
        & np.isfinite(gbar)
        & np.isfinite(gobs)
        & np.isfinite(gobs_error)
        & (radius > 0.0)
        & (gbar > 0.0)
        & (gobs > 0.0)
        & (gobs_error > 0.0)
    )
    predicted_acceleration = np.full(len(radius), np.nan, dtype=np.float64)
    predicted_acceleration[valid] = predict_acceleration(law, gbar[valid])
    predicted_velocity = np.sqrt(
        np.maximum(predicted_acceleration * radius * KPC_M, 0.0)
    ) / 1000.0
    return {
        "radius_kpc": radius,
        "observed_velocity_km_s": observed_velocity,
        "velocity_error_km_s": velocity_error,
        "gbar_m_s2": gbar,
        "gobs_m_s2": gobs,
        "gobs_error_m_s2": gobs_error,
        "predicted_acceleration_m_s2": predicted_acceleration,
        "predicted_velocity_km_s": predicted_velocity,
        "valid": valid,
    }


def catalog_curve(
    curve: RotationCurve,
    law: str,
    *,
    disk_mass_to_light: float = 0.5,
    bulge_mass_to_light: float = 0.7,
) -> dict[str, np.ndarray]:
    return evaluated_curve(
        curve,
        disk_mass_to_light=disk_mass_to_light,
        bulge_mass_to_light=bulge_mass_to_light,
        distance_mpc=curve.metadata.distance_mpc,
        inclination_deg=curve.metadata.inclination_deg,
        law=law,
    )


def published_fit_curve(
    curve: RotationCurve, fit: Li2018Fit, law: str = "li2018_rar_mond"
) -> dict[str, np.ndarray]:
    if curve.metadata.name != fit.galaxy:
        raise ValueError("curve and published fit refer to different galaxies")
    return evaluated_curve(
        curve,
        disk_mass_to_light=fit.disk_mass_to_light,
        bulge_mass_to_light=(
            0.0 if fit.bulge_mass_to_light is None else fit.bulge_mass_to_light
        ),
        distance_mpc=fit.distance_mpc,
        inclination_deg=fit.inclination_deg,
        law=law,
    )


def precision_mask(
    evaluation: dict[str, np.ndarray], *, fractional_error_max: float = 0.1
) -> np.ndarray:
    # The epsilon reproduces the strict comparison applied to the unrounded
    # source values: two displayed 10.0% SPARC points would otherwise enter
    # because of binary floating-point representation.
    return evaluation["valid"] & (
        evaluation["velocity_error_km_s"] + 1.0e-12
        < fractional_error_max * evaluation["observed_velocity_km_s"]
    )


def reduced_chi_square(
    evaluation: dict[str, np.ndarray], *, fitted_parameters: int
) -> float:
    valid = evaluation["valid"]
    residual = (
        evaluation["gobs_m_s2"][valid]
        - evaluation["predicted_acceleration_m_s2"][valid]
    ) / evaluation["gobs_error_m_s2"][valid]
    degrees = int(valid.sum()) - int(fitted_parameters)
    if degrees <= 0:
        return float("nan")
    return float(np.sum(np.square(residual)) / degrees)
