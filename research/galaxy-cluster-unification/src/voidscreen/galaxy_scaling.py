"""Residual-blind galaxy predictors and galaxy-dependent cage responses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import KPC_M, PackedDataset


@dataclass(frozen=True)
class GalaxyPredictors:
    names: tuple[str, ...]
    mass_proxy_1e9_msun: np.ndarray
    central_stellar_surface_density_msun_pc2: np.ndarray
    concentration_rdisk_over_reff: np.ndarray


def load_sparc_structural_predictors(
    table1_path: Path, galaxy_names: tuple[str, ...]
) -> GalaxyPredictors:
    """Load predictors from SPARC Table 1 without reading velocity columns."""
    rows: dict[str, tuple[float, float, float]] = {}
    with Path(table1_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                name = line[0:11].strip()
                luminosity_1e9_lsun = float(line[40:47])
                effective_radius_kpc = float(line[56:61])
                disk_scale_kpc = float(line[71:76])
                disk_surface_brightness = float(line[77:85])
                hi_mass_1e9_msun = float(line[86:93])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Malformed SPARC structural metadata at {table1_path}:{line_number}"
                ) from exc
            mass_proxy = 0.5 * luminosity_1e9_lsun + 1.33 * hi_mass_1e9_msun
            central_stellar_surface_density = 0.5 * disk_surface_brightness
            concentration = disk_scale_kpc / effective_radius_kpc
            rows[name] = (
                mass_proxy,
                central_stellar_surface_density,
                concentration,
            )

    missing = [name for name in galaxy_names if name not in rows]
    if missing:
        raise ValueError(f"SPARC Table 1 is missing retained galaxies: {missing[:5]}")
    values = np.asarray([rows[name] for name in galaxy_names], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Galaxy predictors must be finite and positive")
    return GalaxyPredictors(
        names=galaxy_names,
        mass_proxy_1e9_msun=values[:, 0],
        central_stellar_surface_density_msun_pc2=values[:, 1],
        concentration_rdisk_over_reff=values[:, 2],
    )


def normalize_positive_by_training_median(
    values: np.ndarray, training_galaxies: np.ndarray
) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=np.float64)
    training = np.asarray(training_galaxies, dtype=bool)
    if array.ndim != 1 or training.shape != array.shape:
        raise ValueError("Predictor and training mask must be one-dimensional and aligned")
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError("Predictor values must be finite and positive")
    median = float(np.median(array[training]))
    if not np.isfinite(median) or median <= 0.0:
        raise ValueError("Training-fold median must be finite and positive")
    return array / median, median


def local_acceleration_screened_velocity(
    packed: PackedDataset,
    baryonic_v2_km2_s2: np.ndarray,
    *,
    log10_velocity_scale_km_s: float,
    log10_gstar_m_s2: float,
    screening_power: float,
    environment_by_galaxy: np.ndarray | None = None,
    environment_exponent: float = 0.0,
) -> np.ndarray:
    """Add a flat contribution activated as the local baryonic field weakens."""
    if screening_power <= 0.0:
        raise ValueError("screening_power must be positive")
    baryonic = np.asarray(baryonic_v2_km2_s2, dtype=np.float64)
    if baryonic.shape != packed.radius_kpc.shape or np.any(baryonic <= 0.0):
        raise ValueError("baryonic velocity squared must be positive and point-aligned")
    radius_m = packed.radius_kpc * KPC_M
    gbar = baryonic * 1e6 / radius_m
    gstar = 10.0**log10_gstar_m_s2
    activation = 1.0 / (1.0 + np.power(gbar / gstar, screening_power))
    response = _environment_response(
        packed.n_galaxies, environment_by_galaxy, environment_exponent
    )
    velocity_scale = 10.0**log10_velocity_scale_km_s
    extra_v2 = velocity_scale**2 * response[packed.galaxy_index] * activation
    return np.sqrt(np.maximum(baryonic + extra_v2, 1e-12))


def catalog_scaled_screened_velocity(
    packed: PackedDataset,
    baryonic_v2_km2_s2: np.ndarray,
    *,
    mass_by_galaxy: np.ndarray,
    transition_driver_by_galaxy: np.ndarray,
    log10_velocity_scale_km_s: float,
    log10_transition_scale_lengths: float,
    mass_amplitude_exponent: float,
    transition_exponent: float,
    environment_by_galaxy: np.ndarray | None = None,
    environment_exponent: float = 0.0,
) -> np.ndarray:
    """Screen with a mass-scaled amplitude and a structural transition radius."""
    mass = _positive_galaxy_array(mass_by_galaxy, packed.n_galaxies, "mass")
    driver = _positive_galaxy_array(
        transition_driver_by_galaxy, packed.n_galaxies, "transition driver"
    )
    baryonic = np.asarray(baryonic_v2_km2_s2, dtype=np.float64)
    velocity_scale = 10.0**log10_velocity_scale_km_s
    transition_scale = 10.0**log10_transition_scale_lengths
    transition = (
        transition_scale
        * packed.disk_scale_kpc
        * np.power(driver, transition_exponent)
    )
    radius = packed.radius_kpc
    activation = radius**2 / (
        radius**2 + transition[packed.galaxy_index] ** 2
    )
    response = np.power(mass, mass_amplitude_exponent)
    response *= _environment_response(
        packed.n_galaxies, environment_by_galaxy, environment_exponent
    )
    extra_v2 = velocity_scale**2 * response[packed.galaxy_index] * activation
    return np.sqrt(np.maximum(baryonic + extra_v2, 1e-12))


def _positive_galaxy_array(values: np.ndarray, count: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,) or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(f"{label} must contain one finite positive value per galaxy")
    return array


def _environment_response(
    count: int,
    environment_by_galaxy: np.ndarray | None,
    environment_exponent: float,
) -> np.ndarray:
    if environment_by_galaxy is None:
        return np.ones(count, dtype=np.float64)
    environment = _positive_galaxy_array(
        environment_by_galaxy, count, "environment"
    )
    return np.power(environment, environment_exponent)
