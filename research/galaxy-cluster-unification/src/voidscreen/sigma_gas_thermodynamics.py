"""Physical APEC-to-gas conversions for the post-V19X3 source-state stage."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

FloatArray = NDArray[np.float64]

KPC_CM = 3.0856775814913673e21
SOLAR_MASS_G = 1.988409870698051e33
PROTON_MASS_G = 1.67262192595e-24
KEV_ERG = 1.602176634e-9


def _positive(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return array


def apec_emission_measure_cm3(
    normalization: ArrayLike,
    angular_diameter_distance_cm: ArrayLike,
    redshift: ArrayLike,
) -> FloatArray:
    """Recover ``integral n_e n_H dV`` from the XSPEC APEC normalization."""

    norm = _positive(normalization, name="normalization")
    distance = _positive(
        angular_diameter_distance_cm, name="angular_diameter_distance_cm"
    )
    z = np.asarray(redshift, dtype=float)
    if not np.all(np.isfinite(z)) or np.any(z < 0.0):
        raise ValueError("redshift must be finite and non-negative")
    return norm * 4.0 * math.pi * (distance * (1.0 + z)) ** 2 * 1.0e14


def uniform_slab_thermodynamics(
    normalization: ArrayLike,
    temperature_keV: ArrayLike,
    angular_diameter_distance_cm: ArrayLike,
    redshift: ArrayLike,
    projected_area_kpc2: ArrayLike,
    line_of_sight_depth_kpc: ArrayLike,
    *,
    electron_to_hydrogen_ratio: float = 1.2,
    mean_mass_per_electron_proton_masses: float = 1.17,
    mean_particle_mass_proton_masses: float = 0.61,
    adiabatic_index: float = 5.0 / 3.0,
) -> dict[str, FloatArray]:
    """Convert one or more APEC fits into a uniform-slab gas state.

    The depth is a measurement/deprojection nuisance and must be propagated as
    a posterior draw.  It is not a gravity parameter.
    """

    temperature = _positive(temperature_keV, name="temperature_keV")
    area = _positive(projected_area_kpc2, name="projected_area_kpc2")
    depth = _positive(line_of_sight_depth_kpc, name="line_of_sight_depth_kpc")
    for value, name in (
        (electron_to_hydrogen_ratio, "electron_to_hydrogen_ratio"),
        (
            mean_mass_per_electron_proton_masses,
            "mean_mass_per_electron_proton_masses",
        ),
        (mean_particle_mass_proton_masses, "mean_particle_mass_proton_masses"),
        (adiabatic_index, "adiabatic_index"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")

    emission_measure = apec_emission_measure_cm3(
        normalization, angular_diameter_distance_cm, redshift
    )
    area_cm2 = area * KPC_CM**2
    depth_cm = depth * KPC_CM
    emission_measure_per_area = emission_measure / area_cm2
    electron_density = np.sqrt(
        electron_to_hydrogen_ratio * emission_measure_per_area / depth_cm
    )
    surface_density_g_cm2 = (
        mean_mass_per_electron_proton_masses
        * PROTON_MASS_G
        * electron_density
        * depth_cm
    )
    surface_density_msun_kpc2 = (
        surface_density_g_cm2 * KPC_CM**2 / SOLAR_MASS_G
    )
    total_particle_to_electron = (
        mean_mass_per_electron_proton_masses / mean_particle_mass_proton_masses
    )
    pressure = (
        total_particle_to_electron * electron_density * temperature * KEV_ERG
    )
    entropy = temperature * electron_density ** (-2.0 / 3.0)
    sound_speed = np.sqrt(
        adiabatic_index
        * temperature
        * KEV_ERG
        / (mean_particle_mass_proton_masses * PROTON_MASS_G)
    ) / 1.0e5
    gas_mass = surface_density_msun_kpc2 * area
    return {
        "emission_measure_cm3": emission_measure,
        "emission_measure_per_area_cm5": emission_measure_per_area,
        "electron_density_cm3": electron_density,
        "gas_surface_density_msun_kpc2": surface_density_msun_kpc2,
        "gas_mass_msun": gas_mass,
        "thermal_pressure_erg_cm3": pressure,
        "entropy_proxy_keV_cm2": entropy,
        "sound_speed_km_s": sound_speed,
    }


def compression_mach_number(
    density_compression: ArrayLike, *, adiabatic_index: float = 5.0 / 3.0
) -> FloatArray:
    """Invert the Rankine-Hugoniot density jump for the upstream Mach number."""

    compression = _positive(density_compression, name="density_compression")
    if not math.isfinite(adiabatic_index) or adiabatic_index <= 1.0:
        raise ValueError("adiabatic_index must be finite and greater than one")
    maximum = (adiabatic_index + 1.0) / (adiabatic_index - 1.0)
    if np.any(compression < 1.0) or np.any(compression >= maximum):
        raise ValueError("density_compression must lie in [1, strong-shock limit)")
    mach_squared = 2.0 * compression / (
        (adiabatic_index + 1.0) - (adiabatic_index - 1.0) * compression
    )
    return np.sqrt(mach_squared)


def temperature_jump_from_mach(
    mach_number: ArrayLike, *, adiabatic_index: float = 5.0 / 3.0
) -> FloatArray:
    """Return the ideal-gas downstream/upstream temperature jump."""

    mach = _positive(mach_number, name="mach_number")
    if np.any(mach < 1.0):
        raise ValueError("mach_number must be at least one")
    gamma = float(adiabatic_index)
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("adiabatic_index must be finite and greater than one")
    mach_squared = mach**2
    pressure_jump = (2.0 * gamma * mach_squared - (gamma - 1.0)) / (
        gamma + 1.0
    )
    density_jump = (gamma + 1.0) * mach_squared / (
        (gamma - 1.0) * mach_squared + 2.0
    )
    return pressure_jump / density_jump


def temperature_jump_mach_number(
    temperature_jump: ArrayLike, *, adiabatic_index: float = 5.0 / 3.0
) -> FloatArray:
    """Numerically invert the ideal-gas temperature jump."""

    jump = _positive(temperature_jump, name="temperature_jump")
    if np.any(jump < 1.0):
        raise ValueError("temperature_jump must be at least one")
    gamma = float(adiabatic_index)
    output = np.empty_like(jump)
    for index, value in np.ndenumerate(jump):
        if math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-14):
            output[index] = 1.0
            continue
        root = brentq(
            lambda mach, target=float(value): float(
                temperature_jump_from_mach(mach, adiabatic_index=gamma) - target
            ),
            1.0,
            1.0e3,
        )
        output[index] = root
    return output


def shock_speed_km_s(mach_number: ArrayLike, upstream_sound_speed_km_s: ArrayLike) -> FloatArray:
    mach = _positive(mach_number, name="mach_number")
    sound_speed = _positive(
        upstream_sound_speed_km_s, name="upstream_sound_speed_km_s"
    )
    return mach * sound_speed


def json_scalars(state: dict[str, FloatArray]) -> dict[str, Any]:
    """Convert a scalar gas state to plain JSON values for audit reports."""

    output: dict[str, Any] = {}
    for key, value in state.items():
        array = np.asarray(value)
        output[key] = float(array) if array.ndim == 0 else array.tolist()
    return output
