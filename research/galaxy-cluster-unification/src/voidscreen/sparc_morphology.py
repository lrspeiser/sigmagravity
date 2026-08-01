"""Residual-blind disk/bulge geometry inputs derived from the SPARC snapshot."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import iv, kv


G_KPC_KM2_S2_MSUN = 4.300917270036279e-6


@dataclass(frozen=True)
class SparcProfile:
    galaxy: str
    radius_kpc: np.ndarray
    observed_velocity_km_s: np.ndarray
    velocity_error_km_s: np.ndarray
    gas_velocity_km_s: np.ndarray
    disk_velocity_unit_ml_km_s: np.ndarray
    bulge_velocity_unit_ml_km_s: np.ndarray
    disk_surface_brightness: np.ndarray
    bulge_surface_brightness: np.ndarray


def parse_sparc_metadata(path: Path) -> pd.DataFrame:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(
                    {
                        "galaxy": line[0:11].strip(),
                        "hubble_type": int(line[12:14]),
                        "distance_mpc": float(line[15:21]),
                        "inclination_deg": float(line[30:34]),
                        "luminosity_3p6_billion_solar": float(line[40:47]),
                        "effective_radius_kpc": float(line[56:61]),
                        "effective_surface_brightness": float(line[62:70]),
                        "disk_scale_kpc": float(line[71:76]),
                        "disk_central_surface_brightness": float(line[77:85]),
                        "HI_mass_billion_solar": float(line[86:93]),
                        "HI_radius_kpc": float(line[94:99]),
                        "flat_velocity_km_s": float(line[100:105]),
                        "flat_velocity_error_km_s": float(line[106:111]),
                        "quality": int(line[112:115]),
                    }
                )
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Malformed SPARC metadata at {path}:{line_number}") from exc
    frame = pd.DataFrame(rows)
    if len(frame) != 175 or frame["galaxy"].duplicated().any():
        raise ValueError("SPARC metadata must contain 175 unique galaxies")
    return frame


def parse_sparc_profile(path: Path) -> SparcProfile:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"Expected eight SPARC columns at {path}:{line_number}")
            try:
                rows.append([float(value) for value in parts[:8]])
            except ValueError as exc:
                raise ValueError(f"Non-numeric SPARC profile at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"No SPARC profile rows in {path}")
    values = np.asarray(rows, dtype=float)
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    return SparcProfile(
        galaxy=Path(path).name.removesuffix("_rotmod.dat"),
        radius_kpc=values[:, 0],
        observed_velocity_km_s=values[:, 1],
        velocity_error_km_s=values[:, 2],
        gas_velocity_km_s=values[:, 3],
        disk_velocity_unit_ml_km_s=values[:, 4],
        bulge_velocity_unit_ml_km_s=values[:, 5],
        disk_surface_brightness=values[:, 6],
        bulge_surface_brightness=values[:, 7],
    )


def exponential_disk_velocity_squared_per_solar_mass(
    radius_kpc: np.ndarray, disk_scale_kpc: float
) -> np.ndarray:
    radius = np.asarray(radius_kpc, dtype=float)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius_kpc must be finite and positive")
    if not math.isfinite(disk_scale_kpc) or disk_scale_kpc <= 0.0:
        raise ValueError("disk_scale_kpc must be finite and positive")
    y = radius / (2.0 * disk_scale_kpc)
    bessel = iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y)
    return 2.0 * G_KPC_KM2_S2_MSUN / disk_scale_kpc * np.square(y) * bessel


def fit_exponential_disk_luminosity(
    profile: SparcProfile, disk_scale_kpc: float
) -> dict[str, float]:
    velocity = profile.disk_velocity_unit_ml_km_s
    valid = np.isfinite(velocity) & (velocity > 0.0)
    if valid.sum() < 3:
        return {
            "disk_luminosity_fit_solar": math.nan,
            "disk_velocity_fractional_rms": math.nan,
            "disk_fit_points": int(valid.sum()),
        }
    shape = exponential_disk_velocity_squared_per_solar_mass(
        profile.radius_kpc[valid], disk_scale_kpc
    )
    observed_squared = np.square(velocity[valid])
    mass = float(np.dot(shape, observed_squared) / np.dot(shape, shape))
    predicted = np.sqrt(np.maximum(mass * shape, 0.0))
    fractional_rms = float(
        np.sqrt(np.mean(np.square(predicted / velocity[valid] - 1.0)))
    )
    return {
        "disk_luminosity_fit_solar": mass,
        "disk_velocity_fractional_rms": fractional_rms,
        "disk_fit_points": int(valid.sum()),
    }


def fit_hernquist_bulge(profile: SparcProfile) -> dict[str, float]:
    velocity = profile.bulge_velocity_unit_ml_km_s
    valid = np.isfinite(velocity) & (velocity > 1.0e-3)
    if valid.sum() < 3:
        return {
            "bulge_luminosity_fit_solar": 0.0,
            "bulge_scale_fit_kpc": math.nan,
            "bulge_velocity_fractional_rms": math.nan,
            "bulge_fit_points": int(valid.sum()),
        }
    radius = profile.radius_kpc[valid]
    observed = velocity[valid]

    def residual(vector: np.ndarray) -> np.ndarray:
        mass = 10.0 ** vector[0]
        scale = 10.0 ** vector[1]
        predicted_squared = G_KPC_KM2_S2_MSUN * mass * radius / np.square(
            radius + scale
        )
        return np.log(np.sqrt(predicted_squared) / observed)

    outer_mass_guess = max(
        float(np.median(np.square(observed) * np.square(radius) / (G_KPC_KM2_S2_MSUN * radius))),
        1.0e6,
    )
    result = least_squares(
        residual,
        np.asarray([math.log10(outer_mass_guess), math.log10(max(np.median(radius) / 3.0, 0.02))]),
        bounds=(np.asarray([5.0, -3.0]), np.asarray([13.5, 2.5])),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=5000,
    )
    mass = float(10.0 ** result.x[0])
    scale = float(10.0 ** result.x[1])
    fractional_rms = float(np.sqrt(np.mean(np.square(np.expm1(residual(result.x))))))
    return {
        "bulge_luminosity_fit_solar": mass,
        "bulge_scale_fit_kpc": scale,
        "bulge_velocity_fractional_rms": fractional_rms,
        "bulge_fit_points": int(valid.sum()),
    }


def profile_light_integrals(profile: SparcProfile) -> dict[str, float]:
    radius_pc = profile.radius_kpc * 1000.0
    disk = float(2.0 * math.pi * np.trapezoid(profile.disk_surface_brightness * radius_pc, radius_pc))
    bulge = float(2.0 * math.pi * np.trapezoid(profile.bulge_surface_brightness * radius_pc, radius_pc))
    return {
        "disk_profile_light_solar": max(disk, 0.0),
        "bulge_profile_light_solar": max(bulge, 0.0),
    }


def build_sparc_morphology_catalog(
    sparc_directory: Path,
    *,
    disk_mass_to_light: float = 0.5,
    bulge_mass_to_light: float = 0.7,
    helium_factor: float = 1.33,
) -> pd.DataFrame:
    directory = Path(sparc_directory)
    metadata = parse_sparc_metadata(directory / "table1.dat")
    rows = []
    for record in metadata.to_dict(orient="records"):
        profile = parse_sparc_profile(
            directory / "rotmod" / f"{record['galaxy']}_rotmod.dat"
        )
        disk_fit = fit_exponential_disk_luminosity(profile, record["disk_scale_kpc"])
        bulge_fit = fit_hernquist_bulge(profile)
        integrals = profile_light_integrals(profile)
        disk_mass = disk_mass_to_light * disk_fit["disk_luminosity_fit_solar"]
        bulge_mass = bulge_mass_to_light * bulge_fit["bulge_luminosity_fit_solar"]
        gas_mass = helium_factor * record["HI_mass_billion_solar"] * 1.0e9
        stellar_mass = disk_mass + bulge_mass
        baryonic_mass = stellar_mass + gas_mass
        stellar_bt = bulge_mass / stellar_mass if stellar_mass > 0.0 else math.nan
        baryonic_bt = bulge_mass / baryonic_mass if baryonic_mass > 0.0 else math.nan
        row = {
            **record,
            **disk_fit,
            **bulge_fit,
            **integrals,
            "disk_mass_solar": disk_mass,
            "bulge_mass_solar": bulge_mass,
            "gas_mass_solar": gas_mass,
            "stellar_mass_solar": stellar_mass,
            "baryonic_mass_solar": baryonic_mass,
            "stellar_bulge_fraction": stellar_bt,
            "baryonic_bulge_fraction": baryonic_bt,
            "gas_fraction": gas_mass / baryonic_mass if baryonic_mass > 0.0 else math.nan,
            "bulge_scale_over_disk_scale": (
                bulge_fit["bulge_scale_fit_kpc"] / record["disk_scale_kpc"]
                if math.isfinite(bulge_fit["bulge_scale_fit_kpc"])
                else math.nan
            ),
        }
        rows.append(row)
    output = pd.DataFrame(rows)
    if len(output) != 175 or output["galaxy"].duplicated().any():
        raise RuntimeError("morphology catalog lost SPARC systems")
    return output.sort_values("galaxy", kind="stable").reset_index(drop=True)
