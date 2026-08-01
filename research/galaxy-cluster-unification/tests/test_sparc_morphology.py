from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voidscreen.sparc_morphology import (
    G_KPC_KM2_S2_MSUN,
    SparcProfile,
    exponential_disk_velocity_squared_per_solar_mass,
    fit_hernquist_bulge,
    parse_sparc_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def test_profile_parser_retains_disk_bulge_surface_brightness_columns() -> None:
    profile = parse_sparc_profile(ROOT / "data/raw/sparc/rotmod/NGC7814_rotmod.dat")
    assert profile.galaxy == "NGC7814"
    assert len(profile.radius_kpc) == 18
    assert profile.bulge_velocity_unit_ml_km_s[0] > profile.disk_velocity_unit_ml_km_s[0]
    assert profile.bulge_surface_brightness[0] > profile.disk_surface_brightness[0]


def test_hernquist_fit_recovers_synthetic_mass_and_scale() -> None:
    radius = np.geomspace(0.1, 30.0, 80)
    mass = 3.0e10
    scale = 1.2
    velocity = np.sqrt(G_KPC_KM2_S2_MSUN * mass * radius / np.square(radius + scale))
    zeros = np.zeros_like(radius)
    profile = SparcProfile(
        galaxy="synthetic",
        radius_kpc=radius,
        observed_velocity_km_s=velocity,
        velocity_error_km_s=np.ones_like(radius),
        gas_velocity_km_s=zeros,
        disk_velocity_unit_ml_km_s=zeros,
        bulge_velocity_unit_ml_km_s=velocity,
        disk_surface_brightness=zeros,
        bulge_surface_brightness=np.ones_like(radius),
    )
    result = fit_hernquist_bulge(profile)
    assert result["bulge_luminosity_fit_solar"] == pytest.approx(mass, rel=1.0e-9)
    assert result["bulge_scale_fit_kpc"] == pytest.approx(scale, rel=1.0e-9)
    assert result["bulge_velocity_fractional_rms"] < 1.0e-10


def test_exponential_disk_kernel_scales_linearly_with_mass() -> None:
    radius = np.geomspace(0.1, 20.0, 50)
    unit = exponential_disk_velocity_squared_per_solar_mass(radius, 2.5)
    assert np.all(unit > 0.0)
    np.testing.assert_allclose(7.0e9 * unit / (1.0e9 * unit), 7.0)
