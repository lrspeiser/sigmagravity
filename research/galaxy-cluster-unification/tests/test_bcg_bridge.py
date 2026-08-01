from __future__ import annotations

import numpy as np

from voidscreen.bcg_bridge import (
    acceleration_from_log_mass,
    calibrate_log_offset,
    mfl_log_acceleration_at_radius,
    physical_radius_kpc,
    sersic_projected_fraction,
)
from voidscreen.data import KPC_M
from voidscreen.unified import G_SI, M_SUN_KG


def test_sersic_effective_radius_contains_half_projected_mass() -> None:
    fractions = sersic_projected_fraction(np.ones(4), np.asarray([1.0, 2.0, 4.0, 8.0]))
    assert np.allclose(fractions, 0.5, rtol=0.0, atol=1e-12)


def test_mass_acceleration_matches_definition() -> None:
    expected = G_SI * 1e11 * M_SUN_KG / (10.0 * KPC_M) ** 2
    assert np.isclose(acceleration_from_log_mass(11.0, 10.0), expected)


def test_isothermal_mfl_scaling_is_inverse_radius() -> None:
    at_re = np.log10(acceleration_from_log_mass(11.0, 5.0))
    at_two_re = mfl_log_acceleration_at_radius(11.0, 5.0, 10.0, -2.0)
    assert np.isclose(at_two_re, at_re - np.log10(2.0))


def test_calibration_offset_centers_constant_bias() -> None:
    report = calibrate_log_offset([-10.2, -9.2, -8.2], [-10.0, -9.0, -8.0])
    assert np.isclose(report["offset_dex"], 0.2)
    assert report["rms_residual_dex"] < 1e-12


def test_angular_radius_conversion() -> None:
    radius = physical_radius_kpc(100.0, 206.26480624709636)
    assert np.isclose(radius, 100.0)
