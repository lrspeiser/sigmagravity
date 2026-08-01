import numpy as np
import pytest

from voidscreen.raw_lensing import (
    C_M_S,
    KPC_M,
    finite_ratio_or_none,
    loglog_interpolate_with_tails,
    pseudo_elliptical_deflection,
    spherical_deflection_radians,
)


def test_finite_ratio_or_none_rejects_failed_root_scores():
    assert finite_ratio_or_none(1.0, 2.0) == 0.5
    assert finite_ratio_or_none(1.0, float("inf")) is None
    assert finite_ratio_or_none(float("nan"), 2.0) is None
    assert finite_ratio_or_none(1.0, 0.0) is None


def test_loglog_interpolation_uses_explicit_outer_slope():
    result = loglog_interpolate_with_tails(
        [1.0, 10.0, 100.0], [1.0, 10.0], [100.0, 10.0], outer_slope=-2.0
    )
    assert result[0] == pytest.approx(100.0)
    assert result[1] == pytest.approx(10.0)
    assert result[2] == pytest.approx(0.1)


def test_spherical_deflection_recovers_point_mass_result():
    mass_kg = 1.0e14 * 1.988409870698051e30
    gravitational_constant = 6.67430e-11
    impact_kpc = 100.0

    def acceleration(radius_kpc):
        radius_m = np.asarray(radius_kpc) * KPC_M
        return gravitational_constant * mass_kg / radius_m**2

    measured = spherical_deflection_radians(
        [impact_kpc], acceleration, maximum_radius_kpc=1.0e8, integration_points=3000
    )[0]
    expected = (
        4.0
        * gravitational_constant
        * mass_kg
        / (impact_kpc * KPC_M * C_M_S**2)
    )
    assert measured == pytest.approx(expected, rel=2.0e-5)


def test_pseudo_elliptical_deflection_is_radial_at_unit_axis_ratio():
    radial = lambda radius: np.full_like(np.asarray(radius), 2.0, dtype=float)
    alpha_x, alpha_y = pseudo_elliptical_deflection(
        [3.0],
        [4.0],
        radial,
        axis_ratio=1.0,
        phi_radian=0.7,
        center_x_arcsec=0.0,
        center_y_arcsec=0.0,
    )
    assert alpha_x[0] == pytest.approx(1.2)
    assert alpha_y[0] == pytest.approx(1.6)
