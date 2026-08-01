import numpy as np
import pytest

from voidscreen.member_lensing import (
    circularized_member_deflection,
    member_geometry_delta_deflection,
    point_mass_einstein_radius_squared_arcsec2,
    softened_member_deflection,
)


def test_einstein_strength_scales_linearly_with_mass_and_distance_ratio():
    result = point_mass_einstein_radius_squared_arcsec2(
        [1.0e10, 2.0e10],
        lens_angular_distance_m=2.0e25,
        distance_ratio=0.5,
    )
    doubled_ratio = point_mass_einstein_radius_squared_arcsec2(
        [1.0e10],
        lens_angular_distance_m=2.0e25,
        distance_ratio=1.0,
    )
    assert result[1] == pytest.approx(2.0 * result[0])
    assert doubled_ratio[0] == pytest.approx(2.0 * result[0])


def test_centered_member_matches_its_circularized_control():
    delta_x, delta_y = member_geometry_delta_deflection(
        [1.0, 3.0],
        [2.0, -4.0],
        [0.0],
        [0.0],
        [0.7],
        [0.3],
    )
    np.testing.assert_allclose(delta_x, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(delta_y, 0.0, atol=1.0e-14)


def test_circularized_formula_matches_numeric_angular_average():
    angles = np.linspace(0.0, 2.0 * np.pi, 20_000, endpoint=False)
    member_radius = 2.3
    actual_x, actual_y = softened_member_deflection(
        [4.1],
        [0.0],
        member_radius * np.cos(angles),
        member_radius * np.sin(angles),
        np.full(len(angles), 1.0 / len(angles)),
        np.full(len(angles), 0.4),
    )
    smooth_x, smooth_y = circularized_member_deflection(
        [4.1], [0.0], [member_radius], [1.0], [0.4]
    )
    assert actual_x[0] == pytest.approx(smooth_x[0], rel=2.0e-6)
    assert actual_y[0] == pytest.approx(smooth_y[0], abs=2.0e-12)
