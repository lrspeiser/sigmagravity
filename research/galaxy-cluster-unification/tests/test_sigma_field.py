import numpy as np
import pytest

from voidscreen.axisymmetric_permittivity import AxisymmetricGrid
from voidscreen.sigma_field import (
    geometric_radial_faces,
    local_sigma_equilibrium,
    sigma_permittivity,
    solve_axisymmetric_sigma,
    solve_spherical_sigma,
)


def test_local_sigma_equilibrium_has_screened_and_broken_phases():
    result = local_sigma_equilibrium([0.0, 0.75, 1.0, 2.0], 1.0)
    np.testing.assert_allclose(result, [1.0, 0.5, 0.0, 0.0])


def test_sigma_permittivity_recovers_newtonian_and_vacuum_limits():
    result = sigma_permittivity([0.0, 0.5, 1.0], 0.8)
    np.testing.assert_allclose(result, [1.0, 0.8, 0.2])
    with pytest.raises(ValueError):
        sigma_permittivity([1.1], 0.8)


def test_uniform_spherical_vacuum_stays_at_vacuum_solution():
    faces = geometric_radial_faces(80, 0.01, 20.0)
    result = solve_spherical_sigma(
        faces,
        np.zeros(len(faces) - 1),
        rho_s_g_cm3=1.0,
        length_kpc=1.0,
        outer_sigma=1.0,
    )
    assert result.converged
    np.testing.assert_allclose(result.field, 1.0, atol=2.0e-6)


def test_larger_empty_cavity_builds_more_sigma():
    centers = []
    for radius in (0.3, 1.0, 3.0, 10.0):
        faces = geometric_radial_faces(120, radius * 1.0e-4, radius)
        result = solve_spherical_sigma(
            faces,
            np.zeros(len(faces) - 1),
            rho_s_g_cm3=1.0,
            length_kpc=1.0,
            outer_sigma=0.0,
        )
        centers.append(result.field[0])
    assert np.all(np.diff(centers) >= -1.0e-5)
    assert centers[0] < 0.05
    assert centers[-1] > 0.8


def test_empty_cavity_turns_on_near_linear_stability_radius_pi_L():
    center_fields = []
    for radius in (3.1, 3.2):
        faces = geometric_radial_faces(160, radius * 1.0e-5, radius)
        result = solve_spherical_sigma(
            faces,
            np.zeros(len(faces) - 1),
            rho_s_g_cm3=1.0,
            length_kpc=1.0,
            outer_sigma=0.0,
        )
        assert result.converged
        center_fields.append(result.field[0])
    assert center_fields[0] < 0.02
    assert center_fields[1] > 0.2


def test_uniform_axisymmetric_vacuum_stays_at_vacuum_solution():
    grid = AxisymmetricGrid(12, 10, 6.0, 5.0)
    result = solve_axisymmetric_sigma(
        grid,
        np.zeros((12, 10)),
        rho_s_g_cm3=1.0,
        length=0.8,
        outer_sigma=1.0,
    )
    assert result.converged
    np.testing.assert_allclose(result.field, 1.0, atol=2.0e-5)
