from __future__ import annotations

import numpy as np
import pytest

from voidscreen.axisymmetric_permittivity import (
    AxisymmetricGrid,
    double_exponential_density,
    logistic_permittivity,
    midplane_inward_acceleration,
    miyamoto_nagai_density,
    miyamoto_nagai_potential,
    normalize_density,
    represented_mass,
    solve_axisymmetric_helmholtz_smoothing,
    solve_axisymmetric_potential,
)


def test_density_normalization_uses_full_reflected_axisymmetric_volume() -> None:
    grid = AxisymmetricGrid(40, 32, 8.0, 6.0)
    radial, vertical = grid.mesh()
    density = normalize_density(grid, np.exp(-radial) * np.exp(-vertical), target_mass=3.2)
    assert represented_mass(grid, density) == pytest.approx(3.2, rel=1.0e-13)


def test_constant_permittivity_rescales_potential_and_acceleration() -> None:
    grid = AxisymmetricGrid(72, 72, 12.0, 12.0)
    density = miyamoto_nagai_density(
        grid, mass=1.0, radial_scale=1.0, vertical_scale=0.3
    )
    boundary = lambda radial, vertical: miyamoto_nagai_potential(
        radial,
        vertical,
        mass=1.0,
        radial_scale=1.0,
        vertical_scale=0.3,
    )
    newtonian = solve_axisymmetric_potential(
        grid, density, np.ones_like(density), boundary_potential=boundary
    )
    epsilon = 0.25
    scaled = solve_axisymmetric_potential(
        grid,
        density,
        np.full_like(density, epsilon),
        far_permittivity=epsilon,
        boundary_potential=lambda radial, vertical: boundary(radial, vertical) / epsilon,
    )
    np.testing.assert_allclose(scaled, newtonian / epsilon, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        midplane_inward_acceleration(grid, scaled),
        midplane_inward_acceleration(grid, newtonian) / epsilon,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


@pytest.mark.parametrize(
    ("radial_scale", "vertical_scale", "maximum_relative_error"),
    [(0.0, 0.7, 0.035), (1.0, 0.3, 0.075)],
)
def test_newtonian_solver_recovers_miyamoto_nagai_midplane_force(
    radial_scale: float, vertical_scale: float, maximum_relative_error: float
) -> None:
    grid = AxisymmetricGrid(128, 128, 16.0, 16.0)
    density = miyamoto_nagai_density(
        grid,
        mass=1.0,
        radial_scale=radial_scale,
        vertical_scale=vertical_scale,
    )
    boundary = lambda radial, vertical: miyamoto_nagai_potential(
        radial,
        vertical,
        mass=1.0,
        radial_scale=radial_scale,
        vertical_scale=vertical_scale,
    )
    potential = solve_axisymmetric_potential(
        grid, density, np.ones_like(density), boundary_potential=boundary
    )
    radius = grid.radial_centers
    numerical = midplane_inward_acceleration(grid, potential)
    analytic = radius / np.power(
        np.square(radius) + (radial_scale + vertical_scale) ** 2, 1.5
    )
    mask = (radius >= 0.75) & (radius <= 8.0)
    relative = np.abs(numerical[mask] / analytic[mask] - 1.0)
    assert np.quantile(relative, 0.95) <= maximum_relative_error


def test_helmholtz_zero_length_is_identity_and_positive_smoothing_spreads_source() -> None:
    grid = AxisymmetricGrid(64, 64, 10.0, 10.0)
    source = double_exponential_density(
        grid, mass=1.0, radial_scale=1.0, vertical_scale=0.2
    )
    np.testing.assert_array_equal(
        solve_axisymmetric_helmholtz_smoothing(grid, source, 0.0), source
    )
    smoothed = solve_axisymmetric_helmholtz_smoothing(grid, source, 0.7)
    assert smoothed.max() < source.max()
    assert smoothed[-2, -2] > source[-2, -2]
    assert np.all(smoothed >= 0.0)


def test_logistic_permittivity_has_declared_limits() -> None:
    density = np.asarray([0.0, 1.0e-6, 1.0, 1.0e6])
    epsilon = logistic_permittivity(
        density, minimum_permittivity=0.1, critical_density=1.0, sharpness=2.0
    )
    assert epsilon[0] == pytest.approx(0.1)
    assert epsilon[2] == pytest.approx(0.55)
    assert epsilon[-1] == pytest.approx(1.0, rel=1.0e-12)
    assert np.all(np.diff(epsilon) > 0.0)
