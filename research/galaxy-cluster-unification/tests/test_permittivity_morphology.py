from __future__ import annotations

import numpy as np

from voidscreen.axisymmetric_permittivity import AxisymmetricGrid
from voidscreen.permittivity_morphology import (
    MorphologyParameters,
    composite_galaxy_density,
    solve_morphology_response,
)


def baseline(**changes) -> MorphologyParameters:
    values = {
        "stellar_bulge_fraction": 0.3,
        "disk_vertical_scale_over_Rdisk": 0.2,
        "bulge_scale_over_Rdisk": 0.2,
        "gas_fraction": 0.2,
        "gas_radial_scale_over_Rdisk": 2.5,
        "gas_vertical_scale_over_Rdisk": 0.1,
        "minimum_permittivity": 0.1,
        "log10_critical_density_dimensionless": -2.0,
        "sharpness": 2.0,
        "smoothing_length_over_Rdisk": 0.3,
    }
    values.update(changes)
    return MorphologyParameters(**values)


def test_composite_mass_fractions_sum_to_one() -> None:
    grid = AxisymmetricGrid(48, 56, 10.0, 5.0)
    density, components, masses = composite_galaxy_density(grid, baseline())
    assert np.isfinite(density).all()
    assert sum(masses.values()) == 1.0
    assert masses == {"disk": 0.56, "bulge": 0.24, "gas": 0.2}
    np.testing.assert_allclose(density, sum(components.values()))


def test_unit_permittivity_recovers_newtonian_response_for_any_morphology() -> None:
    grid = AxisymmetricGrid(56, 64, 10.0, 5.0)
    response = solve_morphology_response(
        grid,
        baseline(minimum_permittivity=1.0),
        response_radii=np.asarray([1.0, 2.2, 4.0, 6.0, 8.0]),
    )
    np.testing.assert_allclose(response["midplane_acceleration_enhancement"], 1.0)
    np.testing.assert_allclose(response["geometry_only_enhancement"], 1.0)
    assert abs(response["outer_speed_slope_change"]) < 1.0e-12
    probe = response["above_plane_probe"]
    assert abs(probe["radial_acceleration_enhancement"] - 1.0) < 1.0e-12
    assert abs(probe["vertical_acceleration_enhancement"] - 1.0) < 1.0e-12
    assert abs(probe["constitutive_direction_ratio_change"] - 1.0) < 1.0e-12


def test_disk_and_bulge_extremes_return_finite_responses() -> None:
    grid = AxisymmetricGrid(56, 64, 10.0, 5.0)
    for bulge_fraction in (0.0, 1.0):
        response = solve_morphology_response(
            grid,
            baseline(stellar_bulge_fraction=bulge_fraction),
            response_radii=np.asarray([1.0, 2.2, 4.0, 6.0, 8.0]),
        )
        assert np.isfinite(response["midplane_acceleration_enhancement"]).all()
        assert np.isfinite(response["modified_speed_log_slope"])
        assert response["epsilon_minimum_realized"] >= 0.1
        assert response["epsilon_maximum_realized"] <= 1.0
