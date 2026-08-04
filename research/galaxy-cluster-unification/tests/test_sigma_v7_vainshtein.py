from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v7_vainshtein import (
    audit_spherical_vainshtein_carrier,
    bigravity_mixing_coefficients,
    carrier_range_for_transition_m,
    equal_density_screening_residual,
    screening_coordinate,
    transition_mean_density_kg_m3,
    vainshtein_radius_m,
)

M_SUN_KG = 1.98847e30
KPC_M = 3.085677581491367e19


def test_vainshtein_radius_and_transition_density_are_reciprocal() -> None:
    mass = 1.0e11 * M_SUN_KG
    radius = 20.0 * KPC_M
    carrier_range = carrier_range_for_transition_m(radius, mass)
    assert float(vainshtein_radius_m(mass, carrier_range)) == pytest.approx(radius)
    assert float(screening_coordinate(radius, mass, carrier_range)) == pytest.approx(1.0)
    mean_density = 3.0 * mass / (4.0 * np.pi * radius**3)
    assert transition_mean_density_kg_m3(carrier_range) == pytest.approx(mean_density)


def test_equal_mean_density_systems_have_identical_screening_for_every_range() -> None:
    ranges = np.geomspace(1.0 * KPC_M, 1.0e7 * KPC_M, 1001)
    residual = equal_density_screening_residual(
        1.0e11 * M_SUN_KG,
        20.0 * KPC_M,
        1.0e14 * M_SUN_KG,
        200.0 * KPC_M,
        ranges,
    )
    assert residual["mass_over_radius_cubed_ratio"] == pytest.approx(1.0)
    assert residual["maximum_relative_screening_coordinate_difference"] < 2.0e-15


def test_one_metric_bigravity_has_finite_fixed_exterior_enhancement() -> None:
    angles = np.linspace(0.0, np.pi / 2.0, 10001)
    coefficients = bigravity_mixing_coefficients(angles)
    assert np.min(coefficients["newton_coefficient"]) >= 0.0
    assert np.min(coefficients["yukawa_coefficient"]) >= 0.0
    assert np.max(coefficients["short_range_dynamics_factor"]) == pytest.approx(2.0)
    assert np.max(coefficients["short_range_lensing_factor"]) == pytest.approx(1.5)


def test_spherical_screen_cannot_separate_equal_density_systems_or_close_gap() -> None:
    audit = audit_spherical_vainshtein_carrier(
        carrier_ranges_m=np.geomspace(1.0 * KPC_M, 1.0e7 * KPC_M, 2001),
        mixing_angles_rad=np.linspace(0.0, np.pi / 2.0, 2001),
        protected_mass_kg=1.0e11 * M_SUN_KG,
        protected_radius_m=20.0 * KPC_M,
        target_mass_kg=1.0e14 * M_SUN_KG,
        target_radius_m=200.0 * KPC_M,
        required_lensing_enhancement=3.0,
    )
    assert not audit["gates"]["protected_system_screened_while_target_unscreened"]
    assert not audit["gates"]["useful_lensing_amplitude"]
    assert audit["gates"]["positive_mixing_coefficients"]


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (vainshtein_radius_m, (-1.0, 1.0)),
        (vainshtein_radius_m, (1.0, 0.0)),
        (screening_coordinate, (0.0, 1.0, 1.0)),
        (transition_mean_density_kg_m3, (0.0,)),
        (carrier_range_for_transition_m, (1.0, -1.0)),
        (bigravity_mixing_coefficients, (-0.1,)),
    ],
)
def test_invalid_vainshtein_inputs_are_rejected(function, arguments) -> None:
    with pytest.raises(ValueError):
        function(*arguments)
