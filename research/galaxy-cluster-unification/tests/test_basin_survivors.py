from __future__ import annotations

import numpy as np
import pytest

from voidscreen.basin_survivors import (
    algebraic_inverse_field_scaling,
    canonical_scalar_exterior_energy_fraction,
    direct_force_amplitude_for_field_energy_fraction,
    nonlinear_flux_law_scaling,
    uniform_vacuum_radial_acceleration_m_s2,
)


def test_flat_btfr_flux_law_uniquely_selects_aqual_power() -> None:
    powers = np.linspace(0.25, 5.0, 10_000)
    radial = np.asarray(
        [nonlinear_flux_law_scaling(value).circular_speed_radial_exponent for value in powers]
    )
    mass = np.asarray(
        [
            nonlinear_flux_law_scaling(value).circular_speed_fourth_power_mass_exponent
            for value in powers
        ]
    )
    best = np.argmin(np.square(radial) + np.square(mass - 1.0))
    assert powers[best] == pytest.approx(2.0, abs=5.0e-4)
    exact = nonlinear_flux_law_scaling(2.0)
    assert exact.acceleration_mass_exponent == pytest.approx(0.5)
    assert exact.acceleration_radial_exponent == pytest.approx(-1.0)
    assert exact.circular_speed_radial_exponent == pytest.approx(0.0)
    assert exact.circular_speed_fourth_power_mass_exponent == pytest.approx(1.0)


def test_algebraic_map_of_one_inverse_field_cannot_get_flat_and_btfr() -> None:
    # Flat v requires n=0 for X~M/r, but then Phi~X^0 is constant and has no force.
    flat_candidate = algebraic_inverse_field_scaling(0.0)
    assert flat_candidate.circular_speed_squared_radial_exponent == 0.0
    assert flat_candidate.acceleration_mass_exponent == 0.0

    btfr_candidate = algebraic_inverse_field_scaling(0.5)
    assert btfr_candidate.acceleration_mass_exponent == 0.5
    assert btfr_candidate.circular_speed_squared_radial_exponent == pytest.approx(-0.5)


def test_canonical_field_energy_is_compactness_suppressed() -> None:
    compactness = 1.0e-6
    fraction = canonical_scalar_exterior_energy_fraction(1.0, compactness)
    assert fraction == pytest.approx(1.0e-6)
    required_force = direct_force_amplitude_for_field_energy_fraction(5.0, compactness)
    assert required_force == pytest.approx(1.0e7)


def test_uniform_positive_vacuum_energy_is_outward_and_harmonic() -> None:
    radii = np.asarray([1.0, 2.0, 4.0])
    acceleration = uniform_vacuum_radial_acceleration_m_s2(radii, 1.0e-26)
    assert np.all(acceleration > 0.0)
    np.testing.assert_allclose(acceleration / radii, acceleration[0] / radii[0])
