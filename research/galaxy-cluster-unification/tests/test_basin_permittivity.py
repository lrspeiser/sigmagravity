from __future__ import annotations

import numpy as np
import pytest

from voidscreen.basin_action import G_SI, KPC_M, M_SUN_KG
from voidscreen.basin_permittivity import (
    confined_slab_acceleration_m_s2,
    confined_slab_flat_velocity_km_s,
    required_confinement_half_height_kpc,
    spherical_permittivity_acceleration_m_s2,
)


def test_spherical_constant_permittivity_only_rescales_inverse_square() -> None:
    radius = np.asarray([1.0, 2.0, 4.0]) * KPC_M
    acceleration = spherical_permittivity_acceleration_m_s2(
        radius, 1.0e11 * M_SUN_KG, 0.2
    )
    np.testing.assert_allclose(acceleration / acceleration[0], [1.0, 0.25, 0.0625])


def test_confined_slab_gives_inverse_radius_and_flat_speed() -> None:
    radius = np.asarray([2.0, 4.0, 8.0]) * KPC_M
    mass = 1.0e11 * M_SUN_KG
    height = 5.0 * KPC_M
    acceleration = confined_slab_acceleration_m_s2(radius, mass, height)
    np.testing.assert_allclose(acceleration / acceleration[0], [1.0, 0.5, 0.25])
    speed_squared = radius * acceleration
    np.testing.assert_allclose(speed_squared, speed_squared[0])


def test_height_velocity_inversion_round_trips() -> None:
    mass = np.asarray([1.0e8, 1.0e10, 1.0e12])
    velocity = np.asarray([25.0, 100.0, 300.0])
    height = required_confinement_half_height_kpc(mass, velocity)
    recovered = confined_slab_flat_velocity_km_s(mass, height)
    np.testing.assert_allclose(recovered, velocity, rtol=1.0e-14)


def test_spherical_formula_matches_gauss_law() -> None:
    radius = 3.0 * KPC_M
    mass = 2.0e10 * M_SUN_KG
    epsilon = 0.4
    result = spherical_permittivity_acceleration_m_s2(radius, mass, epsilon)
    assert result == pytest.approx(G_SI * mass / (epsilon * radius**2))
