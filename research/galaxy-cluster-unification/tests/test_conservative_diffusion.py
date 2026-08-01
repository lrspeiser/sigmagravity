import numpy as np

from voidscreen.conservative_diffusion import (
    gaussian_tail_upper_bound,
    low_acceleration_activation,
    radial_shape_activation,
    redistributed_cumulative_mass,
)


def test_redistribution_conserves_total_and_contraction_moves_mass_inward():
    radius = np.linspace(0.1, 10.0, 100)
    mass = radius**2
    mass /= mass[-1]
    contracted, error = redistributed_cumulative_mass(
        radius,
        mass,
        r80=8.0,
        position_scale=0.8,
        width_over_r80=0.01,
    )
    assert error < 1e-12
    assert contracted[50] > mass[50]
    assert np.isclose(contracted[-1], mass[-1], rtol=1e-3)


def test_activation_and_solar_tail_have_expected_limits():
    assert low_acceleration_activation(1e-12, a0_m_s2=1.2e-10, power=2.0) > 0.99
    assert low_acceleration_activation(1e-3, a0_m_s2=1.2e-10, power=2.0) < 1e-12
    assert gaussian_tail_upper_bound(evaluation_radius=60.0, source_radius=1.0, sigma=0.5) < 1e-100


def test_radial_shape_activation_is_bounded_and_monotonic():
    low = radial_shape_activation(0.4, midpoint=0.6, width=0.1)
    middle = radial_shape_activation(0.6, midpoint=0.6, width=0.1)
    high = radial_shape_activation(0.8, midpoint=0.6, width=0.1)
    assert 0.0 < low < middle < high < 1.0
    assert middle == 0.5
