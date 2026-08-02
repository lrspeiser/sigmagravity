from __future__ import annotations

import numpy as np

from voidscreen.componentwise_mond_lensing import (
    componentwise_simple_mond_excess_deflection,
    dimensionless_simple_mond_excess_deflection,
)


def test_point_mass_kernel_has_screened_and_deep_mond_limits():
    values = dimensionless_simple_mond_excess_deflection(
        np.asarray([1.0e-5, 1.0, 1.0e3])
    )
    assert values[0] < 1.0e-3
    assert values[0] < values[1] < values[2]
    assert np.isclose(values[2], 2.0 * np.pi, rtol=2.0e-3)


def test_deep_componentwise_deflection_scales_as_square_root_mass():
    first_east, first_north = componentwise_simple_mond_excess_deflection(
        [1000.0],
        [0.0],
        [0.0],
        [0.0],
        [1.0e10],
        kpc_per_arcsec=1.0,
        distance_ratio=1.0,
        softening_kpc=0.1,
    )
    second_east, second_north = componentwise_simple_mond_excess_deflection(
        [1000.0],
        [0.0],
        [0.0],
        [0.0],
        [4.0e10],
        kpc_per_arcsec=1.0,
        distance_ratio=1.0,
        softening_kpc=0.1,
    )
    assert np.isclose(second_east[0] / first_east[0], 2.0, rtol=3.0e-3)
    assert abs(first_north[0]) < 1.0e-14
    assert abs(second_north[0]) < 1.0e-14


def test_componentwise_vectors_rotate_and_sum_before_observation():
    alpha_east, alpha_north = componentwise_simple_mond_excess_deflection(
        [10.0, 0.0],
        [0.0, 10.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0e10, 1.0e10],
        kpc_per_arcsec=1.0,
        distance_ratio=0.5,
        softening_kpc=[1.0, 1.0],
    )
    assert np.isclose(alpha_east[0], alpha_north[1])
    assert np.isclose(alpha_north[0], alpha_east[1])
    assert alpha_east[0] > 0.0
    assert alpha_north[1] > 0.0
