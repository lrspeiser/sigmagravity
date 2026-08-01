import math

import numpy as np
import pytest

from voidscreen.spherical_spacetime import (
    closed_sphere_area_enhancement,
    global_closed_acceleration,
    hard_cavity_best_axis_enhancement,
    hard_cavity_flow_components,
    hard_cavity_isotropic_rms_enhancement,
    local_mass_curvature_acceleration,
    stellar_area_covering_fraction,
)


def test_closed_sphere_recovers_flat_space_and_strengthens_flux():
    tiny = closed_sphere_area_enhancement([0.0, 1.0e-6])
    assert tiny[0] == 1.0
    assert np.isclose(tiny[1], 1.0 + 1.0e-12 / 3.0)
    assert closed_sphere_area_enhancement([1.0])[0] > 1.0


def test_global_and_local_curvature_recover_newtonian_limit():
    assert np.allclose(global_closed_acceleration([1.0e-10], [1.0], 1.0e9, maximum_x=2.9), 1.0e-10)
    assert np.allclose(local_mass_curvature_acceleration([1.0e-10], [1.0], 0.0), 1.0e-10)


def test_closed_sphere_rejects_antipodal_pole():
    with pytest.raises(ValueError):
        closed_sphere_area_enhancement([0.95 * math.pi])


def test_hard_cavity_surface_redirects_instead_of_absorbing_flow():
    radial, polar = hard_cavity_flow_components([0.0, math.pi / 2.0], [1.0, 1.0])
    assert np.allclose(radial, [0.0, 0.0])
    assert np.allclose(polar, [0.0, -1.5])
    assert np.isclose(hard_cavity_best_axis_enhancement([1.0])[0], 1.5)
    assert np.isclose(hard_cavity_isotropic_rms_enhancement([1.0])[0], math.sqrt(1.5))


def test_stellar_cavity_covering_fraction_is_tiny_for_a_galaxy():
    fraction = stellar_area_covering_fraction([1.0e11], [3.0])[0]
    assert fraction < 1.0e-10
