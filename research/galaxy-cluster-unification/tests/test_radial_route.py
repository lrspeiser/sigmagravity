import numpy as np

from voidscreen.arc_invariants import generalized_solar_diagnostics
from voidscreen.radial_route import (
    potential_transition_scale,
    remap_extra_acceleration,
    remapped_solar_diagnostics,
)


def test_zero_fraction_and_unit_scale_are_exact_parents():
    radius = np.geomspace(1.0, 100.0, 50)
    extra = 3.0 / radius
    assert np.array_equal(
        remap_extra_acceleration(
            radius, extra, route_fraction=0.0, radial_scale=0.8
        ),
        extra,
    )
    assert np.array_equal(
        remap_extra_acceleration(
            radius, extra, route_fraction=0.7, radial_scale=1.0
        ),
        extra,
    )


def test_power_law_flux_has_analytic_remap():
    radius = np.geomspace(1.0, 100.0, 50)
    extra = 3.0 / radius
    fraction = 0.25
    scale = 0.8
    changed = remap_extra_acceleration(
        radius, extra, route_fraction=fraction, radial_scale=scale
    )
    expected_factor = (1.0 - fraction) + fraction / scale
    assert np.allclose(changed, expected_factor * extra, rtol=2e-14)


def test_point_mass_like_extra_is_unchanged_by_radial_scale():
    radius = np.geomspace(1.0, 100.0, 50)
    extra = 7.0 / radius**2
    changed = remap_extra_acceleration(
        radius, extra, route_fraction=1.0, radial_scale=1.2
    )
    assert np.allclose(changed, extra, rtol=2e-14)


def test_potential_transition_is_smooth_bounded_and_parent_preserving():
    depth = np.array([2.0e-8, 2.0e-6, 2.0e-4])
    parent = potential_transition_scale(depth, log_scale_amplitude=0.0)
    changed = potential_transition_scale(depth, log_scale_amplitude=0.1)
    assert np.array_equal(parent, np.ones(3))
    assert changed[0] < 1.0
    assert np.isclose(changed[1], 1.0)
    assert changed[2] > 1.0
    assert np.all(changed >= np.exp(-0.1))
    assert np.all(changed <= np.exp(0.1))


def test_variable_scale_remap_matches_local_power_law_solution():
    radius = np.geomspace(1.0, 100.0, 50)
    extra = 3.0 / radius
    scale = np.linspace(0.8, 1.2, len(radius))
    changed = remap_extra_acceleration(
        radius, extra, route_fraction=1.0, radial_scale=scale
    )
    assert np.allclose(changed, extra / scale, rtol=2e-14)


def test_solar_parent_matches_existing_diagnostics_when_addition_is_linear():
    parameters = {
        "residence_strength": 1.2300686115098052,
        "alpha": 0.75,
        "apogee_ratio": 100.0,
        "screen_a0_m_s2": 1.2e-10,
        "screen_exponent": 1.0,
        "screen_scale": 1.0,
        "mass_radius_delta": 0.0,
        "extent_leak": 0.0,
        "invariant_mode": "potential_depth",
        "invariant_power": 1.2,
        "invariant_scale": 2.0e-6,
        "secondary_path_ratio_power": 0.25,
        "photon_extra_multiplier": 1.75,
    }
    legacy = generalized_solar_diagnostics(**parameters)
    routed = remapped_solar_diagnostics(
        response_parameters=parameters,
        route_fraction=0.0,
        radial_scale=0.8,
    )
    assert np.isclose(
        routed["Earth_orbit_fractional_change"],
        legacy["Earth_orbit_fractional_change"],
        rtol=2e-5,
        atol=1e-16,
    )
    assert np.isclose(
        routed["Mercury_precession_mas_per_century"],
        legacy["Mercury_precession_mas_per_century"],
        rtol=2e-5,
        atol=1e-7,
    )
    assert routed["Cassini_proxy_pass"] == legacy["Cassini_proxy_pass"]
