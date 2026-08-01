import numpy as np

from voidscreen.arc_invariants import (
    generalized_add_one,
    generalized_arc_response,
    generalized_residence_coordinate,
    generalized_screen,
    invariant_multiplier,
    spherical_profile_invariants,
)
from voidscreen.arc_apogee import acceleration_screen, residence_coordinate


def test_point_mass_invariants_have_expected_limits():
    radius = np.geomspace(1.0, 100.0, 20)
    gbar = 2.0e-10 / radius**2
    result = spherical_profile_invariants(radius, gbar)
    assert np.allclose(result["potential_path_ratio"], 1.0, rtol=2e-12)
    assert np.allclose(result["enclosed_mass_log_slope"], 0.0, atol=2e-12)


def test_invariant_modes_are_neutral_at_zero_power():
    for mode in ("none", "path_ratio", "mass_growth", "coherence_length", "potential_depth"):
        value = invariant_multiplier(
            mode,
            potential_depth=[1e-7, 1e-6],
            potential_length_kpc=[10.0, 100.0],
            potential_path_ratio=[1.0, 3.0],
            enclosed_mass_log_slope=[0.0, 1.0],
            power=0.0,
            scale=10.0,
        )
        assert np.allclose(value, 1.0)


def test_photon_multiplier_changes_only_extra_lensing_channel():
    common = dict(
        gbar_m_s2=np.array([1.0e-11]),
        radius_kpc=np.array([10.0]),
        total_baryonic_mass_solar=np.array([1.0e10]),
        concentration=np.array([0.6]),
        residence_strength=1.5,
        alpha=0.75,
        apogee_ratio=100.0,
        screen_a0_m_s2=1.2e-10,
        screen_exponent=1.0,
    )
    base = generalized_arc_response(**common, photon_extra_multiplier=1.0)
    slip = generalized_arc_response(**common, photon_extra_multiplier=3.0)
    assert np.allclose(base["dynamical_enhancement"], slip["dynamical_enhancement"])
    assert np.allclose(
        slip["lensing_enhancement"] - 1.0,
        3.0 * (base["dynamical_enhancement"] - 1.0),
    )


def test_mass_delta_changes_scale_across_mass_but_not_at_pivot():
    common = dict(
        gbar_m_s2=np.array([1e-11, 1e-11]),
        radius_kpc=np.array([10.0, 10.0]),
        total_baryonic_mass_solar=np.array([1e10, 1e12]),
        concentration=np.array([0.6, 0.6]),
        residence_strength=1.0,
        alpha=1.0,
        apogee_ratio=1e6,
        screen_a0_m_s2=1.2e-10,
        screen_exponent=1.0,
    )
    base = generalized_arc_response(**common, mass_radius_delta=0.0)
    changed = generalized_arc_response(**common, mass_radius_delta=0.1)
    assert np.isclose(base["scale_radius_kpc"][0], changed["scale_radius_kpc"][0])
    assert changed["scale_radius_kpc"][1] > base["scale_radius_kpc"][1]


def test_secondary_path_factor_combines_with_primary_invariant():
    common = dict(
        gbar_m_s2=np.array([1e-11]),
        radius_kpc=np.array([10.0]),
        total_baryonic_mass_solar=np.array([1e10]),
        concentration=np.array([0.6]),
        residence_strength=1.0,
        alpha=0.75,
        apogee_ratio=100.0,
        screen_a0_m_s2=1.2e-10,
        screen_exponent=1.0,
        invariant_mode="potential_depth",
        invariant_power=1.0,
        invariant_scale=1e-6,
        potential_depth=np.array([1e-6]),
        potential_path_ratio=np.array([4.0]),
    )
    base = generalized_arc_response(**common, secondary_path_ratio_power=0.0)
    combined = generalized_arc_response(**common, secondary_path_ratio_power=0.5)
    assert np.allclose(
        combined["unit_fractional_response"],
        2.0 * base["unit_fractional_response"],
    )


def test_structural_softness_parents_reproduce_existing_operators_exactly():
    value = np.geomspace(1.0e-4, 1.0e4, 50)
    assert np.allclose(generalized_add_one(value, 1.0), 1.0 + value)
    gbar = value * 1.2e-10
    assert np.allclose(
        generalized_screen(gbar, a0_m_s2=1.2e-10, exponent=1.3, softness=1.0),
        acceleration_screen(gbar, a0_m_s2=1.2e-10, exponent=1.3),
    )
    radius = np.geomspace(0.1, 1000.0, 50)
    assert np.allclose(
        generalized_residence_coordinate(
            radius, 10.0, alpha=0.75, apogee_ratio=100.0, softness=1.0
        ),
        residence_coordinate(radius, 10.0, alpha=0.75, apogee_ratio=100.0),
    )


def test_potential_path_cross_one_is_product_and_zero_removes_only_cross_term():
    common = dict(
        gbar_m_s2=np.array([1e-11]),
        radius_kpc=np.array([10.0]),
        total_baryonic_mass_solar=np.array([1e10]),
        concentration=np.array([0.6]),
        residence_strength=1.0,
        alpha=0.75,
        apogee_ratio=100.0,
        screen_a0_m_s2=1.2e-10,
        screen_exponent=1.0,
        invariant_mode="potential_depth",
        invariant_power=1.0,
        invariant_scale=1e-6,
        potential_depth=np.array([1e-6]),
        potential_path_ratio=np.array([4.0]),
        secondary_path_ratio_power=0.5,
    )
    product = generalized_arc_response(**common, potential_path_cross=1.0)
    additive = generalized_arc_response(**common, potential_path_cross=0.0)
    assert np.allclose(product["invariant_multiplier"], 4.0)
    assert np.allclose(additive["invariant_multiplier"], 3.0)


def test_lensing_addition_softness_changes_light_but_not_dynamics():
    common = dict(
        gbar_m_s2=np.array([1e-11]),
        radius_kpc=np.array([10.0]),
        total_baryonic_mass_solar=np.array([1e10]),
        concentration=np.array([0.6]),
        residence_strength=1.0,
        alpha=0.75,
        apogee_ratio=100.0,
        screen_a0_m_s2=1.2e-10,
        screen_exponent=1.0,
        photon_extra_multiplier=1.75,
    )
    parent = generalized_arc_response(**common, lensing_addition_softness=1.0)
    changed = generalized_arc_response(**common, lensing_addition_softness=1.2)
    assert np.allclose(parent["dynamical_enhancement"], changed["dynamical_enhancement"])
    assert not np.allclose(parent["lensing_enhancement"], changed["lensing_enhancement"])


def test_extent_and_potential_scale_couplings_move_scale_radius():
    common = dict(
        gbar_m_s2=np.array([1e-11]),
        radius_kpc=np.array([10.0]),
        total_baryonic_mass_solar=np.array([1e10]),
        concentration=np.array([0.6]),
        residence_strength=1.0,
        alpha=0.75,
        apogee_ratio=100.0,
        screen_a0_m_s2=1.2e-10,
        screen_exponent=1.0,
        invariant_mode="potential_depth",
        invariant_power=1.0,
        invariant_scale=1e-6,
        potential_depth=np.array([1e-6]),
    )
    parent = generalized_arc_response(**common)
    extent = generalized_arc_response(**common, extent_scale_coupling=0.1)
    potential = generalized_arc_response(**common, potential_scale_coupling=0.1)
    assert extent["scale_radius_kpc"][0] < parent["scale_radius_kpc"][0]
    assert potential["scale_radius_kpc"][0] > parent["scale_radius_kpc"][0]
