import numpy as np

from voidscreen.potential_channel_qumond import (
    inward_monotone_majorant,
    path_diluted_channel_exponent,
    path_diluted_potential_channel_acceleration,
    potential_channel_acceleration,
    potential_channel_exponent,
    rar_qumond_boost,
    system_potential_path_coordinate,
)


def test_channel_exponent_has_declared_limits():
    depth = np.array([0.0, 1.0e-6, 1.0])
    exponent = potential_channel_exponent(
        depth,
        transition_depth=1.0e-6,
        transition_power=2.0,
        endpoint_exponent=4.0,
    )
    assert exponent[0] == 1.0
    assert np.isclose(exponent[1], 2.5)
    assert np.isclose(exponent[2], 4.0, rtol=1e-10)


def test_one_channel_is_exact_fixed_rar_qumond_boost():
    gbar = np.geomspace(1.0e-13, 1.0e-8, 20)
    response = potential_channel_acceleration(
        gbar,
        np.zeros_like(gbar),
        a0_m_s2=1.2e-10,
        transition_depth=1.0e-6,
        transition_power=2.0,
        endpoint_exponent=4.0,
    )
    assert np.allclose(response["enhancement"], rar_qumond_boost(gbar, 1.2e-10))


def test_deep_potential_raises_boost_toward_fourth_power():
    gbar = np.array([6.0e-11])
    response = potential_channel_acceleration(
        gbar,
        np.array([1.0]),
        a0_m_s2=1.2e-10,
        transition_depth=1.0e-6,
        transition_power=2.0,
        endpoint_exponent=4.0,
    )
    base = rar_qumond_boost(gbar, 1.2e-10)
    assert np.allclose(response["enhancement"], base**4, rtol=1e-10)


def test_high_acceleration_limit_is_exponentially_newtonian():
    response = potential_channel_acceleration(
        np.array([1.0]),
        np.array([1.0e-5]),
        a0_m_s2=1.2e-10,
        transition_depth=1.0e-6,
        transition_power=2.0,
        endpoint_exponent=4.0,
    )
    assert response["enhancement"][0] == 1.0


def test_path_dilution_has_inverse_square_root_primary_limit():
    response = path_diluted_channel_exponent(
        np.array([1.0, 1.0]),
        np.array([1.0, 9.0]),
        transition_depth=1.0e-6,
        transition_power=2.0,
        extra_spatial_channels=3.0,
        path_power=0.5,
    )
    assert np.allclose(response["channel_exponent"], [4.0, 2.0], rtol=1e-10)


def test_path_ratio_below_point_mass_is_clipped_to_one():
    response = path_diluted_channel_exponent(
        np.array([1.0]),
        np.array([0.5]),
        transition_depth=1.0e-6,
        transition_power=2.0,
        extra_spatial_channels=3.0,
        path_power=0.5,
    )
    assert response["clipped_potential_path_ratio"][0] == 1.0
    assert np.isclose(response["channel_exponent"][0], 4.0, rtol=1e-10)


def test_zero_extra_channels_is_exact_fixed_rar_response():
    gbar = np.geomspace(1.0e-13, 1.0e-8, 12)
    response = path_diluted_potential_channel_acceleration(
        gbar,
        np.full_like(gbar, 1.0e-5),
        np.full_like(gbar, 10.0),
        a0_m_s2=1.2e-10,
        transition_depth=1.0e-6,
        transition_power=2.0,
        extra_spatial_channels=0.0,
        path_power=0.5,
    )
    assert np.allclose(response["enhancement"], rar_qumond_boost(gbar, 1.2e-10))


def test_system_path_coordinate_is_scale_free_and_uses_separate_extrema():
    depth = np.array([4.0, 3.0, 2.0]) / 100.0**2
    radius = np.array([1.0, 2.0, 4.0])
    gbar = np.array([1.0, 1.5, 0.5])
    coordinate = system_potential_path_coordinate(
        depth,
        radius,
        gbar,
        light_speed_m_s=100.0,
    )
    assert np.isclose(coordinate, 4.0 / 3.0)
    rescaled = system_potential_path_coordinate(
        depth,
        7.0 * radius,
        gbar / 7.0,
        light_speed_m_s=100.0,
    )
    assert np.isclose(rescaled, coordinate)


def test_inward_monotone_majorant_is_minimal_and_nonincreasing():
    values = np.array([1.0, 2.0, 1.5, 1.8, 1.2])
    envelope = inward_monotone_majorant(values)
    assert np.allclose(envelope, [2.0, 2.0, 1.8, 1.8, 1.2])
    assert np.all(envelope >= values)
    assert np.all(np.diff(envelope) <= 0.0)
