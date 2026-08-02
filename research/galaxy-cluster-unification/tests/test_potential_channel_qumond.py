import numpy as np

from voidscreen.potential_channel_qumond import (
    potential_channel_acceleration,
    potential_channel_exponent,
    rar_qumond_boost,
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
