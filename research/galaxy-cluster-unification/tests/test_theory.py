from __future__ import annotations

import numpy as np

from voidscreen.theory import h7a_acceleration, h7a_parameters, h7s_acceleration
from voidscreen.unified import A0_M_S2


def test_h7a_has_declared_deep_and_newtonian_limits() -> None:
    vector = np.asarray([0.0, -6.0, 0.2])
    deep_gbar = A0_M_S2 * 1e-3
    deep = h7a_acceleration(
        np.asarray([deep_gbar]), np.asarray([1e-6]), vector
    )[0]
    expected_deep = np.sqrt(A0_M_S2 * deep_gbar)
    assert np.isclose(deep, expected_deep, rtol=0.05)

    high_gbar = A0_M_S2 * 1e5
    high = h7a_acceleration(
        np.asarray([high_gbar]), np.asarray([1e-6]), vector
    )[0]
    assert high / high_gbar - 1.0 <= 1e-5


def test_h7a_environment_increases_response_scale_and_acceleration() -> None:
    vector = np.asarray([1.0, -6.0, 0.1])
    gbar = np.full(2, 1e-11)
    predicted = h7a_acceleration(gbar, np.asarray([1e-9, 1e-3]), vector)
    assert predicted[1] > predicted[0]
    parameters = h7a_parameters(vector)
    assert parameters == {"F": 10.0, "chi_t": 1e-6, "w_dex": 0.1}


def test_h7s_has_declared_deep_and_newtonian_limits() -> None:
    vector = np.asarray([0.0, -6.0, 0.2])
    deep_gbar = A0_M_S2 * 1e-3
    deep = h7s_acceleration(
        np.asarray([deep_gbar]), np.asarray([1e-6]), vector
    )[0]
    expected_deep = np.sqrt(A0_M_S2 * deep_gbar)
    assert np.isclose(deep, expected_deep, rtol=0.05)

    high_gbar = A0_M_S2 * 1e5
    high = h7s_acceleration(
        np.asarray([high_gbar]), np.asarray([1e-6]), vector
    )[0]
    assert high / high_gbar - 1.0 <= 1e-5
