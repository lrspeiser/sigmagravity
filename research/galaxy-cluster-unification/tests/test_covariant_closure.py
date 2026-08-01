import numpy as np
import pytest

from voidscreen.covariant_closure import (
    aqual_characteristics,
    causal_catchup_characteristics,
    equilibrium_sigma_from_density,
    sigma_metric_lensing_acceleration,
)


def test_zero_slip_returns_the_dynamical_acceleration():
    result = sigma_metric_lensing_acceleration(
        [1.0, 2.0], [3.0, 5.0], [0.2, 1.0], zeta=0.0
    )
    np.testing.assert_allclose(result, [3.0, 5.0])


def test_conformal_limit_cancels_extra_lensing_at_broken_vacuum():
    result = sigma_metric_lensing_acceleration([1.0], [4.0], [1.0], zeta=-2.0)
    assert result[0] == pytest.approx(1.0)


def test_dense_sigma_zero_limit_is_newtonian_for_any_slip():
    result = sigma_metric_lensing_acceleration([2.0], [2.0], [0.0], zeta=50.0)
    assert result[0] == pytest.approx(2.0)


def test_naive_covariant_aqual_is_hyperbolic_but_has_a_wider_scalar_cone():
    result = aqual_characteristics(
        [1.0e-12, 1.0e-10],
        [1.0, 0.0],
        a0_m_s2=1.0e-10,
        activation=1.0,
        eta=0.6,
    )
    assert np.all(result["mu_time_kinetic"] > 0.0)
    assert np.all(result["parallel_gradient_coefficient"] > 0.0)
    assert result["parallel_speed_squared_over_c2"][0] > 1.9
    assert result["parallel_speed_squared_over_c2"][1] == pytest.approx(1.0)


def test_equilibrium_sigma_density_proxy_has_dense_and_vacuum_limits():
    sigma = equilibrium_sigma_from_density(
        [0.0, 0.75e-24, 1.0e-24, 2.0e-24], rho_screen_g_cm3=1.0e-24
    )
    np.testing.assert_allclose(sigma, [1.0, 0.5, 0.0, 0.0])


def test_causal_catchup_completion_is_on_or_inside_metric_light_cone():
    result = causal_catchup_characteristics(
        [1.0e-12, 1.0e-10],
        [1.0, 0.0],
        a0_m_s2=1.0e-10,
        activation=1.0,
        eta=0.6,
        delta=10.0,
    )
    assert np.all(result["q_time_coefficient"] > 0.0)
    assert np.all(result["parallel_speed_squared_over_c2"] <= 1.0)
    assert np.all(result["perpendicular_speed_squared_over_c2"] <= 1.0)
    assert result["parallel_speed_squared_over_c2"][0] < 1.0
    assert result["parallel_speed_squared_over_c2"][1] == pytest.approx(1.0)


def test_delta_zero_is_fastest_causal_longitudinal_choice():
    result = causal_catchup_characteristics(
        [1.0e-12],
        [1.0],
        a0_m_s2=1.0e-10,
        eta=0.6,
        delta=0.0,
    )
    assert result["parallel_speed_squared_over_c2"][0] == pytest.approx(1.0)
    assert result["perpendicular_speed_squared_over_c2"][0] < 1.0
