from __future__ import annotations

import numpy as np

from voidscreen.eaq import (
    aether_action_function,
    aether_feedback_shape,
    beta_from_gamma_bound,
    exterior_feedback_ratio,
    high_field_mode_speeds_squared,
    point_source_minimum_range_over_radius,
    ppn_restricted_aether_coefficients,
    required_exponential_coupling,
    scalar_tensor_gamma_minus_one,
    standard_mu_from_y,
)


def test_action_derivative_generates_standard_mu() -> None:
    y = np.geomspace(1e-5, 1e5, 1000)
    step = 1e-4
    upper = aether_action_function(y * np.exp(step))
    lower = aether_action_function(y * np.exp(-step))
    derivative = (upper - lower) / (2.0 * step * y)
    reconstructed_mu = 1.0 - 0.5 * derivative
    assert np.allclose(reconstructed_mu, standard_mu_from_y(y), rtol=2e-5, atol=2e-8)


def test_feedback_shape_is_positive_and_cancellation_safe() -> None:
    y = np.geomspace(1e-30, 1e12, 1000)
    shape = aether_feedback_shape(y)
    assert np.all(np.isfinite(shape))
    assert np.all(shape > 0.0)
    assert np.isclose(shape[0], (2.0 / 3.0) * y[0] ** 1.5, rtol=1e-12)


def test_ppn_restriction_is_luminal_and_healthy() -> None:
    c1 = 2e-5
    c14 = 1e-5
    coefficients = ppn_restricted_aether_coefficients(c1, c14)
    speeds = high_field_mode_speeds_squared(c1, c14)
    assert coefficients["c13"] == 0.0
    assert coefficients["c2"] > 0.0
    assert speeds == {"tensor": 1.0, "vector": 2.0, "scalar": 1.0}


def test_scalar_coupling_saturates_declared_gamma_bound() -> None:
    bound = 2.3e-5
    beta = beta_from_gamma_bound(bound)
    assert np.isclose(abs(scalar_tensor_gamma_minus_one(beta)), bound)


def test_prefit_environment_mapping_and_range_are_fixed() -> None:
    eta = required_exponential_coupling(100.0, 4.504241726150346e-6)
    assert np.isclose(np.exp(eta * 4.504241726150346e-6), 10.0)
    minimum_range = point_source_minimum_range_over_radius(0.05)
    assert np.isclose(np.exp(-1.0 / minimum_range), 0.95)


def test_spherical_exterior_feedback_is_positive_and_scales_with_beta() -> None:
    common = {
        "radius_m": 10.0 * 3.085677581491367e19,
        "gbar_m_s2": 1e-11,
        "target_chi": 2e-6,
        "eta_per_chi": 5e5,
        "range_over_radius": point_source_minimum_range_over_radius(0.05),
        "grid_points": 2000,
    }
    first = exterior_feedback_ratio(beta=1e-5, **common)
    second = exterior_feedback_ratio(beta=2e-5, **common)
    assert first > 0.0
    assert np.isclose(second, first / 2.0)
