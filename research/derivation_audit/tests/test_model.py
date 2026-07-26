import numpy as np

from sigma_sprint.model import (
    DEFAULT_G_DAGGER,
    deep_btfr_velocity4,
    enhancement_kernel,
    infer_B,
    nu,
    predict_acceleration,
    q_B,
    q_potential,
    q_z,
)


def test_q_derivative_is_canonical_response():
    z = np.logspace(-10, 10, 301)
    B = 1.73
    epsilon = 1e-6
    finite_difference = (
        q_potential(z * np.exp(epsilon), B) - q_potential(z * np.exp(-epsilon), B)
    ) / (2.0 * epsilon * z)
    np.testing.assert_allclose(finite_difference, q_z(z, B), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        q_z(z, B), nu(DEFAULT_G_DAGGER * np.sqrt(z), B), rtol=2e-14
    )
    b_epsilon = 1e-6
    z_for_B = z[z <= 1e4]
    finite_q_B = (
        q_potential(z_for_B, B + b_epsilon) - q_potential(z_for_B, B - b_epsilon)
    ) / (2.0 * b_epsilon)
    np.testing.assert_allclose(finite_q_B, q_B(z_for_B), rtol=2e-7, atol=2e-9)


def test_asymptotic_limits():
    low = 1e-12 * DEFAULT_G_DAGGER
    high = 1e12 * DEFAULT_G_DAGGER
    assert np.isclose(enhancement_kernel(low) * np.sqrt(low / DEFAULT_G_DAGGER), 1.0)
    assert enhancement_kernel(high) < 1e-17
    assert abs(nu(high, 3.0) - 1.0) < 1e-15


def test_infer_B_round_trip():
    gbar = np.logspace(-13, -8, 100)
    B = np.linspace(0.2, 9.0, len(gbar))
    gtot = predict_acceleration(gbar, B)
    np.testing.assert_allclose(infer_B(gbar, gtot), B, rtol=2e-13, atol=2e-13)


def test_deep_btfr_dimensions_and_degeneracy():
    mass = 2.0e40
    value = deep_btfr_velocity4(mass, 2.0)
    doubled_B_lower_gdagger = deep_btfr_velocity4(
        mass, 4.0, g_dagger=DEFAULT_G_DAGGER / 4.0
    )
    assert value > 0
    assert np.isclose(value, doubled_B_lower_gdagger)
