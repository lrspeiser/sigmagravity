import numpy as np

from voidscreen.sigma_actions import (
    conformal_symmetron_acceleration,
    refracted_aqual_acceleration,
    refracted_aqual_free_function,
    refracted_aqual_free_function_sigma_derivative,
    refracted_aqual_mu,
    scalar_field_stress_energy_profile,
    solve_coupled_spherical_sigma,
)
from voidscreen.sigma_field import geometric_radial_faces, radial_cell_centers


def test_refracted_aqual_recovers_newton_when_sigma_is_zero():
    gbar = np.geomspace(1.0e-13, 1.0e-8, 20)
    prediction = refracted_aqual_acceleration(
        gbar, np.zeros_like(gbar), a0_m_s2=1.2e-10, eta=0.8
    )
    np.testing.assert_allclose(prediction, gbar, rtol=2.0e-15)


def test_refracted_aqual_closed_form_satisfies_field_equation():
    gbar = np.geomspace(1.0e-13, 1.0e-8, 20)
    sigma = np.linspace(0.05, 1.0, 20)
    prediction = refracted_aqual_acceleration(
        gbar, sigma, a0_m_s2=1.2e-10, activation=1.7, eta=0.6
    )
    mu = refracted_aqual_mu(
        prediction, sigma, a0_m_s2=1.2e-10, activation=1.7, eta=0.6
    )
    np.testing.assert_allclose(mu * prediction, gbar, rtol=2.0e-14)


def test_deep_aqual_point_mass_has_flat_circular_speed():
    radius = np.geomspace(10.0, 1000.0, 100)
    gbar = 1.0e-11 * (10.0 / radius) ** 2
    prediction = refracted_aqual_acceleration(
        gbar, np.ones_like(radius), a0_m_s2=1.2e-10
    )
    speed_squared = prediction * radius
    slope = np.polyfit(np.log10(radius[-30:]), np.log10(speed_squared[-30:]), 1)[0]
    assert abs(slope) < 0.01


def test_free_function_derivative_is_mu():
    X = np.geomspace(1.0e-5, 1.0e3, 30)
    sigma = np.full_like(X, 0.7)
    step = 1.0e-5 * X
    upper = refracted_aqual_free_function(X + step, sigma, activation=1.3, eta=0.4)
    lower = refracted_aqual_free_function(X - step, sigma, activation=1.3, eta=0.4)
    derivative = (upper - lower) / (2.0 * step)
    expected = (1.0 - 0.4 * sigma**2) * np.sqrt(X) / (
        np.sqrt(X) + 1.3 * sigma**2
    )
    np.testing.assert_allclose(derivative, expected, rtol=3.0e-7)


def test_conformal_symmetron_force_needs_a_sigma_gradient():
    radius = np.geomspace(1.0, 100.0, 40)
    gbar = np.full_like(radius, 1.0e-11)
    uniform = conformal_symmetron_acceleration(
        gbar, radius, np.full_like(radius, 0.5), alpha=1.0e-5
    )
    rising = conformal_symmetron_acceleration(
        gbar, radius, np.linspace(0.1, 0.9, len(radius)), alpha=1.0e-5
    )
    np.testing.assert_allclose(uniform, gbar)
    assert np.all(rising > gbar)


def test_free_function_sigma_derivative_matches_finite_difference():
    X = np.geomspace(1.0e-3, 1.0e2, 20)
    sigma = np.linspace(0.1, 0.9, 20)
    step = 1.0e-6
    numerical = (
        refracted_aqual_free_function(X, sigma + step, activation=1.2, eta=0.5)
        - refracted_aqual_free_function(X, sigma - step, activation=1.2, eta=0.5)
    ) / (2.0 * step)
    analytic = refracted_aqual_free_function_sigma_derivative(
        X, sigma, activation=1.2, eta=0.5
    )
    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-7, atol=2.0e-8)


def test_zero_backreaction_recovers_uncoupled_vacuum_field():
    faces = geometric_radial_faces(100, 0.01, 20.0)
    radius = radial_cell_centers(faces)
    result = solve_coupled_spherical_sigma(
        faces,
        np.zeros_like(radius),
        np.full_like(radius, 1.0e-12),
        rho_s_g_cm3=1.0,
        length_kpc=1.0,
        a0_m_s2=1.2e-10,
        eta=0.6,
        backreaction=0.0,
        initial_sigma=np.ones_like(radius),
    )
    assert result.converged
    np.testing.assert_allclose(result.field, 1.0, atol=2.0e-5)


def test_uniform_broken_vacuum_has_no_relative_scalar_energy():
    radius = np.geomspace(0.1, 100.0, 80)
    density, mass, acceleration = scalar_field_stress_energy_profile(
        radius,
        np.ones_like(radius),
        length_kpc=3.0,
        a0_m_s2=1.2e-10,
        backreaction=1.0e-3,
    )
    np.testing.assert_allclose(density, 0.0, atol=3.0e-53)
    np.testing.assert_allclose(mass, 0.0, atol=3.0e20)
    np.testing.assert_allclose(acceleration, 0.0, atol=1.0e-21)


def test_scalar_energy_mass_scales_inverse_with_backreaction():
    radius = np.geomspace(0.1, 100.0, 80)
    sigma = 1.0 - np.exp(-radius / 5.0)
    _, mass_one, _ = scalar_field_stress_energy_profile(
        radius,
        sigma,
        length_kpc=3.0,
        a0_m_s2=1.2e-10,
        backreaction=1.0e-3,
    )
    _, mass_two, _ = scalar_field_stress_energy_profile(
        radius,
        sigma,
        length_kpc=3.0,
        a0_m_s2=1.2e-10,
        backreaction=2.0e-3,
    )
    np.testing.assert_allclose(mass_one, 2.0 * mass_two, rtol=2.0e-14)
    assert np.all(np.diff(mass_one) >= 0.0)
