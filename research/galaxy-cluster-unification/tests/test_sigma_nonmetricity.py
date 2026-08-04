from __future__ import annotations

import numpy as np

from voidscreen.sigma_nonmetricity import (
    dimensionless_action_invariant,
    nonminimal_scalar_weak_laplacians,
    regular_isolated_branch_has_zero_slip,
    slip_nonmetricity,
    standard_action_primitive,
    standard_mu,
    standard_mu_spherical_acceleration,
    stegr_nonmetricity,
    weyl_trace_nonmetricity,
)


def test_static_stegr_invariant_reduces_to_two_potential_form() -> None:
    rng = np.random.default_rng(7101)
    grad_psi = rng.normal(size=(1000, 3))
    grad_phi = rng.normal(size=(1000, 3))
    expected = 4.0 * np.sum(grad_psi * grad_phi, axis=1) - 2.0 * np.sum(
        grad_phi**2, axis=1
    )
    assert np.allclose(stegr_nonmetricity(grad_psi, grad_phi), expected, atol=2e-14)


def test_independent_quadratic_combination_is_slip_gradient() -> None:
    rng = np.random.default_rng(7102)
    grad_psi = rng.normal(size=(1000, 3))
    grad_phi = rng.normal(size=(1000, 3))
    expected = np.sum((grad_psi - grad_phi) ** 2, axis=1)
    assert np.allclose(slip_nonmetricity(grad_psi, grad_phi), expected, atol=2e-14)


def test_equal_potential_action_invariant_is_acceleration_squared() -> None:
    acceleration_scale = 1.2e-10
    gradient = np.array([[1.0, 2.0, -3.0], [-4.0, 0.5, 2.0]]) * acceleration_scale
    expected = np.sum(gradient**2, axis=1) / acceleration_scale**2
    assert np.allclose(
        dimensionless_action_invariant(gradient, gradient, acceleration_scale), expected
    )


def test_action_derivative_is_standard_mu() -> None:
    x = np.geomspace(1e-8, 1e8, 2000)
    step = 1e-5
    derivative = (
        standard_action_primitive(x * np.exp(step))
        - standard_action_primitive(x * np.exp(-step))
    ) / (2.0 * step * x)
    assert np.allclose(derivative, standard_mu(x), rtol=2e-5, atol=2e-8)


def test_spherical_solution_round_trips_and_has_required_limits() -> None:
    acceleration_scale = 1.2e-10
    gbar = acceleration_scale * np.geomspace(1e-6, 1e8, 1000)
    gravity = standard_mu_spherical_acceleration(gbar, acceleration_scale)
    reconstructed = standard_mu((gravity / acceleration_scale) ** 2) * gravity
    assert np.allclose(reconstructed, gbar, rtol=2e-12)
    assert abs(gravity[0] / np.sqrt(gbar[0] * acceleration_scale) - 1.0) < 0.001
    assert gravity[-1] / gbar[-1] - 1.0 < 1e-12


def test_positive_elliptic_branch_has_zero_slip() -> None:
    assert regular_isolated_branch_has_zero_slip(1e-12)
    assert not regular_isolated_branch_has_zero_slip(0.0)


def test_weyl_trace_invariant_selects_sum_of_metric_potentials() -> None:
    rng = np.random.default_rng(7103)
    grad_psi = rng.normal(size=(1000, 3))
    grad_phi = rng.normal(size=(1000, 3))
    expected = 4.0 * np.sum(np.square(grad_psi + grad_phi), axis=1)
    assert np.allclose(
        weyl_trace_nonmetricity(grad_psi, grad_phi), expected, atol=8e-14
    )


def test_plain_nonminimal_scalar_cancels_from_linear_weyl_laplacian() -> None:
    baryonic = np.array([0.5, 2.0, 7.0])
    scalar = np.array([-3.0, 1.5, 9.0])
    response = nonminimal_scalar_weak_laplacians(baryonic, scalar)
    assert np.array_equal(response["photon_weyl"], baryonic)
    assert np.array_equal(response["spatial_phi"] - response["matter_psi"], scalar)
