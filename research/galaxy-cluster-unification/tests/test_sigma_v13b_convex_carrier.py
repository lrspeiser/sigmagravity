from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v13b_convex_carrier import (
    ConvexCarrierParameters,
    carrier_characteristic_speeds,
    carrier_hamiltonian_density,
    carrier_legendre_state,
    carrier_phase_space_hessian,
    carrier_radial_curvature,
    carrier_response_mu,
    carrier_shape,
    numerical_flux_jacobian,
)


def test_convex_shape_has_aqual_limits_and_positive_floor() -> None:
    epsilon = 1.0e-6
    radius = np.geomspace(1.0e-9, 1.0e9, 2000)
    shape = carrier_shape(radius, epsilon=epsilon)
    mu = carrier_response_mu(radius, epsilon=epsilon)
    radial = carrier_radial_curvature(radius, epsilon=epsilon)
    assert np.all(shape >= 0.0)
    assert np.all(mu >= epsilon)
    assert np.all(radial >= epsilon)
    assert np.all(mu < 1.0)
    assert np.all(radial <= 1.0)
    assert carrier_response_mu(0.0, epsilon=epsilon) == pytest.approx(epsilon)
    assert carrier_response_mu(1.0, epsilon=epsilon) == pytest.approx(
        0.5 * (1.0 + epsilon)
    )


def test_phase_space_hessian_is_strictly_convex_and_bounded() -> None:
    params = ConvexCarrierParameters(epsilon=1.0e-6)
    phase = np.asarray([0.7, -1.2, 0.4, 2.1])
    hessian = carrier_phase_space_hessian(phase[0], phase[1:], parameters=params)
    eigenvalues = np.linalg.eigvalsh(hessian)
    ratio = np.linalg.norm(phase) / params.acceleration_scale
    assert eigenvalues[0] == pytest.approx(
        carrier_response_mu(ratio, epsilon=params.epsilon)
    )
    assert eigenvalues[-1] == pytest.approx(
        carrier_radial_curvature(ratio, epsilon=params.epsilon)
    )
    assert np.all(eigenvalues > 0.0)
    assert np.all(eigenvalues < 1.0)


def test_analytic_hessian_matches_independent_flux_jacobian() -> None:
    params = ConvexCarrierParameters(epsilon=1.0e-4)
    analytic = carrier_phase_space_hessian(
        0.7,
        (-1.2, 0.4, 2.1),
        parameters=params,
    )
    numeric = numerical_flux_jacobian(
        0.7,
        (-1.2, 0.4, 2.1),
        parameters=params,
    )
    assert np.max(np.abs(analytic - numeric)) < 1.0e-9


def test_static_slice_is_exact_aqual_legendre_state() -> None:
    params = ConvexCarrierParameters(epsilon=1.0e-6)
    gradient = (0.2, 0.0, 0.0)
    state = carrier_legendre_state(0.0, gradient, parameters=params)
    hamiltonian = carrier_hamiltonian_density(0.0, gradient, parameters=params)
    assert state["momentum"] == pytest.approx(0.0)
    assert state["lagrangian_density"] == pytest.approx(-hamiltonian)
    assert state["momentum_map_residual"] == pytest.approx(0.0)


def test_legendre_map_is_unique_on_large_signed_velocities() -> None:
    params = ConvexCarrierParameters(epsilon=1.0e-6)
    for velocity in (-100.0, -1.0, -0.01, 0.0, 0.01, 1.0, 100.0):
        state = carrier_legendre_state(
            velocity,
            (0.7, -0.2, 0.1),
            parameters=params,
        )
        assert abs(state["momentum_map_residual"]) < 1.0e-10
        assert abs(state["legendre_reconstruction_residual"]) < 1.0e-12
        assert state["hamiltonian_momentum_curvature"] >= params.epsilon
        assert state["lagrangian_time_kinetic_curvature"] > 0.0


def test_arbitrary_background_characteristics_remain_causal() -> None:
    params = ConvexCarrierParameters(epsilon=1.0e-6)
    rng = np.random.default_rng(13131)
    maximum_speed = 0.0
    for _ in range(1000):
        phase = rng.normal(size=4) * 10.0 ** rng.uniform(-6.0, 6.0)
        direction = rng.normal(size=3)
        row = carrier_characteristic_speeds(
            phase[0],
            phase[1:],
            direction,
            parameters=params,
        )
        maximum_speed = max(
            maximum_speed,
            float(row["maximum_absolute_characteristic_speed"]),
        )
        assert row["hyperbolic"]
        assert row["causal_in_preferred_unit_cone"]
        assert float(row["maximum_absolute_characteristic_speed"]) <= (
            float(row["largest_relevant_hessian_eigenvalue"]) + 1.0e-12
        )
    assert maximum_speed < 1.0


def test_convex_carrier_has_no_linear_charge_energy_near_origin() -> None:
    params = ConvexCarrierParameters(epsilon=1.0e-3)
    small = 1.0e-6
    positive = carrier_hamiltonian_density(
        small,
        (0.0, 0.0, 0.0),
        parameters=params,
    )
    negative = carrier_hamiltonian_density(
        -small,
        (0.0, 0.0, 0.0),
        parameters=params,
    )
    expected = 0.5 * params.epsilon * small**2
    assert positive == pytest.approx(negative, rel=1.0e-12)
    assert positive == pytest.approx(expected, rel=1.0e-3)


def test_invalid_convex_carrier_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        ConvexCarrierParameters(epsilon=0.0).validated()
    with pytest.raises(ValueError):
        carrier_shape(-1.0, epsilon=1.0e-6)
    with pytest.raises(ValueError):
        carrier_characteristic_speeds(
            0.0,
            (1.0, 0.0, 0.0),
            (0.0, 0.0),
        )
