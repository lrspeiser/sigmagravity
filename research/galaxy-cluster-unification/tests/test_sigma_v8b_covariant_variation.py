from __future__ import annotations

import itertools

import numpy as np
import pytest

from voidscreen.sigma_v8b_covariant_variation import (
    audit_v8b_covariant_subgate,
    audit_v8b_metric_noether_subgate,
    completion_metric_algebraic_euler_density,
    completion_minisuperspace_velocity_hessian,
    completion_point_lagrangian_density,
    completion_vector_euler_derivative,
    flrw_clock_kinetic_coefficient,
    flrw_stability_limit,
    scalar_principal_third_derivative_cancellation,
)


def symmetric_rank_three(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(4, 4, 4))
    return sum(np.transpose(raw, axes) for axes in itertools.permutations(range(3))) / 6.0


def test_covariant_scalar_principal_third_derivatives_cancel() -> None:
    result = scalar_principal_third_derivative_cancellation(
        np.array([1.0, 0.2, -0.1, 0.3]),
        np.array(
            [
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.2, 0.0],
                [0.0, 0.2, 0.8, 0.1],
                [0.0, 0.0, 0.1, 1.2],
            ]
        ),
        symmetric_rank_three(),
        clock_displacement=0.4,
    )
    assert result.first_divergence_term == pytest.approx(
        -result.second_divergence_term
    )
    assert result.sum == pytest.approx(0.0, abs=1.0e-14)


def test_vector_euler_derivative_vanishes_on_q_equals_q0_background() -> None:
    result = completion_vector_euler_derivative(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.diag([0.0, 1.0, 1.0, 1.0]),
        np.array([0.5, 0.1, 0.2, 0.3]),
        np.eye(4),
        clock_displacement=0.0,
    )
    assert np.allclose(result, 0.0)


def test_selected_flrw_clock_bound_is_twelve_sevenths() -> None:
    alpha = 16.0 / 9.0
    limit = flrw_stability_limit(k_2=2.0, alpha=alpha)
    assert limit == pytest.approx(12.0 / 7.0)
    stable = flrw_clock_kinetic_coefficient(
        k_2=2.0,
        alpha=alpha,
        horndeski_length=1.0,
        hubble_inverse_length=0.99 * limit,
        q_0_inverse_length=1.0,
    )
    unstable = flrw_clock_kinetic_coefficient(
        k_2=2.0,
        alpha=alpha,
        horndeski_length=1.0,
        hubble_inverse_length=1.01 * limit,
        q_0_inverse_length=1.0,
    )
    assert stable.stable
    assert stable.total_coefficient > 0.0
    assert not unstable.stable
    assert unstable.total_coefficient < 0.0


def test_minisuperspace_hessian_has_no_mixing_at_q0() -> None:
    result = completion_minisuperspace_velocity_hessian(
        scale_factor=2.0,
        lapse=1.5,
        scale_factor_velocity=0.2,
        scalar_velocity=0.75,
        q_0_inverse_length=0.5,
        coefficient=0.7,
    )
    assert result[0, 0] == pytest.approx(0.0)
    assert result[0, 1] == pytest.approx(0.0)
    assert result[1, 0] == pytest.approx(0.0)
    assert result[1, 1] < 0.0


def test_v8b_covariant_subgate_passes_only_as_necessary_gate() -> None:
    audit = audit_v8b_covariant_subgate(
        k_2=2.0,
        alpha=16.0 / 9.0,
        physical_parameter_count=5,
        maximum_physical_parameters=5,
    )
    assert all(audit["gates"].values())
    assert audit["selected_flrw_stability_limit_LH2_H_Q0"] == pytest.approx(
        12.0 / 7.0
    )


def test_invalid_covariant_gate_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        flrw_stability_limit(k_2=0.0, alpha=16.0 / 9.0)
    with pytest.raises(ValueError):
        flrw_clock_kinetic_coefficient(
            k_2=2.0,
            alpha=16.0 / 9.0,
            horndeski_length=-1.0,
            hubble_inverse_length=1.0,
            q_0_inverse_length=1.0,
        )


def test_metric_euler_density_is_symmetric_and_vanishes_at_q0() -> None:
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    vector = np.array([1.0, 0.0, 0.0, 0.0])
    gradient = np.array([0.5, 0.1, 0.2, 0.3])
    hessian = np.eye(4)
    euler = completion_metric_algebraic_euler_density(
        metric,
        vector,
        gradient,
        hessian,
        q_0=0.5,
    )
    assert np.allclose(euler, euler.T)
    assert np.allclose(euler, 0.0)
    assert completion_point_lagrangian_density(
        metric,
        vector,
        gradient,
        hessian,
        q_0=0.5,
    ) == pytest.approx(0.0)


def test_metric_noether_subgate_finite_difference() -> None:
    audit = audit_v8b_metric_noether_subgate()
    assert all(audit["gates"].values())
    assert audit["metric_directional_variation_relative_error"] < 1.0e-9
