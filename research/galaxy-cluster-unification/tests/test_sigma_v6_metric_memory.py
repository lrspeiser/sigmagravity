from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v6_metric_memory import (
    bounded_tensor_coherence,
    detuned_massive_memory_step_response,
    detuned_static_tensor_transfer,
    hessian_power_regularities,
    hessian_response,
    metric_memory_activation,
    metric_memory_chi,
    nonlinear_superposition_residual,
    repeated_massless_memory_step_response,
    retarded_convolution,
    rotation_covariance_residual,
    static_trace_free_projector,
    time_symmetric_convolution,
    trace_free_symmetric,
    v6a_perturbation_action,
    v6b_metric_memory_chi,
    v6c_total_constitutive_coefficient,
    v6d_cubic_orientation_correction,
    v6d_deep_acceleration_enhancement,
    v6d_parallel_ellipticity_coefficient,
    v6d_total_constitutive_coefficient,
)


def test_retarded_response_has_no_pre_source_support() -> None:
    source = np.zeros(101)
    source[50] = 1.0
    kernel = np.exp(-np.arange(30) / 5.0)
    response = retarded_convolution(source, kernel)
    assert np.all(response[:50] == 0.0)
    assert response[50] == pytest.approx(1.0)


def test_time_symmetric_variation_has_advanced_support() -> None:
    source = np.zeros(101)
    source[50] = 1.0
    kernel = np.exp(-np.arange(30) / 5.0)
    response = time_symmetric_convolution(source, kernel)
    assert response[49] > 0.0
    assert response[51] == pytest.approx(response[49])


def test_zero_source_and_fixed_zero_state_give_zero_memory() -> None:
    response = retarded_convolution(np.zeros(100), np.ones(10))
    assert np.all(response == 0.0)


def test_trace_free_hessian_response_is_rotation_covariant() -> None:
    matrix = np.array([[2.0, 1.0, -0.5], [1.0, -1.0, 0.3], [-0.5, 0.3, 4.0]])
    angle = 0.73
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert abs(np.trace(trace_free_symmetric(matrix))) < 1.0e-14
    assert rotation_covariance_residual(matrix, rotation) < 1.0e-14


def test_nonlinear_hessian_response_changes_component_order() -> None:
    first = np.diag([2.0, -1.0, -1.0])
    second = np.array([[0.0, 1.2, 0.0], [1.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert nonlinear_superposition_residual(first, second) > 0.01
    assert np.linalg.norm(hessian_response(first)) < np.linalg.norm(first)


def test_frozen_activation_has_required_asymptotes() -> None:
    tiny = 1.0e-12
    assert metric_memory_activation(tiny) / tiny == pytest.approx(1.0, rel=1.1e-6)
    high = 1.0e10
    assert metric_memory_activation(high) / high < 1.0e-15


def test_metric_memory_invariant_is_nonnegative_and_label_free() -> None:
    value = metric_memory_chi([0.0, 1.0], [4.0, 9.0], 0.5)
    assert np.allclose(value, [1.0, 2.5])


def test_invalid_metric_memory_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        retarded_convolution([], [1.0])
    with pytest.raises(ValueError):
        trace_free_symmetric(np.eye(2))
    with pytest.raises(ValueError):
        hessian_response(np.eye(3), 0.0)
    with pytest.raises(ValueError):
        metric_memory_activation(-1.0)
    with pytest.raises(ValueError):
        metric_memory_chi(1.0, 1.0, -1.0)


def test_v6a_square_root_hessian_has_a_first_variation_cusp() -> None:
    step = 1.0e-10
    center = float(v6a_perturbation_action(0.0, 1.0, 4.0, 0.7))
    right = (float(v6a_perturbation_action(step, 1.0, 4.0, 0.7)) - center) / step
    left = (center - float(v6a_perturbation_action(-step, 1.0, 4.0, 0.7))) / step
    assert right == pytest.approx(1.4, rel=3.0e-5)
    assert left == pytest.approx(-1.4, rel=3.0e-5)
    assert right - left > 2.7


def test_hessian_power_regularities_identify_the_safe_nonlinear_range() -> None:
    assert not hessian_power_regularities(0.5)["first_variation_exists"]
    assert not hessian_power_regularities(0.75)["finite_quadratic_variation"]
    assert hessian_power_regularities(1.0)["spectrum_role"] == "changes_quadratic_operator"
    assert (
        hessian_power_regularities(1.5)["spectrum_role"]
        == "nonlinear_only_about_zero_background"
    )


def test_v6b_coherence_is_bounded_and_enters_at_fourth_order() -> None:
    amplitude = np.geomspace(1.0e-6, 1.0e-3, 200)
    scalar_chi = amplitude**2
    oriented_chi = v6b_metric_memory_chi(scalar_chi, amplitude**2, 2.0, 1.0)
    correction = oriented_chi - scalar_chi
    slope = np.polyfit(np.log(amplitude), np.log(correction), 1)[0]
    assert slope == pytest.approx(4.0, rel=2.0e-5)
    coherence = bounded_tensor_coherence([0.0, 1.0, 1.0e30], 1.0)
    assert coherence[0] == 0.0
    assert np.all((coherence >= 0.0) & (coherence <= 1.0))
    assert coherence[-1] == pytest.approx(1.0)


def test_twice_retarded_static_projector_has_no_wavenumber_growth() -> None:
    direction = np.array([1.0, 2.0, -3.0])
    baseline = static_trace_free_projector(direction)
    expected_norm = np.sqrt(2.0 / 3.0)
    assert np.linalg.norm(baseline) == pytest.approx(expected_norm)
    for scale in (1.0e-12, 1.0, 1.0e12):
        assert np.allclose(static_trace_free_projector(scale * direction), baseline)


def test_invalid_v6b_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        hessian_power_regularities(0.0)
    with pytest.raises(ValueError):
        bounded_tensor_coherence(-1.0, 1.0)
    with pytest.raises(ValueError):
        bounded_tensor_coherence(1.0, 0.0)
    with pytest.raises(ValueError):
        v6b_metric_memory_chi(1.0, 1.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        static_trace_free_projector([0.0, 0.0, 0.0])


def test_repeated_massless_memory_has_linear_secular_envelope() -> None:
    wavenumber = 1.7
    peak_indices = np.arange(10, 1010)
    peak_times = (np.pi / 2.0 + 2.0 * np.pi * peak_indices) / wavenumber
    response = np.abs(repeated_massless_memory_step_response(peak_times, wavenumber))
    late_slope = np.polyfit(peak_times[-500:], response[-500:], 1)[0]
    assert late_slope == pytest.approx(1.0 / (2.0 * wavenumber), rel=1.0e-3)
    assert response[-1] > 50.0 * response[0]


def test_detuned_massive_memory_is_bounded_without_resonance() -> None:
    wavenumber = 1.3
    memory_mass = 0.4
    time = np.linspace(0.0, 4000.0 * 2.0 * np.pi / wavenumber, 200_000)
    response = detuned_massive_memory_step_response(time, wavenumber, memory_mass)
    omega_squared = wavenumber**2 + memory_mass**2
    analytic_bound = 2.0 / omega_squared + 2.0 / memory_mass**2
    assert np.max(np.abs(response)) <= analytic_bound * (1.0 + 1.0e-12)
    early = np.max(np.abs(response[:50_000]))
    late = np.max(np.abs(response[-50_000:]))
    assert late <= 1.01 * early


def test_detuned_static_transfer_is_bounded_and_has_no_uv_growth() -> None:
    wavenumber = np.geomspace(1.0e-12, 1.0e12, 1000)
    transfer = detuned_static_tensor_transfer(wavenumber, 2.0)
    assert np.all((transfer >= 0.0) & (transfer <= 1.0))
    assert np.all(np.diff(transfer) >= -1.0e-15)
    assert transfer[0] < 1.0e-20
    assert transfer[-1] == pytest.approx(1.0)


def test_invalid_detuned_memory_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        repeated_massless_memory_step_response(-1.0, 1.0)
    with pytest.raises(ValueError):
        repeated_massless_memory_step_response(1.0, 0.0)
    with pytest.raises(ValueError):
        detuned_massive_memory_step_response(1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        detuned_static_tensor_transfer(-1.0, 1.0)


def test_v6c_inside_chi_placement_has_negative_deep_response() -> None:
    invariant = np.geomspace(1.0e-16, 1.0e-6, 100)
    for strength in (0.1, 0.5, 0.99):
        coefficient = v6c_total_constitutive_coefficient(invariant, strength)
        assert coefficient[0] == pytest.approx(-strength, abs=1.0e-7)
        assert np.any(coefficient < 0.0)


def test_v6d_preserves_quadratic_cancellation_and_positive_ellipticity() -> None:
    invariant = np.geomspace(1.0e-16, 1.0e16, 100_000)
    for strength in (0.0, 0.5, 0.9, 0.99, 0.999):
        mu = v6d_total_constitutive_coefficient(invariant, strength)
        parallel = v6d_parallel_ellipticity_coefficient(invariant, strength)
        assert np.all(mu > 0.0)
        assert np.all(parallel > 0.0)
        expected_deep_slope = 1.5 * (1.0 - strength)
        assert mu[0] / np.sqrt(invariant[0]) == pytest.approx(
            expected_deep_slope, rel=2.0e-5, abs=2.0e-8
        )


def test_v6d_orientation_changes_only_cubic_and_higher_terms() -> None:
    invariant = np.geomspace(1.0e-10, 1.0e-5, 500)
    base = v6d_cubic_orientation_correction(invariant, 0.0)
    oriented = v6d_cubic_orientation_correction(invariant, 0.8)
    difference = oriented - base
    slope = np.polyfit(np.log(invariant), np.log(difference), 1)[0]
    assert slope == pytest.approx(1.5, rel=1.0e-3)
    assert v6d_deep_acceleration_enhancement(0.99) == pytest.approx(10.0)


def test_invalid_v6c_v6d_constitutive_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        v6c_total_constitutive_coefficient(-1.0, 1.0)
    with pytest.raises(ValueError):
        v6d_cubic_orientation_correction(1.0, -1.0)
    with pytest.raises(ValueError):
        v6d_total_constitutive_coefficient(1.0, 1.1)
    with pytest.raises(ValueError):
        v6d_parallel_ellipticity_coefficient(0.0, 0.5)
    with pytest.raises(ValueError):
        v6d_deep_acceleration_enhancement(1.0)
