from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v6_metric_memory import (
    hessian_response,
    metric_memory_activation,
    metric_memory_chi,
    nonlinear_superposition_residual,
    retarded_convolution,
    rotation_covariance_residual,
    time_symmetric_convolution,
    trace_free_symmetric,
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
