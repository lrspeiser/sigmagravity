from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v13b_convex_carrier import carrier_response_mu
from voidscreen.sigma_v13c_khronon_completion import (
    KhrononCompletionParameters,
    dimensionless_khronon_static_function,
    effective_adm_lambda,
    khronon_completion_row,
    scalar_shift_block,
    scalar_shift_reduced_kinetic_coefficient,
    static_susceptibility,
    temporal_excess_curvature,
    traceless_tensor_modifier_contraction,
)


def test_static_khronon_function_reproduces_carrier_mu() -> None:
    epsilon = 1.0e-6
    ratio = np.geomspace(1.0e-6, 1.0e6, 1000)
    step = 1.0e-5 * ratio
    upper = dimensionless_khronon_static_function(
        ratio + step,
        epsilon=epsilon,
    )
    lower = dimensionless_khronon_static_function(
        ratio - step,
        epsilon=epsilon,
    )
    derivative = (upper - lower) / (2.0 * step)
    recovered = derivative / (2.0 * ratio)
    expected = static_susceptibility(ratio, epsilon=epsilon)
    assert np.max(np.abs(recovered - expected)) < 2.0e-7
    assert np.max(
        np.abs(
            expected
            - (carrier_response_mu(ratio, epsilon=epsilon) - 1.0)
        )
    ) < 1.0e-15


def test_temporal_curvature_identity_is_stable_at_high_field() -> None:
    params = KhrononCompletionParameters(epsilon=1.0e-6)
    ratio = np.geomspace(1.0e-8, 1.0e14, 1000)
    exact = temporal_excess_curvature(ratio, parameters=params)
    mu = carrier_response_mu(ratio, epsilon=params.epsilon)
    assert np.max(np.abs(exact - (1.0 / mu - 1.0)) / exact) < 0.02
    assert np.all(exact > 0.0)


def test_selected_completion_enters_high_field_ghost_interval() -> None:
    params = KhrononCompletionParameters(epsilon=1.0e-6)
    row = khronon_completion_row(1.0e5, parameters=params)
    assert 1.0 / 3.0 < row["adm_lambda"] < 1.0
    assert row["in_standard_ghost_interval"]
    assert row["analytic_reduced_kinetic_coefficient"] < 0.0
    assert not row["positive_reduced_scalar_kinetic"]


def test_deep_field_and_high_field_lie_on_opposite_kinetic_branches() -> None:
    params = KhrononCompletionParameters(epsilon=1.0e-6)
    deep = khronon_completion_row(0.0, parameters=params)
    high = khronon_completion_row(10.0, parameters=params)
    assert deep["adm_lambda"] < 1.0 / 3.0
    assert deep["positive_reduced_scalar_kinetic"]
    assert 1.0 / 3.0 < high["adm_lambda"] < 1.0
    assert not high["positive_reduced_scalar_kinetic"]


def test_direct_shift_elimination_matches_closed_form() -> None:
    for lam in (-10.0, 0.0, 0.5, 0.9, 1.1, 10.0):
        row = scalar_shift_block(lam)
        expected = float(scalar_shift_reduced_kinetic_coefficient(lam))
        assert row["direct_schur_kinetic_coefficient"] == pytest.approx(
            expected
        )
        assert abs(row["schur_identity_residual"]) < 1.0e-12


def test_every_positive_completion_weight_has_a_high_field_ghost() -> None:
    for weight in (1.0e-6, 1.0e-3, 1.0, 100.0, 1.0e6):
        params = KhrononCompletionParameters(
            epsilon=1.0e-6,
            completion_weight=weight,
        )
        ratio = max(1.0e5, 10.0 * weight)
        lam = float(effective_adm_lambda(ratio, parameters=params))
        assert 1.0 / 3.0 < lam < 1.0


def test_trace_completion_does_not_change_tt_quadratic_kinetic() -> None:
    plus = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    cross = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ) / np.sqrt(2.0)
    assert traceless_tensor_modifier_contraction(
        plus,
        trace_curvature=100.0,
    ) == pytest.approx(0.0)
    assert traceless_tensor_modifier_contraction(
        cross,
        trace_curvature=100.0,
    ) == pytest.approx(0.0)


def test_invalid_khronon_completion_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        KhrononCompletionParameters(completion_weight=0.0).validated()
    with pytest.raises(ValueError):
        temporal_excess_curvature(-1.0)
    with pytest.raises(ValueError):
        scalar_shift_block(float("nan"))
    with pytest.raises(ValueError):
        traceless_tensor_modifier_contraction(
            np.eye(2),
            trace_curvature=1.0,
        )
