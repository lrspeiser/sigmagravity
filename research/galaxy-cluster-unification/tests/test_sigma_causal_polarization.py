from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_causal_polarization import (
    bounded_disformal_fraction,
    flrw_polarization_mode,
    local_transport_eigenvalues,
    maximum_characteristic_speed,
    minimum_static_operator_eigenvalue,
    signed_trace_bandpass,
    static_reduced_kinetic_hessian,
    static_reduced_velocity_lagrangian,
    transition_bandpass,
    transition_bandpass_y_derivative,
    weak_transport_gradient_contraction,
    weak_transport_tensor,
)


def test_transition_bandpass_has_frozen_limits_and_peak() -> None:
    ratio = np.array([0.0, 1.0e-5, 1.0, 1.0e5])
    source = transition_bandpass(ratio)
    assert source[0] == 0.0
    assert source[1] == pytest.approx(1.0e-20, rel=1.0e-12)
    assert source[2] == 0.25
    assert source[3] == pytest.approx(1.0e-20, rel=1.0e-12)


def test_disformal_fraction_is_bounded_for_all_field_strengths() -> None:
    ratio = np.geomspace(1.0e-12, 1.0e12, 1000)
    for alpha in (0.0, 0.1, 1.0, 10.0, 1.0e6):
        fraction = bounded_disformal_fraction(ratio, alpha)
        assert np.all(fraction >= 0.0)
        assert np.all(fraction < 1.0)
        assert np.max(fraction) <= alpha / (1.0 + alpha) + 1.0e-15


def test_local_scalar_cones_are_lorentzian_and_not_superluminal() -> None:
    ratio = np.geomspace(1.0e-12, 1.0e12, 1000)
    for orientation in ("spacelike", "timelike"):
        eigenvalues = local_transport_eigenvalues(
            ratio, 3.0, orientation=orientation
        )
        assert np.all(eigenvalues["time_magnitude"] > 0.0)
        assert np.all(eigenvalues["parallel_spatial"] > 0.0)
        assert np.all(eigenvalues["transverse_spatial"] > 0.0)
        speed = maximum_characteristic_speed(ratio, 3.0, orientation=orientation)
        assert np.all(speed > 0.0)
        assert np.all(speed <= 1.0)


def test_static_massive_branch_is_unique() -> None:
    ratio = np.geomspace(1.0e-12, 1.0e12, 1000)
    eigenvalue = minimum_static_operator_eigenvalue(ratio, 20.0)
    assert np.all(eigenvalue == 1.0)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        transition_bandpass([-1.0])
    with pytest.raises(ValueError):
        bounded_disformal_fraction([1.0], -0.1)
    with pytest.raises(ValueError):
        local_transport_eigenvalues([1.0], 1.0, orientation="null")


def test_transition_bandpass_y_derivative_matches_finite_difference() -> None:
    y = np.geomspace(1.0e-5, 1.0e5, 1000)
    step = 1.0e-6
    plus = np.square(y * np.exp(step)) / np.square(
        1.0 + np.square(y * np.exp(step))
    )
    minus = np.square(y * np.exp(-step)) / np.square(
        1.0 + np.square(y * np.exp(-step))
    )
    finite = (plus - minus) / (2.0 * step * y)
    assert np.allclose(
        finite, transition_bandpass_y_derivative(y), rtol=2.0e-8, atol=1.0e-12
    )


def test_signed_trace_source_is_real_even_and_smooth_through_zero() -> None:
    value = np.linspace(-2.0, 2.0, 1001)
    source = signed_trace_bandpass(value)
    assert np.all(np.isfinite(source))
    assert np.allclose(source, signed_trace_bandpass(-value))
    assert signed_trace_bandpass(0.0) == 0.0
    assert transition_bandpass_y_derivative(0.0) == 0.0


def test_weak_transport_chain_rule_matches_directional_difference() -> None:
    rng = np.random.default_rng(7501)
    vector = rng.normal(size=(200, 3))
    polarization_gradient = rng.normal(size=(200, 3))
    direction = rng.normal(size=(200, 3))
    direction /= np.sqrt(np.mean(np.square(direction)))
    alpha = 2.7
    step = 1.0e-6

    def energy(argument: np.ndarray) -> float:
        tensor = weak_transport_tensor(argument, alpha)
        return float(
            np.sum(
                np.einsum(
                    "...i,...ij,...j->...",
                    polarization_gradient,
                    tensor,
                    polarization_gradient,
                )
            )
        )

    finite = (energy(vector + step * direction) - energy(vector - step * direction)) / (
        2.0 * step
    )
    analytic_vector = weak_transport_gradient_contraction(
        vector, polarization_gradient, alpha
    )
    analytic = float(np.sum(analytic_vector * direction))
    assert abs(finite - analytic) / max(abs(finite), abs(analytic), 1.0e-15) < 1.0e-8


def test_v5b_flrw_scalar_mode_is_healthy_and_subluminal() -> None:
    hubble_ratio = np.geomspace(1.0e-12, 1.0e12, 1000)
    mode = flrw_polarization_mode(hubble_ratio, 10.0)
    assert np.all(mode["time_kinetic"] >= 1.0)
    assert np.all(mode["spatial_gradient"] == 1.0)
    assert np.all(mode["sound_speed_squared"] > 0.0)
    assert np.all(mode["sound_speed_squared"] <= 1.0)
    assert np.all(mode["mass_squared_times_L_squared"] > 0.0)


def test_static_reduced_hessian_matches_centered_finite_difference() -> None:
    weyl = np.array([4.0, 1.0, 0.5])
    gradient = np.array([0.7, -0.2, 0.4])
    trace = np.array([1.4, 0.5, -0.3])
    analytic = static_reduced_kinetic_hessian(
        weyl, gradient, trace, 0.2, 2.0
    )
    step = 1.0e-4
    origin = np.zeros(3)
    finite = np.empty((3, 3))
    base = static_reduced_velocity_lagrangian(
        origin, weyl, gradient, trace, 0.2, 2.0
    )
    for row in range(3):
        row_step = np.zeros(3)
        row_step[row] = step
        finite[row, row] = (
            static_reduced_velocity_lagrangian(
                origin + row_step, weyl, gradient, trace, 0.2, 2.0
            )
            - 2.0 * base
            + static_reduced_velocity_lagrangian(
                origin - row_step, weyl, gradient, trace, 0.2, 2.0
            )
        ) / step**2
        for column in range(row):
            column_step = np.zeros(3)
            column_step[column] = step
            finite[row, column] = finite[column, row] = (
                static_reduced_velocity_lagrangian(
                    origin + row_step + column_step,
                    weyl,
                    gradient,
                    trace,
                    0.2,
                    2.0,
                )
                - static_reduced_velocity_lagrangian(
                    origin + row_step - column_step,
                    weyl,
                    gradient,
                    trace,
                    0.2,
                    2.0,
                )
                - static_reduced_velocity_lagrangian(
                    origin - row_step + column_step,
                    weyl,
                    gradient,
                    trace,
                    0.2,
                    2.0,
                )
                + static_reduced_velocity_lagrangian(
                    origin - row_step - column_step,
                    weyl,
                    gradient,
                    trace,
                    0.2,
                    2.0,
                )
            ) / (4.0 * step**2)
    assert np.allclose(finite, analytic, rtol=2.0e-6, atol=2.0e-7)


def test_v5b_static_kinetic_matrix_loses_the_stegr_lapse_null() -> None:
    control = static_reduced_kinetic_hessian(
        np.zeros(3), np.zeros(3), np.zeros(3), 0.0, 0.0
    )
    assert np.linalg.matrix_rank(control, tol=1.0e-10) == 2
    assert np.linalg.det(control) == 0.0

    source_only = static_reduced_kinetic_hessian(
        np.zeros(3),
        np.zeros(3),
        np.array([np.sqrt(2.0), 0.0, 0.0]),
        0.2,
        0.0,
    )
    assert np.linalg.matrix_rank(source_only, tol=1.0e-10) == 3
    assert source_only[0, 0] < 0.0

    transport_only = static_reduced_kinetic_hessian(
        np.array([4.0, 1.0, 0.5]),
        np.array([0.7, -0.2, 0.4]),
        np.zeros(3),
        0.0,
        2.0,
    )
    assert np.linalg.matrix_rank(transport_only, tol=1.0e-10) == 3
    assert abs(np.linalg.det(transport_only)) > 1.0e-3
