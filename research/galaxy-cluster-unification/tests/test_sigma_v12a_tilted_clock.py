from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v12a_tilted_clock import (
    audit_v12a_tilted_clock,
    global_tilt_parameter_condition,
    simple_aest_interpolation_derivatives,
    tilted_aest_clock_susceptibility,
    tilted_clock_kinematics,
)


def test_tilted_clock_kinematic_identities() -> None:
    q_clock = 0.7
    gradient = np.asarray([0.3, -0.4, 0.2])
    aether = np.asarray([0.5, 0.1, -0.2])
    row = tilted_clock_kinematics(q_clock, gradient, aether)
    expected_q = np.sqrt(1.0 + aether @ aether) * q_clock + aether @ gradient
    assert row["Q"] == pytest.approx(expected_q)
    assert row["Y"] == pytest.approx(-(q_clock**2) + gradient @ gradient + expected_q**2)
    assert row["projected_cauchy_residual"] == pytest.approx(
        (aether @ aether) * (gradient @ gradient) - (aether @ gradient) ** 2
    )
    assert row["projected_cauchy_identity_relative_error"] < 1.0e-15


def test_exact_projected_axis_has_regular_interpolation_limit() -> None:
    aether = np.asarray([2.0, -0.5, 0.7])
    q_clock = 1.3
    chi = np.sqrt(1.0 + aether @ aether)
    gradient = -q_clock * aether / chi
    row = tilted_aest_clock_susceptibility(
        q_clock,
        gradient,
        aether,
        a_sigma=1.0,
        k_b=1.0,
        k_2=2.0,
    )
    assert row["Y"] == pytest.approx(0.0, abs=1.0e-12)
    assert row["interpolation_curvature_term"] == pytest.approx(0.0)
    assert row["clock_susceptibility"] == pytest.approx(8.0 + 6.0 * (aether @ aether))
    assert row["susceptibility_positive"]


def test_selected_parameter_row_has_global_positive_bound() -> None:
    condition = global_tilt_parameter_condition(k_b=1.0, k_2=2.0)
    assert condition["minimum_k2"] == pytest.approx(9.0 / 8.0)
    assert condition["tilt_squared_margin"] == pytest.approx(3.5)
    assert condition["globally_positive_sufficient_condition"]


def test_interpolation_curvature_obeys_analytic_half_aether_bound() -> None:
    rng = np.random.default_rng(1206)
    for _ in range(1000):
        aether = rng.normal(size=3)
        gradient = rng.normal(size=3)
        row = tilted_aest_clock_susceptibility(
            float(rng.normal()),
            gradient,
            aether,
            a_sigma=1.0,
            k_b=1.0,
            k_2=2.0,
        )
        assert row["interpolation_curvature_term"] <= 0.5 * (aether @ aether) + 1.0e-12
        assert row["bound_satisfied"]
        assert row["susceptibility_positive"]


def test_kinematics_remain_stable_on_large_projected_axis() -> None:
    aether = np.asarray([3.0e5, -4.0e5, 1.0e5])
    q_clock = -8.0e5
    chi = np.sqrt(1.0 + aether @ aether)
    gradient = -q_clock * aether / chi
    row = tilted_clock_kinematics(q_clock, gradient, aether)
    exact_axis_q = q_clock / chi
    assert row["Y"] <= 1.0e-18 * max(1.0, q_clock**2)
    assert row["Q"] == pytest.approx(exact_axis_q, rel=1.0e-12, abs=1.0e-12)
    assert row["projected_cauchy_residual"] >= 0.0
    assert row["projected_cauchy_identity_relative_error"] < 1.0e-12


def test_flat_clock_reproduces_homogeneous_bracket_magnitude() -> None:
    row = tilted_aest_clock_susceptibility(
        1.0,
        np.zeros(3),
        np.zeros(3),
        a_sigma=1.0,
        k_b=1.0,
        k_2=2.0,
    )
    assert row["clock_susceptibility"] == pytest.approx(8.0)
    assert row["analytic_lower_bound"] == pytest.approx(8.0)


def test_tilted_clock_audit_passes_without_claiming_full_delta_eff() -> None:
    report = audit_v12a_tilted_clock(
        a_sigma=1.0,
        k_b=1.0,
        k_2=2.0,
        random_trials=2000,
        logarithmic_amplitude_limit=3.0,
        random_seed=12006,
    )
    assert report["tilted_reduced_aest_susceptibility_globally_positive"]
    assert all(report["gates"].values())
    assert not report["dhost_spatial_operator_included"]
    assert not report["complete_delta_eff_proven_invertible"]
    assert not report["complete_dirac_chain_derived"]
    assert not report["physical_degree_count_proven_unchanged"]
    assert not report["theory_viable"]
    assert not report["observational_data_accessed"]


def test_invalid_interpolation_and_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        simple_aest_interpolation_derivatives(-1.0)
    with pytest.raises(ValueError):
        global_tilt_parameter_condition(k_b=2.0, k_2=2.0)
