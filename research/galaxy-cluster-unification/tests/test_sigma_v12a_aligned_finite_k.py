from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v12a_aligned_finite_k import (
    activation_weight_bound,
    aligned_hessian_invariants,
    audit_v12a_aligned_finite_k,
    critical_wave_number_ratio,
    negative_strength_sufficient_condition,
    normalized_aligned_coefficients,
    normalized_primary_secondary_symbol,
)


def test_aligned_hessian_invariants_restore_clock_gradient() -> None:
    row = aligned_hessian_invariants(
        2.0,
        -0.4,
        0.7,
        np.asarray([0.3, -0.2, 0.5]),
    )
    assert row["L3"] == pytest.approx(-(2.0**2) * 0.4**2 + 2.0**3 * 0.7 * 0.4)
    assert row["L4"] == pytest.approx(-(2.0**2) * 0.4**2 + 2.0**2 * (0.3**2 + 0.2**2 + 0.5**2))
    assert row["L5"] == pytest.approx(2.0**4 * 0.4**2)


def test_aligned_reduction_matches_direct_four_tensor_contractions() -> None:
    q_clock = 1.7
    v_star = -0.35
    clock_gradient = np.asarray([0.2, -0.4, 0.1])
    extrinsic_curvature = np.diag([0.1, -0.3, 0.6])
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    scalar_covector = np.asarray([q_clock, 0.0, 0.0, 0.0])
    scalar_vector = metric @ scalar_covector
    hessian = np.zeros((4, 4))
    hessian[0, 0] = v_star
    hessian[0, 1:] = clock_gradient
    hessian[1:, 0] = clock_gradient
    hessian[1:, 1:] = -q_clock * extrinsic_curvature
    raised_hessian = metric @ hessian @ metric

    box = float(np.einsum("mn,mn", metric, hessian))
    projected_hessian = float(np.einsum("m,mn,n", scalar_vector, hessian, scalar_vector))
    direct = {
        "L3": box * projected_hessian,
        "L4": float(
            np.einsum(
                "m,mr,rn,n",
                scalar_vector,
                hessian,
                raised_hessian,
                scalar_covector,
            )
        ),
        "L5": projected_hessian**2,
    }
    reduced = aligned_hessian_invariants(
        q_clock,
        v_star,
        float(np.trace(extrinsic_curvature)),
        clock_gradient,
    )
    for invariant in ("L3", "L4", "L5"):
        assert reduced[invariant] == pytest.approx(direct[invariant])


def test_positive_selected_sign_has_exact_finite_k_counterexample() -> None:
    coefficients = normalized_aligned_coefficients(
        2.0,
        background_clock_ratio=1.0,
        orientation_strength=1.0,
    )
    assert coefficients["activation"] == pytest.approx(0.06902684899626334)
    assert coefficients["A4_bar"] == pytest.approx(-0.07855626076096922)
    wave = critical_wave_number_ratio(
        2.0,
        background_clock_ratio=1.0,
        orientation_strength=1.0,
        k_2=2.0,
    )
    assert wave == pytest.approx(3.567874736625391)
    row = normalized_primary_secondary_symbol(
        2.0,
        wave,
        background_clock_ratio=1.0,
        orientation_strength=1.0,
        k_2=2.0,
    )
    assert row["positive_core"] == pytest.approx(0.0, abs=1.0e-12)


def test_negative_strength_interval_is_analytic_and_nontrivial() -> None:
    condition = negative_strength_sufficient_condition(
        orientation_strength=-1.0,
        background_kinetic_ratio=-1.0,
    )
    assert condition["minimum_orientation_strength"] == pytest.approx(-4.0 * np.sqrt(2.0))
    assert condition["sufficient_condition_satisfied"]
    assert condition["interaction_nonzero"]


def test_activation_weight_obeys_global_bound() -> None:
    for x_value in np.linspace(-100.0, 100.0, 4001):
        row = activation_weight_bound(
            float(x_value),
            background_kinetic_ratio=-1.0,
        )
        assert row["bound_satisfied"]
        assert row["weighted_activation"] <= np.sqrt(2.0) + 1.0e-12


def test_negative_sentinel_has_nonnegative_A4_and_no_aligned_root() -> None:
    for clock in np.linspace(-20.0, 20.0, 4001):
        coefficients = normalized_aligned_coefficients(
            float(clock),
            background_clock_ratio=1.0,
            orientation_strength=-1.0,
        )
        assert coefficients["A4_bar"] >= -1.0e-14
        assert (
            critical_wave_number_ratio(
                float(clock),
                background_clock_ratio=1.0,
                orientation_strength=-1.0,
                k_2=2.0,
            )
            is None
        )


def test_aligned_finite_k_audit_falsifies_only_positive_row() -> None:
    report = audit_v12a_aligned_finite_k(
        k_2=2.0,
        background_clock_ratio=1.0,
        selected_positive_strength=1.0,
        surviving_negative_strength=-1.0,
        counterexample_clock_ratio=2.0,
        random_trials=2000,
        logarithmic_clock_limit=3.0,
        logarithmic_wave_limit=3.0,
        random_seed=12007,
    )
    assert all(report["gates"].values())
    assert not report["selected_positive_row_survives"]
    assert report["negative_branch_aligned_finite_k_regular"]
    assert not report["full_tilted_anisotropic_symbol_derived"]
    assert not report["complete_delta_eff_proven_invertible"]
    assert not report["physical_degree_count_proven_unchanged"]
    assert not report["theory_viable"]
    assert not report["observational_data_accessed"]
    assert not report["raw_holdout_opened"]


def test_invalid_wave_and_sign_protocol_are_rejected() -> None:
    with pytest.raises(ValueError):
        normalized_primary_secondary_symbol(
            1.0,
            -1.0,
            background_clock_ratio=1.0,
            orientation_strength=-1.0,
            k_2=2.0,
        )
    with pytest.raises(ValueError):
        audit_v12a_aligned_finite_k(
            k_2=2.0,
            background_clock_ratio=1.0,
            selected_positive_strength=-1.0,
            surviving_negative_strength=-1.0,
            counterexample_clock_ratio=2.0,
            random_trials=10,
            logarithmic_clock_limit=1.0,
            logarithmic_wave_limit=1.0,
            random_seed=1,
        )
