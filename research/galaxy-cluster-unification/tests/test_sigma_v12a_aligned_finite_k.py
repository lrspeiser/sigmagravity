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
    normalized_full_aligned_symbol,
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


def test_positive_selected_sign_clock_only_block_has_superseded_zero() -> None:
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
    assert not row["projection_complete"]


def test_full_null_projection_cancels_dhost_gradient_exactly() -> None:
    row = normalized_full_aligned_symbol(
        2.0,
        3.567874736625391,
        background_clock_ratio=1.0,
        orientation_strength=1.0,
        k_b=1.0,
        k_2=2.0,
    )
    terms = row["gradient_terms"]
    assert terms["dhost_clock"] == pytest.approx(-0.3142250430438769)
    assert terms["class_ia_sum"] == pytest.approx(0.0, abs=1.0e-15)
    assert terms["aest_maxwell"] == pytest.approx(0.25)
    assert terms["full_null_projected"] == pytest.approx(0.25)
    assert row["positive_core"] > 8.0
    assert row["symbol_nonzero"]
    assert row["projection_complete_on_aligned_branch"]


def test_negative_strength_interval_controls_only_unprojected_A4() -> None:
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


def test_negative_sentinel_has_nonnegative_unprojected_A4_block() -> None:
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


def test_aligned_finite_k_audit_corrects_positive_falsification() -> None:
    report = audit_v12a_aligned_finite_k(
        k_b=1.0,
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
    assert not report["prior_positive_sign_falsification_valid"]
    assert report["selected_positive_row_survives_aligned_gate"]
    assert report["negative_row_survives_aligned_gate"]
    assert not report["orientation_sign_constrained_by_aligned_gate"]
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
            k_b=1.0,
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
