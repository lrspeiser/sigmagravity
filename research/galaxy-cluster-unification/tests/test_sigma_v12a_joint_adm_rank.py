from __future__ import annotations

import numpy as np

from voidscreen.sigma_v12a_joint_adm_rank import (
    audit_v12a_joint_adm_rank,
    degenerate_dhost_kinetic_matrix,
    dewit_signature_block,
    finite_difference_hessian,
    joint_aest_dhost_hessian,
    matrix_inertia,
)


def test_dhost_schur_identity_has_one_null_direction() -> None:
    rng = np.random.default_rng(12)
    metric = dewit_signature_block(rng)
    dhost, null = degenerate_dhost_kinetic_matrix(metric, rng.normal(size=6))
    assert np.linalg.norm(dhost @ null) < 1.0e-12
    assert matrix_inertia(dhost) == (1, 1, 5)


def test_positive_aether_block_preserves_null_and_adds_three_positive_modes() -> None:
    rng = np.random.default_rng(13)
    metric = dewit_signature_block(rng)
    dhost, null = degenerate_dhost_kinetic_matrix(metric, rng.normal(size=6))
    aether_metric = np.asarray(
        [
            [1.7, 0.2, -0.1],
            [0.2, 0.9, 0.15],
            [-0.1, 0.15, 1.2],
        ]
    )
    joint = joint_aest_dhost_hessian(
        dhost,
        k_b=1.0,
        aether_metric=aether_metric,
    )
    joint_null = np.concatenate((null, np.zeros(3)))
    assert np.linalg.norm(joint @ joint_null) < 1.0e-12
    assert matrix_inertia(joint) == (1, 1, 8)


def test_nonpositive_aether_metric_is_rejected() -> None:
    rng = np.random.default_rng(131)
    metric = dewit_signature_block(rng)
    dhost, _ = degenerate_dhost_kinetic_matrix(metric, rng.normal(size=6))
    with np.testing.assert_raises_regex(ValueError, "positive definite"):
        joint_aest_dhost_hessian(
            dhost,
            k_b=1.0,
            aether_metric=np.diag([1.0, 1.0, -1.0]),
        )


def test_linear_momentum_shift_does_not_change_velocity_hessian() -> None:
    rng = np.random.default_rng(14)
    metric = dewit_signature_block(rng)
    dhost, _ = degenerate_dhost_kinetic_matrix(metric, rng.normal(size=6))
    joint = joint_aest_dhost_hessian(dhost, k_b=1.0)
    numerical = finite_difference_hessian(joint, rng.normal(size=10), step=1.0e-4)
    assert np.max(np.abs(numerical - joint)) < 1.0e-6


def test_joint_rank_audit_passes_without_overclaiming_dirac_completion() -> None:
    report = audit_v12a_joint_adm_rank(
        k_b=1.0,
        random_trials=100,
        random_seed=12002,
        finite_difference_step=1.0e-4,
    )
    assert report["joint_kinetic_rank_subgate_pass"]
    assert report["random_audit"]["minimum_aether_block_eigenvalue"] > 0.0
    assert not report["constraint_reduced"]
    assert report["dhost_primary_constraint_kinematically_present"]
    assert not report["dhost_secondary_constraint_derived"]
    assert not report["complete_dirac_chain_derived"]
    assert not report["physical_degree_count_proven_unchanged"]
    assert not report["theory_viable"]
    assert not report["observational_data_accessed"]
