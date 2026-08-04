from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v12a_joint_adm_rank import dewit_signature_block
from voidscreen.sigma_v12a_primary_dirac import (
    audit_v12a_primary_dirac,
    dhost_canonical_momenta,
    dhost_primary_constraint,
    full_dirac_matrix,
    primary_secondary_bracket_block,
    reduced_physical_dof,
)


def test_canonical_primary_is_velocity_independent_on_degenerate_row() -> None:
    rng = np.random.default_rng(1204)
    metric = dewit_signature_block(rng)
    mixing = rng.normal(size=6)
    scalar_coefficient = float(mixing @ np.linalg.solve(metric, mixing))
    metric_affine = rng.normal(size=6)
    scalar_affine = float(rng.normal())
    for _ in range(20):
        p_star, momentum = dhost_canonical_momenta(
            metric,
            mixing,
            scalar_velocity_coefficient=scalar_coefficient,
            scalar_velocity=float(rng.normal()),
            metric_velocity=rng.normal(size=6),
            scalar_affine=scalar_affine,
            metric_affine=metric_affine,
            sqrt_h=1.7,
        )
        primary = dhost_primary_constraint(
            p_star,
            momentum,
            metric,
            mixing,
            scalar_affine=scalar_affine,
            metric_affine=metric_affine,
            sqrt_h=1.7,
        )
        assert primary == pytest.approx(0.0, abs=1.0e-12)


def test_nondegenerate_scalar_coefficient_does_not_define_primary() -> None:
    rng = np.random.default_rng(1205)
    metric = dewit_signature_block(rng)
    mixing = rng.normal(size=6)
    degenerate = float(mixing @ np.linalg.solve(metric, mixing))
    residuals = []
    for velocity in (-2.0, 3.0):
        p_star, momentum = dhost_canonical_momenta(
            metric,
            mixing,
            scalar_velocity_coefficient=degenerate + 0.4,
            scalar_velocity=velocity,
            metric_velocity=np.zeros(6),
            scalar_affine=0.0,
            metric_affine=np.zeros(6),
            sqrt_h=1.0,
        )
        residuals.append(
            dhost_primary_constraint(
                p_star,
                momentum,
                metric,
                mixing,
                scalar_affine=0.0,
                metric_affine=np.zeros(6),
                sqrt_h=1.0,
            )
        )
    assert residuals == pytest.approx([-1.6, 2.4])


def test_full_dirac_determinant_is_effective_bracket_schur_square() -> None:
    c_matrix = np.asarray([[2.0, 0.3], [0.3, 1.4]])
    e_row = np.asarray([0.4, -0.2])
    d_column = np.asarray([0.1, 0.5])
    delta = 1.7
    mixed, effective = primary_secondary_bracket_block(
        c_matrix,
        e_row,
        d_column,
        delta,
    )
    secondary = np.asarray([[0.0, 0.2, -0.3], [-0.2, 0.0, 0.4], [0.3, -0.4, 0.0]])
    dirac = full_dirac_matrix(mixed, secondary)
    expected = (np.linalg.det(c_matrix) * effective) ** 2
    assert np.linalg.det(dirac) == pytest.approx(expected)
    assert np.linalg.matrix_rank(dirac) == 6


def test_zero_effective_bracket_makes_complete_dirac_matrix_singular() -> None:
    c_matrix = np.asarray([[1.2, 0.1], [0.1, 0.9]])
    e_row = np.asarray([0.3, -0.4])
    d_column = np.asarray([0.5, 0.2])
    delta = float(e_row @ np.linalg.solve(c_matrix, d_column))
    mixed, effective = primary_secondary_bracket_block(
        c_matrix,
        e_row,
        d_column,
        delta,
    )
    assert effective == pytest.approx(0.0, abs=1.0e-15)
    assert abs(np.linalg.det(full_dirac_matrix(mixed))) < 1.0e-20


def test_regular_pair_leaves_published_aest_six_dof_unchanged() -> None:
    assert reduced_physical_dof(12, 4, 4) == pytest.approx(6.0)
    assert reduced_physical_dof(13, 4, 6) == pytest.approx(6.0)


def test_primary_audit_advances_without_claiming_complete_chain() -> None:
    report = audit_v12a_primary_dirac(random_trials=100, random_seed=12004)
    assert report["all_identity_gates_pass"]
    assert report["primary_constraint_derived"]
    assert report["secondary_constraint_existence_proven"]
    assert not report["explicit_secondary_density_derived"]
    assert not report["model_specific_effective_bracket_computed"]
    assert not report["complete_dirac_chain_derived"]
    assert not report["physical_degree_count_proven_unchanged"]
    assert not report["theory_viable"]
    assert not report["observational_data_accessed"]


def test_invalid_dirac_blocks_are_rejected() -> None:
    with pytest.raises(ValueError):
        primary_secondary_bracket_block(
            np.zeros((2, 2)),
            np.zeros(2),
            np.zeros(2),
            0.0,
        )
    with pytest.raises(ValueError):
        full_dirac_matrix(np.eye(2), np.eye(2))
