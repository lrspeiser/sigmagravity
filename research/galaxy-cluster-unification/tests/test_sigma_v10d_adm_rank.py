from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v10d_adm_rank import (
    audit_v10d_adm_rank,
    carrier_metric_velocity_map,
    einstein_hilbert_velocity_hessian,
    hessian_inertia,
    metric_carrier_velocity_hessian,
    symmetric_orthonormal_components,
    symmetric_orthonormal_matrix,
)


def test_symmetric_orthonormal_round_trip_preserves_frobenius_norm() -> None:
    components = np.array([1.0, -2.0, 3.0, 0.4, -0.7, 1.2])
    matrix = symmetric_orthonormal_matrix(components)
    assert symmetric_orthonormal_components(matrix) == pytest.approx(components)
    assert np.sum(matrix**2) == pytest.approx(np.sum(components**2))


def test_dewitt_hessian_has_one_negative_and_five_positive_directions() -> None:
    assert hessian_inertia(einstein_hilbert_velocity_hessian()) == {
        "negative": 1,
        "zero": 0,
        "positive": 5,
    }


def test_carrier_velocity_shift_is_triangular_for_anisotropic_background() -> None:
    background = np.array([[2.0, 0.3, -0.1], [0.3, -1.0, 0.4], [-0.1, 0.4, 0.7]])
    velocity_map = carrier_metric_velocity_map(background)
    transform = np.block(
        [
            [np.eye(6), np.zeros((6, 6))],
            [-velocity_map, np.eye(6)],
        ]
    )
    assert np.linalg.det(transform) == pytest.approx(1.0)
    assert hessian_inertia(metric_carrier_velocity_hessian(background)) == {
        "negative": 1,
        "zero": 0,
        "positive": 11,
    }


def test_adm_rank_audit_passes_only_aether_rest_subgate() -> None:
    report = audit_v10d_adm_rank(
        k_b=1.0,
        beta=np.sqrt(2.0 / 11.0),
        scalar_clock_coefficient=4.0,
        random_samples=100,
    )
    assert all(report["gates"].values())
    assert report["constraint_count"]["physical_degrees_of_freedom"] == 12
    assert report["unresolved"]["full_metric_characteristic_cones_on_anisotropic_P"] is False


def test_invalid_adm_rank_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        carrier_metric_velocity_map(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        audit_v10d_adm_rank(
            k_b=1.0,
            beta=0.4,
            scalar_clock_coefficient=0.0,
            random_samples=20,
        )
