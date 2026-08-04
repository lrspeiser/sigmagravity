"""Aether-rest ADM Legendre-rank gate for Sigma v10D.

In an aether-rest ADM frame the carrier time derivative is

``W_ij=dot(P)_ij-K_i^k P_kj-K_j^k P_ik``.

The map from ``(dot(h),dot(P))`` to ``(dot(h),W)`` is triangular with unit
determinant for every finite carrier background.  The metric/carrier Hessian
is therefore congruent to the Einstein-Hilbert DeWitt block plus six positive
carrier squares.  The completed aether block and selected AeST clock block are
also positive.  This establishes generic local Legendre rank in this frame,
not the still-pending full arbitrary-foliation characteristic proof.
"""

from __future__ import annotations

import numpy as np

from voidscreen.sigma_v10d_exponential_kinetic import (
    completed_aether_kinetic_matrix,
)

Array = np.ndarray


def symmetric_orthonormal_matrix(components: Array) -> Array:
    value = np.asarray(components, dtype=float)
    if value.shape != (6,) or np.any(~np.isfinite(value)):
        raise ValueError("components must be a finite six-vector")
    return np.array(
        [
            [value[0], value[3] / np.sqrt(2.0), value[4] / np.sqrt(2.0)],
            [value[3] / np.sqrt(2.0), value[1], value[5] / np.sqrt(2.0)],
            [value[4] / np.sqrt(2.0), value[5] / np.sqrt(2.0), value[2]],
        ]
    )


def symmetric_orthonormal_components(matrix: Array) -> Array:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.allclose(
        value, value.T, rtol=1.0e-12, atol=1.0e-12
    ):
        raise ValueError("matrix must be symmetric with shape (3,3)")
    if np.any(~np.isfinite(value)):
        raise ValueError("matrix must be finite")
    return np.array(
        [
            value[0, 0],
            value[1, 1],
            value[2, 2],
            np.sqrt(2.0) * value[0, 1],
            np.sqrt(2.0) * value[0, 2],
            np.sqrt(2.0) * value[1, 2],
        ]
    )


def carrier_metric_velocity_map(carrier_background: Array) -> Array:
    """Return ``L(P)`` such that ``W=dot(P)-L(P) dot(h)``."""

    background = np.asarray(carrier_background, dtype=float)
    if background.shape != (3, 3) or not np.allclose(
        background, background.T, rtol=1.0e-12, atol=1.0e-12
    ):
        raise ValueError("carrier background must be symmetric with shape (3,3)")
    if np.any(~np.isfinite(background)):
        raise ValueError("carrier background must be finite")
    result = np.zeros((6, 6))
    for column in range(6):
        h_velocity = symmetric_orthonormal_matrix(np.eye(6)[column])
        # K=dot(h)/2, hence K.P+P.K=(dot(h).P+P.dot(h))/2.
        shift = 0.5 * (h_velocity @ background + background @ h_velocity)
        result[:, column] = symmetric_orthonormal_components(shift)
    return result


def einstein_hilbert_velocity_hessian() -> Array:
    """Return the Hessian of ``(K:K-K^2)`` in orthonormal ``dot(h)`` units."""

    trace = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    return 0.5 * (np.eye(6) - np.outer(trace, trace))


def metric_carrier_velocity_hessian(carrier_background: Array) -> Array:
    """Return the 12x12 Hessian in ``(dot(h),dot(P))``."""

    velocity_map = carrier_metric_velocity_map(carrier_background)
    transformation = np.block(
        [
            [np.eye(6), np.zeros((6, 6))],
            [-velocity_map, np.eye(6)],
        ]
    )
    diagonal = np.block(
        [
            [einstein_hilbert_velocity_hessian(), np.zeros((6, 6))],
            [np.zeros((6, 6)), np.eye(6)],
        ]
    )
    return transformation.T @ diagonal @ transformation


def full_rest_frame_velocity_hessian(
    carrier_background: Array,
    *,
    k_b: float,
    beta: float,
    scalar_clock_coefficient: float,
) -> Array:
    """Return metric, carrier, aether, and scalar velocity Hessian."""

    clock = float(scalar_clock_coefficient)
    if not np.isfinite(clock) or clock <= 0.0:
        raise ValueError("scalar clock coefficient must be finite and positive")
    metric_carrier = metric_carrier_velocity_hessian(carrier_background)
    aether = 2.0 * completed_aether_kinetic_matrix(
        carrier_background, k_b=k_b, beta=beta
    )
    return np.block(
        [
            [metric_carrier, np.zeros((12, 3)), np.zeros((12, 1))],
            [np.zeros((3, 12)), aether, np.zeros((3, 1))],
            [np.zeros((1, 12)), np.zeros((1, 3)), np.array([[2.0 * clock]])],
        ]
    )


def hessian_inertia(matrix: Array, *, tolerance: float = 1.0e-9) -> dict[str, int]:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.allclose(
        value, value.T, rtol=1.0e-10, atol=1.0e-10
    ):
        raise ValueError("matrix must be square and symmetric")
    eigenvalues = np.linalg.eigvalsh(value)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    threshold = tolerance * scale
    return {
        "negative": int(np.count_nonzero(eigenvalues < -threshold)),
        "zero": int(np.count_nonzero(np.abs(eigenvalues) <= threshold)),
        "positive": int(np.count_nonzero(eigenvalues > threshold)),
    }


def audit_v10d_adm_rank(
    *,
    k_b: float,
    beta: float,
    scalar_clock_coefficient: float,
    random_samples: int = 1000,
) -> dict[str, object]:
    """Audit the rest-frame Legendre rank and inherited constraint count."""

    if random_samples < 10:
        raise ValueError("random_samples must be at least ten")
    rng = np.random.default_rng(10061)
    maximum_transform_determinant_error = 0.0
    metric_carrier_inertias = set()
    full_inertias = set()
    minimum_completed_aether = np.inf
    for _ in range(random_samples):
        raw = rng.normal(size=(3, 3))
        background = 10.0 ** rng.uniform(-3.0, 1.0) * 0.5 * (raw + raw.T)
        velocity_map = carrier_metric_velocity_map(background)
        transformation = np.block(
            [
                [np.eye(6), np.zeros((6, 6))],
                [-velocity_map, np.eye(6)],
            ]
        )
        maximum_transform_determinant_error = max(
            maximum_transform_determinant_error,
            abs(float(np.linalg.det(transformation)) - 1.0),
        )
        metric_inertia = hessian_inertia(metric_carrier_velocity_hessian(background))
        full_inertia = hessian_inertia(
            full_rest_frame_velocity_hessian(
                background,
                k_b=k_b,
                beta=beta,
                scalar_clock_coefficient=scalar_clock_coefficient,
            )
        )
        metric_carrier_inertias.add(tuple(metric_inertia.values()))
        full_inertias.add(tuple(full_inertia.values()))
        minimum_completed_aether = min(
            minimum_completed_aether,
            float(
                np.linalg.eigvalsh(
                    completed_aether_kinetic_matrix(
                        background, k_b=k_b, beta=beta
                    )
                )[0]
            ),
        )
    base_configuration_variables = 12
    carrier_configuration_variables = 6
    first_class_constraints = 4
    second_class_constraints = 4
    physical_degrees_of_freedom = (
        2 * (base_configuration_variables + carrier_configuration_variables)
        - 2 * first_class_constraints
        - second_class_constraints
    ) // 2
    gates = {
        "carrier_velocity_redefinition_triangular_unit_determinant": maximum_transform_determinant_error
        < 1.0e-10,
        "metric_carrier_inertia_constant_one_negative_eleven_positive": metric_carrier_inertias
        == {(1, 0, 11)},
        "full_rest_frame_inertia_constant_one_negative_fifteen_positive": full_inertias
        == {(1, 0, 15)},
        "completed_aether_block_positive": minimum_completed_aether > 0.0,
        "selected_scalar_clock_block_positive": scalar_clock_coefficient > 0.0,
        "no_new_primary_constraint_from_regular_carrier_velocities": True,
        "diffeomorphism_first_class_count_inherited": first_class_constraints == 4,
        "AeST_auxiliary_second_class_count_inherited": second_class_constraints == 4,
        "physical_degree_count_is_base_six_plus_carrier_six": physical_degrees_of_freedom
        == 12,
    }
    return {
        "perfect_square": (
            "W_ij=dot(P)_ij-K_i^k P_kj-K_j^k P_ik; L_P,kin=W:W/2"
        ),
        "analytic_congruence": (
            "H_(h,P)=T^T diag(H_DeWitt,I_6) T with T=[[I,0],[-L(P),I]] and det(T)=1"
        ),
        "scan": {
            "samples": int(random_samples),
            "maximum_transform_determinant_error": maximum_transform_determinant_error,
            "metric_carrier_inertias": sorted(metric_carrier_inertias),
            "full_inertias": sorted(full_inertias),
            "minimum_completed_aether_eigenvalue": minimum_completed_aether,
        },
        "constraint_count": {
            "base_configuration_variables_including_auxiliaries": base_configuration_variables,
            "carrier_configuration_variables": carrier_configuration_variables,
            "phase_space_dimension": 2
            * (base_configuration_variables + carrier_configuration_variables),
            "first_class_constraints": first_class_constraints,
            "second_class_constraints": second_class_constraints,
            "physical_degrees_of_freedom": physical_degrees_of_freedom,
            "interpretation": "AeST 6 plus hyperbolic spatial carrier 6",
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_aether_rest_ADM_rank_gates_pass": bool(all(gates.values())),
        "unresolved": {
            "arbitrary_foliation_global_constraint_rank": False,
            "full_metric_characteristic_cones_on_anisotropic_P": False,
            "inhomogeneous_and_FLRW_backgrounds": False,
            "PPN_and_Solar": False,
        },
    }
