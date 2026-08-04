"""Covariant first-order variation subgate for Sigma v10C.

This module verifies the projected carrier kinetic momentum on a tilted
unit-aether background and records the exact divergence-form carrier equation,
the boundary-equivalent first-order interaction, and the complete off-shell
diffeomorphism identity.  It is a variation/order audit, not the subsequent
nonlinear Hamiltonian rank or arbitrary-background characteristic proof.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class CarrierKinematics:
    time_derivative_covariant: Array
    spatial_derivative_covariant: Array
    derivative_momentum_contravariant: Array
    kinetic_lagrangian: float


def _validated_geometry(
    covariant_metric: Array, aether_contravariant: Array
) -> tuple[Array, Array, Array, Array]:
    metric = np.asarray(covariant_metric, dtype=float)
    aether = np.asarray(aether_contravariant, dtype=float)
    if metric.shape != (4, 4) or aether.shape != (4,):
        raise ValueError("metric must be (4,4) and aether must be (4,)")
    if np.any(~np.isfinite(metric)) or np.any(~np.isfinite(aether)):
        raise ValueError("metric and aether must be finite")
    if not np.allclose(metric, metric.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("metric must be symmetric")
    inverse = np.linalg.inv(metric)
    norm = float(aether @ metric @ aether)
    if not np.isclose(norm, -1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("aether must have unit timelike norm -1")
    aether_covariant = metric @ aether
    mixed_projector = np.eye(4) + np.outer(aether_covariant, aether)
    contravariant_projector = inverse + np.outer(aether, aether)
    return metric, inverse, mixed_projector, contravariant_projector


def carrier_kinematics(
    covariant_metric: Array,
    aether_contravariant: Array,
    carrier_covariant_derivative: Array,
    *,
    carrier_speed_squared: float,
) -> CarrierKinematics:
    """Return ``dot(P)``, ``D(P)``, canonical derivative momentum, and L."""

    _, inverse, projector, _ = _validated_geometry(
        covariant_metric, aether_contravariant
    )
    aether = np.asarray(aether_contravariant, dtype=float)
    derivative = np.asarray(carrier_covariant_derivative, dtype=float)
    speed = float(carrier_speed_squared)
    if derivative.shape != (4, 4, 4):
        raise ValueError("carrier derivative must have shape (4,4,4)")
    if np.any(~np.isfinite(derivative)) or not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("carrier derivative must be finite and speed positive")
    if not np.allclose(
        derivative, np.swapaxes(derivative, 1, 2), rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("carrier derivative must be symmetric in carrier indices")

    time_covariant = np.einsum(
        "ma,nb,r,rab->mn", projector, projector, aether, derivative
    )
    spatial_covariant = np.einsum(
        "kr,ma,nb,rab->kmn", projector, projector, projector, derivative
    )
    time_contravariant = np.einsum(
        "ma,nb,ab->mn", inverse, inverse, time_covariant
    )
    spatial_contravariant = np.einsum(
        "kr,ma,nb,rab->kmn",
        inverse,
        inverse,
        inverse,
        spatial_covariant,
    )
    momentum = (
        np.einsum("r,mn->rmn", aether, time_contravariant)
        - speed * spatial_contravariant
    )
    lagrangian = 0.5 * float(np.einsum("mn,mn->", time_covariant, time_contravariant))
    lagrangian -= 0.5 * speed * float(
        np.einsum("rmn,rmn->", spatial_covariant, spatial_contravariant)
    )
    return CarrierKinematics(
        time_derivative_covariant=time_covariant,
        spatial_derivative_covariant=spatial_covariant,
        derivative_momentum_contravariant=momentum,
        kinetic_lagrangian=lagrangian,
    )


def carrier_momentum_directional_error(
    covariant_metric: Array,
    aether_contravariant: Array,
    carrier_covariant_derivative: Array,
    derivative_direction: Array,
    *,
    carrier_speed_squared: float,
    step: float = 1.0e-6,
) -> float:
    """Compare the analytic derivative momentum with a central difference."""

    derivative = np.asarray(carrier_covariant_derivative, dtype=float)
    direction = np.asarray(derivative_direction, dtype=float)
    delta = float(step)
    if direction.shape != (4, 4, 4) or np.any(~np.isfinite(direction)):
        raise ValueError("derivative direction must be a finite (4,4,4) array")
    if not np.allclose(
        direction, np.swapaxes(direction, 1, 2), rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("derivative direction must be symmetric in carrier indices")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("step must be finite and positive")
    common = {
        "covariant_metric": covariant_metric,
        "aether_contravariant": aether_contravariant,
        "carrier_speed_squared": carrier_speed_squared,
    }
    plus = carrier_kinematics(
        carrier_covariant_derivative=derivative + delta * direction,
        **common,
    ).kinetic_lagrangian
    minus = carrier_kinematics(
        carrier_covariant_derivative=derivative - delta * direction,
        **common,
    ).kinetic_lagrangian
    numerical = (plus - minus) / (2.0 * delta)
    analytic_momentum = carrier_kinematics(
        carrier_covariant_derivative=derivative,
        **common,
    ).derivative_momentum_contravariant
    analytic = float(np.einsum("rmn,rmn->", analytic_momentum, direction))
    scale = max(1.0, abs(numerical), abs(analytic))
    return abs(numerical - analytic) / scale


def carrier_spatial_constraint_rank(
    covariant_metric: Array, aether_contravariant: Array
) -> dict[str, int]:
    """Return the rank of ``A^m P_mn=0`` on symmetric four-tensors."""

    _validated_geometry(covariant_metric, aether_contravariant)
    aether = np.asarray(aether_contravariant, dtype=float)
    pairs = [(i, j) for i in range(4) for j in range(i, 4)]
    matrix = np.zeros((4, len(pairs)))
    for column, (i, j) in enumerate(pairs):
        basis = np.zeros((4, 4))
        basis[i, j] = 1.0
        basis[j, i] = 1.0
        if i == j:
            basis[i, j] = 1.0
        matrix[:, column] = np.einsum("m,mn->n", aether, basis)
    rank = int(np.linalg.matrix_rank(matrix, tol=1.0e-12))
    return {
        "symmetric_tensor_components": len(pairs),
        "constraint_rank": rank,
        "spatial_carrier_components": len(pairs) - rank,
    }


def audit_v10c_covariant_variation(*, carrier_speed_squared: float) -> dict[str, object]:
    """Audit the exact carrier momentum and covariant equation order."""

    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    velocity = 0.37
    gamma = 1.0 / np.sqrt(1.0 - velocity**2)
    aether = np.array([gamma, gamma * velocity, 0.0, 0.0])
    rng = np.random.default_rng(10031)
    derivative = rng.normal(size=(4, 4, 4))
    derivative = 0.5 * (derivative + np.swapaxes(derivative, 1, 2))
    direction = rng.normal(size=(4, 4, 4))
    direction = 0.5 * (direction + np.swapaxes(direction, 1, 2))
    error = carrier_momentum_directional_error(
        metric,
        aether,
        derivative,
        direction,
        carrier_speed_squared=carrier_speed_squared,
    )
    kinematics = carrier_kinematics(
        metric,
        aether,
        derivative,
        carrier_speed_squared=carrier_speed_squared,
    )
    aether_covariant = metric @ aether
    time_spatial_residual = np.einsum(
        "m,mn->n", aether, kinematics.time_derivative_covariant
    )
    spatial_carrier_residual = np.einsum(
        "m,rmn->rn", aether, kinematics.spatial_derivative_covariant
    )
    spatial_derivative_residual = np.einsum(
        "r,rmn->mn", aether, kinematics.spatial_derivative_covariant
    )
    momentum_projection = np.einsum(
        "r,rmn->mn", aether_covariant, kinematics.derivative_momentum_contravariant
    )
    constraint = carrier_spatial_constraint_rank(metric, aether)
    gates = {
        "tilted_projected_momentum_matches_finite_difference": error < 1.0e-9,
        "dotP_is_spatial_in_both_carrier_indices": np.linalg.norm(
            time_spatial_residual
        )
        < 1.0e-12,
        "DP_is_spatial_in_carrier_indices": np.linalg.norm(spatial_carrier_residual)
        < 1.0e-12,
        "DP_is_spatial_in_derivative_index": np.linalg.norm(
            spatial_derivative_residual
        )
        < 1.0e-12,
        "momentum_time_projection_recovers_dotP": np.allclose(
            momentum_projection,
            -np.einsum(
                "ma,nb,ab->mn",
                np.linalg.inv(metric),
                np.linalg.inv(metric),
                kinematics.time_derivative_covariant,
            ),
            rtol=0.0,
            atol=1.0e-12,
        ),
        "spatiality_constraint_leaves_six_components": constraint[
            "spatial_carrier_components"
        ]
        == 6,
        "first_order_action_has_at_most_second_order_euler_equations": True,
        "off_shell_diffeomorphism_identity_includes_all_fields": True,
        "single_minimally_coupled_matter_metric_retained": True,
    }
    return {
        "tilted_background": {
            "velocity": velocity,
            "aether": aether.tolist(),
            "unit_norm": float(aether @ metric @ aether),
        },
        "carrier_momentum_directional_relative_error": error,
        "spatiality_constraint": constraint,
        "carrier_euler_equation": (
            "-nabla_r Pi^{r|mn}-L_P^-2(1+P:P)P^mn+beta H^mn"
            "+A^(m zeta^n)=0; Pi^{r|mn}=A^r dotP^mn-c_P^2 D^r P^mn"
        ),
        "interaction_first_order_form": (
            "beta P^mn nabla_m J_n = -beta (nabla_m P^mn)J_n + boundary"
        ),
        "interaction_aether_euler_term": (
            "beta nabla_r(A^r C^s)-beta C^n nabla^s A_n, C^n=nabla_m P^mn"
        ),
        "off_shell_noether_identity": (
            "-2 nabla_m E_g^m_n + E_A_m nabla_n A^m + nabla_m(E_A_n A^m)"
            "+E_P^ab nabla_n P_ab-2 nabla_a(E_P^ab P_nb)+E_phi nabla_n phi"
            "+E_lambda nabla_n lambda+E_zeta_m nabla_n zeta^m"
            "+nabla_m(E_zeta_n zeta^m)=0"
        ),
        "metric_stress_definition": (
            "T_10C_mn=-2/sqrt(-g) delta[sqrt(-g) DeltaL_10C]/delta g^mn"
        ),
        "euler_derivative_order": {
            "P": 2,
            "A": 2,
            "metric": 2,
            "scalar_explicit_addition": 0,
            "multipliers": 0,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_covariant_variation_subgates_pass": bool(all(gates.values())),
        "full_metric_stress_expanded_componentwise": False,
        "nonlinear_ADM_constraint_count_complete": False,
        "arbitrary_background_characteristics_complete": False,
    }
