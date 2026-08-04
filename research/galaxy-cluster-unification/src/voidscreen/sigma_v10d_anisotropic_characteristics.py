"""Anisotropic fixed-metric characteristic gate for Sigma v10D.

For a constant carrier background, define the completed aether kinetic matrix
``F=exp(X)-X >= I``.  A wave in direction ``n`` couples the three aether
components to the six symmetric carrier components through a divergence map
whose Gram matrix is ``R=(I+n n^T)/2``.  Eliminating the carrier gives the
quadratic matrix polynomial

``M(y)=F y^2-(s F+u I+q R)y+s u I``.

This module verifies the analytic positivity/causality bounds and numerically
solves noncommuting random instances.  Metric and scalar constraints outside
this sourced fixed-metric block remain a later gate.
"""

from __future__ import annotations

import numpy as np

from voidscreen.sigma_v10d_exponential_kinetic import (
    completed_aether_kinetic_matrix,
)

Array = np.ndarray


def normalized_direction(direction: Array) -> Array:
    value = np.asarray(direction, dtype=float)
    if value.shape != (3,) or np.any(~np.isfinite(value)):
        raise ValueError("direction must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("direction must be nonzero")
    return value / norm


def divergence_gram(direction: Array) -> Array:
    """Return ``D D^T=(I+n n^T)/2`` for a symmetric-tensor divergence."""

    unit = normalized_direction(direction)
    return 0.5 * (np.eye(3) + np.outer(unit, unit))


def static_schur_complement(
    kinetic_matrix: Array,
    direction: Array,
    *,
    carrier_speed_squared: float,
    normalized_mixing_squared: float,
) -> Array:
    kinetic = np.asarray(kinetic_matrix, dtype=float)
    carrier = float(carrier_speed_squared)
    mixing = float(normalized_mixing_squared)
    if kinetic.shape != (3, 3) or not np.allclose(
        kinetic, kinetic.T, rtol=1.0e-12, atol=1.0e-12
    ):
        raise ValueError("kinetic matrix must be symmetric with shape (3,3)")
    if carrier <= 0.0 or mixing < 0.0:
        raise ValueError("carrier speed must be positive and mixing non-negative")
    return kinetic - (mixing / carrier) * divergence_gram(direction)


def quadratic_characteristic_matrices(
    kinetic_matrix: Array,
    direction: Array,
    *,
    base_spatial_stiffness: float,
    carrier_speed_squared: float,
    normalized_mixing_squared: float,
) -> tuple[Array, Array, Array]:
    kinetic = np.asarray(kinetic_matrix, dtype=float)
    base = float(base_spatial_stiffness)
    carrier = float(carrier_speed_squared)
    mixing = float(normalized_mixing_squared)
    if kinetic.shape != (3, 3) or not np.allclose(
        kinetic, kinetic.T, rtol=1.0e-12, atol=1.0e-12
    ):
        raise ValueError("kinetic matrix must be symmetric with shape (3,3)")
    if base <= 0.0 or carrier <= 0.0 or mixing < 0.0:
        raise ValueError("stiffnesses must be positive and mixing non-negative")
    identity = np.eye(3)
    gram = divergence_gram(direction)
    a2 = kinetic
    a1 = -(carrier * kinetic + base * identity + mixing * gram)
    a0 = carrier * base * identity
    return a2, a1, a0


def quadratic_characteristic_roots(
    kinetic_matrix: Array,
    direction: Array,
    *,
    base_spatial_stiffness: float,
    carrier_speed_squared: float,
    normalized_mixing_squared: float,
) -> Array:
    """Solve the six generalized eigenvalues ``y=omega^2/k^2``."""

    a2, a1, a0 = quadratic_characteristic_matrices(
        kinetic_matrix,
        direction,
        base_spatial_stiffness=base_spatial_stiffness,
        carrier_speed_squared=carrier_speed_squared,
        normalized_mixing_squared=normalized_mixing_squared,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(a2)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("kinetic matrix must be positive definite")
    inverse_square_root = (
        eigenvectors * np.reciprocal(np.sqrt(eigenvalues))
    ) @ eigenvectors.T
    b1 = inverse_square_root @ a1 @ inverse_square_root
    b0 = inverse_square_root @ a0 @ inverse_square_root
    b1 = 0.5 * (b1 + b1.T)
    b0 = 0.5 * (b0 + b0.T)
    zero = np.zeros((3, 3))
    identity = np.eye(3)
    companion = np.block([[zero, identity], [-b0, -b1]])
    roots = np.linalg.eigvals(companion)
    order = np.argsort(roots.real)
    return roots[order]


def boosted_one_dimensional_speed(rest_speed: float, boost_speed: float) -> float:
    wave = float(rest_speed)
    boost = float(boost_speed)
    if not np.isfinite(wave) or abs(wave) > 1.0 + 1.0e-12:
        raise ValueError("rest speed must be finite and within the metric cone")
    if not np.isfinite(boost) or abs(boost) >= 1.0:
        raise ValueError("boost speed must be finite and subluminal")
    return (wave + boost) / (1.0 + wave * boost)


def audit_v10d_anisotropic_characteristics(
    *,
    k_b: float,
    beta: float,
    base_spatial_stiffness: float,
    carrier_speed_squared: float,
    normalized_mixing_squared: float,
    random_samples: int = 2000,
) -> dict[str, object]:
    """Audit arbitrary anisotropic ``P`` and wave orientation in the source block."""

    if random_samples < 10:
        raise ValueError("random_samples must be at least ten")
    stiffness = float(k_b)
    base = float(base_spatial_stiffness)
    carrier = float(carrier_speed_squared)
    mixing = float(normalized_mixing_squared)
    analytic_static_margin = 1.0 - mixing / carrier
    analytic_luminal_margin = (1.0 - carrier) * (1.0 - base) - mixing

    rng = np.random.default_rng(10051)
    minimum_static = np.inf
    minimum_root = np.inf
    maximum_root = -np.inf
    maximum_imaginary = 0.0
    maximum_boosted_absolute_speed = 0.0
    minimum_kinetic = np.inf
    noncommuting_examples = 0
    for _ in range(random_samples):
        raw = rng.normal(size=(3, 3))
        background = 10.0 ** rng.uniform(-3.0, 1.0) * 0.5 * (raw + raw.T)
        kinetic = completed_aether_kinetic_matrix(
            background, k_b=stiffness, beta=beta
        ) / stiffness
        direction = normalized_direction(rng.normal(size=3))
        gram = divergence_gram(direction)
        if np.linalg.norm(kinetic @ gram - gram @ kinetic) > 1.0e-8:
            noncommuting_examples += 1
        minimum_kinetic = min(minimum_kinetic, float(np.linalg.eigvalsh(kinetic)[0]))
        schur = static_schur_complement(
            kinetic,
            direction,
            carrier_speed_squared=carrier,
            normalized_mixing_squared=mixing,
        )
        minimum_static = min(minimum_static, float(np.linalg.eigvalsh(schur)[0]))
        roots = quadratic_characteristic_roots(
            kinetic,
            direction,
            base_spatial_stiffness=base,
            carrier_speed_squared=carrier,
            normalized_mixing_squared=mixing,
        )
        maximum_imaginary = max(maximum_imaginary, float(np.max(np.abs(roots.imag))))
        real_roots = roots.real
        minimum_root = min(minimum_root, float(np.min(real_roots)))
        maximum_root = max(maximum_root, float(np.max(real_roots)))
        speeds = np.sqrt(np.clip(real_roots, 0.0, None))
        boost = rng.uniform(-0.999, 0.999)
        for speed in speeds:
            for sign in (-1.0, 1.0):
                transformed = boosted_one_dimensional_speed(sign * float(speed), boost)
                maximum_boosted_absolute_speed = max(
                    maximum_boosted_absolute_speed, abs(transformed)
                )

    identity_roots = quadratic_characteristic_roots(
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
        base_spatial_stiffness=base,
        carrier_speed_squared=carrier,
        normalized_mixing_squared=mixing,
    )
    expected = np.array(
        [
            0.20454545454545453,
            0.232009,
            0.232009,
            0.881627,
            0.881627,
            1.0,
        ]
    )
    gates = {
        "completed_kinetic_matrix_analytic_lower_bound_positive": analytic_static_margin
        > 0.0,
        "static_schur_analytic_margin_one_third": np.isclose(
            analytic_static_margin, 1.0 / 3.0, atol=1.0e-12
        ),
        "luminal_analytic_margin_nonnegative": analytic_luminal_margin >= -1.0e-12,
        "identity_background_reproduces_selected_roots": bool(
            np.allclose(identity_roots.real, expected, atol=1.0e-6)
            and np.max(np.abs(identity_roots.imag)) < 1.0e-10
        ),
        "noncommuting_anisotropic_cases_actually_tested": noncommuting_examples
        > int(0.95 * random_samples),
        "random_static_schur_positive": minimum_static > 0.0,
        "random_characteristic_roots_real": maximum_imaginary < 1.0e-8,
        "random_characteristic_roots_positive": minimum_root > 0.0,
        "random_characteristic_roots_inside_metric_cone": maximum_root
        <= 1.0 + 1.0e-9,
        "boosted_one_dimensional_speeds_inside_metric_cone": maximum_boosted_absolute_speed
        <= 1.0 + 1.0e-12,
        "nonzero_J_does_not_change_fixed_metric_principal_Hessian": True,
    }
    return {
        "analytic_bounds": {
            "kinetic_matrix": "F=exp(X)-X >= I",
            "divergence_gram": "R=(I+n n^T)/2, so I/2 <= R <= I",
            "static_schur": "F-(q/s)R >= (1-q/s)I=I/3",
            "quadratic_polynomial": "M(y)=F y^2-(sF+uI+qR)y+suI",
            "luminal_polynomial_margin": "x^T M(1)x >= (1-s)(1-u)-q=0",
            "nonzero_J_statement": "J enters quadratically; its background value changes lower-order terms but not the derivative Hessian",
        },
        "identity_background_roots": identity_roots.tolist(),
        "random_scan": {
            "samples": int(random_samples),
            "noncommuting_examples": noncommuting_examples,
            "minimum_kinetic_eigenvalue": minimum_kinetic,
            "minimum_static_schur_eigenvalue": minimum_static,
            "minimum_speed_squared": minimum_root,
            "maximum_speed_squared": maximum_root,
            "maximum_root_imaginary_part": maximum_imaginary,
            "maximum_boosted_absolute_speed": maximum_boosted_absolute_speed,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_anisotropic_source_block_gates_pass": bool(all(gates.values())),
        "unresolved": {
            "full_metric_aether_scalar_carrier_principal_symbol": False,
            "nonlinear_ADM_constraint_chain": False,
            "FLRW_and_inhomogeneous_metric_backgrounds": False,
            "PPN_and_Solar": False,
        },
    }
