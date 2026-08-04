"""Canonical primary and conditional Dirac structure for Sigma v12A.

The calculation uses the standard auxiliary-gradient ADM variables for a
quadratic DHOST scalar.  It proves the explicit primary momentum identity and
the algebraic form of the coupled AeST-auxiliary/DHOST Dirac determinant.  It
does not evaluate the model-specific secondary bracket on general fields.
"""

from __future__ import annotations

import numpy as np

from voidscreen.sigma_v12a_joint_adm_rank import dewit_signature_block


def dhost_canonical_momenta(
    metric_block: np.ndarray,
    scalar_metric_mixing: np.ndarray,
    *,
    scalar_velocity_coefficient: float,
    scalar_velocity: float,
    metric_velocity: np.ndarray,
    scalar_affine: float,
    metric_affine: np.ndarray,
    sqrt_h: float,
) -> tuple[float, np.ndarray]:
    """Return ``(p_*, pi_A)`` in the published DHOST ADM convention.

    For

    ``L/(N sqrt(h)) = A V_*^2 + 2 B_A V_* K_A + K_AB K_A K_B
                       + 2 C_A K_A + 2 C_0 V_* - U``,

    the scalar and metric momenta are

    ``p_*=2 sqrt(h)(A V_*+B.K+C_0)`` and
    ``pi=sqrt(h)(K.K+B V_*+C)``.
    """

    metric = np.asarray(metric_block, dtype=float)
    mixing = np.asarray(scalar_metric_mixing, dtype=float)
    velocity = np.asarray(metric_velocity, dtype=float)
    affine = np.asarray(metric_affine, dtype=float)
    volume = float(sqrt_h)
    scalar_coefficient = float(scalar_velocity_coefficient)
    v_star = float(scalar_velocity)
    c_zero = float(scalar_affine)
    if (
        metric.ndim != 2
        or metric.shape[0] != metric.shape[1]
        or mixing.shape != (metric.shape[0],)
        or velocity.shape != mixing.shape
        or affine.shape != mixing.shape
        or not np.allclose(metric, metric.T, rtol=0.0, atol=1.0e-12)
        or np.any(~np.isfinite(metric))
        or np.any(~np.isfinite(mixing))
        or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(affine))
        or not np.all(np.isfinite([volume, scalar_coefficient, v_star, c_zero]))
        or volume <= 0.0
    ):
        raise ValueError("canonical momentum inputs must be finite and compatible")
    p_star = 2.0 * volume * (scalar_coefficient * v_star + mixing @ velocity + c_zero)
    metric_momentum = volume * (metric @ velocity + mixing * v_star + affine)
    return float(p_star), metric_momentum


def dhost_primary_constraint(
    p_star: float,
    metric_momentum: np.ndarray,
    metric_block: np.ndarray,
    scalar_metric_mixing: np.ndarray,
    *,
    scalar_affine: float,
    metric_affine: np.ndarray,
    sqrt_h: float,
) -> float:
    """Evaluate the Class-Ia canonical primary constraint ``Psi_Sigma``."""

    metric = np.asarray(metric_block, dtype=float)
    mixing = np.asarray(scalar_metric_mixing, dtype=float)
    momentum = np.asarray(metric_momentum, dtype=float)
    affine = np.asarray(metric_affine, dtype=float)
    volume = float(sqrt_h)
    if (
        metric.ndim != 2
        or metric.shape[0] != metric.shape[1]
        or mixing.shape != (metric.shape[0],)
        or momentum.shape != mixing.shape
        or affine.shape != mixing.shape
        or np.any(~np.isfinite(metric))
        or np.any(~np.isfinite(mixing))
        or np.any(~np.isfinite(momentum))
        or np.any(~np.isfinite(affine))
        or not np.isfinite(float(p_star))
        or not np.isfinite(float(scalar_affine))
        or not np.isfinite(volume)
        or volume <= 0.0
    ):
        raise ValueError("primary-constraint inputs must be finite and compatible")
    try:
        null_metric_part = np.linalg.solve(metric, mixing)
    except np.linalg.LinAlgError as error:
        raise ValueError("the metric kinetic block must be invertible") from error
    return float(
        p_star
        - 2.0 * null_metric_part @ momentum
        + 2.0 * volume * (null_metric_part @ affine - float(scalar_affine))
    )


def primary_secondary_bracket_block(
    aest_auxiliary_bracket: np.ndarray,
    primary_to_aest_secondary: np.ndarray,
    aest_primary_to_dhost_secondary: np.ndarray,
    dhost_primary_secondary_bracket: float,
) -> tuple[np.ndarray, float]:
    """Return the mixed primary/secondary bracket block and Schur bracket.

    With primaries ``(p_A, Psi)`` and secondaries ``(S_B, Omega)``, the block is

    ``M=[[C,D],[E,Delta]]``.

    Eliminating the AeST auxiliary pairs leaves the effective DHOST bracket
    ``Delta_eff=Delta-E C^-1 D``.
    """

    c_matrix = np.asarray(aest_auxiliary_bracket, dtype=float)
    e_row = np.asarray(primary_to_aest_secondary, dtype=float)
    d_column = np.asarray(aest_primary_to_dhost_secondary, dtype=float)
    delta = float(dhost_primary_secondary_bracket)
    if (
        c_matrix.ndim != 2
        or c_matrix.shape[0] != c_matrix.shape[1]
        or e_row.shape != (c_matrix.shape[0],)
        or d_column.shape != (c_matrix.shape[0],)
        or np.any(~np.isfinite(c_matrix))
        or np.any(~np.isfinite(e_row))
        or np.any(~np.isfinite(d_column))
        or not np.isfinite(delta)
    ):
        raise ValueError("Dirac bracket inputs must be finite and compatible")
    try:
        reduced_column = np.linalg.solve(c_matrix, d_column)
    except np.linalg.LinAlgError as error:
        raise ValueError("the AeST auxiliary bracket block must be invertible") from error
    effective = float(delta - e_row @ reduced_column)
    mixed = np.block(
        [
            [c_matrix, d_column[:, None]],
            [e_row[None, :], np.asarray([[delta]])],
        ]
    )
    return mixed, effective


def full_dirac_matrix(
    primary_secondary_block: np.ndarray,
    secondary_secondary_block: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the antisymmetric Dirac matrix for primary/secondary pairs."""

    mixed = np.asarray(primary_secondary_block, dtype=float)
    if mixed.ndim != 2 or mixed.shape[0] != mixed.shape[1]:
        raise ValueError("the primary-secondary block must be square")
    size = mixed.shape[0]
    if secondary_secondary_block is None:
        secondary = np.zeros((size, size))
    else:
        secondary = np.asarray(secondary_secondary_block, dtype=float)
    if (
        secondary.shape != (size, size)
        or np.any(~np.isfinite(mixed))
        or np.any(~np.isfinite(secondary))
        or not np.allclose(secondary, -secondary.T, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("the secondary bracket block must be finite and antisymmetric")
    zeros = np.zeros_like(mixed)
    return np.block([[zeros, mixed], [-mixed.T, secondary]])


def reduced_physical_dof(
    configuration_variables: int,
    first_class_constraints: int,
    second_class_constraints: int,
) -> float:
    """Return ``(2N-2F-S)/2`` for a regular constrained Hamiltonian system."""

    n_config = int(configuration_variables)
    n_first = int(first_class_constraints)
    n_second = int(second_class_constraints)
    if n_config < 0 or n_first < 0 or n_second < 0:
        raise ValueError("constraint counts must be non-negative")
    return (2.0 * n_config - 2.0 * n_first - n_second) / 2.0


def audit_v12a_primary_dirac(
    *,
    random_trials: int,
    random_seed: int,
) -> dict[str, object]:
    """Audit the exact primary identity and conditional Dirac determinant."""

    if random_trials < 1:
        raise ValueError("the primary-Dirac audit requires at least one trial")
    rng = np.random.default_rng(int(random_seed))
    maximum_primary_residual = 0.0
    maximum_determinant_relative_residual = 0.0
    minimum_conditional_singular_value = np.inf
    for _ in range(int(random_trials)):
        metric = dewit_signature_block(rng)
        mixing = rng.normal(size=6)
        metric_affine = rng.normal(size=6)
        scalar_affine = float(rng.normal())
        metric_velocity = rng.normal(size=6)
        scalar_velocity = float(rng.normal())
        sqrt_h = float(np.exp(rng.uniform(-1.0, 1.0)))
        scalar_coefficient = float(mixing @ np.linalg.solve(metric, mixing))
        p_star, metric_momentum = dhost_canonical_momenta(
            metric,
            mixing,
            scalar_velocity_coefficient=scalar_coefficient,
            scalar_velocity=scalar_velocity,
            metric_velocity=metric_velocity,
            scalar_affine=scalar_affine,
            metric_affine=metric_affine,
            sqrt_h=sqrt_h,
        )
        primary = dhost_primary_constraint(
            p_star,
            metric_momentum,
            metric,
            mixing,
            scalar_affine=scalar_affine,
            metric_affine=metric_affine,
            sqrt_h=sqrt_h,
        )
        maximum_primary_residual = max(maximum_primary_residual, abs(primary))

        raw = rng.normal(size=(2, 2))
        orthogonal, _ = np.linalg.qr(raw)
        eigenvalues = rng.uniform(0.2, 2.0, size=2)
        c_matrix = orthogonal @ np.diag(eigenvalues) @ orthogonal.T
        e_row = rng.normal(size=2)
        d_column = rng.normal(size=2)
        target_effective = float(rng.choice(np.asarray([-1.0, 1.0])) * rng.uniform(0.2, 2.0))
        delta = float(e_row @ np.linalg.solve(c_matrix, d_column) + target_effective)
        mixed, effective = primary_secondary_bracket_block(
            c_matrix,
            e_row,
            d_column,
            delta,
        )
        raw_secondary = rng.normal(size=(3, 3))
        secondary = raw_secondary - raw_secondary.T
        dirac = full_dirac_matrix(mixed, secondary)
        expected_determinant = (np.linalg.det(c_matrix) * effective) ** 2
        determinant = float(np.linalg.det(dirac))
        determinant_scale = max(1.0, abs(expected_determinant))
        maximum_determinant_relative_residual = max(
            maximum_determinant_relative_residual,
            abs(determinant - expected_determinant) / determinant_scale,
        )
        minimum_conditional_singular_value = min(
            minimum_conditional_singular_value,
            float(np.min(np.linalg.svd(dirac, compute_uv=False))),
        )

    base_dof = reduced_physical_dof(12, 4, 4)
    conditional_combined_dof = reduced_physical_dof(13, 4, 6)
    identity_gates = {
        "canonical_primary_velocity_identity": maximum_primary_residual < 1.0e-11,
        "conditional_dirac_determinant_identity": (maximum_determinant_relative_residual < 1.0e-10),
        "conditional_random_dirac_matrices_regular": (minimum_conditional_singular_value > 1.0e-12),
        "conditional_physical_dof_unchanged": (base_dof == conditional_combined_dof == 6.0),
    }
    return {
        "candidate": "Sigma v12A same-AeST-clock luminal DHOST geometry",
        "canonical_primary": {
            "formula": "Psi=p_*-2(K^-1 B)^A pi_A+2 sqrt(h)[(K^-1 B)^A C_A-C_0] approximately 0",
            "aest_metric_momentum_shift_in_published_reduced_variables": 0.0,
            "commutes_with_aest_auxiliary_primaries": ["Pi_mu", "Pi_nu"],
            "maximum_velocity_identity_residual": maximum_primary_residual,
        },
        "secondary_existence": {
            "formula": "Omega_Sigma={Psi_Sigma,H_0} approximately 0",
            "p_phi_coefficient": 1.0,
            "reason": "Psi commutes with the AeST auxiliary primaries, so its preservation cannot fix their multipliers; the shift-symmetric AeST lower-derivative terms modify Omega_rest but not the unit p_phi coefficient.",
            "explicit_local_density_derived": False,
        },
        "conditional_dirac_structure": {
            "primary_order": ["Pi_mu", "Pi_nu", "Psi_Sigma"],
            "secondary_order": ["S_mu", "S_nu", "Omega_Sigma"],
            "mixed_block": "M=[[C,D],[E,Delta]]",
            "effective_bracket": "Delta_eff=Delta-E C^-1 D",
            "full_determinant": "det(D_Dirac)=det(C)^2 Delta_eff^2",
            "maximum_determinant_relative_residual": (maximum_determinant_relative_residual),
            "minimum_random_singular_value": minimum_conditional_singular_value,
            "base_aest_physical_dof": base_dof,
            "combined_physical_dof_if_regular": conditional_combined_dof,
        },
        "identity_gates": {name: bool(value) for name, value in identity_gates.items()},
        "all_identity_gates_pass": bool(all(identity_gates.values())),
        "primary_constraint_derived": True,
        "secondary_constraint_existence_proven": True,
        "explicit_secondary_density_derived": False,
        "model_specific_effective_bracket_computed": False,
        "complete_dirac_chain_derived": False,
        "physical_degree_count_proven_unchanged": False,
        "theory_viable": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
