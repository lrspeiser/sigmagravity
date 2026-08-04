"""Joint AeST--DHOST kinetic-rank subgate for Sigma v12A."""

from __future__ import annotations

import numpy as np


def dewit_signature_block(rng: np.random.Generator, dimension: int = 6) -> np.ndarray:
    """Return a random symmetric block with one negative DeWitt direction."""

    if dimension < 2:
        raise ValueError("the metric block needs at least two directions")
    raw = rng.normal(size=(dimension, dimension))
    orthogonal, _ = np.linalg.qr(raw)
    eigenvalues = np.concatenate(([-rng.uniform(0.2, 2.0)], rng.uniform(0.2, 2.0, dimension - 1)))
    return orthogonal @ np.diag(eigenvalues) @ orthogonal.T


def degenerate_dhost_kinetic_matrix(
    metric_block: np.ndarray,
    scalar_metric_mixing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the canonical one-null DHOST velocity Hessian.

    For velocities ``(V_*, K_A)``, degeneracy is the Schur identity
    ``A=B^T K^-1 B``.  The corresponding null vector is
    ``(1,-K^-1 B)``.
    """

    metric = np.asarray(metric_block, dtype=float)
    mixing = np.asarray(scalar_metric_mixing, dtype=float)
    if (
        metric.ndim != 2
        or metric.shape[0] != metric.shape[1]
        or mixing.shape != (metric.shape[0],)
        or np.any(~np.isfinite(metric))
        or np.any(~np.isfinite(mixing))
        or not np.allclose(metric, metric.T, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("DHOST blocks must be finite, symmetric, and compatible")
    try:
        solved = np.linalg.solve(metric, mixing)
    except np.linalg.LinAlgError as error:
        raise ValueError("the metric kinetic block must be invertible") from error
    scalar = float(mixing @ solved)
    hessian = np.block(
        [
            [np.asarray([[scalar]]), mixing[None, :]],
            [mixing[:, None], metric],
        ]
    )
    null = np.concatenate(([1.0], -solved))
    null /= np.linalg.norm(null)
    return hessian, null


def joint_aest_dhost_hessian(
    dhost_hessian: np.ndarray,
    *,
    k_b: float,
    aether_metric: np.ndarray | None = None,
    aether_dimension: int = 3,
) -> np.ndarray:
    """Add the independent positive Maxwell-aether velocity block.

    ``aether_metric`` represents the positive spatial inverse-metric form seen
    by the electric Maxwell velocities.  It is the identity in a local
    orthonormal ADM basis, but accepting a general positive-definite form makes
    the congruence/basis independence of the rank statement explicit.
    """

    dhost = np.asarray(dhost_hessian, dtype=float)
    vector_coupling = float(k_b)
    if aether_metric is None:
        electric_metric = np.eye(int(aether_dimension))
    else:
        electric_metric = np.asarray(aether_metric, dtype=float)
        aether_dimension = int(electric_metric.shape[0]) if electric_metric.ndim == 2 else 0
    if (
        dhost.ndim != 2
        or dhost.shape[0] != dhost.shape[1]
        or not np.allclose(dhost, dhost.T, rtol=0.0, atol=1.0e-12)
        or np.any(~np.isfinite(dhost))
        or vector_coupling <= 0.0
        or not np.isfinite(vector_coupling)
        or aether_dimension < 1
        or electric_metric.shape != (aether_dimension, aether_dimension)
        or np.any(~np.isfinite(electric_metric))
        or not np.allclose(electric_metric, electric_metric.T, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("joint Hessian inputs are outside their finite domain")
    try:
        np.linalg.cholesky(electric_metric)
    except np.linalg.LinAlgError as error:
        raise ValueError("the Maxwell electric metric must be positive definite") from error
    aether = vector_coupling * electric_metric
    return np.block(
        [
            [dhost, np.zeros((dhost.shape[0], aether_dimension))],
            [np.zeros((aether_dimension, dhost.shape[0])), aether],
        ]
    )


def matrix_inertia(matrix: np.ndarray, *, tolerance: float = 1.0e-10) -> tuple[int, int, int]:
    """Return counts of negative, zero, and positive eigenvalues."""

    hessian = np.asarray(matrix, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("inertia requires a square matrix")
    eigenvalues = np.linalg.eigvalsh(hessian)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    threshold = float(tolerance) * scale
    return (
        int(np.sum(eigenvalues < -threshold)),
        int(np.sum(np.abs(eigenvalues) <= threshold)),
        int(np.sum(eigenvalues > threshold)),
    )


def finite_difference_hessian(
    hessian: np.ndarray,
    linear_momentum_shift: np.ndarray,
    *,
    step: float,
) -> np.ndarray:
    """Differentiate a quadratic plus arbitrary linear AeST momentum shift."""

    matrix = np.asarray(hessian, dtype=float)
    shift = np.asarray(linear_momentum_shift, dtype=float)
    spacing = float(step)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or shift.shape != (matrix.shape[0],)
        or spacing <= 0.0
        or not np.isfinite(spacing)
    ):
        raise ValueError("finite-difference Hessian inputs are invalid")

    def lagrangian(velocity: np.ndarray) -> float:
        return float(0.5 * velocity @ matrix @ velocity + shift @ velocity)

    dimension = matrix.shape[0]
    origin = np.zeros(dimension)
    result = np.zeros_like(matrix)
    base = lagrangian(origin)
    for i in range(dimension):
        ei = np.zeros(dimension)
        ei[i] = spacing
        result[i, i] = (lagrangian(ei) - 2.0 * base + lagrangian(-ei)) / spacing**2
        for j in range(i + 1, dimension):
            ej = np.zeros(dimension)
            ej[j] = spacing
            mixed = (
                lagrangian(ei + ej)
                - lagrangian(ei - ej)
                - lagrangian(-ei + ej)
                + lagrangian(-ei - ej)
            ) / (4.0 * spacing**2)
            result[i, j] = result[j, i] = mixed
    return result


def audit_v12a_joint_adm_rank(
    *,
    k_b: float,
    random_trials: int,
    random_seed: int,
    finite_difference_step: float,
) -> dict[str, object]:
    if random_trials < 1:
        raise ValueError("joint-rank audit requires at least one trial")
    rng = np.random.default_rng(int(random_seed))
    maximum_dhost_null_residual = 0.0
    maximum_joint_null_residual = 0.0
    maximum_fd_residual = 0.0
    observed_dhost_inertias: set[tuple[int, int, int]] = set()
    observed_joint_inertias: set[tuple[int, int, int]] = set()
    minimum_aether_eigenvalue = np.inf
    representative: dict[str, object] | None = None
    for trial in range(int(random_trials)):
        metric = dewit_signature_block(rng)
        mixing = rng.normal(size=6)
        dhost, null = degenerate_dhost_kinetic_matrix(metric, mixing)
        electric_basis = rng.normal(size=(3, 3))
        electric_orthogonal, _ = np.linalg.qr(electric_basis)
        electric_eigenvalues = rng.uniform(0.2, 2.0, size=3)
        electric_metric = (
            electric_orthogonal @ np.diag(electric_eigenvalues) @ electric_orthogonal.T
        )
        joint = joint_aest_dhost_hessian(
            dhost,
            k_b=float(k_b),
            aether_metric=electric_metric,
        )
        joint_null = np.concatenate((null, np.zeros(3)))
        dhost_residual = float(np.linalg.norm(dhost @ null))
        joint_residual = float(np.linalg.norm(joint @ joint_null))
        maximum_dhost_null_residual = max(maximum_dhost_null_residual, dhost_residual)
        maximum_joint_null_residual = max(maximum_joint_null_residual, joint_residual)
        observed_dhost_inertias.add(matrix_inertia(dhost))
        observed_joint_inertias.add(matrix_inertia(joint))
        minimum_aether_eigenvalue = min(
            minimum_aether_eigenvalue,
            float(k_b) * float(np.min(electric_eigenvalues)),
        )
        if trial == 0:
            shift = rng.normal(size=joint.shape[0])
            numerical = finite_difference_hessian(joint, shift, step=float(finite_difference_step))
            maximum_fd_residual = float(np.max(np.abs(numerical - joint)))
            representative = {
                "dhost_inertia": matrix_inertia(dhost),
                "joint_inertia": matrix_inertia(joint),
                "dhost_null_residual": dhost_residual,
                "joint_null_residual": joint_residual,
                "finite_difference_hessian_residual": maximum_fd_residual,
                "linear_momentum_shift_norm": float(np.linalg.norm(shift)),
                "aether_metric_eigenvalues": sorted((float(k_b) * electric_eigenvalues).tolist()),
            }

    expected_dhost_inertia = (1, 1, 5)
    expected_joint_inertia = (1, 1, 8)
    gates = {
        "dhost_primary_null_preserved": maximum_dhost_null_residual < 1.0e-11,
        "joint_primary_null_preserved": maximum_joint_null_residual < 1.0e-11,
        "dhost_inertia_constant": observed_dhost_inertias == {expected_dhost_inertia},
        "joint_inertia_adds_only_positive_aether_modes": observed_joint_inertias
        == {expected_joint_inertia},
        "aether_velocity_block_positive": minimum_aether_eigenvalue > 0.0,
        "linear_aest_terms_do_not_change_hessian": maximum_fd_residual < 1.0e-6,
        "no_lapse_velocity_added": True,
    }
    return {
        "candidate": "Sigma v12A same-AeST-clock luminal DHOST geometry",
        "adm_velocity_basis": {
            "dhost": "(V_*, six K_ij components)",
            "aether": "three physical Maxwell aether velocities",
            "lapse_shift": "no velocities",
        },
        "analytic_block_identity": {
            "dhost": "H_D=[[B^T K^-1 B,B^T],[B,K]]",
            "dhost_null": "n_D=(1,-K^-1 B)",
            "aest_maxwell": "H_A=K_B G_E with G_E positive definite (I_3 in an orthonormal ADM basis)",
            "joint": "H_total=diag(H_D,H_A)",
            "linear_terms": "J^mu nabla_mu phi is affine in K_ij and aether velocity and shifts momenta without changing H_total",
        },
        "why_block_identity_is_background_independent": {
            "maxwell": "F_mu_nu=partial_mu A_nu-partial_nu A_mu contains no metric connection; the electric velocity form is a positive spatial-metric congruence",
            "scalar_first_derivatives": "Y, Q, and F(Y,Q) contain no V_* after B_mu=nabla_mu phi is introduced as an auxiliary coordinate",
            "aether_scalar_mixing": "A^nu nabla_nu A^mu B_mu is first order in the aether/metric velocities and contains no V_*",
            "constraint": "A^2=-1 is algebraic",
        },
        "random_audit": {
            "trials": int(random_trials),
            "maximum_dhost_null_residual": maximum_dhost_null_residual,
            "maximum_joint_null_residual": maximum_joint_null_residual,
            "maximum_finite_difference_hessian_residual": maximum_fd_residual,
            "minimum_aether_block_eigenvalue": minimum_aether_eigenvalue,
            "observed_dhost_inertias": [list(value) for value in sorted(observed_dhost_inertias)],
            "observed_joint_inertias": [list(value) for value in sorted(observed_joint_inertias)],
            "representative": representative,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "joint_kinetic_rank_subgate_pass": bool(all(gates.values())),
        "constraint_reduced": False,
        "dhost_primary_constraint_kinematically_present": True,
        "dhost_secondary_constraint_derived": False,
        "complete_dirac_chain_derived": False,
        "physical_degree_count_proven_unchanged": False,
        "complete_metric_stress_derived": False,
        "arbitrary_background_characteristics_proven": False,
        "theory_viable": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
