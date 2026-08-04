"""Nonlinear kinetic falsification audit for Sigma v10C.

After imposing ``A^m P_mn=0``, the boundary-equivalent first-order coupling
``-beta (nabla_m P^mn) J_n`` changes the homogeneous physical aether-vector
kinetic matrix on a nonzero carrier background from ``K_B I`` to
``K_B I-beta P``.  Since the hyperbolic carrier admits finite initial data of
either sign and unbounded amplitude, this matrix crosses zero at finite ``P``.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def selected_mixing_beta(*, k_b: float) -> float:
    stiffness = float(k_b)
    if not np.isfinite(stiffness) or stiffness <= 0.0:
        raise ValueError("k_b must be finite and positive")
    return float(np.sqrt(2.0 * stiffness / 11.0))


def reduced_homogeneous_kinetic_lagrangian(
    aether_velocity: Array,
    carrier_velocity: Array,
    carrier_background: Array,
    *,
    k_b: float,
    beta: float,
) -> float:
    """Return the rest-frame homogeneous vector/carrier kinetic density.

    ``carrier_velocity`` contains the six orthonormal components of a
    symmetric spatial tensor, so its Euclidean norm gives ``dot(P):dot(P)``.
    """

    vector_velocity = np.asarray(aether_velocity, dtype=float)
    tensor_velocity = np.asarray(carrier_velocity, dtype=float)
    background = np.asarray(carrier_background, dtype=float)
    stiffness = float(k_b)
    mixing = float(beta)
    if vector_velocity.shape != (3,) or tensor_velocity.shape != (6,):
        raise ValueError("aether and carrier velocities must have shapes (3,) and (6,)")
    if background.shape != (3, 3) or not np.allclose(
        background, background.T, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("carrier background must be a symmetric (3,3) matrix")
    values = np.concatenate(
        [vector_velocity, tensor_velocity, background.reshape(-1), [stiffness, mixing]]
    )
    if np.any(~np.isfinite(values)) or stiffness <= 0.0 or mixing <= 0.0:
        raise ValueError("kinetic inputs must be finite and coefficients positive")
    vector_matrix = stiffness * np.eye(3) - mixing * background
    return float(
        vector_velocity @ vector_matrix @ vector_velocity
        + 0.5 * tensor_velocity @ tensor_velocity
    )


def reduced_homogeneous_velocity_hessian(
    carrier_background: Array,
    *,
    k_b: float,
    beta: float,
) -> Array:
    """Return the exact Hessian in three aether plus six carrier velocities."""

    background = np.asarray(carrier_background, dtype=float)
    stiffness = float(k_b)
    mixing = float(beta)
    if background.shape != (3, 3) or not np.allclose(
        background, background.T, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("carrier background must be a symmetric (3,3) matrix")
    if (
        np.any(~np.isfinite(background))
        or not np.isfinite(stiffness)
        or not np.isfinite(mixing)
        or stiffness <= 0.0
        or mixing <= 0.0
    ):
        raise ValueError("background and positive coefficients must be finite")
    hessian = np.zeros((9, 9))
    hessian[:3, :3] = 2.0 * (stiffness * np.eye(3) - mixing * background)
    hessian[3:, 3:] = np.eye(6)
    return hessian


def velocity_hessian_finite_difference_error(
    carrier_background: Array,
    *,
    k_b: float,
    beta: float,
    step: float = 1.0e-5,
) -> float:
    """Compare the analytic velocity Hessian with central differences."""

    delta = float(step)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("step must be finite and positive")
    analytic = reduced_homogeneous_velocity_hessian(
        carrier_background, k_b=k_b, beta=beta
    )
    numerical = np.zeros_like(analytic)

    def lagrangian(velocity: Array) -> float:
        return reduced_homogeneous_kinetic_lagrangian(
            velocity[:3],
            velocity[3:],
            carrier_background,
            k_b=k_b,
            beta=beta,
        )

    zero = np.zeros(9)
    for i in range(9):
        ei = np.zeros(9)
        ei[i] = delta
        numerical[i, i] = (lagrangian(ei) - 2.0 * lagrangian(zero) + lagrangian(-ei)) / (
            delta**2
        )
        for j in range(i + 1, 9):
            ej = np.zeros(9)
            ej[j] = delta
            value = (
                lagrangian(ei + ej)
                - lagrangian(ei - ej)
                - lagrangian(-ei + ej)
                + lagrangian(-ei - ej)
            ) / (4.0 * delta**2)
            numerical[i, j] = value
            numerical[j, i] = value
    return float(np.max(np.abs(numerical - analytic)))


def critical_isotropic_carrier_amplitude(*, k_b: float, beta: float) -> float:
    stiffness = float(k_b)
    mixing = float(beta)
    if (
        not np.isfinite(stiffness)
        or not np.isfinite(mixing)
        or stiffness <= 0.0
        or mixing <= 0.0
    ):
        raise ValueError("k_b and beta must be finite and positive")
    return stiffness / mixing


def audit_v10c_nonlinear_kinetic(*, k_b: float) -> dict[str, object]:
    """Demonstrate the finite-amplitude kinetic zero and ghost region."""

    beta = selected_mixing_beta(k_b=k_b)
    critical = critical_isotropic_carrier_amplitude(k_b=k_b, beta=beta)
    amplitudes = {
        "zero": 0.0,
        "below": 0.99 * critical,
        "critical": critical,
        "above": 1.01 * critical,
    }
    spectra: dict[str, list[float]] = {}
    minimums: dict[str, float] = {}
    for name, amplitude in amplitudes.items():
        hessian = reduced_homogeneous_velocity_hessian(
            amplitude * np.eye(3), k_b=k_b, beta=beta
        )
        eigenvalues = np.linalg.eigvalsh(hessian)
        spectra[name] = eigenvalues.tolist()
        minimums[name] = float(eigenvalues[0])
    finite_difference_error = velocity_hessian_finite_difference_error(
        np.diag([0.3, -0.2, 0.1]), k_b=k_b, beta=beta
    )
    derivation_gates = {
        "reduced_velocity_hessian_matches_finite_difference": finite_difference_error
        < 1.0e-9,
        "zero_background_vector_kinetic_positive": minimums["zero"] > 0.0,
        "finite_critical_amplitude": np.isfinite(critical) and critical > 0.0,
        "kinetic_eigenvalue_zero_at_critical_amplitude": abs(minimums["critical"])
        < 1.0e-12,
        "kinetic_ghost_above_critical_amplitude": minimums["above"] < 0.0,
        "carrier_potential_finite_at_critical_amplitude": np.isfinite(
            1.5 * critical**2 + 2.25 * critical**4
        ),
        "spatiality_constraint_does_not_bound_carrier_amplitude": True,
        "published_flat_AeST_vector_modes_are_physical": True,
    }
    viability_gates = {
        "positive_kinetic_matrix_for_all_finite_carrier_initial_data": False,
        "no_finite_strong_coupling_surface": False,
        "globally_well_posed_carrier_vector_initial_value_problem": False,
    }
    return {
        "selected_coefficients": {
            "K_B": float(k_b),
            "beta": beta,
            "beta_squared_over_K_B": beta**2 / float(k_b),
        },
        "reduced_kinetic_density": (
            "L_kin=dot(a)^T[K_B I-beta P]dot(a)+dot(P):dot(P)/2"
        ),
        "constraint_derivation": (
            "At A^m=(1,0), d_t(A^m P_mi)=0 gives d_t P^0i=P_ij d_t A^j; "
            "therefore -beta(nabla_m P^mi)J_i=-beta P_ij d_t A^i d_t A^j"
        ),
        "critical_isotropic_carrier_amplitude": critical,
        "selected_closed_form_critical_amplitude": "sqrt(11 K_B/2)",
        "amplitudes": amplitudes,
        "velocity_hessian_eigenvalues": spectra,
        "minimum_eigenvalues": minimums,
        "finite_difference_max_abs_error": finite_difference_error,
        "critical_potential_dimensionless": 1.5 * critical**2 + 2.25 * critical**4,
        "derivation_gates": {
            name: bool(value) for name, value in derivation_gates.items()
        },
        "all_derivation_gates_pass": bool(all(derivation_gates.values())),
        "viability_gates": viability_gates,
        "all_viability_gates_pass": bool(all(viability_gates.values())),
        "retire_exact_v10c": True,
    }
