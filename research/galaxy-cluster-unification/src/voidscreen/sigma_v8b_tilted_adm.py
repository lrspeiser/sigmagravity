"""Necessary tilted-ADM kinetic gate for the Sigma v8B completion.

This module studies a homogeneous scalar on a local ADM patch while allowing
the unit aether to be tilted relative to the slice normal.  In this setting the
v8B projected Hessian contains a scalar normal acceleration.  An exact boundary
subtraction converts it to a first-order metric--aether--scalar Lagrangian.

The resulting ten-by-ten Hessian covers the six spatial-metric velocities,
three spatial-aether velocities, and scalar normal velocity.  Full rank gives a
conditional six-degree-of-freedom patch when combined with four diffeomorphism
constraints.  It is a necessary local gate, not the full inhomogeneous Dirac
analysis or a Hamiltonian-positivity proof.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import brentq

Array = np.ndarray
TORCH_DTYPE = torch.float64


@dataclass(frozen=True)
class CompletionAdmIdentity:
    original_density: float
    first_order_density: float
    boundary_derivative_density: float
    residual: float


@dataclass(frozen=True)
class KineticPoint:
    base_hessian: Array
    combined_hessian: Array
    base_singular_values: Array
    combined_singular_values: Array
    base_inertia: tuple[int, int, int]
    combined_inertia: tuple[int, int, int]
    determinant_ratio: float


@dataclass(frozen=True)
class CanonicalPoint:
    lagrangian: float
    canonical_energy: float
    momenta: Array


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def aether_spatial_norm(aether_velocity: float) -> float:
    """Return ``|A_i|=v/sqrt(1-v^2)`` for the local ADM frame."""

    velocity = _finite(aether_velocity, name="aether_velocity")
    if not 0.0 <= velocity < 1.0:
        raise ValueError("aether_velocity must lie in [0, 1)")
    return velocity / np.sqrt(1.0 - velocity**2)


def _completion_antiderivative_terms(
    *,
    spatial_norm_squared: float,
    sigma: float,
    q_0: float,
) -> tuple[float, float, float, float]:
    x = _finite(spatial_norm_squared, name="spatial_norm_squared")
    scalar_velocity = _finite(sigma, name="sigma")
    q_zero = _finite(q_0, name="q_0")
    if x < 0.0:
        raise ValueError("spatial_norm_squared must be non-negative")
    chi = np.sqrt(1.0 + x)
    displacement = chi * scalar_velocity - q_zero
    antiderivative = x * displacement**3 / (3.0 * chi)
    derivative_x = (
        displacement**3 / (3.0 * chi)
        + x * displacement**2 * scalar_velocity / (2.0 * chi**2)
        - x * displacement**3 / (6.0 * chi**3)
    )
    return chi, displacement, antiderivative, derivative_x


def completion_adm_boundary_identity(
    *,
    spatial_norm_squared: float,
    sigma: float,
    sigma_normal_derivative: float,
    extrinsic_trace: float,
    aether_extrinsic_projection: float,
    aether_electric_contraction: float,
    q_0: float,
    coefficient: float,
) -> CompletionAdmIdentity:
    """Check the exact boundary reduction of the homogeneous completion.

    With ``x=A_i A^i``, ``K_A=K^ij A_i A_j`` and ``A.E=A^i E_i``, one has
    ``L_n x=2(A.E-K_A)``.  The acceleration term is removed using
    ``F=x(Q-Q0)^3/(3 chi)``.
    """

    x = _finite(spatial_norm_squared, name="spatial_norm_squared")
    scalar_velocity = _finite(sigma, name="sigma")
    scalar_acceleration = _finite(
        sigma_normal_derivative,
        name="sigma_normal_derivative",
    )
    trace = _finite(extrinsic_trace, name="extrinsic_trace")
    projected = _finite(
        aether_extrinsic_projection,
        name="aether_extrinsic_projection",
    )
    electric = _finite(
        aether_electric_contraction,
        name="aether_electric_contraction",
    )
    coupling = _finite(coefficient, name="coefficient")
    _, displacement, antiderivative, derivative_x = (
        _completion_antiderivative_terms(
            spatial_norm_squared=x,
            sigma=scalar_velocity,
            q_0=q_0,
        )
    )
    original = coupling * displacement**2 * (
        x * scalar_acceleration - scalar_velocity * (trace + projected)
    )
    first_order = -coupling * (
        trace * (antiderivative + scalar_velocity * displacement**2)
        + projected * (scalar_velocity * displacement**2 - 2.0 * derivative_x)
        + 2.0 * derivative_x * electric
    )
    x_derivative = 2.0 * (electric - projected)
    boundary = coupling * (
        x * displacement**2 * scalar_acceleration
        + derivative_x * x_derivative
        + trace * antiderivative
    )
    return CompletionAdmIdentity(
        original_density=original,
        first_order_density=first_order,
        boundary_derivative_density=boundary,
        residual=original - first_order - boundary,
    )


def _symmetric_matrix_from_velocities(velocities: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.stack((velocities[0], velocities[3], velocities[4])),
            torch.stack((velocities[3], velocities[1], velocities[5])),
            torch.stack((velocities[4], velocities[5], velocities[2])),
        )
    )


def _simple_aqual_free_function(y: torch.Tensor, *, identically_zero: bool) -> torch.Tensor:
    if identically_zero:
        return y * 0.0
    root = torch.sqrt(y)
    return y - 2.0 * root + 2.0 * torch.log1p(root)


def _homogeneous_adm_lagrangian(
    velocities: torch.Tensor,
    *,
    aether_velocity: float,
    a_sigma_over_q0: float,
    ell_h_q0: float,
    k_b: float,
    k_2: float,
    alpha: float,
    include_completion: bool,
) -> torch.Tensor:
    extrinsic = _symmetric_matrix_from_velocities(velocities)
    electric = velocities[6:9]
    sigma = velocities[9]
    spatial_norm = aether_spatial_norm(aether_velocity)
    aether = torch.tensor((spatial_norm, 0.0, 0.0), dtype=TORCH_DTYPE)
    x = spatial_norm**2
    chi = np.sqrt(1.0 + x)
    q_value = chi * sigma
    displacement = q_value - 1.0
    trace = torch.trace(extrinsic)
    projected = aether @ (extrinsic @ aether)
    electric_contraction = aether @ electric
    y = x * sigma**2
    ratio = float(a_sigma_over_q0)
    aqual = _simple_aqual_free_function(
        y / ratio**2,
        identically_zero=x == 0.0,
    )
    base = (
        torch.sum(extrinsic**2)
        - trace**2
        + k_b * torch.sum(electric**2)
        + 2.0 * (2.0 - k_b) * sigma * electric_contraction
        - (2.0 - k_b) * y
        - (2.0 - k_b) * ratio**2 * aqual
        + 2.0 * k_2 * displacement**2
    )
    if not include_completion:
        return base
    coupling = (alpha - 1.0) * ell_h_q0**2
    antiderivative = x * displacement**3 / (3.0 * chi)
    derivative_x = (
        displacement**3 / (3.0 * chi)
        + x * displacement**2 * sigma / (2.0 * chi**2)
        - x * displacement**3 / (6.0 * chi**3)
    )
    completion = -coupling * (
        trace * (antiderivative + sigma * displacement**2)
        + projected * (sigma * displacement**2 - 2.0 * derivative_x)
        + 2.0 * derivative_x * electric_contraction
    )
    return base + completion


def homogeneous_adm_lagrangian(
    velocities: Iterable[float],
    *,
    aether_velocity: float,
    a_sigma_over_q0: float,
    ell_h_q0: float,
    k_b: float = 1.0,
    k_2: float = 2.0,
    alpha: float = 16.0 / 9.0,
    include_completion: bool = True,
) -> float:
    """Evaluate the dimensionless local homogeneous ADM Lagrangian.

    This public scalar evaluator permits finite-difference checks that are
    independent of the automatic-differentiation Hessian used by the audit.
    The velocity order is ``K_xx, K_yy, K_zz, K_xy, K_xz, K_yz, E_x, E_y,
    E_z, sigma``.
    """

    values = np.asarray(tuple(velocities), dtype=float)
    if values.shape != (10,) or np.any(~np.isfinite(values)):
        raise ValueError("velocities must contain ten finite values")
    tensor = torch.tensor(values, dtype=TORCH_DTYPE)
    result = _homogeneous_adm_lagrangian(
        tensor,
        aether_velocity=aether_velocity,
        a_sigma_over_q0=a_sigma_over_q0,
        ell_h_q0=ell_h_q0,
        k_b=k_b,
        k_2=k_2,
        alpha=alpha,
        include_completion=bool(include_completion),
    )
    value = float(result.detach().cpu())
    if not np.isfinite(value):
        raise ValueError("homogeneous ADM Lagrangian is not finite")
    return value


def centered_finite_difference_hessian(
    velocities: Iterable[float],
    *,
    aether_velocity: float,
    a_sigma_over_q0: float,
    ell_h_q0: float,
    step: float = 2.0e-4,
    k_b: float = 1.0,
    k_2: float = 2.0,
    alpha: float = 16.0 / 9.0,
    include_completion: bool = True,
) -> Array:
    """Return an independent centered finite-difference velocity Hessian."""

    center = np.asarray(tuple(velocities), dtype=float)
    if center.shape != (10,) or np.any(~np.isfinite(center)):
        raise ValueError("velocities must contain ten finite values")
    delta = _finite(step, name="step")
    if delta <= 0.0:
        raise ValueError("step must be positive")

    def evaluate(values: Array) -> float:
        return homogeneous_adm_lagrangian(
            values,
            aether_velocity=aether_velocity,
            a_sigma_over_q0=a_sigma_over_q0,
            ell_h_q0=ell_h_q0,
            k_b=k_b,
            k_2=k_2,
            alpha=alpha,
            include_completion=include_completion,
        )

    size = center.size
    hessian = np.empty((size, size), dtype=float)
    center_value = evaluate(center)
    for row in range(size):
        row_step = np.zeros(size, dtype=float)
        row_step[row] = delta
        hessian[row, row] = (
            evaluate(center + row_step)
            - 2.0 * center_value
            + evaluate(center - row_step)
        ) / delta**2
        for column in range(row):
            column_step = np.zeros(size, dtype=float)
            column_step[column] = delta
            hessian[row, column] = hessian[column, row] = (
                evaluate(center + row_step + column_step)
                - evaluate(center + row_step - column_step)
                - evaluate(center - row_step + column_step)
                + evaluate(center - row_step - column_step)
            ) / (4.0 * delta**2)
    return hessian


def homogeneous_canonical_point(
    velocities: Iterable[float],
    *,
    aether_velocity: float,
    a_sigma_over_q0: float,
    ell_h_q0: float,
    k_b: float = 1.0,
    k_2: float = 2.0,
    alpha: float = 16.0 / 9.0,
    include_completion: bool = True,
) -> CanonicalPoint:
    """Return the local velocity momenta and canonical energy density."""

    values = np.asarray(tuple(velocities), dtype=float)
    if values.shape != (10,) or np.any(~np.isfinite(values)):
        raise ValueError("velocities must contain ten finite values")
    tensor = torch.tensor(values, dtype=TORCH_DTYPE, requires_grad=True)
    lagrangian = _homogeneous_adm_lagrangian(
        tensor,
        aether_velocity=aether_velocity,
        a_sigma_over_q0=a_sigma_over_q0,
        ell_h_q0=ell_h_q0,
        k_b=k_b,
        k_2=k_2,
        alpha=alpha,
        include_completion=bool(include_completion),
    )
    momenta_tensor = torch.autograd.grad(lagrangian, tensor)[0]
    energy_tensor = tensor @ momenta_tensor - lagrangian
    momenta = momenta_tensor.detach().cpu().numpy()
    lagrangian_value = float(lagrangian.detach().cpu())
    energy_value = float(energy_tensor.detach().cpu())
    if not (
        np.isfinite(lagrangian_value)
        and np.isfinite(energy_value)
        and np.all(np.isfinite(momenta))
    ):
        raise ValueError("homogeneous canonical point is not finite")
    return CanonicalPoint(
        lagrangian=lagrangian_value,
        canonical_energy=energy_value,
        momenta=momenta,
    )


def _inertia(eigenvalues: Array, *, tolerance: float = 1.0e-9) -> tuple[int, int, int]:
    negative = int(np.sum(eigenvalues < -tolerance))
    zero = int(np.sum(np.abs(eigenvalues) <= tolerance))
    positive = int(np.sum(eigenvalues > tolerance))
    return negative, zero, positive


def homogeneous_kinetic_point(
    *,
    aether_velocity: float,
    q_over_q0: float,
    a_sigma_over_q0: float,
    ell_h_q0: float,
    background_kinematics: Iterable[float] | None = None,
    k_b: float = 1.0,
    k_2: float = 2.0,
    alpha: float = 16.0 / 9.0,
) -> KineticPoint:
    """Return the base and combined local ADM velocity Hessians."""

    velocity = _finite(aether_velocity, name="aether_velocity")
    q_ratio = _finite(q_over_q0, name="q_over_q0")
    acceleration_ratio = _finite(a_sigma_over_q0, name="a_sigma_over_q0")
    length_ratio = _finite(ell_h_q0, name="ell_h_q0")
    vector_coupling = _finite(k_b, name="k_b")
    clock = _finite(k_2, name="k_2")
    completion = _finite(alpha, name="alpha")
    if not 0.0 <= velocity < 1.0:
        raise ValueError("aether_velocity must lie in [0, 1)")
    if q_ratio <= 0.0 or acceleration_ratio <= 0.0:
        raise ValueError("q_over_q0 and a_sigma_over_q0 must be positive")
    if length_ratio < 0.0 or not 0.0 < vector_coupling < 2.0:
        raise ValueError("ell_h_q0 must be non-negative and K_B must lie in (0, 2)")
    if clock <= 0.0 or completion < 1.0:
        raise ValueError("K_2 must be positive and alpha must be at least one")
    if background_kinematics is None:
        background = np.zeros(9, dtype=float)
    else:
        background = np.asarray(tuple(background_kinematics), dtype=float)
        if background.shape != (9,) or np.any(~np.isfinite(background)):
            raise ValueError("background_kinematics must contain nine finite values")
    spatial_norm = aether_spatial_norm(velocity)
    chi = np.sqrt(1.0 + spatial_norm**2)
    initial = np.concatenate((background, np.array((q_ratio / chi,))))
    tensor = torch.tensor(initial, dtype=TORCH_DTYPE, requires_grad=True)
    common = {
        "aether_velocity": velocity,
        "a_sigma_over_q0": acceleration_ratio,
        "ell_h_q0": length_ratio,
        "k_b": vector_coupling,
        "k_2": clock,
        "alpha": completion,
    }
    base_tensor = torch.autograd.functional.hessian(
        lambda values: _homogeneous_adm_lagrangian(
            values,
            include_completion=False,
            **common,
        ),
        tensor,
    )
    combined_tensor = torch.autograd.functional.hessian(
        lambda values: _homogeneous_adm_lagrangian(
            values,
            include_completion=True,
            **common,
        ),
        tensor,
    )
    base = base_tensor.detach().cpu().numpy()
    combined = combined_tensor.detach().cpu().numpy()
    base_singular = np.linalg.svd(base, compute_uv=False)
    combined_singular = np.linalg.svd(combined, compute_uv=False)
    base_eigenvalues = np.linalg.eigvalsh(base)
    combined_eigenvalues = np.linalg.eigvalsh(combined)
    base_sign, base_logdet = np.linalg.slogdet(base)
    combined_sign, combined_logdet = np.linalg.slogdet(combined)
    determinant_ratio = float(
        combined_sign / base_sign * np.exp(combined_logdet - base_logdet)
    )
    return KineticPoint(
        base_hessian=base,
        combined_hessian=combined,
        base_singular_values=base_singular,
        combined_singular_values=combined_singular,
        base_inertia=_inertia(base_eigenvalues),
        combined_inertia=_inertia(combined_eigenvalues),
        determinant_ratio=determinant_ratio,
    )


def find_outside_envelope_singularity(
    *,
    aether_velocity: float = 0.97,
    lower_q_ratio: float = 2.8,
    upper_q_ratio: float = 2.9,
    a_sigma_over_q0: float = 1.0,
    ell_h_q0: float = 1.0,
) -> dict[str, float]:
    """Locate a finite nonlinear rank-changing surface outside the safe patch."""

    def determinant_ratio(q_ratio: float) -> float:
        return homogeneous_kinetic_point(
            aether_velocity=aether_velocity,
            q_over_q0=q_ratio,
            a_sigma_over_q0=a_sigma_over_q0,
            ell_h_q0=ell_h_q0,
        ).determinant_ratio

    lower = determinant_ratio(lower_q_ratio)
    upper = determinant_ratio(upper_q_ratio)
    if lower * upper >= 0.0:
        raise ValueError("singularity bracket must straddle a determinant sign change")
    root = float(brentq(determinant_ratio, lower_q_ratio, upper_q_ratio, xtol=1.0e-11))
    point = homogeneous_kinetic_point(
        aether_velocity=aether_velocity,
        q_over_q0=root,
        a_sigma_over_q0=a_sigma_over_q0,
        ell_h_q0=ell_h_q0,
    )
    return {
        "aether_velocity": float(aether_velocity),
        "invariant_Y_over_Q_squared": float(aether_velocity**2),
        "q_over_q0": root,
        "determinant_ratio": point.determinant_ratio,
        "minimum_combined_singular_value": float(point.combined_singular_values[-1]),
    }


def audit_tilted_adm_kinetic_gate(
    *,
    deterministic_velocities: Iterable[float],
    deterministic_q_ratios: Iterable[float],
    a_sigma_ratios: Iterable[float],
    maximum_ell_h_q0: float,
    maximum_background_kinematic: float,
    random_samples: int,
    random_seed: int,
) -> dict[str, object]:
    """Scan the declared necessary local kinetic envelope."""

    velocities = tuple(float(item) for item in deterministic_velocities)
    q_ratios = tuple(float(item) for item in deterministic_q_ratios)
    acceleration_ratios = tuple(float(item) for item in a_sigma_ratios)
    maximum_length = _finite(maximum_ell_h_q0, name="maximum_ell_h_q0")
    maximum_background = _finite(
        maximum_background_kinematic,
        name="maximum_background_kinematic",
    )
    samples = int(random_samples)
    if not velocities or not q_ratios or not acceleration_ratios:
        raise ValueError("deterministic scan axes must be non-empty")
    if maximum_length < 0.0 or maximum_background < 0.0 or samples < 0:
        raise ValueError("scan maxima and random_samples must be non-negative")
    records: list[tuple[KineticPoint, dict[str, float | str]]] = []
    for acceleration_ratio in acceleration_ratios:
        for velocity in velocities:
            for q_ratio in q_ratios:
                point = homogeneous_kinetic_point(
                    aether_velocity=velocity,
                    q_over_q0=q_ratio,
                    a_sigma_over_q0=acceleration_ratio,
                    ell_h_q0=maximum_length,
                )
                records.append(
                    (
                        point,
                        {
                            "kind": "deterministic",
                            "aether_velocity": velocity,
                            "q_over_q0": q_ratio,
                            "a_sigma_over_q0": acceleration_ratio,
                            "ell_h_q0": maximum_length,
                        },
                    )
                )
    rng = np.random.default_rng(int(random_seed))
    log_min = np.log10(min(acceleration_ratios))
    log_max = np.log10(max(acceleration_ratios))
    for _ in range(samples):
        velocity = float(rng.uniform(min(velocities), max(velocities)))
        q_ratio = float(rng.uniform(min(q_ratios), max(q_ratios)))
        acceleration_ratio = float(10.0 ** rng.uniform(log_min, log_max))
        length_ratio = float(rng.uniform(0.0, maximum_length))
        background = rng.uniform(-maximum_background, maximum_background, size=9)
        point = homogeneous_kinetic_point(
            aether_velocity=velocity,
            q_over_q0=q_ratio,
            a_sigma_over_q0=acceleration_ratio,
            ell_h_q0=length_ratio,
            background_kinematics=background,
        )
        records.append(
            (
                point,
                {
                    "kind": "random",
                    "aether_velocity": velocity,
                    "q_over_q0": q_ratio,
                    "a_sigma_over_q0": acceleration_ratio,
                    "ell_h_q0": length_ratio,
                },
            )
        )
    minimum_ratio_point, minimum_ratio_location = min(
        records,
        key=lambda item: item[0].determinant_ratio,
    )
    minimum_singular_point, minimum_singular_location = min(
        records,
        key=lambda item: item[0].combined_singular_values[-1],
    )
    nonpositive_determinants = int(
        sum(
        item.determinant_ratio <= 0.0 for item, _ in records
        )
    )
    inertia_mismatches = int(
        sum(item.base_inertia != item.combined_inertia for item, _ in records)
    )
    base_rank_failures = int(
        sum(item.base_singular_values[-1] <= 1.0e-8 for item, _ in records)
    )
    combined_rank_failures = int(
        sum(item.combined_singular_values[-1] <= 1.0e-8 for item, _ in records)
    )
    boundary_identity = completion_adm_boundary_identity(
        spatial_norm_squared=0.7,
        sigma=0.8,
        sigma_normal_derivative=-0.2,
        extrinsic_trace=0.13,
        aether_extrinsic_projection=-0.07,
        aether_electric_contraction=0.11,
        q_0=1.0,
        coefficient=7.0 / 9.0,
    )
    representative_velocity = 0.5 * (min(velocities) + max(velocities))
    representative_q_ratio = 0.5 * (min(q_ratios) + max(q_ratios))
    representative_acceleration_ratio = float(
        np.sqrt(min(acceleration_ratios) * max(acceleration_ratios))
    )
    representative_length_ratio = 0.8 * maximum_length
    representative_background = maximum_background * np.array(
        (0.5, -0.4, 0.3, 0.2, -0.1, 0.35, 0.4, -0.3, 0.1),
        dtype=float,
    )
    representative_spatial_norm = aether_spatial_norm(representative_velocity)
    representative_chi = np.sqrt(1.0 + representative_spatial_norm**2)
    representative_velocities = np.concatenate(
        (
            representative_background,
            np.array((representative_q_ratio / representative_chi,)),
        )
    )
    representative_point = homogeneous_kinetic_point(
        aether_velocity=representative_velocity,
        q_over_q0=representative_q_ratio,
        a_sigma_over_q0=representative_acceleration_ratio,
        ell_h_q0=representative_length_ratio,
        background_kinematics=representative_background,
    )
    finite_difference_hessian = centered_finite_difference_hessian(
        representative_velocities,
        aether_velocity=representative_velocity,
        a_sigma_over_q0=representative_acceleration_ratio,
        ell_h_q0=representative_length_ratio,
    )
    hessian_difference = (
        finite_difference_hessian - representative_point.combined_hessian
    )
    hessian_relative_error = float(
        np.linalg.norm(hessian_difference)
        / np.linalg.norm(representative_point.combined_hessian)
    )
    singularity = find_outside_envelope_singularity()
    completed_subgates = {
        "tilted_scalar_acceleration_removed_by_exact_boundary_term": bool(
            abs(boundary_identity.residual) < 1.0e-12
        ),
        "declared_envelope_base_legendre_map_full_rank": bool(
            base_rank_failures == 0
        ),
        "declared_envelope_combined_legendre_map_full_rank": bool(
            combined_rank_failures == 0
        ),
        "declared_envelope_has_no_determinant_sign_change": bool(
            nonpositive_determinants == 0
        ),
        "declared_envelope_preserves_kinetic_inertia": bool(
            inertia_mismatches == 0
        ),
        "autodiff_hessian_matches_centered_finite_difference": (
            hessian_relative_error < 1.0e-6
        ),
        "finite_outside_envelope_rank_surface_identified": abs(
            singularity["determinant_ratio"]
        )
        < 1.0e-7,
    }
    unresolved_kill_gates = {
        "full_inhomogeneous_dirac_algebra_closed": False,
        "global_legendre_map_regular": False,
        "hamiltonian_bounded_on_all_claimed_backgrounds": False,
        "full_coupled_characteristic_determinant_causal": False,
    }
    return {
        "velocity_order": [
            "K_xx",
            "K_yy",
            "K_zz",
            "K_xy",
            "K_xz",
            "K_yz",
            "E_x",
            "E_y",
            "E_z",
            "sigma",
        ],
        "declared_envelope": {
            "aether_velocity_maximum": max(velocities),
            "invariant_Y_over_Q_squared_maximum_on_homogeneous_patch": max(
                velocities
            )
            ** 2,
            "q_over_q0_minimum": min(q_ratios),
            "q_over_q0_maximum": max(q_ratios),
            "a_sigma_over_q0_minimum": min(acceleration_ratios),
            "a_sigma_over_q0_maximum": max(acceleration_ratios),
            "ell_h_q0_maximum": maximum_length,
            "absolute_K_and_E_over_q0_random_maximum": maximum_background,
        },
        "sample_counts": {
            "deterministic": len(velocities)
            * len(q_ratios)
            * len(acceleration_ratios),
            "random": samples,
            "total": len(records),
        },
        "boundary_identity_residual": float(boundary_identity.residual),
        "representative_hessian_finite_difference_relative_error": (
            hessian_relative_error
        ),
        "representative_hessian_finite_difference_maximum_absolute_error": float(
            np.max(np.abs(hessian_difference))
        ),
        "minimum_determinant_ratio": minimum_ratio_point.determinant_ratio,
        "minimum_determinant_ratio_location": minimum_ratio_location,
        "minimum_combined_singular_value": float(
            minimum_singular_point.combined_singular_values[-1]
        ),
        "minimum_combined_singular_value_location": minimum_singular_location,
        "nonpositive_determinants": nonpositive_determinants,
        "base_rank_failures": base_rank_failures,
        "combined_rank_failures": combined_rank_failures,
        "inertia_mismatches": inertia_mismatches,
        "representative_inertia": records[0][0].combined_inertia,
        "outside_envelope_singularity": singularity,
        "conditional_local_dof_count": 6.0,
        "completed_subgates": completed_subgates,
        "all_completed_subgates_pass": bool(all(completed_subgates.values())),
        "unresolved_kill_gates": unresolved_kill_gates,
        "full_hamiltonian_gate_pass": bool(
            all(completed_subgates.values()) and all(unresolved_kill_gates.values())
        ),
    }
