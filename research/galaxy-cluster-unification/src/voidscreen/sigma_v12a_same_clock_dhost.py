"""Theory-only selection checks for the Sigma v12A same-clock DHOST lane."""

from __future__ import annotations

import numpy as np

from voidscreen.sigma_degenerate_action import (
    luminal_class_ia_coefficients,
    normalized_dhost_residuals,
)
from voidscreen.sigma_v8_aest_galileon import aest_linear_spectrum


def same_clock_activation(
    scalar_kinetic_ratio,
    *,
    background_kinetic_ratio: float,
) -> np.ndarray:
    """Return the v12A background-zero, high-field-suppressed activation.

    With ``x=X/q_sigma^2``, ``x0=-Q0^2/q_sigma^2`` and ``d=x-x0``, the
    dimensionless shape is

    ``d^2 / [(1+d^2)^(3/2) sqrt(1+x^2)]``.

    It and its first derivative vanish on the AeST clock background.  The
    extra ``sqrt(1+x^2)`` factor also bounds the ``x^2 A3^2`` combinations in
    the dependent Class-Ia coefficients.
    """

    ratio = np.asarray(scalar_kinetic_ratio, dtype=float)
    background = float(background_kinetic_ratio)
    if np.any(~np.isfinite(ratio)) or not np.isfinite(background):
        raise ValueError("kinetic ratios must be finite")
    offset = ratio - background
    return np.square(offset) / (
        np.power(1.0 + np.square(offset), 1.5) * np.sqrt(1.0 + np.square(ratio))
    )


def v12a_dimensionless_coefficients(
    scalar_kinetic_ratio,
    *,
    background_kinetic_ratio: float,
    orientation_strength: float,
) -> dict[str, np.ndarray]:
    """Return normalized ``A1..A5`` for constant-F luminal Class Ia.

    Dimensional factors ``F0/q_sigma^4`` and ``F0/q_sigma^6`` are restored in
    the action document.  The normalized identities use ``F=1`` and
    ``F_X=0`` and are sufficient to audit degeneracy and bounded shape.
    """

    ratio = np.asarray(scalar_kinetic_ratio, dtype=float)
    strength = float(orientation_strength)
    if np.any(~np.isfinite(ratio)) or not np.isfinite(strength):
        raise ValueError("coefficient inputs must be finite")
    a3 = strength * same_clock_activation(ratio, background_kinetic_ratio=background_kinetic_ratio)
    return luminal_class_ia_coefficients(ratio, 1.0, 0.0, a3)


def static_dhost_invariants(
    scalar_gradient,
    scalar_hessian,
) -> dict[str, float]:
    """Return the static directional reductions of ``L3``, ``L4`` and ``L5``.

    On a locally Cartesian static background these are

    ``L3=tr(H) g.H.g``, ``L4=g.H^2.g`` and ``L5=(g.H.g)^2``.
    """

    gradient = np.asarray(scalar_gradient, dtype=float)
    hessian = np.asarray(scalar_hessian, dtype=float)
    if (
        gradient.shape != (3,)
        or hessian.shape != (3, 3)
        or np.any(~np.isfinite(gradient))
        or np.any(~np.isfinite(hessian))
        or not np.allclose(hessian, hessian.T, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("static DHOST probe requires a finite 3-vector and symmetric 3x3 Hessian")
    projected = float(gradient @ hessian @ gradient)
    projected_square = float(gradient @ hessian @ hessian @ gradient)
    return {
        "trace_hessian": float(np.trace(hessian)),
        "gradient_hessian_gradient": projected,
        "L3": float(np.trace(hessian) * projected),
        "L4": projected_square,
        "L5": projected**2,
    }


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q @ np.diag(signs)
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def audit_v12a_same_clock_dhost(
    *,
    k_b: float,
    k_2: float,
    lambda_s: float,
    orientation_strength: float,
    background_kinetic_ratios: list[float],
    signed_scan_limit: float,
    signed_scan_points: int,
    high_acceleration_ratio: float,
    random_rotation_trials: int,
    random_seed: int,
) -> dict[str, object]:
    if signed_scan_limit <= 1.0 or signed_scan_points < 3:
        raise ValueError("coefficient scan must span a finite signed interval")
    if high_acceleration_ratio <= 1.0 or random_rotation_trials < 1:
        raise ValueError("high-field and rotation audits require positive coverage")
    spectrum = aest_linear_spectrum(k_b=k_b, k_2=k_2, lambda_s=lambda_s)
    magnitude = np.geomspace(1.0e-12, float(signed_scan_limit), signed_scan_points)
    ratio = np.concatenate((-magnitude[::-1], [0.0], magnitude))
    coefficient_audits: list[dict[str, float]] = []
    maximum_residual = 0.0
    maximum_coefficient = 0.0
    for background in background_kinetic_ratios:
        coefficients = v12a_dimensionless_coefficients(
            ratio,
            background_kinetic_ratio=float(background),
            orientation_strength=float(orientation_strength),
        )
        residuals = normalized_dhost_residuals(
            ratio, np.ones_like(ratio), np.zeros_like(ratio), coefficients
        )
        local_residual = max(float(np.max(np.abs(residual))) for residual in residuals.values())
        local_coefficient = max(
            float(np.max(np.abs(coefficient))) for coefficient in coefficients.values()
        )
        at_background = v12a_dimensionless_coefficients(
            np.asarray([background]),
            background_kinetic_ratio=float(background),
            orientation_strength=float(orientation_strength),
        )
        maximum_residual = max(maximum_residual, local_residual)
        maximum_coefficient = max(maximum_coefficient, local_coefficient)
        coefficient_audits.append(
            {
                "background_kinetic_ratio": float(background),
                "maximum_normalized_degeneracy_residual": local_residual,
                "maximum_absolute_normalized_coefficient": local_coefficient,
                "maximum_background_coefficient": max(
                    float(np.max(np.abs(value))) for value in at_background.values()
                ),
            }
        )

    gradient = np.asarray([1.0, 0.0, 0.0])
    isotropic = np.eye(3)
    rank_one = np.diag([3.0, 0.0, 0.0])
    isotropic_invariants = static_dhost_invariants(gradient, isotropic)
    rank_one_invariants = static_dhost_invariants(gradient, rank_one)
    directional_difference = max(
        abs(rank_one_invariants[name] - isotropic_invariants[name]) for name in ("L3", "L4", "L5")
    )

    rng = np.random.default_rng(int(random_seed))
    rotation_residual = 0.0
    for _ in range(int(random_rotation_trials)):
        trial_gradient = rng.normal(size=3)
        trial_hessian = rng.normal(size=(3, 3))
        trial_hessian = 0.5 * (trial_hessian + trial_hessian.T)
        rotation = _random_rotation(rng)
        before = static_dhost_invariants(trial_gradient, trial_hessian)
        after = static_dhost_invariants(
            rotation @ trial_gradient,
            rotation @ trial_hessian @ rotation.T,
        )
        for name in ("trace_hessian", "L3", "L4", "L5"):
            scale = max(1.0, abs(before[name]), abs(after[name]))
            rotation_residual = max(rotation_residual, abs(before[name] - after[name]) / scale)

    # On the static AeST branch, x-x0=Y/q_sigma^2=(g/a_sigma)^2.
    high_delta = float(high_acceleration_ratio) ** 2
    representative_background = float(background_kinetic_ratios[0])
    high_activation = float(
        same_clock_activation(
            representative_background + high_delta,
            background_kinetic_ratio=representative_background,
        )
    )
    background_activation = float(
        same_clock_activation(
            representative_background,
            background_kinetic_ratio=representative_background,
        )
    )
    near_background_step = 1.0e-6
    derivative_at_background = float(
        (
            same_clock_activation(
                representative_background + near_background_step,
                background_kinetic_ratio=representative_background,
            )
            - same_clock_activation(
                representative_background - near_background_step,
                background_kinetic_ratio=representative_background,
            )
        )
        / (2.0 * near_background_step)
    )
    gates = {
        "one_physical_metric": True,
        "no_new_memory_field": True,
        "five_or_fewer_universal_constants": True,
        "background_interaction_zero": abs(background_activation) < 1.0e-15,
        "background_first_derivative_zero": abs(derivative_at_background) < 1.0e-12,
        "luminal_class_ia_degeneracy": maximum_residual < 1.0e-12,
        "normalized_coefficients_finite": bool(np.isfinite(maximum_coefficient)),
        "flat_propagating_modes_positive": bool(spectrum["positive_propagating_modes"]),
        "flat_propagating_modes_causal": bool(spectrum["causal_propagating_modes"]),
        "tensor_front_luminal": spectrum["tensor_speed_squared"] == 1.0,
        "high_acceleration_activation_below_1e-5": high_activation < 1.0e-5,
        "equal_trace_geometry_distinguished": directional_difference > 1.0,
        "static_invariants_rotation_covariant": rotation_residual < 1.0e-12,
    }
    return {
        "candidate": "Sigma v12A same-AeST-clock luminal DHOST geometry",
        "mechanism_reset": {
            "retired_family": "independent material or memory fields carrying positive spatial strain",
            "new_family": "no new field; exact degenerate Hessian operators of the existing AeST scalar",
            "why_materially_distinct": "The directional state is a derivative of the baryon-forced AeST clock and the higher-derivative mode is removed algebraically; no material coordinates, response multipliers, or free halo profile are added.",
        },
        "action_identity": {
            "basis": "S=S_AeST+integral sqrt(-g)[A3 L3+A4 L4+A5 L5]",
            "tensor_condition": "A1=A2=0",
            "dependent_coefficients": "A4=-A3-X^2 A3^2/(8F0); A5=X A3^2/(2F0)",
            "activation": "d^2/[(1+d^2)^(3/2) sqrt(1+x^2)], x=X/q_sigma^2, d=x-x0",
            "background": "x0=-Q0^2/q_sigma^2",
        },
        "provisional_constants": [
            "a_sigma",
            "mu_sigma",
            "K_B",
            "K_2",
            "lambda_D",
        ],
        "flat_spectrum": spectrum,
        "coefficient_scan": {
            "signed_limit": float(signed_scan_limit),
            "total_points": int(ratio.size),
            "backgrounds": coefficient_audits,
            "maximum_normalized_degeneracy_residual": maximum_residual,
            "maximum_absolute_normalized_coefficient": maximum_coefficient,
        },
        "geometry_probe": {
            "gradient": gradient.tolist(),
            "isotropic_hessian": isotropic.tolist(),
            "rank_one_hessian": rank_one.tolist(),
            "same_trace": bool(
                isotropic_invariants["trace_hessian"] == rank_one_invariants["trace_hessian"]
            ),
            "isotropic_invariants": isotropic_invariants,
            "rank_one_invariants": rank_one_invariants,
            "maximum_directional_difference": directional_difference,
            "maximum_rotation_covariance_residual": rotation_residual,
        },
        "screening_probe": {
            "high_acceleration_ratio": float(high_acceleration_ratio),
            "high_field_activation": high_activation,
            "background_activation": background_activation,
            "numerical_background_first_derivative": derivative_at_background,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_selection_gates_pass": bool(all(gates.values())),
        "full_joint_adm_degeneracy_proven": False,
        "complete_metric_stress_derived": False,
        "arbitrary_background_characteristics_proven": False,
        "theory_viable": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
