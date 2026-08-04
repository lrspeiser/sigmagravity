from __future__ import annotations

import numpy as np


def bounded_even_activation(signed_kinetic_ratio) -> np.ndarray:
    """Return the v5C trial shape ``X_hat^2/(1+X_hat^2)^(3/2)``.

    It is globally real, smooth and even, vanishes quadratically at zero, and
    falls as ``1/|X_hat|`` on either large timelike or spacelike branch.  That
    final power is selected because the DHOST identities contain ``X^2 A3^2``;
    it keeps every dependent coefficient bounded at high field.
    """
    value = np.asarray(signed_kinetic_ratio, dtype=float)
    if np.any(~np.isfinite(value)):
        raise ValueError("signed_kinetic_ratio must be finite")
    squared = np.square(value)
    return squared / np.power(1.0 + squared, 1.5)


def luminal_class_ia_coefficients(
    scalar_kinetic,
    tensor_coupling,
    tensor_coupling_derivative,
    a3_coefficient,
) -> dict[str, np.ndarray]:
    """Return the quadratic luminal Class-Ia DHOST coefficients.

    The convention is ``X=g^ab grad_a(phi) grad_b(phi)``.  Requiring the tensor
    cone to equal the matter light cone sets ``A1=A2=0``.  Degeneracy then fixes
    ``A4`` and ``A5`` from ``F``, ``F_X`` and the freely selected ``A3``:

    ``A4=[48 F_X^2-8(F-X F_X)A3-X^2 A3^2]/(8F)``
    ``A5=(4F_X+X A3)A3/(2F)``.
    """
    kinetic = np.asarray(scalar_kinetic, dtype=float)
    coupling = np.asarray(tensor_coupling, dtype=float)
    derivative = np.asarray(tensor_coupling_derivative, dtype=float)
    a3 = np.asarray(a3_coefficient, dtype=float)
    try:
        kinetic, coupling, derivative, a3 = np.broadcast_arrays(
            kinetic, coupling, derivative, a3
        )
    except ValueError as error:
        raise ValueError("the DHOST coefficient inputs must broadcast") from error
    if np.any(~np.isfinite(kinetic + coupling + derivative + a3)):
        raise ValueError("the DHOST coefficient inputs must be finite")
    if np.any(coupling <= 0.0):
        raise ValueError("tensor_coupling must be positive")
    a4 = (
        48.0 * np.square(derivative)
        - 8.0 * (coupling - kinetic * derivative) * a3
        - np.square(kinetic * a3)
    ) / (8.0 * coupling)
    a5 = (4.0 * derivative + kinetic * a3) * a3 / (2.0 * coupling)
    return {
        "A1": np.zeros_like(a4),
        "A2": np.zeros_like(a4),
        "A3": a3,
        "A4": a4,
        "A5": a5,
    }


def luminal_class_ia_residuals(
    scalar_kinetic,
    tensor_coupling,
    tensor_coupling_derivative,
    coefficients: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return unscaled identities that must vanish for the selected class."""
    kinetic = np.asarray(scalar_kinetic, dtype=float)
    coupling = np.asarray(tensor_coupling, dtype=float)
    derivative = np.asarray(tensor_coupling_derivative, dtype=float)
    a1 = np.asarray(coefficients["A1"], dtype=float)
    a2 = np.asarray(coefficients["A2"], dtype=float)
    a3 = np.asarray(coefficients["A3"], dtype=float)
    a4 = np.asarray(coefficients["A4"], dtype=float)
    a5 = np.asarray(coefficients["A5"], dtype=float)
    return {
        "tensor_speed": a1,
        "class_i": a1 + a2,
        "a4_degeneracy": 8.0 * coupling * a4
        - (
            48.0 * np.square(derivative)
            - 8.0 * (coupling - kinetic * derivative) * a3
            - np.square(kinetic * a3)
        ),
        "a5_degeneracy": 2.0 * coupling * a5
        - (4.0 * derivative + kinetic * a3) * a3,
    }


def normalized_dhost_residuals(
    scalar_kinetic,
    tensor_coupling,
    tensor_coupling_derivative,
    coefficients: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return scale-safe residuals for numerical scans of the identities."""
    raw = luminal_class_ia_residuals(
        scalar_kinetic,
        tensor_coupling,
        tensor_coupling_derivative,
        coefficients,
    )
    kinetic = np.asarray(scalar_kinetic, dtype=float)
    coupling = np.asarray(tensor_coupling, dtype=float)
    derivative = np.asarray(tensor_coupling_derivative, dtype=float)
    a3 = np.asarray(coefficients["A3"], dtype=float)
    a4 = np.asarray(coefficients["A4"], dtype=float)
    a5 = np.asarray(coefficients["A5"], dtype=float)
    scale4 = np.maximum.reduce(
        (
            np.ones_like(a4),
            np.abs(8.0 * coupling * a4),
            np.abs(48.0 * np.square(derivative)),
            np.abs(8.0 * (coupling - kinetic * derivative) * a3),
            np.abs(np.square(kinetic * a3)),
        )
    )
    scale5 = np.maximum.reduce(
        (
            np.ones_like(a5),
            np.abs(2.0 * coupling * a5),
            np.abs(4.0 * derivative * a3),
            np.abs(kinetic * np.square(a3)),
        )
    )
    return {
        "tensor_speed": raw["tensor_speed"],
        "class_i": raw["class_i"],
        "a4_degeneracy": raw["a4_degeneracy"] / scale4,
        "a5_degeneracy": raw["a5_degeneracy"] / scale5,
    }


def v5c_trial_coefficients(
    scalar_kinetic_ratio,
    orientation_strength: float,
) -> dict[str, np.ndarray]:
    """Return dimensionless coefficient shapes for the provisional v5C row.

    This uses ``F/F0=1``, ``F_X=0`` and
    ``A3/(F0/q_sigma^4)=lambda*activation(X/q_sigma^2)``.  Dimensional factors
    are restored in the action document; this helper tests only the universal
    signed shape and the exact degeneracy relations.
    """
    ratio = np.asarray(scalar_kinetic_ratio, dtype=float)
    strength = float(orientation_strength)
    if not np.isfinite(strength):
        raise ValueError("orientation_strength must be finite")
    a3 = strength * bounded_even_activation(ratio)
    return luminal_class_ia_coefficients(ratio, 1.0, 0.0, a3)


def k_mouflage_static_parallel_speed_squared(
    scalar_kinetic,
    kinetic_derivative,
    kinetic_second_derivative,
) -> np.ndarray:
    """Return the parallel scalar speed on a static spacelike ``P(X)`` branch.

    With ``X<0`` and a static gradient, the time coefficient is ``P_X`` and
    the parallel spatial coefficient is ``P_X+2 X P_XX``.  Strict hyperbolicity
    requires both positive.  A derivative screen needs ``P_X`` to grow as
    ``-X`` grows, hence ``P_XX<0`` and this ratio is necessarily superluminal.
    """
    kinetic = np.asarray(scalar_kinetic, dtype=float)
    first = np.asarray(kinetic_derivative, dtype=float)
    second = np.asarray(kinetic_second_derivative, dtype=float)
    try:
        kinetic, first, second = np.broadcast_arrays(kinetic, first, second)
    except ValueError as error:
        raise ValueError("the k-mouflage inputs must broadcast") from error
    if np.any(~np.isfinite(kinetic + first + second)):
        raise ValueError("the k-mouflage inputs must be finite")
    if np.any(kinetic >= 0.0) or np.any(first <= 0.0):
        raise ValueError("the static branch requires X<0 and P_X>0")
    parallel = first + 2.0 * kinetic * second
    if np.any(parallel <= 0.0):
        raise FloatingPointError("the parallel gradient coefficient is non-positive")
    return parallel / first


def newton_yukawa_acceleration_ratio(
    radius_over_range,
    scalar_strength: float,
) -> np.ndarray:
    """Return ``g/(GM/r^2)`` for an attractive massive conformal scalar.

    The fixed v5C row is canonical when its DHOST operators are inactive.  Its
    most favorable linear exterior law is therefore Newton plus a Yukawa force,
    ``1+alpha(1+x)exp(-x)`` with ``x=r/L`` and non-negative strength.
    """
    ratio = np.asarray(radius_over_range, dtype=float)
    strength = float(scalar_strength)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("radius_over_range must be finite and non-negative")
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("scalar_strength must be finite and non-negative")
    return 1.0 + strength * (1.0 + ratio) * np.exp(-ratio)


def newton_yukawa_log_acceleration_slope(
    radius_over_range,
    scalar_strength: float,
) -> np.ndarray:
    """Return ``d log(g)/d log(r)`` for Newton plus attractive Yukawa.

    The exact expression is ``-2-alpha*x^2 exp(-x)/F(x)``.  It is never
    shallower than ``-2`` for a non-negative scalar force, so this exterior
    cannot supply the ``-1`` acceleration slope of a flat rotation curve.
    """
    ratio = np.asarray(radius_over_range, dtype=float)
    strength = float(scalar_strength)
    force_ratio = newton_yukawa_acceleration_ratio(ratio, strength)
    return -2.0 - strength * np.square(ratio) * np.exp(-ratio) / force_ratio


def newton_yukawa_circular_speed_ratio(
    radius_ratio: float,
    inner_radius_over_range,
    scalar_strength: float,
) -> np.ndarray:
    """Return ``v_c(r2)/v_c(r1)`` with ``r2=radius_ratio*r1``."""
    scale = float(radius_ratio)
    inner = np.asarray(inner_radius_over_range, dtype=float)
    if not np.isfinite(scale) or scale <= 1.0:
        raise ValueError("radius_ratio must be finite and greater than one")
    if np.any(~np.isfinite(inner)) or np.any(inner < 0.0):
        raise ValueError("inner_radius_over_range must be finite and non-negative")
    inner_force = newton_yukawa_acceleration_ratio(inner, scalar_strength)
    outer_force = newton_yukawa_acceleration_ratio(
        scale * inner, scalar_strength
    )
    # g=(GM/r^2)F and v_c^2=r g, hence v2/v1=sqrt(F2/(scale F1)).
    return np.sqrt(outer_force / (scale * inner_force))
