"""Exact tensor-cone falsification for Sigma v10D.

On an axisymmetric carrier background with a wave along the symmetry axis,
the two transverse-traceless metric polarizations are an invariant spin-2
sector.  After the carrier time derivative is diagonalized, its covariant
spatial derivative supplies an unmatched positive spatial stiffness.  The
resulting squared tensor speed is ``1+c_P^2 (p_parallel-p_perp)^2``.

This is a theory-only necessary gate.  It does not use observational data or
assume a value for the carrier anisotropy inferred from an astronomical fit.
"""

from __future__ import annotations

import numpy as np

from voidscreen.sigma_v10d_adm_rank import (
    symmetric_orthonormal_matrix,
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


def tt_polarization_basis(direction: Array) -> tuple[Array, Array]:
    """Return Frobenius-orthonormal plus and cross tensors transverse to ``n``."""

    unit = normalized_direction(direction)
    axes = np.eye(3)
    trial = axes[int(np.argmin(np.abs(axes @ unit)))]
    first = np.cross(unit, trial)
    first /= np.linalg.norm(first)
    second = np.cross(unit, first)
    plus = (np.outer(first, first) - np.outer(second, second)) / np.sqrt(2.0)
    cross = (np.outer(first, second) + np.outer(second, first)) / np.sqrt(2.0)
    return plus, cross


def axisymmetric_carrier_background(
    direction: Array, *, perpendicular: float, parallel: float
) -> Array:
    unit = normalized_direction(direction)
    p_perpendicular = float(perpendicular)
    p_parallel = float(parallel)
    if not np.isfinite(p_perpendicular) or not np.isfinite(p_parallel):
        raise ValueError("carrier eigenvalues must be finite")
    return p_perpendicular * np.eye(3) + (p_parallel - p_perpendicular) * np.outer(
        unit, unit
    )


def linearized_spatial_connection_residual(
    carrier_background: Array,
    metric_perturbation: Array,
    direction: Array,
) -> Array:
    """Return the connection residual in ``delta(D_l P_ij)``.

    With ``r=delta(P)-(h.P+P.h)/2``, the principal spatial derivative is
    ``delta(D_l P_ij)=n_l r_ij+R_lij``.  This function returns ``R`` without
    the common plane-wave factor ``i k``.
    """

    background = np.asarray(carrier_background, dtype=float)
    metric = np.asarray(metric_perturbation, dtype=float)
    unit = normalized_direction(direction)
    for value, name in ((background, "carrier background"), (metric, "metric")):
        if value.shape != (3, 3) or np.any(~np.isfinite(value)):
            raise ValueError(f"{name} must be a finite 3x3 matrix")
        if not np.allclose(value, value.T, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{name} must be symmetric")
    residual = np.zeros((3, 3, 3))
    for derivative_index in range(3):
        for first_index in range(3):
            for second_index in range(3):
                total = 0.0
                for contracted in range(3):
                    total += (
                        unit[first_index] * metric[derivative_index, contracted]
                        - unit[contracted]
                        * metric[derivative_index, first_index]
                    ) * background[contracted, second_index]
                    total += (
                        unit[second_index] * metric[derivative_index, contracted]
                        - unit[contracted]
                        * metric[derivative_index, second_index]
                    ) * background[first_index, contracted]
                residual[derivative_index, first_index, second_index] = -0.5 * total
    return residual


def tensor_carrier_characteristic_roots(
    carrier_background: Array,
    direction: Array,
    *,
    carrier_speed_squared: float,
) -> Array:
    """Return the eight TT-metric plus symmetric-carrier squared speeds."""

    background = np.asarray(carrier_background, dtype=float)
    if background.shape != (3, 3) or np.any(~np.isfinite(background)):
        raise ValueError("carrier background must be a finite 3x3 matrix")
    if not np.allclose(background, background.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("carrier background must be symmetric")
    speed = float(carrier_speed_squared)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("carrier speed squared must be finite and positive")
    unit = normalized_direction(direction)
    polarizations = tt_polarization_basis(unit)
    residual_map = np.column_stack(
        [
            linearized_spatial_connection_residual(background, tensor, unit).reshape(
                -1
            )
            for tensor in polarizations
        ]
    )
    derivative_map = np.zeros((27, 6))
    for column in range(6):
        carrier = symmetric_orthonormal_matrix(np.eye(6)[column])
        derivative_map[:, column] = np.einsum("l,ij->lij", unit, carrier).reshape(
            -1
        )
    covariant_derivative_map = np.column_stack([residual_map, derivative_map])

    kinetic = np.diag([0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    gradient = np.zeros((8, 8))
    gradient[:2, :2] = 0.5 * np.eye(2)
    gradient += speed * covariant_derivative_map.T @ covariant_derivative_map
    inverse_sqrt = np.diag(np.reciprocal(np.sqrt(np.diag(kinetic))))
    normalized = inverse_sqrt @ gradient @ inverse_sqrt
    return np.linalg.eigvalsh(0.5 * (normalized + normalized.T))


def axisymmetric_tensor_speed_squared(
    *, carrier_speed_squared: float, anisotropy: float
) -> float:
    speed = float(carrier_speed_squared)
    difference = float(anisotropy)
    if not np.isfinite(speed) or speed <= 0.0 or not np.isfinite(difference):
        raise ValueError("speed must be positive and inputs finite")
    return 1.0 + speed * difference**2


def stable_relative_speed_excess(speed_squared: float) -> float:
    value = float(speed_squared)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("speed squared must be finite and non-negative")
    root = np.sqrt(value)
    return (value - 1.0) / (root + 1.0)


def audit_v10d_tensor_cone(
    *,
    carrier_speed_squared: float,
    speed_tolerance: float,
    demonstration_anisotropy: float,
    scan_maximum_anisotropy: float,
    scan_samples: int,
) -> dict[str, object]:
    """Audit the exact axisymmetric spin-2 characteristic sector."""

    speed = float(carrier_speed_squared)
    tolerance = float(speed_tolerance)
    demonstration = float(demonstration_anisotropy)
    maximum = float(scan_maximum_anisotropy)
    count = int(scan_samples)
    values = np.asarray([speed, tolerance, demonstration, maximum], dtype=float)
    if (
        np.any(~np.isfinite(values))
        or speed <= 0.0
        or tolerance <= 0.0
        or demonstration <= 0.0
        or maximum <= 0.0
        or count < 10
    ):
        raise ValueError("audit inputs are outside their finite positive domains")

    direction = np.array([0.0, 0.0, 1.0])
    scan = np.linspace(0.0, maximum, count)
    numerical_maxima = []
    analytic_maxima = []
    maximum_root_error = 0.0
    for difference in scan:
        background = axisymmetric_carrier_background(
            direction, perpendicular=0.0, parallel=float(difference)
        )
        roots = tensor_carrier_characteristic_roots(
            background, direction, carrier_speed_squared=speed
        )
        analytic = axisymmetric_tensor_speed_squared(
            carrier_speed_squared=speed, anisotropy=float(difference)
        )
        numerical_maxima.append(float(roots[-1]))
        analytic_maxima.append(analytic)
        maximum_root_error = max(maximum_root_error, abs(float(roots[-1]) - analytic))

    demonstration_squared = axisymmetric_tensor_speed_squared(
        carrier_speed_squared=speed, anisotropy=demonstration
    )
    demonstration_squared_excess = speed * demonstration**2
    demonstration_excess = demonstration_squared_excess / (
        np.sqrt(1.0 + demonstration_squared_excess) + 1.0
    )
    maximum_allowed_anisotropy = np.sqrt(
        (2.0 * tolerance + tolerance**2) / speed
    )
    isotropic_roots = tensor_carrier_characteristic_roots(
        np.zeros((3, 3)), direction, carrier_speed_squared=speed
    )
    anisotropic_roots = tensor_carrier_characteristic_roots(
        axisymmetric_carrier_background(
            direction, perpendicular=0.0, parallel=demonstration
        ),
        direction,
        carrier_speed_squared=speed,
    )
    gates = {
        "isotropic_carrier_preserves_luminal_TT_cone": bool(
            np.allclose(isotropic_roots[-2:], 1.0, atol=1.0e-14)
        ),
        "axisymmetric_numerical_roots_match_exact_formula": maximum_root_error
        < 1.0e-12,
        "anisotropic_carrier_preserves_metric_TT_cone": float(anisotropic_roots[-1])
        <= 1.0 + 1.0e-14,
        "demonstration_background_satisfies_declared_speed_tolerance": demonstration_excess
        <= tolerance,
    }
    return {
        "analytic_result": {
            "field_redefinition": "r_ij=delta(P)_ij-(h_i^k P_kj+P_i^k h_kj)/2",
            "axisymmetric_background": "P=diag(p_perp,p_perp,p_parallel), k along symmetry axis",
            "connection_residual_norm": "R:R=(Delta p)^2(h:h)/2",
            "TT_speed_squared": "c_TT^2=1+c_P^2(Delta p)^2",
            "consequence": "c_TT exceeds the physical metric cone for every nonzero carrier anisotropy",
        },
        "isotropic_roots": isotropic_roots.tolist(),
        "demonstration": {
            "anisotropy": demonstration,
            "roots": anisotropic_roots.tolist(),
            "TT_speed_squared": demonstration_squared,
            "relative_TT_speed_excess": demonstration_excess,
            "declared_relative_speed_tolerance": tolerance,
            "maximum_anisotropy_consistent_with_tolerance": float(
                maximum_allowed_anisotropy
            ),
        },
        "scan": {
            "samples": count,
            "maximum_anisotropy": maximum,
            "maximum_numerical_root": max(numerical_maxima),
            "maximum_analytic_root": max(analytic_maxima),
            "maximum_root_error": maximum_root_error,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_tensor_cone_gates_pass": bool(all(gates.values())),
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
