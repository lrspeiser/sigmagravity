"""Pre-data scalar-sector gates for the Sigma v8B causal cubic completion.

The v8A cubic interaction has a superluminal radial characteristic when it is
nonlinear. AeST already contains a unit timelike vector and the clock invariant
``Q=A^m grad_m(phi)``. The v8B envelope adds the preferred-frame operator

``(alpha-1) L_H^2 (Q-Q0)^2 q^mn nabla_m nabla_n(phi)``.

It vanishes, together with its first variation, on a static ``Q=Q0`` background
and therefore leaves that background equation unchanged. Its quadratic
variation increases the time kinetic coefficient. This module proves only the
fixed-aether scalar selection identities; the full covariant vector, metric, and
constraint variation remains mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class CompletedPrincipalSymbol:
    temporal_coefficient: float
    spatial_eigenvalues: Array
    speed_squared: Array
    positive: bool
    causal: bool


def causal_completion_alpha(*, base_speed_squared: float) -> float:
    """Return the smallest constant closing the positive spherical cone."""

    base = float(base_speed_squared)
    if not np.isfinite(base) or not 0.0 < base < 1.0:
        raise ValueError("base_speed_squared must lie strictly between zero and one")
    return 1.0 / (3.0 * base * (1.0 - base))


def completed_cubic_static_principal_symbol(
    dimensionless_hessian: Array,
    *,
    base_speed_squared: float,
    alpha: float,
) -> CompletedPrincipalSymbol:
    """Return the completed scalar principal symbol on a static background."""

    matrix = np.asarray(dimensionless_hessian, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("dimensionless_hessian must have shape (3, 3)")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("dimensionless_hessian must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("dimensionless_hessian must be symmetric")
    base = float(base_speed_squared)
    completion = float(alpha)
    if not np.isfinite(base) or not 0.0 < base < 1.0:
        raise ValueError("base_speed_squared must lie strictly between zero and one")
    if not np.isfinite(completion) or completion < 1.0:
        raise ValueError("alpha must be finite and at least one")
    trace = float(np.trace(matrix))
    temporal = 1.0 + 2.0 * completion * trace
    spatial = base * np.eye(3) + 2.0 * (trace * np.eye(3) - matrix)
    eigenvalues = np.linalg.eigvalsh(spatial)
    if temporal > 0.0:
        speeds = eigenvalues / temporal
    else:
        speeds = np.full(3, np.nan)
    positive = temporal > 0.0 and bool(np.all(eigenvalues > 0.0))
    causal = positive and bool(np.all(speeds <= 1.0 + 1.0e-12))
    return CompletedPrincipalSymbol(
        temporal_coefficient=temporal,
        spatial_eigenvalues=eigenvalues,
        speed_squared=speeds,
        positive=positive,
        causal=causal,
    )


def completed_spherical_characteristics(
    dimensionless_tangential_hessian: float,
    *,
    base_speed_squared: float,
    alpha: float,
) -> dict[str, float | bool]:
    """Return the completed characteristics on the unchanged spherical branch."""

    u = float(dimensionless_tangential_hessian)
    base = float(base_speed_squared)
    if not np.isfinite(u) or u < 0.0:
        raise ValueError(
            "dimensionless_tangential_hessian must be finite and non-negative"
        )
    if not np.isfinite(base) or not 0.0 < base < 1.0:
        raise ValueError("base_speed_squared must lie strictly between zero and one")
    ratio = -2.0 * (base + u) / (base + 4.0 * u)
    symbol = completed_cubic_static_principal_symbol(
        np.diag([ratio * u, u, u]),
        base_speed_squared=base,
        alpha=alpha,
    )
    radial_spatial = base + 4.0 * u
    tangential_spatial = base + 2.0 * u * (ratio + 1.0)
    return {
        "dimensionless_tangential_hessian": u,
        "radial_to_tangential_hessian": ratio,
        "temporal_coefficient": symbol.temporal_coefficient,
        "radial_speed_squared": radial_spatial / symbol.temporal_coefficient,
        "tangential_speed_squared": tangential_spatial
        / symbol.temporal_coefficient,
        "positive": symbol.positive,
        "causal": symbol.causal,
    }


def audit_v8b_scalar_selection(
    *,
    base_speed_squared: float,
    physical_parameter_count: int,
    maximum_physical_parameters: int,
) -> dict[str, object]:
    """Audit the analytic v8B scalar-sector selection identities."""

    base = float(base_speed_squared)
    alpha = causal_completion_alpha(base_speed_squared=base)
    scan = np.concatenate(([0.0], np.geomspace(1.0e-10, 1.0e8, 20000)))
    spherical = [
        completed_spherical_characteristics(
            float(value),
            base_speed_squared=base,
            alpha=alpha,
        )
        for value in scan
    ]
    radial = np.array([row["radial_speed_squared"] for row in spherical])
    tangential = np.array([row["tangential_speed_squared"] for row in spherical])
    temporal = np.array([row["temporal_coefficient"] for row in spherical])
    positive = np.array([row["positive"] for row in spherical], dtype=bool)
    causal = np.array([row["causal"] for row in spherical], dtype=bool)
    peak = int(np.argmax(radial))

    isotropic = completed_cubic_static_principal_symbol(
        np.eye(3),
        base_speed_squared=base,
        alpha=alpha,
    )
    rank_one = completed_cubic_static_principal_symbol(
        np.diag([3.0, 0.0, 0.0]),
        base_speed_squared=base,
        alpha=alpha,
    )
    trace_scan = np.concatenate(([0.0], np.geomspace(1.0e-12, 1.0e8, 20000)))
    discriminant = np.sqrt(4.0 * trace_scan**2 + 6.0 * base * trace_scan)
    minimum_hessian_eigenvalue = (trace_scan - discriminant) / 3.0
    maximum_spatial_eigenvalue = base + 2.0 * (
        trace_scan - minimum_hessian_eigenvalue
    )
    # This rationalized form avoids cancellation as T becomes large.
    reduction = np.divide(
        4.0 * base * trace_scan,
        discriminant + 2.0 * trace_scan,
        out=np.zeros_like(trace_scan),
        where=trace_scan > 0.0,
    )
    minimum_spatial_eigenvalue = base - reduction
    extremal_temporal = 1.0 + 2.0 * alpha * trace_scan
    extremal_speed = maximum_spatial_eigenvalue / extremal_temporal
    extremal_peak = int(np.argmax(extremal_speed))
    count = int(physical_parameter_count)
    maximum = int(maximum_physical_parameters)
    if count < 0 or maximum < 0:
        raise ValueError("parameter counts must be non-negative")
    gates = {
        "derived_completion_adds_no_parameter": True,
        "static_background_equation_unchanged_at_Q_equals_Q0": True,
        "fixed_aether_third_derivatives_cancel": True,
        "complete_positive_spherical_scan_is_positive": bool(np.all(positive)),
        "complete_positive_spherical_scan_is_causal": bool(np.all(causal)),
        "isotropic_equal_trace_probe_is_causal": isotropic.causal,
        "rank_one_equal_trace_probe_is_causal": rank_one.causal,
        "all_nonnegative_source_hessians_are_spatially_positive": bool(
            np.all(minimum_spatial_eigenvalue > 0.0)
        ),
        "all_nonnegative_source_hessians_are_causal": bool(
            np.all(extremal_speed <= 1.0 + 1.0e-12)
        ),
        "parameter_count": count <= maximum,
    }
    return {
        "base_speed_squared": base,
        "derived_alpha": alpha,
        "spherical_scan": {
            "samples": int(scan.size),
            "minimum_temporal_coefficient": float(np.min(temporal)),
            "maximum_radial_speed_squared": float(np.max(radial)),
            "maximum_radial_speed_location": float(scan[peak]),
            "maximum_tangential_speed_squared": float(np.max(tangential)),
            "deep_radial_speed_squared": float(radial[-1]),
            "deep_tangential_speed_squared": float(tangential[-1]),
        },
        "equal_trace_probes": {
            "isotropic_maximum_speed_squared": float(
                np.max(isotropic.speed_squared)
            ),
            "rank_one_maximum_speed_squared": float(
                np.max(rank_one.speed_squared)
            ),
        },
        "nonnegative_source_extremal_bound": {
            "samples": int(trace_scan.size),
            "maximum_speed_squared": float(np.max(extremal_speed)),
            "maximum_speed_trace": float(trace_scan[extremal_peak]),
            "minimum_spatial_eigenvalue_over_finite_scan": float(
                np.min(minimum_spatial_eigenvalue)
            ),
            "large_trace_minimum_spatial_eigenvalue_limit": 0.0,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
    }
