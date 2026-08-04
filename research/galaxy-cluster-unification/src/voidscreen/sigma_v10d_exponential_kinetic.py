"""Theory-only selection for the Sigma v10D exponential kinetic completion.

V10D supplements the retired v10C action with the fixed covariant spatial
matrix function ``K_B J.[exp(X)-I].J``, where ``X=(beta/K_B)P``.  After the
constraint-induced ``-beta J.P.J`` term is included, the physical aether
kinetic matrix is ``K_B[exp(X)-X]``.  Every real eigenvalue obeys
``exp(x)-x >= 1``, so the v10C finite-amplitude kinetic zero is removed without
an extra constant or an amplitude cutoff.
"""

from __future__ import annotations

import numpy as np

from voidscreen.sigma_v10c_hyperbolic_aether_tidal import mixed_hyperbolic_channel

Array = np.ndarray


def completed_eigenvalue(x: Array | float) -> Array:
    value = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(value)):
        raise ValueError("x must be finite")
    return np.exp(value) - value


def symmetric_matrix_exponential(matrix: Array) -> Array:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.allclose(
        value, value.T, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("matrix must be symmetric with shape (3,3)")
    if np.any(~np.isfinite(value)):
        raise ValueError("matrix must be finite")
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    result = (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T
    return 0.5 * (result + result.T)


def completed_aether_kinetic_matrix(
    carrier_background: Array,
    *,
    k_b: float,
    beta: float,
) -> Array:
    background = np.asarray(carrier_background, dtype=float)
    stiffness = float(k_b)
    mixing = float(beta)
    if background.shape != (3, 3) or not np.allclose(
        background, background.T, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("carrier background must be symmetric with shape (3,3)")
    if (
        np.any(~np.isfinite(background))
        or not np.isfinite(stiffness)
        or not np.isfinite(mixing)
        or stiffness <= 0.0
        or mixing <= 0.0
    ):
        raise ValueError("background and positive coefficients must be finite")
    x_matrix = (mixing / stiffness) * background
    return stiffness * (
        symmetric_matrix_exponential(x_matrix) - x_matrix
    )


def completed_channel_speed_squared(
    *,
    kinetic_factor: float,
    base_spatial_stiffness: float,
    carrier_speed_squared: float,
    normalized_mixing_squared: float,
) -> Array:
    """Return roots of ``(u-f y)(s-y)-q y=0``."""

    factor = float(kinetic_factor)
    base = float(base_spatial_stiffness)
    carrier = float(carrier_speed_squared)
    mixing = float(normalized_mixing_squared)
    values = np.asarray([factor, base, carrier, mixing], dtype=float)
    if np.any(~np.isfinite(values)) or factor <= 0.0 or base <= 0.0 or carrier <= 0.0:
        raise ValueError("kinetic factor and stiffnesses must be finite and positive")
    if mixing < 0.0:
        raise ValueError("mixing must be non-negative")
    coefficient = base + factor * carrier + mixing
    discriminant = coefficient**2 - 4.0 * factor * base * carrier
    if discriminant < -1.0e-14:
        return np.array([np.nan, np.nan])
    root = np.sqrt(max(discriminant, 0.0))
    return np.array(
        [
            (coefficient - root) / (2.0 * factor),
            (coefficient + root) / (2.0 * factor),
        ]
    )


def completed_static_block(
    *,
    kinetic_factor: float,
    carrier_speed_squared: float,
    normalized_mixing_squared: float,
) -> Array:
    factor = float(kinetic_factor)
    carrier = float(carrier_speed_squared)
    mixing = float(normalized_mixing_squared)
    if factor <= 0.0 or carrier <= 0.0 or mixing < 0.0:
        raise ValueError("static block coefficients are outside their positive domain")
    return np.array([[factor, -np.sqrt(mixing)], [-np.sqrt(mixing), carrier]])


def audit_v10d_exponential_kinetic(
    *,
    k_b: float,
    u: float,
    carrier_speed_squared: float,
    normalized_mixing_squared: float,
    physical_parameter_count: int,
    maximum_physical_parameters: int,
) -> dict[str, object]:
    """Audit global amplitude positivity and inherited local cones."""

    stiffness = float(k_b)
    beta = np.sqrt(stiffness * float(normalized_mixing_squared))
    x_samples = np.linspace(-20.0, 20.0, 200_001)
    factors = completed_eigenvalue(x_samples)
    minimum_index = int(np.argmin(factors))
    sampled_minimum_x = float(x_samples[minimum_index])
    sampled_minimum = float(factors[minimum_index])

    rng = np.random.default_rng(10041)
    matrix_minima = []
    for scale in (0.0, 0.1, 1.0, 10.0, 100.0):
        raw = rng.normal(size=(3, 3))
        background = scale * 0.5 * (raw + raw.T)
        matrix_minima.append(
            float(
                np.linalg.eigvalsh(
                    completed_aether_kinetic_matrix(
                        background, k_b=stiffness, beta=beta
                    )
                )[0]
            )
        )

    factors_for_cones = np.unique(
        np.concatenate([np.array([1.0]), np.geomspace(1.0, 1.0e8, 4000)])
    )
    longitudinal_speeds = np.array(
        [
            completed_channel_speed_squared(
                kinetic_factor=factor,
                base_spatial_stiffness=u,
                carrier_speed_squared=carrier_speed_squared,
                normalized_mixing_squared=normalized_mixing_squared,
            )
            for factor in factors_for_cones
        ]
    )
    transverse_speeds = np.array(
        [
            completed_channel_speed_squared(
                kinetic_factor=factor,
                base_spatial_stiffness=u,
                carrier_speed_squared=carrier_speed_squared,
                normalized_mixing_squared=normalized_mixing_squared / 2.0,
            )
            for factor in factors_for_cones
        ]
    )
    static_minima = np.array(
        [
            np.linalg.eigvalsh(
                completed_static_block(
                    kinetic_factor=factor,
                    carrier_speed_squared=carrier_speed_squared,
                    normalized_mixing_squared=normalized_mixing_squared,
                )
            )[0]
            for factor in factors_for_cones
        ]
    )
    zero_background_reference = mixed_hyperbolic_channel(
        base_speed_squared=u,
        carrier_speed_squared=carrier_speed_squared,
        normalized_mixing_squared=normalized_mixing_squared,
    ).speed_squared
    count = int(physical_parameter_count)
    maximum = int(maximum_physical_parameters)
    gates = {
        "analytic_scalar_minimum_is_one_at_zero": bool(
            np.isclose(completed_eigenvalue(0.0), 1.0, rtol=0.0, atol=1.0e-15)
        ),
        "sampled_scalar_factor_never_below_one": sampled_minimum >= 1.0 - 1.0e-12,
        "sampled_minimum_occurs_at_zero": abs(sampled_minimum_x) < 2.1e-4,
        "random_matrix_kinetic_eigenvalues_at_least_KB": min(matrix_minima)
        >= stiffness * (1.0 - 1.0e-10),
        "zero_background_longitudinal_cones_unchanged": bool(
            np.allclose(longitudinal_speeds[0], zero_background_reference, atol=1.0e-12)
        ),
        "amplitude_scan_longitudinal_cones_positive_causal": bool(
            np.all(longitudinal_speeds > 0.0)
            and np.all(longitudinal_speeds <= 1.0 + 1.0e-12)
        ),
        "amplitude_scan_transverse_cones_positive_causal": bool(
            np.all(transverse_speeds > 0.0)
            and np.all(transverse_speeds <= 1.0 + 1.0e-12)
        ),
        "amplitude_scan_static_block_positive": bool(np.all(static_minima > 0.0)),
        "no_new_physical_constant": count <= maximum,
    }
    return {
        "coefficients": {
            "K_B": stiffness,
            "u": float(u),
            "c_P_squared": float(carrier_speed_squared),
            "beta_squared_over_K_B": float(normalized_mixing_squared),
            "beta": float(beta),
        },
        "completion": {
            "X": "(beta/K_B) P",
            "added_term": "K_B J_m [exp(X)^m_n-q^m_n] J^n",
            "reduced_kinetic_matrix": "K_B[exp(X)-X]",
            "eigenvalue_function": "K_B[exp(x)-x]",
            "analytic_global_minimum": stiffness,
            "minimum_location": 0.0,
        },
        "scalar_amplitude_scan": {
            "minimum_x": sampled_minimum_x,
            "minimum_factor": sampled_minimum,
            "x_min": float(x_samples[0]),
            "x_max": float(x_samples[-1]),
            "samples": int(x_samples.size),
        },
        "random_matrix_minimum_kinetic_eigenvalues": matrix_minima,
        "cone_scan": {
            "kinetic_factor_min": float(factors_for_cones[0]),
            "kinetic_factor_max": float(factors_for_cones[-1]),
            "samples": int(factors_for_cones.size),
            "maximum_longitudinal_speed_squared": float(np.max(longitudinal_speeds)),
            "minimum_longitudinal_speed_squared": float(np.min(longitudinal_speeds)),
            "maximum_transverse_speed_squared": float(np.max(transverse_speeds)),
            "minimum_transverse_speed_squared": float(np.min(transverse_speeds)),
            "minimum_static_eigenvalue": float(np.min(static_minima)),
        },
        "zero_background_longitudinal_speed_squared": zero_background_reference.tolist(),
        "selection_gates": {name: bool(value) for name, value in gates.items()},
        "all_selection_gates_pass": bool(all(gates.values())),
        "unresolved": {
            "full_nonlinear_ADM_constraint_count": False,
            "tilted_metric_carrier_velocity_Hessian": False,
            "nonzero_J_and_nonzero_P_characteristics": False,
            "complete_metric_stress_and_PPN": False,
            "global_hyperbolicity": False,
            "numerical_PDE_convergence": False,
        },
        "all_mandatory_theory_gates_pass": False,
    }
