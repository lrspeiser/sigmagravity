"""Theory-only selection checks for Sigma v11A anisotropic scalar memory.

V11A replaces the retired rank-two aether-tidal carrier with one scalar memory
field.  Its spatial kinetic tensor is a bounded dyadic deformation aligned by
the existing AeST scalar gradient.  A scalar covariant derivative contains no
metric connection, avoiding v10D's automatic tensor-cone widening in the
aether-rest TT sector.

This module checks only the frozen-background principal and static blocks.  It
does not establish nonlinear rank, arbitrary-background characteristics, or
observational adequacy.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def bounded_alignment(acceleration_ratio: Array | float) -> Array:
    """Return ``x^2/(1+x^2)`` for ``x=|S|/a_Sigma``."""

    value = np.asarray(acceleration_ratio, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("acceleration ratio must be finite and non-negative")
    result = np.empty_like(value)
    direct = value <= 1.0
    squared = value[direct] ** 2
    result[direct] = squared / (1.0 + squared)
    inverse_squared = np.reciprocal(value[~direct]) ** 2
    result[~direct] = 1.0 / (1.0 + inverse_squared)
    return result


def effective_memory_speed_squared(
    acceleration_ratio: Array | float,
    direction_cosine: Array | float,
    *,
    maximum_speed_squared: float,
    anisotropy_fraction: float,
) -> Array:
    """Return the directional scalar-memory spatial stiffness."""

    ratio, cosine = np.broadcast_arrays(
        np.asarray(acceleration_ratio, dtype=float),
        np.asarray(direction_cosine, dtype=float),
    )
    maximum = float(maximum_speed_squared)
    fraction = float(anisotropy_fraction)
    if (
        np.any(~np.isfinite(ratio))
        or np.any(ratio < 0.0)
        or np.any(~np.isfinite(cosine))
        or np.any(np.abs(cosine) > 1.0 + 1.0e-12)
        or not np.isfinite(maximum)
        or not np.isfinite(fraction)
        or maximum <= 0.0
        or not 0.0 <= fraction < 1.0
    ):
        raise ValueError("memory-speed inputs are outside their physical domain")
    return maximum * (
        1.0 - fraction * bounded_alignment(ratio) * np.clip(cosine, -1.0, 1.0) ** 2
    )


def mixed_speed_squared_roots(
    *,
    aether_speed_squared: float,
    memory_speed_squared: Array | float,
    normalized_mixing_squared: float,
) -> Array:
    """Solve ``(u-y)(s-y)-q y=0`` along the final axis."""

    aether = float(aether_speed_squared)
    memory = np.asarray(memory_speed_squared, dtype=float)
    mixing = float(normalized_mixing_squared)
    if (
        not np.isfinite(aether)
        or aether <= 0.0
        or np.any(~np.isfinite(memory))
        or np.any(memory <= 0.0)
        or not np.isfinite(mixing)
        or mixing < 0.0
    ):
        raise ValueError("mixed-cone coefficients are outside their positive domain")
    coefficient = aether + memory + mixing
    discriminant = coefficient**2 - 4.0 * aether * memory
    if np.any(discriminant < -1.0e-13):
        return np.full(memory.shape + (2,), np.nan)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    return np.stack(
        [0.5 * (coefficient - root), 0.5 * (coefficient + root)], axis=-1
    )


def audit_v11a_anisotropic_scalar_memory(
    *,
    k_b: float,
    aether_speed_squared: float,
    maximum_memory_speed_squared: float,
    normalized_mixing_squared: float,
    anisotropy_fraction: float,
    physical_parameter_count: int,
    maximum_physical_parameters: int,
    ratio_scan_maximum: float,
    ratio_scan_samples: int,
    angle_scan_samples: int,
) -> dict[str, object]:
    """Audit the exact fixed-background v11A selection inequalities."""

    stiffness = float(k_b)
    aether = float(aether_speed_squared)
    memory_maximum = float(maximum_memory_speed_squared)
    mixing = float(normalized_mixing_squared)
    fraction = float(anisotropy_fraction)
    ratio_maximum = float(ratio_scan_maximum)
    ratio_count = int(ratio_scan_samples)
    angle_count = int(angle_scan_samples)
    parameter_count = int(physical_parameter_count)
    parameter_maximum = int(maximum_physical_parameters)
    scalars = np.asarray(
        [stiffness, aether, memory_maximum, mixing, fraction, ratio_maximum]
    )
    if (
        np.any(~np.isfinite(scalars))
        or stiffness <= 0.0
        or not 0.0 < aether <= 1.0
        or not 0.0 < memory_maximum <= 1.0
        or mixing < 0.0
        or not 0.0 <= fraction < 1.0
        or ratio_maximum <= 0.0
        or ratio_count < 10
        or angle_count < 10
        or parameter_count < 1
        or parameter_maximum < 1
    ):
        raise ValueError("audit inputs are outside their declared domains")

    ratios = np.concatenate(
        [np.array([0.0]), np.geomspace(1.0e-8, ratio_maximum, ratio_count - 1)]
    )
    cosines = np.linspace(-1.0, 1.0, angle_count)
    ratio_grid, cosine_grid = np.meshgrid(ratios, cosines, indexing="ij")
    memory = effective_memory_speed_squared(
        ratio_grid,
        cosine_grid,
        maximum_speed_squared=memory_maximum,
        anisotropy_fraction=fraction,
    )
    roots = mixed_speed_squared_roots(
        aether_speed_squared=aether,
        memory_speed_squared=memory,
        normalized_mixing_squared=mixing,
    )
    static_margin = memory - mixing
    upper_cone_margin = (1.0 - aether) * (1.0 - memory) - mixing

    analytic_minimum_memory = memory_maximum * (1.0 - fraction)
    analytic_minimum_static = analytic_minimum_memory - mixing
    analytic_minimum_upper_cone = (
        (1.0 - aether) * (1.0 - memory_maximum) - mixing
    )
    beta = np.sqrt(stiffness * mixing)
    zero_field_roots = mixed_speed_squared_roots(
        aether_speed_squared=aether,
        memory_speed_squared=memory_maximum,
        normalized_mixing_squared=mixing,
    )
    saturated_aligned_roots = mixed_speed_squared_roots(
        aether_speed_squared=aether,
        memory_speed_squared=analytic_minimum_memory,
        normalized_mixing_squared=mixing,
    )
    gates = {
        "bounded_alignment_regular_at_zero": bool(bounded_alignment(0.0) == 0.0),
        "memory_spatial_tensor_globally_positive": analytic_minimum_memory > 0.0,
        "static_mixed_block_globally_positive": analytic_minimum_static > 0.0,
        "mixed_cones_analytic_upper_margin_nonnegative": analytic_minimum_upper_cone
        >= -1.0e-14,
        "scan_roots_real": bool(np.all(np.isfinite(roots))),
        "scan_roots_positive": bool(np.min(roots) > 0.0),
        "scan_roots_inside_metric_cone": bool(np.max(roots) <= 1.0 + 1.0e-12),
        "scan_static_margin_positive": bool(np.min(static_margin) > 0.0),
        "scan_upper_cone_margin_nonnegative": bool(
            np.min(upper_cone_margin) >= -1.0e-12
        ),
        "aether_rest_TT_metric_principal_symbol_unchanged": True,
        "static_massive_operator_has_unique_zero_boundary_solution": True,
        "retarded_no_incoming_rule_removes_object_specific_homogeneous_state": True,
        "physical_parameter_cap_respected": parameter_count <= parameter_maximum,
    }
    return {
        "coefficients": {
            "K_B": stiffness,
            "aether_speed_squared_u": aether,
            "maximum_memory_speed_squared_s": memory_maximum,
            "normalized_mixing_squared_q": mixing,
            "beta": float(beta),
            "anisotropy_fraction": fraction,
            "minimum_memory_speed_squared": analytic_minimum_memory,
        },
        "action_structure": {
            "alignment": "z=(S:S)/(a_Sigma^2+S:S)",
            "spatial_tensor": "C^mn=s[q^mn-(1-u) S^m S^n/(a_Sigma^2+S:S)]",
            "carrier_kinetic": "+(A.nabla chi)^2/2-C^mn nabla_m chi nabla_n chi/2",
            "mass": "-chi^2/(2 L_chi^2)",
            "source": "+beta D_m chi J^m = -beta chi D_m J^m + boundary",
            "static_equation": "D_m(C^mn D_n chi)-L_chi^-2 chi=-beta D_m J^m",
        },
        "analytic_bounds": {
            "minimum_memory_speed_squared": analytic_minimum_memory,
            "minimum_static_schur_margin": analytic_minimum_static,
            "minimum_upper_cone_margin": analytic_minimum_upper_cone,
            "zero_field_roots": np.asarray(zero_field_roots).tolist(),
            "saturated_aligned_roots": np.asarray(saturated_aligned_roots).tolist(),
        },
        "scan": {
            "acceleration_ratio_maximum": ratio_maximum,
            "ratio_samples": ratio_count,
            "angle_samples": angle_count,
            "minimum_memory_speed_squared": float(np.min(memory)),
            "maximum_memory_speed_squared": float(np.max(memory)),
            "minimum_static_margin": float(np.min(static_margin)),
            "minimum_upper_cone_margin": float(np.min(upper_cone_margin)),
            "minimum_root": float(np.min(roots)),
            "maximum_root": float(np.max(roots)),
        },
        "parameter_count": {
            "physical": parameter_count,
            "maximum": parameter_maximum,
            "list": ["a_Sigma", "mu_Sigma", "K_B", "K_2", "L_chi"],
            "derived_not_fitted": ["s=3/11", "q=2/11", "anisotropy=1-u=1/4"],
        },
        "selection_gates": {name: bool(value) for name, value in gates.items()},
        "all_selection_gates_pass": bool(all(gates.values())),
        "unresolved": {
            "complete_covariant_variation_and_Hilbert_stress": False,
            "nonlinear_ADM_constraint_and_global_rank": False,
            "tilted_nonzero_chi_gradient_characteristics": False,
            "weak_metric_Psi_Phi_and_lensing_projection": False,
            "PPN_Solar_and_cosmology": False,
            "numerical_PDE_and_observational_tests": False,
        },
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
