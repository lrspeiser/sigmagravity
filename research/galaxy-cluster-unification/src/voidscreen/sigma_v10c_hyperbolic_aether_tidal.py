"""Theory-only selection for Sigma v10C hyperbolic aether-tidal carrier.

V10C gives the v10B spatial polarization a positive time kinetic term and
derives its speed and mixing by imposing two exact conditions: a threefold
longitudinal static response capacity and a fastest mixed characteristic equal
to the physical metric light cone.  A fixed spatial-aether magnetic
counterterm lowers the bare transverse aether speed to the existing AeST
scalar speed, so one cone construction covers both source sectors.

These are flat-background and frozen-projector selection identities.  The
full nonlinear constraint algebra, tilted-background characteristics, PPN,
and cosmology remain mandatory before observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voidscreen.sigma_v10a_spatial_polarization import (
    local_algebraic_carrier_response,
    point_mass_tidal_hessian,
    potential_convexity_spectrum,
    trace_stf_decomposition,
)

Array = np.ndarray


@dataclass(frozen=True)
class HyperbolicChannel:
    """High-frequency characteristic result for one canonical channel."""

    base_speed_squared: float
    carrier_speed_squared: float
    normalized_mixing_squared: float
    speed_squared: Array
    positive: bool
    causal: bool


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def v10c_derived_coefficients(
    *,
    maximum_sourced_base_speed_squared: float,
    static_mixing_fraction: float,
    k_b: float,
) -> dict[str, float]:
    """Solve the static-capacity and upper-cone equations.

    Let ``u`` be the largest squared speed among sourced AeST scalar/vector
    modes, ``s=c_P^2``, and ``q=beta^2/K_B``.  Requiring static fraction
    ``q/s=f`` and a luminal upper mixed root gives

    ``q=f s=(1-u)(1-s)``.
    """

    base_speed = _finite_scalar(
        maximum_sourced_base_speed_squared,
        name="maximum_sourced_base_speed_squared",
    )
    fraction = _finite_scalar(static_mixing_fraction, name="static_mixing_fraction")
    aether_stiffness = _finite_scalar(k_b, name="k_b")
    if not 0.0 < base_speed < 1.0:
        raise ValueError("maximum_sourced_base_speed_squared must lie in (0, 1)")
    if fraction <= 0.0 or aether_stiffness <= 0.0:
        raise ValueError("static_mixing_fraction and k_b must be positive")
    carrier_speed = (1.0 - base_speed) / (fraction + 1.0 - base_speed)
    normalized_mixing = fraction * carrier_speed
    beta = np.sqrt(aether_stiffness * normalized_mixing)
    return {
        "sourced_base_speed_squared": base_speed,
        "aether_vector_speed_squared_after_counterterm": base_speed,
        "carrier_speed_squared": carrier_speed,
        "normalized_mixing_beta_squared_over_KB": normalized_mixing,
        "mixing_beta": float(beta),
        "static_mixing_fraction": fraction,
        "longitudinal_static_capacity": 1.0 / (1.0 - fraction),
        "aether_magnetic_counterterm_fraction": 1.0 - base_speed,
    }


def mixed_hyperbolic_channel(
    *,
    base_speed_squared: float,
    carrier_speed_squared: float,
    normalized_mixing_squared: float,
) -> HyperbolicChannel:
    """Return roots of the high-frequency mixed characteristic polynomial.

    The canonical channel has

    ``(u-y)(s-y)-q y=0`` where ``y=omega^2/k^2``.
    """

    base = _finite_scalar(base_speed_squared, name="base_speed_squared")
    carrier = _finite_scalar(carrier_speed_squared, name="carrier_speed_squared")
    mixing = _finite_scalar(
        normalized_mixing_squared, name="normalized_mixing_squared"
    )
    if base <= 0.0 or carrier <= 0.0 or mixing < 0.0:
        raise ValueError("squared speeds must be positive and mixing non-negative")
    total = base + carrier + mixing
    product = base * carrier
    discriminant = total**2 - 4.0 * product
    if discriminant < -1.0e-14:
        speeds = np.array([np.nan, np.nan])
    else:
        root = np.sqrt(max(discriminant, 0.0))
        speeds = np.array([(total - root) / 2.0, (total + root) / 2.0])
    positive = bool(np.all(np.isfinite(speeds)) and np.all(speeds > 0.0))
    causal = positive and bool(np.all(speeds <= 1.0 + 1.0e-12))
    return HyperbolicChannel(
        base_speed_squared=base,
        carrier_speed_squared=carrier,
        normalized_mixing_squared=mixing,
        speed_squared=speeds,
        positive=positive,
        causal=causal,
    )


def static_principal_channel(
    *,
    k_b: float,
    carrier_spatial_stiffness: float,
    mixing_beta: float,
    canonical_factor: float,
) -> dict[str, object]:
    """Return the high-k static aether-acceleration--carrier block."""

    aether = _finite_scalar(k_b, name="k_b")
    carrier = _finite_scalar(
        carrier_spatial_stiffness, name="carrier_spatial_stiffness"
    )
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    factor = _finite_scalar(canonical_factor, name="canonical_factor")
    if aether <= 0.0 or carrier <= 0.0 or not 0.0 <= factor <= 1.0:
        raise ValueError("stiffnesses must be positive and canonical_factor in [0,1]")
    canonical = beta * factor
    matrix = np.array([[aether, -canonical], [-canonical, carrier]])
    eigenvalues = np.linalg.eigvalsh(matrix)
    determinant = float(np.linalg.det(matrix))
    effective = aether - canonical**2 / carrier
    return {
        "canonical_mixing": canonical,
        "matrix": matrix,
        "eigenvalues": eigenvalues,
        "determinant": determinant,
        "aether_schur_complement": effective,
        "static_response_capacity": np.inf if effective <= 0.0 else aether / effective,
        "positive": bool(np.all(eigenvalues > 0.0)),
    }


def cone_margin(
    *,
    base_speed_squared: float,
    carrier_speed_squared: float,
    normalized_mixing_squared: float,
) -> dict[str, float | bool]:
    """Return analytic positivity and luminal-cone margins."""

    base = _finite_scalar(base_speed_squared, name="base_speed_squared")
    carrier = _finite_scalar(carrier_speed_squared, name="carrier_speed_squared")
    mixing = _finite_scalar(
        normalized_mixing_squared, name="normalized_mixing_squared"
    )
    positive_margin = base * carrier - mixing
    causal_margin = (1.0 - base) * (1.0 - carrier) - mixing
    return {
        "positive_gradient_margin": positive_margin,
        "luminal_upper_cone_margin": causal_margin,
        "positive": bool(positive_margin > 0.0),
        "causal": bool(causal_margin >= -1.0e-12),
    }


def retarded_source_structure() -> dict[str, object]:
    """Return causal/source-uniqueness properties of the hyperbolic carrier."""

    return {
        "carrier_equation_type": "massive quasilinear hyperbolic",
        "carrier_has_time_kinetic_term": True,
        "finite_front_set_by_principal_cone": True,
        "equal_preferred_time_yukawa_constraint_tail": False,
        "static_zero_boundary_solution_unique_from_strict_convexity": True,
        "object_specific_homogeneous_static_profile_allowed": False,
        "universal_dynamic_boundary_condition": "retarded/no incoming carrier radiation",
        "free_carrier_waves_exist": True,
        "free_waves_are_universal_initial_data_not_object_fit": True,
        "nonlinear_global_well_posedness_proved": False,
    }


def linear_metric_structure() -> dict[str, object]:
    """Return the same frozen linear one-metric source identity as v10B."""

    return {
        "static_aether_acceleration": "J_i=partial_i Psi+O(2)",
        "linear_lapse_equation_correction": "beta partial_i partial_j P_ij",
        "linear_spatial_traceless_equation_correction": 0.0,
        "base_no_slip_relation_retained_at_linear_static_order": True,
        "delta_Psi_equals_delta_Phi_equals_delta_Weyl": True,
        "flat_TT_source": 0.0,
        "flat_tensor_speed_squared": 1.0,
        "nonlinear_metric_variation_complete": False,
    }


def audit_v10c_selection(
    *,
    maximum_sourced_base_speed_squared: float,
    static_mixing_fraction: float,
    k_b: float,
    existing_cluster_amplification_target: float,
    physical_parameter_count: int,
    maximum_physical_parameters: int,
) -> dict[str, object]:
    """Run deterministic no-observation v10C selection checks."""

    coefficients = v10c_derived_coefficients(
        maximum_sourced_base_speed_squared=maximum_sourced_base_speed_squared,
        static_mixing_fraction=static_mixing_fraction,
        k_b=k_b,
    )
    base_speed = coefficients["sourced_base_speed_squared"]
    carrier_speed = coefficients["carrier_speed_squared"]
    normalized_mixing = coefficients["normalized_mixing_beta_squared_over_KB"]
    beta = coefficients["mixing_beta"]
    longitudinal = mixed_hyperbolic_channel(
        base_speed_squared=base_speed,
        carrier_speed_squared=carrier_speed,
        normalized_mixing_squared=normalized_mixing,
    )
    transverse = mixed_hyperbolic_channel(
        base_speed_squared=base_speed,
        carrier_speed_squared=carrier_speed,
        normalized_mixing_squared=normalized_mixing / 2.0,
    )
    static_longitudinal = static_principal_channel(
        k_b=k_b,
        carrier_spatial_stiffness=carrier_speed,
        mixing_beta=beta,
        canonical_factor=1.0,
    )
    static_transverse = static_principal_channel(
        k_b=k_b,
        carrier_spatial_stiffness=carrier_speed,
        mixing_beta=beta,
        canonical_factor=1.0 / np.sqrt(2.0),
    )
    margins = {
        "longitudinal": cone_margin(
            base_speed_squared=base_speed,
            carrier_speed_squared=carrier_speed,
            normalized_mixing_squared=normalized_mixing,
        ),
        "transverse": cone_margin(
            base_speed_squared=base_speed,
            carrier_speed_squared=carrier_speed,
            normalized_mixing_squared=normalized_mixing / 2.0,
        ),
    }
    target = _finite_scalar(
        existing_cluster_amplification_target,
        name="existing_cluster_amplification_target",
    )
    if target <= 1.0:
        raise ValueError("existing_cluster_amplification_target must exceed one")
    capacity = float(static_longitudinal["static_response_capacity"])
    gap_closure = (capacity - 1.0) / (target - 1.0)

    isotropic = local_algebraic_carrier_response(beta * np.eye(3))
    rank_one = local_algebraic_carrier_response(beta * np.diag([3.0, 0.0, 0.0]))
    tidal = local_algebraic_carrier_response(
        beta * point_mass_tidal_hessian(1.0, 1.0)
    )
    convexity = {
        str(magnitude): potential_convexity_spectrum(
            np.array([magnitude, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        for magnitude in (0.0, 0.1, 1.0, 10.0)
    }
    retarded = retarded_source_structure()
    metric = linear_metric_structure()
    count = int(physical_parameter_count)
    maximum = int(maximum_physical_parameters)
    if count < 0 or maximum < 0:
        raise ValueError("parameter counts must be non-negative")

    selection_gates = {
        "longitudinal_static_principal_positive": bool(
            static_longitudinal["positive"]
        ),
        "transverse_static_principal_positive": bool(static_transverse["positive"]),
        "longitudinal_mixed_cone_positive_causal": bool(
            longitudinal.positive and longitudinal.causal
        ),
        "transverse_mixed_cone_positive_causal": bool(
            transverse.positive and transverse.causal
        ),
        "unmixed_carrier_cone_positive_causal": bool(0.0 < carrier_speed <= 1.0),
        "aether_vector_bare_cone_positive_causal": bool(0.0 < base_speed <= 1.0),
        "upper_longitudinal_cone_exactly_luminal": bool(
            np.isclose(longitudinal.speed_squared[-1], 1.0, atol=1.0e-12)
        ),
        "longitudinal_capacity_closes_75_percent_gap": gap_closure >= 0.75,
        "carrier_potential_strictly_convex": bool(
            all(item["strictly_convex"] for item in convexity.values())
        ),
        "static_source_state_unique": bool(
            retarded["static_zero_boundary_solution_unique_from_strict_convexity"]
        ),
        "retarded_finite_front_replaces_instantaneous_tail": bool(
            retarded["finite_front_set_by_principal_cone"]
            and not retarded["equal_preferred_time_yukawa_constraint_tail"]
        ),
        "nonzero_trace_and_STF_response": bool(
            trace_stf_decomposition(isotropic)["trace"] != 0.0
            and trace_stf_decomposition(rank_one)["stf_norm"] > 0.0
        ),
        "nonzero_spherical_tidal_response": bool(np.linalg.norm(tidal) > 0.0),
        "linear_same_metric_dynamics_and_Weyl": bool(
            metric["delta_Psi_equals_delta_Phi_equals_delta_Weyl"]
        ),
        "flat_TT_luminal_and_unsourced": bool(
            metric["flat_TT_source"] == 0.0
            and metric["flat_tensor_speed_squared"] == 1.0
        ),
        "parameter_count": count <= maximum,
    }
    unresolved = {
        "full_covariant_Euler_Lagrange_equations": False,
        "full_nonlinear_ADM_constraint_count": False,
        "tilted_and_inhomogeneous_characteristic_cones": False,
        "nonlinear_global_hyperbolicity": False,
        "complete_weak_metric_and_stress_energy_derivation": False,
        "Solar_PPN_and_compact_source_screening": False,
        "FLRW_background_and_all_mode_stability": False,
        "numerical_PDE_convergence": False,
    }
    return {
        "coefficients": coefficients,
        "static_channels": {
            "longitudinal": static_longitudinal,
            "transverse": static_transverse,
        },
        "hyperbolic_channels": {
            "longitudinal": {
                "speed_squared": longitudinal.speed_squared,
                "positive": longitudinal.positive,
                "causal": longitudinal.causal,
            },
            "transverse": {
                "speed_squared": transverse.speed_squared,
                "positive": transverse.positive,
                "causal": transverse.causal,
            },
            "unmixed_carrier_speed_squared": carrier_speed,
            "unmixed_tensor_speed_squared": 1.0,
        },
        "cone_margins": margins,
        "response": {
            "longitudinal_capacity": capacity,
            "existing_target": target,
            "gap_closure_fraction": gap_closure,
        },
        "geometry": {
            "isotropic_response": isotropic,
            "isotropic_decomposition": trace_stf_decomposition(isotropic),
            "rank_one_response": rank_one,
            "rank_one_decomposition": trace_stf_decomposition(rank_one),
            "spherical_tidal_response": tidal,
            "spherical_tidal_decomposition": trace_stf_decomposition(tidal),
        },
        "convexity": convexity,
        "retarded_source_structure": retarded,
        "linear_metric_structure": metric,
        "selection_gates": selection_gates,
        "all_selection_gates_pass": bool(all(selection_gates.values())),
        "unresolved_mandatory_gates": unresolved,
        "all_mandatory_theory_gates_pass": False,
    }
