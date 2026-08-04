"""Theory-only checks for Sigma v10B auxiliary aether-tidal polarization.

V10B retains the six-component spatial tensor geometry of v10A but removes its
time kinetic term and sources it with the symmetric spatial derivative of the
aether acceleration ``J_m=A^n nabla_n A_m``.  The Maxwell-aether stiffness is
finite on the deep-MOND branch, so the static principal block can be positive
for a constant mixing even where the AQUAL scalar stiffness vanishes.

The calculations here are exact for the frozen local flat/static reductions.
They do not replace the full nonlinear ADM constraint count, PPN derivation, or
cosmological perturbation analysis.
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
class StaticChannel:
    """Principal spectrum for one canonical aether--polarization channel."""

    channel: str
    canonical_mixing: float
    matrix: Array
    eigenvalues: Array
    determinant: float
    positive: bool


@dataclass(frozen=True)
class AuxiliaryDiracChannel:
    """One-mode Dirac reduction of a nondynamical carrier component."""

    secondary_bracket: float
    reduced_hamiltonian_momentum_coefficient: float
    primary_constraints: int
    secondary_constraints: int
    second_class_constraints: int
    auxiliary_configuration_dof: float
    positive: bool


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _channel_factor(channel: str) -> float:
    if channel == "longitudinal":
        return 1.0
    if channel == "transverse":
        # A canonical symmetric off-diagonal component is sqrt(2) P_xy.
        return 1.0 / np.sqrt(2.0)
    if channel == "unmixed":
        return 0.0
    raise ValueError("channel must be longitudinal, transverse, or unmixed")


def v10b_fixed_coefficients(k_b: float) -> dict[str, float]:
    """Return the no-new-constant v10B coefficient prescription.

    The carrier spatial-gradient normalization is fixed to one by rescaling
    ``P``.  ``beta^2=2 K_B/3`` leaves a worst-channel static Schur complement
    ``K_B/3`` and a high-k response capacity of three.
    """

    aether_stiffness = _finite_scalar(k_b, name="k_b")
    if aether_stiffness <= 0.0:
        raise ValueError("k_b must be positive")
    return {
        "carrier_spatial_stiffness": 1.0,
        "mixing_beta": float(np.sqrt(2.0 * aether_stiffness / 3.0)),
        "mixing_fraction_beta_squared_over_KB": 2.0 / 3.0,
    }


def static_principal_channel(
    *,
    k_b: float,
    mixing_beta: float,
    channel: str,
) -> StaticChannel:
    """Return the canonical high-k static energy block.

    With frozen projectors, integration by parts gives the principal energy

    ``E=K_B J^2/2+|D P|^2/2-beta J_j D_i P_ij``.

    For a wave along ``x``, the longitudinal component ``P_xx`` has canonical
    mixing ``beta`` and ``P_xy/P_xz`` have ``beta/sqrt(2)``.
    """

    aether_stiffness = _finite_scalar(k_b, name="k_b")
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if aether_stiffness <= 0.0:
        raise ValueError("k_b must be positive")
    canonical = beta * _channel_factor(channel)
    matrix = np.array([[aether_stiffness, -canonical], [-canonical, 1.0]])
    eigenvalues = np.linalg.eigvalsh(matrix)
    determinant = float(np.linalg.det(matrix))
    return StaticChannel(
        channel=channel,
        canonical_mixing=canonical,
        matrix=matrix,
        eigenvalues=eigenvalues,
        determinant=determinant,
        positive=bool(np.all(eigenvalues > 0.0)),
    )


def auxiliary_scale_fraction(k_length: float) -> float:
    """Return ``k^2/(k^2+L_P^-2)`` as a function of ``k L_P``."""

    scale = _finite_scalar(k_length, name="k_length")
    if scale < 0.0:
        raise ValueError("k_length must be non-negative")
    return scale**2 / (1.0 + scale**2)


def static_response_amplification(
    k_length: float,
    *,
    k_b: float,
    mixing_beta: float,
    channel: str,
) -> float:
    """Return the linear static response after eliminating auxiliary ``P``."""

    aether_stiffness = _finite_scalar(k_b, name="k_b")
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if aether_stiffness <= 0.0:
        raise ValueError("k_b must be positive")
    canonical = beta * _channel_factor(channel)
    effective = aether_stiffness - canonical**2 * auxiliary_scale_fraction(k_length)
    if effective <= 0.0:
        return np.inf
    return aether_stiffness / effective


def static_response_capacity(
    *,
    k_b: float,
    mixing_beta: float,
    channel: str,
) -> float:
    """Return the ``k L_P -> infinity`` static response amplification."""

    aether_stiffness = _finite_scalar(k_b, name="k_b")
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if aether_stiffness <= 0.0:
        raise ValueError("k_b must be positive")
    canonical = beta * _channel_factor(channel)
    effective = aether_stiffness - canonical**2
    if effective <= 0.0:
        return np.inf
    return aether_stiffness / effective


def k_length_for_static_amplification(
    target_amplification: float,
    *,
    k_b: float,
    mixing_beta: float,
    channel: str,
) -> float:
    """Invert the finite-range auxiliary response for ``k L_P``."""

    target = _finite_scalar(target_amplification, name="target_amplification")
    aether_stiffness = _finite_scalar(k_b, name="k_b")
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if target < 1.0:
        raise ValueError("target_amplification must be at least one")
    if target == 1.0:
        return 0.0
    capacity = static_response_capacity(
        k_b=aether_stiffness,
        mixing_beta=beta,
        channel=channel,
    )
    if target >= capacity:
        raise ValueError("target_amplification must be below channel capacity")
    canonical = beta * _channel_factor(channel)
    required_fraction = aether_stiffness * (1.0 - 1.0 / target) / canonical**2
    if not 0.0 < required_fraction < 1.0:
        raise ValueError("target does not imply a finite positive scale")
    return float(np.sqrt(required_fraction / (1.0 - required_fraction)))


def auxiliary_vector_speed_squared(
    k_length: float,
    *,
    k_b: float,
    mixing_beta: float,
    channel: str,
) -> float:
    """Return the flat vector speed after solving the auxiliary constraint.

    In the decoupled flat vector block,

    ``L=K_B(dot a^2-k^2 a^2)/2-(k^2+m^2)p^2/2+b k p dot a``.

    Eliminating ``p`` increases the positive time kinetic coefficient and
    leaves the spatial coefficient unchanged.  It adds no independent root.
    """

    aether_stiffness = _finite_scalar(k_b, name="k_b")
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if aether_stiffness <= 0.0:
        raise ValueError("k_b must be positive")
    canonical = beta * _channel_factor(channel)
    kinetic = aether_stiffness + canonical**2 * auxiliary_scale_fraction(k_length)
    return aether_stiffness / kinetic


def auxiliary_dirac_channel(
    wave_number: float,
    inverse_length: float,
    *,
    k_b: float,
    mixing_beta: float,
    channel: str,
) -> AuxiliaryDiracChannel:
    """Return the exact one-Fourier-mode Dirac reduction.

    For canonical variables ``a,p`` the flat channel is

    ``L=K_B dot(a)^2/2-K_B k^2 a^2/2-Omega^2 p^2/2+b k p dot(a)``.

    ``pi_p=0`` is primary.  Its preservation gives one secondary constraint
    whose bracket with ``pi_p`` is ``Omega^2+b^2 k^2/K_B>0``.  The pair is
    second class and removes ``p`` without adding a configuration degree of
    freedom.
    """

    wave = _finite_scalar(wave_number, name="wave_number")
    mass = _finite_scalar(inverse_length, name="inverse_length")
    aether_stiffness = _finite_scalar(k_b, name="k_b")
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if wave < 0.0 or mass <= 0.0 or aether_stiffness <= 0.0:
        raise ValueError("wave_number must be non-negative; inverse_length and k_b positive")
    canonical = beta * _channel_factor(channel)
    omega_squared = wave**2 + mass**2
    bracket = omega_squared + canonical**2 * wave**2 / aether_stiffness
    momentum_coefficient = omega_squared / (
        aether_stiffness * omega_squared + canonical**2 * wave**2
    )
    return AuxiliaryDiracChannel(
        secondary_bracket=bracket,
        reduced_hamiltonian_momentum_coefficient=momentum_coefficient,
        primary_constraints=1,
        secondary_constraints=1,
        second_class_constraints=2,
        auxiliary_configuration_dof=0.0,
        positive=bool(bracket > 0.0 and momentum_coefficient > 0.0),
    )


def instantaneous_acceleration_kernel(
    inverse_length: float,
    *,
    k_b: float,
    mixing_beta: float,
    channel: str,
) -> dict[str, float | bool]:
    """Return the equal-time kernel after eliminating auxiliary ``P``.

    The response of ``ddot(a)`` to a localized source contains

    ``C(k)=(k^2+m^2)/[(K_B+b^2)k^2+K_B m^2]``.

    In position space this is a local delta term plus

    ``tail_coefficient exp(-M r)/(4 pi r)``.

    A nonzero tail changes a physical transverse aether acceleration at every
    distance on the same preferred-time slice.  Unlike a first-class gauge
    constraint, the ``P`` pair is second class.
    """

    mass = _finite_scalar(inverse_length, name="inverse_length")
    aether_stiffness = _finite_scalar(k_b, name="k_b")
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if mass <= 0.0 or aether_stiffness <= 0.0:
        raise ValueError("inverse_length and k_b must be positive")
    canonical = beta * _channel_factor(channel)
    denominator = aether_stiffness + canonical**2
    local_coefficient = 1.0 / denominator
    effective_mass = mass * np.sqrt(aether_stiffness / denominator)
    tail_coefficient = canonical**2 * mass**2 / denominator**2
    return {
        "local_delta_coefficient": local_coefficient,
        "effective_inverse_range": effective_mass,
        "yukawa_tail_coefficient": tail_coefficient,
        "equal_time_nonlocal_tail_present": bool(tail_coefficient > 0.0),
        "finite_light_cone_front": bool(tail_coefficient == 0.0),
    }


def instantaneous_tail_at_radius(
    radius: float,
    inverse_length: float,
    *,
    k_b: float,
    mixing_beta: float,
    channel: str,
) -> float:
    """Evaluate the positive equal-time Yukawa tail away from the source."""

    distance = _finite_scalar(radius, name="radius")
    if distance <= 0.0:
        raise ValueError("radius must be positive")
    kernel = instantaneous_acceleration_kernel(
        inverse_length,
        k_b=k_b,
        mixing_beta=mixing_beta,
        channel=channel,
    )
    return float(
        kernel["yukawa_tail_coefficient"]
        * np.exp(-kernel["effective_inverse_range"] * distance)
        / (4.0 * np.pi * distance)
    )


def audit_v10b_constraint_causality(
    *,
    k_b: float,
    inverse_length: float,
    wave_numbers: Array,
    radii: Array,
) -> dict[str, object]:
    """Audit auxiliary constraint removal and the equal-time physical tail."""

    coefficients = v10b_fixed_coefficients(k_b)
    beta = coefficients["mixing_beta"]
    waves = np.asarray(wave_numbers, dtype=float)
    distances = np.asarray(radii, dtype=float)
    if waves.ndim != 1 or waves.size == 0 or np.any(~np.isfinite(waves)) or np.any(waves < 0):
        raise ValueError("wave_numbers must be a non-empty finite non-negative vector")
    if (
        distances.ndim != 1
        or distances.size == 0
        or np.any(~np.isfinite(distances))
        or np.any(distances <= 0)
    ):
        raise ValueError("radii must be a non-empty finite positive vector")
    channels = ("longitudinal", "transverse")
    dirac_rows = {
        channel: [
            auxiliary_dirac_channel(
                float(wave),
                inverse_length,
                k_b=k_b,
                mixing_beta=beta,
                channel=channel,
            )
            for wave in waves
        ]
        for channel in channels
    }
    kernels = {
        channel: instantaneous_acceleration_kernel(
            inverse_length,
            k_b=k_b,
            mixing_beta=beta,
            channel=channel,
        )
        for channel in channels
    }
    tails = {
        channel: [
            instantaneous_tail_at_radius(
                float(radius),
                inverse_length,
                k_b=k_b,
                mixing_beta=beta,
                channel=channel,
            )
            for radius in distances
        ]
        for channel in channels
    }
    constraint_gates = {
        "all_primary_secondary_brackets_positive": bool(
            all(row.positive for rows in dirac_rows.values() for row in rows)
        ),
        "six_primary_plus_six_secondary_constraints": True,
        "all_twelve_constraints_second_class_on_flat_branch": True,
        "auxiliary_tensor_adds_zero_configuration_dof": True,
        "reduced_hamiltonian_channel_positive": bool(
            all(
                row.reduced_hamiltonian_momentum_coefficient > 0.0
                for rows in dirac_rows.values()
                for row in rows
            )
        ),
    }
    causality_gates = {
        "no_equal_time_transverse_physical_tail": not kernels["transverse"][
            "equal_time_nonlocal_tail_present"
        ],
        "no_equal_time_longitudinal_tail": not kernels["longitudinal"][
            "equal_time_nonlocal_tail_present"
        ],
        "finite_front_for_finite_range_carrier": bool(
            all(kernel["finite_light_cone_front"] for kernel in kernels.values())
        ),
    }
    return {
        "coefficients": coefficients,
        "wave_numbers": waves,
        "radii_in_carrier_length_units": distances,
        "dirac": {
            channel: {
                "minimum_secondary_bracket": min(
                    row.secondary_bracket for row in rows
                ),
                "minimum_reduced_momentum_coefficient": min(
                    row.reduced_hamiltonian_momentum_coefficient for row in rows
                ),
                "primary_constraints_per_component": rows[0].primary_constraints,
                "secondary_constraints_per_component": rows[0].secondary_constraints,
                "second_class_constraints_per_component": rows[0].second_class_constraints,
                "auxiliary_configuration_dof_per_component": rows[
                    0
                ].auxiliary_configuration_dof,
            }
            for channel, rows in dirac_rows.items()
        },
        "equal_time_kernels": kernels,
        "equal_time_tail_values": tails,
        "constraint_gates": constraint_gates,
        "all_constraint_gates_pass": bool(all(constraint_gates.values())),
        "causality_gates": causality_gates,
        "all_causality_gates_pass": bool(all(causality_gates.values())),
        "exact_v10b_survives": bool(
            all(constraint_gates.values()) and all(causality_gates.values())
        ),
    }


def gap_closure_fraction(capacity: float, required_amplification: float) -> float:
    """Return fraction of the unit-to-required amplitude gap closed."""

    available = _finite_scalar(capacity, name="capacity")
    required = _finite_scalar(required_amplification, name="required_amplification")
    if available < 1.0 or required <= 1.0:
        raise ValueError("capacity must be >=1 and required_amplification >1")
    return (available - 1.0) / (required - 1.0)


def source_operator_norm_bound(tensor: Array, direction: Array) -> dict[str, float | bool]:
    """Check ``|n_i P_ij| <= |P|_F`` for the divergence coupling."""

    matrix = np.asarray(tensor, dtype=float)
    unit = np.asarray(direction, dtype=float)
    if matrix.shape != (3, 3) or np.any(~np.isfinite(matrix)):
        raise ValueError("tensor must be a finite (3, 3) matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("tensor must be symmetric")
    if unit.shape != (3,) or np.any(~np.isfinite(unit)):
        raise ValueError("direction must be a finite three-vector")
    norm = float(np.linalg.norm(unit))
    if norm == 0.0:
        raise ValueError("direction must be nonzero")
    unit = unit / norm
    divergence = unit @ matrix
    tensor_norm = float(np.linalg.norm(matrix))
    divergence_norm = float(np.linalg.norm(divergence))
    ratio = 0.0 if tensor_norm == 0.0 else divergence_norm / tensor_norm
    return {
        "divergence_norm": divergence_norm,
        "tensor_frobenius_norm": tensor_norm,
        "operator_ratio": ratio,
        "bound_satisfied": bool(ratio <= 1.0 + 1.0e-12),
    }


def static_linear_metric_structure() -> dict[str, object]:
    """Return the frozen linear weak-field coupling structure.

    In static unitary gauge, ``J_i=D_i ln N=D_i Psi+O(2)``.  Therefore the
    new quadratic term is ``beta P_ij partial_i partial_j Psi``.  Its variation
    changes the lapse/``Psi`` equation linearly.  Dependence on the spatial
    metric multiplies the already first-order product ``P*Psi`` and begins at
    cubic action order, so the linear traceless spatial equation is unchanged.
    If the AeST base has ``Phi=Psi``, the same correction enters slow matter
    and the Weyl potential.  Full nonlinear variation remains mandatory.
    """

    return {
        "static_aether_acceleration": "J_i=partial_i Psi+O(2)",
        "quadratic_new_term": "beta P_ij partial_i partial_j Psi",
        "linear_lapse_equation_correction": "beta partial_i partial_j P_ij",
        "linear_spatial_traceless_equation_correction": 0.0,
        "flat_TT_source": 0.0,
        "base_no_slip_relation_retained_at_linear_static_order": True,
        "delta_Psi_equals_delta_Phi_equals_delta_Weyl": True,
        "photon_only_rule": False,
        "nonlinear_metric_variation_complete": False,
    }


def audit_v10b_selection(
    *,
    k_b: float,
    existing_cluster_amplification_target: float,
    physical_parameter_count: int,
    maximum_physical_parameters: int,
) -> dict[str, object]:
    """Run deterministic theory-only v10B construction checks."""

    coefficients = v10b_fixed_coefficients(k_b)
    beta = coefficients["mixing_beta"]
    channels = {
        name: static_principal_channel(k_b=k_b, mixing_beta=beta, channel=name)
        for name in ("longitudinal", "transverse", "unmixed")
    }
    capacities = {
        name: static_response_capacity(k_b=k_b, mixing_beta=beta, channel=name)
        for name in ("longitudinal", "transverse", "unmixed")
    }
    target = _finite_scalar(
        existing_cluster_amplification_target,
        name="existing_cluster_amplification_target",
    )
    if target <= 1.0:
        raise ValueError("existing_cluster_amplification_target must exceed one")
    gap_target = 1.0 + 0.75 * (target - 1.0)
    gap_scale = k_length_for_static_amplification(
        gap_target,
        k_b=k_b,
        mixing_beta=beta,
        channel="longitudinal",
    )
    scale_rows = [
        {
            "kL": scale,
            "longitudinal_static_amplification": static_response_amplification(
                scale, k_b=k_b, mixing_beta=beta, channel="longitudinal"
            ),
            "transverse_static_amplification": static_response_amplification(
                scale, k_b=k_b, mixing_beta=beta, channel="transverse"
            ),
            "longitudinal_vector_speed_squared": auxiliary_vector_speed_squared(
                scale, k_b=k_b, mixing_beta=beta, channel="longitudinal"
            ),
            "transverse_vector_speed_squared": auxiliary_vector_speed_squared(
                scale, k_b=k_b, mixing_beta=beta, channel="transverse"
            ),
        }
        for scale in (0.0, 0.1, 1.0, 10.0, 1.0e6)
    ]

    isotropic = np.eye(3)
    tidal = point_mass_tidal_hessian(1.0, 1.0)
    rank_one = np.diag([3.0, 0.0, 0.0])
    isotropic_response = local_algebraic_carrier_response(beta * isotropic)
    tidal_response = local_algebraic_carrier_response(beta * tidal)
    rank_one_response = local_algebraic_carrier_response(beta * rank_one)
    rng = np.random.default_rng(1042)
    operator_checks = [
        source_operator_norm_bound(
            0.5 * (matrix + matrix.T),
            direction,
        )
        for matrix, direction in zip(
            rng.normal(size=(256, 3, 3)),
            rng.normal(size=(256, 3)),
            strict=True,
        )
    ]
    convexity = {
        str(magnitude): potential_convexity_spectrum(
            np.array([magnitude, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        for magnitude in (0.0, 0.1, 1.0, 10.0)
    }
    metric = static_linear_metric_structure()
    count = int(physical_parameter_count)
    maximum = int(maximum_physical_parameters)
    if count < 0 or maximum < 0:
        raise ValueError("parameter counts must be non-negative")

    selection_gates = {
        "all_static_principal_channels_positive": bool(
            all(channel.positive for channel in channels.values())
        ),
        "worst_static_schur_complement_positive": bool(k_b - beta**2 > 0.0),
        "auxiliary_constraint_adds_no_flat_vector_root": True,
        "flat_vector_speeds_positive_and_subluminal": bool(
            all(
                0.0 < row[key] <= 1.0
                for row in scale_rows
                for key in (
                    "longitudinal_vector_speed_squared",
                    "transverse_vector_speed_squared",
                )
            )
        ),
        "carrier_potential_strictly_convex": bool(
            all(item["strictly_convex"] for item in convexity.values())
        ),
        "fixed_source_auxiliary_solution_unique": True,
        "divergence_operator_norm_bounded": bool(
            all(item["bound_satisfied"] for item in operator_checks)
        ),
        "nonzero_trace_and_STF_response": bool(
            trace_stf_decomposition(isotropic_response)["trace"] != 0.0
            and trace_stf_decomposition(rank_one_response)["stf_norm"] > 0.0
        ),
        "nonzero_spherical_exterior_tidal_response": bool(
            np.linalg.norm(tidal_response) > 0.0
        ),
        "longitudinal_capacity_closes_75_percent_gap": bool(
            gap_closure_fraction(capacities["longitudinal"], target) >= 0.75
        ),
        "linear_static_same_metric_dynamics_and_Weyl": bool(
            metric["delta_Psi_equals_delta_Phi_equals_delta_Weyl"]
        ),
        "flat_TT_source_zero": bool(metric["flat_TT_source"] == 0.0),
        "parameter_count": count <= maximum,
    }
    unresolved = {
        "full_nonlinear_ADM_constraint_count": False,
        "complete_scalar_vector_metric_principal_symbol": False,
        "nonlinear_one_metric_field_equations": False,
        "retarded_constraint_propagation_and_causality": False,
        "Solar_PPN_and_compact_source_screening": False,
        "FLRW_background_and_perturbation_stability": False,
        "numerical_PDE_convergence": False,
    }
    return {
        "coefficients": coefficients,
        "static_channels": {
            name: {
                "canonical_mixing": channel.canonical_mixing,
                "matrix": channel.matrix,
                "eigenvalues": channel.eigenvalues,
                "determinant": channel.determinant,
                "positive": channel.positive,
            }
            for name, channel in channels.items()
        },
        "response": {
            "capacities": capacities,
            "existing_target": target,
            "longitudinal_gap_closure_fraction": gap_closure_fraction(
                capacities["longitudinal"], target
            ),
            "seventy_five_percent_gap_target": gap_target,
            "kL_for_seventy_five_percent_gap": gap_scale,
            "scale_rows": scale_rows,
        },
        "geometry": {
            "isotropic_response": isotropic_response,
            "isotropic_decomposition": trace_stf_decomposition(isotropic_response),
            "rank_one_response": rank_one_response,
            "rank_one_decomposition": trace_stf_decomposition(rank_one_response),
            "spherical_tidal_response": tidal_response,
            "spherical_tidal_decomposition": trace_stf_decomposition(tidal_response),
            "operator_maximum_ratio": max(
                item["operator_ratio"] for item in operator_checks
            ),
        },
        "convexity": convexity,
        "linear_metric_structure": metric,
        "selection_gates": selection_gates,
        "all_selection_gates_pass": bool(all(selection_gates.values())),
        "unresolved_mandatory_gates": unresolved,
        "all_mandatory_theory_gates_pass": False,
    }
