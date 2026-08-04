"""Theory-only selection checks for the Sigma v10A spatial carrier.

The proposed carrier ``P_mn`` is symmetric and orthogonal to the AeST aether,
so it has six components in the preferred spatial slice.  It is sourced by the
projected symmetric derivative of the AeST spatial scalar gradient.  The trace
can respond to an isotropic interior while the traceless components retain
tidal orientation.

This module deliberately checks only identities that follow from the frozen
selection action: a necessary flat scalar--carrier subblock, convexity of the
carrier potential for a fixed source, tensor covariance, and response
capacity.  It does not claim the full AeST--aether--metric constraint system is
healthy.  In particular, the nonlinear quasistatic AQUAL stiffness and the
transverse aether sectors are separate mandatory gates before observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voidscreen.sigma_v8_aest_galileon import aest_linear_spectrum

Array = np.ndarray


@dataclass(frozen=True)
class MixedSpectrum:
    """Necessary high-frequency scalar--longitudinal-carrier spectrum."""

    gradient_matrix: Array
    speed_squared: Array
    determinant: float
    positive: bool
    causal: bool


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _symmetric_matrix(value: Array, *, name: str) -> Array:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3)")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric")
    return matrix


def carrier_coefficients(base_scalar_speed_squared: float) -> dict[str, float]:
    """Return the fixed v10A coefficient prescription.

    ``c_P^2=1-c_s^2`` and ``beta=c_s^2/2`` add no physical constant.  They
    select the rational values ``1/4`` and ``3/8`` at the frozen AeST point
    ``c_s^2=3/4``.  This is a construction prescription, not a claimed
    symmetry derivation.
    """

    scalar_speed = _finite_scalar(
        base_scalar_speed_squared, name="base_scalar_speed_squared"
    )
    if not 0.0 < scalar_speed < 1.0:
        raise ValueError("base_scalar_speed_squared must lie strictly between zero and one")
    return {
        "carrier_speed_squared": 1.0 - scalar_speed,
        "mixing_beta": 0.5 * scalar_speed,
    }


def mixed_scalar_carrier_spectrum(
    *,
    base_scalar_speed_squared: float,
    carrier_speed_squared: float,
    mixing_beta: float,
) -> MixedSpectrum:
    """Diagonalize the normalized flat scalar--carrier gradient block.

    For a scalar plane wave only ``P_ij n_i n_j`` mixes.  With unit time
    kinetic terms, the high-frequency spatial matrix is

    ``[[c_s^2, beta], [beta, c_P^2]]``.

    The five orthogonal carrier polarizations retain speed ``c_P^2``.  This is
    necessary but not sufficient because the AeST vector and constraints have
    not been included.
    """

    scalar_speed = _finite_scalar(
        base_scalar_speed_squared, name="base_scalar_speed_squared"
    )
    carrier_speed = _finite_scalar(
        carrier_speed_squared, name="carrier_speed_squared"
    )
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if scalar_speed <= 0.0 or carrier_speed <= 0.0:
        raise ValueError("squared speeds must be positive")
    matrix = np.array([[scalar_speed, beta], [beta, carrier_speed]], dtype=float)
    speeds = np.linalg.eigvalsh(matrix)
    determinant = float(np.linalg.det(matrix))
    positive = bool(np.all(speeds > 0.0))
    causal = positive and bool(np.all(speeds <= 1.0 + 1.0e-12))
    return MixedSpectrum(
        gradient_matrix=matrix,
        speed_squared=speeds,
        determinant=determinant,
        positive=positive,
        causal=causal,
    )


def linear_response_amplification(
    k_length: float,
    *,
    base_scalar_stiffness: float,
    carrier_speed_squared: float,
    mixing_beta: float,
) -> float:
    """Return the linear scalar response relative to its uncoupled value.

    Eliminating a carrier with mass ``L_P^-1`` gives

    ``K_eff=K_s-beta^2/[c_P^2+(k L_P)^-2]``.

    The result is only the selected normalized flat subblock.  It must not be
    substituted for the nonlinear AeST quasistatic constitutive matrix.
    """

    scale = _finite_scalar(k_length, name="k_length")
    stiffness = _finite_scalar(base_scalar_stiffness, name="base_scalar_stiffness")
    carrier_speed = _finite_scalar(
        carrier_speed_squared, name="carrier_speed_squared"
    )
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if scale < 0.0:
        raise ValueError("k_length must be non-negative")
    if stiffness <= 0.0 or carrier_speed <= 0.0:
        raise ValueError("stiffnesses must be positive")
    if scale == 0.0:
        return 1.0
    effective = stiffness - beta**2 / (carrier_speed + scale**-2)
    if effective <= 0.0:
        return np.inf
    return stiffness / effective


def asymptotic_response_capacity(
    *,
    base_scalar_stiffness: float,
    carrier_speed_squared: float,
    mixing_beta: float,
) -> float:
    """Return the ``k L_P -> infinity`` amplification of the selected block."""

    stiffness = _finite_scalar(base_scalar_stiffness, name="base_scalar_stiffness")
    carrier_speed = _finite_scalar(
        carrier_speed_squared, name="carrier_speed_squared"
    )
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if stiffness <= 0.0 or carrier_speed <= 0.0:
        raise ValueError("stiffnesses must be positive")
    effective = stiffness - beta**2 / carrier_speed
    if effective <= 0.0:
        return np.inf
    return stiffness / effective


def k_length_for_amplification(
    target_amplification: float,
    *,
    base_scalar_stiffness: float,
    carrier_speed_squared: float,
    mixing_beta: float,
) -> float:
    """Invert :func:`linear_response_amplification` for ``k L_P``."""

    target = _finite_scalar(target_amplification, name="target_amplification")
    stiffness = _finite_scalar(base_scalar_stiffness, name="base_scalar_stiffness")
    carrier_speed = _finite_scalar(
        carrier_speed_squared, name="carrier_speed_squared"
    )
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if target < 1.0:
        raise ValueError("target_amplification must be at least one")
    if target == 1.0:
        return 0.0
    capacity = asymptotic_response_capacity(
        base_scalar_stiffness=stiffness,
        carrier_speed_squared=carrier_speed,
        mixing_beta=beta,
    )
    if target >= capacity:
        raise ValueError("target_amplification must be below the asymptotic capacity")
    removed_stiffness = stiffness * (1.0 - 1.0 / target)
    inverse_scale_squared = beta**2 / removed_stiffness - carrier_speed
    if inverse_scale_squared <= 0.0:
        raise ValueError("target does not have a finite positive scale")
    return float(1.0 / np.sqrt(inverse_scale_squared))


def carrier_potential_hessian(polarization: Array) -> Array:
    """Return the six-dimensional Hessian of ``V=p^2/2+(p^2)^2/4``."""

    vector = np.asarray(polarization, dtype=float)
    if vector.shape != (6,) or np.any(~np.isfinite(vector)):
        raise ValueError("polarization must be a finite six-vector")
    norm_squared = float(vector @ vector)
    return (1.0 + norm_squared) * np.eye(6) + 2.0 * np.outer(vector, vector)


def potential_convexity_spectrum(polarization: Array) -> dict[str, object]:
    """Return exact radial/transverse eigenvalues of the carrier potential."""

    vector = np.asarray(polarization, dtype=float)
    hessian = carrier_potential_hessian(vector)
    eigenvalues = np.linalg.eigvalsh(hessian)
    norm_squared = float(vector @ vector)
    return {
        "norm_squared": norm_squared,
        "eigenvalues": eigenvalues,
        "analytic_transverse_eigenvalue": 1.0 + norm_squared,
        "analytic_radial_eigenvalue": 1.0 + 3.0 * norm_squared,
        "strictly_convex": bool(np.all(eigenvalues > 0.0)),
    }


def local_algebraic_carrier_response(dimensionless_source: Array) -> Array:
    """Solve ``(1+P:P)P=H`` for a local dimensionless source ``H``.

    This is the zero-gradient algebraic limit of the convex carrier equation.
    Its unique solution is parallel to the source.  The full PDE also contains
    the positive spatial-gradient term and requires boundary conditions.
    """

    source = _symmetric_matrix(dimensionless_source, name="dimensionless_source")
    magnitude = float(np.linalg.norm(source))
    if magnitude == 0.0:
        return np.zeros((3, 3), dtype=float)
    # The positive root of r^3+r=h, written without a numerical root choice.
    response_magnitude = (2.0 / np.sqrt(3.0)) * np.sinh(
        np.arcsinh(1.5 * np.sqrt(3.0) * magnitude) / 3.0
    )
    return response_magnitude * source / magnitude


def trace_stf_decomposition(tensor: Array) -> dict[str, object]:
    """Return trace and symmetric trace-free parts of a spatial tensor."""

    matrix = _symmetric_matrix(tensor, name="tensor")
    trace = float(np.trace(matrix))
    stf = matrix - trace * np.eye(3) / 3.0
    return {
        "trace": trace,
        "stf": stf,
        "stf_norm": float(np.linalg.norm(stf)),
        "frobenius_norm": float(np.linalg.norm(matrix)),
    }


def point_mass_tidal_hessian(
    mass: float,
    radius: float,
    direction: Array | None = None,
) -> Array:
    """Return ``(M/r^3)(3 n n-I)`` in dimensionless ``G=1`` units."""

    source_mass = _finite_scalar(mass, name="mass")
    distance = _finite_scalar(radius, name="radius")
    if source_mass <= 0.0 or distance <= 0.0:
        raise ValueError("mass and radius must be positive")
    if direction is None:
        unit = np.array([1.0, 0.0, 0.0])
    else:
        unit = np.asarray(direction, dtype=float)
        if unit.shape != (3,) or np.any(~np.isfinite(unit)):
            raise ValueError("direction must be a finite three-vector")
        norm = float(np.linalg.norm(unit))
        if norm == 0.0:
            raise ValueError("direction must be nonzero")
        unit = unit / norm
    return source_mass / distance**3 * (3.0 * np.outer(unit, unit) - np.eye(3))


def equal_acceleration_geometry_pair(
    *,
    mass_ratio: float = 100.0,
    radius_ratio: float = 10.0,
) -> dict[str, object]:
    """Compare two point-mass Hessians with a declared equal-force scaling."""

    mass_scale = _finite_scalar(mass_ratio, name="mass_ratio")
    radius_scale = _finite_scalar(radius_ratio, name="radius_ratio")
    if mass_scale <= 0.0 or radius_scale <= 0.0:
        raise ValueError("ratios must be positive")
    first = point_mass_tidal_hessian(1.0, 1.0)
    second = point_mass_tidal_hessian(mass_scale, radius_scale)
    return {
        "mass_ratio": mass_scale,
        "radius_ratio": radius_scale,
        "surface_acceleration_ratio": mass_scale / radius_scale**2,
        "tidal_hessian_norm_ratio": float(np.linalg.norm(second) / np.linalg.norm(first)),
        "first_hessian": first,
        "second_hessian": second,
    }


def rotation_covariance_error(source: Array, rotation: Array) -> float:
    """Return relative covariance error of the nonlinear algebraic response."""

    matrix = _symmetric_matrix(source, name="source")
    transform = np.asarray(rotation, dtype=float)
    if transform.shape != (3, 3) or np.any(~np.isfinite(transform)):
        raise ValueError("rotation must be a finite (3, 3) matrix")
    if not np.allclose(transform.T @ transform, np.eye(3), rtol=0.0, atol=1.0e-12):
        raise ValueError("rotation must be orthogonal")
    rotated_source = transform @ matrix @ transform.T
    direct = local_algebraic_carrier_response(rotated_source)
    expected = transform @ local_algebraic_carrier_response(matrix) @ transform.T
    scale = max(float(np.linalg.norm(expected)), np.finfo(float).eps)
    return float(np.linalg.norm(direct - expected) / scale)


def nonlinear_additivity_error(first_source: Array, second_source: Array) -> float:
    """Return relative failure of response additivity for two sources."""

    first = _symmetric_matrix(first_source, name="first_source")
    second = _symmetric_matrix(second_source, name="second_source")
    combined = local_algebraic_carrier_response(first + second)
    separate = local_algebraic_carrier_response(first) + local_algebraic_carrier_response(
        second
    )
    scale = max(float(np.linalg.norm(combined)), np.finfo(float).eps)
    return float(np.linalg.norm(combined - separate) / scale)


def simple_aqual_longitudinal_stiffness(acceleration_ratio: float) -> float:
    """Return ``d[x mu(x)]/dx`` for ``mu=x/(1+x)``.

    This is a diagnostic proxy for the unresolved nonlinear quasistatic gate.
    It is not substituted into the frozen AeST finite-frequency spectrum.
    """

    ratio = _finite_scalar(acceleration_ratio, name="acceleration_ratio")
    if ratio < 0.0:
        raise ValueError("acceleration_ratio must be non-negative")
    return ratio * (ratio + 2.0) / (1.0 + ratio) ** 2


def simple_aqual_static_stiffnesses(acceleration_ratio: float) -> dict[str, float]:
    """Return transverse and longitudinal principal stiffnesses.

    For ``mu(x)=x/(1+x)``, perturbing the flux ``mu(x) S_i`` about a
    constant spatial gradient gives

    ``K_T=mu`` and ``K_L=mu+x dmu/dx``.
    """

    ratio = _finite_scalar(acceleration_ratio, name="acceleration_ratio")
    if ratio < 0.0:
        raise ValueError("acceleration_ratio must be non-negative")
    transverse = ratio / (1.0 + ratio)
    longitudinal = simple_aqual_longitudinal_stiffness(ratio)
    return {
        "transverse": transverse,
        "longitudinal": longitudinal,
    }


def static_high_k_mixed_spectrum(
    acceleration_ratio: float,
    *,
    propagation_cosine: float,
    carrier_speed_squared: float,
    mixing_beta: float,
) -> dict[str, object]:
    """Return the exact quasistatic high-k scalar--carrier gradient block.

    ``propagation_cosine`` is the cosine between the perturbation wavevector
    and the background AQUAL field.  The scalar stiffness is

    ``K(theta)=K_T+(K_L-K_T) cos(theta)^2``.

    The carrier mass and convex potential enter at order ``k^0`` and cannot
    change the sign of this order-``k^2`` block as ``k -> infinity``.
    """

    cosine = _finite_scalar(propagation_cosine, name="propagation_cosine")
    carrier_speed = _finite_scalar(
        carrier_speed_squared, name="carrier_speed_squared"
    )
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if abs(cosine) > 1.0:
        raise ValueError("propagation_cosine must lie between minus one and one")
    if carrier_speed <= 0.0:
        raise ValueError("carrier_speed_squared must be positive")
    stiffnesses = simple_aqual_static_stiffnesses(acceleration_ratio)
    scalar_stiffness = stiffnesses["transverse"] + (
        stiffnesses["longitudinal"] - stiffnesses["transverse"]
    ) * cosine**2
    matrix = np.array(
        [[scalar_stiffness, beta], [beta, carrier_speed]], dtype=float
    )
    eigenvalues = np.linalg.eigvalsh(matrix)
    determinant = float(np.linalg.det(matrix))
    schur = scalar_stiffness - beta**2 / carrier_speed
    return {
        "acceleration_ratio": float(acceleration_ratio),
        "propagation_cosine": cosine,
        "AQUAL_transverse_stiffness": stiffnesses["transverse"],
        "AQUAL_longitudinal_stiffness": stiffnesses["longitudinal"],
        "directional_scalar_stiffness": scalar_stiffness,
        "gradient_matrix": matrix,
        "gradient_eigenvalues": eigenvalues,
        "determinant": determinant,
        "scalar_schur_complement": schur,
        "elliptic": bool(np.all(eigenvalues > 0.0)),
    }


def constant_mixing_ellipticity_thresholds(
    *,
    carrier_speed_squared: float,
    mixing_beta: float,
) -> dict[str, float | bool]:
    """Return simple-mu field thresholds for constant derivative mixing.

    A globally elliptic nonzero constant mixing is impossible because
    ``K_T`` and ``K_L`` both tend to zero as ``x -> 0``.
    """

    carrier_speed = _finite_scalar(
        carrier_speed_squared, name="carrier_speed_squared"
    )
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if carrier_speed <= 0.0:
        raise ValueError("carrier_speed_squared must be positive")
    required_stiffness = beta**2 / carrier_speed
    if required_stiffness >= 1.0:
        transverse_threshold = np.inf
        longitudinal_threshold = np.inf
    elif required_stiffness <= 0.0:
        transverse_threshold = 0.0
        longitudinal_threshold = 0.0
    else:
        transverse_threshold = required_stiffness / (1.0 - required_stiffness)
        longitudinal_threshold = 1.0 / np.sqrt(1.0 - required_stiffness) - 1.0
    return {
        "required_AQUAL_stiffness": required_stiffness,
        "transverse_acceleration_ratio_threshold": float(transverse_threshold),
        "longitudinal_acceleration_ratio_threshold": float(longitudinal_threshold),
        "globally_elliptic_for_all_nonnegative_accelerations": bool(beta == 0.0),
        "carrier_decoupled": bool(beta == 0.0),
    }


def naive_aqual_schur_diagnostic(
    acceleration_ratio: float,
    *,
    carrier_speed_squared: float,
    mixing_beta: float,
) -> dict[str, float | bool]:
    """Expose, without resolving, the constant-mixing deep-AQUAL warning.

    If the selected mixing acted directly on an isolated AQUAL scalar with no
    help from the AeST metric/aether constraints, its high-k Schur complement
    would be ``K_AQUAL-beta^2/c_P^2``.  A negative value kills that decoupled
    proxy.  Only a full reduction can determine whether the covariant action
    has exactly this block, so the result is a mandatory next gate rather than
    a completed-theory claim.
    """

    carrier_speed = _finite_scalar(
        carrier_speed_squared, name="carrier_speed_squared"
    )
    beta = _finite_scalar(mixing_beta, name="mixing_beta")
    if carrier_speed <= 0.0:
        raise ValueError("carrier_speed_squared must be positive")
    stiffness = simple_aqual_longitudinal_stiffness(acceleration_ratio)
    schur = stiffness - beta**2 / carrier_speed
    return {
        "acceleration_ratio": float(acceleration_ratio),
        "AQUAL_longitudinal_stiffness": stiffness,
        "constant_mixing_schur_complement": schur,
        "proxy_elliptic": bool(schur > 0.0),
    }


def audit_v10a_selection(
    *,
    k_b: float,
    k_2: float,
    lambda_s: float,
    physical_parameter_count: int,
    maximum_physical_parameters: int,
    existing_cluster_amplification_target: float,
) -> dict[str, object]:
    """Run the deterministic, observation-free v10A selection audit."""

    base = aest_linear_spectrum(k_b=k_b, k_2=k_2, lambda_s=lambda_s)
    scalar_speed = float(base["scalar_speed_squared"])
    coefficients = carrier_coefficients(scalar_speed)
    carrier_speed = coefficients["carrier_speed_squared"]
    beta = coefficients["mixing_beta"]
    mixed = mixed_scalar_carrier_spectrum(
        base_scalar_speed_squared=scalar_speed,
        carrier_speed_squared=carrier_speed,
        mixing_beta=beta,
    )
    capacity = asymptotic_response_capacity(
        base_scalar_stiffness=scalar_speed,
        carrier_speed_squared=carrier_speed,
        mixing_beta=beta,
    )
    target = _finite_scalar(
        existing_cluster_amplification_target,
        name="existing_cluster_amplification_target",
    )
    if target <= 1.0:
        raise ValueError("existing_cluster_amplification_target must exceed one")
    target_k_length = k_length_for_amplification(
        target,
        base_scalar_stiffness=scalar_speed,
        carrier_speed_squared=carrier_speed,
        mixing_beta=beta,
    )
    seventy_five_percent_target = 1.0 + 0.75 * (target - 1.0)
    seventy_five_percent_k_length = k_length_for_amplification(
        seventy_five_percent_target,
        base_scalar_stiffness=scalar_speed,
        carrier_speed_squared=carrier_speed,
        mixing_beta=beta,
    )

    convexity = {
        str(magnitude): potential_convexity_spectrum(
            np.array([magnitude, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        for magnitude in (0.0, 0.1, 1.0, 10.0)
    }
    isotropic_source = np.eye(3)
    rank_one_source = np.diag([3.0, 0.0, 0.0])
    tidal_source = np.diag([2.0, -1.0, -1.0])
    isotropic_response = local_algebraic_carrier_response(isotropic_source)
    rank_one_response = local_algebraic_carrier_response(rank_one_source)
    tidal_response = local_algebraic_carrier_response(tidal_source)
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    covariance_error = rotation_covariance_error(rank_one_source, rotation)
    additivity_error = nonlinear_additivity_error(
        np.diag([2.0, -1.0, -1.0]),
        rotation @ np.diag([2.0, -1.0, -1.0]) @ rotation.T,
    )
    geometry_pair = equal_acceleration_geometry_pair()
    deep_proxy = [
        naive_aqual_schur_diagnostic(
            ratio,
            carrier_speed_squared=carrier_speed,
            mixing_beta=beta,
        )
        for ratio in (0.01, 0.1, 0.3, 1.0, 10.0)
    ]
    count = int(physical_parameter_count)
    maximum = int(maximum_physical_parameters)
    if count < 0 or maximum < 0:
        raise ValueError("parameter counts must be non-negative")

    selection_gates = {
        "base_finite_frequency_modes_positive": bool(base["positive_propagating_modes"]),
        "base_finite_frequency_modes_causal": bool(base["causal_propagating_modes"]),
        "mixed_scalar_carrier_subblock_positive": mixed.positive,
        "mixed_scalar_carrier_subblock_causal": mixed.causal,
        "five_unmixed_carrier_polarizations_positive_causal": bool(
            0.0 < carrier_speed <= 1.0
        ),
        "carrier_potential_strictly_convex": bool(
            all(item["strictly_convex"] for item in convexity.values())
        ),
        "fixed_source_carrier_state_unique": True,
        "nonzero_isotropic_trace_response": bool(
            abs(float(np.trace(isotropic_response))) > 0.0
        ),
        "nonzero_spherical_exterior_tidal_response": bool(
            np.linalg.norm(tidal_response) > 0.0
        ),
        "trace_and_stf_channels_present": bool(
            trace_stf_decomposition(isotropic_response)["trace"] != 0.0
            and trace_stf_decomposition(rank_one_response)["stf_norm"] > 0.0
        ),
        "rotation_covariant": covariance_error < 1.0e-12,
        "nonlinear_before_summation": additivity_error > 1.0e-3,
        "equal_acceleration_geometry_discriminated": bool(
            np.isclose(geometry_pair["surface_acceleration_ratio"], 1.0)
            and not np.isclose(geometry_pair["tidal_hessian_norm_ratio"], 1.0)
        ),
        "linear_capacity_reaches_existing_target": capacity >= target,
        "parameter_count": count <= maximum,
    }
    unresolved_mandatory_gates = {
        "full_AeST_metric_aether_carrier_constraint_count": False,
        "full_scalar_vector_tensor_principal_symbol": False,
        "global_nonlinear_quasistatic_ellipticity": False,
        "one_metric_weak_field_Psi_Phi_Weyl_derivation": False,
        "PPN_and_Solar_screening": False,
        "cosmological_background_and_stability": False,
        "retarded_no_incoming_source_uniqueness": False,
    }
    return {
        "base_spectrum": base,
        "derived_coefficients": coefficients,
        "mixed_spectrum": {
            "gradient_matrix": mixed.gradient_matrix,
            "speed_squared": mixed.speed_squared,
            "determinant": mixed.determinant,
            "positive": mixed.positive,
            "causal": mixed.causal,
        },
        "linear_capacity": {
            "asymptotic_amplification": capacity,
            "existing_cluster_target": target,
            "kL_for_existing_target": target_k_length,
            "seventy_five_percent_gap_target": seventy_five_percent_target,
            "kL_for_seventy_five_percent_gap": seventy_five_percent_k_length,
        },
        "carrier_potential_convexity": convexity,
        "geometry": {
            "isotropic_response": isotropic_response,
            "isotropic_decomposition": trace_stf_decomposition(isotropic_response),
            "rank_one_response": rank_one_response,
            "rank_one_decomposition": trace_stf_decomposition(rank_one_response),
            "tidal_response": tidal_response,
            "tidal_decomposition": trace_stf_decomposition(tidal_response),
            "rotation_covariance_relative_error": covariance_error,
            "nonlinear_additivity_relative_error": additivity_error,
            "equal_acceleration_pair": geometry_pair,
        },
        "deep_AQUAL_decoupled_proxy_warning": deep_proxy,
        "selection_gates": selection_gates,
        "all_selection_gates_pass": bool(all(selection_gates.values())),
        "unresolved_mandatory_gates": unresolved_mandatory_gates,
        "all_mandatory_theory_gates_pass": False,
    }
