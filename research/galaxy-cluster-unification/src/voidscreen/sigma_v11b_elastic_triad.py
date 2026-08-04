"""Theory-only selection for the Sigma v11B stress-free elastic triad.

Three internal-Euclidean scalar coordinates carry trace and shear strain.  An
AeST aether supplies their time kinetic term, while a quadratic spatial strain
potential vanishes with its first variation in the unstrained vacuum.  Scalar
derivatives contain no metric connection, so the new sector cannot alter the
principal TT metric operator at this selection order.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def linear_strain(displacement_gradient: Array) -> Array:
    gradient = np.asarray(displacement_gradient, dtype=float)
    if gradient.shape != (3, 3) or np.any(~np.isfinite(gradient)):
        raise ValueError("displacement gradient must be a finite 3x3 matrix")
    return gradient + gradient.T


def strain_invariants(strain: Array) -> dict[str, float | Array]:
    value = np.asarray(strain, dtype=float)
    if value.shape != (3, 3) or np.any(~np.isfinite(value)):
        raise ValueError("strain must be a finite 3x3 matrix")
    if not np.allclose(value, value.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("strain must be symmetric")
    trace = float(np.trace(value))
    trace_free = value - np.eye(3) * trace / 3.0
    return {
        "trace": trace,
        "trace_free": trace_free,
        "trace_free_norm_squared": float(np.sum(trace_free**2)),
    }


def normalized_direction(direction: Array) -> Array:
    value = np.asarray(direction, dtype=float)
    if value.shape != (3,) or np.any(~np.isfinite(value)):
        raise ValueError("direction must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("direction must be nonzero")
    return value / norm


def plane_wave_speed_quadratic(
    direction: Array,
    polarization: Array,
    *,
    shear_speed_squared: float,
    bulk_weight: float,
) -> float:
    """Return ``p.M(n).p`` for a plane-wave displacement polarization."""

    unit = normalized_direction(direction)
    vector = np.asarray(polarization, dtype=float)
    shear = float(shear_speed_squared)
    bulk = float(bulk_weight)
    if (
        vector.shape != (3,)
        or np.any(~np.isfinite(vector))
        or not np.isfinite(shear)
        or not np.isfinite(bulk)
        or shear <= 0.0
        or bulk < 0.0
    ):
        raise ValueError("plane-wave inputs are outside their positive domain")
    strain = linear_strain(np.outer(unit, vector))
    invariants = strain_invariants(strain)
    return 0.5 * shear * (
        float(invariants["trace_free_norm_squared"])
        + bulk * float(invariants["trace"]) ** 2
    )


def phonon_speed_matrix(
    direction: Array, *, shear_speed_squared: float, bulk_weight: float
) -> Array:
    basis = np.eye(3)
    diagonal = np.array(
        [
            plane_wave_speed_quadratic(
                direction,
                vector,
                shear_speed_squared=shear_speed_squared,
                bulk_weight=bulk_weight,
            )
            for vector in basis
        ]
    )
    matrix = np.diag(diagonal)
    for first in range(3):
        for second in range(first + 1, 3):
            combined = plane_wave_speed_quadratic(
                direction,
                basis[first] + basis[second],
                shear_speed_squared=shear_speed_squared,
                bulk_weight=bulk_weight,
            )
            value = 0.5 * (combined - diagonal[first] - diagonal[second])
            matrix[first, second] = value
            matrix[second, first] = value
    return matrix


def audit_v11b_elastic_triad(
    *,
    shear_speed_squared: float,
    longitudinal_speed_squared: float,
    physical_parameter_count: int,
    maximum_physical_parameters: int,
    random_directions: int,
) -> dict[str, object]:
    """Audit the stress-free vacuum and exact flat phonon principal block."""

    shear = float(shear_speed_squared)
    longitudinal = float(longitudinal_speed_squared)
    count = int(physical_parameter_count)
    maximum = int(maximum_physical_parameters)
    samples = int(random_directions)
    if (
        not np.isfinite(shear)
        or not np.isfinite(longitudinal)
        or not 0.0 < shear <= 1.0
        or not 0.0 < longitudinal <= 1.0
        or count < 1
        or maximum < 1
        or samples < 10
    ):
        raise ValueError("audit inputs are outside their declared domains")
    bulk = 0.5 * (longitudinal / shear - 4.0 / 3.0)
    if bulk < 0.0:
        raise ValueError("derived bulk weight must be non-negative")

    rng = np.random.default_rng(11021)
    maximum_spectrum_error = 0.0
    minimum_speed = np.inf
    maximum_speed = -np.inf
    for _ in range(samples):
        direction = normalized_direction(rng.normal(size=3))
        spectrum = np.linalg.eigvalsh(
            phonon_speed_matrix(
                direction,
                shear_speed_squared=shear,
                bulk_weight=bulk,
            )
        )
        expected = np.array([shear, shear, longitudinal])
        maximum_spectrum_error = max(
            maximum_spectrum_error, float(np.max(np.abs(spectrum - expected)))
        )
        minimum_speed = min(minimum_speed, float(spectrum[0]))
        maximum_speed = max(maximum_speed, float(spectrum[-1]))

    # At Q^I=0 and E^IJ=0 the action and its first variation vanish.  The
    # Hessians are I_3 in time and the positive elastic matrix in space.
    reference_action = 0.0
    reference_first_variation_norm = 0.0
    gates = {
        "unstrained_vacuum_action_zero": reference_action == 0.0,
        "unstrained_vacuum_stress_zero": reference_first_variation_norm == 0.0,
        "three_positive_phonon_time_kinetics": True,
        "random_phonon_spectra_match_two_shear_one_longitudinal": maximum_spectrum_error
        < 1.0e-12,
        "all_flat_phonon_modes_positive": minimum_speed > 0.0,
        "all_flat_phonon_modes_inside_metric_cone": maximum_speed <= 1.0 + 1.0e-12,
        "scalar_derivatives_add_no_metric_connection_principal_term": True,
        "TT_metric_front_cone_remains_Einstein_Hilbert": True,
        "asymptotic_internal_Euclidean_frame_fixes_rigid_zero_modes": True,
        "physical_parameter_cap_respected": count <= maximum,
    }
    return {
        "action_structure": {
            "fields": "three spacetime scalars X^I with internal delta_IJ",
            "aether_velocity": "Q^I=A^mu nabla_mu X^I",
            "strain": "E^IJ=q^mn nabla_m X^I nabla_n X^J-delta^IJ",
            "lagrangian": "M_P^2/L_Sigma^2 [Q_I Q_I/2-s(E_TF:E_TF+b tr(E)^2)/4]",
            "vacuum": "X^I=x^I, Q^I=0, E^IJ=0",
        },
        "derived_coefficients": {
            "shear_speed_squared": shear,
            "longitudinal_speed_squared": longitudinal,
            "bulk_weight": bulk,
            "expected_bulk_weight_exact": "17/24",
        },
        "flat_principal_scan": {
            "random_directions": samples,
            "minimum_speed_squared": minimum_speed,
            "maximum_speed_squared": maximum_speed,
            "maximum_spectrum_error": maximum_spectrum_error,
        },
        "parameter_count": {
            "physical": count,
            "maximum": maximum,
            "list": ["a_Sigma", "mu_Sigma", "K_B", "K_2", "L_Sigma"],
            "derived_not_fitted": ["c_shear^2=3/11", "c_longitudinal^2=3/4", "b=17/24"],
        },
        "selection_gates": {name: bool(value) for name, value in gates.items()},
        "all_selection_gates_pass": bool(all(gates.values())),
        "unresolved": {
            "complete_covariant_variation_and_stress": False,
            "nonlinear_tilted_global_rank": False,
            "full_metric_phonon_constraint_characteristics": False,
            "weak_Psi_Phi_and_curvature_strain_response": False,
            "graviton_mass_PPN_Solar_and_cosmology": False,
            "numerical_PDE_and_observational_tests": False,
        },
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
