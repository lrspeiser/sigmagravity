"""Direct flat-background characteristic regression for Sigma v12A.

The local quadratic action is first varied with every ADM metric and aether
component retained.  Only after the Euler polynomial

``P(s)=s^2 K+s(A-A.T)-B``

has been formed do we impose the three spatial-diffeomorphism gauges per real
Fourier phase.  The remaining singular quadratic pencil is linearized as a
generalized eigenproblem.  Its finite roots reproduce the six local AeST
degrees of freedom: two tensor, two vector, one finite-frequency scalar, and
one zero-frequency scalar sector.

The exact ``Y=0`` endpoint of the frozen simple AeST interpolation is not
twice differentiable in the square-root representation used by Torch.  The
audit therefore approaches it from a preregistered positive aether-tilt
sentinel and extrapolates the finite-wave results in ``1/k^2``.  It is a flat
linear regression only; it does not establish tilted-background hyperbolicity
or the nonlinear Hamiltonian degree count.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy import linalg

from voidscreen.sigma_v12a_tilted_principal import (
    FIELD_COUNT,
    TiltedPrincipalBackground,
    mode_lagrangian_hessian,
)

# With the Fourier wave vector on axis 3, spatial diffeomorphisms set
# h13=h23=h33=0.  The indices are applied to each real sine/cosine phase only
# after the complete Euler matrices have been constructed.
SPATIAL_GAUGE_BASE_INDICES = (6, 8, 9)


def _phase_doubled_indices(base_indices: tuple[int, ...]) -> tuple[int, ...]:
    return (*base_indices, *(index + FIELD_COUNT for index in base_indices))


def gauge_fixed_characteristic_pencil(
    background: TiltedPrincipalBackground,
    *,
    wave_number_ratio: float,
) -> dict[str, object]:
    """Return the generalized Euler pencil after spatial gauge fixing.

    For

    ``L=1/2 qdot.T K qdot+qdot.T A q+1/2 q.T B q``

    the constant-coefficient Euler equation is

    ``K qddot+(A-A.T) qdot-B q=0``.

    Setting ``q=exp(s t) u`` gives the quadratic matrix polynomial used here.
    """

    mode = mode_lagrangian_hessian(
        background,
        wave_number_ratio=wave_number_ratio,
    )
    full_kinetic = np.asarray(mode["K"], dtype=float)
    full_mixing = np.asarray(mode["A"], dtype=float)
    full_potential = np.asarray(mode["B"], dtype=float)
    full_gyroscopic = full_mixing - full_mixing.T

    gauge_indices = _phase_doubled_indices(SPATIAL_GAUGE_BASE_INDICES)
    retained_indices = np.asarray(
        [index for index in range(2 * FIELD_COUNT) if index not in set(gauge_indices)],
        dtype=int,
    )
    kinetic = full_kinetic[np.ix_(retained_indices, retained_indices)]
    gyroscopic = full_gyroscopic[np.ix_(retained_indices, retained_indices)]
    potential = full_potential[np.ix_(retained_indices, retained_indices)]

    size = retained_indices.size
    zero = np.zeros((size, size), dtype=float)
    identity = np.eye(size, dtype=float)
    left = np.block([[zero, identity], [potential, -gyroscopic]])
    right = np.block([[identity, zero], [zero, kinetic]])
    return {
        "background": asdict(background),
        "wave_number_ratio": float(wave_number_ratio),
        "full_euler_dimension": int(2 * FIELD_COUNT),
        "spatial_gauge_indices": list(gauge_indices),
        "retained_indices": retained_indices,
        "gauge_fixed_euler_dimension": int(size),
        "K": kinetic,
        "C": gyroscopic,
        "B": potential,
        "left_pencil": left,
        "right_pencil": right,
        "gauge_applied_after_full_euler_matrices": True,
    }


def _group_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean_real": float(np.mean(values.real)),
        "minimum_real": float(np.min(values.real)),
        "maximum_real": float(np.max(values.real)),
        "maximum_absolute_imaginary": float(np.max(np.abs(values.imag))),
    }


def generalized_characteristic_eigensystem(
    background: TiltedPrincipalBackground,
    *,
    wave_number_ratio: float,
    infinity_tolerance: float = 1.0e-6,
) -> dict[str, object]:
    """Return finite roots using homogeneous generalized eigenvalues.

    Away from the flat clock, roundoff can turn an exact Class-Ia infinite
    constraint root into an enormous finite quotient.  Classifying the
    homogeneous pair ``(alpha,beta)`` before division keeps those roots at
    infinity and avoids mistaking a numerical constraint root for a physical
    characteristic.
    """

    tolerance = float(infinity_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("infinity_tolerance must be finite and positive")
    pencil = gauge_fixed_characteristic_pencil(
        background,
        wave_number_ratio=wave_number_ratio,
    )
    left = np.asarray(pencil["left_pencil"], dtype=float)
    right = np.asarray(pencil["right_pencil"], dtype=float)
    homogeneous, eigenvectors = linalg.eig(
        left,
        right,
        homogeneous_eigvals=True,
    )
    alpha = homogeneous[0]
    beta = homogeneous[1]
    finite_mask = np.abs(beta) > tolerance * np.maximum(1.0, np.abs(alpha))
    finite_roots = alpha[finite_mask] / beta[finite_mask]
    return {
        **pencil,
        "finite_roots": finite_roots,
        "finite_eigenvectors": eigenvectors[:, finite_mask],
        "finite_generalized_root_count": int(np.count_nonzero(finite_mask)),
        "infinite_generalized_root_count": int(np.count_nonzero(~finite_mask)),
        "minimum_finite_homogeneous_beta_margin": float(
            np.min(
                np.abs(beta[finite_mask])
                / (tolerance * np.maximum(1.0, np.abs(alpha[finite_mask])))
            )
        ),
        "maximum_infinite_homogeneous_beta_margin": float(
            np.max(
                np.abs(beta[~finite_mask])
                / (tolerance * np.maximum(1.0, np.abs(alpha[~finite_mask])))
            )
        ),
    }


def flat_characteristic_spectrum(
    background: TiltedPrincipalBackground,
    *,
    wave_number_ratio: float,
    finite_frequency_threshold: float = 1.0e-3,
) -> dict[str, object]:
    """Solve and classify the gauge-fixed finite characteristic roots.

    The real sine/cosine representation duplicates every complex Fourier
    amplitude.  A physical configuration degree of freedom therefore gives
    four finite generalized roots: two time directions times two real phases.
    """

    threshold = float(finite_frequency_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("finite_frequency_threshold must be finite and positive")
    pencil = generalized_characteristic_eigensystem(
        background,
        wave_number_ratio=wave_number_ratio,
    )
    kinetic = np.asarray(pencil["K"], dtype=float)
    gyroscopic = np.asarray(pencil["C"], dtype=float)
    potential = np.asarray(pencil["B"], dtype=float)
    finite_roots = np.asarray(pencil["finite_roots"], dtype=complex)
    finite_vectors = np.asarray(pencil["finite_eigenvectors"], dtype=complex)
    wave_number = float(wave_number_ratio)
    speed_squared = -(finite_roots / wave_number) ** 2
    order = np.argsort(speed_squared.real)
    finite_roots = finite_roots[order]
    finite_vectors = finite_vectors[:, order]
    speed_squared = speed_squared[order]

    maximum_polynomial_residual = 0.0
    propagating_energies: list[float] = []
    serialized_roots: list[dict[str, float | None]] = []
    euler_dimension = kinetic.shape[0]
    kinetic_norm = float(np.linalg.norm(kinetic))
    gyroscopic_norm = float(np.linalg.norm(gyroscopic))
    potential_norm = float(np.linalg.norm(potential))
    for root, vector, speed in zip(
        finite_roots,
        finite_vectors.T,
        speed_squared,
        strict=True,
    ):
        amplitude = vector[:euler_dimension]
        amplitude_norm = float(np.linalg.norm(amplitude))
        if amplitude_norm <= 0.0:
            polynomial_residual = np.inf
            normalized_amplitude = amplitude
        else:
            normalized_amplitude = amplitude / amplitude_norm
            polynomial = (
                root**2 * kinetic + root * gyroscopic - potential
            )
            scale = max(
                1.0,
                abs(root) ** 2 * kinetic_norm
                + abs(root) * gyroscopic_norm
                + potential_norm,
            )
            polynomial_residual = float(
                np.linalg.norm(polynomial @ normalized_amplitude) / scale
            )
        maximum_polynomial_residual = max(
            maximum_polynomial_residual,
            polynomial_residual,
        )

        energy: float | None = None
        if (
            speed.real > threshold
            and abs(speed.imag) < 1.0e-7
            and abs(root.real) < 1.0e-6 * max(1.0, abs(root.imag))
        ):
            omega = abs(float(root.imag))
            energy = 0.25 * float(
                np.real(
                    np.vdot(
                        normalized_amplitude,
                        (omega**2 * kinetic - potential) @ normalized_amplitude,
                    )
                )
            )
            propagating_energies.append(energy)
        serialized_roots.append(
            {
                "growth_rate_real": float(root.real),
                "growth_rate_imaginary": float(root.imag),
                "speed_squared_real": float(speed.real),
                "speed_squared_imaginary": float(speed.imag),
                "normalized_mode_energy": energy,
                "polynomial_residual": polynomial_residual,
            }
        )

    if speed_squared.size != 24:
        zero_group = speed_squared[:0]
        scalar_group = speed_squared[:0]
        luminal_group = speed_squared[:0]
    else:
        zero_group = speed_squared[:4]
        scalar_group = speed_squared[4:8]
        luminal_group = speed_squared[8:]
    finite_count = int(finite_roots.size)
    inferred_degrees = float(finite_count / 4.0)
    return {
        "background": pencil["background"],
        "wave_number_ratio": wave_number,
        "full_euler_dimension": pencil["full_euler_dimension"],
        "gauge_fixed_euler_dimension": pencil["gauge_fixed_euler_dimension"],
        "gauge_applied_after_full_euler_matrices": True,
        "finite_generalized_root_count": finite_count,
        "infinite_generalized_root_count": int(
            pencil["infinite_generalized_root_count"]
        ),
        "minimum_finite_homogeneous_beta_margin": pencil[
            "minimum_finite_homogeneous_beta_margin"
        ],
        "maximum_infinite_homogeneous_beta_margin": pencil[
            "maximum_infinite_homogeneous_beta_margin"
        ],
        "inferred_linear_physical_degrees_of_freedom": inferred_degrees,
        "groups": {
            "zero_frequency_sector": _group_summary(zero_group),
            "finite_scalar": _group_summary(scalar_group),
            "tensor_vector_luminal": _group_summary(luminal_group),
        },
        "maximum_polynomial_residual": maximum_polynomial_residual,
        "propagating_energy_root_count": len(propagating_energies),
        "minimum_normalized_propagating_mode_energy": (
            float(min(propagating_energies)) if propagating_energies else None
        ),
        "all_identified_finite_frequency_mode_energies_positive": bool(
            len(propagating_energies) == 20 and min(propagating_energies) > 0.0
        ),
        "roots": serialized_roots,
    }


def _linear_intercept(x_values: np.ndarray, y_values: np.ndarray) -> float:
    coefficients = np.polyfit(x_values, y_values, deg=1)
    return float(coefficients[1])


def audit_v12a_flat_characteristics(
    *,
    k_b: float,
    k_2: float,
    background_clock_ratio: float,
    aligned_tilt_sentinel: float,
    orientation_strengths: tuple[float, float],
    wave_number_sentinels: tuple[float, float, float],
    scalar_speed_squared_target: float,
    scalar_limit_tolerance: float,
    luminal_limit_tolerance: float,
    polynomial_residual_tolerance: float,
) -> dict[str, object]:
    """Regress the full quadratic pencil against the flat AeST spectrum."""

    strengths = tuple(float(value) for value in orientation_strengths)
    wave_numbers = tuple(float(value) for value in wave_number_sentinels)
    if len(strengths) != 2 or not strengths[0] < 0.0 < strengths[1]:
        raise ValueError("one negative and one positive orientation strength are required")
    if len(wave_numbers) != 3 or any(value <= 0.0 for value in wave_numbers):
        raise ValueError("three positive wave-number sentinels are required")
    if not wave_numbers[0] < wave_numbers[1] < wave_numbers[2]:
        raise ValueError("wave-number sentinels must be strictly increasing")
    if aligned_tilt_sentinel <= 0.0:
        raise ValueError("aligned_tilt_sentinel must be positive")

    branch_rows: dict[str, list[dict[str, object]]] = {}
    principal_limits: dict[str, dict[str, float]] = {}
    maximum_polynomial_residual = 0.0
    maximum_speed_squared_imaginary = 0.0
    minimum_propagating_energy = np.inf
    valid_counts = True
    all_positive_energies = True
    branch_speed_arrays: dict[str, list[np.ndarray]] = {}
    for strength in strengths:
        name = "negative" if strength < 0.0 else "positive"
        rows = []
        branch_speed_arrays[name] = []
        for wave_number in wave_numbers:
            background = TiltedPrincipalBackground(
                scalar_clock_ratio=float(background_clock_ratio),
                aether_parallel=0.0,
                aether_perpendicular=float(aligned_tilt_sentinel),
                background_clock_ratio=float(background_clock_ratio),
                orientation_strength=float(strength),
                k_b=float(k_b),
                k_2=float(k_2),
            )
            row = flat_characteristic_spectrum(
                background,
                wave_number_ratio=wave_number,
            )
            rows.append(row)
            speeds = np.asarray(
                [
                    complex(
                        item["speed_squared_real"],
                        item["speed_squared_imaginary"],
                    )
                    for item in row["roots"]
                ]
            )
            branch_speed_arrays[name].append(speeds)
            valid_counts = bool(
                valid_counts
                and row["finite_generalized_root_count"] == 24
                and row["infinite_generalized_root_count"] == 16
                and row["inferred_linear_physical_degrees_of_freedom"] == 6.0
                and row["groups"]["zero_frequency_sector"]["count"] == 4
                and row["groups"]["finite_scalar"]["count"] == 4
                and row["groups"]["tensor_vector_luminal"]["count"] == 16
            )
            maximum_polynomial_residual = max(
                maximum_polynomial_residual,
                float(row["maximum_polynomial_residual"]),
            )
            maximum_speed_squared_imaginary = max(
                maximum_speed_squared_imaginary,
                *(abs(value.imag) for value in speeds),
            )
            all_positive_energies = bool(
                all_positive_energies
                and row["all_identified_finite_frequency_mode_energies_positive"]
            )
            minimum_propagating_energy = min(
                minimum_propagating_energy,
                float(row["minimum_normalized_propagating_mode_energy"]),
            )
        branch_rows[name] = rows

        inverse_wave_squared = np.asarray([1.0 / value**2 for value in wave_numbers])
        zero_means = np.asarray(
            [row["groups"]["zero_frequency_sector"]["mean_real"] for row in rows]
        )
        scalar_means = np.asarray(
            [row["groups"]["finite_scalar"]["mean_real"] for row in rows]
        )
        luminal_minima = np.asarray(
            [row["groups"]["tensor_vector_luminal"]["minimum_real"] for row in rows]
        )
        luminal_maxima = np.asarray(
            [row["groups"]["tensor_vector_luminal"]["maximum_real"] for row in rows]
        )
        principal_limits[name] = {
            "orientation_strength": strength,
            "zero_frequency_speed_squared": _linear_intercept(
                inverse_wave_squared,
                zero_means,
            ),
            "scalar_speed_squared": _linear_intercept(
                inverse_wave_squared,
                scalar_means,
            ),
            "luminal_minimum_speed_squared": _linear_intercept(
                inverse_wave_squared,
                luminal_minima,
            ),
            "luminal_maximum_speed_squared": _linear_intercept(
                inverse_wave_squared,
                luminal_maxima,
            ),
        }

    maximum_sign_difference = 0.0
    for negative, positive in zip(
        branch_speed_arrays["negative"],
        branch_speed_arrays["positive"],
        strict=True,
    ):
        maximum_sign_difference = max(
            maximum_sign_difference,
            float(np.max(np.abs(negative - positive))),
        )
    scalar_residual = max(
        abs(values["scalar_speed_squared"] - float(scalar_speed_squared_target))
        for values in principal_limits.values()
    )
    luminal_residual = max(
        *(
            abs(values["luminal_minimum_speed_squared"] - 1.0)
            for values in principal_limits.values()
        ),
        *(
            abs(values["luminal_maximum_speed_squared"] - 1.0)
            for values in principal_limits.values()
        ),
    )
    zero_residual = max(
        abs(values["zero_frequency_speed_squared"])
        for values in principal_limits.values()
    )
    gates = {
        "full_euler_matrices_built_before_spatial_gauge": all(
            row["gauge_applied_after_full_euler_matrices"]
            for rows in branch_rows.values()
            for row in rows
        ),
        "six_linear_degrees_from_24_finite_roots": valid_counts,
        "quadratic_polynomial_residual": maximum_polynomial_residual
        < float(polynomial_residual_tolerance),
        "squared_roots_numerically_real": maximum_speed_squared_imaginary < 1.0e-8,
        "zero_frequency_sector_reproduced": zero_residual < 1.0e-8,
        "scalar_front_reproduces_local_half": scalar_residual
        < float(scalar_limit_tolerance),
        "tensor_vector_fronts_luminal": luminal_residual
        < float(luminal_limit_tolerance),
        "finite_frequency_mode_energies_positive": all_positive_energies
        and minimum_propagating_energy > 0.0,
        "v12a_sign_absent_from_flat_quadratic_spectrum": maximum_sign_difference
        < 1.0e-12,
    }
    return {
        "candidate": "Sigma v12A same-AeST-clock luminal DHOST geometry",
        "calculation": {
            "euler_polynomial": "P(s)=s^2 K+s(A-A^T)-B",
            "linearized_pencil": "[[0,I],[B,-(A-A^T)]] u=s [[I,0],[0,K]] u",
            "spatial_gauge": "h13=h23=h33=0 per real phase after full EOM construction",
            "real_phase_multiplicity": (
                "four finite roots per physical configuration degree: +/- time roots "
                "times sine/cosine phases"
            ),
            "flat_endpoint_regulator": (
                "positive aether-tilt sentinel because the sqrt(Y) representation is not "
                "twice differentiable at exact Y=0"
            ),
        },
        "fixed_values": {
            "k_b": float(k_b),
            "k_2": float(k_2),
            "background_clock_ratio": float(background_clock_ratio),
            "aligned_tilt_sentinel": float(aligned_tilt_sentinel),
            "orientation_strengths": list(strengths),
            "wave_number_sentinels": list(wave_numbers),
            "scalar_speed_squared_target": float(scalar_speed_squared_target),
        },
        "principal_limits": principal_limits,
        "diagnostics": {
            "maximum_polynomial_residual": maximum_polynomial_residual,
            "maximum_speed_squared_imaginary": maximum_speed_squared_imaginary,
            "minimum_normalized_propagating_mode_energy": minimum_propagating_energy,
            "maximum_positive_negative_sign_spectrum_difference": maximum_sign_difference,
            "maximum_scalar_limit_residual": scalar_residual,
            "maximum_luminal_limit_residual": luminal_residual,
            "maximum_zero_frequency_limit_residual": zero_residual,
        },
        "rows": branch_rows,
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_flat_characteristic_gates_pass": bool(all(gates.values())),
        "flat_linear_six_degree_count_reproduced": bool(valid_counts),
        "flat_finite_frequency_energy_positive": bool(all_positive_energies),
        "zero_frequency_jeans_sector_resolved": False,
        "tilted_background_characteristics_proven": False,
        "nonconstant_background_characteristics_proven": False,
        "nonlinear_physical_degree_count_proven": False,
        "theory_viable": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
