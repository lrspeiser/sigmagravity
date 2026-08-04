"""Constraint-solved modal energy gate for the Sigma v12A quadratic action.

For the singular quadratic Lagrangian

``L = qdot.K.qdot/2 + qdot.A.q + q.B.q/2``

the finite generalized eigenvectors solve the lapse, shift, aether, and DHOST
algebraic equations after spatial gauge fixing.  On an oscillatory root
``s=i omega``, the time-averaged canonical energy is

``E = u^dagger (omega^2 K-B) u / 4``.

It is independently equal to the Krein derivative

``E = omega u^dagger (2 omega K-i C) u / 4``, ``C=A-A.T``.

Agreement of the two expressions verifies that the reported sign is the
finite physical-mode sign, not an unevaluated coordinate constraint.
"""

from __future__ import annotations

import math

import numpy as np

from voidscreen.sigma_v12a_general_covector import (
    GeneralCovectorBackground,
    boosted_unitary_background,
    general_covector_characteristic_eigensystem,
    rotate_background_to_wave_axis,
    solve_tilted_constant_branch_scalar_clock,
)


def _wave_direction(angle_degrees: float) -> tuple[float, float, float]:
    angle = math.radians(float(angle_degrees))
    return (math.sin(angle), 0.0, math.cos(angle))


def constrained_modal_energy_spectrum(
    background: GeneralCovectorBackground,
    *,
    wave_number_ratio: float,
    oscillatory_growth_fraction_tolerance: float = 1.0e-4,
    minimum_frequency_fraction: float = 1.0e-8,
    relative_energy_tolerance: float = 1.0e-8,
) -> dict[str, object]:
    """Return canonical/Krein energy signs of positive-frequency finite modes."""

    wave_number = float(wave_number_ratio)
    growth_tolerance = float(oscillatory_growth_fraction_tolerance)
    frequency_threshold = float(minimum_frequency_fraction)
    energy_tolerance = float(relative_energy_tolerance)
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (
            wave_number,
            growth_tolerance,
            frequency_threshold,
            energy_tolerance,
        )
    ):
        raise ValueError("energy-spectrum wave number and tolerances must be positive")

    eigensystem = general_covector_characteristic_eigensystem(
        background,
        wave_number_ratio=wave_number,
    )
    kinetic = np.asarray(eigensystem["K"], dtype=float)
    gyroscopic = np.asarray(eigensystem["C"], dtype=float)
    potential = np.asarray(eigensystem["B"], dtype=float)
    roots = np.asarray(eigensystem["finite_roots"], dtype=complex)
    eigenvectors = np.asarray(eigensystem["finite_eigenvectors"], dtype=complex)
    modes: list[dict[str, float | str]] = []
    maximum_growth = 0.0
    maximum_polynomial_residual = 0.0
    matrix_norms = (
        float(np.linalg.norm(kinetic)),
        float(np.linalg.norm(gyroscopic)),
        float(np.linalg.norm(potential)),
    )
    for root, state in zip(roots, eigenvectors.T, strict=True):
        growth_fraction = abs(float(root.real)) / wave_number
        frequency_fraction = float(root.imag) / wave_number
        maximum_growth = max(maximum_growth, growth_fraction)
        if growth_fraction > growth_tolerance or frequency_fraction <= frequency_threshold:
            continue
        amplitude = state[: kinetic.shape[0]]
        amplitude_norm = float(np.linalg.norm(amplitude))
        if amplitude_norm <= 0.0:
            continue
        amplitude = amplitude / amplitude_norm
        omega = float(root.imag)
        kinetic_energy = 0.25 * omega**2 * float(
            np.real(np.vdot(amplitude, kinetic @ amplitude))
        )
        potential_energy = -0.25 * float(
            np.real(np.vdot(amplitude, potential @ amplitude))
        )
        canonical_energy = kinetic_energy + potential_energy
        krein_energy = 0.25 * omega * float(
            np.real(
                np.vdot(
                    amplitude,
                    (2.0 * omega * kinetic - 1.0j * gyroscopic) @ amplitude,
                )
            )
        )
        energy_scale = max(
            np.finfo(float).tiny,
            abs(kinetic_energy) + abs(potential_energy),
        )
        normalized_energy = canonical_energy / energy_scale
        identity_residual = abs(canonical_energy - krein_energy) / energy_scale
        polynomial = root**2 * kinetic + root * gyroscopic - potential
        polynomial_scale = max(
            1.0,
            abs(root) ** 2 * matrix_norms[0]
            + abs(root) * matrix_norms[1]
            + matrix_norms[2],
        )
        polynomial_residual = float(
            np.linalg.norm(polynomial @ amplitude) / polynomial_scale
        )
        maximum_polynomial_residual = max(
            maximum_polynomial_residual,
            polynomial_residual,
        )
        if normalized_energy < -energy_tolerance:
            sign = "negative"
        elif normalized_energy > energy_tolerance:
            sign = "positive"
        else:
            sign = "zero"
        modes.append(
            {
                "frequency_over_wave_number": frequency_fraction,
                "growth_over_wave_number": growth_fraction,
                "kinetic_energy": kinetic_energy,
                "potential_energy": potential_energy,
                "canonical_energy": canonical_energy,
                "krein_energy": krein_energy,
                "normalized_energy": normalized_energy,
                "canonical_krein_identity_residual": identity_residual,
                "polynomial_residual": polynomial_residual,
                "energy_sign": sign,
            }
        )

    negative_modes = [mode for mode in modes if mode["energy_sign"] == "negative"]
    zero_modes = [mode for mode in modes if mode["energy_sign"] == "zero"]
    minimum_mode = (
        min(modes, key=lambda mode: float(mode["normalized_energy"]))
        if modes
        else None
    )
    maximum_identity_residual = max(
        (
            float(mode["canonical_krein_identity_residual"])
            for mode in modes
        ),
        default=0.0,
    )
    return {
        "wave_number_ratio": wave_number,
        "finite_generalized_root_count": int(
            eigensystem["finite_generalized_root_count"]
        ),
        "infinite_generalized_root_count": int(
            eigensystem["infinite_generalized_root_count"]
        ),
        "positive_frequency_oscillatory_mode_count": len(modes),
        "negative_energy_mode_count": len(negative_modes),
        "zero_energy_mode_count": len(zero_modes),
        "minimum_energy_mode": minimum_mode,
        "minimum_normalized_energy": (
            float(minimum_mode["normalized_energy"])
            if minimum_mode is not None
            else None
        ),
        "maximum_normalized_growth": maximum_growth,
        "maximum_canonical_krein_identity_residual": maximum_identity_residual,
        "maximum_polynomial_residual": maximum_polynomial_residual,
        "finite_descriptor_root_structure_preserved": bool(
            eigensystem["finite_generalized_root_count"] == 24
            and eigensystem["infinite_generalized_root_count"] == 16
        ),
        "all_identified_finite_mode_energies_positive": bool(
            modes and not negative_modes and not zero_modes
        ),
        "modes": sorted(
            modes,
            key=lambda mode: float(mode["frequency_over_wave_number"]),
        ),
    }


def common_time_energy_row(
    *,
    k_b: float,
    k_2: float,
    orientation_strength: float,
    tilt_magnitude: float,
    boost_velocity: float,
    wave_angles_degrees: tuple[float, ...],
    wave_number_ratio: float,
    principal_growth_threshold: float,
    minimum_frequency_fraction: float,
    relative_energy_tolerance: float,
) -> dict[str, object]:
    """Evaluate one time direction against every supplied wave direction."""

    scalar_clock = solve_tilted_constant_branch_scalar_clock(
        tilt_magnitude=float(tilt_magnitude),
        k_b=float(k_b),
        k_2=float(k_2),
    )
    boosted = boosted_unitary_background(
        scalar_clock_ratio=scalar_clock,
        aether_spatial_covector=(0.0, 0.0, float(tilt_magnitude)),
        boost_velocity=(0.0, 0.0, float(boost_velocity)),
        orientation_strength=float(orientation_strength),
        k_b=float(k_b),
        k_2=float(k_2),
    )
    rows = [
        {
            "wave_angle_degrees": float(angle),
            **constrained_modal_energy_spectrum(
                rotate_background_to_wave_axis(
                    boosted,
                    _wave_direction(float(angle)),
                ),
                wave_number_ratio=float(wave_number_ratio),
                oscillatory_growth_fraction_tolerance=float(
                    principal_growth_threshold
                ),
                minimum_frequency_fraction=float(minimum_frequency_fraction),
                relative_energy_tolerance=float(relative_energy_tolerance),
            ),
        }
        for angle in wave_angles_degrees
    ]
    minimum_rows = [
        row for row in rows if row["minimum_normalized_energy"] is not None
    ]
    worst = min(
        minimum_rows,
        key=lambda row: float(row["minimum_normalized_energy"]),
    )
    root_structure = all(
        bool(row["finite_descriptor_root_structure_preserved"])
        for row in rows
    )
    maximum_growth = max(float(row["maximum_normalized_growth"]) for row in rows)
    return {
        "k_b": float(k_b),
        "k_2": float(k_2),
        "orientation_strength": float(orientation_strength),
        "tilt_magnitude": float(tilt_magnitude),
        "on_shell_scalar_clock": scalar_clock,
        "boost_velocity_parallel_to_aether": float(boost_velocity),
        "aether_rest_frame_velocity": float(
            float(tilt_magnitude) / np.sqrt(1.0 + float(tilt_magnitude) ** 2)
        ),
        "root_structure_preserved_all_directions": root_structure,
        "maximum_normalized_growth": maximum_growth,
        "minimum_normalized_energy_all_directions": float(
            worst["minimum_normalized_energy"]
        ),
        "minimum_canonical_energy_all_directions": float(
            worst["minimum_energy_mode"]["canonical_energy"]
        ),
        "worst_wave_angle_degrees": float(worst["wave_angle_degrees"]),
        "negative_energy_mode_count_all_directions": sum(
            int(row["negative_energy_mode_count"]) for row in rows
        ),
        "common_time_kinematically_valid": bool(
            root_structure
            and maximum_growth <= float(principal_growth_threshold)
        ),
        "all_directions_positive_energy": bool(
            all(row["all_identified_finite_mode_energies_positive"] for row in rows)
        ),
        "maximum_canonical_krein_identity_residual": max(
            float(row["maximum_canonical_krein_identity_residual"])
            for row in rows
        ),
        "maximum_polynomial_residual": max(
            float(row["maximum_polynomial_residual"]) for row in rows
        ),
        "rows": rows,
    }


def scan_common_time_energy(
    *,
    k_b: float,
    k_2: float,
    orientation_strength: float,
    tilt_magnitude: float,
    boost_velocities: tuple[float, ...],
    wave_angles_degrees: tuple[float, ...],
    wave_number_ratio: float,
    principal_growth_threshold: float,
    minimum_frequency_fraction: float,
    relative_energy_tolerance: float,
) -> dict[str, object]:
    """Maximize the worst modal energy over a fixed common-time scan."""

    candidates = [
        common_time_energy_row(
            k_b=float(k_b),
            k_2=float(k_2),
            orientation_strength=float(orientation_strength),
            tilt_magnitude=float(tilt_magnitude),
            boost_velocity=float(velocity),
            wave_angles_degrees=wave_angles_degrees,
            wave_number_ratio=float(wave_number_ratio),
            principal_growth_threshold=float(principal_growth_threshold),
            minimum_frequency_fraction=float(minimum_frequency_fraction),
            relative_energy_tolerance=float(relative_energy_tolerance),
        )
        for velocity in boost_velocities
    ]
    valid = [row for row in candidates if row["common_time_kinematically_valid"]]
    best = (
        max(
            valid,
            key=lambda row: float(
                row["minimum_normalized_energy_all_directions"]
            ),
        )
        if valid
        else None
    )
    return {
        "candidate_time_count": len(candidates),
        "kinematically_valid_time_count": len(valid),
        "best_maximin_time": best,
        "any_common_time_all_directions_positive_energy": any(
            bool(row["all_directions_positive_energy"]) for row in valid
        ),
        "candidates": candidates,
    }
