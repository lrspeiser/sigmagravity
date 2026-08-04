"""On-shell constant-background cone screen for Sigma v12A.

The arbitrary-covector machinery can diagnose off-shell field configurations,
but a theory decision should first impose the background field equations.  For
constant gradients the shift-symmetric scalar equation is automatic.  A
misaligned aether obeys ``dL/dQ=0``.  This module solves that branch, screens
the already-present ``K_B,K_2`` constants, and requires one time direction per
background that is common to every sampled wave direction.

This remains a constant-background principal-symbol screen.  It does not
replace the reduced Hamiltonian on a complete curved solution.
"""

from __future__ import annotations

import math

import numpy as np

from voidscreen.sigma_v12a_general_covector import (
    GeneralCovectorBackground,
    boosted_unitary_background,
    general_covector_background_invariants,
    general_covector_characteristic_row,
    rotate_background_to_wave_axis,
    solve_tilted_constant_branch_scalar_clock,
)


def _wave_direction(angle_degrees: float) -> tuple[float, float, float]:
    angle = math.radians(float(angle_degrees))
    return (math.sin(angle), 0.0, math.cos(angle))


def _unitary_background(
    *,
    scalar_clock: float,
    tilt_magnitude: float,
    angle_degrees: float,
    orientation_strength: float,
    k_b: float,
    k_2: float,
) -> GeneralCovectorBackground:
    angle = math.radians(float(angle_degrees))
    tilt = float(tilt_magnitude)
    return GeneralCovectorBackground(
        scalar_covector=(float(scalar_clock), 0.0, 0.0, 0.0),
        aether_spatial_covector=(
            tilt * math.sin(angle),
            0.0,
            tilt * math.cos(angle),
        ),
        orientation_strength=float(orientation_strength),
        k_b=float(k_b),
        k_2=float(k_2),
    ).validated()


def screen_on_shell_parameter_pair(
    *,
    k_b: float,
    k_2: float,
    orientation_strength: float,
    tilt_magnitudes: tuple[float, ...],
    relative_angles_degrees: tuple[float, ...],
    wave_number_ratio: float,
    principal_growth_threshold: float,
    metric_cone_frequency_tolerance: float,
    polynomial_residual_tolerance: float,
    aether_eom_tolerance: float,
) -> dict[str, object]:
    """Screen one existing AeST parameter pair on its constant tilted branch."""

    scalar_speed_squared = (2.0 - float(k_b)) / (float(k_2) * float(k_b))
    rows: list[dict[str, object]] = []
    maximum_growth = 0.0
    maximum_frequency = 0.0
    maximum_residual = 0.0
    maximum_aether_residual = 0.0
    for tilt in tilt_magnitudes:
        scalar_clock = solve_tilted_constant_branch_scalar_clock(
            tilt_magnitude=float(tilt),
            k_b=float(k_b),
            k_2=float(k_2),
        )
        for angle in relative_angles_degrees:
            background = _unitary_background(
                scalar_clock=scalar_clock,
                tilt_magnitude=float(tilt),
                angle_degrees=float(angle),
                orientation_strength=float(orientation_strength),
                k_b=float(k_b),
                k_2=float(k_2),
            )
            row = general_covector_characteristic_row(
                background,
                wave_number_ratio=float(wave_number_ratio),
                principal_growth_threshold=float(principal_growth_threshold),
                metric_cone_frequency_tolerance=float(
                    metric_cone_frequency_tolerance
                ),
            )
            row["tilt_magnitude"] = float(tilt)
            row["relative_angle_degrees"] = float(angle)
            rows.append(row)
            maximum_growth = max(
                maximum_growth,
                float(row["maximum_normalized_exponential_growth"]),
            )
            maximum_frequency = max(
                maximum_frequency,
                float(row["maximum_absolute_frequency_over_metric_light"]),
            )
            maximum_residual = max(
                maximum_residual,
                float(row["maximum_polynomial_residual"]),
            )
            maximum_aether_residual = max(
                maximum_aether_residual,
                float(
                    row["background_invariants"][
                        "projected_aether_eom_residual"
                    ]
                ),
            )

    root_structure = all(
        row["finite_generalized_root_count"] == 24
        and row["infinite_generalized_root_count"] == 16
        for row in rows
    )
    no_metric_timelike_roots = all(
        row["metric_timelike_oscillatory_characteristic_root_count"] == 0
        for row in rows
    )
    gates = {
        "flat_scalar_positive_and_subluminal": 0.0 < scalar_speed_squared <= 1.0,
        "constant_aether_equation_satisfied": maximum_aether_residual
        <= float(aether_eom_tolerance),
        "finite_constraint_root_structure_preserved": root_structure,
        "quadratic_polynomial_residuals_controlled": maximum_residual
        <= float(polynomial_residual_tolerance),
        "scalar_unitary_time_hyperbolic_on_screen": maximum_growth
        <= float(principal_growth_threshold),
        "no_metric_timelike_oscillatory_covectors_at_finite_k_tolerance": (
            no_metric_timelike_roots
        ),
    }
    return {
        "k_b": float(k_b),
        "k_2": float(k_2),
        "orientation_strength": float(orientation_strength),
        "flat_scalar_speed_squared": scalar_speed_squared,
        "row_count": len(rows),
        "maximum_normalized_exponential_growth": maximum_growth,
        "maximum_absolute_frequency_over_metric_light": maximum_frequency,
        "maximum_polynomial_residual": maximum_residual,
        "maximum_projected_aether_eom_residual": maximum_aether_residual,
        "gates": gates,
        "all_parameter_screen_gates_pass": bool(all(gates.values())),
        "rows": rows,
    }


def scan_common_time_on_shell_branch(
    *,
    k_b: float,
    k_2: float,
    orientation_strength: float,
    tilt_magnitude: float,
    boost_velocities: tuple[float, ...],
    wave_angles_degrees: tuple[float, ...],
    wave_number_ratio: float,
    principal_growth_threshold: float,
    metric_cone_frequency_tolerance: float,
    polynomial_residual_tolerance: float,
) -> dict[str, object]:
    """Find one boosted time direction common to all sampled wave directions."""

    scalar_clock = solve_tilted_constant_branch_scalar_clock(
        tilt_magnitude=float(tilt_magnitude),
        k_b=float(k_b),
        k_2=float(k_2),
    )
    candidates: list[dict[str, object]] = []
    for velocity in boost_velocities:
        boosted = boosted_unitary_background(
            scalar_clock_ratio=scalar_clock,
            aether_spatial_covector=(0.0, 0.0, float(tilt_magnitude)),
            boost_velocity=(0.0, 0.0, float(velocity)),
            orientation_strength=float(orientation_strength),
            k_b=float(k_b),
            k_2=float(k_2),
        )
        rows = [
            general_covector_characteristic_row(
                rotate_background_to_wave_axis(
                    boosted,
                    _wave_direction(angle),
                ),
                wave_number_ratio=float(wave_number_ratio),
                principal_growth_threshold=float(principal_growth_threshold),
                metric_cone_frequency_tolerance=float(
                    metric_cone_frequency_tolerance
                ),
            )
            for angle in wave_angles_degrees
        ]
        root_structure = all(
            row["finite_generalized_root_count"] == 24
            and row["infinite_generalized_root_count"] == 16
            for row in rows
        )
        maximum_growth = max(
            float(row["maximum_normalized_exponential_growth"]) for row in rows
        )
        maximum_frequency = max(
            float(row["maximum_absolute_frequency_over_metric_light"])
            for row in rows
        )
        maximum_residual = max(
            float(row["maximum_polynomial_residual"]) for row in rows
        )
        timelike_root_count = sum(
            int(row["metric_timelike_oscillatory_characteristic_root_count"])
            for row in rows
        )
        energies = [
            float(row["minimum_oscillatory_mode_energy"])
            for row in rows
            if row["minimum_oscillatory_mode_energy"] is not None
        ]
        candidate = {
            "boost_velocity_parallel_to_aether": float(velocity),
            "root_structure_preserved": root_structure,
            "maximum_normalized_exponential_growth": maximum_growth,
            "maximum_absolute_frequency_over_metric_light": maximum_frequency,
            "metric_timelike_oscillatory_characteristic_root_count": (
                timelike_root_count
            ),
            "maximum_polynomial_residual": maximum_residual,
            "minimum_coordinate_time_energy": min(energies) if energies else None,
            "common_time_at_declared_threshold": bool(
                root_structure
                and maximum_growth <= float(principal_growth_threshold)
                and timelike_root_count == 0
                and maximum_residual <= float(polynomial_residual_tolerance)
            ),
            "rows": rows,
        }
        candidates.append(candidate)

    structurally_valid = [
        row for row in candidates if bool(row["root_structure_preserved"])
    ]
    ranked = structurally_valid if structurally_valid else candidates
    best = min(
        ranked,
        key=lambda row: (
            float(row["maximum_normalized_exponential_growth"]),
            int(row["metric_timelike_oscillatory_characteristic_root_count"]),
        ),
    )
    invariants = general_covector_background_invariants(
        _unitary_background(
            scalar_clock=scalar_clock,
            tilt_magnitude=float(tilt_magnitude),
            angle_degrees=0.0,
            orientation_strength=float(orientation_strength),
            k_b=float(k_b),
            k_2=float(k_2),
        )
    )
    return {
        "k_b": float(k_b),
        "k_2": float(k_2),
        "orientation_strength": float(orientation_strength),
        "tilt_magnitude": float(tilt_magnitude),
        "on_shell_scalar_clock": scalar_clock,
        "on_shell_background_invariants": invariants,
        "candidate_time_count": len(candidates),
        "best_common_time": best,
        "common_time_found_at_declared_threshold": bool(
            best["common_time_at_declared_threshold"]
        ),
        "candidates": candidates,
    }


def principal_cone_convergence(
    *,
    k_b: float,
    k_2: float,
    orientation_strength: float,
    tilt_magnitude: float,
    boost_velocity: float,
    wave_angles_degrees: tuple[float, ...],
    wave_numbers: tuple[float, ...],
    principal_growth_threshold: float,
    metric_cone_frequency_tolerance: float,
) -> dict[str, object]:
    """Extrapolate the fastest finite-k frequency toward the principal limit."""

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
    rows: list[dict[str, float]] = []
    for wave_number in wave_numbers:
        direction_rows = [
            general_covector_characteristic_row(
                rotate_background_to_wave_axis(
                    boosted,
                    _wave_direction(angle),
                ),
                wave_number_ratio=float(wave_number),
                principal_growth_threshold=float(principal_growth_threshold),
                metric_cone_frequency_tolerance=float(
                    metric_cone_frequency_tolerance
                ),
            )
            for angle in wave_angles_degrees
        ]
        rows.append(
            {
                "wave_number_ratio": float(wave_number),
                "finite_constraint_root_structure_preserved": all(
                    row["finite_generalized_root_count"] == 24
                    and row["infinite_generalized_root_count"] == 16
                    for row in direction_rows
                ),
                "maximum_normalized_exponential_growth": max(
                    float(row["maximum_normalized_exponential_growth"])
                    for row in direction_rows
                ),
                "maximum_absolute_frequency_over_metric_light": max(
                    float(row["maximum_absolute_frequency_over_metric_light"])
                    for row in direction_rows
                ),
                "maximum_polynomial_residual": max(
                    float(row["maximum_polynomial_residual"])
                    for row in direction_rows
                ),
            }
        )
    inverse_wave = np.asarray(
        [1.0 / float(row["wave_number_ratio"]) for row in rows],
        dtype=float,
    )
    frequency_excess = np.asarray(
        [float(row["maximum_absolute_frequency_over_metric_light"]) - 1.0 for row in rows],
        dtype=float,
    )
    slope, intercept = np.polyfit(inverse_wave, frequency_excess, 1)
    return {
        "tilt_magnitude": float(tilt_magnitude),
        "boost_velocity_parallel_to_aether": float(boost_velocity),
        "orientation_strength": float(orientation_strength),
        "frequency_excess_fit_intercept": float(intercept),
        "frequency_excess_fit_inverse_wave_slope": float(slope),
        "rows": rows,
    }
