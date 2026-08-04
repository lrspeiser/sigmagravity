"""Finite-tilt characteristic diagnostics in scalar-unitary slicing.

This module deliberately makes a narrower claim than an invariant cone audit.
It asks whether the physical metric time used by the scalar-unitary ADM
calculation is itself a valid hyperbolic, positive-energy slicing throughout a
frozen finite-tilt grid.  A failure requires a later common-Cauchy-covector
calculation before the covariant theory can be retired: one bad slicing need
not imply that every metric-timelike slicing is bad.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from voidscreen.sigma_v12a_flat_characteristics import (
    generalized_characteristic_eigensystem,
)
from voidscreen.sigma_v12a_tilted_principal import TiltedPrincipalBackground


def unitary_slicing_characteristic_row(
    background: TiltedPrincipalBackground,
    *,
    wave_number_ratio: float,
    principal_growth_threshold: float,
    metric_cone_frequency_tolerance: float,
    oscillatory_growth_tolerance: float = 2.0e-3,
    finite_frequency_threshold: float = 1.0e-3,
) -> dict[str, object]:
    """Return compact root, cone, and quadratic-energy diagnostics."""

    wave_number = float(wave_number_ratio)
    growth_threshold = float(principal_growth_threshold)
    cone_tolerance = float(metric_cone_frequency_tolerance)
    oscillatory_tolerance = float(oscillatory_growth_tolerance)
    frequency_threshold = float(finite_frequency_threshold)
    if any(
        (not np.isfinite(value) or value <= 0.0)
        for value in (
            wave_number,
            growth_threshold,
            cone_tolerance,
            oscillatory_tolerance,
            frequency_threshold,
        )
    ):
        raise ValueError("characteristic thresholds and wave number must be positive")

    eigensystem = generalized_characteristic_eigensystem(
        background,
        wave_number_ratio=wave_number,
    )
    roots = np.asarray(eigensystem["finite_roots"], dtype=complex)
    eigenvectors = np.asarray(eigensystem["finite_eigenvectors"], dtype=complex)
    kinetic = np.asarray(eigensystem["K"], dtype=float)
    gyroscopic = np.asarray(eigensystem["C"], dtype=float)
    potential = np.asarray(eigensystem["B"], dtype=float)
    euler_dimension = kinetic.shape[0]
    growth_fractions = np.abs(roots.real) / wave_number
    frequency_fractions = np.abs(roots.imag) / wave_number

    energies: list[float] = []
    maximum_polynomial_residual = 0.0
    kinetic_norm = float(np.linalg.norm(kinetic))
    gyroscopic_norm = float(np.linalg.norm(gyroscopic))
    potential_norm = float(np.linalg.norm(potential))
    for root, vector, growth_fraction, frequency_fraction in zip(
        roots,
        eigenvectors.T,
        growth_fractions,
        frequency_fractions,
        strict=True,
    ):
        amplitude = vector[:euler_dimension]
        amplitude_norm = float(np.linalg.norm(amplitude))
        if amplitude_norm <= 0.0:
            maximum_polynomial_residual = np.inf
            continue
        amplitude = amplitude / amplitude_norm
        polynomial = root**2 * kinetic + root * gyroscopic - potential
        scale = max(
            1.0,
            abs(root) ** 2 * kinetic_norm
            + abs(root) * gyroscopic_norm
            + potential_norm,
        )
        maximum_polynomial_residual = max(
            maximum_polynomial_residual,
            float(np.linalg.norm(polynomial @ amplitude) / scale),
        )
        if (
            growth_fraction < oscillatory_tolerance
            and frequency_fraction > frequency_threshold
        ):
            omega = abs(float(root.imag))
            energies.append(
                0.25
                * float(
                    np.real(
                        np.vdot(
                            amplitude,
                            (omega**2 * kinetic - potential) @ amplitude,
                        )
                    )
                )
            )

    maximum_growth = float(np.max(growth_fractions))
    maximum_frequency = float(np.max(frequency_fractions))
    minimum_energy = float(min(energies)) if energies else None
    negative_energy_count = sum(value < -1.0e-8 for value in energies)
    principal_growth_count = int(
        np.count_nonzero(growth_fractions > growth_threshold)
    )
    outside_metric_cone_count = int(
        np.count_nonzero(frequency_fractions > 1.0 + cone_tolerance)
    )
    return {
        "background": asdict(background),
        "wave_number_ratio": wave_number,
        "finite_generalized_root_count": int(
            eigensystem["finite_generalized_root_count"]
        ),
        "infinite_generalized_root_count": int(
            eigensystem["infinite_generalized_root_count"]
        ),
        "minimum_finite_homogeneous_beta_margin": eigensystem[
            "minimum_finite_homogeneous_beta_margin"
        ],
        "maximum_infinite_homogeneous_beta_margin": eigensystem[
            "maximum_infinite_homogeneous_beta_margin"
        ],
        "maximum_normalized_exponential_growth": maximum_growth,
        "principal_exponential_root_count": principal_growth_count,
        "maximum_absolute_frequency_over_metric_light": maximum_frequency,
        "outside_metric_cone_root_count": outside_metric_cone_count,
        "oscillatory_finite_frequency_energy_root_count": len(energies),
        "minimum_oscillatory_mode_energy": minimum_energy,
        "negative_oscillatory_energy_root_count": negative_energy_count,
        "maximum_polynomial_residual": maximum_polynomial_residual,
        "scalar_unitary_time_hyperbolic_at_declared_threshold": principal_growth_count
        == 0,
        "frequencies_inside_metric_cone_at_declared_tolerance": outside_metric_cone_count
        == 0,
        "identified_oscillatory_energies_nonnegative": negative_energy_count == 0,
    }


def _extremum_row(
    current: dict[str, object] | None,
    candidate: dict[str, object],
    *,
    key: str,
    minimum: bool,
) -> dict[str, object]:
    value = candidate[key]
    if value is None:
        return current if current is not None else candidate
    if current is None or current[key] is None:
        return candidate
    if (minimum and float(value) < float(current[key])) or (
        not minimum and float(value) > float(current[key])
    ):
        return candidate
    return current


def audit_v12a_tilted_characteristics(
    *,
    k_b: float,
    k_2: float,
    background_clock_ratio: float,
    orientation_strengths: tuple[float, float],
    scalar_clock_ratios: tuple[float, ...],
    tilt_magnitudes: tuple[float, ...],
    relative_angles_degrees: tuple[float, ...],
    grid_wave_number: float,
    convergence_wave_numbers: tuple[float, float, float],
    convergence_sentinels: tuple[dict[str, float | str], ...],
    principal_growth_threshold: float,
    metric_cone_frequency_tolerance: float,
    polynomial_residual_tolerance: float,
) -> dict[str, object]:
    """Audit whether scalar-unitary metric time survives a frozen tilt grid."""

    strengths = tuple(float(value) for value in orientation_strengths)
    clocks = tuple(float(value) for value in scalar_clock_ratios)
    tilts = tuple(float(value) for value in tilt_magnitudes)
    angles = tuple(float(value) for value in relative_angles_degrees)
    convergence_waves = tuple(float(value) for value in convergence_wave_numbers)
    if len(strengths) != 2 or not strengths[0] < 0.0 < strengths[1]:
        raise ValueError("one negative and one positive orientation strength are required")
    if any(value == 0.0 or not np.isfinite(value) for value in clocks):
        raise ValueError("scalar clock ratios must be finite and nonzero")
    if any(value <= 0.0 or not np.isfinite(value) for value in tilts):
        raise ValueError("tilt magnitudes must be finite and positive")
    if any(not 0.0 <= value <= 90.0 for value in angles):
        raise ValueError("relative angles must lie in [0,90] degrees")
    if len(convergence_waves) != 3 or not (
        convergence_waves[0] < convergence_waves[1] < convergence_waves[2]
    ):
        raise ValueError("three increasing convergence wave numbers are required")

    rows: list[dict[str, object]] = []
    maximum_growth_row: dict[str, object] | None = None
    maximum_frequency_row: dict[str, object] | None = None
    minimum_energy_row: dict[str, object] | None = None
    root_structure_failures = 0
    growth_failures = 0
    metric_cone_failures = 0
    energy_failures = 0
    residual_failures = 0
    branch_failures = {
        "negative": {"growth": 0, "metric_cone": 0, "energy": 0},
        "positive": {"growth": 0, "metric_cone": 0, "energy": 0},
    }
    for strength in strengths:
        branch = "negative" if strength < 0.0 else "positive"
        for scalar_clock in clocks:
            for tilt in tilts:
                for angle_degrees in angles:
                    angle = np.deg2rad(angle_degrees)
                    background = TiltedPrincipalBackground(
                        scalar_clock_ratio=scalar_clock,
                        aether_parallel=tilt * float(np.cos(angle)),
                        aether_perpendicular=tilt * float(np.sin(angle)),
                        background_clock_ratio=float(background_clock_ratio),
                        orientation_strength=strength,
                        k_b=float(k_b),
                        k_2=float(k_2),
                    )
                    row = unitary_slicing_characteristic_row(
                        background,
                        wave_number_ratio=float(grid_wave_number),
                        principal_growth_threshold=float(principal_growth_threshold),
                        metric_cone_frequency_tolerance=float(
                            metric_cone_frequency_tolerance
                        ),
                    )
                    row = {
                        "branch": branch,
                        "tilt_magnitude": tilt,
                        "relative_angle_degrees": angle_degrees,
                        **row,
                    }
                    rows.append(row)
                    maximum_growth_row = _extremum_row(
                        maximum_growth_row,
                        row,
                        key="maximum_normalized_exponential_growth",
                        minimum=False,
                    )
                    maximum_frequency_row = _extremum_row(
                        maximum_frequency_row,
                        row,
                        key="maximum_absolute_frequency_over_metric_light",
                        minimum=False,
                    )
                    minimum_energy_row = _extremum_row(
                        minimum_energy_row,
                        row,
                        key="minimum_oscillatory_mode_energy",
                        minimum=True,
                    )
                    if not (
                        row["finite_generalized_root_count"] == 24
                        and row["infinite_generalized_root_count"] == 16
                    ):
                        root_structure_failures += 1
                    if not row["scalar_unitary_time_hyperbolic_at_declared_threshold"]:
                        growth_failures += 1
                        branch_failures[branch]["growth"] += 1
                    if not row["frequencies_inside_metric_cone_at_declared_tolerance"]:
                        metric_cone_failures += 1
                        branch_failures[branch]["metric_cone"] += 1
                    if not row["identified_oscillatory_energies_nonnegative"]:
                        energy_failures += 1
                        branch_failures[branch]["energy"] += 1
                    if row["maximum_polynomial_residual"] >= float(
                        polynomial_residual_tolerance
                    ):
                        residual_failures += 1

    convergence: dict[str, list[dict[str, object]]] = {}
    for sentinel in convergence_sentinels:
        name = str(sentinel["name"])
        angle = np.deg2rad(float(sentinel["relative_angle_degrees"]))
        tilt = float(sentinel["tilt_magnitude"])
        background = TiltedPrincipalBackground(
            scalar_clock_ratio=float(sentinel["scalar_clock_ratio"]),
            aether_parallel=tilt * float(np.cos(angle)),
            aether_perpendicular=tilt * float(np.sin(angle)),
            background_clock_ratio=float(background_clock_ratio),
            orientation_strength=float(sentinel["orientation_strength"]),
            k_b=float(k_b),
            k_2=float(k_2),
        )
        convergence[name] = [
            unitary_slicing_characteristic_row(
                background,
                wave_number_ratio=wave_number,
                principal_growth_threshold=float(principal_growth_threshold),
                metric_cone_frequency_tolerance=float(metric_cone_frequency_tolerance),
            )
            for wave_number in convergence_waves
        ]

    gates = {
        "finite_constraint_root_structure_preserved": root_structure_failures == 0,
        "quadratic_polynomial_residuals_controlled": residual_failures == 0,
        "scalar_unitary_metric_time_hyperbolic_across_grid": growth_failures == 0,
        "all_unitary_time_frequencies_inside_metric_cone": metric_cone_failures == 0,
        "all_identified_oscillatory_energies_nonnegative": energy_failures == 0,
    }
    return {
        "candidate": "Sigma v12A same-AeST-clock luminal DHOST geometry",
        "calculation": {
            "scope": "finite constant backgrounds in scalar-unitary metric-time slicing",
            "warning": (
                "failure of this slicing does not prove absence of another common "
                "metric-timelike Cauchy covector"
            ),
            "root_filter": (
                "homogeneous generalized eigenpairs keep the Class-Ia constraint roots "
                "at infinity before alpha/beta division"
            ),
        },
        "grid": {
            "orientation_strengths": list(strengths),
            "scalar_clock_ratios": list(clocks),
            "tilt_magnitudes": list(tilts),
            "relative_angles_degrees": list(angles),
            "wave_number_ratio": float(grid_wave_number),
            "total_backgrounds": len(rows),
        },
        "thresholds": {
            "principal_growth_fraction": float(principal_growth_threshold),
            "metric_cone_frequency_excess": float(metric_cone_frequency_tolerance),
            "polynomial_residual": float(polynomial_residual_tolerance),
        },
        "failure_counts": {
            "root_structure": root_structure_failures,
            "principal_growth": growth_failures,
            "metric_cone_frequency": metric_cone_failures,
            "negative_oscillatory_energy": energy_failures,
            "polynomial_residual": residual_failures,
            "by_branch": branch_failures,
        },
        "extrema": {
            "maximum_growth_row": maximum_growth_row,
            "maximum_frequency_row": maximum_frequency_row,
            "minimum_energy_row": minimum_energy_row,
        },
        "convergence_sentinels": convergence,
        "rows": rows,
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_tilted_unitary_slicing_gates_pass": bool(all(gates.values())),
        "scalar_unitary_slicing_fails_some_finite_backgrounds": bool(
            growth_failures > 0
        ),
        "invariant_common_cauchy_covector_proven_absent": False,
        "covariant_theory_falsified_by_this_subgate": False,
        "nonconstant_background_characteristics_proven": False,
        "theory_viable": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
