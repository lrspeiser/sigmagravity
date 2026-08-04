from __future__ import annotations

from voidscreen.sigma_v12a_tilted_characteristics import (
    audit_v12a_tilted_characteristics,
    unitary_slicing_characteristic_row,
)
from voidscreen.sigma_v12a_tilted_principal import TiltedPrincipalBackground


def test_small_tilt_remains_oscillatory_in_unitary_time() -> None:
    row = unitary_slicing_characteristic_row(
        TiltedPrincipalBackground(1.0, 0.0, 0.5),
        wave_number_ratio=300.0,
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
    )
    assert row["finite_generalized_root_count"] == 24
    assert row["infinite_generalized_root_count"] == 16
    assert row["scalar_unitary_time_hyperbolic_at_declared_threshold"]
    assert row["frequencies_inside_metric_cone_at_declared_tolerance"]
    assert row["identified_oscillatory_energies_nonnegative"]


def test_large_perpendicular_tilt_has_persistent_principal_growth() -> None:
    row = unitary_slicing_characteristic_row(
        TiltedPrincipalBackground(1.0, 0.0, 2.0),
        wave_number_ratio=1000.0,
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
    )
    assert row["finite_generalized_root_count"] == 24
    assert row["maximum_normalized_exponential_growth"] > 0.45
    assert row["principal_exponential_root_count"] == 4
    assert not row["scalar_unitary_time_hyperbolic_at_declared_threshold"]


def test_off_clock_and_parallel_rows_expose_distinct_failures() -> None:
    superluminal = unitary_slicing_characteristic_row(
        TiltedPrincipalBackground(0.5, 0.0, 0.5),
        wave_number_ratio=300.0,
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
    )
    negative_energy = unitary_slicing_characteristic_row(
        TiltedPrincipalBackground(2.0, 0.5, 0.0, orientation_strength=-1.0),
        wave_number_ratio=300.0,
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
    )
    assert superluminal["maximum_absolute_frequency_over_metric_light"] > 1.19
    assert not superluminal["frequencies_inside_metric_cone_at_declared_tolerance"]
    assert negative_energy["minimum_oscillatory_mode_energy"] < -3.0
    assert not negative_energy["identified_oscillatory_energies_nonnegative"]


def test_small_tilted_audit_reports_slicing_failure_without_covariant_overclaim() -> None:
    report = audit_v12a_tilted_characteristics(
        k_b=1.0,
        k_2=2.0,
        background_clock_ratio=1.0,
        orientation_strengths=(-1.0, 1.0),
        scalar_clock_ratios=(0.5, 1.0, 2.0),
        tilt_magnitudes=(0.5, 2.0),
        relative_angles_degrees=(0.0, 90.0),
        grid_wave_number=300.0,
        convergence_wave_numbers=(100.0, 300.0, 1000.0),
        convergence_sentinels=(
            {
                "name": "growth",
                "scalar_clock_ratio": 1.0,
                "tilt_magnitude": 2.0,
                "relative_angle_degrees": 90.0,
                "orientation_strength": 1.0,
            },
        ),
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
        polynomial_residual_tolerance=1.0e-7,
    )
    assert report["gates"]["finite_constraint_root_structure_preserved"]
    assert report["gates"]["quadratic_polynomial_residuals_controlled"]
    assert not report["all_tilted_unitary_slicing_gates_pass"]
    assert report["scalar_unitary_slicing_fails_some_finite_backgrounds"]
    assert not report["invariant_common_cauchy_covector_proven_absent"]
    assert not report["covariant_theory_falsified_by_this_subgate"]
    assert not report["theory_viable"]
    assert not report["observational_data_accessed"]
    assert not report["raw_holdout_opened"]
