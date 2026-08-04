from __future__ import annotations

from voidscreen.sigma_v12a_on_shell_cones import (
    principal_cone_convergence,
    scan_common_time_on_shell_branch,
    screen_on_shell_parameter_pair,
)


def test_k2_four_passes_small_tilt_on_shell_screen() -> None:
    report = screen_on_shell_parameter_pair(
        k_b=1.0,
        k_2=4.0,
        orientation_strength=-1.0,
        tilt_magnitudes=(0.1, 0.5),
        relative_angles_degrees=(0.0, 90.0),
        wave_number_ratio=300.0,
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
        polynomial_residual_tolerance=1.0e-7,
        aether_eom_tolerance=1.0e-10,
    )
    assert report["flat_scalar_speed_squared"] == 0.25
    assert report["row_count"] == 4
    assert report["all_parameter_screen_gates_pass"]


def test_one_time_direction_is_used_for_every_wave_direction() -> None:
    report = scan_common_time_on_shell_branch(
        k_b=1.0,
        k_2=4.0,
        orientation_strength=-1.0,
        tilt_magnitude=2.0,
        boost_velocities=(-0.925, 0.0),
        wave_angles_degrees=(0.0, 45.0, 90.0, 135.0, 180.0),
        wave_number_ratio=300.0,
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
        polynomial_residual_tolerance=1.0e-7,
    )
    best = report["best_common_time"]
    assert report["common_time_found_at_declared_threshold"]
    assert best["boost_velocity_parallel_to_aether"] == -0.925
    assert len(best["rows"]) == 5
    assert best["maximum_normalized_exponential_growth"] < 0.001


def test_k2_four_frequency_excess_extrapolates_to_metric_cone() -> None:
    report = principal_cone_convergence(
        k_b=1.0,
        k_2=4.0,
        orientation_strength=-1.0,
        tilt_magnitude=0.5,
        boost_velocity=0.0,
        wave_angles_degrees=(90.0,),
        wave_numbers=(100.0, 300.0, 1000.0),
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
    )
    frequencies = [
        row["maximum_absolute_frequency_over_metric_light"]
        for row in report["rows"]
    ]
    assert frequencies[0] > frequencies[1] > frequencies[2]
    assert abs(report["frequency_excess_fit_intercept"]) < 1.0e-5


def test_original_k2_two_has_persistent_on_shell_cone_excess() -> None:
    report = principal_cone_convergence(
        k_b=1.0,
        k_2=2.0,
        orientation_strength=-1.0,
        tilt_magnitude=0.5,
        boost_velocity=0.0,
        wave_angles_degrees=(90.0,),
        wave_numbers=(300.0, 1000.0, 3000.0),
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
    )
    assert report["frequency_excess_fit_intercept"] > 0.002
