from __future__ import annotations

import math

import pytest

from voidscreen.sigma_v12a_general_covector import (
    GeneralCovectorBackground,
    boosted_unitary_background,
    rotate_background_to_wave_axis,
    solve_tilted_constant_branch_scalar_clock,
)
from voidscreen.sigma_v12a_reduced_energy import (
    constrained_modal_energy_spectrum,
    scan_common_time_energy,
)


def test_flat_finite_frequency_modes_have_positive_canonical_energy() -> None:
    report = constrained_modal_energy_spectrum(
        GeneralCovectorBackground(
            scalar_covector=(1.0, 0.0, 0.0, 0.0),
            aether_spatial_covector=(1.0e-5, 0.0, 0.0),
            orientation_strength=-1.0,
            k_b=1.0,
            k_2=4.0,
        ),
        wave_number_ratio=300.0,
        minimum_frequency_fraction=0.05,
    )
    assert report["finite_descriptor_root_structure_preserved"]
    assert report["positive_frequency_oscillatory_mode_count"] == 10
    assert report["negative_energy_mode_count"] == 0
    assert report["all_identified_finite_mode_energies_positive"]
    assert report["maximum_canonical_krein_identity_residual"] < 1.0e-9


def _rest_frame_background(*, orientation_strength: float) -> GeneralCovectorBackground:
    tilt = 0.5
    scalar_clock = solve_tilted_constant_branch_scalar_clock(
        tilt_magnitude=tilt,
        k_b=1.0,
        k_2=4.0,
    )
    background = boosted_unitary_background(
        scalar_clock_ratio=scalar_clock,
        aether_spatial_covector=(0.0, 0.0, tilt),
        boost_velocity=(0.0, 0.0, tilt / math.sqrt(1.0 + tilt**2)),
        orientation_strength=orientation_strength,
        k_b=1.0,
        k_2=4.0,
    )
    return rotate_background_to_wave_axis(background, (0.0, 0.0, 1.0))


def test_selected_v12a_row_has_constraint_solved_negative_energy_mode() -> None:
    report = constrained_modal_energy_spectrum(
        _rest_frame_background(orientation_strength=-1.0),
        wave_number_ratio=300.0,
    )
    mode = report["minimum_energy_mode"]
    assert report["positive_frequency_oscillatory_mode_count"] == 12
    assert report["negative_energy_mode_count"] == 2
    assert mode["kinetic_energy"] > 0.0
    assert mode["potential_energy"] < 0.0
    assert mode["canonical_energy"] < -0.09
    assert mode["normalized_energy"] < -0.6
    assert report["maximum_canonical_krein_identity_residual"] < 1.0e-9
    assert report["maximum_polynomial_residual"] < 1.0e-9


def test_negative_mode_is_inherited_when_dhost_coupling_is_zero() -> None:
    report = constrained_modal_energy_spectrum(
        _rest_frame_background(orientation_strength=0.0),
        wave_number_ratio=300.0,
    )
    assert report["negative_energy_mode_count"] == 2
    assert report["minimum_normalized_energy"] < -0.3


def test_no_scanned_common_time_makes_all_directions_positive() -> None:
    report = scan_common_time_energy(
        k_b=1.0,
        k_2=4.0,
        orientation_strength=-1.0,
        tilt_magnitude=0.5,
        boost_velocities=(0.445, 0.446, 0.45),
        wave_angles_degrees=(0.0, 90.0, 180.0),
        wave_number_ratio=300.0,
        principal_growth_threshold=0.01,
        minimum_frequency_fraction=1.0e-8,
        relative_energy_tolerance=1.0e-8,
    )
    assert report["kinematically_valid_time_count"] == 3
    assert not report["any_common_time_all_directions_positive_energy"]
    assert (
        report["best_maximin_time"][
            "minimum_normalized_energy_all_directions"
        ]
        < -0.8
    )


def test_low_frequency_threshold_does_not_hide_endpoint_negative_mode() -> None:
    tilt = 0.5
    scalar_clock = solve_tilted_constant_branch_scalar_clock(
        tilt_magnitude=tilt,
        k_b=1.99,
        k_2=2.0,
    )
    background = boosted_unitary_background(
        scalar_clock_ratio=scalar_clock,
        aether_spatial_covector=(0.0, 0.0, tilt),
        boost_velocity=(0.0, 0.0, tilt / math.sqrt(1.0 + tilt**2)),
        orientation_strength=0.0,
        k_b=1.99,
        k_2=2.0,
    )
    report = constrained_modal_energy_spectrum(
        rotate_background_to_wave_axis(background, (1.0, 0.0, 0.0)),
        wave_number_ratio=300.0,
        minimum_frequency_fraction=1.0e-8,
    )
    assert report["negative_energy_mode_count"] == 2
    assert report["minimum_energy_mode"]["frequency_over_wave_number"] < 1.0e-5
    assert report["minimum_energy_mode"]["canonical_energy"] < 0.0


def test_invalid_modal_energy_protocol_is_rejected() -> None:
    with pytest.raises(ValueError):
        constrained_modal_energy_spectrum(
            _rest_frame_background(orientation_strength=-1.0),
            wave_number_ratio=0.0,
        )
