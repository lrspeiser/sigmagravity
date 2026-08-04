from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v12a_general_covector import (
    GeneralCovectorBackground,
    boosted_unitary_background,
    general_covector_background_invariants,
    general_covector_characteristic_row,
    lorentz_invariant_background_residuals,
    solve_tilted_constant_branch_scalar_clock,
    unitary_hessian_parity,
)
from voidscreen.sigma_v12a_tilted_characteristics import (
    unitary_slicing_characteristic_row,
)
from voidscreen.sigma_v12a_tilted_principal import TiltedPrincipalBackground


def test_manifest_density_reproduces_unitary_adm_hessian() -> None:
    residuals = unitary_hessian_parity(
        scalar_clock_ratio=0.7,
        aether_parallel=0.3,
        aether_perpendicular=0.2,
        orientation_strength=-1.0,
        wave_number_ratio=30.0,
        k_b=1.0,
        k_2=4.0,
    )
    assert max(residuals.values()) < 1.0e-12


def test_boost_preserves_scalar_aether_background_invariants() -> None:
    scalar_clock = solve_tilted_constant_branch_scalar_clock(
        tilt_magnitude=0.5,
        k_b=1.0,
        k_2=4.0,
    )
    reference = GeneralCovectorBackground(
        scalar_covector=(scalar_clock, 0.0, 0.0, 0.0),
        aether_spatial_covector=(0.0, 0.0, 0.5),
        k_b=1.0,
        k_2=4.0,
    )
    transformed = boosted_unitary_background(
        scalar_clock_ratio=scalar_clock,
        aether_spatial_covector=(0.0, 0.0, 0.5),
        boost_velocity=(0.1, -0.05, 0.2),
        k_b=1.0,
        k_2=4.0,
    )
    residuals = lorentz_invariant_background_residuals(reference, transformed)
    assert max(residuals.values()) < 1.0e-12


def test_tilted_branch_solves_projected_constant_aether_equation() -> None:
    scalar_clock = solve_tilted_constant_branch_scalar_clock(
        tilt_magnitude=0.5,
        k_b=1.0,
        k_2=4.0,
    )
    background = GeneralCovectorBackground(
        scalar_covector=(scalar_clock, 0.0, 0.0, 0.0),
        aether_spatial_covector=(0.0, 0.0, 0.5),
        k_b=1.0,
        k_2=4.0,
    )
    invariants = general_covector_background_invariants(background)
    assert scalar_clock == pytest.approx(1.075970051032027, abs=1.0e-12)
    assert invariants["projected_scalar_norm_y"] > 0.0
    assert invariants["projected_aether_eom_residual"] < 1.0e-12


def test_general_covector_pencil_matches_no_boost_unitary_pencil() -> None:
    scalar_clock = solve_tilted_constant_branch_scalar_clock(
        tilt_magnitude=0.5,
        k_b=1.0,
        k_2=4.0,
    )
    general = general_covector_characteristic_row(
        GeneralCovectorBackground(
            scalar_covector=(scalar_clock, 0.0, 0.0, 0.0),
            aether_spatial_covector=(0.5, 0.0, 0.0),
            orientation_strength=-1.0,
            k_b=1.0,
            k_2=4.0,
        ),
        wave_number_ratio=300.0,
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
    )
    unitary = unitary_slicing_characteristic_row(
        TiltedPrincipalBackground(
            scalar_clock,
            0.0,
            0.5,
            orientation_strength=-1.0,
            k_b=1.0,
            k_2=4.0,
        ),
        wave_number_ratio=300.0,
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
    )
    assert general["finite_generalized_root_count"] == 24
    assert general["infinite_generalized_root_count"] == 16
    assert general["maximum_normalized_exponential_growth"] == pytest.approx(
        unitary["maximum_normalized_exponential_growth"],
        abs=1.0e-9,
    )
    assert general["maximum_absolute_frequency_over_metric_light"] == pytest.approx(
        unitary["maximum_absolute_frequency_over_metric_light"],
        abs=1.0e-9,
    )


def test_invalid_general_covector_protocol_is_rejected() -> None:
    with pytest.raises(ValueError):
        GeneralCovectorBackground(
            scalar_covector=(0.0, 1.0, 0.0, 0.0),
            aether_spatial_covector=(0.0, 0.0, 0.0),
        ).validated()
    with pytest.raises(ValueError):
        solve_tilted_constant_branch_scalar_clock(tilt_magnitude=0.0)
    with pytest.raises(ValueError):
        general_covector_characteristic_row(
            GeneralCovectorBackground(
                scalar_covector=(1.0, 0.0, 0.0, 0.0),
                aether_spatial_covector=(0.0, 0.0, 0.0),
            ),
            wave_number_ratio=0.0,
            principal_growth_threshold=0.01,
            metric_cone_frequency_tolerance=0.01,
        )


def test_characteristic_covector_norm_matches_metric_definition() -> None:
    row = general_covector_characteristic_row(
        GeneralCovectorBackground(
            scalar_covector=(0.5, 0.0, 0.0, 0.0),
            aether_spatial_covector=(0.5, 0.0, 0.0),
        ),
        wave_number_ratio=300.0,
        principal_growth_threshold=0.01,
        metric_cone_frequency_tolerance=0.01,
    )
    frequency = float(row["maximum_absolute_frequency_over_metric_light"])
    norm = float(
        row["minimum_normalized_oscillatory_characteristic_covector_norm"]
    )
    assert norm == pytest.approx(1.0 - frequency**2, abs=1.0e-12)
    assert np.isfinite(norm)
