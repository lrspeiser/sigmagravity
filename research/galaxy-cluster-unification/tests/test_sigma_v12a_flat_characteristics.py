from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v12a_flat_characteristics import (
    audit_v12a_flat_characteristics,
    flat_characteristic_spectrum,
    gauge_fixed_characteristic_pencil,
    generalized_characteristic_eigensystem,
)
from voidscreen.sigma_v12a_tilted_principal import TiltedPrincipalBackground


def test_characteristic_pencil_forms_full_eom_before_spatial_gauge() -> None:
    row = gauge_fixed_characteristic_pencil(
        TiltedPrincipalBackground(1.0, 0.0, 1.0e-5),
        wave_number_ratio=100.0,
    )
    assert row["full_euler_dimension"] == 26
    assert row["gauge_fixed_euler_dimension"] == 20
    assert row["spatial_gauge_indices"] == [6, 8, 9, 19, 21, 22]
    assert row["gauge_applied_after_full_euler_matrices"]
    assert np.asarray(row["left_pencil"]).shape == (40, 40)
    assert np.asarray(row["right_pencil"]).shape == (40, 40)
    assert np.asarray(row["C"]) == pytest.approx(-np.asarray(row["C"]).T)


def test_direct_flat_pencil_reproduces_six_aest_modes() -> None:
    row = flat_characteristic_spectrum(
        TiltedPrincipalBackground(1.0, 0.0, 1.0e-5),
        wave_number_ratio=300.0,
    )
    assert row["finite_generalized_root_count"] == 24
    assert row["infinite_generalized_root_count"] == 16
    assert row["inferred_linear_physical_degrees_of_freedom"] == 6.0
    assert row["groups"]["zero_frequency_sector"]["count"] == 4
    assert abs(row["groups"]["zero_frequency_sector"]["mean_real"]) < 1.0e-12
    assert row["groups"]["finite_scalar"]["count"] == 4
    assert row["groups"]["finite_scalar"]["mean_real"] == pytest.approx(
        0.5,
        abs=2.0e-5,
    )
    assert row["groups"]["tensor_vector_luminal"]["count"] == 16
    assert row["groups"]["tensor_vector_luminal"]["minimum_real"] == pytest.approx(
        1.0,
        abs=2.0e-5,
    )
    assert row["groups"]["tensor_vector_luminal"]["maximum_real"] == pytest.approx(
        1.0,
        abs=2.0e-5,
    )
    assert row["maximum_polynomial_residual"] < 1.0e-9
    assert row["propagating_energy_root_count"] == 20
    assert row["all_identified_finite_frequency_mode_energies_positive"]
    assert row["minimum_normalized_propagating_mode_energy"] > 0.0


def test_v12a_sign_is_absent_on_the_flat_clock_background() -> None:
    rows = [
        flat_characteristic_spectrum(
            TiltedPrincipalBackground(
                1.0,
                0.0,
                1.0e-5,
                orientation_strength=strength,
            ),
            wave_number_ratio=100.0,
        )
        for strength in (-1.0, 1.0)
    ]
    spectra = [
        np.asarray([value["speed_squared_real"] for value in row["roots"]])
        for row in rows
    ]
    assert spectra[0] == pytest.approx(spectra[1], abs=1.0e-12)


def test_homogeneous_eigenvalue_filter_keeps_dhost_constraint_at_infinity() -> None:
    row = generalized_characteristic_eigensystem(
        TiltedPrincipalBackground(2.0, 0.0, 0.5),
        wave_number_ratio=300.0,
    )
    assert row["finite_generalized_root_count"] == 24
    assert row["infinite_generalized_root_count"] == 16
    assert row["minimum_finite_homogeneous_beta_margin"] > 100.0
    assert row["maximum_infinite_homogeneous_beta_margin"] < 1.0


def test_flat_characteristic_audit_passes_without_widening_scope() -> None:
    report = audit_v12a_flat_characteristics(
        k_b=1.0,
        k_2=2.0,
        background_clock_ratio=1.0,
        aligned_tilt_sentinel=1.0e-5,
        orientation_strengths=(-1.0, 1.0),
        wave_number_sentinels=(100.0, 300.0, 1000.0),
        scalar_speed_squared_target=0.5,
        scalar_limit_tolerance=1.0e-5,
        luminal_limit_tolerance=1.0e-6,
        polynomial_residual_tolerance=1.0e-8,
    )
    assert all(report["gates"].values())
    assert report["all_flat_characteristic_gates_pass"]
    assert report["flat_linear_six_degree_count_reproduced"]
    assert report["flat_finite_frequency_energy_positive"]
    assert not report["zero_frequency_jeans_sector_resolved"]
    assert not report["tilted_background_characteristics_proven"]
    assert not report["nonconstant_background_characteristics_proven"]
    assert not report["nonlinear_physical_degree_count_proven"]
    assert not report["theory_viable"]
    assert not report["observational_data_accessed"]
    assert not report["raw_holdout_opened"]


def test_invalid_characteristic_protocol_is_rejected() -> None:
    with pytest.raises(ValueError):
        flat_characteristic_spectrum(
            TiltedPrincipalBackground(1.0, 0.0, 1.0e-5),
            wave_number_ratio=100.0,
            finite_frequency_threshold=0.0,
        )
    with pytest.raises(ValueError):
        audit_v12a_flat_characteristics(
            k_b=1.0,
            k_2=2.0,
            background_clock_ratio=1.0,
            aligned_tilt_sentinel=1.0e-5,
            orientation_strengths=(1.0, 2.0),
            wave_number_sentinels=(100.0, 300.0, 1000.0),
            scalar_speed_squared_target=0.5,
            scalar_limit_tolerance=1.0e-5,
            luminal_limit_tolerance=1.0e-6,
            polynomial_residual_tolerance=1.0e-8,
        )
