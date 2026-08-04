from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v11b_elastic_triad import (
    audit_v11b_elastic_triad,
    linear_strain,
    phonon_speed_matrix,
    plane_wave_speed_quadratic,
    strain_invariants,
)


def test_strain_split_preserves_trace_and_frobenius_identity() -> None:
    strain = linear_strain(np.array([[1.0, 0.2, 0.0], [-0.3, 0.4, 0.1], [0.0, 0.7, -0.2]]))
    invariants = strain_invariants(strain)
    trace_free = invariants["trace_free"]
    assert np.trace(trace_free) == pytest.approx(0.0, abs=1.0e-14)
    assert np.sum(strain**2) == pytest.approx(
        invariants["trace_free_norm_squared"] + invariants["trace"] ** 2 / 3.0
    )


def test_transverse_and_longitudinal_plane_wave_speeds_are_selected_values() -> None:
    direction = np.array([0.0, 0.0, 1.0])
    shear = 3.0 / 11.0
    bulk = 17.0 / 24.0
    transverse = plane_wave_speed_quadratic(
        direction,
        np.array([1.0, 0.0, 0.0]),
        shear_speed_squared=shear,
        bulk_weight=bulk,
    )
    longitudinal = plane_wave_speed_quadratic(
        direction,
        direction,
        shear_speed_squared=shear,
        bulk_weight=bulk,
    )
    assert transverse == pytest.approx(3.0 / 11.0)
    assert longitudinal == pytest.approx(3.0 / 4.0)


def test_random_direction_speed_matrix_has_rotation_independent_spectrum() -> None:
    matrix = phonon_speed_matrix(
        np.array([0.2, -0.7, 0.4]),
        shear_speed_squared=3.0 / 11.0,
        bulk_weight=17.0 / 24.0,
    )
    assert np.linalg.eigvalsh(matrix) == pytest.approx(
        [3.0 / 11.0, 3.0 / 11.0, 3.0 / 4.0]
    )


def test_v11b_selection_passes_only_the_flat_architecture_gate() -> None:
    report = audit_v11b_elastic_triad(
        shear_speed_squared=3.0 / 11.0,
        longitudinal_speed_squared=3.0 / 4.0,
        physical_parameter_count=5,
        maximum_physical_parameters=5,
        random_directions=100,
    )
    assert all(report["selection_gates"].values())
    assert report["derived_coefficients"]["bulk_weight"] == pytest.approx(17.0 / 24.0)
    assert report["flat_principal_scan"]["maximum_speed_squared"] == pytest.approx(0.75)
    assert not report["unresolved"]["nonlinear_tilted_global_rank"]
    assert not report["observational_data_accessed"]


def test_invalid_elastic_triad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        phonon_speed_matrix(
            np.zeros(3), shear_speed_squared=3.0 / 11.0, bulk_weight=17.0 / 24.0
        )
    with pytest.raises(ValueError):
        audit_v11b_elastic_triad(
            shear_speed_squared=0.9,
            longitudinal_speed_squared=0.1,
            physical_parameter_count=5,
            maximum_physical_parameters=5,
            random_directions=100,
        )
