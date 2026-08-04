from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v12a_tilted_principal import (
    TiltedPrincipalBackground,
    audit_v12a_tilted_principal,
    mode_lagrangian_hessian,
    reduced_dirac_block,
)


def test_mode_hessian_retains_all_metric_components_and_is_symmetric() -> None:
    background = TiltedPrincipalBackground(1.2, 0.3, -0.4)
    row = mode_lagrangian_hessian(background, wave_number_ratio=1.0)
    for name in ("K", "B"):
        matrix = np.asarray(row[name])
        assert matrix.shape == (26, 26)
        assert matrix == pytest.approx(matrix.T, abs=1.0e-12)
    assert np.asarray(row["A"]).shape == (26, 26)


@pytest.mark.parametrize("strength", [1.0, -1.0])
def test_tilted_dirac_block_follows_exact_class_ia_null(strength: float) -> None:
    row = reduced_dirac_block(
        TiltedPrincipalBackground(
            2.0,
            0.3,
            0.4,
            orientation_strength=strength,
        ),
        wave_number_ratio=1.0,
    )
    assert row["kinetic_nullity"] == 8
    assert row["dhost_nullity"] == 2
    assert row["expected_eight_total_two_dhost_nullity"]
    assert row["clock_normalized_null_residual"] < 1.0e-12
    assert row["null_conformal_ratio_residual"] < 1.0e-12
    assert row["dhost_primary_self_bracket_norm"] < 1.0e-12
    assert row["shift_dhost_primary_cross_norm"] < 1.0e-12
    assert row["shift_dhost_secondary_cross_norm"] < 1.0e-12
    assert np.all(np.asarray(row["dirac_eigenvalues"]) < 0.0)
    assert row["dirac_invertible"]


def test_dynamical_aether_removes_apparent_wave_number_dependence() -> None:
    background = TiltedPrincipalBackground(
        0.5,
        -0.6,
        0.8,
        orientation_strength=1.0,
    )
    eigenvalues = [
        np.asarray(
            reduced_dirac_block(background, wave_number_ratio=wave)["dirac_eigenvalues"]
        )
        for wave in (0.1, 1.0, 10.0)
    ]
    assert eigenvalues[0] == pytest.approx(eigenvalues[1], rel=1.0e-12, abs=1.0e-12)
    assert eigenvalues[2] == pytest.approx(eigenvalues[1], rel=1.0e-12, abs=1.0e-12)


def test_aligned_limit_recovers_continuum_minus_4K2_bracket() -> None:
    row = reduced_dirac_block(
        TiltedPrincipalBackground(1.0, 0.0, 1.0e-4),
        wave_number_ratio=10.0,
    )
    continuum_eigenvalues = 2.0 * np.asarray(row["dirac_eigenvalues"])
    assert continuum_eigenvalues == pytest.approx([-8.0, -8.0], rel=1.0e-8)


def test_small_tilted_audit_keeps_scope_flags_closed() -> None:
    report = audit_v12a_tilted_principal(
        k_b=1.0,
        k_2=2.0,
        background_clock_ratio=1.0,
        positive_orientation_strength=1.0,
        negative_orientation_strength=-1.0,
        random_trials=2,
        logarithmic_clock_limit=1.0,
        logarithmic_tilt_limit=1.0,
        wave_number_sentinels=(0.1, 1.0, 10.0),
        wave_invariance_trials=1,
        aligned_limit_tilt=1.0e-4,
        random_seed=3,
    )
    assert all(report["gates"].values())
    assert not report["previous_aligned_maxwell_stabilization_valid"]
    assert not report["aligned_sign_conclusion_changed"]
    assert report["constant_background_delta_eff_proven_invertible"]
    assert not report["nonconstant_background_delta_eff_proven_invertible"]
    assert not report["complete_physical_characteristic_matrix_scored"]
    assert not report["physical_degree_count_proven_unchanged"]
    assert not report["theory_viable"]
    assert not report["observational_data_accessed"]
    assert not report["raw_holdout_opened"]


def test_invalid_background_and_protocol_are_rejected() -> None:
    with pytest.raises(ValueError):
        TiltedPrincipalBackground(0.0, 0.0, 1.0).validated()
    with pytest.raises(ValueError):
        reduced_dirac_block(
            TiltedPrincipalBackground(1.0, 0.0, 1.0),
            wave_number_ratio=0.0,
        )
    with pytest.raises(ValueError):
        audit_v12a_tilted_principal(
            k_b=1.0,
            k_2=2.0,
            background_clock_ratio=1.0,
            positive_orientation_strength=-1.0,
            negative_orientation_strength=-1.0,
            random_trials=1,
            logarithmic_clock_limit=1.0,
            logarithmic_tilt_limit=1.0,
            wave_number_sentinels=(0.1, 1.0, 10.0),
            wave_invariance_trials=1,
            aligned_limit_tilt=1.0e-4,
            random_seed=1,
        )
