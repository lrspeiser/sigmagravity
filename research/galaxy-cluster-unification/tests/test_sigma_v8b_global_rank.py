from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v8b_global_rank import (
    asymptotic_schur_coefficient,
    audit_global_rank_falsification,
    critical_aether_tilt,
    derived_causal_alpha,
    find_first_rank_surface,
    isotropic_extrinsic_rank_surface,
    leading_mixing_vector,
    rank_surface_mode_diagnostics,
)
from voidscreen.sigma_v8b_tilted_adm import (
    aether_spatial_norm,
    homogeneous_adm_lagrangian,
    homogeneous_canonical_point,
    homogeneous_kinetic_point,
)


def test_analytic_asymptotic_mixing_matches_large_q_hessian() -> None:
    aether_velocity = 0.97
    q_ratio = 1.0e5
    spatial_norm = aether_spatial_norm(aether_velocity)
    chi = np.sqrt(1.0 + spatial_norm**2)
    sigma = q_ratio / chi
    completion_coefficient = 7.0 / 9.0
    point = homogeneous_kinetic_point(
        aether_velocity=aether_velocity,
        q_over_q0=q_ratio,
        a_sigma_over_q0=1.0,
        ell_h_q0=1.0,
    )
    numerical = (
        point.combined_hessian[:9, 9] - point.base_hessian[:9, 9]
    ) / (-completion_coefficient * sigma**2)
    analytic = leading_mixing_vector(spatial_norm**2)
    relative_error = np.linalg.norm(numerical - analytic) / np.linalg.norm(analytic)
    assert relative_error < 3.0e-5


def test_selected_asymptotic_coefficient_has_finite_critical_tilt() -> None:
    critical = critical_aether_tilt(k_b=1.0)
    assert critical["critical_spatial_norm_squared"] == pytest.approx(
        4.369518710154625
    )
    assert critical["critical_aether_velocity"] == pytest.approx(0.9020884486250939)
    assert abs(critical["coefficient_at_root"]) < 1.0e-10
    assert asymptotic_schur_coefficient(16.0, k_b=1.0) > 0.0


@pytest.mark.parametrize(
    ("length_ratio", "expected_q_ratio"),
    [
        (0.25, 9.6648172943),
        (0.5, 5.1337312702),
        (1.0, 2.8649430865),
        (2.0, 1.7403283185),
    ],
)
def test_every_nonzero_tested_completion_length_has_rank_surface(
    length_ratio: float,
    expected_q_ratio: float,
) -> None:
    root = find_first_rank_surface(
        aether_velocity=0.97,
        a_sigma_over_q0=1.0,
        ell_h_q0=length_ratio,
    )
    assert root == pytest.approx(expected_q_ratio, rel=1.0e-8)


def test_rank_surface_is_finite_mixed_and_adds_negative_direction() -> None:
    mode = rank_surface_mode_diagnostics()
    assert abs(mode["null_eigenvalue"]) < 1.0e-8
    assert mode["minimum_singular_value"] < 1.0e-8
    assert mode["sector_power"]["metric"] > 0.4
    assert mode["sector_power"]["aether"] > 0.4
    assert np.isfinite(mode["canonical_energy_at_zero_K_E"])
    assert mode["all_canonical_momenta_finite"]


@pytest.mark.parametrize("k_b", [1.0, 1.6, 1.7, 1.8, 1.95])
def test_high_k_b_escape_has_finite_extrinsic_curvature_surface(k_b: float) -> None:
    surface = isotropic_extrinsic_rank_surface(k_b=k_b)
    assert abs(surface["determinant_ratio_at_root"]) < 1.0e-8
    assert surface["minimum_singular_value_at_root"] < 1.0e-8
    assert abs(surface["determinant_affine_residual_at_two"]) < 1.0e-8
    assert np.isfinite(surface["canonical_energy_at_surface"])
    assert surface["all_momenta_finite"]
    assert surface["inertia_below_surface"] != surface["inertia_above_surface"]


def test_high_k_b_row_is_linearly_healthy_but_not_globally_regular() -> None:
    mode = derived_causal_alpha(k_b=1.7)
    assert mode["scalar_speed_squared"] == pytest.approx(0.1632352941176471)
    assert mode["alpha"] == pytest.approx(2.4404017374140397)
    surface = isotropic_extrinsic_rank_surface(k_b=1.7)
    assert surface["isotropic_extrinsic_trace_over_q0"] == pytest.approx(
        1.8488346177453117
    )


def test_canonical_momenta_match_independent_centered_gradient() -> None:
    values = np.array((0.02, -0.01, 0.03, 0.01, 0.0, -0.02, 0.03, 0.0, 0.01, 0.7))
    point = homogeneous_canonical_point(
        values,
        aether_velocity=0.6,
        a_sigma_over_q0=1.0,
        ell_h_q0=0.8,
    )
    step = 1.0e-6
    numerical = np.empty(10)
    for index in range(10):
        offset = np.zeros(10)
        offset[index] = step
        plus = homogeneous_adm_lagrangian(
            values + offset,
            aether_velocity=0.6,
            a_sigma_over_q0=1.0,
            ell_h_q0=0.8,
        )
        minus = homogeneous_adm_lagrangian(
            values - offset,
            aether_velocity=0.6,
            a_sigma_over_q0=1.0,
            ell_h_q0=0.8,
        )
        numerical[index] = (plus - minus) / (2.0 * step)
    assert np.allclose(point.momenta, numerical, rtol=1.0e-8, atol=1.0e-8)
    assert point.canonical_energy == pytest.approx(values @ point.momenta - point.lagrangian)


def test_global_rank_gate_retires_exact_v8b_without_data() -> None:
    audit = audit_global_rank_falsification(
        ell_h_q0_values=(0.25, 0.5, 1.0, 2.0),
        a_sigma_over_q0_values=(1.0e-4, 1.0, 100.0),
        k_b_escape_values=(1.0, 1.6, 1.7, 1.8, 1.95),
    )
    assert all(audit["completed_falsification_subgates"].values())
    assert audit["all_falsification_subgates_pass"]
    assert not audit["global_legendre_map_regular"]
    assert audit["candidate_retired"]
    assert not audit["raw_holdout_opened"]
    assert audit["inertia_before_surface"] == (1, 0, 9)
    assert audit["inertia_after_surface"] == (2, 0, 8)
    assert set(audit["isotropic_extrinsic_rank_surfaces_by_K_B"]) == {
        "1",
        "1.6",
        "1.7",
        "1.8",
        "1.95",
    }


def test_invalid_global_rank_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        asymptotic_schur_coefficient(-1.0)
    with pytest.raises(ValueError):
        critical_aether_tilt(k_b=1.8)
    with pytest.raises(ValueError):
        find_first_rank_surface(
            aether_velocity=0.97,
            a_sigma_over_q0=1.0,
            ell_h_q0=1.0,
            maximum_q_ratio=1.0,
        )
    with pytest.raises(ValueError):
        audit_global_rank_falsification(
            ell_h_q0_values=(),
            a_sigma_over_q0_values=(1.0,),
        )
