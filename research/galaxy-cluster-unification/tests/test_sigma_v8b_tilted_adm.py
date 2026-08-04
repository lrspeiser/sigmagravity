from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v8b_tilted_adm import (
    aether_spatial_norm,
    audit_tilted_adm_kinetic_gate,
    centered_finite_difference_hessian,
    completion_adm_boundary_identity,
    find_outside_envelope_singularity,
    homogeneous_adm_lagrangian,
    homogeneous_kinetic_point,
)


def test_boundary_reduction_identity_holds_on_random_tilted_inputs() -> None:
    rng = np.random.default_rng(80804)
    residuals = []
    for _ in range(64):
        values = rng.uniform(-1.0, 1.0, size=7)
        identity = completion_adm_boundary_identity(
            spatial_norm_squared=float(rng.uniform(0.0, 5.0)),
            sigma=float(values[0]),
            sigma_normal_derivative=float(values[1]),
            extrinsic_trace=float(values[2]),
            aether_extrinsic_projection=float(values[3]),
            aether_electric_contraction=float(values[4]),
            q_0=float(values[5]),
            coefficient=float(values[6]),
        )
        residuals.append(abs(identity.residual))
    assert max(residuals) < 1.0e-12


def test_autodiff_hessian_matches_independent_centered_difference() -> None:
    aether_velocity = 0.63
    q_ratio = 1.17
    acceleration_ratio = 0.37
    length_ratio = 0.8
    background = np.array(
        (0.03, -0.02, 0.01, 0.015, -0.01, 0.02, 0.04, -0.03, 0.01)
    )
    spatial_norm = aether_spatial_norm(aether_velocity)
    chi = np.sqrt(1.0 + spatial_norm**2)
    velocities = np.concatenate((background, np.array((q_ratio / chi,))))
    point = homogeneous_kinetic_point(
        aether_velocity=aether_velocity,
        q_over_q0=q_ratio,
        a_sigma_over_q0=acceleration_ratio,
        ell_h_q0=length_ratio,
        background_kinematics=background,
    )
    numerical = centered_finite_difference_hessian(
        velocities,
        aether_velocity=aether_velocity,
        a_sigma_over_q0=acceleration_ratio,
        ell_h_q0=length_ratio,
    )
    relative_error = np.linalg.norm(numerical - point.combined_hessian) / np.linalg.norm(
        point.combined_hessian
    )
    assert relative_error < 1.0e-6


def test_completion_hessian_equals_base_at_aligned_clock_minimum() -> None:
    point = homogeneous_kinetic_point(
        aether_velocity=0.0,
        q_over_q0=1.0,
        a_sigma_over_q0=1.0,
        ell_h_q0=1.0,
    )
    assert np.array_equal(point.base_hessian, point.combined_hessian)
    assert point.determinant_ratio == pytest.approx(1.0)
    assert point.combined_inertia == (1, 0, 9)


def test_selected_envelope_point_preserves_rank_and_inertia() -> None:
    point = homogeneous_kinetic_point(
        aether_velocity=0.9,
        q_over_q0=0.75,
        a_sigma_over_q0=1.0e-4,
        ell_h_q0=1.0,
    )
    assert point.base_inertia == (1, 0, 9)
    assert point.combined_inertia == point.base_inertia
    assert point.combined_singular_values[-1] > 1.0e-8
    assert point.determinant_ratio > 0.9


def test_finite_rank_changing_surface_exists_outside_declared_envelope() -> None:
    singularity = find_outside_envelope_singularity()
    assert singularity["q_over_q0"] == pytest.approx(2.8649430865, rel=1.0e-8)
    assert singularity["invariant_Y_over_Q_squared"] == pytest.approx(0.9409)
    assert abs(singularity["determinant_ratio"]) < 1.0e-7
    assert singularity["minimum_combined_singular_value"] < 1.0e-8
    before = homogeneous_kinetic_point(
        aether_velocity=0.97,
        q_over_q0=2.8,
        a_sigma_over_q0=1.0,
        ell_h_q0=1.0,
    )
    after = homogeneous_kinetic_point(
        aether_velocity=0.97,
        q_over_q0=2.9,
        a_sigma_over_q0=1.0,
        ell_h_q0=1.0,
    )
    assert before.combined_inertia == (1, 0, 9)
    assert after.combined_inertia == (2, 0, 8)


def test_gate_passes_only_completed_local_subgates() -> None:
    audit = audit_tilted_adm_kinetic_gate(
        deterministic_velocities=(0.0, 0.45, 0.9),
        deterministic_q_ratios=(0.5, 1.0, 1.5),
        a_sigma_ratios=(1.0e-4, 1.0, 100.0),
        maximum_ell_h_q0=1.0,
        maximum_background_kinematic=0.1,
        random_samples=8,
        random_seed=80804,
    )
    assert audit["sample_counts"] == {
        "deterministic": 27,
        "random": 8,
        "total": 35,
    }
    assert all(audit["completed_subgates"].values())
    assert not any(audit["unresolved_kill_gates"].values())
    assert not audit["full_hamiltonian_gate_pass"]
    assert audit["conditional_local_dof_count"] == 6.0


def test_invalid_tilted_adm_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        aether_spatial_norm(1.0)
    with pytest.raises(ValueError):
        homogeneous_adm_lagrangian(
            np.zeros(9),
            aether_velocity=0.2,
            a_sigma_over_q0=1.0,
            ell_h_q0=1.0,
        )
    with pytest.raises(ValueError):
        centered_finite_difference_hessian(
            np.zeros(10),
            aether_velocity=0.2,
            a_sigma_over_q0=1.0,
            ell_h_q0=1.0,
            step=0.0,
        )
    with pytest.raises(ValueError):
        homogeneous_kinetic_point(
            aether_velocity=0.2,
            q_over_q0=0.0,
            a_sigma_over_q0=1.0,
            ell_h_q0=1.0,
        )
