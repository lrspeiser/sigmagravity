from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_degenerate_action import normalized_dhost_residuals
from voidscreen.sigma_v12a_same_clock_dhost import (
    audit_v12a_same_clock_dhost,
    same_clock_activation,
    static_dhost_invariants,
    v12a_dimensionless_coefficients,
)


def test_activation_and_first_derivative_vanish_on_clock_background() -> None:
    background = -3.0
    assert same_clock_activation(background, background_kinetic_ratio=background) == pytest.approx(
        0.0
    )
    step = 1.0e-6
    derivative = (
        same_clock_activation(background + step, background_kinetic_ratio=background)
        - same_clock_activation(background - step, background_kinetic_ratio=background)
    ) / (2.0 * step)
    assert derivative == pytest.approx(0.0, abs=1.0e-12)


def test_v12a_coefficients_obey_luminal_class_ia_identities() -> None:
    ratio = np.linspace(-100.0, 100.0, 10001)
    coefficients = v12a_dimensionless_coefficients(
        ratio,
        background_kinetic_ratio=-1.0,
        orientation_strength=1.0,
    )
    residuals = normalized_dhost_residuals(
        ratio, np.ones_like(ratio), np.zeros_like(ratio), coefficients
    )
    for residual in residuals.values():
        assert np.max(np.abs(residual)) < 1.0e-12
    for coefficient in coefficients.values():
        assert np.all(np.isfinite(coefficient))


def test_equal_trace_hessians_have_different_directional_invariants() -> None:
    gradient = np.asarray([1.0, 0.0, 0.0])
    isotropic = static_dhost_invariants(gradient, np.eye(3))
    rank_one = static_dhost_invariants(gradient, np.diag([3.0, 0.0, 0.0]))
    assert isotropic["trace_hessian"] == rank_one["trace_hessian"] == 3.0
    assert isotropic["L3"] == pytest.approx(3.0)
    assert rank_one["L3"] == pytest.approx(9.0)
    assert isotropic["L4"] == pytest.approx(1.0)
    assert rank_one["L4"] == pytest.approx(9.0)


def test_high_acceleration_activation_is_automatically_screened() -> None:
    background = -1.0
    high_field = same_clock_activation(
        background + 1.0e10,
        background_kinetic_ratio=background,
    )
    assert high_field < 1.0e-5


def test_v12a_selection_passes_but_does_not_claim_theory_viability() -> None:
    report = audit_v12a_same_clock_dhost(
        k_b=1.0,
        k_2=2.0,
        lambda_s=1.0,
        orientation_strength=1.0,
        background_kinetic_ratios=[-100.0, -1.0, -0.01],
        signed_scan_limit=1.0e8,
        signed_scan_points=1001,
        high_acceleration_ratio=1.0e5,
        random_rotation_trials=100,
        random_seed=12001,
    )
    assert report["all_selection_gates_pass"]
    assert not report["full_joint_adm_degeneracy_proven"]
    assert not report["complete_metric_stress_derived"]
    assert not report["arbitrary_background_characteristics_proven"]
    assert not report["theory_viable"]
    assert not report["observational_data_accessed"]
