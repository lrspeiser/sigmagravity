from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v11c_biot_stretch_rank import (
    audit_v11c_biot_stretch_rank,
    biot_energy_from_singular_values,
    critical_axial_stretch,
    rank_one_biot_stiffness,
    tilted_biot_hessian,
    tilted_biot_lagrangian,
    vacuum_longitudinal_hessian,
)

FIXED = {
    "aether_tilt": 0.5,
    "shear_speed_squared": 3.0 / 11.0,
    "bulk_weight": 17.0 / 24.0,
    "transverse_stretch": 0.1,
}


def test_biot_energy_matches_direct_stretch_definition() -> None:
    sigma = np.asarray([0.1, 0.1, 10.0])
    direct = biot_energy_from_singular_values(
        sigma,
        shear_speed_squared=3.0 / 11.0,
        bulk_weight=17.0 / 24.0,
    )
    assert direct == pytest.approx((3.0 / 11.0) * (5103.0 / 50.0))
    assert tilted_biot_lagrangian(0.0, axial_stretch=10.0, **FIXED) == pytest.approx(-direct)


def test_simple_longitudinal_channel_is_repaired() -> None:
    assert vacuum_longitudinal_hessian(
        aether_tilt=0.5,
        shear_speed_squared=3.0 / 11.0,
        bulk_weight=17.0 / 24.0,
    ) == pytest.approx(13.0 / 12.0)


def test_exact_mixed_shear_counterexample_is_negative_inside_gl_plus() -> None:
    stiffness = rank_one_biot_stiffness(
        shear_speed_squared=3.0 / 11.0,
        bulk_weight=17.0 / 24.0,
        transverse_stretch=0.1,
        axial_stretch=10.0,
    )
    assert stiffness == pytest.approx(57.0 / 11.0)
    assert stiffness > 1.0 / FIXED["aether_tilt"] ** 2
    assert tilted_biot_hessian(axial_stretch=10.0, **FIXED) == pytest.approx(-13.0 / 33.0)
    assert 0.1**2 * 10.0 > 0.0


def test_finite_rank_surface_crosses_with_comoving_timelike_material() -> None:
    critical = critical_axial_stretch(**FIXED)
    assert critical == pytest.approx(398.0 / 45.0)
    assert tilted_biot_hessian(axial_stretch=0.99 * critical, **FIXED) > 0.0
    assert tilted_biot_hessian(axial_stretch=critical, **FIXED) == pytest.approx(0.0, abs=1.0e-12)
    assert tilted_biot_hessian(axial_stretch=1.01 * critical, **FIXED) < 0.0


def test_v11c_audit_triggers_third_failure_reset_without_data() -> None:
    report = audit_v11c_biot_stretch_rank(
        **FIXED,
        counterexample_axial_stretch=10.0,
        finite_difference_step=1.0e-4,
    )
    assert report["gates"]["analytic_finite_difference_agree"]
    assert report["gates"]["critical_material_flow_timelike"]
    assert not report["gates"]["globally_positive_material_velocity_hessian"]
    assert not report["all_nonlinear_rank_gates_pass"]
    assert report["third_post_reset_same_gate_failure"]
    assert report["mechanism_reset_required"]
    assert not report["observational_data_accessed"]
