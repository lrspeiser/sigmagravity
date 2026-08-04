from __future__ import annotations

import pytest

from voidscreen.sigma_v11b_tilted_rank import (
    audit_v11b_tilted_rank,
    material_coordinate_velocity,
    positive_rank_surface,
    tilted_flow_hessian,
)

COEFFICIENTS = {
    "aether_tilt": 0.5,
    "shear_speed_squared": 3.0 / 11.0,
    "bulk_weight": 17.0 / 24.0,
}


def test_finite_rank_surface_is_inside_timelike_material_domain() -> None:
    critical = positive_rank_surface(**COEFFICIENTS)
    assert critical == pytest.approx(1.6835994477509886)
    assert abs(material_coordinate_velocity(critical, aether_tilt=0.5)) < 1.0
    assert tilted_flow_hessian(critical, **COEFFICIENTS) == pytest.approx(0.0, abs=1.0e-12)


def test_hessian_crosses_before_material_light_boundary() -> None:
    critical = positive_rank_surface(**COEFFICIENTS)
    assert tilted_flow_hessian(0.99 * critical, **COEFFICIENTS) > 0.0
    assert tilted_flow_hessian(1.01 * critical, **COEFFICIENTS) < 0.0
    assert abs(material_coordinate_velocity(1.01 * critical, aether_tilt=0.5)) < 1.0


def test_v11b_tilted_audit_retires_candidate_without_data() -> None:
    report = audit_v11b_tilted_rank(**COEFFICIENTS)
    assert report["gates"]["crossing_occurs_with_timelike_material_flow"]
    assert report["gates"]["above_surface_negative"]
    assert not report["gates"]["globally_positive_material_velocity_hessian"]
    assert not report["all_nonlinear_rank_gates_pass"]
    assert not report["observational_data_accessed"]
