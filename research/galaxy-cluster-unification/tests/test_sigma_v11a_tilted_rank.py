from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v11a_tilted_rank import (
    audit_v11a_tilted_rank,
    critical_memory_gradient,
    finite_difference_scalar_hessian,
    tilted_scalar_velocity_hessian,
)

COEFFICIENTS = {
    "aether_tilt": 0.5,
    "acceleration_scale": 1.0,
    "memory_speed_squared": 3.0 / 11.0,
    "anisotropy_fraction": 1.0 / 4.0,
    "base_scalar_velocity_hessian": 8.0,
}


def test_critical_configuration_is_finite_and_exact() -> None:
    critical = critical_memory_gradient(**COEFFICIENTS)
    assert critical["scalar_velocity"] == pytest.approx(np.sqrt(3.0))
    assert critical["memory_spatial_gradient"] == pytest.approx(np.sqrt(1056.0))
    hessian = tilted_scalar_velocity_hessian(
        critical["scalar_velocity"],
        critical["memory_spatial_gradient"],
        **COEFFICIENTS,
    )
    assert hessian == pytest.approx(0.0, abs=1.0e-12)


def test_scalar_hessian_crosses_from_positive_to_negative() -> None:
    critical = critical_memory_gradient(**COEFFICIENTS)
    velocity = critical["scalar_velocity"]
    gradient = critical["memory_spatial_gradient"]
    below = tilted_scalar_velocity_hessian(
        velocity, 0.99 * gradient, **COEFFICIENTS
    )
    above = tilted_scalar_velocity_hessian(
        velocity, 1.01 * gradient, **COEFFICIENTS
    )
    assert below > 0.0
    assert above < 0.0


def test_analytic_hessian_matches_five_point_difference() -> None:
    critical = critical_memory_gradient(**COEFFICIENTS)
    numerical = finite_difference_scalar_hessian(
        critical["scalar_velocity"],
        0.9 * critical["memory_spatial_gradient"],
        step=1.0e-3,
        **COEFFICIENTS,
    )
    analytic = tilted_scalar_velocity_hessian(
        critical["scalar_velocity"],
        0.9 * critical["memory_spatial_gradient"],
        **COEFFICIENTS,
    )
    assert numerical == pytest.approx(analytic, rel=1.0e-7, abs=1.0e-7)


def test_larger_base_hessian_only_moves_finite_surface() -> None:
    low = critical_memory_gradient(**COEFFICIENTS)
    higher_coefficients = {**COEFFICIENTS, "base_scalar_velocity_hessian": 80.0}
    high = critical_memory_gradient(**higher_coefficients)
    assert high["memory_spatial_gradient"] == pytest.approx(
        np.sqrt(10.0) * low["memory_spatial_gradient"]
    )
    assert np.isfinite(high["memory_spatial_gradient"])


def test_v11a_tilted_rank_audit_fails_global_positivity_without_data() -> None:
    report = audit_v11a_tilted_rank(
        **COEFFICIENTS,
        finite_difference_step=1.0e-3,
    )
    assert report["gates"]["analytic_hessian_matches_finite_difference"]
    assert report["gates"]["critical_surface_zero"]
    assert report["gates"]["above_surface_negative"]
    assert not report["gates"]["globally_positive_scalar_velocity_hessian"]
    assert not report["all_nonlinear_rank_gates_pass"]
    assert not report["observational_data_accessed"]


def test_invalid_tilted_rank_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        critical_memory_gradient(**{**COEFFICIENTS, "aether_tilt": 0.0})
    with pytest.raises(ValueError):
        critical_memory_gradient(**{**COEFFICIENTS, "aether_tilt": 1.0})
