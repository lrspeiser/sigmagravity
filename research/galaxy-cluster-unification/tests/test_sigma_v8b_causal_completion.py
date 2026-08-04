from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v8b_causal_completion import (
    audit_v8b_scalar_selection,
    causal_completion_alpha,
    completed_cubic_static_principal_symbol,
    completed_spherical_characteristics,
)


def test_derived_completion_coefficient_is_sixteen_ninths() -> None:
    assert causal_completion_alpha(base_speed_squared=0.75) == pytest.approx(
        16.0 / 9.0
    )


def test_completed_spherical_branch_is_causal_at_old_failure_point() -> None:
    alpha = causal_completion_alpha(base_speed_squared=0.75)
    result = completed_spherical_characteristics(
        0.2,
        base_speed_squared=0.75,
        alpha=alpha,
    )
    assert result["positive"]
    assert result["causal"]
    assert result["radial_speed_squared"] < 1.0


def test_completion_saturates_light_cone_at_analytic_peak() -> None:
    alpha = causal_completion_alpha(base_speed_squared=0.75)
    result = completed_spherical_characteristics(
        0.1875,
        base_speed_squared=0.75,
        alpha=alpha,
    )
    assert result["radial_speed_squared"] == pytest.approx(1.0)


def test_completed_deep_spherical_limits_are_subluminal() -> None:
    alpha = causal_completion_alpha(base_speed_squared=0.75)
    result = completed_spherical_characteristics(
        1.0e9,
        base_speed_squared=0.75,
        alpha=alpha,
    )
    assert result["radial_speed_squared"] == pytest.approx(0.75, rel=1.0e-8)
    assert result["tangential_speed_squared"] == pytest.approx(
        0.1875, rel=1.0e-8
    )


def test_equal_trace_isotropic_and_rank_one_probes_are_causal() -> None:
    alpha = causal_completion_alpha(base_speed_squared=0.75)
    for hessian in (np.eye(3), np.diag([3.0, 0.0, 0.0])):
        result = completed_cubic_static_principal_symbol(
            hessian,
            base_speed_squared=0.75,
            alpha=alpha,
        )
        assert result.positive
        assert result.causal


def test_v8b_scalar_selection_passes_narrow_gate() -> None:
    audit = audit_v8b_scalar_selection(
        base_speed_squared=0.75,
        physical_parameter_count=5,
        maximum_physical_parameters=5,
    )
    assert all(audit["gates"].values())
    assert audit["spherical_scan"]["maximum_radial_speed_squared"] <= 1.0 + 1.0e-12
    bound = audit["nonnegative_source_extremal_bound"]
    assert bound["maximum_speed_squared"] <= 1.0 + 1.0e-12
    assert bound["minimum_spatial_eigenvalue_over_finite_scan"] > 0.0


@pytest.mark.parametrize("base", [0.0, 1.0, np.nan])
def test_invalid_base_speed_is_rejected(base: float) -> None:
    with pytest.raises(ValueError):
        causal_completion_alpha(base_speed_squared=base)
