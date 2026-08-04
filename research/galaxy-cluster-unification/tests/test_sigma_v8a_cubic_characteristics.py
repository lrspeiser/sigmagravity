from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v8_aest_galileon import (
    cubic_static_principal_symbol,
    negative_cubic_branch_limit,
    positive_cubic_causality_limit,
    spherical_positive_cubic_characteristics,
)


def test_flat_selected_aest_scalar_symbol_is_positive_and_causal() -> None:
    symbol = cubic_static_principal_symbol(
        np.zeros((3, 3)),
        base_speed_squared=0.75,
    )
    assert symbol.temporal_coefficient == pytest.approx(1.0)
    assert np.allclose(symbol.spatial_eigenvalues, 0.75)
    assert np.allclose(symbol.speed_squared, 0.75)
    assert symbol.positive
    assert symbol.causal


def test_spherical_exterior_formula_matches_direct_principal_symbol() -> None:
    u = 0.2
    result = spherical_positive_cubic_characteristics(
        u,
        base_speed_squared=0.75,
    )
    ratio = float(result["radial_to_tangential_hessian"])
    direct = cubic_static_principal_symbol(
        np.diag([ratio * u, u, u]),
        base_speed_squared=0.75,
    )
    assert result["radial_speed_squared"] == pytest.approx(
        direct.speed_squared[-1]
    )
    assert result["tangential_speed_squared"] == pytest.approx(
        direct.speed_squared[0]
    )


def test_positive_branch_crosses_light_at_analytic_threshold() -> None:
    limit = positive_cubic_causality_limit(base_speed_squared=0.75)
    u = limit["dimensionless_tangential_hessian"]
    at_limit = spherical_positive_cubic_characteristics(
        u,
        base_speed_squared=0.75,
    )
    above = spherical_positive_cubic_characteristics(
        1.001 * u,
        base_speed_squared=0.75,
    )
    assert u == pytest.approx(0.08071891388307384)
    assert at_limit["radial_speed_squared"] == pytest.approx(1.0)
    assert limit["maximum_nonlinear_fraction_of_total_flux"] == pytest.approx(
        0.17712434446770473
    )
    assert above["radial_speed_squared"] > 1.0
    assert not above["causal"]


def test_positive_branch_deep_limit_is_four_thirds_radially() -> None:
    deep = spherical_positive_cubic_characteristics(
        1.0e8,
        base_speed_squared=0.75,
    )
    assert deep["positive"]
    assert deep["radial_speed_squared"] == pytest.approx(4.0 / 3.0, rel=1.0e-8)
    assert deep["tangential_speed_squared"] == pytest.approx(1.0 / 3.0, rel=1.0e-8)
    assert not deep["causal"]


def test_negative_sign_ends_where_radial_ellipticity_vanishes() -> None:
    limit = negative_cubic_branch_limit(base_speed_squared=0.75)
    assert limit["dimensionless_tangential_hessian_magnitude"] == pytest.approx(
        0.1875
    )
    assert limit["radial_spatial_coefficient_at_endpoint"] == pytest.approx(0.0)
    assert limit["maximum_dimensionless_flux"] == pytest.approx(0.0703125)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (cubic_static_principal_symbol, (np.zeros((2, 2)),)),
        (spherical_positive_cubic_characteristics, (-1.0,)),
    ],
)
def test_invalid_characteristic_inputs_are_rejected(function, arguments) -> None:
    with pytest.raises(ValueError):
        function(*arguments, base_speed_squared=0.75)
