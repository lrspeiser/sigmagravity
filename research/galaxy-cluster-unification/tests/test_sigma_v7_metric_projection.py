from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v7_metric_projection import (
    audit_v7c_metric_projection,
    conformal_helicity0_projection,
    massive_spin2_helicity_decomposition,
    null_metric_contraction,
    static_disformal_metric,
)


def test_conformal_helicity_zero_cancels_exactly_from_weyl_potential() -> None:
    scalar = np.linspace(-5.0, 5.0, 101)
    projection = conformal_helicity0_projection(scalar)
    assert np.allclose(projection.psi, -0.5 * scalar)
    assert np.allclose(projection.phi, 0.5 * scalar)
    assert np.array_equal(projection.weyl, np.zeros_like(scalar))


def test_vdvz_extra_force_is_scalar_but_extra_weyl_is_zero() -> None:
    result = massive_spin2_helicity_decomposition()
    assert result["helicity0"] == {"psi": 1.0 / 3.0, "phi": -1.0 / 3.0, "weyl": 0.0}
    assert result["total"]["psi"] == pytest.approx(4.0 / 3.0)
    assert result["total"]["phi"] == pytest.approx(2.0 / 3.0)
    assert result["total"]["weyl"] == pytest.approx(1.0)
    assert result["ppn_gamma"] == pytest.approx(0.5)
    assert result["cavendish_normalized_weyl"] == pytest.approx(0.75)


def test_static_disformal_term_is_spatial_and_direction_dependent() -> None:
    gradient = np.array([2.0, 0.0, 0.0])
    metric = static_disformal_metric(gradient, 0.5)
    assert np.array_equal(metric[0, :], np.zeros(4))
    assert np.array_equal(metric[:, 0], np.zeros(4))
    assert metric[1, 1] == pytest.approx(-2.0)
    assert float(
        null_metric_contraction(metric, np.array([1.0, 0.0, 0.0]))
    ) == pytest.approx(-2.0)
    assert float(
        null_metric_contraction(metric, np.array([0.0, 1.0, 0.0]))
    ) == pytest.approx(0.0)


def test_frozen_scalar_only_v7c_fails_physical_lensing_projection() -> None:
    audit = audit_v7c_metric_projection(
        scalar_samples=np.linspace(-3.0, 3.0, 61),
        conformal_cancellation_tolerance=1.0e-15,
        minimum_nonzero_null_response=1.0e-12,
        disformal_mapping_frozen=False,
        complete_scalar_equation_frozen=False,
        coupled_tensor_equation_frozen=False,
    )
    assert audit["maximum_absolute_conformal_weyl_response"] == 0.0
    assert audit["gates"]["conformal_projection_identity"]
    assert audit["gates"]["vdvz_helicity_decomposition"]
    assert not audit["gates"]["action_derived_nonzero_weyl_or_null_response"]
    assert not audit["gates"]["complete_scalar_metric_mapping"]
    assert not audit["gates"]["coupled_tensor_closure_if_used"]


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (conformal_helicity0_projection, (np.array([np.nan]),)),
        (massive_spin2_helicity_decomposition, (-0.1,)),
        (static_disformal_metric, (np.zeros(2), 1.0)),
        (static_disformal_metric, (np.zeros(3), np.nan)),
        (null_metric_contraction, (np.zeros((3, 3)), np.array([1.0, 0.0, 0.0]))),
        (null_metric_contraction, (np.zeros((4, 4)), np.array([2.0, 0.0, 0.0]))),
    ],
)
def test_invalid_metric_projection_inputs_are_rejected(function, arguments) -> None:
    with pytest.raises(ValueError):
        function(*arguments)
