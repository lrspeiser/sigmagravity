from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v14_gauge_carrier import (
    constant_curvature_riemann,
    curved_improvement_divergence,
    electric_weyl_tensor,
    fourth_order_propagator_residues,
    minimal_covariant_gauge_residual,
    partially_massless_gauge_residual,
    riemann_symmetry_residuals,
    tracefree_stress_double_divergence,
)

METRIC = np.diag([-1.0, 1.0, 1.0, 1.0])
GRADIENT = np.asarray([1.0, 0.2, -0.4, 0.6])


def test_minimal_covariant_field_strength_fails_on_curvature() -> None:
    flat = np.zeros((4, 4, 4, 4))
    curved = constant_curvature_riemann(METRIC, curvature=0.2)
    assert np.max(np.abs(minimal_covariant_gauge_residual(flat, GRADIENT))) == 0.0
    assert np.max(np.abs(minimal_covariant_gauge_residual(curved, GRADIENT))) > 0.0


def test_partially_massless_term_cancels_only_constant_curvature() -> None:
    curvature = 0.2
    background = constant_curvature_riemann(METRIC, curvature=curvature)
    constant_residual = partially_massless_gauge_residual(
        background,
        GRADIENT,
        metric=METRIC,
        curvature_counterterm=curvature,
    )
    assert np.max(np.abs(constant_residual)) < 1.0e-14

    weyl = electric_weyl_tensor(0.3 * np.diag([-2.0, 1.0, 1.0]))
    symmetries = riemann_symmetry_residuals(weyl, metric=METRIC)
    assert max(symmetries.values()) < 1.0e-14
    curved_residual = partially_massless_gauge_residual(
        background + weyl,
        GRADIENT,
        metric=METRIC,
        curvature_counterterm=curvature,
    )
    expected = minimal_covariant_gauge_residual(weyl, GRADIENT)
    assert np.max(np.abs(curved_residual - expected)) < 1.0e-14
    assert np.max(np.abs(curved_residual)) > 0.0


def test_obvious_neutral_sources_lose_conservation_on_general_backgrounds() -> None:
    tracefree_residual = tracefree_stress_double_divergence(
        wave_covector_squared=1.7,
        stress_trace=2.3,
    )
    assert tracefree_residual == pytest.approx(-1.7 * 2.3 / 4.0)
    ricci = np.diag([0.4, 0.2, -0.1, 0.3])
    curved_residual = curved_improvement_divergence(ricci, GRADIENT)
    assert np.max(np.abs(curved_residual)) > 0.0


def test_fourth_order_completion_has_an_opposite_residue_mode() -> None:
    residues = fourth_order_propagator_residues(massive_pole_squared=3.0)
    assert residues.massless == pytest.approx(1.0 / 3.0)
    assert residues.massive == pytest.approx(-1.0 / 3.0)
    assert residues.massless * residues.massive < 0.0


def test_invalid_gauge_carrier_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        constant_curvature_riemann(np.eye(3), curvature=1.0)
    with pytest.raises(ValueError):
        electric_weyl_tensor(np.eye(3))
    with pytest.raises(ValueError):
        minimal_covariant_gauge_residual(np.zeros((3, 3, 3, 3)), GRADIENT)
    with pytest.raises(ValueError):
        partially_massless_gauge_residual(
            np.zeros((4, 4, 4, 4)),
            np.ones(3),
            metric=METRIC,
            curvature_counterterm=0.0,
        )
    with pytest.raises(ValueError):
        tracefree_stress_double_divergence(
            wave_covector_squared=1.0,
            stress_trace=1.0,
            spacetime_dimension=1,
        )
    with pytest.raises(ValueError):
        fourth_order_propagator_residues(massive_pole_squared=0.0)
