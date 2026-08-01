import math

import numpy as np
import pytest

from voidscreen.basin_metric import (
    C_M_S,
    basin_metric_coefficients,
    beta_for_response_ratio,
    lensing_to_dynamics_extra_ratio,
    weak_field_potentials,
)


def test_pure_conformal_response_cancels_from_lensing() -> None:
    coefficients = basin_metric_coefficients(alpha=2.0, beta=0.0)
    assert coefficients.dynamics == 2.0
    assert coefficients.spatial_curvature == -2.0
    assert coefficients.weyl_half == 0.0
    assert lensing_to_dynamics_extra_ratio(2.0, 0.0) == 0.0


def test_pure_disformal_response_has_half_ratio() -> None:
    assert lensing_to_dynamics_extra_ratio(0.0, 3.0) == 0.5


def test_no_slip_limit_has_equal_dynamics_and_lensing_response() -> None:
    alpha = 0.4
    beta = 2.0 * alpha
    coefficients = basin_metric_coefficients(alpha, beta)
    assert coefficients.dynamics == pytest.approx(coefficients.spatial_curvature)
    assert lensing_to_dynamics_extra_ratio(alpha, beta) == pytest.approx(1.0)
    assert beta_for_response_ratio(alpha, 1.0) == pytest.approx(beta)


def test_screened_field_recovers_gr() -> None:
    newtonian = np.asarray([-1.0e10, -2.0e10])
    psi, phi, weyl = weak_field_potentials(
        newtonian, np.zeros(2), alpha=0.3, beta=0.9
    )
    assert np.array_equal(psi, newtonian)
    assert np.array_equal(phi, newtonian)
    assert np.array_equal(weyl, newtonian)


def test_dynamics_blind_limit_keeps_lensing_response() -> None:
    assert math.isinf(lensing_to_dynamics_extra_ratio(1.0, 1.0))
    psi, phi, weyl = weak_field_potentials(
        0.0, 1.0e-8, alpha=1.0, beta=1.0
    )
    assert psi == pytest.approx(0.0)
    assert phi == pytest.approx(-C_M_S**2 * 1.0e-8)
    assert weyl == pytest.approx(0.5 * phi)
