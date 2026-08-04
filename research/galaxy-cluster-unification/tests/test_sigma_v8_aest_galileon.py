from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v8_aest_galileon import (
    aest_linear_spectrum,
    aest_metric_projection,
    audit_v8a_selection,
    cubic_galileon_eom,
    simple_aqual_free_function,
    simple_mu,
)


def test_fixed_aqual_function_derivative_is_simple_mu() -> None:
    y = np.geomspace(1.0e-6, 1.0e6, 1000)
    step = 1.0e-5 * y
    derivative = (
        simple_aqual_free_function(y + step)
        - simple_aqual_free_function(y - step)
    ) / (2.0 * step)
    assert np.allclose(derivative, simple_mu(np.sqrt(y)), rtol=2.0e-6, atol=2.0e-8)
    assert simple_aqual_free_function(0.0) == pytest.approx(0.0)


def test_aest_scalar_changes_dynamics_and_weyl_with_same_sign() -> None:
    projection = aest_metric_projection(2.0, 3.0)
    assert float(projection.psi) == pytest.approx(5.0)
    assert float(projection.phi) == pytest.approx(5.0)
    assert float(projection.weyl) == pytest.approx(5.0)


def test_cubic_galileon_discriminates_equal_trace_hessians() -> None:
    isotropic = np.eye(3)
    rank_one = np.diag([3.0, 0.0, 0.0])
    assert np.trace(isotropic) == pytest.approx(np.trace(rank_one))
    assert float(cubic_galileon_eom(isotropic)) == pytest.approx(6.0)
    assert float(cubic_galileon_eom(rank_one)) == pytest.approx(0.0)
    assert float(cubic_galileon_eom(isotropic, 2.0)) == pytest.approx(24.0)


def test_selected_aest_base_has_positive_subluminal_propagating_modes() -> None:
    spectrum = aest_linear_spectrum(k_b=1.0, k_2=2.0, lambda_s=1.0)
    assert spectrum["tensor_speed_squared"] == pytest.approx(1.0)
    assert spectrum["vector_speed_squared"] == pytest.approx(1.0)
    assert spectrum["scalar_speed_squared"] == pytest.approx(0.75)
    assert spectrum["positive_propagating_modes"]
    assert spectrum["causal_propagating_modes"]


def test_selected_aest_base_does_not_overclaim_zero_frequency_health() -> None:
    spectrum = aest_linear_spectrum(k_b=1.0, k_2=2.0, lambda_s=1.0)
    assert spectrum["zero_frequency_constant_mode_hamiltonian"] == pytest.approx(0.0)
    assert spectrum["zero_frequency_linearly_growing_mode_present"]
    assert spectrum["ir_jeans_like_sector_present"]
    assert not spectrum["zero_frequency_sector_positive_all_momenta"]
    assert not spectrum["complete_flat_hamiltonian_positive_all_momenta"]


def test_v8a_passes_selection_but_not_unchecked_nonlinear_gates() -> None:
    audit = audit_v8a_selection(
        k_b=1.0,
        k_2=2.0,
        lambda_s=1.0,
        physical_parameter_count=5,
        maximum_physical_parameters=5,
    )
    assert all(audit["gates"].values())
    assert audit["metric_projection"]["scalar_delta_weyl"] == pytest.approx(1.0)
    assert audit["geometry_stress_test"]["response_difference"] == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (simple_mu, (-1.0,)),
        (simple_aqual_free_function, (-1.0,)),
        (aest_metric_projection, (np.nan, 0.0)),
        (cubic_galileon_eom, (np.zeros((2, 2)),)),
        (cubic_galileon_eom, (np.ones((3, 3)), -1.0)),
    ],
)
def test_invalid_v8a_inputs_are_rejected(function, arguments) -> None:
    with pytest.raises(ValueError):
        function(*arguments)
