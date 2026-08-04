from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v10a_spatial_polarization import (
    asymptotic_response_capacity,
    audit_v10a_selection,
    carrier_coefficients,
    constant_mixing_ellipticity_thresholds,
    equal_acceleration_geometry_pair,
    k_length_for_amplification,
    local_algebraic_carrier_response,
    mixed_scalar_carrier_spectrum,
    naive_aqual_schur_diagnostic,
    nonlinear_additivity_error,
    point_mass_tidal_hessian,
    potential_convexity_spectrum,
    rotation_covariance_error,
    simple_aqual_static_stiffnesses,
    static_high_k_mixed_spectrum,
    trace_stf_decomposition,
)


def test_selected_flat_mixed_spectrum_is_positive_and_subluminal() -> None:
    coefficients = carrier_coefficients(0.75)
    assert coefficients["carrier_speed_squared"] == pytest.approx(0.25)
    assert coefficients["mixing_beta"] == pytest.approx(0.375)
    spectrum = mixed_scalar_carrier_spectrum(
        base_scalar_speed_squared=0.75,
        carrier_speed_squared=0.25,
        mixing_beta=0.375,
    )
    assert spectrum.determinant == pytest.approx(0.046875)
    assert spectrum.speed_squared == pytest.approx([0.049306090567, 0.950693909433])
    assert spectrum.positive
    assert spectrum.causal


def test_selected_flat_response_capacity_brackets_cluster_target() -> None:
    capacity = asymptotic_response_capacity(
        base_scalar_stiffness=0.75,
        carrier_speed_squared=0.25,
        mixing_beta=0.375,
    )
    assert capacity == pytest.approx(4.0)
    k_length = k_length_for_amplification(
        3.14465,
        base_scalar_stiffness=0.75,
        carrier_speed_squared=0.25,
        mixing_beta=0.375,
    )
    assert k_length == pytest.approx(6.333, rel=2.0e-4)


@pytest.mark.parametrize("magnitude", [0.0, 0.1, 1.0, 10.0, 100.0])
def test_carrier_potential_is_strictly_convex(magnitude: float) -> None:
    vector = np.array([magnitude, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = potential_convexity_spectrum(vector)
    assert result["strictly_convex"]
    assert np.asarray(result["eigenvalues"])[:5] == pytest.approx(1.0 + magnitude**2)
    assert np.asarray(result["eigenvalues"])[5] == pytest.approx(1.0 + 3.0 * magnitude**2)


def test_local_response_is_unique_parallel_and_saturating() -> None:
    source = np.diag([3.0, 0.0, 0.0])
    response = local_algebraic_carrier_response(source)
    magnitude = float(np.linalg.norm(response))
    assert (1.0 + magnitude**2) * response == pytest.approx(source)
    assert response[1:, :] == pytest.approx(0.0)
    weak = local_algebraic_carrier_response(1.0e-4 * source)
    strong = local_algebraic_carrier_response(1.0e4 * source)
    assert np.linalg.norm(weak) / np.linalg.norm(1.0e-4 * source) > 0.999
    assert np.linalg.norm(strong) / np.linalg.norm(1.0e4 * source) < 0.01


def test_trace_and_shear_channels_are_both_nonzero() -> None:
    isotropic = trace_stf_decomposition(local_algebraic_carrier_response(np.eye(3)))
    tidal = trace_stf_decomposition(
        local_algebraic_carrier_response(np.diag([2.0, -1.0, -1.0]))
    )
    assert isotropic["trace"] != pytest.approx(0.0)
    assert isotropic["stf_norm"] == pytest.approx(0.0, abs=1.0e-14)
    assert tidal["trace"] == pytest.approx(0.0, abs=1.0e-14)
    assert tidal["stf_norm"] > 0.0


def test_spherical_exterior_has_nonzero_tidal_carrier_source() -> None:
    hessian = point_mass_tidal_hessian(1.0, 2.0)
    assert np.trace(hessian) == pytest.approx(0.0)
    assert np.linalg.norm(hessian) == pytest.approx(np.sqrt(6.0) / 8.0)
    response = local_algebraic_carrier_response(hessian)
    assert np.linalg.norm(response) > 0.0


def test_equal_force_pair_is_distinguished_by_curvature() -> None:
    pair = equal_acceleration_geometry_pair(mass_ratio=100.0, radius_ratio=10.0)
    assert pair["surface_acceleration_ratio"] == pytest.approx(1.0)
    assert pair["tidal_hessian_norm_ratio"] == pytest.approx(0.1)


def test_nonlinear_response_is_rotation_covariant_but_nonadditive() -> None:
    angle = 0.61
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    first = np.diag([2.0, -1.0, -1.0])
    second = rotation @ first @ rotation.T
    assert rotation_covariance_error(first, rotation) < 1.0e-14
    assert nonlinear_additivity_error(first, second) > 0.1


def test_deep_aqual_proxy_warning_is_not_hidden_by_flat_pass() -> None:
    deep = naive_aqual_schur_diagnostic(
        0.1,
        carrier_speed_squared=0.25,
        mixing_beta=0.375,
    )
    high = naive_aqual_schur_diagnostic(
        10.0,
        carrier_speed_squared=0.25,
        mixing_beta=0.375,
    )
    assert not deep["proxy_elliptic"]
    assert high["proxy_elliptic"]


def test_exact_static_stiffness_has_transverse_and_longitudinal_channels() -> None:
    stiffnesses = simple_aqual_static_stiffnesses(1.0)
    assert stiffnesses["transverse"] == pytest.approx(0.5)
    assert stiffnesses["longitudinal"] == pytest.approx(0.75)
    assert simple_aqual_static_stiffnesses(0.0) == {
        "transverse": 0.0,
        "longitudinal": 0.0,
    }


def test_constant_mixing_fails_exact_high_k_quasistatic_gate() -> None:
    zero = static_high_k_mixed_spectrum(
        0.0,
        propagation_cosine=0.0,
        carrier_speed_squared=0.25,
        mixing_beta=0.375,
    )
    assert zero["gradient_eigenvalues"] == pytest.approx(
        [-0.270284707521, 0.520284707521]
    )
    assert zero["determinant"] == pytest.approx(-0.140625)
    assert not zero["elliptic"]

    thresholds = constant_mixing_ellipticity_thresholds(
        carrier_speed_squared=0.25,
        mixing_beta=0.375,
    )
    assert thresholds["required_AQUAL_stiffness"] == pytest.approx(0.5625)
    assert thresholds["transverse_acceleration_ratio_threshold"] == pytest.approx(
        9.0 / 7.0
    )
    assert thresholds["longitudinal_acceleration_ratio_threshold"] == pytest.approx(
        4.0 / np.sqrt(7.0) - 1.0
    )
    assert not thresholds["globally_elliptic_for_all_nonnegative_accelerations"]


def test_only_zero_constant_mixing_is_globally_elliptic_in_simple_mu_limit() -> None:
    thresholds = constant_mixing_ellipticity_thresholds(
        carrier_speed_squared=0.25,
        mixing_beta=0.0,
    )
    assert thresholds["globally_elliptic_for_all_nonnegative_accelerations"]
    assert thresholds["carrier_decoupled"]


def test_v10a_passes_selection_only_and_retains_mandatory_blockers() -> None:
    audit = audit_v10a_selection(
        k_b=1.0,
        k_2=2.0,
        lambda_s=1.0,
        physical_parameter_count=5,
        maximum_physical_parameters=5,
        existing_cluster_amplification_target=3.14465,
    )
    assert audit["all_selection_gates_pass"]
    assert all(audit["selection_gates"].values())
    assert not audit["all_mandatory_theory_gates_pass"]
    assert not any(audit["unresolved_mandatory_gates"].values())
    assert any(
        not row["proxy_elliptic"] for row in audit["deep_AQUAL_decoupled_proxy_warning"]
    )


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (carrier_coefficients, (1.0,)),
        (local_algebraic_carrier_response, (np.zeros((2, 2)),)),
        (point_mass_tidal_hessian, (-1.0, 1.0)),
        (potential_convexity_spectrum, (np.zeros(5),)),
    ],
)
def test_invalid_inputs_are_rejected(function, arguments) -> None:
    with pytest.raises(ValueError):
        function(*arguments)
