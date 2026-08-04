from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v10b_auxiliary_aether_tidal import (
    audit_v10b_constraint_causality,
    audit_v10b_selection,
    auxiliary_dirac_channel,
    auxiliary_vector_speed_squared,
    gap_closure_fraction,
    instantaneous_acceleration_kernel,
    instantaneous_tail_at_radius,
    k_length_for_static_amplification,
    source_operator_norm_bound,
    static_linear_metric_structure,
    static_principal_channel,
    static_response_amplification,
    static_response_capacity,
    v10b_fixed_coefficients,
)


def test_fixed_coefficients_leave_exact_positive_static_margin() -> None:
    coefficients = v10b_fixed_coefficients(1.0)
    beta = coefficients["mixing_beta"]
    assert beta == pytest.approx(np.sqrt(2.0 / 3.0))
    longitudinal = static_principal_channel(
        k_b=1.0,
        mixing_beta=beta,
        channel="longitudinal",
    )
    assert longitudinal.eigenvalues == pytest.approx(
        [1.0 - np.sqrt(2.0 / 3.0), 1.0 + np.sqrt(2.0 / 3.0)]
    )
    assert longitudinal.determinant == pytest.approx(1.0 / 3.0)
    assert longitudinal.positive


def test_transverse_and_unmixed_static_channels_are_positive() -> None:
    beta = v10b_fixed_coefficients(1.0)["mixing_beta"]
    transverse = static_principal_channel(
        k_b=1.0,
        mixing_beta=beta,
        channel="transverse",
    )
    unmixed = static_principal_channel(
        k_b=1.0,
        mixing_beta=beta,
        channel="unmixed",
    )
    assert transverse.canonical_mixing == pytest.approx(1.0 / np.sqrt(3.0))
    assert transverse.eigenvalues == pytest.approx(
        [1.0 - 1.0 / np.sqrt(3.0), 1.0 + 1.0 / np.sqrt(3.0)]
    )
    assert transverse.positive
    assert unmixed.eigenvalues == pytest.approx([1.0, 1.0])
    assert unmixed.positive


def test_auxiliary_static_response_is_bounded_by_declared_capacities() -> None:
    beta = v10b_fixed_coefficients(1.0)["mixing_beta"]
    assert static_response_capacity(
        k_b=1.0, mixing_beta=beta, channel="longitudinal"
    ) == pytest.approx(3.0)
    assert static_response_capacity(
        k_b=1.0, mixing_beta=beta, channel="transverse"
    ) == pytest.approx(1.5)
    assert static_response_amplification(
        0.0, k_b=1.0, mixing_beta=beta, channel="longitudinal"
    ) == pytest.approx(1.0)
    assert static_response_amplification(
        1.0e8, k_b=1.0, mixing_beta=beta, channel="longitudinal"
    ) == pytest.approx(3.0)


def test_longitudinal_capacity_closes_more_than_75_percent_of_spent_gap() -> None:
    closure = gap_closure_fraction(3.0, 3.14465)
    assert closure == pytest.approx(0.9326, rel=1.0e-4)
    target = 1.0 + 0.75 * (3.14465 - 1.0)
    scale = k_length_for_static_amplification(
        target,
        k_b=1.0,
        mixing_beta=np.sqrt(2.0 / 3.0),
        channel="longitudinal",
    )
    assert scale == pytest.approx(3.51, rel=2.0e-3)


@pytest.mark.parametrize("channel, expected", [("longitudinal", 0.6), ("transverse", 0.75)])
def test_auxiliary_constraint_lowers_vector_speed_without_new_root(
    channel: str, expected: float
) -> None:
    beta = v10b_fixed_coefficients(1.0)["mixing_beta"]
    assert auxiliary_vector_speed_squared(
        0.0, k_b=1.0, mixing_beta=beta, channel=channel
    ) == pytest.approx(1.0)
    assert auxiliary_vector_speed_squared(
        1.0e8, k_b=1.0, mixing_beta=beta, channel=channel
    ) == pytest.approx(expected)


def test_divergence_source_operator_has_unit_norm_bound() -> None:
    saturating = source_operator_norm_bound(
        np.diag([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    assert saturating["operator_ratio"] == pytest.approx(1.0)
    assert saturating["bound_satisfied"]
    rng = np.random.default_rng(20)
    for _ in range(100):
        raw = rng.normal(size=(3, 3))
        tensor = 0.5 * (raw + raw.T)
        result = source_operator_norm_bound(tensor, rng.normal(size=3))
        assert result["bound_satisfied"]


def test_linear_static_metric_channel_is_dynamical_and_weyl_active() -> None:
    structure = static_linear_metric_structure()
    assert structure["linear_lapse_equation_correction"] != 0.0
    assert structure["linear_spatial_traceless_equation_correction"] == pytest.approx(0.0)
    assert structure["base_no_slip_relation_retained_at_linear_static_order"]
    assert structure["delta_Psi_equals_delta_Phi_equals_delta_Weyl"]
    assert structure["flat_TT_source"] == pytest.approx(0.0)
    assert not structure["photon_only_rule"]
    assert not structure["nonlinear_metric_variation_complete"]


def test_v10b_passes_selection_but_not_full_theory_gates() -> None:
    audit = audit_v10b_selection(
        k_b=1.0,
        existing_cluster_amplification_target=3.14465,
        physical_parameter_count=5,
        maximum_physical_parameters=5,
    )
    assert audit["all_selection_gates_pass"]
    assert all(audit["selection_gates"].values())
    assert audit["response"]["capacities"]["longitudinal"] == pytest.approx(3.0)
    assert audit["static_channels"]["longitudinal"]["determinant"] == pytest.approx(
        1.0 / 3.0
    )
    assert not audit["all_mandatory_theory_gates_pass"]
    assert not any(audit["unresolved_mandatory_gates"].values())


def test_auxiliary_dirac_pair_removes_carrier_with_positive_hamiltonian() -> None:
    beta = v10b_fixed_coefficients(1.0)["mixing_beta"]
    channel = auxiliary_dirac_channel(
        2.0,
        1.0,
        k_b=1.0,
        mixing_beta=beta,
        channel="longitudinal",
    )
    assert channel.secondary_bracket == pytest.approx(5.0 + 8.0 / 3.0)
    assert channel.reduced_hamiltonian_momentum_coefficient > 0.0
    assert channel.primary_constraints == 1
    assert channel.secondary_constraints == 1
    assert channel.second_class_constraints == 2
    assert channel.auxiliary_configuration_dof == pytest.approx(0.0)
    assert channel.positive


def test_finite_range_auxiliary_constraint_has_instantaneous_physical_tail() -> None:
    beta = v10b_fixed_coefficients(1.0)["mixing_beta"]
    longitudinal = instantaneous_acceleration_kernel(
        1.0,
        k_b=1.0,
        mixing_beta=beta,
        channel="longitudinal",
    )
    transverse = instantaneous_acceleration_kernel(
        1.0,
        k_b=1.0,
        mixing_beta=beta,
        channel="transverse",
    )
    assert longitudinal["local_delta_coefficient"] == pytest.approx(3.0 / 5.0)
    assert longitudinal["effective_inverse_range"] == pytest.approx(np.sqrt(3.0 / 5.0))
    assert longitudinal["yukawa_tail_coefficient"] == pytest.approx(6.0 / 25.0)
    assert transverse["local_delta_coefficient"] == pytest.approx(3.0 / 4.0)
    assert transverse["effective_inverse_range"] == pytest.approx(np.sqrt(3.0 / 4.0))
    assert transverse["yukawa_tail_coefficient"] == pytest.approx(3.0 / 16.0)
    assert longitudinal["equal_time_nonlocal_tail_present"]
    assert transverse["equal_time_nonlocal_tail_present"]
    assert not longitudinal["finite_light_cone_front"]
    assert not transverse["finite_light_cone_front"]
    assert instantaneous_tail_at_radius(
        10.0,
        1.0,
        k_b=1.0,
        mixing_beta=beta,
        channel="transverse",
    ) > 0.0


def test_v10b_constraint_passes_but_causality_gate_fails() -> None:
    audit = audit_v10b_constraint_causality(
        k_b=1.0,
        inverse_length=1.0,
        wave_numbers=np.array([0.0, 0.1, 1.0, 10.0]),
        radii=np.array([0.1, 1.0, 10.0]),
    )
    assert audit["all_constraint_gates_pass"]
    assert all(audit["constraint_gates"].values())
    assert not audit["all_causality_gates_pass"]
    assert not any(audit["causality_gates"].values())
    assert not audit["exact_v10b_survives"]


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (v10b_fixed_coefficients, (0.0,)),
        (source_operator_norm_bound, (np.zeros((2, 2)), np.ones(3))),
        (gap_closure_fraction, (0.5, 3.0)),
    ],
)
def test_invalid_inputs_are_rejected(function, arguments) -> None:
    with pytest.raises(ValueError):
        function(*arguments)
