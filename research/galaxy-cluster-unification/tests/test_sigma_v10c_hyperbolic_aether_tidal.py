from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v10c_hyperbolic_aether_tidal import (
    audit_v10c_selection,
    cone_margin,
    mixed_hyperbolic_channel,
    retarded_source_structure,
    static_principal_channel,
    v10c_derived_coefficients,
)


def test_derived_coefficients_are_exact_selected_rationals() -> None:
    coefficients = v10c_derived_coefficients(
        maximum_sourced_base_speed_squared=3.0 / 4.0,
        static_mixing_fraction=2.0 / 3.0,
        k_b=1.0,
    )
    assert coefficients["carrier_speed_squared"] == pytest.approx(3.0 / 11.0)
    assert coefficients["normalized_mixing_beta_squared_over_KB"] == pytest.approx(
        2.0 / 11.0
    )
    assert coefficients["mixing_beta"] == pytest.approx(np.sqrt(2.0 / 11.0))
    assert coefficients["aether_vector_speed_squared_after_counterterm"] == pytest.approx(
        3.0 / 4.0
    )
    assert coefficients["longitudinal_static_capacity"] == pytest.approx(3.0)


def test_longitudinal_characteristic_has_exact_luminal_upper_root() -> None:
    channel = mixed_hyperbolic_channel(
        base_speed_squared=3.0 / 4.0,
        carrier_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=2.0 / 11.0,
    )
    assert channel.speed_squared == pytest.approx([9.0 / 44.0, 1.0])
    assert channel.positive
    assert channel.causal
    margin = cone_margin(
        base_speed_squared=3.0 / 4.0,
        carrier_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=2.0 / 11.0,
    )
    assert margin["luminal_upper_cone_margin"] == pytest.approx(0.0, abs=1.0e-15)
    assert margin["positive_gradient_margin"] == pytest.approx(1.0 / 44.0)


def test_transverse_characteristics_are_strictly_inside_metric_cone() -> None:
    channel = mixed_hyperbolic_channel(
        base_speed_squared=3.0 / 4.0,
        carrier_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=1.0 / 11.0,
    )
    expected = np.array([(49.0 - np.sqrt(817.0)) / 88.0, (49.0 + np.sqrt(817.0)) / 88.0])
    assert channel.speed_squared == pytest.approx(expected)
    assert channel.speed_squared[-1] < 1.0
    assert channel.positive
    assert channel.causal


def test_static_block_is_positive_with_threefold_capacity() -> None:
    channel = static_principal_channel(
        k_b=1.0,
        carrier_spatial_stiffness=3.0 / 11.0,
        mixing_beta=np.sqrt(2.0 / 11.0),
        canonical_factor=1.0,
    )
    assert channel["determinant"] == pytest.approx(1.0 / 11.0)
    assert channel["aether_schur_complement"] == pytest.approx(1.0 / 3.0)
    assert channel["static_response_capacity"] == pytest.approx(3.0)
    assert channel["eigenvalues"] == pytest.approx(
        [(7.0 - np.sqrt(38.0)) / 11.0, (7.0 + np.sqrt(38.0)) / 11.0]
    )
    assert channel["positive"]


def test_hyperbolic_retarded_completion_removes_auxiliary_equal_time_tail() -> None:
    structure = retarded_source_structure()
    assert structure["carrier_has_time_kinetic_term"]
    assert structure["finite_front_set_by_principal_cone"]
    assert not structure["equal_preferred_time_yukawa_constraint_tail"]
    assert structure["static_zero_boundary_solution_unique_from_strict_convexity"]
    assert not structure["object_specific_homogeneous_static_profile_allowed"]
    assert structure["free_carrier_waves_exist"]
    assert not structure["nonlinear_global_well_posedness_proved"]


def test_v10c_passes_selection_only() -> None:
    audit = audit_v10c_selection(
        maximum_sourced_base_speed_squared=0.75,
        static_mixing_fraction=2.0 / 3.0,
        k_b=1.0,
        existing_cluster_amplification_target=3.14465,
        physical_parameter_count=5,
        maximum_physical_parameters=5,
    )
    assert audit["all_selection_gates_pass"]
    assert all(audit["selection_gates"].values())
    assert audit["hyperbolic_channels"]["longitudinal"]["speed_squared"] == pytest.approx(
        [9.0 / 44.0, 1.0]
    )
    assert audit["response"]["longitudinal_capacity"] == pytest.approx(3.0)
    assert audit["response"]["gap_closure_fraction"] > 0.93
    assert not audit["all_mandatory_theory_gates_pass"]
    assert not any(audit["unresolved_mandatory_gates"].values())


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            v10c_derived_coefficients,
            {
                "maximum_sourced_base_speed_squared": 1.0,
                "static_mixing_fraction": 2.0 / 3.0,
                "k_b": 1.0,
            },
        ),
        (
            mixed_hyperbolic_channel,
            {
                "base_speed_squared": -1.0,
                "carrier_speed_squared": 0.2,
                "normalized_mixing_squared": 0.1,
            },
        ),
    ],
)
def test_invalid_inputs_are_rejected(function, kwargs) -> None:
    with pytest.raises(ValueError):
        function(**kwargs)
