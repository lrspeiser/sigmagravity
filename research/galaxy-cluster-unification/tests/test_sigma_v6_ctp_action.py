from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v6_ctp_action import (
    advanced_impulse_response,
    constraint_pair_kinetic_hessian,
    kinetic_signature,
    localized_initial_data_count,
    retarded_impulse_response,
)


def test_minimal_v6d_localization_has_equal_positive_and_negative_directions() -> None:
    hessian = constraint_pair_kinetic_hessian([1, 5])
    signature = kinetic_signature(hessian)
    assert hessian.shape == (12, 12)
    assert signature["positive"] == 6
    assert signature["negative"] == 6
    assert signature["null"] == 0
    assert signature["rank"] == 12


def test_changing_constraint_sign_does_not_remove_negative_directions() -> None:
    positive = kinetic_signature(constraint_pair_kinetic_hessian([1, 5], 2.0))
    negative = kinetic_signature(constraint_pair_kinetic_hessian([1, 5], -2.0))
    assert positive["positive"] == negative["positive"] == 6
    assert positive["negative"] == negative["negative"] == 6


def test_localization_doubles_response_fields_and_requires_cauchy_data() -> None:
    count = localized_initial_data_count([1, 5])
    assert count == {
        "desired_retarded_response_components": 6,
        "localized_configuration_components": 12,
        "localized_second_order_initial_data": 24,
        "extra_multiplier_configuration_components": 6,
    }


def test_retarded_and_advanced_boundary_choices_have_opposite_support() -> None:
    time = np.linspace(-5.0, 5.0, 2001)
    retarded = retarded_impulse_response(time, 0.0, 1.3)
    advanced = advanced_impulse_response(time, 0.0, 1.3)
    assert np.all(retarded[time < 0.0] == 0.0)
    assert np.max(np.abs(retarded[time > 0.0])) > 0.5
    assert np.all(advanced[time > 0.0] == 0.0)
    assert np.max(np.abs(advanced[time < 0.0])) > 0.5


def test_invalid_ctp_audit_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        constraint_pair_kinetic_hessian([])
    with pytest.raises(ValueError):
        constraint_pair_kinetic_hessian([1, 0])
    with pytest.raises(ValueError):
        constraint_pair_kinetic_hessian([1], 0.0)
    with pytest.raises(ValueError):
        kinetic_signature(np.ones((2, 3)))
    with pytest.raises(ValueError):
        localized_initial_data_count([-1])
    with pytest.raises(ValueError):
        retarded_impulse_response([0.0], 0.0, 0.0)
    with pytest.raises(ValueError):
        advanced_impulse_response([np.nan], 0.0, 1.0)
