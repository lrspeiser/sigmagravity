from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v10d_exponential_kinetic import (
    audit_v10d_exponential_kinetic,
    completed_aether_kinetic_matrix,
    completed_channel_speed_squared,
    completed_eigenvalue,
    symmetric_matrix_exponential,
)


def test_exponential_minus_x_has_global_minimum_one() -> None:
    samples = np.linspace(-20.0, 20.0, 100_001)
    values = completed_eigenvalue(samples)
    assert np.min(values) == pytest.approx(1.0)
    assert samples[np.argmin(values)] == pytest.approx(0.0)


def test_completed_matrix_is_positive_for_large_mixed_sign_background() -> None:
    background = np.array(
        [[30.0, 4.0, -2.0], [4.0, -25.0, 3.0], [-2.0, 3.0, 10.0]]
    )
    result = completed_aether_kinetic_matrix(
        background, k_b=1.0, beta=np.sqrt(2.0 / 11.0)
    )
    assert np.min(np.linalg.eigvalsh(result)) >= 1.0 - 1.0e-10


def test_zero_background_cones_match_v10c_and_move_inward_with_factor() -> None:
    zero = completed_channel_speed_squared(
        kinetic_factor=1.0,
        base_spatial_stiffness=3.0 / 4.0,
        carrier_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=2.0 / 11.0,
    )
    completed = completed_channel_speed_squared(
        kinetic_factor=10.0,
        base_spatial_stiffness=3.0 / 4.0,
        carrier_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=2.0 / 11.0,
    )
    assert zero == pytest.approx([9.0 / 44.0, 1.0])
    assert np.all(completed > 0.0)
    assert np.max(completed) < 1.0


def test_v10d_selection_passes_but_retains_mandatory_gates() -> None:
    report = audit_v10d_exponential_kinetic(
        k_b=1.0,
        u=3.0 / 4.0,
        carrier_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=2.0 / 11.0,
        physical_parameter_count=5,
        maximum_physical_parameters=5,
    )
    assert all(report["selection_gates"].values())
    assert report["all_selection_gates_pass"] is True
    assert report["all_mandatory_theory_gates_pass"] is False


def test_invalid_exponential_completion_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        symmetric_matrix_exponential(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        completed_aether_kinetic_matrix(np.eye(3), k_b=0.0, beta=0.4)
