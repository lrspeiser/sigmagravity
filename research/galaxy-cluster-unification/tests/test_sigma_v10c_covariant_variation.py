from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v10c_covariant_variation import (
    audit_v10c_covariant_variation,
    carrier_kinematics,
    carrier_momentum_directional_error,
    carrier_spatial_constraint_rank,
)


def tilted_unit_aether(velocity: float) -> np.ndarray:
    gamma = 1.0 / np.sqrt(1.0 - velocity**2)
    return np.array([gamma, gamma * velocity, 0.0, 0.0])


def symmetric_derivative(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = rng.normal(size=(4, 4, 4))
    return 0.5 * (result + np.swapaxes(result, 1, 2))


def test_projected_carrier_momentum_matches_tilted_finite_difference() -> None:
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    error = carrier_momentum_directional_error(
        metric,
        tilted_unit_aether(0.42),
        symmetric_derivative(1),
        symmetric_derivative(2),
        carrier_speed_squared=3.0 / 11.0,
    )
    assert error < 1.0e-9


def test_projected_derivatives_are_spatial_on_tilted_background() -> None:
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    aether = tilted_unit_aether(0.31)
    result = carrier_kinematics(
        metric,
        aether,
        symmetric_derivative(3),
        carrier_speed_squared=3.0 / 11.0,
    )
    assert np.linalg.norm(
        np.einsum("m,mn->n", aether, result.time_derivative_covariant)
    ) < 1.0e-12
    assert np.linalg.norm(
        np.einsum("m,rmn->rn", aether, result.spatial_derivative_covariant)
    ) < 1.0e-12
    assert np.linalg.norm(
        np.einsum("r,rmn->mn", aether, result.spatial_derivative_covariant)
    ) < 1.0e-12


def test_spatiality_constraint_has_rank_four_and_leaves_six_components() -> None:
    result = carrier_spatial_constraint_rank(
        np.diag([-1.0, 1.0, 1.0, 1.0]), tilted_unit_aether(0.55)
    )
    assert result == {
        "symmetric_tensor_components": 10,
        "constraint_rank": 4,
        "spatial_carrier_components": 6,
    }


def test_covariant_variation_subgate_passes_without_claiming_adm_completion() -> None:
    report = audit_v10c_covariant_variation(carrier_speed_squared=3.0 / 11.0)
    assert all(report["gates"].values())
    assert report["all_covariant_variation_subgates_pass"] is True
    assert report["nonlinear_ADM_constraint_count_complete"] is False
    assert report["arbitrary_background_characteristics_complete"] is False


def test_invalid_carrier_variation_inputs_are_rejected() -> None:
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        carrier_kinematics(
            metric,
            np.array([1.0, 0.2, 0.0, 0.0]),
            symmetric_derivative(4),
            carrier_speed_squared=3.0 / 11.0,
        )
    with pytest.raises(ValueError):
        carrier_kinematics(
            metric,
            tilted_unit_aether(0.2),
            symmetric_derivative(5),
            carrier_speed_squared=0.0,
        )
