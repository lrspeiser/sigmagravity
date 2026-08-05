from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_source_invariants import (
    anisotropic_stress,
    axial_orientation_deg,
    component_overlap,
    projected_baroclinicity,
    relative_current,
    thermodynamic_gradient_stress,
)


def test_component_overlap_has_expected_limits_and_species_symmetry() -> None:
    gas = np.array([0.0, 1.0, 1.0, 3.0])
    stars = np.array([0.0, 0.0, 1.0, 1.0])
    result = component_overlap(gas, stars)
    assert result == pytest.approx([0.0, 0.0, 1.0, 0.75])
    assert component_overlap(stars, gas) == pytest.approx(result)


def test_relative_current_is_common_boost_invariant_and_rotation_covariant() -> None:
    gas = np.array([[3.0, 4.0], [2.0, -1.0]])
    stars = np.array([[0.0, 0.0], [-1.0, -1.0]])
    boost = np.array([17.0, -9.0])
    vector, norm = relative_current(gas, stars, 4.0, 3.0)
    boosted_vector, boosted_norm = relative_current(
        gas + boost, stars + boost, 4.0, 3.0
    )
    assert boosted_vector == pytest.approx(vector)
    assert boosted_norm == pytest.approx(norm)
    assert norm == pytest.approx([1.0, 0.36])

    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated_vector, rotated_norm = relative_current(
        gas @ rotation.T, stars @ rotation.T, 4.0, 3.0
    )
    assert rotated_vector == pytest.approx(vector @ rotation.T)
    assert rotated_norm == pytest.approx(norm)


def test_anisotropic_stress_is_symmetric_trace_free_and_rotation_invariant() -> None:
    stress = np.array([[4.0, 2.0], [0.0, 2.0]])
    trace_free, norm = anisotropic_stress(stress, 2.0)
    np.testing.assert_allclose(trace_free, [[1.0, 1.0], [1.0, -1.0]])
    assert np.trace(trace_free) == pytest.approx(0.0, abs=1e-14)
    assert norm == pytest.approx(1.0)

    angle = np.radians(31.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated, rotated_norm = anisotropic_stress(rotation @ stress @ rotation.T, 2.0)
    assert rotated == pytest.approx(rotation @ trace_free @ rotation.T)
    assert rotated_norm == pytest.approx(norm)


def test_gradient_stress_is_trace_free_and_recovers_a_manufactured_axis() -> None:
    _y, x = np.mgrid[-2:3, -2:3]
    density = np.exp(0.4 * x)
    entropy = np.exp(0.7 * x)
    tensor = thermodynamic_gradient_stress(density, entropy)
    assert np.max(np.abs(np.trace(tensor, axis1=-2, axis2=-1))) < 1e-14
    assert axial_orientation_deg(tensor) == pytest.approx(90.0)


def test_projected_baroclinicity_separates_parallel_and_orthogonal_gradients() -> None:
    y, x = np.mgrid[-2:3, -2:3]
    density = np.exp(0.3 * x)
    parallel_pressure = np.exp(0.8 * x)
    orthogonal_pressure = np.exp(0.8 * y)
    signed_parallel, squared_parallel = projected_baroclinicity(
        density, parallel_pressure
    )
    signed_orthogonal, squared_orthogonal = projected_baroclinicity(
        density, orthogonal_pressure
    )
    assert np.max(np.abs(signed_parallel)) < 1e-14
    assert np.max(np.abs(squared_parallel)) < 1e-14
    assert np.abs(signed_orthogonal) == pytest.approx(np.ones_like(signed_orthogonal))
    assert squared_orthogonal == pytest.approx(np.ones_like(squared_orthogonal))


def test_invalid_or_unidentifiable_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        component_overlap([-1.0], [2.0])
    with pytest.raises(ValueError, match="positive"):
        relative_current([[1.0, 0.0]], [[0.0, 0.0]], 0.0, 0.0)
    with pytest.raises(ValueError, match="isotropic"):
        axial_orientation_deg(np.eye(2))
