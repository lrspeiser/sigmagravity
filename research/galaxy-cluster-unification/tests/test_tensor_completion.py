import numpy as np
import pytest

from voidscreen.tensor_completion import (
    axisymmetric_tidal_eigenvalues,
    predict_tensor_acceleration,
    spherical_tidal_eigenvalues,
    tensor_completion,
)


def test_spherical_poisson_closure_has_expected_trace():
    eigenvalues = spherical_tidal_eigenvalues([1.0e-10], [10.0], [2.0e-25])
    expected = 4.0 * np.pi * 6.67430e-11 * 2.0e-22
    assert np.sum(eigenvalues[0]) == pytest.approx(expected)


def test_spherical_tidal_shape_is_only_local_to_mean_density_ratio():
    gbar = np.array([2.0e-11, 7.0e-11, 1.0e-10])
    radius_kpc = np.array([4.0, 20.0, 80.0])
    density_g_cm3 = np.array([8.0e-25, 3.0e-26, 2.0e-27])
    eigenvalues = spherical_tidal_eigenvalues(
        gbar, radius_kpc, density_g_cm3
    )
    radius_m = radius_kpc * 3.085677581491367e19
    mean_density_kg_m3 = (
        3.0 * gbar / (4.0 * np.pi * 6.67430e-11 * radius_m)
    )
    density_ratio = density_g_cm3 * 1000.0 / mean_density_kg_m3
    tangential = gbar / radius_m
    normalized = eigenvalues / tangential[:, None]
    expected = np.column_stack(
        [3.0 * density_ratio - 2.0, np.ones(3), np.ones(3)]
    )
    assert normalized == pytest.approx(expected)


def test_completion_tensor_and_projection_are_bounded():
    result = tensor_completion(
        [[-2.0e-31, 1.0e-31, 1.0e-31]],
        [[1.0, 1.0, 0.0]],
        "tensor_dominance",
        [0.1, -30.0, 2.0, 2.0],
    )
    assert np.all(result["completion_tensor_eigenvalues"] >= 0.1)
    assert np.all(result["completion_tensor_eigenvalues"] <= 1.0)
    assert 0.1 <= result["projected_completion_fraction"][0] <= 1.0


def test_dominance_model_recovers_more_along_dominant_direction():
    radial = tensor_completion(
        [[-4.0e-32, 1.0e-32, 2.0e-32]],
        [[1.0, 0.0, 0.0]],
        "tensor_dominance",
        [0.2, -30.0, 2.0, 2.0],
    )
    transverse = tensor_completion(
        [[-4.0e-32, 1.0e-32, 2.0e-32]],
        [[0.0, 1.0, 0.0]],
        "tensor_dominance",
        [0.2, -30.0, 2.0, 2.0],
    )
    assert radial["projected_completion_fraction"][0] > transverse[
        "projected_completion_fraction"
    ][0]


def test_isotropic_tensor_does_not_depend_on_direction():
    first = tensor_completion(
        [[-2.0e-32, 1.0e-32, 1.0e-32]],
        [[1.0, 0.0, 0.0]],
        "tensor_isotropic",
        [0.1, -30.0, 2.0],
    )
    second = tensor_completion(
        [[-2.0e-32, 1.0e-32, 1.0e-32]],
        [[0.0, 0.0, 1.0]],
        "tensor_isotropic",
        [0.1, -30.0, 2.0],
    )
    assert first["projected_completion_fraction"] == pytest.approx(
        second["projected_completion_fraction"]
    )


def test_axisymmetric_reconstruction_and_acceleration_are_finite():
    radius = np.asarray([1.0, 2.0, 4.0])
    gbar = np.asarray([2.0e-10, 1.5e-10, 1.0e-10])
    density = np.asarray([1.0e-24, 5.0e-25, 2.0e-25])
    eigenvalues = axisymmetric_tidal_eigenvalues(gbar, radius, density)
    result = predict_tensor_acceleration(
        gbar,
        eigenvalues,
        "tensor_alignment",
        [0.1, -30.0, 1.0, 2.0],
    )
    assert eigenvalues.shape == (3, 3)
    assert np.all(np.isfinite(result["predicted_acceleration_m_s2"]))
    assert np.all(result["predicted_acceleration_m_s2"] >= gbar)
