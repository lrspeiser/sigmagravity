from __future__ import annotations

import numpy as np

from voidscreen.required_field_decomposition import (
    angular_harmonics,
    convergence_and_jacobian_determinant,
    predictor_correlations,
    radial_vector_decomposition,
    sign_change_cells,
    vector_rms,
)


def grid():
    axis = np.linspace(-4.0, 4.0, 17)
    return np.meshgrid(axis, axis, indexing="ij")


def test_radial_decomposition_reconstructs_field_exactly():
    x, y = grid()
    alpha_x = 0.3 * x + 0.1 * y
    alpha_y = 0.3 * y - 0.1 * x
    result = radial_vector_decomposition(
        alpha_x,
        alpha_y,
        x,
        y,
        np.linspace(0.0, np.hypot(4.0, 4.0), 9),
    )
    assert np.allclose(result.monopole_x + result.angular_x, alpha_x)
    assert np.allclose(result.monopole_y + result.angular_y, alpha_y)
    assert result.table.samples.sum() == x.size


def test_quadratic_potential_has_expected_convergence_zero_curl_and_criticality():
    x, y = grid()
    alpha_x = 1.2 * x
    alpha_y = 0.4 * y
    convergence, curl, determinant = convergence_and_jacobian_determinant(
        alpha_x,
        alpha_y,
        0.5,
    )
    assert np.allclose(convergence, 0.8)
    assert np.allclose(curl, 0.0)
    assert np.allclose(determinant, -0.12)
    transition = determinant.copy()
    transition[:8] *= -1.0
    assert sign_change_cells(transition) > 0


def test_harmonics_and_correlations_identify_constructed_signal():
    x, y = grid()
    angle = np.arctan2(y, x)
    radial = 1.0 + 0.5 * np.cos(2.0 * angle)
    mask = np.hypot(x, y) > 1.0
    harmonics = angular_harmonics(radial, x, y, mask, [1, 2, 3])
    assert harmonics.set_index("mode").loc[2, "amplitude"] > 0.2
    correlations = predictor_correlations(
        {"aligned": radial, "reversed": -radial},
        {"required": radial},
        mask,
    ).set_index("predictor")
    assert correlations.loc["aligned", "spearman_rho"] > 0.99
    assert correlations.loc["reversed", "spearman_rho"] < -0.99
    assert vector_rms(radial, np.zeros_like(radial), mask) > 1.0
