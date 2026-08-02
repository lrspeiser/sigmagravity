from __future__ import annotations

import numpy as np

from voidscreen.mass_planarity import (
    baryonic_mass_planarity,
    planarity_blended_coherence,
)


def gaussian_density(axis: np.ndarray, widths: tuple[float, float, float]) -> np.ndarray:
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.exp(
        -0.5
        * (
            (x / widths[0]) ** 2
            + (y / widths[1]) ** 2
            + (z / widths[2]) ** 2
        )
    )


def test_planarity_distinguishes_sheet_filament_and_ball() -> None:
    axis = np.linspace(-6.0, 6.0, 49)
    spacing = float(axis[1] - axis[0])
    sheet = baryonic_mass_planarity(gaussian_density(axis, (2.0, 1.5, 0.12)), spacing)
    filament = baryonic_mass_planarity(gaussian_density(axis, (2.0, 0.2, 0.2)), spacing)
    ball = baryonic_mass_planarity(gaussian_density(axis, (1.0, 1.0, 1.0)), spacing)
    assert sheet.planarity > 0.95
    assert filament.planarity < 0.05
    assert ball.planarity < 1e-12


def test_planarity_is_rotation_and_translation_invariant() -> None:
    axis = np.linspace(-4.0, 4.0, 41)
    spacing = float(axis[1] - axis[0])
    density = gaussian_density(axis, (1.4, 0.8, 0.25))
    baseline = baryonic_mass_planarity(density, spacing)
    rotated = baryonic_mass_planarity(np.transpose(density, (1, 2, 0)), spacing)
    shifted = baryonic_mass_planarity(np.roll(density, (2, -1, 1), axis=(0, 1, 2)), spacing)
    assert abs(baseline.planarity - rotated.planarity) < 1e-12
    assert abs(baseline.planarity - shifted.planarity) < 2e-4


def test_planarity_blend_has_exact_endpoints_and_bounds() -> None:
    coherence = np.linspace(0.0, 1.0, 25).reshape(5, 5)
    assert np.array_equal(planarity_blended_coherence(coherence, 0.0), coherence)
    assert np.array_equal(planarity_blended_coherence(coherence, 1.0), np.ones_like(coherence))
    blended = planarity_blended_coherence(coherence, 0.4)
    assert np.min(blended) >= 0.4
    assert np.max(blended) <= 1.0
    assert np.allclose(blended, 0.4 + 0.6 * coherence)
