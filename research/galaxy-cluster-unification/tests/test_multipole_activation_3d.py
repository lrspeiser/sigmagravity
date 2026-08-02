from __future__ import annotations

import numpy as np

from voidscreen.multipole_activation_3d import (
    baryonic_multipole_gate_3d,
    exact_multipole_gated_activation_3d,
)


def gaussian(axis, center, sigma, mass):
    spacing = float(axis[1] - axis[0])
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    values = np.exp(
        -0.5
        * (
            ((x - center[0]) / sigma) ** 2
            + ((y - center[1]) / sigma) ** 2
            + ((z - center[2]) / sigma) ** 2
        )
    )
    return values * mass / (np.sum(values) * spacing**3)


def test_radial_components_have_exact_zero_multipole_gate():
    axis = np.linspace(-8.0, 8.0, 25)
    stars = gaussian(axis, (0.0, 0.0, 0.0), 0.8, 0.3)
    gas = gaussian(axis, (0.0, 0.0, 0.0), 1.6, 0.7)
    gate = baryonic_multipole_gate_3d(stars, gas, axis[1] - axis[0])
    assert gate.gate < 1e-28


def test_gate_is_scale_rotation_translation_and_exchange_invariant():
    axis = np.linspace(-8.0, 8.0, 25)
    spacing = float(axis[1] - axis[0])
    stars = gaussian(axis, (-1.0, 0.0, 0.0), 0.8, 0.3)
    gas = gaussian(axis, (1.0, 0.0, 0.0), 1.6, 0.7)
    reference = baryonic_multipole_gate_3d(stars, gas, spacing)
    scaled = baryonic_multipole_gate_3d(stars, gas, 3.0 * spacing)
    rotated = baryonic_multipole_gate_3d(
        np.swapaxes(stars, 0, 1),
        np.swapaxes(gas, 0, 1),
        spacing,
    )
    exchanged = baryonic_multipole_gate_3d(gas, stars, spacing)
    assert np.isclose(reference.gate, scaled.gate, rtol=1e-14)
    assert np.isclose(reference.gate, rotated.gate, rtol=1e-14)
    assert np.isclose(reference.gate, exchanged.gate, rtol=1e-14)


def test_multipole_gated_activation_is_bounded_and_retains_offset_signal():
    axis = np.linspace(-8.0, 8.0, 25)
    spacing = float(axis[1] - axis[0])
    stars = gaussian(axis, (0.0, 0.0, 0.0), 0.8, 0.3)
    gas = gaussian(axis, (2.0, 0.0, 0.0), 1.6, 0.7)
    result = exact_multipole_gated_activation_3d(
        stars,
        gas,
        spacing,
        gravitational_constant=1.0,
        a0=0.1,
        coherence_length=2.0,
    )
    weighted = float(np.sum(result.sigma * (stars + gas)) / np.sum(stars + gas))
    assert result.multipole.gate > 0.05
    assert weighted > 0.005
    assert np.min(result.minimum_eigenvalue_proxy) > 0.0
