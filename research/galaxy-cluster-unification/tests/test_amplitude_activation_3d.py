from __future__ import annotations

import numpy as np

from voidscreen.amplitude_activation_3d import exact_amplitude_multipole_activation_3d
from voidscreen.multipole_activation_3d import exact_multipole_gated_activation_3d


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


def test_radial_components_have_exact_zero_amplitude():
    axis = np.linspace(-8.0, 8.0, 25)
    stars = gaussian(axis, (0.0, 0.0, 0.0), 0.8, 0.3)
    gas = gaussian(axis, (0.0, 0.0, 0.0), 1.6, 0.7)
    result = exact_amplitude_multipole_activation_3d(
        stars,
        gas,
        axis[1] - axis[0],
        gravitational_constant=1.0,
        a0=0.1,
        coherence_length=2.0,
    )
    assert result.amplitude_gate < 2e-16
    assert float(np.max(result.sigma)) < 2e-16


def test_amplitude_interpretation_is_exact_square_root_and_enhances_offset_field():
    axis = np.linspace(-8.0, 8.0, 25)
    spacing = float(axis[1] - axis[0])
    stars = gaussian(axis, (0.0, 0.0, 0.0), 0.8, 0.3)
    gas = gaussian(axis, (2.0, 0.0, 0.0), 1.6, 0.7)
    kwargs = {
        "gravitational_constant": 1.0,
        "a0": 0.1,
        "coherence_length": 2.0,
    }
    amplitude = exact_amplitude_multipole_activation_3d(
        stars,
        gas,
        spacing,
        **kwargs,
    )
    intensity = exact_multipole_gated_activation_3d(
        stars,
        gas,
        spacing,
        **kwargs,
    )
    assert np.isclose(
        amplitude.amplitude_gate,
        np.sqrt(amplitude.multipole.gate),
        rtol=1e-15,
    )
    assert float(np.sum(amplitude.sigma)) > float(np.sum(intensity.sigma))
    assert float(np.min(amplitude.minimum_eigenvalue_proxy)) > 0.0
    assert float(np.max(amplitude.sigma)) <= 1.0
