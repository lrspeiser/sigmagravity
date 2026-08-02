from __future__ import annotations

import numpy as np

from voidscreen.compound_activation_3d import exact_compound_path_activation_3d


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


def settings():
    return {
        "gravitational_constant": 1.0,
        "a0": 0.1,
        "coherence_length": 2.0,
        "coherence_power": 2.0,
    }


def test_cocentered_radial_components_have_exact_compound_null():
    axis = np.linspace(-8.0, 8.0, 25)
    stars = gaussian(axis, (0.0, 0.0, 0.0), 0.8, 0.3)
    gas = gaussian(axis, (0.0, 0.0, 0.0), 1.6, 0.7)
    result = exact_compound_path_activation_3d(
        stars,
        gas,
        axis[1] - axis[0],
        **settings(),
    )
    assert result.amplitude_gate < 2e-16
    assert float(np.max(result.sigma)) < 2e-16


def test_compound_identity_and_positive_constitutive_bound():
    axis = np.linspace(-8.0, 8.0, 25)
    stars = gaussian(axis, (0.0, 0.0, 0.0), 0.8, 0.3)
    gas = gaussian(axis, (2.0, 0.0, 0.0), 1.6, 0.7)
    result = exact_compound_path_activation_3d(
        stars,
        gas,
        axis[1] - axis[0],
        **settings(),
    )
    reconstructed = 1.0 - np.power(
        1.0 - result.elementary_probability,
        result.coherent_opportunities,
    )
    active = reconstructed < 1.0 - 1e-6
    assert np.allclose(result.sigma[active], reconstructed[active], rtol=1e-12, atol=1e-15)
    assert float(np.max(result.sigma)) <= 1.0 - 1e-6
    assert float(np.min(result.minimum_eigenvalue_proxy)) > 0.0
