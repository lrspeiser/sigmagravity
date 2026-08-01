import numpy as np

from voidscreen.tensor_routing import (
    anisotropic_gaussian_deposit,
    baryonic_field_frames,
    curl_free_deflection_diagnostic,
    redistributed_cumulative_mass_tensor,
    weighted_radii,
)


def test_tensor_kernel_is_positive_normalized_and_curl_free():
    axis = np.linspace(-100.0, 100.0, 101)
    positions = np.array([[-25.0, 0.0], [25.0, 0.0], [0.0, 30.0]])
    weights = np.array([0.4, 0.4, 0.2])
    center, r50, r80, concentration = weighted_radii(positions, weights)
    frames = baryonic_field_frames(positions, weights, softening=10.0)
    image = anisotropic_gaussian_deposit(
        axis,
        positions,
        weights,
        frames["tidal"],
        geometric_sigma=12.0,
        axis_ratio=2.0,
    )
    diagnostic = curl_free_deflection_diagnostic(image, axis[1] - axis[0])
    assert np.all(np.isfinite(center))
    assert 0.0 < r50 <= r80
    assert 0.0 < concentration <= 1.0
    assert np.min(image) >= 0.0
    assert np.isclose(np.sum(image), 1.0)
    assert diagnostic["relative_curl_norm"] < 1e-12
    assert diagnostic["relative_poisson_residual"] < 1e-12


def test_radial_projection_conserves_mass():
    radius = np.linspace(0.01, 10.0, 300)
    mass = (radius / radius[-1]) ** 2
    routed, error = redistributed_cumulative_mass_tensor(
        radius,
        mass,
        r80=8.0,
        length_over_r80=1.0,
        radius_exponent=-0.5,
        width_over_r80=0.25,
        axis_ratio=1.5,
    )
    assert np.all(np.isfinite(routed))
    assert np.all(np.diff(routed) >= -1e-12)
    assert error < 1e-12
