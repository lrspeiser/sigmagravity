import numpy as np
import pytest

from voidscreen.stellar_morphology_lensing import (
    build_stellar_morphology_deflection_field,
    normalized_light_weights,
    radial_convergence_from_deflection,
)


def test_radial_convergence_recovers_constant_sheet():
    radius = np.geomspace(0.02, 100.0, 2000)
    kappa = 0.17
    alpha = kappa * radius
    recovered = radial_convergence_from_deflection(radius, alpha)
    assert np.max(np.abs(recovered - kappa)) < 1.0e-10


def test_light_weights_preserve_carrier_weighted_annuli():
    axis = (np.arange(128) - 63.5) * 0.5
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    radius = np.hypot(xx, yy)
    light = 2.0 + np.exp(-((xx - 7.0) ** 2 + (yy + 3.0) ** 2) / 12.0)
    carrier = 0.3 * np.exp(-radius / 30.0)
    weights, audit = normalized_light_weights(
        light,
        carrier,
        radius,
        contrast_cap=5.0,
        annulus_width_arcsec=1.0,
        support_radius_arcsec=25.0,
    )
    assert weights.min() >= 0.0
    assert audit["maximum_carrier_weighted_annular_mean_error"] < 1.0e-12
    bins = np.floor(radius).astype(int)
    for index in range(25):
        selected = bins == index
        assert abs(np.mean(carrier[selected] * (weights[selected] - 1.0))) < 1.0e-14


def test_morphology_field_is_curl_free_and_has_zero_circular_mean():
    axis = (np.arange(256) - 127.5) * 0.5
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    light = (
        0.1
        + np.exp(-((xx - 9.0) ** 2 + (yy + 4.0) ** 2) / 30.0)
        + 0.7 * np.exp(-((xx + 13.0) ** 2 + (yy - 7.0) ** 2) / 80.0)
    )

    def alpha(radius):
        return 0.2 * radius * np.exp(-radius / 100.0)

    field = build_stellar_morphology_deflection_field(
        axis,
        light,
        alpha,
        contrast_cap=5.0,
        annulus_width_arcsec=1.0,
        taper_inner_arcsec=25.0,
        support_radius_arcsec=30.0,
        radial_samples=1024,
        circular_radii=256,
        circular_azimuths=720,
    )
    assert field.audit["maximum_annular_convergence_mean_fraction"] < 1.0e-12
    assert field.audit["normalized_curl_RMS"] < 1.0e-12
    phi = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    radii = np.array([2.0, 8.0, 20.0, 29.0])
    x = radii[:, None] * np.cos(phi)[None, :]
    y = radii[:, None] * np.sin(phi)[None, :]
    ax, ay = field.alpha_arcsec(x, y, distance_ratio=0.7)
    mean_radial = np.mean(
        ax * np.cos(phi)[None, :] + ay * np.sin(phi)[None, :], axis=1
    )
    assert np.max(np.abs(mean_radial)) < 2.0e-4


def test_distance_ratio_is_linear():
    axis = (np.arange(128) - 63.5) * 0.5
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    light = 0.2 + np.exp(-((xx - 5.0) ** 2 + yy**2) / 20.0)
    field = build_stellar_morphology_deflection_field(
        axis,
        light,
        lambda r: 0.12 * r * np.exp(-r / 80.0),
        contrast_cap=2.0,
        annulus_width_arcsec=1.0,
        taper_inner_arcsec=20.0,
        support_radius_arcsec=25.0,
        radial_samples=512,
        circular_radii=128,
        circular_azimuths=360,
    )
    a1 = field.alpha_arcsec(7.0, -2.0, distance_ratio=0.2)
    a2 = field.alpha_arcsec(7.0, -2.0, distance_ratio=0.7)
    assert np.allclose(np.asarray(a2) / np.asarray(a1), 3.5)


def test_contrast_strength_scales_the_curl_free_correction_linearly():
    axis = (np.arange(128) - 63.5) * 0.5
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    light = 0.2 + np.exp(-((xx - 5.0) ** 2 + (yy + 2.0) ** 2) / 20.0)
    common = dict(
        contrast_cap=5.0,
        annulus_width_arcsec=1.0,
        taper_inner_arcsec=20.0,
        support_radius_arcsec=25.0,
        radial_samples=512,
        circular_radii=128,
        circular_azimuths=360,
    )
    full = build_stellar_morphology_deflection_field(
        axis, light, lambda r: 0.12 * r * np.exp(-r / 80.0), **common
    )
    weak = build_stellar_morphology_deflection_field(
        axis,
        light,
        lambda r: 0.12 * r * np.exp(-r / 80.0),
        contrast_strength=0.2,
        **common,
    )
    assert np.allclose(weak.raw_alpha_x_arcsec, 0.2 * full.raw_alpha_x_arcsec)
    assert np.allclose(weak.raw_alpha_y_arcsec, 0.2 * full.raw_alpha_y_arcsec)
    assert weak.audit["contrast_strength"] == 0.2


def test_smooth_contrast_modes_are_positive_bounded_and_annularly_conservative():
    axis = np.linspace(-40.0, 40.0, 129)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    radius = np.hypot(xx, yy)
    light = np.exp(-((xx - 8.0) ** 2 + (yy + 3.0) ** 2) / (2.0 * 3.0**2))
    carrier = np.exp(-radius / 20.0)
    for mode in ("tanh", "exponential", "rational"):
        weights, audit = normalized_light_weights(
            light,
            carrier,
            radius,
            contrast_cap=5.0,
            contrast_mode=mode,
            annulus_width_arcsec=2.0,
            support_radius_arcsec=35.0,
        )
        supported = radius <= 35.0
        assert audit["contrast_mode"] == mode
        assert np.all(np.isfinite(weights))
        assert np.all(weights[supported] >= 0.0)
        assert audit["maximum_carrier_weighted_annular_mean_error"] < 1e-12


def test_invalid_contrast_mode_is_rejected():
    radius = np.ones((4, 4))
    with pytest.raises(ValueError, match="contrast_mode"):
        normalized_light_weights(
            np.ones_like(radius),
            np.ones_like(radius),
            radius,
            contrast_cap=5.0,
            contrast_mode="sharpish",
            annulus_width_arcsec=1.0,
            support_radius_arcsec=3.0,
        )
