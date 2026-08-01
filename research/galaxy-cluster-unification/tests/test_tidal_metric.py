import numpy as np

from voidscreen.tidal_metric import (
    build_tidal_correction_field,
    normalized_spin2_tensor,
)


def test_normalized_member_tensor_is_bounded():
    axis = np.linspace(-20.0, 20.0, 41)
    x, y = np.meshgrid(axis, axis)
    qxx, qxy = normalized_spin2_tensor(
        x,
        y,
        np.array([-8.0, 3.0, 11.0]),
        np.array([2.0, -5.0, 7.0]),
        np.array([0.2, 0.5, 0.3]),
        softening_arcsec=1.5,
    )
    assert np.max(np.hypot(qxx, qxy)) <= 1.0 + 1.0e-12


def test_circular_member_ring_is_removed_and_correction_is_curl_free():
    phi = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    field = build_tidal_correction_field(
        30.0 * np.cos(phi),
        30.0 * np.sin(phi),
        np.ones_like(phi),
        softening_arcsec=2.0,
        extra_alpha_arcsec=lambda radius: np.ones_like(radius) * 5.0,
        half_width_arcsec=96.0,
        pixels_per_axis=128,
        polar_mean_radii=96,
        polar_mean_azimuths=192,
    )
    assert field.audit["RMS_Q_eigenvalue"] < 0.035
    assert field.audit["normalized_curl_RMS"] < 1.0e-12


def test_full_tensor_retains_circular_member_stress():
    phi = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    field = build_tidal_correction_field(
        30.0 * np.cos(phi),
        30.0 * np.sin(phi),
        np.ones_like(phi),
        softening_arcsec=2.0,
        extra_alpha_arcsec=lambda radius: np.ones_like(radius) * 5.0,
        half_width_arcsec=96.0,
        pixels_per_axis=128,
        polar_mean_radii=96,
        polar_mean_azimuths=192,
        subtract_circular_mean=False,
    )
    assert field.audit["RMS_Q_eigenvalue"] > 0.10
    assert field.audit["circular_mean_subtracted"] is False
    assert field.audit["normalized_curl_RMS"] < 1.0e-12


def test_correction_interpolation_returns_zero_outside_grid():
    field = build_tidal_correction_field(
        np.array([-8.0, 2.0, 14.0]),
        np.array([1.0, -6.0, 4.0]),
        np.array([0.3, 0.4, 0.3]),
        softening_arcsec=2.0,
        extra_alpha_arcsec=lambda radius: 3.0 * np.ones_like(radius),
        half_width_arcsec=64.0,
        pixels_per_axis=128,
        polar_mean_radii=64,
        polar_mean_azimuths=128,
    )
    ax, ay = field.alpha_arcsec(np.array([0.0, 100.0]), np.array([0.0, 100.0]))
    assert np.isfinite(ax[0]) and np.isfinite(ay[0])
    assert ax[1] == 0.0 and ay[1] == 0.0
