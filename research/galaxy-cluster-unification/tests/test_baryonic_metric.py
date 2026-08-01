import numpy as np

from voidscreen.baryonic_metric import (
    asymmetry_gate,
    build_baryonic_metric_correction_field,
    low_acceleration_gate,
    prepare_baryonic_metric_state,
    prepare_baryonic_metric_workspace,
    remove_baryonic_affine_modes,
    spherical_metric_acceleration,
    weighted_morphology,
)
from voidscreen.tidal_metric import TidalCorrectionField


def test_workspace_does_not_mutate_source_weights():
    weights = np.array([1.0, 2.0, 3.0])
    original = weights.copy()
    prepare_baryonic_metric_workspace(
        [-1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        weights,
        total_mass_msun=1.0e12,
        scale_kpc_per_arcsec=5.0,
        half_width_arcsec=32.0,
        pixels_per_axis=64,
    )
    assert np.array_equal(weights, original)


def test_low_acceleration_gate_and_spherical_limits():
    g = np.array([0.0, 1.2e-10, 1.2e-8])
    gate = low_acceleration_gate(g, 1.2e-10, 2.0)
    assert np.allclose(gate[:2], [1.0, 0.5])
    assert gate[2] < 1.1e-4
    predicted = spherical_metric_acceleration(
        g,
        minimum_permittivity=0.25,
        a0_m_s2=1.2e-10,
        gate_power=2.0,
    )
    assert predicted[0] == 0.0
    assert np.isclose(predicted[1] / g[1], 1.0 / 0.625)
    assert np.isclose(predicted[2] / g[2], 1.0, rtol=1.0e-4)


def test_morphology_and_asymmetry_are_scale_free():
    circular = weighted_morphology([1, -1, 0, 0], [0, 0, 1, -1], [1, 1, 1, 1])
    elongated = weighted_morphology([2, -2, 1, -1], [0, 0, 0, 0], [1, 1, 1, 1])
    assert circular["quadrupole_asymmetry"] < 1.0e-12
    assert elongated["quadrupole_asymmetry"] > 0.99
    assert asymmetry_gate(circular["quadrupole_asymmetry"]) == 0.0
    assert asymmetry_gate(elongated["quadrupole_asymmetry"]) > 0.99


def test_identity_metric_returns_zero_correction():
    workspace = prepare_baryonic_metric_workspace(
        [-2.0, 0.0, 2.0],
        [0.0, 0.5, 0.0],
        [1.0, 2.0, 1.0],
        total_mass_msun=1.0e13,
        scale_kpc_per_arcsec=5.0,
        half_width_arcsec=64.0,
        pixels_per_axis=64,
        point_softening_arcsec=2.0,
    )
    state = prepare_baryonic_metric_state(workspace, 0.35)
    field = build_baryonic_metric_correction_field(
        [-2.0, 0.0, 2.0],
        [0.0, 0.5, 0.0],
        [1.0, 2.0, 1.0],
        total_mass_msun=1.0e13,
        scale_kpc_per_arcsec=5.0,
        minimum_permittivity=1.0,
        a0_m_s2=1.2e-10,
        gate_power=2.0,
        anisotropy=0.0,
        smoothing_r80_fraction=0.35,
        workspace=workspace,
        state=state,
    )
    assert np.max(np.abs(field.alpha_x_arcsec)) < 1.0e-14
    assert np.max(np.abs(field.alpha_y_arcsec)) < 1.0e-14


def test_circular_source_is_removed_but_metric_stays_positive():
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    x = 10.0 * np.cos(angles)
    y = 10.0 * np.sin(angles)
    weights = np.ones_like(x)
    field = build_baryonic_metric_correction_field(
        x,
        y,
        weights,
        total_mass_msun=2.0e13,
        scale_kpc_per_arcsec=5.0,
        minimum_permittivity=0.25,
        a0_m_s2=1.2e-10,
        gate_power=2.0,
        anisotropy=0.6,
        smoothing_r80_fraction=0.35,
        half_width_arcsec=64.0,
        pixels_per_axis=64,
    )
    assert field.audit["asymmetry_gate"] < 1.0e-10
    assert field.audit["metric_minimum_eigenvalue"] > 0.0
    assert field.audit["normalized_curl_RMS"] < 1.0e-10
    assert field.audit["circular_residual_fraction"] < 0.05


def test_baryon_defined_symmetric_affine_removal_is_curl_free():
    axis = np.arange(-64.0, 64.0, 2.0)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    alpha_x = 0.02 * xx + 0.01 * yy + 0.3
    alpha_y = 0.01 * xx - 0.015 * yy - 0.2
    zeros = np.zeros_like(xx)
    field = TidalCorrectionField(
        axis,
        alpha_x,
        alpha_y,
        zeros,
        zeros,
        {"center_x_arcsec": 0.0, "center_y_arcsec": 0.0, "r80_arcsec": 20.0},
    )
    result = remove_baryonic_affine_modes(
        field,
        aperture_r80_fraction=1.0,
        removal_fraction=1.0,
        mode="symmetric",
        taper_outer_factor=2.0,
    )
    radius = np.hypot(xx, yy)
    inner = radius < 18.0
    assert np.sqrt(
        np.mean(result.alpha_x_arcsec[inner] ** 2 + result.alpha_y_arcsec[inner] ** 2)
    ) < 1.0e-12
    assert result.audit["baryon_grid_affine_R2_before"] > 0.999999
    assert result.audit["removed_affine_RMS_arcsec"] > 0.2
    assert result.audit["normalized_curl_RMS"] < 1.0e-10


def test_trace_only_removal_preserves_shear_component():
    axis = np.arange(-64.0, 64.0, 2.0)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    alpha_x = 0.02 * xx + 0.01 * yy
    alpha_y = 0.01 * xx - 0.015 * yy
    zeros = np.zeros_like(xx)
    field = TidalCorrectionField(
        axis,
        alpha_x,
        alpha_y,
        zeros,
        zeros,
        {"center_x_arcsec": 0.0, "center_y_arcsec": 0.0, "r80_arcsec": 20.0},
    )
    result = remove_baryonic_affine_modes(
        field,
        aperture_r80_fraction=1.0,
        removal_fraction=1.0,
        mode="trace",
    )
    radius = np.hypot(xx, yy)
    inner = radius < 18.0
    rms = np.sqrt(
        np.mean(result.alpha_x_arcsec[inner] ** 2 + result.alpha_y_arcsec[inner] ** 2)
    )
    assert rms > 0.05
    assert result.audit["normalized_curl_RMS"] < 1.0e-10
