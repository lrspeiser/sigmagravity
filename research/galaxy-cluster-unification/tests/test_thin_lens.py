from __future__ import annotations

import numpy as np

from voidscreen.thin_lens import (
    C_SI,
    G_SI,
    KPC_M,
    M_SUN_KG,
    RAD_TO_ARCSEC,
    thin_lens_deflection_from_surface_density,
)


def test_resolved_point_mass_recovers_exact_thin_lens_deflection():
    size = 65
    cell_kpc = 1.0
    mass_msun = 1.0e11
    surface = np.zeros((size, size))
    center = size // 2
    surface[center, center] = mass_msun / cell_kpc**2
    field = thin_lens_deflection_from_surface_density(surface, cell_kpc)
    offset = 10
    expected = (
        4.0
        * G_SI
        * mass_msun
        * M_SUN_KG
        / (C_SI**2 * offset * cell_kpc * KPC_M)
        * RAD_TO_ARCSEC
    )
    assert np.isclose(field.input_mass_msun, mass_msun)
    assert np.isclose(field.alpha_east_arcsec[center, center + offset], expected)
    assert abs(field.alpha_north_arcsec[center, center + offset]) < 1.0e-14
    assert np.isclose(field.alpha_north_arcsec[center + offset, center], expected)
    assert abs(field.alpha_east_arcsec[center + offset, center]) < 1.0e-14


def test_symmetric_surface_produces_antisymmetric_sky_components():
    axis = np.linspace(-4.0, 4.0, 33)
    east, north = np.meshgrid(axis, axis, indexing="xy")
    surface = np.exp(-0.5 * (east**2 + north**2))
    field = thin_lens_deflection_from_surface_density(
        surface, axis[1] - axis[0], gravitational_constant=1.0, light_speed=1.0
    )
    scale = float(np.max(np.abs(field.alpha_east_radian)))
    east_antisymmetry = np.max(
        np.abs(field.alpha_east_radian + field.alpha_east_radian[:, ::-1])
    )
    north_antisymmetry = np.max(
        np.abs(field.alpha_north_radian + field.alpha_north_radian[::-1, :])
    )
    transpose_symmetry = np.max(
        np.abs(field.alpha_east_radian - field.alpha_north_radian.T)
    )
    assert east_antisymmetry / scale < 1.0e-12
    assert north_antisymmetry / scale < 1.0e-12
    assert transpose_symmetry / scale < 1.0e-12


def test_thin_lens_rejects_negative_or_invalid_surface_maps():
    invalid = np.ones((9, 9))
    invalid[3, 4] = -1.0
    with np.testing.assert_raises(ValueError):
        thin_lens_deflection_from_surface_density(invalid, 1.0)
    with np.testing.assert_raises(ValueError):
        thin_lens_deflection_from_surface_density(np.ones((4, 4)), 1.0)
