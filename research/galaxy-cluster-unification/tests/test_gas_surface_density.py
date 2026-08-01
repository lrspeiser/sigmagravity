import numpy as np

from voidscreen.gas_surface_density import (
    annular_morphology_factor,
    enclosed_gas_mass_msun,
    projected_gas_surface_density_msun_kpc2,
)


def test_uniform_sphere_center_column_and_enclosed_mass_are_positive():
    sigma = projected_gas_surface_density_msun_kpc2(
        np.array([0.0, 5.0, 10.0]), [0.0], [10.0], [0.01]
    )
    assert sigma[0] > sigma[1] > sigma[2] == 0.0
    mass = enclosed_gas_mass_msun(10.0, [0.0], [10.0], [0.01])
    assert mass > 0.0


def test_multiple_shell_projection_is_additive_and_finite():
    radius = np.linspace(0.0, 30.0, 31)
    combined = projected_gas_surface_density_msun_kpc2(
        radius, [0.0, 10.0], [10.0, 30.0], [0.03, 0.01]
    )
    inner = projected_gas_surface_density_msun_kpc2(
        radius, [0.0], [10.0], [0.03]
    )
    outer = projected_gas_surface_density_msun_kpc2(
        radius, [10.0], [30.0], [0.01]
    )
    assert np.all(np.isfinite(combined))
    assert np.allclose(combined, inner + outer)


def test_annular_morphology_factor_preserves_each_radial_mean():
    axis = np.arange(-20.0, 21.0)
    xx, yy = np.meshgrid(axis, axis)
    image = np.exp(-np.hypot(xx, yy) / 8.0) * (1.0 + 0.5 * (xx > 0.0))
    factor = annular_morphology_factor(
        axis, image, power=0.5, smoothing_sigma_arcsec=2.0
    )
    bins = np.floor(np.hypot(xx, yy)).astype(int)
    for index in np.unique(bins):
        assert np.isclose(np.mean(factor[bins == index]), 1.0)
    assert factor.min() >= 0.25
    assert factor.max() <= 4.0
