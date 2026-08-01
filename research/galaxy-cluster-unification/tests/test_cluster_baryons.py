import numpy as np

from voidscreen.cluster_baryons import (
    block_compress_surface,
    dpie_axis_ratio,
    dpie_surface_density_shape,
    dpie_total_mass_msun,
    normalize_surface_mass,
    sky_to_lens_offsets,
)


def test_sky_to_lens_offsets_uses_west_positive_convention():
    x, y = sky_to_lens_offsets(
        [10.0, 9.999],
        [20.0, 20.001],
        reference_ra_deg=10.0,
        reference_dec_deg=20.0,
    )
    assert x[0] == 0.0
    assert x[1] > 0.0
    assert y[1] > 0.0


def test_dpie_shape_and_mass_are_positive():
    axis = np.linspace(-20.0, 20.0, 41)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    shape = dpie_surface_density_shape(
        xx,
        yy,
        center_x=1.0,
        center_y=-2.0,
        ellipticity=0.4,
        theta_deg=30.0,
        r_core_arcsec=2.0,
        r_cut_arcsec=15.0,
    )
    assert np.all(shape >= 0.0)
    assert np.max(shape) > 0.0
    assert 0.0 < dpie_axis_ratio(0.4) < 1.0
    assert dpie_total_mass_msun(
        sigma_lt_km_s=200.0,
        r_core_arcsec=2.0,
        r_cut_arcsec=15.0,
        scale_kpc_per_arcsec=5.0,
    ) > 0.0


def test_block_compression_conserves_mass():
    axis = np.linspace(-10.0, 10.0, 21)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    shape = np.exp(-0.1 * (xx * xx + yy * yy))
    surface = normalize_surface_mass(shape, 7.5e12)
    x, y, mass = block_compress_surface(axis, surface, blocks_per_axis=5)
    assert len(x) == len(y) == len(mass)
    assert np.isclose(np.sum(mass), 7.5e12, rtol=1e-12)
    assert np.all(x >= axis[0]) and np.all(x <= axis[-1])
