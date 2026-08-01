from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from voidscreen.galaxy_maps import (
    reproject_wcs_to_disk_grid,
    resolved_map_morphology,
    sky_pixels_to_disk_coordinates,
    weighted_radius_quantile,
)


def simple_wcs(center: SkyCoord, shape: tuple[int, int], scale_arcsec: float) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [center.ra.deg, center.dec.deg]
    wcs.wcs.crpix = [shape[1] // 2 + 1.0, shape[0] // 2 + 1.0]
    wcs.wcs.cd = np.array([[-scale_arcsec / 3600.0, 0.0], [0.0, scale_arcsec / 3600.0]])
    return wcs


def test_wcs_projection_preserves_a_registered_center_and_orientation():
    center = SkyCoord(155.0 * u.deg, 25.0 * u.deg)
    shape = (121, 121)
    wcs = simple_wcs(center, shape, 2.0)
    yy, xx = np.indices(shape, dtype=float)
    image = np.exp(-0.5 * (((xx - 60.0) / 12.0) ** 2 + ((yy - 60.0) / 5.0) ** 2))
    axis = np.linspace(-0.5, 0.5, 81)
    disk = reproject_wcs_to_disk_grid(
        image,
        wcs,
        center=center,
        position_angle_deg=90.0,
        inclination_deg=60.0,
        distance_mpc=1.0,
        disk_axis_kpc=axis,
    )
    peak = np.unravel_index(np.argmax(disk), disk.shape)
    assert np.hypot(peak[0] - 40, peak[1] - 40) <= 1.0
    x, y = sky_pixels_to_disk_coordinates(
        np.array([60.0]),
        np.array([60.0]),
        wcs,
        center=center,
        position_angle_deg=90.0,
        inclination_deg=60.0,
        distance_mpc=1.0,
    )
    assert np.hypot(x[0], y[0]) < 1e-6


def test_weighted_radius_and_morphology_are_finite():
    axis = np.linspace(-2.0, 2.0, 65)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    image = np.exp(-np.hypot(xx - 0.1, yy + 0.2) / 0.5)
    r90 = weighted_radius_quantile(xx, yy, image, 0.9)
    metrics = resolved_map_morphology(image, disk_axis_kpc=axis, smoothing_sigma_pixel=2.0)
    assert 0.5 < r90 < 2.0
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["lopsidedness_180"] <= 1.0
