from __future__ import annotations

import numpy as np
from astropy.io.fits import Header

from voidscreen.galaxy_maps import (
    aips_clean_beam_degrees,
    deproject_to_disk_grid,
    hi_moment0_surface_density,
    integrated_hi_mass_solar,
    normalize_surface_density_mass,
    weighted_disk_geometry,
)


def radio_header() -> Header:
    header = Header()
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    header["RESTFREQ"] = 1.42040575e9
    header["HISTORY"] = "AIPS   CLEAN BMAJ=  3.3333E-03 BMIN=  2.7778E-03 BPA= 32.0"
    return header


def test_aips_history_beam_is_parsed():
    assert np.allclose(aips_clean_beam_degrees(radio_header()), (0.0033333, 0.0027778, 32.0))


def test_hi_conversion_and_integrated_mass_have_physical_units():
    image = np.full((11, 11), 100.0)
    header = radio_header()
    surface = hi_moment0_surface_density(image, header, inclination_deg=60.0)
    assert np.all(surface > 0.0)
    assert np.allclose(surface, surface[0, 0])
    expected_pixels_per_beam = np.pi * 11.99988 * 10.00008 / (4.0 * np.log(2.0))
    expected_flux = image.sum() / expected_pixels_per_beam / 1000.0
    expected_mass = 2.356e5 * 4.0**2 * expected_flux
    assert np.isclose(integrated_hi_mass_solar(image, header, distance_mpc=4.0), expected_mass)


def test_geometry_and_deprojection_preserve_an_elliptical_major_axis():
    yy, xx = np.indices((101, 101), dtype=float)
    angle = np.radians(30.0)
    major = np.cos(angle) * (xx - 53.0) + np.sin(angle) * (yy - 48.0)
    minor = -np.sin(angle) * (xx - 53.0) + np.cos(angle) * (yy - 48.0)
    image = np.exp(-0.5 * ((major / 16.0) ** 2 + (minor / 6.0) ** 2))
    geometry = weighted_disk_geometry(image, inclination_deg=60.0)
    assert np.hypot(geometry.center_x_pixel - 53.0, geometry.center_y_pixel - 48.0) < 0.1
    assert abs(geometry.position_angle_pixel_deg - 30.0) < 0.2
    axis = np.linspace(-2.0, 2.0, 81)
    disk = deproject_to_disk_grid(
        image,
        geometry,
        sky_pixel_scale_arcsec=10.0,
        distance_mpc=1.0,
        disk_axis_kpc=axis,
    )
    assert disk.shape == (81, 81)
    assert np.unravel_index(np.argmax(disk), disk.shape) == (40, 40)


def test_mass_normalization_is_exact():
    morphology = np.arange(1.0, 82.0).reshape(9, 9)
    surface = normalize_surface_density_mass(
        morphology, pixel_size_kpc=0.2, total_mass_solar=3.7e8
    )
    assert np.isclose(surface.sum() * 0.2**2, 3.7e8)
