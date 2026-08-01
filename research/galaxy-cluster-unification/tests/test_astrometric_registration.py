from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from voidscreen.astrometric_registration import solve_foreground_star_wcs


def test_foreground_star_solver_recovers_a_shifted_rotated_frame():
    rng = np.random.default_rng(8)
    center = SkyCoord(120.0 * u.deg, 35.0 * u.deg)
    east = rng.uniform(-300.0, 300.0, 120) * u.arcsec
    north = rng.uniform(-300.0, 300.0, 120) * u.arcsec
    sky = center.spherical_offsets_by(east, north)
    conventional = np.column_stack([-east.to_value(u.arcsec), north.to_value(u.arcsec)])
    angle = np.radians(0.7)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    pixel = conventional @ rotation.T + np.array([430.0, 465.0])
    image = rng.normal(1000.0, 2.0, (900, 900))
    for x, y in pixel:
        if 12 <= x < 888 and 12 <= y < 888:
            image[round(y), round(x)] += 1000.0
    settings = {
        "orientation_candidates": ["identity"],
        "sigma_clip": 3.0,
        "detection_threshold_sigma": 8.0,
        "minimum_detection_separation_arcsec": 2.0,
        "galaxy_exclusion_radius_arcsec": 30.0,
        "maximum_detected_sources": 1000,
        "maximum_translation_pixel": 100.0,
        "translation_histogram_bin_pixel": 5.0,
        "initial_match_radius_pixel": 15.0,
        "ransac_residual_pixel": 2.0,
        "ransac_max_trials": 2000,
        "random_seed": 0,
        "final_projection": "TAN",
    }
    fit = solve_foreground_star_wcs(
        image,
        catalog_center=center,
        catalog_pixel_scale_arcsec=1.0,
        gaia_sky=sky,
        settings=settings,
    )
    assert fit.diagnostics["gaia_inliers"] >= 90
    assert fit.diagnostics["median_residual_pixel"] < 0.5
    expected = center.spherical_offsets_by(0.0 * u.deg, 0.0 * u.arcsec)
    x, y = fit.wcs.world_to_pixel(expected)
    assert np.hypot(x - 430.0, y - 465.0) < 1.0
