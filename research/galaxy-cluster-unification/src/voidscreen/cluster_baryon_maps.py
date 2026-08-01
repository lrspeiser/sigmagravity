"""Construct physical cluster baryon maps without lensing information."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter

from voidscreen.gravity_arc_tomography import (
    combine_f160_photometry,
    photometric_membership_weights,
    read_relics_catalog,
)


@dataclass(frozen=True)
class ClusterGrid:
    center: SkyCoord
    axis_kpc: np.ndarray
    x_kpc: np.ndarray
    y_kpc: np.ndarray
    cell_kpc: float
    world: SkyCoord


def physical_grid(center: SkyCoord, *, half_extent_kpc: float, size: int, redshift: float) -> ClusterGrid:
    """Return a square physical grid and its celestial coordinates."""
    axis = np.linspace(-float(half_extent_kpc), float(half_extent_kpc), int(size))
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")
    kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(float(redshift)).value / 60.0)
    world = center.spherical_offsets_by(
        (x_grid / kpc_per_arcsec) * u.arcsec,
        (y_grid / kpc_per_arcsec) * u.arcsec,
    )
    return ClusterGrid(
        center=center,
        axis_kpc=axis,
        x_kpc=x_grid,
        y_kpc=y_grid,
        cell_kpc=float(axis[1] - axis[0]),
        world=world,
    )


def strict_f160_members(catalog, cluster_redshift: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen member mask and selected F160W flux in nJy."""
    flux, significance = combine_f160_photometry(catalog)
    photoz, _soft = photometric_membership_weights(catalog, cluster_redshift)
    selected = (
        photoz
        & (catalog["stel"].to_numpy(float) < 0.8)
        & np.isfinite(flux)
        & np.isfinite(significance)
        & (flux > 0.0)
        & (significance >= 5.0)
    )
    return selected, flux


def f160_stellar_mass_msun(
    flux_njy: np.ndarray,
    *,
    redshift: float,
    mass_to_light_solar: float,
    solar_absolute_ab_magnitude: float,
) -> np.ndarray:
    """Convert observed F160W flux to a shared near-IR stellar-mass scale."""
    flux = np.asarray(flux_njy, dtype=float)
    apparent_ab = 31.4 - 2.5 * np.log10(flux)
    absolute_ab = (
        apparent_ab
        - float(Planck18.distmod(float(redshift)).value)
        + 2.5 * math.log10(1.0 + float(redshift))
    )
    luminosity_solar = np.power(
        10.0, -0.4 * (absolute_ab - float(solar_absolute_ab_magnitude))
    )
    return float(mass_to_light_solar) * luminosity_solar


def member_light_center(catalog, selected: np.ndarray, mass: np.ndarray) -> SkyCoord:
    """Compute a baryonic center from member positions and stellar masses."""
    coordinates = SkyCoord(
        catalog.loc[selected, "RA"].to_numpy(float) * u.deg,
        catalog.loc[selected, "Dec"].to_numpy(float) * u.deg,
        frame="icrs",
    )
    anchor = SkyCoord(
        float(np.median(coordinates.ra.deg)) * u.deg,
        float(np.median(coordinates.dec.deg)) * u.deg,
        frame="icrs",
    )
    east, north = anchor.spherical_offsets_to(coordinates)
    weight = np.asarray(mass, dtype=float)
    return anchor.spherical_offsets_by(
        float(np.average(east.to_value(u.arcsec), weights=weight)) * u.arcsec,
        float(np.average(north.to_value(u.arcsec), weights=weight)) * u.arcsec,
    )


def stellar_surface_density(
    catalog_path: Path,
    image_path: Path,
    segmentation_path: Path,
    *,
    grid: ClusterGrid,
    redshift: float,
    mass_to_light_solar: float,
    solar_absolute_ab_magnitude: float,
) -> tuple[np.ndarray, dict]:
    """Bin measured segmented F160W member pixels onto a physical grid."""
    catalog = read_relics_catalog(catalog_path)
    selected, flux = strict_f160_members(catalog, redshift)
    selected_mass = f160_stellar_mass_msun(
        flux[selected],
        redshift=redshift,
        mass_to_light_solar=mass_to_light_solar,
        solar_absolute_ab_magnitude=solar_absolute_ab_magnitude,
    )
    maximum_id = int(np.max(catalog["id"]))
    mass_by_id = np.zeros(maximum_id + 1, dtype=float)
    selected_ids = catalog.loc[selected, "id"].to_numpy(int)
    mass_by_id[selected_ids] = selected_mass
    with fits.open(image_path, memmap=True) as image_hdul, fits.open(
        segmentation_path, memmap=True
    ) as segmentation_hdul:
        image = np.asarray(image_hdul[0].data)
        segmentation = np.asarray(segmentation_hdul[0].data)
        if image.shape != segmentation.shape:
            raise ValueError("F160W image and segmentation shape differ")
        valid = (
            np.isfinite(image)
            & (image > 0.0)
            & (segmentation >= 0)
            & (segmentation <= maximum_id)
        )
        valid &= mass_by_id[np.clip(segmentation, 0, maximum_id)] > 0.0
        y_pixel, x_pixel = np.nonzero(valid)
        segment_id = segmentation[valid].astype(int)
        light = image[valid].astype(float)
        light_by_id = np.bincount(segment_id, weights=light, minlength=maximum_id + 1)
        usable = light_by_id[selected_ids] > 0.0
        if not np.all(usable):
            missing_ids = selected_ids[~usable]
            raise ValueError(f"selected F160W members lack positive segmented pixels: {missing_ids}")
        pixel_mass = light * mass_by_id[segment_id] / light_by_id[segment_id]
        image_wcs = WCS(image_hdul[0].header)
        ra, dec = image_wcs.pixel_to_world_values(x_pixel, y_pixel)
    coordinates = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    east, north = grid.center.spherical_offsets_to(coordinates)
    kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(float(redshift)).value / 60.0)
    x_kpc = east.to_value(u.arcsec) * kpc_per_arcsec
    y_kpc = north.to_value(u.arcsec) * kpc_per_arcsec
    edge_step = grid.cell_kpc
    edges = np.concatenate(
        ([grid.axis_kpc[0] - 0.5 * edge_step], grid.axis_kpc + 0.5 * edge_step)
    )
    mass_grid, _, _ = np.histogram2d(y_kpc, x_kpc, bins=(edges, edges), weights=pixel_mass)
    surface = mass_grid / grid.cell_kpc**2
    recovered = float(np.sum(mass_grid))
    expected = float(np.sum(selected_mass))
    return surface, {
        "selected_members": int(np.sum(selected)),
        "segmented_pixels": len(pixel_mass),
        "stellar_mass_msun": expected,
        "recovered_stellar_mass_msun": recovered,
        "stellar_mass_recovery_fraction": recovered / expected,
    }


def chandra_rate_map(paths: list[Path], grid: ClusterGrid) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Reproject and exposure-combine Chandra count images."""
    counts = np.zeros_like(grid.x_kpc, dtype=float)
    exposure = np.zeros_like(counts)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        for path in paths:
            with fits.open(path, memmap=True) as hdul:
                data = np.asarray(hdul[0].data, dtype=float)
                header = hdul[0].header
                x_pixel, y_pixel = WCS(header).world_to_pixel(grid.world)
                inside = (
                    np.isfinite(x_pixel)
                    & np.isfinite(y_pixel)
                    & (x_pixel >= 0.0)
                    & (x_pixel <= data.shape[1] - 1.0)
                    & (y_pixel >= 0.0)
                    & (y_pixel <= data.shape[0] - 1.0)
                )
                sampled = map_coordinates(
                    data,
                    [y_pixel, x_pixel],
                    order=1,
                    mode="constant",
                    cval=0.0,
                )
                obs_exposure = float(header["EXPOSURE"])
                counts += np.where(inside, sampled, 0.0)
                exposure += np.where(inside, obs_exposure, 0.0)
                rows.append(
                    {
                        "filename": path.name,
                        "exposure_s": obs_exposure,
                        "covered_grid_fraction": float(np.mean(inside)),
                        "sampled_counts": float(np.sum(sampled[inside])),
                    }
                )
    rate = np.divide(counts, exposure, out=np.zeros_like(counts), where=exposure > 0.0)
    return rate, exposure, rows


def gas_surface_density(
    rate: np.ndarray,
    exposure: np.ndarray,
    *,
    grid: ClusterGrid,
    aperture_kpc: float,
    gas_mass_msun: float,
    morphology_exponent: float = 0.5,
    smoothing_kpc: float = 40.0,
    winsor_quantile: float = 0.995,
    background_mad_sigma: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Convert X-ray morphology into a normalized projected gas-mass map."""
    cleaned = median_filter(rate, size=5, mode="nearest")
    positive = cleaned[(cleaned > 0.0) & (exposure > 0.0)]
    if positive.size == 0:
        raise ValueError("Chandra rate map has no positive covered pixels")
    winsor_cap = float(np.quantile(positive, float(winsor_quantile)))
    cleaned = np.minimum(cleaned, winsor_cap)
    smoothed = gaussian_filter(cleaned, sigma=float(smoothing_kpc) / grid.cell_kpc)
    radius = np.hypot(grid.x_kpc, grid.y_kpc)
    annulus = (
        (radius >= 1.05 * float(aperture_kpc))
        & (radius <= 1.15 * float(aperture_kpc))
        & (exposure > 0.0)
    )
    if np.sum(annulus) < 100:
        raise ValueError("insufficient Chandra background annulus coverage")
    annulus_values = smoothed[annulus]
    background = float(np.median(annulus_values))
    robust_sigma = float(1.4826 * np.median(np.abs(annulus_values - background)))
    threshold = background + float(background_mad_sigma) * robust_sigma
    brightness = np.clip(smoothed - threshold, 0.0, None)
    provisional = brightness * (radius <= float(aperture_kpc))
    if np.sum(provisional) <= 0.0:
        raise ValueError("no positive X-ray gas signal inside normalization aperture")
    gas_x = float(np.sum(grid.x_kpc * provisional) / np.sum(provisional))
    gas_y = float(np.sum(grid.y_kpc * provisional) / np.sum(provisional))
    gas_radius = np.hypot(grid.x_kpc - gas_x, grid.y_kpc - gas_y)
    depth = 2.0 * np.sqrt(np.clip(float(aperture_kpc) ** 2 - gas_radius**2, 0.0, None))
    proxy = np.power(brightness, float(morphology_exponent)) * np.sqrt(depth)
    aperture = gas_radius <= float(aperture_kpc)
    normalization = float(np.sum(proxy[aperture]) * grid.cell_kpc**2)
    if normalization <= 0.0:
        raise ValueError("gas proxy has zero normalization")
    surface = proxy * float(gas_mass_msun) / normalization
    recovered = float(np.sum(surface[aperture]) * grid.cell_kpc**2)
    return surface, {
        "background_rate": background,
        "background_robust_sigma": robust_sigma,
        "background_threshold_rate": threshold,
        "winsor_cap_rate": winsor_cap,
        "gas_center_x_kpc": gas_x,
        "gas_center_y_kpc": gas_y,
        "gas_mass_msun": float(gas_mass_msun),
        "recovered_gas_mass_msun": recovered,
        "gas_mass_recovery_fraction": recovered / float(gas_mass_msun),
        "chandra_grid_coverage_fraction": float(np.mean(exposure > 0.0)),
    }


def surface_moments(surface: np.ndarray, grid: ClusterGrid) -> dict:
    """Return measurable centroid and low multipole morphology coordinates."""
    weight = np.asarray(surface, dtype=float) * grid.cell_kpc**2
    total = float(np.sum(weight))
    if total <= 0.0:
        raise ValueError("surface map has non-positive total mass")
    center_x = float(np.sum(weight * grid.x_kpc) / total)
    center_y = float(np.sum(weight * grid.y_kpc) / total)
    dx = grid.x_kpc - center_x
    dy = grid.y_kpc - center_y
    cxx = float(np.sum(weight * dx * dx) / total)
    cyy = float(np.sum(weight * dy * dy) / total)
    cxy = float(np.sum(weight * dx * dy) / total)
    eigenvalues, eigenvectors = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis_ratio = math.sqrt(max(float(np.min(eigenvalues)), 0.0) / float(np.max(eigenvalues)))
    position_angle = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0
    theta = np.arctan2(dy, dx)
    m1 = np.sum(weight * np.exp(1j * theta)) / total
    m2 = np.sum(weight * np.exp(2j * theta)) / total
    return {
        "mass_msun": total,
        "centroid_x_kpc": center_x,
        "centroid_y_kpc": center_y,
        "rms_radius_kpc": math.sqrt(cxx + cyy),
        "axis_ratio": axis_ratio,
        "position_angle_deg": position_angle,
        "m1_amplitude": float(abs(m1)),
        "m1_phase_deg": float(np.degrees(np.angle(m1)) % 360.0),
        "m2_amplitude": float(abs(m2)),
        "m2_phase_deg": float((0.5 * np.degrees(np.angle(m2))) % 180.0),
    }
