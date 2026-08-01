"""Resolved baryonic-map ingestion for field-equation tests."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io.fits import Header
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates

Array = np.ndarray
ARCSEC_PER_RADIAN = 206264.80624709636
HI_COLUMN_PER_K_KMS_CM2 = 1.823e18
HYDROGEN_COLUMN_PER_SOLAR_PC2 = 1.2489e20
HI_MASS_FACTOR = 2.356e5


@dataclass(frozen=True)
class DiskGeometry:
    center_x_pixel: float
    center_y_pixel: float
    major_x: float
    major_y: float
    inclination_deg: float

    @property
    def position_angle_pixel_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.major_y, self.major_x)) % 180.0)


def _image2(values: Array) -> Array:
    image = np.squeeze(np.asarray(values, dtype=float))
    if image.ndim != 2 or min(image.shape) < 5:
        raise ValueError("image must reduce to a finite two-dimensional array")
    if not np.all(np.isfinite(image)):
        raise ValueError("image must be finite")
    return image


def aips_clean_beam_degrees(header: Header | dict) -> tuple[float, float, float]:
    """Read a CLEAN beam, including AIPS files that store it in HISTORY."""

    if "BMAJ" in header and "BMIN" in header:
        return float(header["BMAJ"]), float(header["BMIN"]), float(header.get("BPA", 0.0))
    history = header.get("HISTORY", [])
    if isinstance(history, str):
        history = [history]
    pattern = re.compile(
        r"CLEAN\s+BMAJ=\s*([+\-0-9.Ee]+)\s+BMIN=\s*([+\-0-9.Ee]+)\s+BPA=\s*([+\-0-9.Ee]+)"
    )
    matches = [pattern.search(str(line)) for line in history]
    parsed = [match for match in matches if match is not None]
    if not parsed:
        raise ValueError("FITS header contains no BMAJ/BMIN or AIPS CLEAN beam")
    match = parsed[-1]
    return tuple(float(match.group(index)) for index in range(1, 4))


def hi_moment0_surface_density(
    moment0_jy_beam_m_s: Array,
    header: Header | dict,
    *,
    inclination_deg: float,
    helium_factor: float = 1.33,
    rest_frequency_hz: float | None = None,
) -> Array:
    """Convert a 21-cm moment-0 map to face-on gas mass per projected pc^2."""

    image = _image2(moment0_jy_beam_m_s)
    if not 0.0 <= inclination_deg < 90.0 or helium_factor <= 0.0:
        raise ValueError("inclination and helium factor are outside their physical ranges")
    bmaj_deg, bmin_deg, _ = aips_clean_beam_degrees(header)
    frequency_ghz = float(
        rest_frequency_hz if rest_frequency_hz is not None else header.get("RESTFREQ", 1.42040575e9)
    ) / 1e9
    bmaj_arcsec = bmaj_deg * 3600.0
    bmin_arcsec = bmin_deg * 3600.0
    kelvin_per_jy_beam = 1.222e6 / (frequency_ghz**2 * bmaj_arcsec * bmin_arcsec)
    brightness_k_kms = np.clip(image, 0.0, None) * kelvin_per_jy_beam / 1000.0
    column_hi_cm2 = HI_COLUMN_PER_K_KMS_CM2 * brightness_k_kms
    line_of_sight_hi = column_hi_cm2 / HYDROGEN_COLUMN_PER_SOLAR_PC2
    return line_of_sight_hi * float(helium_factor) * np.cos(np.radians(inclination_deg))


def integrated_hi_mass_solar(
    moment0_jy_beam_m_s: Array,
    header: Header | dict,
    *,
    distance_mpc: float,
) -> float:
    """Integrate a Jy/beam m/s map into the standard optically thin H I mass."""

    image = _image2(moment0_jy_beam_m_s)
    if distance_mpc <= 0.0:
        raise ValueError("distance_mpc must be positive")
    bmaj_deg, bmin_deg, _ = aips_clean_beam_degrees(header)
    pixel_area_deg2 = abs(float(header["CDELT1"]) * float(header["CDELT2"]))
    beam_area_deg2 = np.pi * bmaj_deg * bmin_deg / (4.0 * np.log(2.0))
    pixels_per_beam = beam_area_deg2 / pixel_area_deg2
    integrated_flux_jy_kms = float(np.sum(image) / pixels_per_beam / 1000.0)
    return HI_MASS_FACTOR * float(distance_mpc) ** 2 * integrated_flux_jy_kms


def weighted_disk_geometry(
    weights: Array,
    *,
    inclination_deg: float,
    center_hint: Sequence[float] | None = None,
    maximum_radius_pixel: float | None = None,
    quantile_floor: float = 0.0,
) -> DiskGeometry:
    """Estimate a centroid and major axis from a non-negative morphology map."""

    image = np.clip(_image2(weights), 0.0, None)
    yy, xx = np.indices(image.shape, dtype=float)
    keep = np.ones(image.shape, dtype=bool)
    if center_hint is not None and maximum_radius_pixel is not None:
        center_x, center_y = (float(value) for value in center_hint)
        keep &= np.hypot(xx - center_x, yy - center_y) <= float(maximum_radius_pixel)
    positive = image[keep & (image > 0.0)]
    if positive.size < 20:
        raise ValueError("too few positive morphology pixels")
    floor = float(np.quantile(positive, quantile_floor))
    morphology = np.where(keep & (image > floor), image - floor, 0.0)
    total = float(np.sum(morphology))
    if total <= 0.0:
        raise ValueError("morphology weights sum to zero")
    center_x = float(np.sum(morphology * xx) / total)
    center_y = float(np.sum(morphology * yy) / total)
    dx = xx - center_x
    dy = yy - center_y
    covariance = np.array(
        [
            [np.sum(morphology * dx * dx), np.sum(morphology * dx * dy)],
            [np.sum(morphology * dx * dy), np.sum(morphology * dy * dy)],
        ]
    ) / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    if major[0] < 0.0:
        major = -major
    return DiskGeometry(
        center_x_pixel=center_x,
        center_y_pixel=center_y,
        major_x=float(major[0]),
        major_y=float(major[1]),
        inclination_deg=float(inclination_deg),
    )


def deproject_to_disk_grid(
    sky_surface_density: Array,
    geometry: DiskGeometry,
    *,
    sky_pixel_scale_arcsec: float,
    distance_mpc: float,
    disk_axis_kpc: Array,
    surface_is_face_on: bool = True,
    interpolation_order: int = 1,
) -> Array:
    """Resample a projected sky image onto a face-on Cartesian disk grid."""

    image = _image2(sky_surface_density)
    axis = np.asarray(disk_axis_kpc, dtype=float)
    if axis.ndim != 1 or axis.size < 5 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("disk_axis_kpc must be a strictly increasing one-dimensional grid")
    if sky_pixel_scale_arcsec <= 0.0 or distance_mpc <= 0.0:
        raise ValueError("pixel scale and distance must be positive")
    x_disk, y_disk = np.meshgrid(axis, axis, indexing="ij")
    projected_minor = y_disk * np.cos(np.radians(geometry.inclination_deg))
    pixel_kpc = (
        float(distance_mpc) * 1000.0 * float(sky_pixel_scale_arcsec) / ARCSEC_PER_RADIAN
    )
    minor_x = -geometry.major_y
    minor_y = geometry.major_x
    source_x = geometry.center_x_pixel + (
        geometry.major_x * x_disk + minor_x * projected_minor
    ) / pixel_kpc
    source_y = geometry.center_y_pixel + (
        geometry.major_y * x_disk + minor_y * projected_minor
    ) / pixel_kpc
    sampled = map_coordinates(
        image,
        [source_y.ravel(), source_x.ravel()],
        order=int(interpolation_order),
        mode="constant",
        cval=0.0,
        prefilter=interpolation_order > 1,
    ).reshape(x_disk.shape)
    if surface_is_face_on:
        return np.clip(sampled, 0.0, None)
    return np.clip(sampled, 0.0, None) * np.cos(np.radians(geometry.inclination_deg))


def optical_morphology_map(
    counts: Array,
    *,
    center_hint: Sequence[float],
    maximum_radius_pixel: float,
    border_fraction: float = 0.15,
    cap_quantile: float = 0.995,
    cap_sigma_above_sky: float | None = None,
    smoothing_sigma_pixel: float = 3.0,
) -> tuple[Array, dict[str, float]]:
    """Make a robust stellar-light morphology map from a calibrated CCD image."""

    image = _image2(counts)
    ny, nx = image.shape
    border = np.ones(image.shape, dtype=bool)
    border[
        int(border_fraction * ny) : int((1.0 - border_fraction) * ny),
        int(border_fraction * nx) : int((1.0 - border_fraction) * nx),
    ] = False
    mean, median, standard_deviation = sigma_clipped_stats(
        image[border], sigma=3.0, maxiters=10
    )
    signal = np.clip(image - float(mean), 0.0, None)
    positive = signal[signal > 0.0]
    quantile_cap = float(np.quantile(positive, cap_quantile))
    sigma_cap = (
        np.inf
        if cap_sigma_above_sky is None
        else float(cap_sigma_above_sky) * float(standard_deviation)
    )
    cap = min(quantile_cap, sigma_cap)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("foreground cap is not finite and positive")
    signal = np.minimum(signal, cap)
    signal = gaussian_filter(signal, float(smoothing_sigma_pixel))
    yy, xx = np.indices(image.shape, dtype=float)
    keep = np.hypot(xx - float(center_hint[0]), yy - float(center_hint[1])) <= float(
        maximum_radius_pixel
    )
    signal = np.where(keep, signal, 0.0)
    return signal, {
        "sky_mean_counts": float(mean),
        "sky_median_counts": float(median),
        "sky_standard_deviation_counts": float(standard_deviation),
        "foreground_quantile_cap_counts": quantile_cap,
        "foreground_sigma_cap_counts": sigma_cap,
        "foreground_cap_counts": cap,
    }


def normalize_surface_density_mass(
    morphology: Array,
    *,
    pixel_size_kpc: float,
    total_mass_solar: float,
) -> Array:
    """Normalize a non-negative map to a declared total mass."""

    image = np.clip(_image2(morphology), 0.0, None)
    if pixel_size_kpc <= 0.0 or total_mass_solar < 0.0:
        raise ValueError("pixel size must be positive and mass non-negative")
    current = float(np.sum(image) * pixel_size_kpc**2)
    if current <= 0.0:
        raise ValueError("morphology has no positive integral")
    return image * float(total_mass_solar) / current


def disk_grid_sky_coordinates(
    center: SkyCoord,
    *,
    position_angle_deg: float,
    inclination_deg: float,
    distance_mpc: float,
    disk_axis_kpc: Array,
) -> SkyCoord:
    """Project a face-on disk grid onto the celestial sphere.

    Position angle is measured from north through east. The first disk axis is
    the photometric major axis and the second is foreshortened by inclination.
    """

    axis = np.asarray(disk_axis_kpc, dtype=float)
    if axis.ndim != 1 or axis.size < 5 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("disk_axis_kpc must be a strictly increasing one-dimensional grid")
    if not 0.0 <= inclination_deg < 90.0 or distance_mpc <= 0.0:
        raise ValueError("inclination or distance is outside its physical range")
    major, minor = np.meshgrid(axis, axis, indexing="ij")
    projected_minor = minor * np.cos(np.radians(inclination_deg))
    angle = np.radians(position_angle_deg)
    east_kpc = major * np.sin(angle) + projected_minor * np.cos(angle)
    north_kpc = major * np.cos(angle) - projected_minor * np.sin(angle)
    radians_per_kpc = 1.0 / (float(distance_mpc) * 1000.0)
    return center.spherical_offsets_by(
        east_kpc * radians_per_kpc * u.rad,
        north_kpc * radians_per_kpc * u.rad,
    )


def reproject_wcs_to_disk_grid(
    sky_image: Array,
    sky_wcs: WCS,
    *,
    center: SkyCoord,
    position_angle_deg: float,
    inclination_deg: float,
    distance_mpc: float,
    disk_axis_kpc: Array,
    interpolation_order: int = 1,
) -> Array:
    """Sample a registered sky image onto a common face-on disk grid."""

    image = _image2(sky_image)
    sky = disk_grid_sky_coordinates(
        center,
        position_angle_deg=position_angle_deg,
        inclination_deg=inclination_deg,
        distance_mpc=distance_mpc,
        disk_axis_kpc=disk_axis_kpc,
    )
    source_x, source_y = sky_wcs.celestial.world_to_pixel(sky)
    sampled = map_coordinates(
        image,
        [source_y.ravel(), source_x.ravel()],
        order=int(interpolation_order),
        mode="constant",
        cval=0.0,
        prefilter=interpolation_order > 1,
    ).reshape(sky.shape)
    return np.clip(sampled, 0.0, None)


def sky_pixels_to_disk_coordinates(
    pixel_x: Array,
    pixel_y: Array,
    sky_wcs: WCS,
    *,
    center: SkyCoord,
    position_angle_deg: float,
    inclination_deg: float,
    distance_mpc: float,
) -> tuple[Array, Array]:
    """Convert registered sky pixels to deprojected major/minor disk coordinates."""

    x = np.asarray(pixel_x, dtype=float)
    y = np.asarray(pixel_y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("pixel coordinate arrays must have the same shape")
    if not 0.0 <= inclination_deg < 90.0 or distance_mpc <= 0.0:
        raise ValueError("inclination or distance is outside its physical range")
    sky = sky_wcs.celestial.pixel_to_world(x, y).icrs
    reference = center.icrs
    east, north = reference.spherical_offsets_to(sky.frame)
    east_kpc = east.to_value(u.rad) * float(distance_mpc) * 1000.0
    north_kpc = north.to_value(u.rad) * float(distance_mpc) * 1000.0
    angle = np.radians(position_angle_deg)
    major = east_kpc * np.sin(angle) + north_kpc * np.cos(angle)
    projected_minor = east_kpc * np.cos(angle) - north_kpc * np.sin(angle)
    minor = projected_minor / np.cos(np.radians(inclination_deg))
    return np.asarray(major, dtype=float), np.asarray(minor, dtype=float)


def weighted_radius_quantile(
    x: Array, y: Array, weights: Array, quantile: float
) -> float:
    """Return a non-negative weighted quantile of radial distance."""

    x_values = np.asarray(x, dtype=float).ravel()
    y_values = np.asarray(y, dtype=float).ravel()
    weight_values = np.asarray(weights, dtype=float).ravel()
    if not (x_values.size == y_values.size == weight_values.size):
        raise ValueError("coordinates and weights must contain the same number of values")
    if not 0.0 < quantile <= 1.0 or np.any(weight_values < 0.0):
        raise ValueError("quantile or weights are outside their valid range")
    keep = np.isfinite(x_values) & np.isfinite(y_values) & (weight_values > 0.0)
    if not np.any(keep):
        raise ValueError("weighted radius sample is empty")
    radius = np.hypot(x_values[keep], y_values[keep])
    selected_weights = weight_values[keep]
    order = np.argsort(radius)
    cumulative = np.cumsum(selected_weights[order])
    target = quantile * cumulative[-1]
    return float(radius[order[min(int(np.searchsorted(cumulative, target)), len(order) - 1)]])


def resolved_map_morphology(
    surface_density: Array,
    *,
    disk_axis_kpc: Array,
    smoothing_sigma_pixel: float,
) -> dict[str, float]:
    """Compute outcome-blind concentration, lopsidedness, and clumpiness."""

    image = np.clip(_image2(surface_density), 0.0, None)
    axis = np.asarray(disk_axis_kpc, dtype=float)
    if image.shape != (len(axis), len(axis)) or smoothing_sigma_pixel <= 0.0:
        raise ValueError("surface map and disk axis are inconsistent")
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    weights = image.ravel()
    radius = np.hypot(xx, yy).ravel()
    order = np.argsort(radius)
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] <= 0.0:
        raise ValueError("surface map has no positive mass")

    def radius_at(fraction: float) -> float:
        index = min(
            int(np.searchsorted(cumulative, fraction * cumulative[-1])), len(order) - 1
        )
        return float(radius[order[index]])

    r20 = radius_at(0.2)
    r80 = radius_at(0.8)
    concentration = 5.0 * np.log10(r80 / max(r20, np.finfo(float).tiny))
    lopsidedness = float(
        np.sum(np.abs(image - image[::-1, ::-1])) / (2.0 * np.sum(image))
    )
    smooth = gaussian_filter(image, float(smoothing_sigma_pixel))
    clumpiness = float(np.sum(np.clip(image - smooth, 0.0, None)) / np.sum(image))
    return {
        "concentration_5log_r80_r20": float(concentration),
        "lopsidedness_180": lopsidedness,
        "clumpiness_positive_highpass": clumpiness,
        "r20_kpc": r20,
        "r80_kpc": r80,
    }
