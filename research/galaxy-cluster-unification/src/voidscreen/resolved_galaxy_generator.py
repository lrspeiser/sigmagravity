"""Gravity-independent extraction and generation of resolved galaxy mass maps.

The compact representation is deliberately about baryonic morphology, not a
law of gravity.  A component is represented by an azimuthal Fourier expansion
of its radial surface-density profile plus a small dictionary of signed,
localized Gaussian residual features.  The latter retain clumps, cavities, and
off-centre structure that a smooth radial model cannot express.

A two-dimensional surface-density map does not uniquely determine a
three-dimensional density field.  The helpers in this module therefore lift a
map through an explicitly declared vertical prior and preserve that ambiguity
instead of presenting one chosen thickness as a measurement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import rfc8785
from scipy.ndimage import gaussian_filter

from voidscreen.galaxy_maps import resolved_map_morphology

Array = np.ndarray
SCHEMA_VERSION = "1.0.0"
GENERATOR_NAME = "radial-fourier-sparse-residual"


def _regular_axis(axis_kpc: Array) -> tuple[Array, float]:
    axis = np.asarray(axis_kpc, dtype=float)
    if axis.ndim != 1 or axis.size < 9 or not np.all(np.isfinite(axis)):
        raise ValueError("axis_kpc must be a finite one-dimensional array with at least 9 cells")
    spacing = np.diff(axis)
    if not np.all(spacing > 0.0) or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-12):
        raise ValueError("axis_kpc must be strictly increasing and regularly spaced")
    return axis, float(spacing[0])


def _surface(surface_density: Array, axis: Array) -> Array:
    surface = np.asarray(surface_density, dtype=float)
    if surface.shape != (axis.size, axis.size):
        raise ValueError("surface density must have shape (len(axis_kpc), len(axis_kpc))")
    if not np.all(np.isfinite(surface)) or np.any(surface < 0.0):
        raise ValueError("surface density must be finite and non-negative")
    if not float(np.sum(surface)) > 0.0:
        raise ValueError("surface density must have positive mass")
    return surface


def _weighted_quantile(values: Array, weights: Array, quantile: float) -> float:
    order = np.argsort(values)
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    index = min(int(np.searchsorted(cumulative, quantile * cumulative[-1])), len(order) - 1)
    return float(values[order[index]])


def _fill_missing(values: Array, valid: Array) -> Array:
    locations = np.arange(len(values), dtype=float)
    if not np.any(valid):
        return np.zeros_like(values)
    return np.interp(locations, locations[valid], values[valid])


def _normalize_mass(surface: Array, target_mass_solar: float, spacing_kpc: float) -> Array:
    clipped = np.clip(np.asarray(surface, dtype=float), 0.0, None)
    mass = float(np.sum(clipped) * spacing_kpc**2)
    if not mass > 0.0:
        raise ValueError("generated component has no positive mass")
    return clipped * (float(target_mass_solar) / mass)


def _base_component(
    parameters: Mapping[str, Any],
    axis_kpc: Array,
    *,
    mass_scale: float,
    radial_scale: float,
    fourier_scale: float,
    residual_scale: float,
    rotation_deg: float,
    center_offset_kpc: Sequence[float],
    axis_ratio_scale: float,
) -> Array:
    axis, spacing = _regular_axis(axis_kpc)
    if mass_scale <= 0.0 or radial_scale <= 0.0 or axis_ratio_scale <= 0.0:
        raise ValueError("mass_scale, radial_scale, and axis_ratio_scale must be positive")
    if fourier_scale < 0.0 or residual_scale < 0.0:
        raise ValueError("fourier_scale and residual_scale must be non-negative")
    if len(center_offset_kpc) != 2:
        raise ValueError("center_offset_kpc must contain x and y")

    center = np.asarray(parameters["centroid_kpc"], dtype=float) + np.asarray(
        center_offset_kpc, dtype=float
    )
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    angle = np.radians(float(rotation_deg))
    cosine, sine = np.cos(angle), np.sin(angle)
    dx = xx - center[0]
    dy = yy - center[1]
    source_x = (cosine * dx + sine * dy) / radial_scale
    source_y = (-sine * dx + cosine * dy) / (radial_scale * axis_ratio_scale)
    radius = np.hypot(source_x, source_y)
    phi = np.arctan2(source_y, source_x)

    radial = parameters["radial_fourier"]
    centres = np.asarray(radial["radius_centers_kpc"], dtype=float)
    axisymmetric = np.asarray(radial["axisymmetric_surface_density"], dtype=float)
    model = np.interp(radius, centres, axisymmetric, left=axisymmetric[0], right=0.0)
    for mode in radial["modes"]:
        m = int(mode["m"])
        cosine_coefficient = np.interp(
            radius,
            centres,
            np.asarray(mode["cosine_surface_density"], dtype=float),
            left=float(mode["cosine_surface_density"][0]),
            right=0.0,
        )
        sine_coefficient = np.interp(
            radius,
            centres,
            np.asarray(mode["sine_surface_density"], dtype=float),
            left=float(mode["sine_surface_density"][0]),
            right=0.0,
        )
        model += fourier_scale * (
            cosine_coefficient * np.cos(m * phi) + sine_coefficient * np.sin(m * phi)
        )
    model = np.where(radius <= float(radial["maximum_radius_kpc"]), model, 0.0)

    for feature in parameters["residual_features"]:
        feature_x = float(feature["x_kpc"]) - float(parameters["centroid_kpc"][0])
        feature_y = float(feature["y_kpc"]) - float(parameters["centroid_kpc"][1])
        transformed_x = center[0] + radial_scale * (
            cosine * feature_x - sine * axis_ratio_scale * feature_y
        )
        transformed_y = center[1] + radial_scale * axis_ratio_scale * (
            sine * feature_x + cosine * feature_y
        )
        sigma = radial_scale * float(feature["sigma_kpc"])
        feature_dx = xx - transformed_x
        feature_dy = yy - transformed_y
        feature_source_x = cosine * feature_dx + sine * feature_dy
        feature_source_y = (-sine * feature_dx + cosine * feature_dy) / axis_ratio_scale
        gaussian = np.exp(
            -0.5 * (feature_source_x**2 + feature_source_y**2) / sigma**2
        )
        model += residual_scale * float(feature["amplitude_surface_density"]) * gaussian

    target_mass = float(parameters["mass_solar"]) * float(mass_scale)
    return _normalize_mass(model, target_mass, spacing)


def extract_component_parameters(
    surface_density: Array,
    axis_kpc: Array,
    *,
    radial_bins: int = 24,
    maximum_fourier_mode: int = 4,
    residual_feature_count: int = 64,
    residual_sigma_pixels: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
) -> dict[str, Any]:
    """Extract a compact, deterministic morphology description from one map."""

    axis, spacing = _regular_axis(axis_kpc)
    surface = _surface(surface_density, axis)
    if radial_bins < 6 or maximum_fourier_mode < 0 or residual_feature_count < 0:
        raise ValueError("extraction controls are outside their supported ranges")
    sigmas = np.asarray(residual_sigma_pixels, dtype=float)
    if sigmas.ndim != 1 or sigmas.size == 0 or np.any(sigmas <= 0.0):
        raise ValueError("residual_sigma_pixels must contain positive values")

    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    weight_sum = float(np.sum(surface))
    centroid_x = float(np.sum(surface * xx) / weight_sum)
    centroid_y = float(np.sum(surface * yy) / weight_sum)
    dx = xx - centroid_x
    dy = yy - centroid_y
    radius = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    maximum_radius = _weighted_quantile(radius.ravel(), surface.ravel(), 0.9995) + spacing
    maximum_radius = max(maximum_radius, 3.0 * spacing)
    edges = np.linspace(0.0, maximum_radius, radial_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    annulus = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, radial_bins - 1)
    inside = radius.ravel() <= maximum_radius

    axisymmetric = np.zeros(radial_bins, dtype=float)
    valid = np.zeros(radial_bins, dtype=bool)
    cosine_coefficients = np.zeros((maximum_fourier_mode, radial_bins), dtype=float)
    sine_coefficients = np.zeros_like(cosine_coefficients)
    flat_surface = surface.ravel()
    flat_phi = phi.ravel()
    for index in range(radial_bins):
        selected = inside & (annulus == index)
        if not np.any(selected):
            continue
        valid[index] = True
        values = flat_surface[selected]
        angles = flat_phi[selected]
        axisymmetric[index] = float(np.mean(values))
        for mode_index, mode in enumerate(range(1, maximum_fourier_mode + 1)):
            cosine_coefficients[mode_index, index] = 2.0 * float(
                np.mean(values * np.cos(mode * angles))
            )
            sine_coefficients[mode_index, index] = 2.0 * float(
                np.mean(values * np.sin(mode * angles))
            )
    axisymmetric = _fill_missing(axisymmetric, valid)
    for index in range(maximum_fourier_mode):
        cosine_coefficients[index] = _fill_missing(cosine_coefficients[index], valid)
        sine_coefficients[index] = _fill_missing(sine_coefficients[index], valid)

    modes = [
        {
            "m": mode,
            "cosine_surface_density": cosine_coefficients[mode - 1].tolist(),
            "sine_surface_density": sine_coefficients[mode - 1].tolist(),
        }
        for mode in range(1, maximum_fourier_mode + 1)
    ]
    parameters: dict[str, Any] = {
        "mass_solar": float(np.sum(surface) * spacing**2),
        "centroid_kpc": [centroid_x, centroid_y],
        "radial_fourier": {
            "maximum_radius_kpc": float(maximum_radius),
            "radius_centers_kpc": centres.tolist(),
            "axisymmetric_surface_density": axisymmetric.tolist(),
            "modes": modes,
        },
        "residual_features": [],
    }
    base = _base_component(
        parameters,
        axis,
        mass_scale=1.0,
        radial_scale=1.0,
        fourier_scale=1.0,
        residual_scale=0.0,
        rotation_deg=0.0,
        center_offset_kpc=(0.0, 0.0),
        axis_ratio_scale=1.0,
    )
    residual = surface - base

    for _ in range(residual_feature_count):
        best: tuple[float, float, int, int, float, Array] | None = None
        for sigma_pixel in sigmas:
            smoothed = gaussian_filter(residual, float(sigma_pixel), mode="constant")
            location = np.unravel_index(int(np.argmax(np.abs(smoothed))), smoothed.shape)
            centre_x = axis[location[0]]
            centre_y = axis[location[1]]
            sigma_kpc = float(sigma_pixel * spacing)
            gaussian = np.exp(
                -0.5 * ((xx - centre_x) ** 2 + (yy - centre_y) ** 2) / sigma_kpc**2
            )
            denominator = float(np.sum(gaussian * gaussian))
            amplitude = float(np.sum(residual * gaussian) / denominator)
            reduction = amplitude**2 * denominator
            candidate = (
                reduction,
                float(sigma_kpc),
                int(location[0]),
                int(location[1]),
                amplitude,
                gaussian,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None or best[0] <= np.finfo(float).eps:
            break
        _, sigma_kpc, x_index, y_index, amplitude, gaussian = best
        parameters["residual_features"].append(
            {
                "x_kpc": float(axis[x_index]),
                "y_kpc": float(axis[y_index]),
                "sigma_kpc": sigma_kpc,
                "amplitude_surface_density": amplitude,
            }
        )
        residual = residual - amplitude * gaussian

    return parameters


def render_component(
    parameters: Mapping[str, Any],
    axis_kpc: Array,
    *,
    mass_scale: float = 1.0,
    radial_scale: float = 1.0,
    fourier_scale: float = 1.0,
    residual_scale: float = 1.0,
    rotation_deg: float = 0.0,
    center_offset_kpc: Sequence[float] = (0.0, 0.0),
    axis_ratio_scale: float = 1.0,
) -> Array:
    """Render one extracted component, optionally changing generative controls."""

    return _base_component(
        parameters,
        axis_kpc,
        mass_scale=mass_scale,
        radial_scale=radial_scale,
        fourier_scale=fourier_scale,
        residual_scale=residual_scale,
        rotation_deg=rotation_deg,
        center_offset_kpc=center_offset_kpc,
        axis_ratio_scale=axis_ratio_scale,
    )


def package_content_hash(package: Mapping[str, Any]) -> str:
    """Return a deterministic hash excluding a pre-existing hash field."""

    payload = dict(package)
    payload.pop("contentSha256", None)
    encoded = rfc8785.dumps(payload)
    return hashlib.sha256(encoded).hexdigest()


def extract_galaxy_parameters(
    galaxy: str,
    axis_kpc: Array,
    gas_surface_density: Array,
    stellar_surface_density: Array,
    *,
    source_observables: Mapping[str, Any] | None = None,
    radial_bins: int = 24,
    maximum_fourier_mode: int = 4,
    residual_feature_count: int = 64,
) -> dict[str, Any]:
    """Extract a gravity-independent, JSON-serializable galaxy package."""

    axis, spacing = _regular_axis(axis_kpc)
    gas = _surface(gas_surface_density, axis)
    stars = _surface(stellar_surface_density, axis)
    controls = {
        "radial_bins": int(radial_bins),
        "maximum_fourier_mode": int(maximum_fourier_mode),
        "residual_feature_count_per_component": int(residual_feature_count),
        "residual_sigma_pixels": [1.0, 2.0, 4.0, 8.0],
    }
    package: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "generator": GENERATOR_NAME,
        "galaxy": str(galaxy),
        "sourceObservables": dict(source_observables or {}),
        "grid": {
            "cellsPerAxis": int(axis.size),
            "minimumKpc": float(axis[0]),
            "maximumKpc": float(axis[-1]),
            "spacingKpc": spacing,
        },
        "extractionControls": controls,
        "components": {
            "gas": extract_component_parameters(
                gas,
                axis,
                radial_bins=radial_bins,
                maximum_fourier_mode=maximum_fourier_mode,
                residual_feature_count=residual_feature_count,
            ),
            "stars": extract_component_parameters(
                stars,
                axis,
                radial_bins=radial_bins,
                maximum_fourier_mode=maximum_fourier_mode,
                residual_feature_count=residual_feature_count,
            ),
        },
        "gravityParameters": {},
        "velocityTargetsUsed": False,
        "verticalStructure": {
            "status": "assumed_prior_not_measured",
            "profiles": ["exponential", "sech_squared"],
            "warning": "A single surface-density map does not identify a unique 3D density.",
        },
    }
    package["contentSha256"] = package_content_hash(package)
    return package


def render_galaxy(
    package: Mapping[str, Any],
    axis_kpc: Array | None = None,
    *,
    component_controls: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Array]:
    """Render gas, stars, and total baryons from an extracted package."""

    if axis_kpc is None:
        grid = package["grid"]
        axis_kpc = np.linspace(
            float(grid["minimumKpc"]), float(grid["maximumKpc"]), int(grid["cellsPerAxis"])
        )
    controls = component_controls or {}
    rendered: dict[str, Array] = {}
    for name in ("gas", "stars"):
        rendered[name] = render_component(
            package["components"][name], axis_kpc, **dict(controls.get(name, {}))
        )
    rendered["total"] = rendered["gas"] + rendered["stars"]
    return rendered


def lift_surface_density_to_volume(
    surface_density: Array,
    axis_kpc: Array,
    z_axis_kpc: Array,
    *,
    scale_height_kpc: float | Array,
    profile: str = "sech_squared",
    midplane_offset_kpc: float | Array = 0.0,
) -> Array:
    """Lift a 2D map to 3D while preserving every projected mass column."""

    axis, _ = _regular_axis(axis_kpc)
    surface = _surface(surface_density, axis)
    z_axis = np.asarray(z_axis_kpc, dtype=float)
    if z_axis.ndim != 1 or z_axis.size < 9 or not np.all(np.diff(z_axis) > 0.0):
        raise ValueError("z_axis_kpc must be a strictly increasing one-dimensional grid")
    if not np.allclose(np.diff(z_axis), np.diff(z_axis)[0], rtol=1e-8, atol=1e-12):
        raise ValueError("z_axis_kpc must be regularly spaced")
    height = np.asarray(scale_height_kpc, dtype=float)
    if height.ndim == 0:
        height = np.full(surface.shape, float(height))
    if height.shape != surface.shape or not np.all(np.isfinite(height)) or np.any(height <= 0.0):
        raise ValueError("scale_height_kpc must be positive and scalar or match the surface map")
    midplane = np.asarray(midplane_offset_kpc, dtype=float)
    if midplane.ndim == 0:
        midplane = np.full(surface.shape, float(midplane))
    if midplane.shape != surface.shape or not np.all(np.isfinite(midplane)):
        raise ValueError("midplane_offset_kpc must be finite and scalar or match the surface map")
    scaled_z = np.abs(z_axis[None, None, :] - midplane[:, :, None]) / height[:, :, None]
    if profile == "exponential":
        weights = np.exp(-scaled_z)
    elif profile == "sech_squared":
        weights = 1.0 / np.cosh(np.clip(scaled_z, 0.0, 350.0)) ** 2
    else:
        raise ValueError("profile must be 'exponential' or 'sech_squared'")
    dz = float(z_axis[1] - z_axis[0])
    weights /= np.sum(weights, axis=2, keepdims=True) * dz
    return surface[:, :, None] * weights


def sample_vertical_realization(
    surface_density: Array,
    axis_kpc: Array,
    z_axis_kpc: Array,
    *,
    r80_kpc: float,
    component: str,
    rng: np.random.Generator,
    profile: str | None = None,
    scale_height_log_sigma: float = 0.35,
    flaring_max: float = 1.0,
    warp_amplitude_deg: float = 0.0,
    warp_phase_deg: float = 0.0,
) -> tuple[Array, dict[str, Any]]:
    """Draw one declared disk-thickness prior and return its 3D density."""

    axis, spacing = _regular_axis(axis_kpc)
    if component not in {"gas", "stars"} or r80_kpc <= 0.0:
        raise ValueError("component or r80_kpc is invalid")
    if scale_height_log_sigma < 0.0 or flaring_max < 0.0:
        raise ValueError("vertical prior widths must be non-negative")
    if not np.isfinite(warp_amplitude_deg) or abs(warp_amplitude_deg) > 45.0:
        raise ValueError("warp_amplitude_deg must be finite and within +/-45 degrees")
    median_fraction = 0.12 if component == "gas" else 0.08
    base_height = max(spacing, median_fraction * float(r80_kpc)) * float(
        np.exp(rng.normal(0.0, scale_height_log_sigma))
    )
    flaring = float(rng.uniform(0.0, flaring_max))
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(xx, yy)
    scale_height = base_height * (1.0 + flaring * radius / float(r80_kpc))
    phi = np.arctan2(yy, xx)
    warp_start = 0.5 * float(r80_kpc)
    warp_slope = np.tan(np.radians(float(warp_amplitude_deg)))
    midplane_offset = (
        warp_slope
        * np.clip(radius - warp_start, 0.0, None)
        * np.cos(phi - np.radians(float(warp_phase_deg)))
    )
    chosen_profile = profile or ("sech_squared" if rng.random() < 0.5 else "exponential")
    density = lift_surface_density_to_volume(
        surface_density,
        axis,
        z_axis_kpc,
        scale_height_kpc=scale_height,
        profile=chosen_profile,
        midplane_offset_kpc=midplane_offset,
    )
    return density, {
        "status": "assumed_prior_not_measured",
        "component": component,
        "profile": chosen_profile,
        "baseScaleHeightKpc": float(base_height),
        "flaringPerR80": flaring,
        "r80Kpc": float(r80_kpc),
        "warpAmplitudeDeg": float(warp_amplitude_deg),
        "warpPhaseDeg": float(warp_phase_deg),
        "warpStartR80": 0.5,
    }


def _radial_profile(surface: Array, axis: Array, bins: int = 24) -> Array:
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(xx, yy)
    edges = np.linspace(0.0, float(np.max(radius)), bins + 1)
    indices = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, bins - 1)
    profile = np.zeros(bins, dtype=float)
    for index in range(bins):
        selected = indices == index
        profile[index] = float(np.mean(surface.ravel()[selected])) if np.any(selected) else 0.0
    return profile


def roundtrip_metrics(reference: Array, generated: Array, axis_kpc: Array) -> dict[str, float]:
    """Measure pixel, radial-profile, mass, and morphology reconstruction error."""

    axis, spacing = _regular_axis(axis_kpc)
    expected = _surface(reference, axis)
    actual = _surface(generated, axis)
    reference_mass = float(np.sum(expected) * spacing**2)
    generated_mass = float(np.sum(actual) * spacing**2)
    normalized_l2 = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    correlation = float(np.corrcoef(expected.ravel(), actual.ravel())[0, 1])
    reference_profile = _radial_profile(expected, axis)
    generated_profile = _radial_profile(actual, axis)
    peak = max(float(np.max(reference_profile)), np.finfo(float).tiny)
    log_profile_rmse = float(
        np.sqrt(
            np.mean(
                (
                    np.log10(generated_profile / peak + 1e-6)
                    - np.log10(reference_profile / peak + 1e-6)
                )
                ** 2
            )
        )
    )
    reference_morphology = resolved_map_morphology(
        expected, disk_axis_kpc=axis, smoothing_sigma_pixel=2.0
    )
    generated_morphology = resolved_map_morphology(
        actual, disk_axis_kpc=axis, smoothing_sigma_pixel=2.0
    )
    return {
        "mass_relative_error": abs(generated_mass - reference_mass) / reference_mass,
        "normalized_l2": normalized_l2,
        "pixel_correlation": correlation,
        "radial_profile_log10_rmse": log_profile_rmse,
        "concentration_absolute_error": abs(
            generated_morphology["concentration_5log_r80_r20"]
            - reference_morphology["concentration_5log_r80_r20"]
        ),
        "lopsidedness_absolute_error": abs(
            generated_morphology["lopsidedness_180"]
            - reference_morphology["lopsidedness_180"]
        ),
        "clumpiness_absolute_error": abs(
            generated_morphology["clumpiness_positive_highpass"]
            - reference_morphology["clumpiness_positive_highpass"]
        ),
    }
