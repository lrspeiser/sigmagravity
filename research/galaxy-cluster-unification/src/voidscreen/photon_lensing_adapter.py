"""Typed projection of a solved 3D photon field into sky-lensing maps.

The adapter consumes an acceleration-like observable whose manifest target is
``photons`` or ``both``.  It does not inspect, fit, or modify the field
equations.  Sky axes, distance geometry, and every scored observation array
are explicit inputs so that no cosmology or massive-tracer rule is inferred.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from .sky_lensing import (
    C_M_S,
    RAD_TO_ARCSEC,
    SkyPhotonDeflection2D,
    photon_deflection_sky,
)

Array = np.ndarray


def _finite_positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _target_array(
    arrays: Mapping[str, Array] | None,
    key: Any,
    label: str,
    *,
    shape: tuple[int, int],
) -> Array:
    if arrays is None:
        raise ValueError(f"{label} requires observation arrays")
    if not isinstance(key, str) or not key:
        raise ValueError(f"{label} requires a non-empty array key")
    if key not in arrays:
        raise ValueError(f"{label} references missing array {key}")
    value = np.asarray(arrays[key], dtype=float)
    if value.shape != shape:
        raise ValueError(f"{label} array must have shape {shape}")
    return value


def _optional_mask(
    target: Mapping[str, Any],
    arrays: Mapping[str, Array] | None,
    shape: tuple[int, int],
) -> Array:
    key = target.get("scoreMaskArrayKey")
    if key is None:
        return np.ones(shape, dtype=bool)
    value = _target_array(arrays, key, "scoreMaskArrayKey", shape=shape)
    return np.isfinite(value) & (value > 0.0)


def _score_component_pair(
    *,
    channel: str,
    unit: str,
    predicted_first: Array,
    predicted_second: Array,
    observed_first_key: str,
    observed_second_key: str,
    uncertainty_key: str,
    target: Mapping[str, Any],
    arrays: Mapping[str, Array] | None,
    support: Array,
    fitted_nuisance_parameters: int,
) -> dict[str, Any]:
    keys = (
        target.get(observed_first_key),
        target.get(observed_second_key),
        target.get(uncertainty_key),
    )
    if all(key is None for key in keys):
        valid = support & np.isfinite(predicted_first) & np.isfinite(predicted_second)
        return {
            "channel": channel,
            "unit": unit,
            "state": "predicted_not_scored",
            "totalPoints": int(2 * predicted_first.size),
            "validPoints": int(2 * valid.sum()),
            "fittedNuisanceParameters": 0,
        }
    if any(key is None for key in keys):
        raise ValueError(
            f"{channel} scoring requires both observed component maps and its uncertainty map"
        )
    shape = predicted_first.shape
    observed_first = _target_array(
        arrays, keys[0], observed_first_key, shape=shape
    )
    observed_second = _target_array(
        arrays, keys[1], observed_second_key, shape=shape
    )
    uncertainty = _target_array(arrays, keys[2], uncertainty_key, shape=shape)
    valid = (
        support
        & np.isfinite(predicted_first)
        & np.isfinite(predicted_second)
        & np.isfinite(observed_first)
        & np.isfinite(observed_second)
        & np.isfinite(uncertainty)
        & (uncertainty > 0.0)
    )
    valid_pixels = int(valid.sum())
    valid_points = 2 * valid_pixels
    minimum_valid = target.get("minimumValidPixels", 25)
    if (
        isinstance(minimum_valid, bool)
        or not isinstance(minimum_valid, int)
        or minimum_valid < 1
    ):
        raise ValueError("minimumValidPixels must be a positive integer")
    if valid_pixels < minimum_valid:
        raise ValueError(
            f"{channel} has too few valid pixels: {valid_pixels} < {minimum_valid}"
        )
    if fitted_nuisance_parameters >= valid_points:
        raise ValueError(
            "fittedNuisanceParameters must be smaller than the scored point count"
        )
    residual_first = predicted_first[valid] - observed_first[valid]
    residual_second = predicted_second[valid] - observed_second[valid]
    residual = np.concatenate([residual_first, residual_second])
    sigma = np.concatenate([uncertainty[valid], uncertainty[valid]])
    inverse_variance = 1.0 / np.square(sigma)
    squared = np.square(residual)
    sum_squared = float(np.sum(squared))
    weighted_squared = float(np.sum(inverse_variance * squared))
    weight_sum = float(np.sum(inverse_variance))
    degrees_freedom = valid_points - fitted_nuisance_parameters
    chi_square = float(np.sum(np.square(residual / sigma)))
    gaussian_log_likelihood = -0.5 * (
        chi_square
        + float(np.sum(np.log(np.square(sigma))))
        + valid_points * math.log(2.0 * math.pi)
    )
    return {
        "channel": channel,
        "unit": unit,
        "state": "scored",
        "totalPoints": int(2 * predicted_first.size),
        "validPoints": valid_points,
        "validPixels": valid_pixels,
        "fittedNuisanceParameters": fitted_nuisance_parameters,
        "sumSquaredResidual": sum_squared,
        "rmse": math.sqrt(sum_squared / valid_points),
        "inverseVarianceWeightedSquaredResidual": weighted_squared,
        "inverseVarianceWeightSum": weight_sum,
        "inverseVarianceWeightedRmse": math.sqrt(weighted_squared / weight_sum),
        "chiSquare": chi_square,
        "degreesFreedom": degrees_freedom,
        "reducedChiSquare": chi_square / degrees_freedom,
        "gaussianLogLikelihood": gaussian_log_likelihood,
    }


def _lensing_invariants(
    alpha_east_radian: Array,
    alpha_north_radian: Array,
    *,
    east_spacing_radian: float,
    north_spacing_radian: float,
) -> dict[str, Array]:
    edge_order = 2 if min(alpha_east_radian.shape) >= 3 else 1
    d_east_d_east = np.gradient(
        alpha_east_radian, east_spacing_radian, axis=1, edge_order=edge_order
    )
    d_east_d_north = np.gradient(
        alpha_east_radian, north_spacing_radian, axis=0, edge_order=edge_order
    )
    d_north_d_east = np.gradient(
        alpha_north_radian, east_spacing_radian, axis=1, edge_order=edge_order
    )
    d_north_d_north = np.gradient(
        alpha_north_radian, north_spacing_radian, axis=0, edge_order=edge_order
    )
    jacobian_ee = 1.0 - d_east_d_east
    jacobian_en = -d_east_d_north
    jacobian_ne = -d_north_d_east
    jacobian_nn = 1.0 - d_north_d_north
    convergence = 0.5 * (d_east_d_east + d_north_d_north)
    shear_1 = 0.5 * (d_east_d_east - d_north_d_north)
    shear_2 = 0.5 * (d_east_d_north + d_north_d_east)
    rotation = 0.5 * (d_north_d_east - d_east_d_north)
    determinant = jacobian_ee * jacobian_nn - jacobian_en * jacobian_ne
    symmetric = np.empty(alpha_east_radian.shape + (2, 2), dtype=float)
    symmetric[..., 0, 0] = jacobian_ee
    symmetric[..., 1, 1] = jacobian_nn
    symmetric[..., 0, 1] = symmetric[..., 1, 0] = 0.5 * (
        jacobian_en + jacobian_ne
    )
    eigenvalues = np.linalg.eigvalsh(symmetric)
    denominator = 1.0 - convergence
    reduced_1 = np.divide(
        shear_1,
        denominator,
        out=np.full_like(shear_1, np.nan),
        where=np.abs(denominator) > 1.0e-12,
    )
    reduced_2 = np.divide(
        shear_2,
        denominator,
        out=np.full_like(shear_2, np.nan),
        where=np.abs(denominator) > 1.0e-12,
    )
    return {
        "convergence": convergence,
        "shear_1": shear_1,
        "shear_2": shear_2,
        "shear_magnitude": np.hypot(shear_1, shear_2),
        "reduced_shear_1": reduced_1,
        "reduced_shear_2": reduced_2,
        "rotation": rotation,
        "jacobian_determinant": determinant,
        "minimum_jacobian_eigenvalue": eigenvalues[..., 0],
        "maximum_jacobian_eigenvalue": eigenvalues[..., 1],
        "absolute_magnification": 1.0 / np.maximum(np.abs(determinant), 1.0e-12),
    }


def _axisymmetric_photon_deflection(
    components: tuple[Array, Array],
    geometry: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    distance_ratio: float,
) -> tuple[Any, dict[str, Any]]:
    """Project an ``(a_r,a_z)`` field through its finite cylindrical domain."""

    if list(geometry.get("axisOrder", [])) != ["r", "z"]:
        raise ValueError(
            "axisymmetric photon lensing requires geometry axisOrder=['r','z']"
        )
    radial, vertical = components
    if radial.ndim != 2 or vertical.ndim != 2 or radial.shape != vertical.shape:
        raise ValueError("axisymmetric photon observable must provide matching 2D (r,z) components")
    if min(radial.shape) < 3:
        raise ValueError("axisymmetric photon field must contain at least three cells per axis")
    raw_spacing = geometry.get("spacing")
    spacing = (
        np.full(2, float(raw_spacing))
        if isinstance(raw_spacing, (int, float))
        else np.asarray(raw_spacing, dtype=float)
    )
    if spacing.shape != (2,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("axisymmetric geometry spacing must contain two positive finite values")
    raw_origin = target.get("gridOriginM", geometry.get("origin"))
    origin = np.asarray(raw_origin, dtype=float)
    if origin.shape != (2,) or np.any(~np.isfinite(origin)):
        raise ValueError("axisymmetric photon lensing requires an explicit origin=[0,z0]")
    if origin[0] != 0.0:
        raise ValueError("axisymmetric photon-lensing radial origin must be exactly r=0")

    raw_shape = target.get("skyShape")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_shape)
        or any(value < 3 or value > 513 for value in raw_shape)
    ):
        raise ValueError("skyShape must contain two integers from 3 through 513")
    sky_shape = (int(raw_shape[0]), int(raw_shape[1]))
    line_samples = target.get("lineOfSightSamples")
    if (
        isinstance(line_samples, bool)
        or not isinstance(line_samples, int)
        or line_samples < 3
        or line_samples > 2049
    ):
        raise ValueError("lineOfSightSamples must be an integer from 3 through 2049")
    if sky_shape[0] * sky_shape[1] * line_samples > 16_777_216:
        raise ValueError("axisymmetric photon projection exceeds 16,777,216 path samples")
    inclination_deg = float(target.get("axisymmetricInclinationDeg"))
    if not math.isfinite(inclination_deg) or not 0.0 <= inclination_deg <= 90.0:
        raise ValueError("axisymmetricInclinationDeg must lie in [0,90]")

    radial_max = float((radial.shape[0] - 1) * spacing[0])
    vertical_min = float(origin[1])
    vertical_max = float(origin[1] + (radial.shape[1] - 1) * spacing[1])
    inclination = math.radians(inclination_deg)
    sin_i = math.sin(inclination)
    cos_i = math.cos(inclination)
    north_bounds = (
        -radial_max * cos_i + vertical_min * sin_i,
        radial_max * cos_i + vertical_max * sin_i,
    )
    east_bounds = (-radial_max, radial_max)
    north_axis = np.linspace(*north_bounds, sky_shape[0])
    east_axis = np.linspace(*east_bounds, sky_shape[1])
    east_grid = np.broadcast_to(east_axis[None, :], sky_shape)
    north_grid = np.broadcast_to(north_axis[:, None], sky_shape)

    lower = np.full(sky_shape, -np.inf, dtype=float)
    upper = np.full(sky_shape, np.inf, dtype=float)
    valid = np.ones(sky_shape, dtype=bool)
    radial_available = np.square(radial_max) - np.square(east_grid)
    valid &= radial_available >= -1.0e-12 * max(radial_max**2, 1.0)
    radial_root = np.sqrt(np.maximum(radial_available, 0.0))
    tolerance = 32.0 * np.finfo(float).eps
    if sin_i > tolerance:
        lower = np.maximum(lower, (north_grid * cos_i - radial_root) / sin_i)
        upper = np.minimum(upper, (north_grid * cos_i + radial_root) / sin_i)
    else:
        valid &= np.hypot(north_grid, east_grid) <= radial_max * (1.0 + 1.0e-12)
    if cos_i > tolerance:
        lower = np.maximum(lower, (vertical_min - north_grid * sin_i) / cos_i)
        upper = np.minimum(upper, (vertical_max - north_grid * sin_i) / cos_i)
    else:
        intrinsic_z = north_grid * sin_i
        valid &= (intrinsic_z >= vertical_min) & (intrinsic_z <= vertical_max)
    valid &= np.isfinite(lower) & np.isfinite(upper) & (upper > lower)

    alpha_east = np.full(sky_shape, np.nan, dtype=float)
    alpha_north = np.full(sky_shape, np.nan, dtype=float)
    path_lengths = np.where(valid, upper - lower, np.nan)
    path_fraction = np.linspace(0.0, 1.0, line_samples)
    multiplier = -2.0 * distance_ratio / C_M_S**2
    for row in range(sky_shape[0]):
        columns = np.flatnonzero(valid[row])
        if not len(columns):
            continue
        row_lower = lower[row, columns]
        row_upper = upper[row, columns]
        line_of_sight = row_lower[:, None] + (
            row_upper - row_lower
        )[:, None] * path_fraction[None, :]
        north = north_axis[row]
        east = east_axis[columns, None]
        intrinsic_x = -north * cos_i + line_of_sight * sin_i
        intrinsic_y = np.broadcast_to(east, intrinsic_x.shape)
        intrinsic_z = north * sin_i + line_of_sight * cos_i
        cylindrical_r = np.hypot(intrinsic_x, intrinsic_y)
        radial_index = np.clip(cylindrical_r / spacing[0], 0.0, radial.shape[0] - 1)
        vertical_index = np.clip(
            (intrinsic_z - vertical_min) / spacing[1], 0.0, radial.shape[1] - 1
        )
        sample_indices = np.vstack([radial_index.ravel(), vertical_index.ravel()])
        sampled_radial = map_coordinates(
            radial,
            sample_indices,
            order=1,
            mode="nearest",
            prefilter=False,
        ).reshape(radial_index.shape)
        sampled_vertical = map_coordinates(
            vertical,
            sample_indices,
            order=1,
            mode="nearest",
            prefilter=False,
        ).reshape(radial_index.shape)
        radial_direction_x = np.divide(
            intrinsic_x,
            cylindrical_r,
            out=np.zeros_like(intrinsic_x),
            where=cylindrical_r > 0.0,
        )
        radial_direction_y = np.divide(
            intrinsic_y,
            cylindrical_r,
            out=np.zeros_like(intrinsic_y),
            where=cylindrical_r > 0.0,
        )
        acceleration_x = sampled_radial * radial_direction_x
        acceleration_east = sampled_radial * radial_direction_y
        acceleration_north = -cos_i * acceleration_x + sin_i * sampled_vertical
        alpha_east[row, columns] = multiplier * np.trapezoid(
            acceleration_east, x=line_of_sight, axis=1
        )
        alpha_north[row, columns] = multiplier * np.trapezoid(
            acceleration_north, x=line_of_sight, axis=1
        )

    deflection = SkyPhotonDeflection2D(
        alpha_east_radian=alpha_east,
        alpha_north_radian=alpha_north,
        alpha_east_arcsec=alpha_east * RAD_TO_ARCSEC,
        alpha_north_arcsec=alpha_north * RAD_TO_ARCSEC,
        distance_ratio=distance_ratio,
        zero_slip_multiplier=-multiplier,
    )
    finite_paths = path_lengths[np.isfinite(path_lengths)]
    diagnostics = {
        "coordinateSystem": "axisymmetric_cylindrical",
        "samplingMode": "axisymmetric_cylindrical_ray_integral",
        "axisOrder": ["r", "z"],
        "originM": origin.tolist(),
        "inclinationDeg": inclination_deg,
        "skyShape": list(sky_shape),
        "northPhysicalBoundsM": list(north_bounds),
        "eastPhysicalBoundsM": list(east_bounds),
        "lineOfSightSamples": line_samples,
        "supportedPixels": int(valid.sum()),
        "minimumPathLengthM": float(np.min(finite_paths)),
        "maximumPathLengthM": float(np.max(finite_paths)),
        "maximumLineOfSightStepM": float(np.max(finite_paths) / (line_samples - 1)),
        "finiteDomainOutsidePolicy": "zero_outside_solved_cylinder",
    }
    return deflection, diagnostics


def evaluate_photon_lensing_map_target(
    model: Mapping[str, Any],
    observables: Mapping[str, Array],
    geometry: Mapping[str, Any],
    target: Mapping[str, Any],
    arrays: Mapping[str, Array] | None,
    *,
    archive_prefix: str,
) -> tuple[dict[str, Any], dict[str, Array]]:
    """Project and optionally score one explicitly typed photon observable."""

    if target.get("schemaVersion") != "sigma-observation-target/1":
        raise ValueError("observation target must use sigma-observation-target/1")
    if target.get("kind") != "photon_lensing_map":
        raise ValueError(f"unsupported observation target kind: {target.get('kind')}")
    target_id = str(target.get("id", ""))
    if not target_id:
        raise ValueError("observation target id is required")
    provenance = target.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("observation target requires provenance")
    target_license = target.get("license")
    if (
        not isinstance(target_license, Mapping)
        or not isinstance(target_license.get("id"), str)
        or not target_license["id"]
        or not isinstance(target_license.get("redistributionAllowed"), bool)
    ):
        raise ValueError("observation target requires an explicit license")
    observable_id = str(target.get("observable", ""))
    definitions = {
        str(value.get("id")): value for value in model.get("observables", [])
    }
    definition = definitions.get(observable_id)
    if definition is None:
        raise ValueError(f"observation target requires unknown observable {observable_id}")
    if definition.get("target") not in {"photons", "both"}:
        raise ValueError("photon_lensing_map requires a photons or both observable")
    if definition.get("rank") != "vector" or definition.get("unit") != "m/s^2":
        raise ValueError("photon_lensing_map requires a vector observable in m/s^2")
    coordinate_system = str(geometry.get("coordinateSystem", ""))
    dimensions = int(geometry.get("dimensions", 0))
    distance_ratio = _finite_positive(target.get("distanceRatio"), "distanceRatio")
    lens_distance = _finite_positive(
        target.get("lensAngularDiameterDistanceM"), "lensAngularDiameterDistanceM"
    )
    axes = (
        target.get("northAxis"),
        target.get("eastAxis"),
        target.get("lineOfSightAxis"),
    )
    if coordinate_system == "cartesian_3d" and dimensions == 3:
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in axes)
            or set(axes) != {0, 1, 2}
        ):
            raise ValueError(
                "northAxis, eastAxis, and lineOfSightAxis must be a permutation of [0,1,2]"
            )
        components = tuple(
            np.asarray(observables[f"{observable_id}__axis{axis}"], dtype=float)
            for axis in range(3)
            if f"{observable_id}__axis{axis}" in observables
        )
        if len(components) != 3 or any(component.ndim != 3 for component in components):
            raise ValueError(f"observable {observable_id} must provide three 3D components")
        if any(component.shape != components[0].shape for component in components):
            raise ValueError(
                f"observable {observable_id} components do not share the grid shape"
            )
        raw_spacing = geometry.get("spacing")
        spacing = (
            np.full(3, float(raw_spacing))
            if isinstance(raw_spacing, (int, float))
            else np.asarray(raw_spacing, dtype=float)
        )
        if (
            spacing.shape != (3,)
            or np.any(~np.isfinite(spacing))
            or np.any(spacing <= 0)
        ):
            raise ValueError("geometry spacing must contain three positive finite values")
        north_axis, east_axis, line_of_sight_axis = axes
        permutation = (north_axis, east_axis, line_of_sight_axis)
        ordered = tuple(
            np.transpose(components[component_axis], axes=permutation)
            for component_axis in (north_axis, east_axis, line_of_sight_axis)
        )
        deflection = photon_deflection_sky(
            ordered,
            float(spacing[line_of_sight_axis]),
            distance_ratio=distance_ratio,
            light_speed=C_M_S,
        )
        north_physical_spacing = float(spacing[north_axis])
        east_physical_spacing = float(spacing[east_axis])
        line_of_sight_spacing: float | None = float(spacing[line_of_sight_axis])
        axis_convention = {
            "arrayRows": "north",
            "arrayColumns": "east",
            "northAxis": north_axis,
            "eastAxis": east_axis,
            "lineOfSightAxis": line_of_sight_axis,
        }
        projection_diagnostics = {
            "coordinateSystem": coordinate_system,
            "samplingMode": "cartesian_grid_trapezoid",
        }
    elif coordinate_system == "axisymmetric_cylindrical" and dimensions == 2:
        if any(value is not None for value in axes):
            raise ValueError(
                "axisymmetric photon lensing does not accept Cartesian sky-axis indices"
            )
        components = tuple(
            np.asarray(observables[f"{observable_id}__axis{axis}"], dtype=float)
            for axis in range(2)
            if f"{observable_id}__axis{axis}" in observables
        )
        if len(components) != 2:
            raise ValueError(
                f"observable {observable_id} must provide radial and vertical components"
            )
        deflection, projection_diagnostics = _axisymmetric_photon_deflection(
            components,
            geometry,
            target,
            distance_ratio=distance_ratio,
        )
        north_bounds = projection_diagnostics["northPhysicalBoundsM"]
        east_bounds = projection_diagnostics["eastPhysicalBoundsM"]
        north_physical_spacing = (north_bounds[1] - north_bounds[0]) / (
            deflection.alpha_east_radian.shape[0] - 1
        )
        east_physical_spacing = (east_bounds[1] - east_bounds[0]) / (
            deflection.alpha_east_radian.shape[1] - 1
        )
        line_of_sight_spacing = None
        axis_convention = {
            "arrayRows": "north",
            "arrayColumns": "east",
            "intrinsicAxes": ["r", "z"],
            "eastBasisIntrinsicXYZ": [0.0, 1.0, 0.0],
            "northBasisIntrinsicXYZ": [
                -math.cos(math.radians(projection_diagnostics["inclinationDeg"])),
                0.0,
                math.sin(math.radians(projection_diagnostics["inclinationDeg"])),
            ],
        }
    else:
        raise ValueError(
            "photon_lensing_map requires Cartesian 3D or axisymmetric cylindrical geometry"
        )
    north_spacing_radian = north_physical_spacing / lens_distance
    east_spacing_radian = east_physical_spacing / lens_distance
    derived = _lensing_invariants(
        deflection.alpha_east_radian,
        deflection.alpha_north_radian,
        east_spacing_radian=east_spacing_radian,
        north_spacing_radian=north_spacing_radian,
    )
    maps = {
        f"{archive_prefix}__alpha_east_radian": deflection.alpha_east_radian,
        f"{archive_prefix}__alpha_north_radian": deflection.alpha_north_radian,
        f"{archive_prefix}__alpha_east_arcsec": deflection.alpha_east_arcsec,
        f"{archive_prefix}__alpha_north_arcsec": deflection.alpha_north_arcsec,
        **{
            f"{archive_prefix}__{name}": value
            for name, value in sorted(derived.items())
        },
    }
    shape = deflection.alpha_east_radian.shape
    support = _optional_mask(target, arrays, shape)
    fitted = target.get("fittedNuisanceParameters", 0)
    if isinstance(fitted, bool) or not isinstance(fitted, int) or fitted < 0:
        raise ValueError("fittedNuisanceParameters must be a non-negative integer")
    deflection_score = _score_component_pair(
        channel="deflection_arcsec",
        unit="arcsec",
        predicted_first=deflection.alpha_east_arcsec,
        predicted_second=deflection.alpha_north_arcsec,
        observed_first_key="observedAlphaEastArcsecArrayKey",
        observed_second_key="observedAlphaNorthArcsecArrayKey",
        uncertainty_key="deflectionUncertaintyArcsecArrayKey",
        target=target,
        arrays=arrays,
        support=support,
        fitted_nuisance_parameters=fitted,
    )
    shear_score = _score_component_pair(
        channel="reduced_shear_dimensionless",
        unit="1",
        predicted_first=derived["reduced_shear_1"],
        predicted_second=derived["reduced_shear_2"],
        observed_first_key="observedReducedShear1ArrayKey",
        observed_second_key="observedReducedShear2ArrayKey",
        uncertainty_key="reducedShearUncertaintyArrayKey",
        target=target,
        arrays=arrays,
        support=support,
        fitted_nuisance_parameters=fitted,
    )
    channel_scores = {
        value["channel"]: value for value in (deflection_score, shear_score)
    }
    scored = [value for value in channel_scores.values() if value["state"] == "scored"]
    state = "scored" if scored else "predicted_not_scored"
    map_units = {
        "alpha_east_radian": "rad",
        "alpha_north_radian": "rad",
        "alpha_east_arcsec": "arcsec",
        "alpha_north_arcsec": "arcsec",
        **{name: "1" for name in derived},
    }
    return (
        {
            "id": target_id,
            "kind": "photon_lensing_map",
            "observable": observable_id,
            "observableTarget": definition.get("target"),
            "state": state,
            "mapShape": list(shape),
            "coordinateSystem": coordinate_system,
            "samplingMode": projection_diagnostics["samplingMode"],
            "axisConvention": axis_convention,
            "distanceRatio": distance_ratio,
            "lensAngularDiameterDistanceM": lens_distance,
            "lineOfSightSpacingM": line_of_sight_spacing,
            "northAngularSpacingRadian": north_spacing_radian,
            "eastAngularSpacingRadian": east_spacing_radian,
            "projection": {
                "equation": "alpha_perp_rad=-(2*distanceRatio/c^2)*integral(a_photon_perp dl)",
                "lightSpeedMPerS": C_M_S,
                "relativisticMultiplier": 2.0,
                "radiansToArcseconds": RAD_TO_ARCSEC,
                "diagnostics": projection_diagnostics,
            },
            "mapArchivePrefix": archive_prefix,
            "mapKeys": {
                name: f"{archive_prefix}__{name}" for name in sorted(map_units)
            },
            "mapUnits": map_units,
            "arrayKeys": {
                key: target.get(key)
                for key in (
                    "observedAlphaEastArcsecArrayKey",
                    "observedAlphaNorthArcsecArrayKey",
                    "deflectionUncertaintyArcsecArrayKey",
                    "observedReducedShear1ArrayKey",
                    "observedReducedShear2ArrayKey",
                    "reducedShearUncertaintyArrayKey",
                    "scoreMaskArrayKey",
                )
                if target.get(key) is not None
            },
            "score": {
                "state": state,
                "totalPoints": sum(value["totalPoints"] for value in channel_scores.values()),
                "validPoints": sum(value["validPoints"] for value in channel_scores.values()),
                "fittedNuisanceParameters": fitted,
                "channels": channel_scores,
            },
            "claimBoundary": [
                "This adapter evaluates the submitted photon field and does not validate or alter its field equation.",
                "Distances are explicit inputs; no redshift or cosmology was inferred.",
                "Map scoring is not a raw multiple-image, source-position, or time-delay likelihood.",
                "Axisymmetric projection integrates only through the finite solved cylinder and treats the field outside it as zero; resolution and domain sensitivity must be reported.",
            ],
        },
        maps,
    )
