"""Theory-neutral adapters from solved fields to observation-space predictions.

The adapters convert a declared massive-tracer acceleration field on a
Cartesian 2D or 3D grid into either an azimuthally averaged circular-speed
curve or a projected, optionally beam-convolved line-of-sight velocity map.
They never change, fit, or re-solve the submitted gravity model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

Array = np.ndarray


def _finite_vector(value: Any, length: int, label: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or np.any(~np.isfinite(result)):
        raise ValueError(f"{label} must contain {length} finite values")
    return result


def _finite_series(value: Any, label: str, *, positive: bool = False) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0 or np.any(~np.isfinite(result)):
        raise ValueError(f"{label} must be a non-empty finite one-dimensional array")
    if positive and np.any(result <= 0):
        raise ValueError(f"{label} must be positive")
    return result


def _observable_definition(
    model: Mapping[str, Any], observable_id: str, target_kind: str
) -> Mapping[str, Any]:
    definitions = {
        str(value.get("id")): value for value in model.get("observables", [])
    }
    if observable_id not in definitions:
        raise ValueError(f"observation target requires unknown observable {observable_id}")
    definition = definitions[observable_id]
    if definition.get("target") != "massive_tracers":
        raise ValueError(f"{target_kind} requires a massive_tracers observable")
    if definition.get("rank") != "vector" or definition.get("unit") != "m/s^2":
        raise ValueError(f"{target_kind} requires a vector observable in m/s^2")
    return definition


def _acceleration_components(
    observables: Mapping[str, Array], observable_id: str, dimensions: int
) -> tuple[Array, ...]:
    components = tuple(
        np.asarray(observables[f"{observable_id}__axis{axis}"], dtype=float)
        for axis in range(dimensions)
        if f"{observable_id}__axis{axis}" in observables
    )
    if len(components) != dimensions:
        raise ValueError(
            f"observable {observable_id} does not provide {dimensions} vector components"
        )
    shape = components[0].shape
    if len(shape) != dimensions or any(value.shape != shape for value in components):
        raise ValueError(f"observable {observable_id} components do not share the grid shape")
    return components


def _coordinate_origin(
    target: Mapping[str, Any], geometry: Mapping[str, Any], shape: Sequence[int], spacing: Array
) -> tuple[Array, str]:
    if target.get("gridOriginM") is not None:
        return _finite_vector(target["gridOriginM"], len(shape), "gridOriginM"), "target"
    if geometry.get("origin") is not None:
        return _finite_vector(geometry["origin"], len(shape), "bundle geometry origin"), "bundle"
    centered = -0.5 * (np.asarray(shape, dtype=float) - 1.0) * spacing
    return centered, "explicit_centered_grid_fallback"


def _score_curve(
    target: Mapping[str, Any], predicted: Array, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    observed_value = target.get("observedSpeedsMPerS")
    if observed_value is None:
        return {
            "state": "predicted_not_scored",
            "totalPoints": int(predicted.size),
            "validPoints": int(np.isfinite(predicted).sum()),
            "fittedNuisanceParameters": 0,
        }
    observed = _finite_series(observed_value, "observedSpeedsMPerS", positive=True)
    if observed.shape != predicted.shape:
        raise ValueError("observedSpeedsMPerS must match radiiM")
    uncertainty_value = target.get("uncertaintiesMPerS")
    covariance_value = target.get("covarianceM2PerS2")
    if (uncertainty_value is None) == (covariance_value is None):
        raise ValueError("scored targets require exactly one uncertainty or covariance input")
    fitted = target.get("fittedNuisanceParameters", 0)
    if isinstance(fitted, bool) or not isinstance(fitted, int) or fitted < 0:
        raise ValueError("fittedNuisanceParameters must be a non-negative integer")
    if fitted >= observed.size:
        raise ValueError(
            "fittedNuisanceParameters must be smaller than the scored point count"
        )
    valid = np.isfinite(predicted) & np.isfinite(observed)
    residual = predicted - observed
    for index, row in enumerate(rows):
        row["observed_speed_m_s"] = float(observed[index])
        row["residual_m_s"] = float(residual[index]) if valid[index] else None
    if valid.sum() <= fitted:
        return {
            "state": "insufficient_valid_points",
            "totalPoints": int(predicted.size),
            "validPoints": int(valid.sum()),
            "fittedNuisanceParameters": int(fitted),
        }
    selected = residual[valid]
    sum_squared = float(np.sum(np.square(selected)))
    if uncertainty_value is not None:
        uncertainty = _finite_series(
            uncertainty_value, "uncertaintiesMPerS", positive=True
        )
        if uncertainty.shape != predicted.shape:
            raise ValueError("uncertaintiesMPerS must match radiiM")
        selected_uncertainty = uncertainty[valid]
        covariance = np.diag(np.square(selected_uncertainty))
        for index, row in enumerate(rows):
            row["uncertainty_m_s"] = float(uncertainty[index])
    else:
        covariance = np.asarray(covariance_value, dtype=float)
        expected_shape = (predicted.size, predicted.size)
        if covariance.shape != expected_shape or np.any(~np.isfinite(covariance)):
            raise ValueError("covarianceM2PerS2 must be a finite square matrix matching radiiM")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
            raise ValueError("covarianceM2PerS2 must be symmetric")
        covariance = covariance[np.ix_(valid, valid)]
        diagonal = np.sqrt(np.diag(covariance))
        for row_index, source_index in enumerate(np.flatnonzero(valid)):
            rows[source_index]["uncertainty_m_s"] = float(diagonal[row_index])
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("observation covariance must be positive definite") from error
    whitened = np.linalg.solve(factor, selected)
    chi_square = float(whitened @ whitened)
    degrees_freedom = int(valid.sum()) - fitted
    log_determinant = 2.0 * float(np.log(np.diag(factor)).sum())
    log_likelihood = -0.5 * (
        chi_square + log_determinant + int(valid.sum()) * math.log(2.0 * math.pi)
    )
    inverse_variance = 1.0 / np.diag(covariance)
    weighted_sum = float(np.sum(inverse_variance * np.square(selected)))
    weight_sum = float(np.sum(inverse_variance))
    return {
        "state": "scored",
        "totalPoints": int(predicted.size),
        "validPoints": int(valid.sum()),
        "fittedNuisanceParameters": int(fitted),
        "sumSquaredResidualM2PerS2": sum_squared,
        "rmseMPerS": math.sqrt(sum_squared / int(valid.sum())),
        "inverseVarianceWeightedRmseMPerS": math.sqrt(weighted_sum / weight_sum),
        "chiSquare": chi_square,
        "degreesFreedom": degrees_freedom,
        "reducedChiSquare": chi_square / degrees_freedom,
        "gaussianLogLikelihood": log_likelihood,
        "inverseVarianceWeightedSquaredResidual": weighted_sum,
        "inverseVarianceWeightSum": weight_sum,
    }


def evaluate_circular_speed_target(
    model: Mapping[str, Any],
    observables: Mapping[str, Array],
    geometry: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Predict and optionally score one circular-speed curve."""

    if target.get("schemaVersion") != "sigma-observation-target/1":
        raise ValueError("observation target must use sigma-observation-target/1")
    if target.get("kind") != "circular_speed_curve":
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
    _observable_definition(model, observable_id, "circular_speed_curve")
    dimensions = int(geometry.get("dimensions", 0))
    if geometry.get("coordinateSystem") not in {"cartesian_2d", "cartesian_3d"}:
        raise ValueError("circular_speed_curve supports Cartesian 2D or 3D grids")
    if dimensions not in {2, 3}:
        raise ValueError("circular_speed_curve requires two or three dimensions")
    components = _acceleration_components(observables, observable_id, dimensions)
    shape = components[0].shape
    raw_spacing = geometry.get("spacing")
    spacing = (
        np.full(dimensions, float(raw_spacing))
        if isinstance(raw_spacing, (int, float))
        else _finite_vector(raw_spacing, dimensions, "geometry spacing")
    )
    if np.any(spacing <= 0):
        raise ValueError("geometry spacing must be positive")
    origin, origin_source = _coordinate_origin(target, geometry, shape, spacing)
    center = _finite_vector(target.get("centerM"), dimensions, "centerM")
    plane_axes = target.get("planeAxes", [0, 1])
    if (
        not isinstance(plane_axes, list)
        or len(plane_axes) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in plane_axes)
        or len(set(plane_axes)) != 2
        or any(value < 0 or value >= dimensions for value in plane_axes)
    ):
        raise ValueError("planeAxes must identify two distinct grid axes")
    radii = _finite_series(target.get("radiiM"), "radiiM", positive=True)
    if np.any(np.diff(radii) <= 0):
        raise ValueError("radiiM must be strictly increasing")
    sample_count = target.get("azimuthalSamples", 128)
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 16 <= sample_count <= 4096
    ):
        raise ValueError("azimuthalSamples must be an integer from 16 through 4096")
    minimum_coverage = float(target.get("minimumAzimuthalCoverage", 0.8))
    if not math.isfinite(minimum_coverage) or not 0 < minimum_coverage <= 1:
        raise ValueError("minimumAzimuthalCoverage must lie in (0,1]")
    angles = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
    cosines = np.cos(angles)
    sines = np.sin(angles)
    predicted = np.full(radii.shape, np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    for index, radius in enumerate(radii):
        positions = np.broadcast_to(center[:, None], (dimensions, sample_count)).copy()
        positions[plane_axes[0]] += radius * cosines
        positions[plane_axes[1]] += radius * sines
        indices = (positions - origin[:, None]) / spacing[:, None]
        sampled = np.vstack(
            [
                map_coordinates(
                    component,
                    indices,
                    order=1,
                    mode="constant",
                    cval=np.nan,
                    prefilter=False,
                )
                for component in components
            ]
        )
        inward = -(
            sampled[plane_axes[0]] * cosines + sampled[plane_axes[1]] * sines
        )
        valid = np.isfinite(inward)
        coverage = float(valid.mean())
        mean_inward = float(np.mean(inward[valid])) if valid.any() else math.nan
        if coverage >= minimum_coverage and mean_inward > 0:
            predicted[index] = math.sqrt(float(radius) * mean_inward)
        rows.append(
            {
                "target_id": target_id,
                "point_index": index,
                "radius_m": float(radius),
                "predicted_speed_m_s": float(predicted[index])
                if math.isfinite(predicted[index])
                else None,
                "observed_speed_m_s": None,
                "uncertainty_m_s": None,
                "residual_m_s": None,
                "azimuthal_coverage": coverage,
                "mean_inward_acceleration_m_s2": mean_inward
                if math.isfinite(mean_inward)
                else None,
            }
        )
    score = _score_curve(target, predicted, rows)
    return (
        {
            "id": target_id,
            "kind": "circular_speed_curve",
            "observable": observable_id,
            "observableTarget": "massive_tracers",
            "state": score["state"],
            "originM": origin.tolist(),
            "originSource": origin_source,
            "centerM": center.tolist(),
            "planeAxes": list(plane_axes),
            "azimuthalSamples": sample_count,
            "minimumAzimuthalCoverage": minimum_coverage,
            "score": score,
        },
        rows,
    )


def _target_array(
    arrays: Mapping[str, Array] | None,
    key: Any,
    label: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> Array:
    if arrays is None:
        raise ValueError(f"{label} requires observation arrays")
    if not isinstance(key, str) or not key:
        raise ValueError(f"{label} requires a non-empty array key")
    if key not in arrays:
        raise ValueError(f"{label} references missing array {key}")
    value = np.asarray(arrays[key], dtype=float)
    if value.ndim != 2 or min(value.shape) < 5:
        raise ValueError(f"{label} array must be a two-dimensional map")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{label} array must have shape {shape}")
    return value


def _optional_target_array(
    arrays: Mapping[str, Array] | None,
    key: Any,
    label: str,
    shape: tuple[int, ...],
) -> Array | None:
    if key is None:
        return None
    return _target_array(arrays, key, label, shape=shape)


def _score_velocity_field(
    target: Mapping[str, Any],
    arrays: Mapping[str, Array] | None,
    predicted: Array,
    support: Array,
) -> tuple[dict[str, Any], dict[str, Array | None]]:
    observed_key = target.get("observedVelocityArrayKey")
    if observed_key is None:
        valid = support & np.isfinite(predicted)
        return (
            {
                "state": "predicted_not_scored",
                "totalPoints": int(predicted.size),
                "validPoints": int(valid.sum()),
                "fittedNuisanceParameters": 0,
            },
            {
                "observed": None,
                "uncertainty": None,
                "residual": None,
                "declaredWeight": None,
                "valid": valid,
            },
        )
    observed_raw = _target_array(
        arrays,
        observed_key,
        "observedVelocityArrayKey",
        shape=predicted.shape,
    )
    uncertainty = _target_array(
        arrays,
        target.get("uncertaintyArrayKey"),
        "uncertaintyArrayKey",
        shape=predicted.shape,
    )
    zero_point = float(target.get("observedVelocityZeroPointMPerS", 0.0))
    if not math.isfinite(zero_point):
        raise ValueError("observedVelocityZeroPointMPerS must be finite")
    observed = observed_raw - zero_point
    valid = (
        support
        & np.isfinite(predicted)
        & np.isfinite(observed)
        & np.isfinite(uncertainty)
        & (uncertainty > 0.0)
    )
    fitted = target.get("fittedNuisanceParameters", 0)
    if isinstance(fitted, bool) or not isinstance(fitted, int) or fitted < 0:
        raise ValueError("fittedNuisanceParameters must be a non-negative integer")
    if int(valid.sum()) <= fitted:
        return (
            {
                "state": "insufficient_valid_points",
                "totalPoints": int(predicted.size),
                "validPoints": int(valid.sum()),
                "fittedNuisanceParameters": int(fitted),
            },
            {
                "observed": observed,
                "uncertainty": uncertainty,
                "residual": predicted - observed,
                "declaredWeight": None,
                "valid": valid,
            },
        )
    residual = predicted - observed
    inverse_variance = np.divide(
        1.0,
        np.square(uncertainty),
        out=np.zeros_like(uncertainty),
        where=valid,
    )
    weighting = str(target.get("weighting", "inverse_variance"))
    if weighting == "inverse_variance":
        declared_weight = inverse_variance
    elif weighting == "intensity_inverse_variance":
        intensity = _target_array(
            arrays,
            target.get("intensityWeightArrayKey"),
            "intensityWeightArrayKey",
            shape=predicted.shape,
        )
        if np.any(intensity[np.isfinite(intensity)] < 0.0):
            raise ValueError("intensity weights must be non-negative")
        valid &= np.isfinite(intensity) & (intensity > 0.0)
        inverse_variance = np.divide(
            1.0,
            np.square(uncertainty),
            out=np.zeros_like(uncertainty),
            where=valid,
        )
        declared_weight = np.where(valid, intensity * inverse_variance, 0.0)
    else:
        raise ValueError(
            "line_of_sight_velocity_field weighting must be inverse_variance or "
            "intensity_inverse_variance"
        )
    valid_count = int(valid.sum())
    if valid_count <= fitted:
        raise ValueError("fittedNuisanceParameters must be smaller than valid pixels")
    selected_residual = residual[valid]
    selected_uncertainty = uncertainty[valid]
    sum_squared = float(np.sum(np.square(selected_residual)))
    inverse_variance_sum = float(np.sum(inverse_variance[valid]))
    inverse_variance_squared = float(
        np.sum(inverse_variance[valid] * np.square(selected_residual))
    )
    declared_weight_sum = float(np.sum(declared_weight[valid]))
    declared_weight_squared = float(
        np.sum(declared_weight[valid] * np.square(selected_residual))
    )
    chi_square = float(np.sum(np.square(selected_residual / selected_uncertainty)))
    degrees_freedom = valid_count - fitted
    log_determinant = float(np.sum(np.log(np.square(selected_uncertainty))))
    log_likelihood = -0.5 * (
        chi_square + log_determinant + valid_count * math.log(2.0 * math.pi)
    )
    return (
        {
            "state": "scored",
            "totalPoints": int(predicted.size),
            "validPoints": valid_count,
            "fittedNuisanceParameters": int(fitted),
            "weighting": weighting,
            "sumSquaredResidualM2PerS2": sum_squared,
            "rmseMPerS": math.sqrt(sum_squared / valid_count),
            "inverseVarianceWeightedRmseMPerS": math.sqrt(
                inverse_variance_squared / inverse_variance_sum
            ),
            "declaredWeightedRmseMPerS": math.sqrt(
                declared_weight_squared / declared_weight_sum
            ),
            "chiSquare": chi_square,
            "degreesFreedom": degrees_freedom,
            "reducedChiSquare": chi_square / degrees_freedom,
            "gaussianLogLikelihood": log_likelihood,
            "inverseVarianceWeightedSquaredResidual": inverse_variance_squared,
            "inverseVarianceWeightSum": inverse_variance_sum,
            "declaredWeightedSquaredResidual": declared_weight_squared,
            "declaredWeightSum": declared_weight_sum,
        },
        {
            "observed": observed,
            "uncertainty": uncertainty,
            "residual": residual,
            "declaredWeight": declared_weight,
            "valid": valid,
        },
    )


def evaluate_line_of_sight_velocity_field_target(
    model: Mapping[str, Any],
    observables: Mapping[str, Array],
    geometry: Mapping[str, Any],
    target: Mapping[str, Any],
    arrays: Mapping[str, Array] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project a massive-tracer acceleration field into a resolved LOS map."""

    if target.get("schemaVersion") != "sigma-observation-target/1":
        raise ValueError("observation target must use sigma-observation-target/1")
    if target.get("kind") != "line_of_sight_velocity_field":
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
    _observable_definition(model, observable_id, "line_of_sight_velocity_field")
    dimensions = int(geometry.get("dimensions", 0))
    if geometry.get("coordinateSystem") not in {"cartesian_2d", "cartesian_3d"}:
        raise ValueError(
            "line_of_sight_velocity_field supports Cartesian 2D or 3D grids"
        )
    if dimensions not in {2, 3}:
        raise ValueError("line_of_sight_velocity_field requires two or three dimensions")
    components = _acceleration_components(observables, observable_id, dimensions)
    shape = components[0].shape
    raw_spacing = geometry.get("spacing")
    spacing = (
        np.full(dimensions, float(raw_spacing))
        if isinstance(raw_spacing, (int, float))
        else _finite_vector(raw_spacing, dimensions, "geometry spacing")
    )
    if np.any(spacing <= 0):
        raise ValueError("geometry spacing must be positive")
    origin, origin_source = _coordinate_origin(target, geometry, shape, spacing)
    center = _finite_vector(target.get("centerM"), dimensions, "centerM")
    plane_axes = target.get("planeAxes", [0, 1])
    if (
        not isinstance(plane_axes, list)
        or len(plane_axes) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in plane_axes)
        or len(set(plane_axes)) != 2
        or any(value < 0 or value >= dimensions for value in plane_axes)
    ):
        raise ValueError("planeAxes must identify two distinct grid axes")
    major = _target_array(
        arrays, target.get("majorCoordinateArrayKey"), "majorCoordinateArrayKey"
    )
    minor = _target_array(
        arrays,
        target.get("minorCoordinateArrayKey"),
        "minorCoordinateArrayKey",
        shape=major.shape,
    )
    inclination_deg = float(target.get("inclinationDeg", math.nan))
    if not math.isfinite(inclination_deg) or not 0.0 < inclination_deg < 90.0:
        raise ValueError("inclinationDeg must lie strictly between 0 and 90 degrees")
    handedness = target.get("handedness")
    if handedness not in {-1, 1}:
        raise ValueError("handedness must be -1 or 1")
    positions = np.broadcast_to(
        center[:, None], (dimensions, major.size)
    ).copy()
    positions[plane_axes[0]] += major.ravel()
    positions[plane_axes[1]] += minor.ravel()
    indices = (positions - origin[:, None]) / spacing[:, None]
    sampled = np.vstack(
        [
            map_coordinates(
                component,
                indices,
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            for component in components
        ]
    )
    radius = np.hypot(major, minor)
    radial_x = np.divide(major, radius, out=np.zeros_like(major), where=radius > 0.0)
    radial_y = np.divide(minor, radius, out=np.zeros_like(minor), where=radius > 0.0)
    inward = -(
        sampled[plane_axes[0]].reshape(major.shape) * radial_x
        + sampled[plane_axes[1]].reshape(major.shape) * radial_y
    )
    nonpositive_policy = str(target.get("nonPositiveInwardPolicy", "exclude"))
    if nonpositive_policy not in {"exclude", "zero_speed"}:
        raise ValueError(
            "nonPositiveInwardPolicy must be exclude or zero_speed"
        )
    circular_speed = np.sqrt(np.maximum(radius * inward, 0.0))
    predicted = (
        float(handedness)
        * math.sin(math.radians(inclination_deg))
        * circular_speed
        * radial_x
    )
    intrinsic_support = np.isfinite(major) & np.isfinite(minor) & np.isfinite(inward)
    if nonpositive_policy == "exclude":
        intrinsic_support &= (radius > 0.0) & (inward > 0.0)
    emission_mask = _optional_target_array(
        arrays,
        target.get("emissionMaskArrayKey"),
        "emissionMaskArrayKey",
        major.shape,
    )
    if emission_mask is not None:
        intrinsic_support &= np.isfinite(emission_mask) & (emission_mask > 0.0)
    legacy_mask_key = target.get("maskArrayKey")
    score_mask_key = target.get("scoreMaskArrayKey")
    if legacy_mask_key is not None and score_mask_key is not None:
        raise ValueError("use either maskArrayKey or scoreMaskArrayKey, not both")
    score_mask = _optional_target_array(
        arrays,
        score_mask_key if score_mask_key is not None else legacy_mask_key,
        "scoreMaskArrayKey" if score_mask_key is not None else "maskArrayKey",
        major.shape,
    )
    beam_kernel_key = target.get("beamKernelArrayKey")
    beam_diagnostics: dict[str, Any] | None = None
    if beam_kernel_key is not None:
        intensity = _target_array(
            arrays,
            target.get("intensityWeightArrayKey"),
            "intensityWeightArrayKey",
            shape=major.shape,
        )
        kernel = _target_array(arrays, beam_kernel_key, "beamKernelArrayKey")
        if any(value % 2 == 0 for value in kernel.shape):
            raise ValueError("beam kernel dimensions must be odd")
        if np.any(~np.isfinite(kernel)) or np.any(kernel < 0.0) or float(kernel.sum()) <= 0:
            raise ValueError("beam kernel must be finite, non-negative, and non-zero")
        kernel = kernel / float(kernel.sum())
        beam_support = intrinsic_support & np.isfinite(intensity) & (intensity > 0.0)
        numerator = fftconvolve(
            np.where(beam_support, predicted * intensity, 0.0), kernel, mode="same"
        )
        denominator = fftconvolve(
            np.where(beam_support, intensity, 0.0), kernel, mode="same"
        )
        prediction_support = denominator > np.finfo(float).tiny
        predicted = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=prediction_support,
        )
        prediction_support &= np.isfinite(predicted)
        beam_diagnostics = {
            "kernelArrayKey": beam_kernel_key,
            "kernelShape": list(kernel.shape),
            "normalizedKernelSum": float(kernel.sum()),
            "intensityArrayKey": target.get("intensityWeightArrayKey"),
        }
    else:
        prediction_support = intrinsic_support & np.isfinite(predicted)
    support = prediction_support.copy()
    if score_mask is not None:
        support &= np.isfinite(score_mask) & (score_mask > 0.0)
    minimum_valid = target.get("minimumValidPixels", 25)
    if (
        isinstance(minimum_valid, bool)
        or not isinstance(minimum_valid, int)
        or minimum_valid < 1
    ):
        raise ValueError("minimumValidPixels must be a positive integer")
    if int((support & np.isfinite(predicted)).sum()) < minimum_valid:
        raise ValueError("too few valid predicted velocity-field pixels")
    score, score_arrays = _score_velocity_field(
        target, arrays, predicted, support
    )
    valid_rows = np.asarray(score_arrays["valid"], dtype=bool)
    observed = score_arrays["observed"]
    uncertainty = score_arrays["uncertainty"]
    residual = score_arrays["residual"]
    declared_weight = score_arrays["declaredWeight"]
    rows: list[dict[str, Any]] = []
    for row_index, column_index in np.argwhere(valid_rows):
        rows.append(
            {
                "target_id": target_id,
                "point_index": int(np.ravel_multi_index((row_index, column_index), major.shape)),
                "row_index": int(row_index),
                "column_index": int(column_index),
                "disk_major_coordinate_m": float(major[row_index, column_index]),
                "disk_minor_coordinate_m": float(minor[row_index, column_index]),
                "circular_radius_m": float(radius[row_index, column_index]),
                "predicted_circular_speed_m_s": float(
                    circular_speed[row_index, column_index]
                ),
                "predicted_velocity_m_s": float(predicted[row_index, column_index]),
                "observed_velocity_m_s": float(observed[row_index, column_index])
                if observed is not None
                else None,
                "uncertainty_m_s": float(uncertainty[row_index, column_index])
                if uncertainty is not None
                else None,
                "residual_m_s": float(residual[row_index, column_index])
                if residual is not None
                else None,
                "declared_weight": float(declared_weight[row_index, column_index])
                if declared_weight is not None
                else None,
                "inward_acceleration_m_s2": float(inward[row_index, column_index]),
            }
        )
    return (
        {
            "id": target_id,
            "kind": "line_of_sight_velocity_field",
            "observable": observable_id,
            "observableTarget": "massive_tracers",
            "state": score["state"],
            "originM": origin.tolist(),
            "originSource": origin_source,
            "centerM": center.tolist(),
            "planeAxes": list(plane_axes),
            "inclinationDeg": inclination_deg,
            "handedness": int(handedness),
            "nonPositiveInwardPolicy": nonpositive_policy,
            "mapShape": list(major.shape),
            "minimumValidPixels": minimum_valid,
            "beamConvolution": beam_diagnostics,
            "arrayKeys": {
                key: target.get(key)
                for key in (
                    "majorCoordinateArrayKey",
                    "minorCoordinateArrayKey",
                    "observedVelocityArrayKey",
                    "uncertaintyArrayKey",
                    "intensityWeightArrayKey",
                    "maskArrayKey",
                    "scoreMaskArrayKey",
                    "emissionMaskArrayKey",
                    "beamKernelArrayKey",
                )
                if target.get(key) is not None
            },
            "score": score,
        },
        rows,
    )


def evaluate_observation_targets(
    model: Mapping[str, Any],
    observables: Mapping[str, Array],
    geometry: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Array] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate supported targets and return deterministic aggregate diagnostics."""

    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("observationTargets must be an array")
    if len(targets) > 32:
        raise ValueError("observationTargets must contain at most 32 targets")
    evaluations: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for target in targets:
        if not isinstance(target, Mapping):
            raise TypeError("observation target must be an object")
        target_id = str(target.get("id", ""))
        if target_id in seen:
            raise ValueError(f"duplicate observation target id: {target_id}")
        seen.add(target_id)
        if target.get("kind") == "circular_speed_curve":
            evaluation, target_rows = evaluate_circular_speed_target(
                model, observables, geometry, target
            )
        elif target.get("kind") == "line_of_sight_velocity_field":
            evaluation, target_rows = evaluate_line_of_sight_velocity_field_target(
                model, observables, geometry, target, arrays
            )
        else:
            raise ValueError(f"unsupported observation target kind: {target.get('kind')}")
        evaluations.append(evaluation)
        rows.extend(target_rows)
    scored = [value["score"] for value in evaluations if value["state"] == "scored"]
    valid_points = sum(int(value["validPoints"]) for value in scored)
    sum_squared = sum(float(value["sumSquaredResidualM2PerS2"]) for value in scored)
    chi_square = sum(float(value["chiSquare"]) for value in scored)
    degrees_freedom = sum(int(value["degreesFreedom"]) for value in scored)
    weighted_sum = sum(
        float(value["inverseVarianceWeightedSquaredResidual"]) for value in scored
    )
    weight_sum = sum(float(value["inverseVarianceWeightSum"]) for value in scored)
    return (
        {
            "schemaVersion": "sigma-observation-evaluation/1",
            "targetKinds": sorted({value["kind"] for value in evaluations}),
            "targetCount": len(evaluations),
            "scoredTargetCount": len(scored),
            "totalPoints": sum(int(value["score"]["totalPoints"]) for value in evaluations),
            "validScoredPoints": valid_points,
            "sumSquaredResidualM2PerS2": sum_squared if scored else None,
            "rmseMPerS": math.sqrt(sum_squared / valid_points) if valid_points else None,
            "inverseVarianceWeightedSquaredResidual": weighted_sum if scored else None,
            "inverseVarianceWeightSum": weight_sum if scored else None,
            "inverseVarianceWeightedRmseMPerS": math.sqrt(weighted_sum / weight_sum)
            if weight_sum
            else None,
            "chiSquare": chi_square if scored else None,
            "degreesFreedom": degrees_freedom if scored else None,
            "reducedChiSquare": chi_square / degrees_freedom if degrees_freedom else None,
            "targets": evaluations,
            "claimBoundary": [
                "Observation targets are evaluated after the field solve and cannot modify the submitted equations.",
                "Circular-speed and line-of-sight velocity-field adapters are massive-tracer mappings, not photon-lensing predictions.",
                "Velocity-field projection uses explicitly declared geometry, handedness, emission support, post-convolution score masks, uncertainties, intensity weights, and beam kernels.",
            ],
        },
        rows,
    )
