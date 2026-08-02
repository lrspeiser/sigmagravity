"""Raw image-plane strong-lensing adapter for typed 3D photon fields."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .photon_lensing_adapter import evaluate_photon_lensing_map_target
from .sky_lensing import (
    RAD_TO_ARCSEC,
    GridSkyDeflectionField,
    assign_observed_roots,
    critical_curve_points,
    find_lens_roots,
    profiled_source,
)

Array = np.ndarray


def _finite_positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _finite_vector(value: Any, length: int, label: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or np.any(~np.isfinite(result)):
        raise ValueError(f"{label} must contain {length} finite values")
    return result


def _positive_integer(
    value: Any, label: str, *, minimum: int, maximum: int
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(f"{label} must be an integer from {minimum} through {maximum}")
    return value


def _observable_definition(
    model: Mapping[str, Any], observable_id: str
) -> Mapping[str, Any]:
    definitions = {
        str(value.get("id")): value for value in model.get("observables", [])
    }
    definition = definitions.get(observable_id)
    if definition is None:
        raise ValueError(f"observation target requires unknown observable {observable_id}")
    if definition.get("target") not in {"photons", "both"}:
        raise ValueError("multiple_image_systems requires a photons or both observable")
    if definition.get("rank") != "vector" or definition.get("unit") != "m/s^2":
        raise ValueError("multiple_image_systems requires a vector observable in m/s^2")
    return definition


def _grid_shape(observables: Mapping[str, Array], observable_id: str) -> tuple[int, ...]:
    components = [
        np.asarray(observables[f"{observable_id}__axis{axis}"])
        for axis in range(3)
        if f"{observable_id}__axis{axis}" in observables
    ]
    if len(components) != 3 or any(value.ndim != 3 for value in components):
        raise ValueError(f"observable {observable_id} must provide three 3D components")
    if any(value.shape != components[0].shape for value in components):
        raise ValueError(f"observable {observable_id} components do not share the grid shape")
    return components[0].shape


def _spacing(geometry: Mapping[str, Any]) -> Array:
    raw = geometry.get("spacing")
    result = (
        np.full(3, float(raw))
        if isinstance(raw, (int, float))
        else np.asarray(raw, dtype=float)
    )
    if result.shape != (3,) or np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("geometry spacing must contain three positive finite values")
    return result


def _origin(
    geometry: Mapping[str, Any], shape: Sequence[int], spacing: Array
) -> tuple[Array, str]:
    if geometry.get("origin") is not None:
        return _finite_vector(geometry["origin"], 3, "geometry origin"), "bundle"
    return (
        -0.5 * (np.asarray(shape, dtype=float) - 1.0) * spacing,
        "explicit_centered_grid_fallback",
    )


def _families(target: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = target.get("families")
    if not isinstance(raw, list) or not raw or len(raw) > 64:
        raise ValueError("families must contain from 1 through 64 image families")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_images = 0
    for family in raw:
        if not isinstance(family, Mapping):
            raise TypeError("each image family must be an object")
        family_id = str(family.get("id", ""))
        if not family_id or family_id in seen:
            raise ValueError(f"invalid or duplicate image family id: {family_id}")
        seen.add(family_id)
        images = np.asarray(family.get("observedImagesArcsec"), dtype=float)
        if (
            images.ndim != 2
            or images.shape[1] != 2
            or len(images) < 2
            or np.any(~np.isfinite(images))
        ):
            raise ValueError(
                f"family {family_id} observedImagesArcsec must have shape (n>=2,2)"
            )
        uncertainties = np.asarray(
            family.get("positionUncertaintiesArcsec"), dtype=float
        )
        if (
            uncertainties.shape != (len(images),)
            or np.any(~np.isfinite(uncertainties))
            or np.any(uncertainties <= 0.0)
        ):
            raise ValueError(
                f"family {family_id} positionUncertaintiesArcsec must contain one positive value per image"
            )
        total_images += len(images)
        result.append(
            {
                **dict(family),
                "id": family_id,
                "distanceRatio": _finite_positive(
                    family.get("distanceRatio"), f"family {family_id} distanceRatio"
                ),
                "images": images,
                "uncertainties": uncertainties,
            }
        )
    if total_images > 512:
        raise ValueError("multiple_image_systems supports at most 512 observed images")
    return result


def _projection_target(
    target: Mapping[str, Any], family: Mapping[str, Any], family_index: int
) -> dict[str, Any]:
    return {
        "schemaVersion": "sigma-observation-target/1",
        "id": f"{target['id']}__{family['id']}",
        "kind": "photon_lensing_map",
        "observable": target["observable"],
        "northAxis": target["northAxis"],
        "eastAxis": target["eastAxis"],
        "lineOfSightAxis": target["lineOfSightAxis"],
        "distanceRatio": family["distanceRatio"],
        "lensAngularDiameterDistanceM": target["lensAngularDiameterDistanceM"],
        "minimumValidPixels": 1,
        "provenance": {
            "kind": "P0735 internal projection",
            "parentTargetId": target["id"],
            "familyId": family["id"],
            "familyIndex": family_index,
        },
        "license": target["license"],
    }


def _score_family(
    images: Array,
    uncertainties: Array,
    roots: Array,
    assignment_pairs: Array,
    complete: bool,
) -> dict[str, Any]:
    matched = len(assignment_pairs)
    if not complete:
        diagnostic = None
        if matched:
            observed_index = assignment_pairs[:, 0]
            root_index = assignment_pairs[:, 1]
            separations = np.linalg.norm(
                roots[root_index] - images[observed_index], axis=1
            )
            diagnostic = float(np.sqrt(np.mean(np.square(separations))))
        return {
            "state": "incomplete_topology",
            "observedImages": len(images),
            "matchedImages": matched,
            "validPoints": 2 * matched,
            "fittedObservationNuisanceParameters": 2,
            "imagePlaneRmsArcsec": None,
            "matchedSubsetDiagnosticRmsArcsec": diagnostic,
            "chiSquare": None,
            "degreesFreedom": None,
            "gaussianLogLikelihood": None,
        }
    observed_index = assignment_pairs[:, 0]
    root_index = assignment_pairs[:, 1]
    residual = roots[root_index] - images[observed_index]
    sigma = uncertainties[observed_index]
    squared_separation = np.sum(np.square(residual), axis=1)
    sum_squared = float(np.sum(squared_separation))
    chi_square = float(np.sum(squared_separation / np.square(sigma)))
    coordinate_count = 2 * len(images)
    degrees_freedom = coordinate_count - 2
    log_determinant = float(2.0 * np.sum(np.log(np.square(sigma))))
    log_likelihood = -0.5 * (
        chi_square + log_determinant + coordinate_count * math.log(2.0 * math.pi)
    )
    inverse_variance = np.repeat(1.0 / np.square(sigma), 2)
    coordinate_residual = residual.ravel()
    weighted_squared = float(
        np.sum(inverse_variance * np.square(coordinate_residual))
    )
    weight_sum = float(np.sum(inverse_variance))
    return {
        "state": "scored",
        "observedImages": len(images),
        "matchedImages": len(images),
        "validPoints": coordinate_count,
        "fittedObservationNuisanceParameters": 2,
        "sumSquaredResidualArcsec2": sum_squared,
        "imagePlaneRmsArcsec": math.sqrt(sum_squared / len(images)),
        "coordinateRmseArcsec": math.sqrt(sum_squared / coordinate_count),
        "inverseVarianceWeightedSquaredResidual": weighted_squared,
        "inverseVarianceWeightSum": weight_sum,
        "inverseVarianceWeightedCoordinateRmseArcsec": math.sqrt(
            weighted_squared / weight_sum
        ),
        "chiSquare": chi_square,
        "degreesFreedom": degrees_freedom,
        "reducedChiSquare": chi_square / degrees_freedom,
        "gaussianLogLikelihood": log_likelihood,
    }


def evaluate_multiple_image_systems_target(
    model: Mapping[str, Any],
    observables: Mapping[str, Array],
    geometry: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    archive_prefix: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Array],
]:
    """Profile sources, find roots, and score observed multiple-image families."""

    if target.get("schemaVersion") != "sigma-observation-target/1":
        raise ValueError("observation target must use sigma-observation-target/1")
    if target.get("kind") != "multiple_image_systems":
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
    definition = _observable_definition(model, observable_id)
    if geometry.get("coordinateSystem") != "cartesian_3d" or int(
        geometry.get("dimensions", 0)
    ) != 3:
        raise ValueError("multiple_image_systems requires a Cartesian 3D grid")
    axes = (
        target.get("northAxis"),
        target.get("eastAxis"),
        target.get("lineOfSightAxis"),
    )
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in axes)
        or set(axes) != {0, 1, 2}
    ):
        raise ValueError(
            "northAxis, eastAxis, and lineOfSightAxis must be a permutation of [0,1,2]"
        )
    shape = _grid_shape(observables, observable_id)
    spacing = _spacing(geometry)
    origin, origin_source = _origin(geometry, shape, spacing)
    sky_center = _finite_vector(target.get("skyCenterM"), 3, "skyCenterM")
    lens_distance = _finite_positive(
        target.get("lensAngularDiameterDistanceM"), "lensAngularDiameterDistanceM"
    )
    root_bound = _finite_positive(
        target.get("rootSearchBoundArcsec"), "rootSearchBoundArcsec"
    )
    root_grid_points = _positive_integer(
        target.get("rootGridPoints", 161),
        "rootGridPoints",
        minimum=21,
        maximum=401,
    )
    closure_tolerance = _finite_positive(
        target.get("closureToleranceArcsec", 2.0e-3),
        "closureToleranceArcsec",
    )
    deduplication_tolerance = _finite_positive(
        target.get("deduplicationToleranceArcsec", 0.2),
        "deduplicationToleranceArcsec",
    )
    jacobian_step = _finite_positive(
        target.get("jacobianStepArcsec", 0.08), "jacobianStepArcsec"
    )
    maximum_seeds = _positive_integer(
        target.get("maximumResidualMinimumSeeds", 64),
        "maximumResidualMinimumSeeds",
        minimum=1,
        maximum=512,
    )
    raw_supplemental = target.get("supplementalGridPoints", [81, 161, 241])
    if (
        not isinstance(raw_supplemental, list)
        or len(raw_supplemental) > 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 21
            or value > 401
            for value in raw_supplemental
        )
    ):
        raise ValueError(
            "supplementalGridPoints must contain at most four integers from 21 through 401"
        )
    include_critical = target.get("includeCriticalCurves", False)
    if not isinstance(include_critical, bool):
        raise TypeError("includeCriticalCurves must be boolean")
    include_residual_minima = target.get("includeResidualMinima", True)
    if not isinstance(include_residual_minima, bool):
        raise TypeError("includeResidualMinima must be boolean")
    critical_grid_points = _positive_integer(
        target.get("criticalCurveGridPoints", 161),
        "criticalCurveGridPoints",
        minimum=21,
        maximum=401,
    )
    families = _families(target)
    north_axis, east_axis, _line_of_sight_axis = axes
    north_coordinates = (
        origin[north_axis]
        + np.arange(shape[north_axis]) * spacing[north_axis]
        - sky_center[north_axis]
    ) / lens_distance * RAD_TO_ARCSEC
    east_coordinates = (
        origin[east_axis]
        + np.arange(shape[east_axis]) * spacing[east_axis]
        - sky_center[east_axis]
    ) / lens_distance * RAD_TO_ARCSEC
    if not (
        north_coordinates[0] <= -root_bound <= root_bound <= north_coordinates[-1]
        and east_coordinates[0] <= -root_bound <= root_bound <= east_coordinates[-1]
    ):
        raise ValueError("rootSearchBoundArcsec must fit inside the published photon map")

    prediction_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    family_evaluations: list[dict[str, Any]] = []
    root_arrays: dict[str, Array] = {}
    for family_index, family in enumerate(families):
        _projection, projected_maps = evaluate_photon_lensing_map_target(
            model,
            observables,
            geometry,
            _projection_target(target, family, family_index),
            None,
            archive_prefix="projection",
        )
        field = GridSkyDeflectionField(
            north_axis_arcsec=north_coordinates,
            east_axis_arcsec=east_coordinates,
            alpha_east_ratio_one_arcsec=projected_maps[
                "projection__alpha_east_arcsec"
            ],
            alpha_north_ratio_one_arcsec=projected_maps[
                "projection__alpha_north_arcsec"
            ],
            distance_ratio=lambda _unused: 1.0,
        )
        images = family["images"]
        if (
            np.any(images[:, 0] < east_coordinates[0])
            or np.any(images[:, 0] > east_coordinates[-1])
            or np.any(images[:, 1] < north_coordinates[0])
            or np.any(images[:, 1] > north_coordinates[-1])
        ):
            raise ValueError(f"family {family['id']} has images outside the photon map")
        source = profiled_source(field, images, 1.0)
        if np.any(~np.isfinite(source)):
            raise ValueError(f"family {family['id']} produced a non-finite source")
        roots = find_lens_roots(
            field,
            source,
            1.0,
            bound_arcsec=root_bound,
            observed_starts_arcsec=images,
            grid_points=root_grid_points,
            closure_tolerance_arcsec=closure_tolerance,
            deduplication_tolerance_arcsec=deduplication_tolerance,
            jacobian_step_arcsec=jacobian_step,
            include_residual_minima=include_residual_minima,
            maximum_residual_minimum_seeds=maximum_seeds,
            supplemental_grid_points=tuple(raw_supplemental),
        )
        assignment = assign_observed_roots(images, roots.roots_arcsec)
        score = _score_family(
            images,
            family["uncertainties"],
            roots.roots_arcsec,
            assignment.pairs,
            assignment.complete,
        )
        pair_by_observed = {
            int(observed_index): int(root_index)
            for observed_index, root_index in assignment.pairs
        }
        for image_index, (observed, uncertainty) in enumerate(
            zip(images, family["uncertainties"], strict=True)
        ):
            root_index = pair_by_observed.get(image_index)
            predicted = (
                roots.roots_arcsec[root_index] if root_index is not None else None
            )
            residual = predicted - observed if predicted is not None else None
            prediction_rows.append(
                {
                    "target_id": target_id,
                    "family_id": family["id"],
                    "family_index": family_index,
                    "image_index": image_index,
                    "assignment_state": "matched"
                    if predicted is not None
                    else "unmatched",
                    "observed_east_arcsec": float(observed[0]),
                    "observed_north_arcsec": float(observed[1]),
                    "position_uncertainty_arcsec": float(uncertainty),
                    "predicted_root_index": root_index,
                    "predicted_east_arcsec": float(predicted[0])
                    if predicted is not None
                    else None,
                    "predicted_north_arcsec": float(predicted[1])
                    if predicted is not None
                    else None,
                    "residual_east_arcsec": float(residual[0])
                    if residual is not None
                    else None,
                    "residual_north_arcsec": float(residual[1])
                    if residual is not None
                    else None,
                    "separation_arcsec": float(np.linalg.norm(residual))
                    if residual is not None
                    else None,
                    "root_closure_arcsec": float(roots.closure_arcsec[root_index])
                    if root_index is not None
                    else None,
                    "root_absolute_magnification": float(
                        roots.absolute_magnification[root_index]
                    )
                    if root_index is not None
                    else None,
                }
            )
        family_prefix = f"{archive_prefix}__family_{family_index:03d}"
        root_arrays[f"{family_prefix}__roots_arcsec"] = roots.roots_arcsec
        root_arrays[f"{family_prefix}__closures_arcsec"] = roots.closure_arcsec
        root_arrays[f"{family_prefix}__absolute_magnifications"] = (
            roots.absolute_magnification
        )
        critical_count = 0
        if include_critical:
            critical = critical_curve_points(
                field,
                1.0,
                bound_arcsec=root_bound,
                grid_points=critical_grid_points,
            )
            root_arrays[f"{family_prefix}__critical_curve_points_arcsec"] = critical
            critical_count = len(critical)
        family_record = {
            "target_id": target_id,
            "family_id": family["id"],
            "family_index": family_index,
            "distance_ratio": family["distanceRatio"],
            "profiled_source_east_arcsec": float(source[0]),
            "profiled_source_north_arcsec": float(source[1]),
            "observed_images": len(images),
            "predicted_roots": len(roots.roots_arcsec),
            "matched_images": assignment.matched_images,
            "complete_observed_assignment": assignment.complete,
            "excess_predicted_roots": max(0, len(roots.roots_arcsec) - len(images)),
            "critical_curve_points": critical_count,
            "state": score["state"],
            "image_plane_rms_arcsec": score["imagePlaneRmsArcsec"],
            "matched_subset_diagnostic_rms_arcsec": score.get(
                "matchedSubsetDiagnosticRmsArcsec"
            ),
            "chi_square": score["chiSquare"],
            "degrees_freedom": score["degreesFreedom"],
            "fitted_observation_nuisance_parameters": 2,
            "gravity_parameters_added": 0,
        }
        family_rows.append(family_record)
        family_evaluations.append(
            {
                "id": family["id"],
                "distanceRatio": family["distanceRatio"],
                "profiledSourceArcsec": source.tolist(),
                "observedImages": len(images),
                "predictedRoots": len(roots.roots_arcsec),
                "matchedImages": assignment.matched_images,
                "completeObservedAssignment": assignment.complete,
                "excessPredictedRoots": max(
                    0, len(roots.roots_arcsec) - len(images)
                ),
                "maximumRootClosureArcsec": float(np.max(roots.closure_arcsec))
                if len(roots.closure_arcsec)
                else None,
                "criticalCurvePoints": critical_count,
                "score": score,
            }
        )

    complete = [
        value for value in family_evaluations if value["score"]["state"] == "scored"
    ]
    all_complete = len(complete) == len(family_evaluations)
    total_observed = sum(value["observedImages"] for value in family_evaluations)
    total_matched = sum(value["matchedImages"] for value in family_evaluations)
    total_points = 2 * total_observed
    nuisance_count = 2 * len(family_evaluations)
    if all_complete:
        sum_squared = sum(
            float(value["score"]["sumSquaredResidualArcsec2"])
            for value in complete
        )
        weighted_squared = sum(
            float(value["score"]["inverseVarianceWeightedSquaredResidual"])
            for value in complete
        )
        weight_sum = sum(
            float(value["score"]["inverseVarianceWeightSum"])
            for value in complete
        )
        chi_square = sum(float(value["score"]["chiSquare"]) for value in complete)
        degrees_freedom = sum(
            int(value["score"]["degreesFreedom"]) for value in complete
        )
        channel_score = {
            "channel": "image_position_arcsec",
            "unit": "arcsec",
            "state": "scored",
            "totalPoints": total_points,
            "validPoints": total_points,
            "fittedNuisanceParameters": nuisance_count,
            "sumSquaredResidual": sum_squared,
            "rmse": math.sqrt(sum_squared / total_points),
            "imagePlaneRmsArcsec": math.sqrt(sum_squared / total_observed),
            "inverseVarianceWeightedSquaredResidual": weighted_squared,
            "inverseVarianceWeightSum": weight_sum,
            "inverseVarianceWeightedRmse": math.sqrt(weighted_squared / weight_sum),
            "chiSquare": chi_square,
            "degreesFreedom": degrees_freedom,
            "reducedChiSquare": chi_square / degrees_freedom,
            "gaussianLogLikelihood": sum(
                float(value["score"]["gaussianLogLikelihood"]) for value in complete
            ),
            "completeTopologyFraction": 1.0,
        }
        state = "scored"
    else:
        complete_images = sum(value["observedImages"] for value in complete)
        complete_sum_squared = sum(
            float(value["score"]["sumSquaredResidualArcsec2"])
            for value in complete
        )
        channel_score = {
            "channel": "image_position_arcsec",
            "unit": "arcsec",
            "state": "incomplete_topology",
            "totalPoints": total_points,
            "validPoints": 2 * total_matched,
            "fittedNuisanceParameters": nuisance_count,
            "sumSquaredResidual": None,
            "rmse": None,
            "imagePlaneRmsArcsec": None,
            "chiSquare": None,
            "degreesFreedom": None,
            "gaussianLogLikelihood": None,
            "completeTopologyFraction": len(complete) / len(family_evaluations),
            "completeFamilyDiagnosticImagePlaneRmsArcsec": math.sqrt(
                complete_sum_squared / complete_images
            )
            if complete_images
            else None,
        }
        state = "incomplete_topology"
    target_score = {
        "state": state,
        "totalPoints": total_points,
        "validPoints": channel_score["validPoints"],
        "fittedNuisanceParameters": nuisance_count,
        "channels": {"image_position_arcsec": channel_score},
    }
    return (
        {
            "id": target_id,
            "kind": "multiple_image_systems",
            "observable": observable_id,
            "observableTarget": definition.get("target"),
            "state": state,
            "originM": origin.tolist(),
            "originSource": origin_source,
            "skyCenterM": sky_center.tolist(),
            "axisConvention": {
                "observedCoordinateOrder": ["east_arcsec", "north_arcsec"],
                "northAxis": north_axis,
                "eastAxis": east_axis,
                "lineOfSightAxis": axes[2],
            },
            "lensAngularDiameterDistanceM": lens_distance,
            "rootSearch": {
                "boundArcsec": root_bound,
                "gridPoints": root_grid_points,
                "closureToleranceArcsec": closure_tolerance,
                "deduplicationToleranceArcsec": deduplication_tolerance,
                "jacobianStepArcsec": jacobian_step,
                "includeResidualMinima": include_residual_minima,
                "maximumResidualMinimumSeeds": maximum_seeds,
                "supplementalGridPoints": raw_supplemental,
            },
            "familyCount": len(family_evaluations),
            "completeFamilyCount": len(complete),
            "incompleteFamilyCount": len(family_evaluations) - len(complete),
            "observedImageCount": total_observed,
            "matchedImageCount": total_matched,
            "fittedObservationNuisanceParameters": nuisance_count,
            "gravityParametersAdded": 0,
            "rootArchivePrefix": archive_prefix,
            "families": family_evaluations,
            "score": target_score,
            "claimBoundary": [
                "Source positions are profiled and counted as two observational nuisance parameters per family.",
                "Missing observed multiplicity produces incomplete_topology and no finite aggregate RMS, chi-square, or likelihood.",
                "Extra predicted roots are disclosed but not classified without an explicit detectability model.",
                "Critical curves are diagnostic unless independent parity or arc-orientation data are supplied.",
            ],
        },
        prediction_rows,
        family_rows,
        root_arrays,
    )
