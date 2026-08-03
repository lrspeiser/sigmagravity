"""Content-addressed jobs and artifacts for the generic field worker.

The Vercel gateway will eventually enqueue this exact job shape.  Local paths
are deliberately excluded from the scientific job identity: the identity is a
hash of the canonical model, verified array bundle, grid/boundary settings,
requested observables, seed, and worker source.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import shutil
import tempfile
import time
import tracemalloc
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rfc8785
import scipy

from .generic_field_worker import GenericFieldSolution, solve_field_manifest
from .observation_adapters import evaluate_observation_targets

Array = np.ndarray
ENGINE_ID = "generic-divergence-field-worker"
ENGINE_VERSION = "1.5.0-preview"
MODEL_HASH_KEYS = (
    "schemaVersion",
    "modelClass",
    "geometry",
    "fields",
    "parameters",
    "equations",
    "observables",
    "dataRequirements",
    "solver",
    "parameterPolicy",
)


def canonical_bytes(value: Any) -> bytes:
    """Return RFC 8785 JSON Canonicalization Scheme bytes."""

    return rfc8785.dumps(_json_ready(value))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        raise TypeError("arrays must be content-hashed, not embedded in canonical JSON")
    return value


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def model_sha256(model: Mapping[str, Any]) -> str:
    """Match the hosted validator's hash of computational model content."""

    return canonical_sha256({key: model[key] for key in MODEL_HASH_KEYS})


def require_model_confirmation(model: Mapping[str, Any]) -> str:
    """Require an explicit confirmation bound to the exact model hash."""

    expected = model_sha256(model)
    source = model.get("source", {})
    if source.get("confirmedCanonical") is not True:
        raise ValueError("model canonical form has not been confirmed by the researcher")
    if source.get("confirmedModelSha256") != expected:
        raise ValueError(
            "model confirmation does not match the exact computational model hash"
        )
    return expected


def document_sha256(model: Mapping[str, Any]) -> str:
    return canonical_sha256(model)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_array(values: Array) -> Array:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot enter a scientific bundle")
    if array.dtype.byteorder not in {"|", "<"}:
        target = array.dtype.newbyteorder("<")
        array = array.astype(target, copy=False)
    return np.ascontiguousarray(array)


def array_content_sha256(values: Array) -> str:
    array = _canonical_array(values)
    header = {
        "schemaVersion": "sigma-array-content/1",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "order": "C",
    }
    digest = hashlib.sha256()
    digest.update(canonical_bytes(header))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _prepare_new_directory(target: Path) -> tuple[Path, Path]:
    resolved = Path(target).resolve()
    if resolved.exists():
        if any(resolved.iterdir()):
            raise FileExistsError(f"immutable output directory is not empty: {resolved}")
        resolved.rmdir()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{resolved.name}-", dir=resolved.parent))
    return resolved, temporary


def _publish_directory(temporary: Path, target: Path) -> None:
    try:
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def write_array_bundle(
    output_directory: Path,
    arrays: Mapping[str, Array],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write and hash a versioned array bundle.

    ``metadata.arrays`` maps public dataset keys to their unit, rank, role, and
    optional ``npzKey``.  Geometry and provenance travel with the data rather
    than being inferred from filenames.
    """

    if metadata.get("schemaVersion") != "sigma-array-bundle-request/1":
        raise ValueError("metadata schemaVersion must be sigma-array-bundle-request/1")
    descriptions = metadata.get("arrays")
    if not isinstance(descriptions, Mapping) or not descriptions:
        raise ValueError("array bundle metadata requires a non-empty arrays object")
    supplied = {str(key): _canonical_array(value) for key, value in arrays.items()}
    target, temporary = _prepare_new_directory(Path(output_directory))
    try:
        records: list[dict[str, Any]] = []
        stored: dict[str, Array] = {}
        for public_key in sorted(descriptions):
            description = dict(descriptions[public_key])
            npz_key = str(description.pop("npzKey", public_key))
            if npz_key not in supplied:
                raise ValueError(f"array {public_key} requires missing npz key {npz_key}")
            array = supplied[npz_key]
            if array.size == 0 or np.any(~np.isfinite(array)):
                raise ValueError(f"array {public_key} must be non-empty and finite")
            if description.get("rank") not in {"scalar", "vector", "tensor2"}:
                raise ValueError(f"array {public_key} requires rank scalar, vector, or tensor2")
            if not isinstance(description.get("unit"), str) or not description["unit"]:
                raise ValueError(f"array {public_key} requires a unit")
            stored[public_key] = array
            records.append(
                {
                    "key": public_key,
                    "npzKey": public_key,
                    **description,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "elementCount": int(array.size),
                    "contentSha256": array_content_sha256(array),
                }
            )
        unused = sorted(set(supplied) - {str(value.get("npzKey", key)) for key, value in descriptions.items()})
        if unused:
            raise ValueError(f"input NPZ contains undeclared arrays: {', '.join(unused)}")
        bundle_core = {
            "schemaVersion": "sigma-array-bundle/1",
            "geometry": metadata.get("geometry"),
            "arrays": records,
            "provenance": metadata.get("provenance"),
            "license": metadata.get("license"),
        }
        if not isinstance(bundle_core["geometry"], Mapping):
            raise TypeError("array bundle metadata requires geometry")
        if not isinstance(bundle_core["provenance"], Mapping):
            raise TypeError("array bundle metadata requires provenance")
        if not isinstance(bundle_core["license"], Mapping):
            raise TypeError("array bundle metadata requires license")
        bundle = {**bundle_core, "bundleSha256": canonical_sha256(bundle_core)}
        with (temporary / "arrays.npz").open("wb") as handle:
            np.savez_compressed(handle, **stored)
        _write_json(temporary / "bundle.json", bundle)
        _publish_directory(temporary, target)
        return bundle
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_array_bundle(directory: Path) -> tuple[dict[str, Any], dict[str, Array]]:
    root = Path(directory).resolve()
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    geometry = bundle.get("geometry")
    if isinstance(geometry, dict):
        if "spacing" in geometry:
            raw_spacing = geometry["spacing"]
            geometry["spacing"] = (
                [float(value) for value in raw_spacing]
                if isinstance(raw_spacing, list)
                else float(raw_spacing)
            )
        if "origin" in geometry:
            geometry["origin"] = [float(value) for value in geometry["origin"]]
    claimed = bundle.get("bundleSha256")
    core = {key: value for key, value in bundle.items() if key != "bundleSha256"}
    if claimed != canonical_sha256(core):
        raise ValueError("array bundle manifest hash mismatch")
    arrays: dict[str, Array] = {}
    with np.load(root / "arrays.npz", allow_pickle=False) as archive:
        declared_keys = {record["npzKey"] for record in bundle.get("arrays", [])}
        if set(archive.files) != declared_keys:
            raise ValueError("array bundle NPZ keys do not match bundle manifest")
        for record in bundle.get("arrays", []):
            values = _canonical_array(archive[record["npzKey"]])
            if list(values.shape) != record.get("shape") or values.dtype.str != record.get("dtype"):
                raise ValueError(f"array {record['key']} shape or dtype mismatch")
            if array_content_sha256(values) != record.get("contentSha256"):
                raise ValueError(f"array {record['key']} content hash mismatch")
            arrays[str(record["key"])] = values
    return bundle, arrays


def _worker_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "field_job.py",
        "generic_field_worker.py",
        "observation_adapters.py",
        "photon_lensing_adapter.py",
        "multiple_image_adapter.py",
        "sky_lensing.py",
    ):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _parameter_accounting(model: Mapping[str, Any]) -> dict[str, Any]:
    parameters = model.get("parameters", {})
    universal = sorted(
        name for name, value in parameters.items() if value.get("scope") == "universal"
    )
    per_object = sorted(
        name for name, value in parameters.items() if value.get("scope") == "per_object"
    )
    return {
        "mode": model.get("parameterPolicy", {}).get("mode"),
        "universalCount": len(universal),
        "universalNames": universal,
        "perObjectCount": len(per_object),
        "perObjectNames": per_object,
    }


def _boundary_values(
    specifications: Mapping[str, Any], arrays: Mapping[str, Array]
) -> dict[str, float | Array]:
    values: dict[str, float | Array] = {}
    for field_name, specification in specifications.items():
        if isinstance(specification, (int, float)):
            values[str(field_name)] = float(specification)
        elif isinstance(specification, Mapping) and "value" in specification:
            values[str(field_name)] = float(specification["value"])
        elif isinstance(specification, Mapping) and "arrayKey" in specification:
            key = str(specification["arrayKey"])
            if key not in arrays:
                raise ValueError(f"boundary {field_name} requires missing array {key}")
            values[str(field_name)] = arrays[key]
        else:
            raise ValueError(f"invalid boundary specification for {field_name}")
    return values


def _array_records(values: Mapping[str, Array]) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "dtype": _canonical_array(array).dtype.str,
            "shape": list(np.asarray(array).shape),
            "contentSha256": array_content_sha256(array),
        }
        for key, array in sorted(values.items())
    ]


def _flatten_observables(solution: GenericFieldSolution) -> dict[str, Array]:
    flattened: dict[str, Array] = {}
    for name, value in sorted(solution.observables.items()):
        if isinstance(value, tuple):
            for axis, component in enumerate(value):
                flattened[f"{name}__axis{axis}"] = _canonical_array(component)
        else:
            flattened[name] = _canonical_array(np.asarray(value))
    return flattened


def _write_npz(path: Path, values: Mapping[str, Array]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **values)


def _write_deterministic_npz(path: Path, values: Mapping[str, Array]) -> None:
    """Write a byte-stable compressed NumPy archive with fixed ZIP metadata."""

    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for key, raw in sorted(values.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, _canonical_array(raw), allow_pickle=False
            )
            member = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _write_residual_history(path: Path, solution: GenericFieldSolution) -> None:
    equation_ids = sorted(solution.equation_residuals)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration", "maximum_relative_update", *equation_ids],
            lineterminator="\n",
        )
        writer.writeheader()
        for record in solution.residual_history:
            writer.writerow(
                {
                    "iteration": record["iteration"],
                    "maximum_relative_update": format(
                        record["maximum_relative_update"], ".17g"
                    ),
                    **{
                        equation_id: format(
                            record["equation_residuals"][equation_id], ".17g"
                        )
                        for equation_id in equation_ids
                    },
                }
            )


def _write_observation_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "target_id",
        "point_index",
        "radius_m",
        "predicted_speed_m_s",
        "observed_speed_m_s",
        "uncertainty_m_s",
        "residual_m_s",
        "azimuthal_coverage",
        "mean_inward_acceleration_m_s2",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_velocity_field_predictions(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    columns = [
        "target_id",
        "point_index",
        "row_index",
        "column_index",
        "disk_major_coordinate_m",
        "disk_minor_coordinate_m",
        "circular_radius_m",
        "predicted_circular_speed_m_s",
        "predicted_velocity_m_s",
        "observed_velocity_m_s",
        "uncertainty_m_s",
        "residual_m_s",
        "declared_weight",
        "inward_acceleration_m_s2",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_multiple_image_predictions(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    columns = [
        "target_id",
        "family_id",
        "family_index",
        "image_index",
        "assignment_state",
        "observed_east_arcsec",
        "observed_north_arcsec",
        "position_uncertainty_arcsec",
        "predicted_root_index",
        "predicted_east_arcsec",
        "predicted_north_arcsec",
        "residual_east_arcsec",
        "residual_north_arcsec",
        "separation_arcsec",
        "root_closure_arcsec",
        "root_absolute_magnification",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_multiple_image_families(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    columns = [
        "target_id",
        "family_id",
        "family_index",
        "distance_ratio",
        "profiled_source_east_arcsec",
        "profiled_source_north_arcsec",
        "observed_images",
        "predicted_roots",
        "matched_images",
        "complete_observed_assignment",
        "excess_predicted_roots",
        "critical_curve_points",
        "state",
        "image_plane_rms_arcsec",
        "matched_subset_diagnostic_rms_arcsec",
        "chi_square",
        "degrees_freedom",
        "fitted_observation_nuisance_parameters",
        "gravity_parameters_added",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _normalize_observation_targets(values: Any) -> list[dict[str, Any]]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise TypeError("observationTargets must be an array")
    normalized: list[dict[str, Any]] = []
    vector_keys = (
        "gridOriginM",
        "centerM",
        "radiiM",
        "observedSpeedsMPerS",
        "uncertaintiesMPerS",
        "skyCenterM",
    )
    for raw in values:
        if not isinstance(raw, Mapping):
            raise TypeError("observation target must be an object")
        target = dict(raw)
        for key in vector_keys:
            if key in target:
                target[key] = [float(value) for value in target[key]]
        if "covarianceM2PerS2" in target:
            target["covarianceM2PerS2"] = [
                [float(value) for value in row] for row in target["covarianceM2PerS2"]
            ]
        if "minimumAzimuthalCoverage" in target:
            target["minimumAzimuthalCoverage"] = float(
                target["minimumAzimuthalCoverage"]
            )
        for key in ("inclinationDeg", "observedVelocityZeroPointMPerS"):
            if key in target:
                target[key] = float(target[key])
        for key in ("distanceRatio", "lensAngularDiameterDistanceM"):
            if key in target:
                target[key] = float(target[key])
        if "families" in target:
            target["families"] = [
                {
                    **dict(family),
                    "distanceRatio": float(family["distanceRatio"]),
                    "observedImagesArcsec": [
                        [float(value) for value in image]
                        for image in family["observedImagesArcsec"]
                    ],
                    "positionUncertaintiesArcsec": [
                        float(value)
                        for value in family["positionUncertaintiesArcsec"]
                    ],
                }
                for family in target["families"]
            ]
        normalized.append(target)
    return normalized


def execute_field_job(
    model: Mapping[str, Any],
    input_bundle_directory: Path,
    request: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    """Execute one immutable field job and write a citation-ready artifact set."""

    if request.get("schemaVersion") != "sigma-field-job-request/1":
        raise ValueError("request schemaVersion must be sigma-field-job-request/1")
    require_model_confirmation(model)
    bundle, arrays = load_array_bundle(input_bundle_directory)
    model_geometry = model.get("geometry", {})
    bundle_geometry = bundle.get("geometry", {})
    for key in ("coordinateSystem", "dimensions"):
        if model_geometry.get(key) != bundle_geometry.get(key):
            raise ValueError(f"model and array bundle disagree on geometry {key}")
    spacing = request.get("spacing", bundle_geometry.get("spacing"))
    dimensions = int(model_geometry.get("dimensions", 0))
    if isinstance(spacing, (int, float)):
        spacing_values = [float(spacing)] * dimensions
    else:
        spacing_values = [float(value) for value in spacing]
    if len(spacing_values) != dimensions or any(value <= 0 for value in spacing_values):
        raise ValueError("job spacing must contain one positive value per dimension")

    required_keys = {str(item["key"]) for item in model.get("dataRequirements", [])}
    missing = sorted(required_keys - set(arrays))
    if missing:
        raise ValueError(f"array bundle is missing model inputs: {', '.join(missing)}")
    source_fields = {key: arrays[key] for key in required_keys}
    available_observables = {str(value["id"]) for value in model.get("observables", [])}
    requested_observables = sorted(
        str(value) for value in request.get("requestedObservables", available_observables)
    )
    unknown_observables = sorted(set(requested_observables) - available_observables)
    if unknown_observables:
        raise ValueError(f"unknown requested observables: {', '.join(unknown_observables)}")
    boundaries = _boundary_values(request.get("boundaryFields", {}), arrays)
    observation_targets = _normalize_observation_targets(
        request.get("observationTargets", [])
    )
    worker_source_sha = _worker_source_sha256()
    job_core = {
        "schemaVersion": "sigma-field-job/1",
        "modelSha256": model_sha256(model),
        "modelDocumentSha256": document_sha256(model),
        "inputBundleSha256": bundle["bundleSha256"],
        "geometry": {
            "coordinateSystem": model_geometry.get("coordinateSystem"),
            "dimensions": dimensions,
            "spacing": spacing_values,
            "origin": bundle_geometry.get("origin"),
            "axisOrder": bundle_geometry.get("axisOrder"),
            "lengthUnit": model_geometry.get("domain", {}).get("lengthUnit"),
        },
        "boundaryFields": request.get("boundaryFields", {}),
        "requestedObservables": requested_observables,
        "observationTargets": observation_targets,
        "solver": model.get("solver"),
        "parameterPolicy": model.get("parameterPolicy"),
        "seed": int(request.get("seed", 0)),
        "worker": {
            "engine": ENGINE_ID,
            "version": ENGINE_VERSION,
            "sourceSha256": worker_source_sha,
        },
    }
    job_sha = canonical_sha256(job_core)
    job = {**job_core, "jobSha256": job_sha, "id": f"fieldjob_{job_sha[:24]}"}
    target, temporary = _prepare_new_directory(Path(output_directory))
    started_at = datetime.now(UTC)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    tracemalloc.start()
    try:
        solution = solve_field_manifest(
            model,
            source_fields,
            spacing_values,
            boundary_values=boundaries,
            grid_geometry=bundle_geometry,
        )
        _current_memory, peak_memory = tracemalloc.get_traced_memory()
        wall_seconds = time.perf_counter() - wall_started
        cpu_seconds = time.process_time() - cpu_started
        fields = {name: _canonical_array(value) for name, value in solution.fields.items()}
        observables = {
            key: value
            for key, value in _flatten_observables(solution).items()
            if key.split("__axis", maxsplit=1)[0] in requested_observables
        }
        observation_rows: list[dict[str, Any]] = []
        observation_maps: dict[str, Array] = {}
        observation_roots: dict[str, Array] = {}
        observation_auxiliary_rows: dict[str, list[dict[str, Any]]] = {}
        observation_evaluation: dict[str, Any] | None = None
        if observation_targets:
            if solution.converged:
                observation_evaluation, observation_rows = evaluate_observation_targets(
                    model,
                    observables,
                    {**bundle_geometry, "spacing": spacing_values},
                    observation_targets,
                    arrays=arrays,
                    map_outputs=observation_maps,
                    root_outputs=observation_roots,
                    auxiliary_rows=observation_auxiliary_rows,
                )
                for evaluation, target_specification in zip(
                    observation_evaluation["targets"], observation_targets, strict=True
                ):
                    evaluation["targetSha256"] = canonical_sha256(target_specification)
                if observation_maps:
                    observation_evaluation["mapArchive"] = {
                        "path": "observation_photon_lensing_maps.npz",
                        "maps": _array_records(observation_maps),
                    }
                if observation_roots:
                    observation_evaluation["rootArchive"] = {
                        "path": "observation_multiple_image_roots.npz",
                        "arrays": _array_records(observation_roots),
                    }
            else:
                observation_evaluation = {
                    "schemaVersion": "sigma-observation-evaluation/1",
                    "state": "unavailable_nonconvergence",
                    "targetKinds": sorted(
                        {str(target.get("kind")) for target in observation_targets}
                    ),
                    "targetCount": len(observation_targets),
                    "scoredTargetCount": 0,
                    "totalPoints": 0,
                    "validScoredPoints": 0,
                    "targets": [],
                }
        _write_npz(temporary / "fields.npz", fields)
        _write_npz(temporary / "observables.npz", observables)
        _write_residual_history(temporary / "residual_history.csv", solution)
        if observation_maps:
            _write_deterministic_npz(
                temporary / "observation_photon_lensing_maps.npz",
                observation_maps,
            )
        if observation_roots:
            _write_deterministic_npz(
                temporary / "observation_multiple_image_roots.npz",
                observation_roots,
            )
        circular_prediction_rows = [
            row for row in observation_rows if "predicted_speed_m_s" in row
        ]
        velocity_field_prediction_rows = [
            row for row in observation_rows if "predicted_velocity_m_s" in row
        ]
        multiple_image_prediction_rows = [
            row for row in observation_rows if "assignment_state" in row
        ]
        multiple_image_family_rows = observation_auxiliary_rows.get(
            "multiple_image_families", []
        )
        evaluated_target_kinds = (
            set(observation_evaluation.get("targetKinds", []))
            if observation_evaluation is not None and solution.converged
            else set()
        )
        if observation_evaluation is not None:
            _write_json(temporary / "observation_scores.json", observation_evaluation)
        if "circular_speed_curve" in evaluated_target_kinds:
            _write_observation_predictions(
                temporary / "observation_predictions.csv", circular_prediction_rows
            )
        if "line_of_sight_velocity_field" in evaluated_target_kinds:
            _write_velocity_field_predictions(
                temporary / "observation_velocity_field_predictions.csv",
                velocity_field_prediction_rows,
            )
        if "multiple_image_systems" in evaluated_target_kinds:
            _write_multiple_image_predictions(
                temporary / "observation_multiple_image_predictions.csv",
                multiple_image_prediction_rows,
            )
            _write_multiple_image_families(
                temporary / "observation_multiple_image_families.csv",
                multiple_image_family_rows,
            )
        _write_json(temporary / "model.json", model)
        _write_json(temporary / "input_bundle.json", bundle)
        _write_json(temporary / "job.json", job)

        scientific_core = {
            "schemaVersion": "sigma-field-result/1",
            "jobId": job["id"],
            "jobSha256": job_sha,
            "state": "succeeded" if solution.converged else "failed_nonconvergence",
            "converged": solution.converged,
            "iterations": solution.iterations,
            "maximumRelativeUpdate": solution.maximum_relative_update,
            "equationResiduals": solution.equation_residuals,
            "fields": _array_records(fields),
            "observables": _array_records(observables),
            "observationEvaluation": observation_evaluation,
            "parameterAccounting": _parameter_accounting(model),
            "numericalMetadata": solution.metadata,
            "claimBoundary": [
                "A converged field validates execution of the submitted numerical equation, not agreement with galaxy or lensing observations.",
                "The current generic worker uses supplied or zero far-field Dirichlet boundaries for isolated manifests.",
                "Observation targets are evaluated after the field solve and cannot alter the field equations.",
                "Resolved velocity maps use explicitly declared projection and beam data; no galaxy-specific gravity parameter is introduced by the adapter.",
                "Photon-lensing maps use a separately typed photon observable and explicitly declared projection geometry; they are not derived from the massive-tracer adapter.",
                "Raw multiple-image scoring profiles source locations but adds no gravity parameter; missing predicted multiplicity is reported without a finite aggregate fit score.",
            ],
        }
        result_sha = canonical_sha256(scientific_core)
        scientific_result = {**scientific_core, "resultSha256": result_sha}
        _write_json(temporary / "scientific_result.json", scientific_result)
        resource_log = {
            "schemaVersion": "sigma-field-resource-log/1",
            "jobId": job["id"],
            "startedAt": started_at.isoformat(),
            "finishedAt": datetime.now(UTC).isoformat(),
            "wallSeconds": wall_seconds,
            "cpuSeconds": cpu_seconds,
            "peakPythonHeapBytes": int(peak_memory),
            "processId": os.getpid(),
        }
        _write_json(temporary / "resource_log.json", resource_log)
        artifact_names = [
            "model.json",
            "input_bundle.json",
            "job.json",
            "scientific_result.json",
            "fields.npz",
            "observables.npz",
            "residual_history.csv",
            "resource_log.json",
        ]
        if observation_evaluation is not None:
            artifact_names.append("observation_scores.json")
        if "circular_speed_curve" in evaluated_target_kinds:
            artifact_names.append("observation_predictions.csv")
        if "line_of_sight_velocity_field" in evaluated_target_kinds:
            artifact_names.append("observation_velocity_field_predictions.csv")
        if "multiple_image_systems" in evaluated_target_kinds:
            artifact_names.extend(
                [
                    "observation_multiple_image_predictions.csv",
                    "observation_multiple_image_families.csv",
                ]
            )
        if observation_maps:
            artifact_names.append("observation_photon_lensing_maps.npz")
        if observation_roots:
            artifact_names.append("observation_multiple_image_roots.npz")
        artifact_index = {
            "schemaVersion": "sigma-field-artifact-index/1",
            "jobId": job["id"],
            "artifacts": [
                {
                    "path": name,
                    "bytes": (temporary / name).stat().st_size,
                    "sha256": file_sha256(temporary / name),
                }
                for name in artifact_names
            ],
        }
        _write_json(temporary / "artifact_index.json", artifact_index)
        manifest_core = {
            "schemaVersion": "sigma-field-run-manifest/1",
            "state": scientific_result["state"],
            "jobId": job["id"],
            "jobSha256": job_sha,
            "scientificResultSha256": result_sha,
            "artifactIndexSha256": file_sha256(temporary / "artifact_index.json"),
            "worker": job["worker"],
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "reproduction": {
                "command": "python scripts/run_generic_field_job.py run --request <request.json>",
                "requiredInputs": [job["modelSha256"], job["inputBundleSha256"]],
            },
        }
        manifest = {
            **manifest_core,
            "manifestSha256": canonical_sha256(manifest_core),
            "createdAt": datetime.now(UTC).isoformat(),
        }
        _write_json(temporary / "manifest.json", manifest)
        _publish_directory(temporary, target)
        return manifest
    except Exception as error:  # noqa: BLE001 - worker failures must become artifacts
        try:
            _current_memory, peak_memory = tracemalloc.get_traced_memory()
            _write_json(temporary / "model.json", model)
            _write_json(temporary / "input_bundle.json", bundle)
            _write_json(temporary / "job.json", job)
            failure_core = {
                "schemaVersion": "sigma-field-failure/1",
                "jobId": job["id"],
                "jobSha256": job_sha,
                "state": "failed",
                "errorType": type(error).__name__,
                "message": str(error),
            }
            failure = {
                **failure_core,
                "failureSha256": canonical_sha256(failure_core),
            }
            _write_json(temporary / "failure.json", failure)
            resource_log = {
                "schemaVersion": "sigma-field-resource-log/1",
                "jobId": job["id"],
                "startedAt": started_at.isoformat(),
                "finishedAt": datetime.now(UTC).isoformat(),
                "wallSeconds": time.perf_counter() - wall_started,
                "cpuSeconds": time.process_time() - cpu_started,
                "peakPythonHeapBytes": int(peak_memory),
                "processId": os.getpid(),
            }
            _write_json(temporary / "resource_log.json", resource_log)
            artifact_names = [
                "model.json",
                "input_bundle.json",
                "job.json",
                "failure.json",
                "resource_log.json",
            ]
            artifact_index = {
                "schemaVersion": "sigma-field-artifact-index/1",
                "jobId": job["id"],
                "artifacts": [
                    {
                        "path": name,
                        "bytes": (temporary / name).stat().st_size,
                        "sha256": file_sha256(temporary / name),
                    }
                    for name in artifact_names
                ],
            }
            _write_json(temporary / "artifact_index.json", artifact_index)
            manifest_core = {
                "schemaVersion": "sigma-field-run-manifest/1",
                "state": "failed",
                "jobId": job["id"],
                "jobSha256": job_sha,
                "failureSha256": failure["failureSha256"],
                "artifactIndexSha256": file_sha256(
                    temporary / "artifact_index.json"
                ),
                "worker": job["worker"],
                "environment": {
                    "python": platform.python_version(),
                    "implementation": platform.python_implementation(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
                "reproduction": {
                    "command": "python scripts/run_generic_field_job.py run --request <request.json>",
                    "requiredInputs": [job["modelSha256"], job["inputBundleSha256"]],
                },
            }
            manifest = {
                **manifest_core,
                "manifestSha256": canonical_sha256(manifest_core),
                "createdAt": datetime.now(UTC).isoformat(),
            }
            _write_json(temporary / "manifest.json", manifest)
            _publish_directory(temporary, target)
            return manifest
        except Exception:  # noqa: BLE001 - preserve the original solver failure
            shutil.rmtree(temporary, ignore_errors=True)
            raise error
    finally:
        tracemalloc.stop()


def _resolve_within(base: Path, value: str | Path, label: str) -> Path:
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the request directory") from error
    return candidate


def execute_request_file(request_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    path = Path(request_path).resolve()
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if envelope.get("schemaVersion") != "sigma-field-job-cli/1":
        raise ValueError("CLI envelope schemaVersion must be sigma-field-job-cli/1")
    base = path.parent
    model_path = _resolve_within(base, envelope["modelPath"], "modelPath")
    bundle_path = _resolve_within(base, envelope["inputBundlePath"], "inputBundlePath")
    output = (
        _resolve_within(base, output_override, "output")
        if output_override is not None
        else _resolve_within(base, envelope["outputDirectory"], "outputDirectory")
    )
    model = json.loads(model_path.read_text(encoding="utf-8"))
    return execute_field_job(model, bundle_path, envelope["request"], output)


def package_array_file(
    arrays_path: Path, metadata_path: Path, output_directory: Path
) -> dict[str, Any]:
    with np.load(Path(arrays_path), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return write_array_bundle(output_directory, arrays, metadata)
