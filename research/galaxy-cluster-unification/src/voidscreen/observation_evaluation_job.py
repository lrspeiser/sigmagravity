"""Content-addressed observation evaluation on an immutable solved field.

This worker deliberately has no field-solver entry point.  It validates and
loads a completed field job's observable archive, applies the public
observation adapters, and publishes a separately hashed result bundle.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
import tracemalloc
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from .field_job import (
    _array_records,
    _normalize_observation_targets,
    _parameter_accounting,
    _prepare_new_directory,
    _publish_directory,
    _write_deterministic_npz,
    _write_json,
    _write_multiple_image_families,
    _write_multiple_image_predictions,
    _write_observation_predictions,
    _write_velocity_field_predictions,
    array_content_sha256,
    canonical_sha256,
    document_sha256,
    file_sha256,
    load_array_bundle,
    model_sha256,
)
from .observation_adapters import evaluate_observation_targets

Array = np.ndarray
ENGINE_ID = "generic-observation-evaluation-worker"
ENGINE_VERSION = "1.2.0-preview"


def _worker_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    names = (
        "observation_evaluation_job.py",
        "field_job.py",
        "observation_adapters.py",
        "photon_lensing_adapter.py",
        "multiple_image_adapter.py",
        "sky_lensing.py",
    )
    import hashlib

    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _verify_field_artifact(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Array]]:
    root = Path(directory).resolve()
    model = _read_json(root / "model.json")
    job = _read_json(root / "job.json")
    result = _read_json(root / "scientific_result.json")
    if result.get("schemaVersion") != "sigma-field-result/1":
        raise ValueError("source scientific result must use sigma-field-result/1")
    if result.get("state") != "succeeded" or result.get("converged") is not True:
        raise ValueError("source field must be a converged successful result")
    result_core = {key: value for key, value in result.items() if key != "resultSha256"}
    if canonical_sha256(result_core) != result.get("resultSha256"):
        raise ValueError("source scientific result hash mismatch")
    job_core = {
        key: value for key, value in job.items() if key not in {"id", "jobSha256"}
    }
    if canonical_sha256(job_core) != job.get("jobSha256"):
        raise ValueError("source field job hash mismatch")
    if job.get("id") != f"fieldjob_{job['jobSha256'][:24]}":
        raise ValueError("source field job identifier mismatch")
    if result.get("jobId") != job.get("id") or result.get("jobSha256") != job.get(
        "jobSha256"
    ):
        raise ValueError("source field job and scientific result disagree")
    if model_sha256(model) != job.get("modelSha256"):
        raise ValueError("source field model hash mismatch")
    if document_sha256(model) != job.get("modelDocumentSha256"):
        raise ValueError("source field model document hash mismatch")

    records = result.get("observables")
    if not isinstance(records, list) or not records:
        raise ValueError("source field result has no observable records")
    record_by_key = {str(record.get("key")): record for record in records}
    if len(record_by_key) != len(records):
        raise ValueError("source field observable records contain duplicate keys")
    observables: dict[str, Array] = {}
    with np.load(root / "observables.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(record_by_key):
            raise ValueError("source observable archive keys do not match its records")
        for key in sorted(record_by_key):
            values = np.ascontiguousarray(archive[key])
            record = record_by_key[key]
            if values.dtype.hasobject:
                raise TypeError("source observable archive cannot contain object arrays")
            if list(values.shape) != record.get("shape") or values.dtype.str != record.get(
                "dtype"
            ):
                raise ValueError(f"source observable {key} shape or dtype mismatch")
            if array_content_sha256(values) != record.get("contentSha256"):
                raise ValueError(f"source observable {key} content hash mismatch")
            observables[key] = values
    if _array_records(observables) != records:
        raise ValueError("source observable record ordering or content mismatch")
    return model, job, result, observables


def _verify_expected_field_reference(
    directory: Path, expected: Mapping[str, Any] | None
) -> dict[str, Any]:
    root = Path(directory).resolve()
    actual = {
        "gatewayJobId": expected.get("gatewayJobId") if expected else None,
        "manifestSha256": expected.get("manifestSha256") if expected else None,
        "modelArtifactSha256": file_sha256(root / "model.json"),
        "jobArtifactSha256": file_sha256(root / "job.json"),
        "scientificResultArtifactSha256": file_sha256(
            root / "scientific_result.json"
        ),
        "observableArchiveSha256": file_sha256(root / "observables.npz"),
    }
    if expected:
        for key in (
            "modelArtifactSha256",
            "jobArtifactSha256",
            "scientificResultArtifactSha256",
            "observableArchiveSha256",
        ):
            if expected.get(key) != actual[key]:
                raise ValueError(f"source field reference {key} mismatch")
    return actual


def execute_observation_evaluation_job(
    field_artifact_directory: Path,
    observation_bundle_directory: Path,
    request: Mapping[str, Any],
    output_directory: Path,
    *,
    expected_field_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate observations without importing or calling the field solver."""

    if request.get("schemaVersion") != "sigma-observation-evaluation-job-request/1":
        raise ValueError(
            "request schemaVersion must be sigma-observation-evaluation-job-request/1"
        )
    model, field_job, field_result, observables = _verify_field_artifact(
        field_artifact_directory
    )
    field_reference = _verify_expected_field_reference(
        field_artifact_directory, expected_field_reference
    )
    observation_bundle, observation_arrays = load_array_bundle(
        observation_bundle_directory
    )
    targets = _normalize_observation_targets(request.get("observationTargets"))
    if not targets:
        raise ValueError("observationTargets must contain at least one target")
    geometry = field_job.get("geometry")
    if not isinstance(geometry, Mapping):
        raise TypeError("source field job has no geometry")
    worker = {
        "engine": ENGINE_ID,
        "version": ENGINE_VERSION,
        "sourceSha256": _worker_source_sha256(),
    }
    evaluation_core = {
        "schemaVersion": "sigma-observation-evaluation-job/1",
        "field": {
            **field_reference,
            "fieldJobId": field_job["id"],
            "fieldJobSha256": field_job["jobSha256"],
            "fieldScientificResultSha256": field_result["resultSha256"],
            "modelSha256": field_job["modelSha256"],
        },
        "observationBundleSha256": observation_bundle["bundleSha256"],
        "observationTargets": targets,
        "worker": worker,
    }
    evaluation_sha = canonical_sha256(evaluation_core)
    evaluation_job = {
        **evaluation_core,
        "jobSha256": evaluation_sha,
        "id": f"observationjob_{evaluation_sha[:24]}",
    }
    target, temporary = _prepare_new_directory(Path(output_directory))
    started_at = datetime.now(UTC)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    tracemalloc.start()
    try:
        observation_maps: dict[str, Array] = {}
        observation_roots: dict[str, Array] = {}
        observation_auxiliary_rows: dict[str, list[dict[str, Any]]] = {}
        evaluation, rows = evaluate_observation_targets(
            model,
            observables,
            geometry,
            targets,
            arrays=observation_arrays,
            map_outputs=observation_maps,
            root_outputs=observation_roots,
            auxiliary_rows=observation_auxiliary_rows,
        )
        for target_result, target_specification in zip(
            evaluation["targets"], targets, strict=True
        ):
            target_result["targetSha256"] = canonical_sha256(target_specification)
        if observation_maps:
            evaluation["mapArchive"] = {
                "path": "observation_photon_lensing_maps.npz",
                "maps": _array_records(observation_maps),
            }
        if observation_roots:
            evaluation["rootArchive"] = {
                "path": "observation_multiple_image_roots.npz",
                "arrays": _array_records(observation_roots),
            }
        circular_rows = [row for row in rows if "predicted_speed_m_s" in row]
        velocity_rows = [row for row in rows if "predicted_velocity_m_s" in row]
        multiple_image_rows = [row for row in rows if "assignment_state" in row]
        multiple_image_family_rows = observation_auxiliary_rows.get(
            "multiple_image_families", []
        )
        target_kinds = set(evaluation["targetKinds"])
        _write_json(temporary / "observation_scores.json", evaluation)
        if "circular_speed_curve" in target_kinds:
            _write_observation_predictions(
                temporary / "observation_predictions.csv", circular_rows
            )
        if "line_of_sight_velocity_field" in target_kinds:
            _write_velocity_field_predictions(
                temporary / "observation_velocity_field_predictions.csv", velocity_rows
            )
        if "multiple_image_systems" in target_kinds:
            _write_multiple_image_predictions(
                temporary / "observation_multiple_image_predictions.csv",
                multiple_image_rows,
            )
            _write_multiple_image_families(
                temporary / "observation_multiple_image_families.csv",
                multiple_image_family_rows,
            )
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
        _write_json(temporary / "evaluation_job.json", evaluation_job)
        _write_json(temporary / "field_reference.json", field_reference)
        _write_json(temporary / "observation_bundle.json", observation_bundle)
        scientific_core = {
            "schemaVersion": "sigma-observation-evaluation-result/1",
            "jobId": evaluation_job["id"],
            "jobSha256": evaluation_sha,
            "state": "succeeded",
            "field": evaluation_job["field"],
            "observationBundleSha256": observation_bundle["bundleSha256"],
            "observationEvaluation": evaluation,
            "parameterAccounting": _parameter_accounting(model),
            "evaluationAddedGravityParameters": 0,
            "claimBoundary": [
                "This job reuses an immutable solved field and does not execute a field equation.",
                "Changing observational data or target declarations changes this evaluation identity, not the source field identity.",
                "Massive-tracer adapters and photon-lensing adapters use separately typed observables and channel-specific scores.",
                "Photon distances and sky axes are explicit; this worker does not infer a cosmology.",
                "Raw multiple-image evaluation profiles two source coordinates per family and returns no finite aggregate score when predicted topology is incomplete.",
            ],
        }
        result_sha = canonical_sha256(scientific_core)
        _write_json(
            temporary / "scientific_result.json",
            {**scientific_core, "resultSha256": result_sha},
        )
        _current_memory, peak_memory = tracemalloc.get_traced_memory()
        resource_log = {
            "schemaVersion": "sigma-observation-evaluation-resource-log/1",
            "jobId": evaluation_job["id"],
            "startedAt": started_at.isoformat(),
            "finishedAt": datetime.now(UTC).isoformat(),
            "wallSeconds": time.perf_counter() - wall_started,
            "cpuSeconds": time.process_time() - cpu_started,
            "peakPythonHeapBytes": int(peak_memory),
            "processId": os.getpid(),
            "fieldSolverInvocations": 0,
        }
        _write_json(temporary / "resource_log.json", resource_log)
        artifact_names = [
            "evaluation_job.json",
            "field_reference.json",
            "observation_bundle.json",
            "observation_scores.json",
            "scientific_result.json",
            "resource_log.json",
        ]
        if "circular_speed_curve" in target_kinds:
            artifact_names.append("observation_predictions.csv")
        if "line_of_sight_velocity_field" in target_kinds:
            artifact_names.append("observation_velocity_field_predictions.csv")
        if "multiple_image_systems" in target_kinds:
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
            "schemaVersion": "sigma-observation-evaluation-artifact-index/1",
            "jobId": evaluation_job["id"],
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
            "schemaVersion": "sigma-observation-evaluation-run-manifest/1",
            "state": "succeeded",
            "jobId": evaluation_job["id"],
            "jobSha256": evaluation_sha,
            "scientificResultSha256": result_sha,
            "artifactIndexSha256": file_sha256(temporary / "artifact_index.json"),
            "worker": worker,
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "reproduction": {
                "command": "python scripts/run_observation_evaluation_job.py run --request <request.json>",
                "requiredInputs": [
                    field_result["resultSha256"],
                    field_reference["observableArchiveSha256"],
                    observation_bundle["bundleSha256"],
                ],
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
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        tracemalloc.stop()


def _resolve_within(base: Path, value: str | Path, label: str) -> Path:
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the request directory") from error
    return candidate


def execute_request_file(
    request_path: Path, output_override: Path | None = None
) -> dict[str, Any]:
    path = Path(request_path).resolve()
    envelope = _read_json(path)
    if envelope.get("schemaVersion") != "sigma-observation-evaluation-job-cli/1":
        raise ValueError(
            "CLI envelope schemaVersion must be sigma-observation-evaluation-job-cli/1"
        )
    base = path.parent
    field_path = _resolve_within(
        base, envelope["fieldArtifactPath"], "fieldArtifactPath"
    )
    bundle_path = _resolve_within(
        base, envelope["observationBundlePath"], "observationBundlePath"
    )
    output = (
        _resolve_within(base, output_override, "output")
        if output_override is not None
        else _resolve_within(base, envelope["outputDirectory"], "outputDirectory")
    )
    return execute_observation_evaluation_job(
        field_path,
        bundle_path,
        envelope["request"],
        output,
        expected_field_reference=envelope.get("fieldReference"),
    )
