"""Immutable jobs for inverse baryon-to-effective-response discovery.

The target of this job is explicitly model-derived.  It may be used to create
candidate forward laws, but it may not enter the later held-out observation
test that evaluates those laws.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import platform
import re
import shutil
import tempfile
import time
import tracemalloc
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from .field_job import (
    _write_deterministic_npz,
    canonical_sha256,
    file_sha256,
    load_array_bundle,
)
from .inverse_response import InverseResponseAnalysis, analyze_stationary_response

Array = np.ndarray
ENGINE_ID = "inverse-stationary-response-worker"
ENGINE_VERSION = "1.0.0-preview"
SYSTEM_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _resolve_relative(base: Path, value: str, label: str) -> Path:
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the request directory") from error
    return candidate


def _worker_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("inverse_response.py", "inverse_response_job.py", "field_job.py"):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _records(bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["key"]): dict(record) for record in bundle.get("arrays", [])}


def _require_array_role(
    records: Mapping[str, Mapping[str, Any]],
    key: str,
    scientific_role: str,
    operational_role: str,
) -> Mapping[str, Any]:
    if key not in records:
        raise ValueError(f"input bundle is missing array {key}")
    record = records[key]
    if record.get("rank") != "scalar":
        raise ValueError(f"array {key} must declare rank=scalar")
    if record.get("scientificRole") != scientific_role:
        raise ValueError(
            f"array {key} must declare scientificRole={scientific_role}"
        )
    if record.get("role") != operational_role:
        raise ValueError(f"array {key} must declare role={operational_role}")
    return record


def _normalized_controls(request: Mapping[str, Any], dimensions: int) -> dict[str, Any]:
    kernel = request.get("kernel")
    if not isinstance(kernel, Mapping):
        raise TypeError("request requires a kernel object")
    shape = kernel.get("shape")
    if not isinstance(shape, list) or len(shape) != dimensions:
        raise ValueError("kernel.shape requires one odd integer per map dimension")
    shape_values = []
    for value in shape:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("kernel.shape values must be integers")
        shape_values.append(value)
    ridge = float(kernel.get("ridge", 1.0e-8))
    smoothness = float(kernel.get("smoothness", 1.0e-4))
    uncertainty = request.get("uncertainty", {})
    nulls = request.get("nullControls", {})
    ensemble_size = uncertainty.get("ensembleSize", 32)
    null_count = nulls.get("count", 19)
    if isinstance(ensemble_size, bool) or not isinstance(ensemble_size, int):
        raise TypeError("uncertainty.ensembleSize must be an integer")
    if not 20 <= ensemble_size <= 512:
        raise ValueError("uncertainty.ensembleSize must lie from 20 to 512")
    if isinstance(null_count, bool) or not isinstance(null_count, int):
        raise TypeError("nullControls.count must be an integer")
    if not 19 <= null_count <= 999:
        raise ValueError("nullControls.count must lie from 19 to 999")
    if nulls.get("kind", "source_radial_angle_shuffle") != "source_radial_angle_shuffle":
        raise ValueError(
            "nullControls.kind must be source_radial_angle_shuffle in v1"
        )
    multipliers = kernel.get("regularizationMultipliers", [0.1, 1.0, 10.0])
    if not isinstance(multipliers, list) or not multipliers:
        raise TypeError("kernel.regularizationMultipliers must be a non-empty array")
    nonnegative = kernel.get("nonnegative", True)
    if not isinstance(nonnegative, bool):
        raise TypeError("kernel.nonnegative must be a boolean")
    return {
        "kernelShape": shape_values,
        "ridge": ridge,
        "smoothness": smoothness,
        "nonnegative": nonnegative,
        "regularizationMultipliers": [float(value) for value in multipliers],
        "ensembleSize": ensemble_size,
        "ensembleSeed": int(uncertainty.get("seed", 0)),
        "nullCount": null_count,
        "nullSeed": int(nulls.get("seed", 1)),
        "nullKind": "source_radial_angle_shuffle",
    }


def _system_inputs(
    request: Mapping[str, Any],
    bundle: Mapping[str, Any],
    arrays: Mapping[str, Array],
    dimensions: int,
) -> tuple[
    list[str],
    list[Array],
    list[Array],
    list[Array],
    list[Array],
    list[dict[str, Any]],
]:
    systems = request.get("systems")
    if not isinstance(systems, list) or not systems:
        raise TypeError("request requires a non-empty systems array")
    records = _records(bundle)
    identifiers: list[str] = []
    sources: list[Array] = []
    targets: list[Array] = []
    uncertainties: list[Array] = []
    masks: list[Array] = []
    audits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for system in systems:
        if not isinstance(system, Mapping):
            raise TypeError("each system must be an object")
        identifier = str(system.get("id", ""))
        if not SYSTEM_ID.fullmatch(identifier) or identifier in seen:
            raise ValueError(
                "system ids must be unique and use 1-64 letters, numbers, dots, underscores, or hyphens"
            )
        seen.add(identifier)
        source_key = str(system.get("sourceKey", ""))
        target_key = str(system.get("targetKey", ""))
        uncertainty_key = str(system.get("uncertaintyKey", ""))
        source_record = _require_array_role(
            records, source_key, "baryonic_input", "source"
        )
        target_record = _require_array_role(
            records, target_key, "model_derived_discovery_target", "auxiliary"
        )
        uncertainty_record = _require_array_role(
            records, uncertainty_key, "nuisance_or_calibration", "uncertainty"
        )
        if target_record.get("unit") != source_record.get("unit"):
            raise ValueError(
                f"system {identifier} target and baryonic source units must match"
            )
        if uncertainty_record.get("unit") != target_record.get("unit"):
            raise ValueError(
                f"system {identifier} target uncertainty unit must match the target"
            )
        source = np.asarray(arrays[source_key], dtype=float)
        target = np.asarray(arrays[target_key], dtype=float)
        uncertainty = np.asarray(arrays[uncertainty_key], dtype=float)
        if source.ndim != dimensions:
            raise ValueError(
                f"system {identifier} arrays must have {dimensions} spatial dimensions"
            )
        if np.any(source < 0.0) or float(np.sum(source)) <= 0.0:
            raise ValueError(
                f"system {identifier} baryonic source must be non-negative with positive total"
            )
        mask_key = system.get("maskKey")
        if mask_key is None:
            mask = np.ones(source.shape, dtype=bool)
        else:
            mask_name = str(mask_key)
            mask_record = _require_array_role(
                records, mask_name, "nuisance_or_calibration", "mask"
            )
            if mask_record.get("unit") not in {"1", "dimensionless"}:
                raise ValueError(f"system {identifier} mask must be dimensionless")
            mask = np.asarray(arrays[mask_name], dtype=float) > 0.5
        identifiers.append(identifier)
        sources.append(source)
        targets.append(target)
        uncertainties.append(uncertainty)
        masks.append(mask)
        audits.append(
            {
                "systemId": identifier,
                "sourceKey": source_key,
                "sourceRole": source_record["scientificRole"],
                "targetKey": target_key,
                "targetRole": target_record["scientificRole"],
                "uncertaintyKey": uncertainty_key,
                "uncertaintyRole": uncertainty_record["scientificRole"],
                "maskKey": mask_key,
                "heldOutRawObservationsUsed": False,
            }
        )
    return identifiers, sources, targets, uncertainties, masks, audits


def _analysis_summary(
    analysis: InverseResponseAnalysis,
    identifiers: list[str],
    controls: Mapping[str, Any],
    spacing: list[float],
    data_role_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schemaVersion": "sigma-inverse-response-result/1",
        "state": "succeeded",
        "method": (
            "stationary_nonnegative_compact_convolution"
            if controls["nonnegative"]
            else "stationary_signed_compact_convolution"
        ),
        "kernelConvention": {
            "boundary": "zero_padded",
            "mode": "linear_same",
            "kernelOrigin": "centered_sample",
            "measure": "physical_volume",
            "spacing": spacing,
            "automaticKernelNormalization": False,
            "normalization": "integral" if controls["nonnegative"] else "l1_integral",
            "reportedNormalizedKernelIntegral": float(
                np.sum(analysis.fit.normalized_kernel) * np.prod(spacing)
            ),
            "reportedNormalizedKernelL1Integral": float(
                np.sum(np.abs(analysis.fit.normalized_kernel)) * np.prod(spacing)
            ),
        },
        "systems": [
            {"id": identifier, **metrics}
            for identifier, metrics in zip(
                identifiers, analysis.fit.system_metrics, strict=True
            )
        ],
        "aggregateMetrics": analysis.fit.aggregate_metrics,
        "amplitude": analysis.fit.amplitude,
        "amplitudeInterval": analysis.amplitude_interval,
        "optimizer": analysis.fit.optimizer,
        "identifiability": analysis.fit.identifiability,
        "nonIdentifiability": analysis.non_identifiability,
        "nullSummary": analysis.null_summary,
        "regularizationSensitivity": list(analysis.regularization_sensitivity),
        "uncertainty": {
            "method": "parametric_target_perturbation",
            "ensembleSize": controls["ensembleSize"],
            "seed": controls["ensembleSeed"],
        },
        "dataRoleAudit": data_role_audit,
        "parameterAccounting": {
            "fittedDiscoveryKernelCells": int(
                analysis.fit.identifiability["kernel_cells"]
            ),
            "fittedUniversalResponseAmplitudes": 1,
            "fittedPerSystemGravityParameters": 0,
            "classification": "hypothesis_generator_not_forward_theory_fit",
        },
        "claimBoundary": [
            "The target maps are model-derived discovery products, not direct dark-matter observations.",
            "The recovered kernel is a response family compatible with submitted assumptions, not a measured path taken by gravity.",
            "A target-derived kernel cannot validate itself; it must be frozen before predicting withheld raw observations.",
            "Rank, regularization sensitivity, and radial-angle null results must remain attached to any candidate formula.",
            "The absolute response amplitude is fitted jointly across systems and is not derived from first principles here.",
        ],
    }


def _write_kernel_csv(path: Path, analysis: InverseResponseAnalysis, spacing: list[float]) -> None:
    shape = analysis.fit.raw_kernel.shape
    center = tuple(size // 2 for size in shape)
    columns = [
        "flat_index",
        *[f"offset_axis{axis}" for axis in range(len(shape))],
        "raw_response",
        "normalized",
        "lower_2_5",
        "median",
        "upper_97_5",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for flat_index, index in enumerate(np.ndindex(shape)):
            writer.writerow(
                {
                    "flat_index": flat_index,
                    **{
                        f"offset_axis{axis}": format(
                            (index[axis] - center[axis]) * spacing[axis], ".17g"
                        )
                        for axis in range(len(shape))
                    },
                    "raw_response": format(analysis.fit.raw_kernel[index], ".17g"),
                    "normalized": format(
                        analysis.fit.normalized_kernel[index], ".17g"
                    ),
                    "lower_2_5": format(analysis.kernel_lower[index], ".17g"),
                    "median": format(analysis.kernel_median[index], ".17g"),
                    "upper_97_5": format(analysis.kernel_upper[index], ".17g"),
                }
            )


def _write_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _kernel_svg(kernel: Array) -> str:
    values = np.asarray(kernel, dtype=float)
    if values.ndim == 3:
        values = values[:, :, values.shape[2] // 2]
    maximum = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    cells = []
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            fraction = float(np.clip(values[row, column] / maximum, -1.0, 1.0))
            strength = abs(fraction)
            if fraction >= 0.0:
                red = int(245 - 160 * strength)
                green = int(248 - 100 * strength)
                blue = int(250 - 25 * strength)
            else:
                red = int(245 + 10 * strength)
                green = int(248 - 150 * strength)
                blue = int(250 - 155 * strength)
            cells.append(
                f'<rect x="{column * 18}" y="{row * 18}" width="18" height="18" '
                f'fill="rgb({red},{green},{blue})"><title>{values[row, column]:.6g}</title></rect>'
            )
    width = values.shape[1] * 18
    height = values.shape[0] * 18
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(cells)}</svg>'


def _write_html_report(path: Path, result: Mapping[str, Any], analysis: InverseResponseAnalysis) -> None:
    systems = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['id']))}</td>"
        f"<td>{row['rmse']:.6g}</td>"
        f"<td>{row['weighted_rmse']:.4f}</td>"
        f"<td>{row['r_squared']:.4f}</td>"
        "</tr>"
        for row in result["systems"]
    )
    limitations = "".join(
        f"<li>{html.escape(str(value))}</li>" for value in result["claimBoundary"]
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Inverse response report</title>
<style>body{{font:16px system-ui;max-width:960px;margin:40px auto;padding:0 20px;color:#18202a}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ccd4dd;text-align:left}}
.metric{{display:inline-block;padding:12px 18px;margin:6px;background:#eef3f7;border-radius:8px}}svg{{max-width:420px;border:1px solid #ccd4dd}}</style></head>
<body><h1>Inverse baryon-to-response discovery report</h1>
<p><strong>Classification:</strong> hypothesis generator, not a forward theory test.</p>
<div class="metric">Amplitude: {result['amplitude']:.6g}</div>
<div class="metric">Weighted RMSE: {result['aggregateMetrics']['weighted_rmse']:.4f}</div>
<div class="metric">Null p: {result['nullSummary']['permutation_p_value']:.4f}</div>
<div class="metric">Non-identifiable: {str(result['nonIdentifiability']['non_identifiable']).lower()}</div>
<h2>Normalized kernel</h2>{_kernel_svg(analysis.fit.normalized_kernel)}
<h2>Per-system reconstruction</h2><table><thead><tr><th>System</th><th>RMSE</th><th>Weighted RMSE</th><th>R²</th></tr></thead><tbody>{systems}</tbody></table>
<h2>What this result cannot establish</h2><ul>{limitations}</ul>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def _write_llm_briefing(path: Path, result: Mapping[str, Any]) -> None:
    path.write_text(
        "# Deterministic inverse-response briefing\n\n"
        "This file summarizes an already-computed result. An LLM must not change scores, "
        "exclude systems, refit the kernel, or convert this discovery result into a theory claim.\n\n"
        f"- Systems: {result['aggregateMetrics']['systems']}\n"
        f"- Fitted universal amplitude: {result['amplitude']:.12g}\n"
        f"- Weighted RMSE: {result['aggregateMetrics']['weighted_rmse']:.12g}\n"
        f"- Radial-angle null p-value: {result['nullSummary']['permutation_p_value']:.12g}\n"
        f"- Signal against declared null: {result['nullSummary']['signal_against_null']}\n"
        f"- Non-identifiable: {result['nonIdentifiability']['non_identifiable']}\n\n"
        "The target maps were model-derived discovery products. The scientific next step is "
        "to freeze a candidate law and predict raw held-out observations without these targets.\n",
        encoding="utf-8",
    )


def execute_inverse_response_request_file(
    request_path: Path,
    output_override: Path | None = None,
) -> dict[str, Any]:
    """Execute one content-addressed inverse-response discovery request."""

    request_path = Path(request_path).resolve()
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    if envelope.get("schemaVersion") != "sigma-inverse-response-job-cli/1":
        raise ValueError("request must use sigma-inverse-response-job-cli/1")
    base = request_path.parent
    bundle_path = _resolve_relative(
        base, str(envelope.get("inputBundlePath", "")), "inputBundlePath"
    )
    output_value = output_override or Path(str(envelope.get("outputDirectory", "artifacts")))
    output = (
        output_value.resolve()
        if output_value.is_absolute()
        else _resolve_relative(base, str(output_value), "outputDirectory")
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    if output.exists():
        output.rmdir()
    bundle, arrays = load_array_bundle(bundle_path)
    geometry = bundle.get("geometry", {})
    dimensions = int(geometry.get("dimensions", 0))
    expected_coordinate = "cartesian_2d" if dimensions == 2 else "cartesian_3d"
    if dimensions not in {2, 3} or geometry.get("coordinateSystem") != expected_coordinate:
        raise ValueError("inverse response requires Cartesian 2D or 3D bundle geometry")
    raw_spacing = geometry.get("spacing")
    spacing = (
        [float(raw_spacing)] * dimensions
        if isinstance(raw_spacing, (int, float))
        else [float(value) for value in raw_spacing]
    )
    if len(spacing) != dimensions or any(value <= 0.0 for value in spacing):
        raise ValueError("bundle spacing requires one positive value per dimension")
    controls = _normalized_controls(envelope, dimensions)
    identifiers, sources, targets, uncertainties, masks, data_role_audit = _system_inputs(
        envelope, bundle, arrays, dimensions
    )
    output_license = envelope.get("outputLicense")
    if not isinstance(output_license, Mapping) or not isinstance(
        output_license.get("redistributionAllowed"), bool
    ):
        raise TypeError("outputLicense requires id and redistributionAllowed")
    if not isinstance(output_license.get("id"), str) or not output_license["id"]:
        raise TypeError("outputLicense.id must be a non-empty string")

    job_core = {
        "schemaVersion": "sigma-inverse-response-job/1",
        "inputBundleSha256": bundle["bundleSha256"],
        "systems": envelope["systems"],
        "geometry": {
            "coordinateSystem": expected_coordinate,
            "dimensions": dimensions,
            "spacing": spacing,
            "lengthUnit": geometry.get("lengthUnit"),
        },
        "controls": controls,
        "outputLicense": dict(output_license),
        "worker": {
            "engine": ENGINE_ID,
            "version": ENGINE_VERSION,
            "sourceSha256": _worker_source_sha256(),
        },
    }
    job_sha = canonical_sha256(job_core)
    job = {**job_core, "jobSha256": job_sha, "id": f"inversejob_{job_sha[:24]}"}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    started_at = datetime.now(UTC)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    tracemalloc.start()
    try:
        analysis = analyze_stationary_response(
            sources,
            targets,
            spacing,
            controls["kernelShape"],
            uncertainties=uncertainties,
            masks=masks,
            ridge=controls["ridge"],
            smoothness=controls["smoothness"],
            nonnegative=controls["nonnegative"],
            ensemble_size=controls["ensembleSize"],
            ensemble_seed=controls["ensembleSeed"],
            null_count=controls["nullCount"],
            null_seed=controls["nullSeed"],
            regularization_multipliers=controls["regularizationMultipliers"],
        )
        result_core = {
            **_analysis_summary(
                analysis, identifiers, controls, spacing, data_role_audit
            ),
            "jobId": job["id"],
        }
        result_sha = canonical_sha256(result_core)
        result = {**result_core, "resultSha256": result_sha}
        _write_json(temporary / "input_bundle.json", bundle)
        _write_json(temporary / "request.json", envelope)
        _write_json(temporary / "job.json", job)
        _write_json(temporary / "scientific_result.json", result)
        _write_deterministic_npz(
            temporary / "kernels.npz",
            {
                "raw_response": analysis.fit.raw_kernel,
                "normalized": analysis.fit.normalized_kernel,
                "lower_2_5": analysis.kernel_lower,
                "median": analysis.kernel_median,
                "upper_97_5": analysis.kernel_upper,
            },
        )
        prediction_arrays: dict[str, Array] = {}
        for identifier, prediction, residual in zip(
            identifiers,
            analysis.fit.predictions,
            analysis.fit.residuals,
            strict=True,
        ):
            prediction_arrays[f"prediction__{identifier}"] = prediction
            prediction_arrays[f"residual__{identifier}"] = residual
        _write_deterministic_npz(
            temporary / "system_predictions.npz", prediction_arrays
        )
        _write_kernel_csv(temporary / "kernel.csv", analysis, spacing)
        _write_rows(
            temporary / "per_system.csv",
            [dict(row) for row in result["systems"]],
        )
        _write_rows(
            temporary / "null_controls.csv",
            [dict(row) for row in analysis.null_controls],
        )
        _write_rows(
            temporary / "regularization_sensitivity.csv",
            [dict(row) for row in analysis.regularization_sensitivity],
        )
        _write_html_report(temporary / "report.html", result, analysis)
        _write_llm_briefing(temporary / "llm_briefing.md", result)
        (temporary / "reproduction.txt").write_text(
            "python scripts/run_inverse_response_job.py run --request <request.json>\n",
            encoding="utf-8",
        )
        _current_memory, peak_memory = tracemalloc.get_traced_memory()
        resource_log = {
            "schemaVersion": "sigma-inverse-response-resource-log/1",
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
            "input_bundle.json",
            "request.json",
            "job.json",
            "scientific_result.json",
            "kernels.npz",
            "system_predictions.npz",
            "kernel.csv",
            "per_system.csv",
            "null_controls.csv",
            "regularization_sensitivity.csv",
            "report.html",
            "llm_briefing.md",
            "reproduction.txt",
            "resource_log.json",
        ]
        artifact_index = {
            "schemaVersion": "sigma-inverse-response-artifact-index/1",
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
            "schemaVersion": "sigma-inverse-response-run-manifest/1",
            "state": "succeeded",
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
                "command": "python scripts/run_inverse_response_job.py run --request <request.json>",
                "requiredInputs": [bundle["bundleSha256"]],
            },
        }
        manifest = {
            **manifest_core,
            "manifestSha256": canonical_sha256(manifest_core),
            "createdAt": datetime.now(UTC).isoformat(),
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(output)
        return manifest
    except Exception as error:  # noqa: BLE001 - numerical failures become artifacts
        try:
            failure_core = {
                "schemaVersion": "sigma-inverse-response-failure/1",
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
            _write_json(temporary / "job.json", job)
            _write_json(temporary / "failure.json", failure)
            artifact_names = ["job.json", "failure.json"]
            artifact_index = {
                "schemaVersion": "sigma-inverse-response-artifact-index/1",
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
                "schemaVersion": "sigma-inverse-response-run-manifest/1",
                "state": "failed",
                "jobId": job["id"],
                "jobSha256": job_sha,
                "failureSha256": failure["failureSha256"],
                "artifactIndexSha256": file_sha256(
                    temporary / "artifact_index.json"
                ),
                "worker": job["worker"],
            }
            manifest = {
                **manifest_core,
                "manifestSha256": canonical_sha256(manifest_core),
                "createdAt": datetime.now(UTC).isoformat(),
            }
            _write_json(temporary / "manifest.json", manifest)
            temporary.rename(output)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return manifest
    finally:
        tracemalloc.stop()
