"""Immutable job wrapper for resolved-galaxy extraction and generation."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
import time
import tracemalloc
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from voidscreen.field_job import (
    array_content_sha256,
    canonical_sha256,
    file_sha256,
    load_array_bundle,
)
from voidscreen.galaxy_maps import resolved_map_morphology
from voidscreen.resolved_galaxy_generator import (
    extract_galaxy_parameters,
    package_content_hash,
    render_galaxy,
    roundtrip_metrics,
    sample_vertical_realization,
)

Array = np.ndarray
ENGINE_ID = "resolved-galaxy-extract-generate-worker"
ENGINE_VERSION = "1.0.0-preview"
KPC_M = 3.085677581491367e19
MSUN_KG = 1.98847e30


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _worker_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("resolved_galaxy_job.py", "resolved_galaxy_generator.py"):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_relative(base: Path, value: str, label: str) -> Path:
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must stay within the job directory") from error
    return candidate


def _controls(value: Any) -> dict[str, int]:
    raw = dict(value or {})
    allowed = {"radialBins", "maximumFourierMode", "residualFeatureCountPerComponent"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown extraction controls: {', '.join(unknown)}")
    controls = {
        "radialBins": int(raw.get("radialBins", 24)),
        "maximumFourierMode": int(raw.get("maximumFourierMode", 4)),
        "residualFeatureCountPerComponent": int(
            raw.get("residualFeatureCountPerComponent", 64)
        ),
    }
    if not 6 <= controls["radialBins"] <= 64:
        raise ValueError("radialBins must be between 6 and 64")
    if not 0 <= controls["maximumFourierMode"] <= 8:
        raise ValueError("maximumFourierMode must be between 0 and 8")
    if not 0 <= controls["residualFeatureCountPerComponent"] <= 256:
        raise ValueError("residualFeatureCountPerComponent must be between 0 and 256")
    return controls


def _vertical(value: Any) -> dict[str, Any]:
    raw = dict(value or {})
    allowed = {"enabled", "realizations", "zCells", "seed"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown vertical controls: {', '.join(unknown)}")
    result = {
        "enabled": bool(raw.get("enabled", True)),
        "realizations": int(raw.get("realizations", 3)),
        "zCells": int(raw.get("zCells", 33)),
        "seed": int(raw.get("seed", 0)),
    }
    if not 1 <= result["realizations"] <= 8:
        raise ValueError("vertical realizations must be between 1 and 8")
    if not 9 <= result["zCells"] <= 129 or result["zCells"] % 2 == 0:
        raise ValueError("zCells must be an odd integer between 9 and 129")
    return result


def _generation_controls(value: Any) -> dict[str, dict[str, Any]]:
    raw = dict(value or {})
    if not set(raw).issubset({"gas", "stars"}):
        raise ValueError("generation controls may contain only gas and stars")
    allowed = {
        "mass_scale",
        "radial_scale",
        "fourier_scale",
        "residual_scale",
        "rotation_deg",
        "center_offset_kpc",
    }
    result: dict[str, dict[str, Any]] = {}
    for component, values in raw.items():
        controls = dict(values)
        unknown = sorted(set(controls) - allowed)
        if unknown:
            raise ValueError(f"unknown {component} generation controls: {', '.join(unknown)}")
        result[component] = controls
    return result


def _bundle_records(bundle: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(record["key"]): record for record in bundle.get("arrays", [])}


def _axis_and_surfaces(bundle_path: Path) -> tuple[dict[str, Any], Array, dict[str, Array]]:
    bundle, arrays = load_array_bundle(bundle_path)
    geometry = bundle.get("geometry", {})
    if geometry.get("coordinateSystem") != "cartesian_2d" or geometry.get("dimensions") != 2:
        raise ValueError("galaxy extraction requires a cartesian_2d array bundle")
    if geometry.get("lengthUnit") != "kpc":
        raise ValueError("galaxy extraction requires geometry lengthUnit kpc")
    records = _bundle_records(bundle)
    required = {"gas_surface_density", "stellar_surface_density"}
    if not required.issubset(arrays):
        raise ValueError("galaxy extraction requires gas_surface_density and stellar_surface_density")
    for key in required:
        if records[key].get("rank") != "scalar" or records[key].get("unit") != "M_sun/kpc^2":
            raise ValueError(f"{key} must be a scalar with unit M_sun/kpc^2")
    gas = np.asarray(arrays["gas_surface_density"], dtype=float)
    stars = np.asarray(arrays["stellar_surface_density"], dtype=float)
    if gas.shape != stars.shape or gas.ndim != 2 or gas.shape[0] != gas.shape[1]:
        raise ValueError("gas and stellar maps must share one square 2D grid")
    if gas.shape[0] > 513:
        raise ValueError("local galaxy jobs are limited to 513 cells per axis")
    spacing_raw = geometry.get("spacing")
    spacing_values = (
        [float(spacing_raw), float(spacing_raw)]
        if not isinstance(spacing_raw, list)
        else [float(item) for item in spacing_raw]
    )
    if len(spacing_values) != 2 or not np.isclose(spacing_values[0], spacing_values[1]):
        raise ValueError("galaxy extraction currently requires equal x/y spacing")
    spacing = spacing_values[0]
    if spacing <= 0.0:
        raise ValueError("grid spacing must be positive")
    axis = (np.arange(gas.shape[0], dtype=float) - 0.5 * (gas.shape[0] - 1)) * spacing
    return bundle, axis, {"gas": gas, "stars": stars, "total": gas + stars}


def _parameter_axis(package: Mapping[str, Any]) -> Array:
    grid = package.get("grid", {})
    cells = int(grid.get("cellsPerAxis", 0))
    if not 9 <= cells <= 513:
        raise ValueError("parameter package grid cellsPerAxis is outside 9..513")
    axis = np.linspace(float(grid["minimumKpc"]), float(grid["maximumKpc"]), cells)
    if not np.isclose(axis[1] - axis[0], float(grid["spacingKpc"]), rtol=1e-8):
        raise ValueError("parameter package grid spacing is inconsistent")
    return axis


def _output_axis(package: Mapping[str, Any], value: Any) -> Array:
    source = _parameter_axis(package)
    if value is None:
        return source
    controls = dict(value)
    if "cellsPerAxis" not in controls or not set(controls).issubset(
        {"cellsPerAxis", "extentScale"}
    ):
        raise ValueError("outputGrid supports cellsPerAxis and extentScale")
    raw_cells = controls["cellsPerAxis"]
    if isinstance(raw_cells, bool) or not isinstance(raw_cells, int):
        raise TypeError("outputGrid.cellsPerAxis must be an integer")
    cells = raw_cells
    if not 9 <= cells <= 513 or cells % 2 == 0:
        raise ValueError("outputGrid.cellsPerAxis must be an odd integer between 9 and 513")
    extent_scale = float(controls.get("extentScale", 1.0))
    if not np.isfinite(extent_scale) or not 1.0 <= extent_scale <= 4.0:
        raise ValueError("outputGrid.extentScale must be finite and between 1 and 4")
    center = 0.5 * float(source[0] + source[-1])
    half_width = 0.5 * float(source[-1] - source[0]) * extent_scale
    return np.linspace(center - half_width, center + half_width, cells)


def _array_bundle(
    arrays: Mapping[str, Array],
    *,
    geometry: Mapping[str, Any],
    unit: str,
    provenance: Mapping[str, Any],
    license_value: Mapping[str, Any],
) -> dict[str, Any]:
    records = []
    for key in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[key], dtype="<f8"))
        records.append(
            {
                "key": key,
                "npzKey": key,
                "unit": unit,
                "rank": "scalar",
                "role": "source",
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "elementCount": int(array.size),
                "contentSha256": array_content_sha256(array),
            }
        )
    core = {
        "schemaVersion": "sigma-array-bundle/1",
        "geometry": dict(geometry),
        "arrays": records,
        "provenance": dict(provenance),
        "license": dict(license_value),
    }
    return {**core, "bundleSha256": canonical_sha256(core)}


def _save_array_product(
    output: Path,
    stem: str,
    arrays: Mapping[str, Array],
    bundle: Mapping[str, Any],
) -> None:
    with (output / f"{stem}.npz").open("wb") as handle:
        np.savez_compressed(
            handle, **{key: np.ascontiguousarray(value, dtype="<f8") for key, value in arrays.items()}
        )
    _write_json(output / f"{stem}_bundle.json", bundle)


def _vertical_products(
    generated: Mapping[str, Array],
    axis: Array,
    controls: Mapping[str, Any],
) -> tuple[dict[str, Array] | None, list[dict[str, Any]], Array | None]:
    if not controls["enabled"]:
        return None, [], None
    radial_resolution = float(axis[1] - axis[0])
    total_morphology = resolved_map_morphology(
        generated["total"], disk_axis_kpc=axis, smoothing_sigma_pixel=2.0
    )
    total_r80 = float(total_morphology["r80_kpc"])
    z_limit = max(8.0 * radial_resolution, 0.8 * total_r80)
    z_axis = np.linspace(-z_limit, z_limit, int(controls["zCells"]))
    first: dict[str, Array] = {}
    metadata: list[dict[str, Any]] = []
    for component_index, component in enumerate(("gas", "stars")):
        morphology = resolved_map_morphology(
            generated[component], disk_axis_kpc=axis, smoothing_sigma_pixel=2.0
        )
        measured_r80 = float(morphology["r80_kpc"])
        # A compact component can collapse into the central pixel when a
        # high-resolution parameter package is intentionally replayed on a
        # coarse commissioning grid. Depth is then unresolved, not zero. Use
        # one radial cell as an explicit resolution floor for the vertical
        # prior and preserve both values in the metadata.
        r80 = max(measured_r80, radial_resolution)
        for realization in range(int(controls["realizations"])):
            sequence = np.random.SeedSequence(
                [int(controls["seed"]), component_index, realization]
            )
            density, description = sample_vertical_realization(
                generated[component],
                axis,
                z_axis,
                r80_kpc=r80,
                component=component,
                rng=np.random.default_rng(sequence),
            )
            dz = float(z_axis[1] - z_axis[0])
            projected = np.sum(density, axis=2) * dz
            error = float(
                np.max(np.abs(projected - generated[component]))
                / max(float(np.max(generated[component])), np.finfo(float).tiny)
            )
            metadata.append(
                {
                    **description,
                    "measuredR80Kpc": measured_r80,
                    "r80ResolutionFloorKpc": radial_resolution,
                    "r80ResolutionFloorApplied": measured_r80 < radial_resolution,
                    "realization": realization,
                    "zCells": len(z_axis),
                    "zLimitKpc": z_limit,
                    "projectionRelativeError": error,
                    "massWeightedZ2Kpc2": float(
                        np.sum(density * z_axis[None, None, :] ** 2) / np.sum(density)
                    ),
                }
            )
            if realization == 0:
                first[component] = density
    first["total"] = first["gas"] + first["stars"]
    return first, metadata, z_axis


def execute_galaxy_request_file(
    request_path: Path, output_override: Path | None = None
) -> dict[str, Any]:
    """Execute one immutable extraction/generation request and publish artifacts."""

    request_path = Path(request_path).resolve()
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    if envelope.get("schemaVersion") != "sigma-galaxy-job-cli/1":
        raise ValueError("request must use sigma-galaxy-job-cli/1")
    operation = envelope.get("operation")
    if operation not in {"extract_roundtrip", "generate"}:
        raise ValueError("operation must be extract_roundtrip or generate")
    base = request_path.parent
    output_value = output_override or Path(str(envelope.get("outputDirectory", "artifacts")))
    output = output_value.resolve() if output_value.is_absolute() else _resolve_relative(
        base, str(output_value), "outputDirectory"
    )
    extraction_controls = _controls(envelope.get("extractionControls"))
    generation_controls = _generation_controls(envelope.get("generationControls"))
    vertical_controls = _vertical(envelope.get("vertical"))
    license_value = dict(envelope.get("outputLicense") or {})
    if not isinstance(license_value.get("id"), str) or not isinstance(
        license_value.get("redistributionAllowed"), bool
    ):
        raise TypeError("outputLicense requires id and redistributionAllowed")
    worker_hash = _worker_source_sha256()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    if output.exists():
        output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    tracemalloc.start()
    started = time.perf_counter()
    try:
        reference: dict[str, Array] | None = None
        source_bundle_hash: str | None = None
        if operation == "extract_roundtrip":
            if envelope.get("outputGrid") is not None:
                raise ValueError("outputGrid is available only for generate jobs")
            bundle_path = _resolve_relative(base, str(envelope["inputBundlePath"]), "inputBundlePath")
            source_bundle, axis, reference = _axis_and_surfaces(bundle_path)
            source_bundle_hash = str(source_bundle["bundleSha256"])
            package = extract_galaxy_parameters(
                str(envelope.get("galaxy", "uploaded-galaxy")),
                axis,
                reference["gas"],
                reference["stars"],
                source_observables=dict(envelope.get("sourceObservables") or {}),
                radial_bins=extraction_controls["radialBins"],
                maximum_fourier_mode=extraction_controls["maximumFourierMode"],
                residual_feature_count=extraction_controls[
                    "residualFeatureCountPerComponent"
                ],
            )
        else:
            package = dict(envelope.get("parameterPackage") or {})
            if package.get("contentSha256") != package_content_hash(package):
                raise ValueError("parameterPackage content hash mismatch")
            axis = _output_axis(package, envelope.get("outputGrid"))

        identity = {
            "schemaVersion": "sigma-galaxy-scientific-job-identity/1",
            "operation": operation,
            "sourceBundleSha256": source_bundle_hash,
            "parameterPackageSha256": package["contentSha256"],
            "extractionControls": extraction_controls,
            "generationControls": generation_controls,
            "vertical": vertical_controls,
            "outputGrid": envelope.get("outputGrid"),
            "outputLicense": license_value,
            "workerSourceSha256": worker_hash,
        }
        job_id = f"galaxyjob_{canonical_sha256(identity)[:24]}"
        generated = render_galaxy(package, axis, component_controls=generation_controls)
        metrics = (
            {
                component: roundtrip_metrics(reference[component], generated[component], axis)
                for component in ("gas", "stars", "total")
            }
            if reference is not None
            else None
        )
        volume, vertical_metadata, z_axis = _vertical_products(
            generated, axis, vertical_controls
        )

        _write_json(temporary / "parameters.json", package)
        if metrics is not None:
            _write_json(temporary / "roundtrip_metrics.json", metrics)
        _write_json(
            temporary / "vertical_priors.json",
            {
                "schemaVersion": "sigma-galaxy-vertical-prior-ensemble/1",
                "status": "assumed_prior_not_measured",
                "items": vertical_metadata,
            },
        )
        spacing = float(axis[1] - axis[0])
        provenance = {
            "kind": "resolved_galaxy_generator_output",
            "scientificJobId": job_id,
            "parameterPackageSha256": package["contentSha256"],
            "operation": operation,
        }
        surface_arrays = {
            "gas_surface_density": generated["gas"],
            "stellar_surface_density": generated["stars"],
            "total_baryonic_surface_density": generated["total"],
        }
        surface_bundle = _array_bundle(
            surface_arrays,
            geometry={
                "coordinateSystem": "cartesian_2d",
                "dimensions": 2,
                "spacing": [spacing, spacing],
                "lengthUnit": "kpc",
                "axisOrder": ["x", "y"],
                "referenceFrame": "intrinsic_face_on_baryonic_map",
            },
            unit="M_sun/kpc^2",
            provenance=provenance,
            license_value=license_value,
        )
        _save_array_product(temporary, "surface_density", surface_arrays, surface_bundle)
        field_surface_arrays = {
            "gas_surface_density": generated["gas"] * MSUN_KG / KPC_M**2,
            "stellar_surface_density": generated["stars"] * MSUN_KG / KPC_M**2,
            "baryon_surface_density": generated["total"] * MSUN_KG / KPC_M**2,
        }
        field_surface_bundle = _array_bundle(
            field_surface_arrays,
            geometry={
                "coordinateSystem": "cartesian_2d",
                "dimensions": 2,
                "spacing": [spacing * KPC_M, spacing * KPC_M],
                "origin": [float(axis[0]) * KPC_M, float(axis[0]) * KPC_M],
                "lengthUnit": "m",
                "axisOrder": ["x", "y"],
                "referenceFrame": "intrinsic_face_on_baryonic_map",
            },
            unit="kg/m^2",
            provenance={**provenance, "unitConversion": "M_sun/kpc^2 to kg/m^2"},
            license_value=license_value,
        )
        _save_array_product(
            temporary, "field_surface_density", field_surface_arrays, field_surface_bundle
        )
        volume_bundle = None
        field_volume_bundle = None
        if volume is not None and z_axis is not None:
            volume_arrays = {
                "gas_volume_density": volume["gas"],
                "stellar_volume_density": volume["stars"],
                "total_baryonic_volume_density": volume["total"],
            }
            volume_bundle = _array_bundle(
                volume_arrays,
                geometry={
                    "coordinateSystem": "cartesian_3d",
                    "dimensions": 3,
                    "spacing": [spacing, spacing, float(z_axis[1] - z_axis[0])],
                    "lengthUnit": "kpc",
                    "axisOrder": ["x", "y", "z"],
                    "referenceFrame": "intrinsic_baryonic_prior_realization",
                },
                unit="M_sun/kpc^3",
                provenance={
                    **provenance,
                    "verticalStatus": "assumed_prior_not_measured",
                    "savedRealization": 0,
                },
                license_value=license_value,
            )
            _save_array_product(temporary, "volume_density", volume_arrays, volume_bundle)
            field_volume_arrays = {
                "gas_density": volume["gas"] * MSUN_KG / KPC_M**3,
                "stellar_density": volume["stars"] * MSUN_KG / KPC_M**3,
                "baryon_density": volume["total"] * MSUN_KG / KPC_M**3,
            }
            field_volume_bundle = _array_bundle(
                field_volume_arrays,
                geometry={
                    "coordinateSystem": "cartesian_3d",
                    "dimensions": 3,
                    "spacing": [
                        spacing * KPC_M,
                        spacing * KPC_M,
                        float(z_axis[1] - z_axis[0]) * KPC_M,
                    ],
                    "origin": [
                        float(axis[0]) * KPC_M,
                        float(axis[0]) * KPC_M,
                        float(z_axis[0]) * KPC_M,
                    ],
                    "lengthUnit": "m",
                    "axisOrder": ["x", "y", "z"],
                    "referenceFrame": "intrinsic_baryonic_prior_realization",
                },
                unit="kg/m^3",
                provenance={
                    **provenance,
                    "verticalStatus": "assumed_prior_not_measured",
                    "savedRealization": 0,
                    "unitConversion": "M_sun/kpc^3 to kg/m^3",
                },
                license_value=license_value,
            )
            _save_array_product(
                temporary, "field_volume_density", field_volume_arrays, field_volume_bundle
            )

        elapsed = time.perf_counter() - started
        _, peak_memory = tracemalloc.get_traced_memory()
        scientific_core = {
            "schemaVersion": "sigma-galaxy-scientific-result/1",
            "state": "succeeded",
            "jobId": job_id,
            "operation": operation,
            "galaxy": package["galaxy"],
            "parameterPackageSha256": package["contentSha256"],
            "sourceBundleSha256": source_bundle_hash,
            "surfaceBundleSha256": surface_bundle["bundleSha256"],
            "fieldSurfaceBundleSha256": field_surface_bundle["bundleSha256"],
            "volumeBundleSha256": None
            if volume_bundle is None
            else volume_bundle["bundleSha256"],
            "fieldVolumeBundleSha256": None
            if field_volume_bundle is None
            else field_volume_bundle["bundleSha256"],
            "roundtripMetrics": metrics,
            "verticalProjectionMaximumRelativeError": max(
                (item["projectionRelativeError"] for item in vertical_metadata), default=None
            ),
            "parameterAccounting": {
                "gravityUniversal": 0,
                "gravityPerObject": 0,
                "baryonicRepresentationValuesAreNotGravityParameters": True,
            },
        }
        scientific_result_sha = canonical_sha256(scientific_core)
        scientific_result = {
            **scientific_core,
            "scientificResultSha256": scientific_result_sha,
        }
        _write_json(temporary / "scientific_result.json", scientific_result)
        resource_log = {
            "schemaVersion": "sigma-galaxy-resource-log/1",
            "elapsedSeconds": elapsed,
            "peakTracedMemoryBytes": int(peak_memory),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gridShape2d": list(generated["total"].shape),
            "gridShape3d": None if volume is None else list(volume["total"].shape),
        }
        _write_json(temporary / "resource_log.json", resource_log)

        artifact_names = sorted(
            path.name for path in temporary.iterdir() if path.is_file()
        )
        artifact_index = {
            "schemaVersion": "sigma-galaxy-artifact-index/1",
            "jobId": job_id,
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
            "schemaVersion": "sigma-galaxy-run-manifest/1",
            "state": "succeeded",
            "jobId": job_id,
            "scientificResultSha256": scientific_result_sha,
            "artifactIndexSha256": file_sha256(temporary / "artifact_index.json"),
            "worker": {
                "engine": ENGINE_ID,
                "version": ENGINE_VERSION,
                "sourceSha256": worker_hash,
            },
            "formulaIndependence": {
                "gravityParameters": 0,
                "velocityTargetsUsedForExtraction": False,
                "theoryNameDispatch": False,
            },
        }
        manifest = {**manifest_core, "manifestSha256": canonical_sha256(manifest_core)}
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        tracemalloc.stop()
