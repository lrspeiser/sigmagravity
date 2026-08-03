"""Immutable job wrapper for resolved-galaxy extraction and generation."""

from __future__ import annotations

import csv
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
ENGINE_VERSION = "1.1.0-preview"
KPC_M = 3.085677581491367e19
MSUN_KG = 1.98847e30
MAX_ENSEMBLE_ARRAY_BYTES = 256 * 1024**2


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
        "axis_ratio_scale",
    }
    result: dict[str, dict[str, Any]] = {}
    for component, values in raw.items():
        controls = dict(values)
        unknown = sorted(set(controls) - allowed)
        if unknown:
            raise ValueError(f"unknown {component} generation controls: {', '.join(unknown)}")
        result[component] = controls
    return result


def _finite_bounded(
    value: Any, default: float, minimum: float, maximum: float, label: str
) -> float:
    result = float(default if value is None else value)
    if not np.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{label} must be finite and between {minimum} and {maximum}")
    return result


def _integer_bounded(value: Any, default: int, minimum: int, maximum: int, label: str) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool) or not isinstance(raw, int) or not minimum <= raw <= maximum:
        raise TypeError(f"{label} must be an integer between {minimum} and {maximum}")
    return raw


def _uncertainty_ensemble(value: Any, package: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value or {})
    allowed = {"enabled", "realizations", "seed", "priors"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown uncertainty ensemble controls: {', '.join(unknown)}")
    priors_raw = dict(raw.get("priors") or {})
    prior_limits = {
        "gas_mass_ln_sigma": (0.0, 1.0),
        "stellar_mass_ln_sigma": (0.0, 1.0),
        "gas_radial_scale_ln_sigma": (0.0, 0.5),
        "stellar_radial_scale_ln_sigma": (0.0, 0.5),
        "angular_structure_ln_sigma": (0.0, 1.0),
        "local_structure_ln_sigma": (0.0, 1.0),
        "center_sigma_kpc": (0.0, 10.0),
        "rotation_sigma_deg": (0.0, 180.0),
        "distance_scale_ln_sigma": (0.0, 0.5),
        "inclination_sigma_deg": (0.0, 20.0),
        "warp_sigma_deg": (0.0, 20.0),
        "co_spatial_unseen_baryon_fraction_max": (0.0, 0.5),
    }
    unknown_priors = sorted(set(priors_raw) - set(prior_limits) - {"reference_inclination_deg"})
    if unknown_priors:
        raise ValueError(f"unknown baryonic uncertainty priors: {', '.join(unknown_priors)}")
    priors = {
        key: _finite_bounded(priors_raw.get(key), 0.0, limits[0], limits[1], key)
        for key, limits in prior_limits.items()
    }
    source_inclination = package.get("sourceObservables", {}).get("inclinationDeg")
    reference_raw = priors_raw.get("reference_inclination_deg", source_inclination)
    reference_inclination = None
    if reference_raw is not None:
        reference_inclination = _finite_bounded(
            reference_raw, 0.0, 0.0, 85.0, "reference_inclination_deg"
        )
    if priors["inclination_sigma_deg"] > 0.0 and reference_inclination is None:
        raise ValueError(
            "inclination_sigma_deg requires reference_inclination_deg or sourceObservables.inclinationDeg"
        )
    priors["reference_inclination_deg"] = reference_inclination
    result = {
        "enabled": bool(raw.get("enabled", False)),
        "realizations": _integer_bounded(
            raw.get("realizations"), 5, 1, 16, "uncertainty ensemble realizations"
        ),
        "seed": _integer_bounded(
            raw.get("seed"), 0, 0, 2**31 - 1, "uncertainty ensemble seed"
        ),
        "priors": priors,
    }
    if not result["enabled"]:
        result["realizations"] = 1
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


def _ensemble_bundle(
    arrays: Mapping[str, Array],
    *,
    spatial_geometry: Mapping[str, Any],
    ensemble_axes: list[dict[str, Any]],
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
                "rank": "scalar_ensemble",
                "shape": list(array.shape),
                "elementCount": int(array.size),
                "contentSha256": array_content_sha256(array),
            }
        )
    core = {
        "schemaVersion": "sigma-galaxy-density-ensemble/1",
        "spatialGeometry": dict(spatial_geometry),
        "ensembleAxes": ensemble_axes,
        "arrays": records,
        "provenance": dict(provenance),
        "license": dict(license_value),
    }
    return {**core, "bundleSha256": canonical_sha256(core)}


def _surface_uncertainty_products(
    package: Mapping[str, Any],
    axis: Array,
    generation_controls: Mapping[str, Mapping[str, Any]],
    controls: Mapping[str, Any],
) -> tuple[dict[str, Array], list[dict[str, Any]]]:
    count = int(controls["realizations"])
    priors = controls["priors"]
    surfaces: dict[str, list[Array]] = {"gas": [], "stars": [], "total": []}
    metadata: list[dict[str, Any]] = []
    reference_inclination = priors["reference_inclination_deg"]
    spacing = float(axis[1] - axis[0])
    for realization in range(count):
        anchor = realization == 0
        rng = np.random.default_rng(
            np.random.SeedSequence([int(controls["seed"]), realization, 811])
        )
        distance_scale = 1.0 if anchor else float(
            np.exp(rng.normal(0.0, priors["distance_scale_ln_sigma"]))
        )
        unseen_fraction = 0.0 if anchor else float(
            rng.uniform(0.0, priors["co_spatial_unseen_baryon_fraction_max"])
        )
        unseen_scale = 1.0 / (1.0 - unseen_fraction)
        if reference_inclination is None:
            inclination = None
            inclination_axis_scale = 1.0
        else:
            inclination = float(reference_inclination) if anchor else float(
                np.clip(
                    rng.normal(reference_inclination, priors["inclination_sigma_deg"]),
                    0.0,
                    85.0,
                )
            )
            inclination_axis_scale = float(
                np.cos(np.radians(reference_inclination))
                / np.cos(np.radians(inclination))
            )
        draw_controls: dict[str, dict[str, Any]] = {}
        component_metadata: dict[str, Any] = {}
        for component in ("gas", "stars"):
            base = dict(generation_controls.get(component, {}))
            mass_sigma = priors[f"{component if component == 'gas' else 'stellar'}_mass_ln_sigma"]
            radial_sigma = priors[
                f"{component if component == 'gas' else 'stellar'}_radial_scale_ln_sigma"
            ]
            mass_draw = 1.0 if anchor else float(np.exp(rng.normal(0.0, mass_sigma)))
            radial_draw = 1.0 if anchor else float(np.exp(rng.normal(0.0, radial_sigma)))
            fourier_draw = 1.0 if anchor else float(
                np.exp(rng.normal(0.0, priors["angular_structure_ln_sigma"]))
            )
            residual_draw = 1.0 if anchor else float(
                np.exp(rng.normal(0.0, priors["local_structure_ln_sigma"]))
            )
            rotation_draw = 0.0 if anchor else float(
                rng.normal(0.0, priors["rotation_sigma_deg"])
            )
            center_draw = (
                np.zeros(2, dtype=float)
                if anchor
                else rng.normal(0.0, priors["center_sigma_kpc"], size=2)
            )
            base_center = np.asarray(base.get("center_offset_kpc", (0.0, 0.0)), dtype=float)
            draw_controls[component] = {
                "mass_scale": float(base.get("mass_scale", 1.0))
                * mass_draw
                * distance_scale**2
                * unseen_scale,
                "radial_scale": float(base.get("radial_scale", 1.0))
                * radial_draw
                * distance_scale,
                "fourier_scale": float(base.get("fourier_scale", 1.0)) * fourier_draw,
                "residual_scale": float(base.get("residual_scale", 1.0)) * residual_draw,
                "rotation_deg": float(base.get("rotation_deg", 0.0)) + rotation_draw,
                "center_offset_kpc": (base_center + center_draw).tolist(),
                "axis_ratio_scale": float(base.get("axis_ratio_scale", 1.0))
                * inclination_axis_scale,
            }
            component_metadata[component] = {
                "massScale": draw_controls[component]["mass_scale"],
                "radialScale": draw_controls[component]["radial_scale"],
                "fourierScale": draw_controls[component]["fourier_scale"],
                "residualScale": draw_controls[component]["residual_scale"],
                "rotationDeg": draw_controls[component]["rotation_deg"],
                "centerOffsetKpc": draw_controls[component]["center_offset_kpc"],
                "axisRatioScale": draw_controls[component]["axis_ratio_scale"],
            }
        rendered = render_galaxy(package, axis, component_controls=draw_controls)
        surface_hashes: dict[str, str] = {}
        masses: dict[str, float] = {}
        morphology: dict[str, dict[str, float]] = {}
        for component in ("gas", "stars", "total"):
            surfaces[component].append(rendered[component])
            surface_hashes[component] = array_content_sha256(rendered[component])
            masses[component] = float(np.sum(rendered[component]) * spacing**2)
            morphology[component] = resolved_map_morphology(
                rendered[component], disk_axis_kpc=axis, smoothing_sigma_pixel=2.0
            )
        metadata.append(
            {
                "realization": realization,
                "anchor": anchor,
                "distanceScale": distance_scale,
                "referenceInclinationDeg": reference_inclination,
                "inclinationDeg": inclination,
                "inclinationDeprojectionAxisRatioScale": inclination_axis_scale,
                "coSpatialUnseenBaryonFraction": unseen_fraction,
                "coSpatialUnseenBaryonAssumption": "unseen mass follows traced baryons proportionally",
                "components": component_metadata,
                "massSolar": masses,
                "morphology": morphology,
                "surfaceContentSha256": surface_hashes,
            }
        )
    return {
        component: np.ascontiguousarray(np.stack(values), dtype="<f8")
        for component, values in surfaces.items()
    }, metadata


def _vertical_ensemble_products(
    surface_ensemble: Mapping[str, Array],
    axis: Array,
    controls: Mapping[str, Any],
    uncertainty_controls: Mapping[str, Any],
) -> tuple[dict[str, Array] | None, list[dict[str, Any]], Array | None]:
    if not controls["enabled"]:
        return None, [], None
    surface_count, cells, _ = surface_ensemble["total"].shape
    vertical_count = int(controls["realizations"])
    z_cells = int(controls["zCells"])
    raw_bytes = surface_count * vertical_count * cells * cells * z_cells * 3 * 8
    if raw_bytes > MAX_ENSEMBLE_ARRAY_BYTES:
        raise ValueError(
            "requested 3D uncertainty ensemble exceeds the 256 MiB raw-array limit"
        )
    radial_resolution = float(axis[1] - axis[0])
    total_r80 = max(
        float(
            resolved_map_morphology(
                surface_ensemble["total"][index],
                disk_axis_kpc=axis,
                smoothing_sigma_pixel=2.0,
            )["r80_kpc"]
        )
        for index in range(surface_count)
    )
    z_limit = max(8.0 * radial_resolution, 0.8 * total_r80)
    z_axis = np.linspace(-z_limit, z_limit, z_cells)
    volumes = {
        component: np.empty(
            (surface_count, vertical_count, cells, cells, z_cells), dtype="<f8"
        )
        for component in ("gas", "stars")
    }
    metadata: list[dict[str, Any]] = []
    warp_sigma = float(uncertainty_controls["priors"]["warp_sigma_deg"])
    for surface_realization in range(surface_count):
        for vertical_realization in range(vertical_count):
            anchor = surface_realization == 0 and vertical_realization == 0
            common_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        int(controls["seed"]),
                        int(uncertainty_controls["seed"]),
                        surface_realization,
                        vertical_realization,
                        977,
                    ]
                )
            )
            warp_amplitude = 0.0 if anchor else float(common_rng.normal(0.0, warp_sigma))
            warp_phase = 0.0 if anchor else float(common_rng.uniform(-180.0, 180.0))
            for component_index, component in enumerate(("gas", "stars")):
                surface = surface_ensemble[component][surface_realization]
                morphology = resolved_map_morphology(
                    surface, disk_axis_kpc=axis, smoothing_sigma_pixel=2.0
                )
                measured_r80 = float(morphology["r80_kpc"])
                r80 = max(measured_r80, radial_resolution)
                sequence = np.random.SeedSequence(
                    [
                        int(controls["seed"]),
                        int(uncertainty_controls["seed"]),
                        surface_realization,
                        vertical_realization,
                        component_index,
                    ]
                )
                density, description = sample_vertical_realization(
                    surface,
                    axis,
                    z_axis,
                    r80_kpc=r80,
                    component=component,
                    rng=np.random.default_rng(sequence),
                    warp_amplitude_deg=warp_amplitude,
                    warp_phase_deg=warp_phase,
                )
                volumes[component][surface_realization, vertical_realization] = density
                dz = float(z_axis[1] - z_axis[0])
                projected = np.sum(density, axis=2) * dz
                error = float(
                    np.max(np.abs(projected - surface))
                    / max(float(np.max(surface)), np.finfo(float).tiny)
                )
                metadata.append(
                    {
                        **description,
                        "surfaceRealization": surface_realization,
                        "verticalRealization": vertical_realization,
                        "anchor": anchor,
                        "measuredR80Kpc": measured_r80,
                        "r80ResolutionFloorKpc": radial_resolution,
                        "r80ResolutionFloorApplied": measured_r80 < radial_resolution,
                        "zCells": len(z_axis),
                        "zLimitKpc": z_limit,
                        "projectionRelativeError": error,
                        "massWeightedZ2Kpc2": float(
                            np.sum(density * z_axis[None, None, :] ** 2)
                            / np.sum(density)
                        ),
                        "volumeContentSha256": array_content_sha256(density),
                    }
                )
    volumes["total"] = volumes["gas"] + volumes["stars"]
    return volumes, metadata, z_axis


def _vertical_products(
    generated: Mapping[str, Array],
    axis: Array,
    controls: Mapping[str, Any],
) -> tuple[dict[str, Array] | None, list[dict[str, Any]], Array | None]:
    stacked = {
        component: np.asarray(generated[component], dtype=float)[None, :, :]
        for component in ("gas", "stars", "total")
    }
    no_uncertainty = {
        "seed": 0,
        "priors": {"warp_sigma_deg": 0.0},
    }
    ensemble, metadata, z_axis = _vertical_ensemble_products(
        stacked, axis, controls, no_uncertainty
    )
    if ensemble is None:
        return None, metadata, z_axis
    first = {component: ensemble[component][0, 0] for component in ensemble}
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
        uncertainty_controls = _uncertainty_ensemble(
            envelope.get("uncertaintyEnsemble"), package
        )

        identity = {
            "schemaVersion": "sigma-galaxy-scientific-job-identity/1",
            "operation": operation,
            "sourceBundleSha256": source_bundle_hash,
            "parameterPackageSha256": package["contentSha256"],
            "extractionControls": extraction_controls,
            "generationControls": generation_controls,
            "vertical": vertical_controls,
            "uncertaintyEnsemble": uncertainty_controls,
            "outputGrid": envelope.get("outputGrid"),
            "outputLicense": license_value,
            "workerSourceSha256": worker_hash,
        }
        job_id = f"galaxyjob_{canonical_sha256(identity)[:24]}"
        generated = render_galaxy(package, axis, component_controls=generation_controls)
        surface_ensemble, uncertainty_metadata = _surface_uncertainty_products(
            package, axis, generation_controls, uncertainty_controls
        )
        for component in ("gas", "stars", "total"):
            if not np.array_equal(surface_ensemble[component][0], generated[component]):
                raise RuntimeError("uncertainty ensemble anchor does not match central generation")
        metrics = (
            {
                component: roundtrip_metrics(reference[component], generated[component], axis)
                for component in ("gas", "stars", "total")
            }
            if reference is not None
            else None
        )
        volume_ensemble, vertical_metadata, z_axis = _vertical_ensemble_products(
            surface_ensemble, axis, vertical_controls, uncertainty_controls
        )
        volume = (
            None
            if volume_ensemble is None
            else {component: volume_ensemble[component][0, 0] for component in volume_ensemble}
        )

        _write_json(temporary / "parameters.json", package)
        if metrics is not None:
            _write_json(temporary / "roundtrip_metrics.json", metrics)
        _write_json(
            temporary / "vertical_priors.json",
            {
                "schemaVersion": "sigma-galaxy-vertical-prior-ensemble/2",
                "status": "assumed_prior_not_measured",
                "surfaceRealizations": int(surface_ensemble["total"].shape[0]),
                "verticalRealizationsPerSurface": int(vertical_controls["realizations"])
                if vertical_controls["enabled"]
                else 0,
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
        surface_ensemble_arrays = {
            "gas_surface_density": surface_ensemble["gas"],
            "stellar_surface_density": surface_ensemble["stars"],
            "total_baryonic_surface_density": surface_ensemble["total"],
        }
        spatial_geometry_2d = {
            "coordinateSystem": "cartesian_2d",
            "dimensions": 2,
            "spacing": [spacing, spacing],
            "origin": [float(axis[0]), float(axis[0])],
            "lengthUnit": "kpc",
            "axisOrder": ["x", "y"],
            "referenceFrame": "intrinsic_face_on_baryonic_map",
        }
        surface_ensemble_bundle = _ensemble_bundle(
            surface_ensemble_arrays,
            spatial_geometry=spatial_geometry_2d,
            ensemble_axes=[
                {
                    "name": "surfaceRealization",
                    "count": int(surface_ensemble["total"].shape[0]),
                    "anchorIndex": 0,
                }
            ],
            unit="M_sun/kpc^2",
            provenance={
                **provenance,
                "uncertaintyStatus": "observation_conditioned_prior_not_posterior",
            },
            license_value=license_value,
        )
        _save_array_product(
            temporary,
            "surface_density_ensemble",
            surface_ensemble_arrays,
            surface_ensemble_bundle,
        )
        surface_quantile_arrays: dict[str, Array] = {}
        for component, key in (
            ("gas", "gas_surface_density"),
            ("stars", "stellar_surface_density"),
            ("total", "total_baryonic_surface_density"),
        ):
            for percentile, quantile in ((16, 0.16), (50, 0.50), (84, 0.84)):
                surface_quantile_arrays[f"{key}_p{percentile}"] = np.quantile(
                    surface_ensemble[component], quantile, axis=0
                )
        surface_quantile_bundle = _array_bundle(
            surface_quantile_arrays,
            geometry=spatial_geometry_2d,
            unit="M_sun/kpc^2",
            provenance={
                **provenance,
                "summaryOf": surface_ensemble_bundle["bundleSha256"],
                "percentiles": [16, 50, 84],
            },
            license_value=license_value,
        )
        _save_array_product(
            temporary,
            "surface_density_quantiles",
            surface_quantile_arrays,
            surface_quantile_bundle,
        )
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
        volume_ensemble_bundle = None
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
            volume_ensemble_arrays = {
                "gas_volume_density": volume_ensemble["gas"],
                "stellar_volume_density": volume_ensemble["stars"],
                "total_baryonic_volume_density": volume_ensemble["total"],
            }
            volume_ensemble_bundle = _ensemble_bundle(
                volume_ensemble_arrays,
                spatial_geometry={
                    "coordinateSystem": "cartesian_3d",
                    "dimensions": 3,
                    "spacing": [spacing, spacing, float(z_axis[1] - z_axis[0])],
                    "origin": [float(axis[0]), float(axis[0]), float(z_axis[0])],
                    "lengthUnit": "kpc",
                    "axisOrder": ["x", "y", "z"],
                    "referenceFrame": "intrinsic_baryonic_prior_realization",
                },
                ensemble_axes=[
                    {
                        "name": "surfaceRealization",
                        "count": int(volume_ensemble["total"].shape[0]),
                        "anchorIndex": 0,
                    },
                    {
                        "name": "verticalRealization",
                        "count": int(volume_ensemble["total"].shape[1]),
                        "anchorIndex": 0,
                    },
                ],
                unit="M_sun/kpc^3",
                provenance={
                    **provenance,
                    "uncertaintyStatus": "observation_conditioned_prior_not_posterior",
                    "projectionTarget": surface_ensemble_bundle["bundleSha256"],
                },
                license_value=license_value,
            )
            _save_array_product(
                temporary,
                "volume_density_ensemble",
                volume_ensemble_arrays,
                volume_ensemble_bundle,
            )

        _write_json(
            temporary / "baryonic_uncertainty_ensemble.json",
            {
                "schemaVersion": "sigma-galaxy-baryonic-uncertainty-ensemble/1",
                "status": "observation_conditioned_prior_not_posterior",
                "gravityParameters": {},
                "velocityTargetsUsed": False,
                "controls": uncertainty_controls,
                "surfaceBundleSha256": surface_ensemble_bundle["bundleSha256"],
                "volumeBundleSha256": None
                if volume_ensemble_bundle is None
                else volume_ensemble_bundle["bundleSha256"],
                "draws": uncertainty_metadata,
                "limitations": [
                    "Prior widths are researcher-declared and are not inferred from a likelihood.",
                    "Inclination uncertainty is represented as a thin-map minor-axis deprojection scale.",
                    "Unseen baryons, when enabled, are assumed to follow traced baryons proportionally.",
                    "Bulge deprojection, dust, beam, PSF, noise, masks, turbulence, and spectral cubes are not inferred here.",
                ],
            },
        )
        with (temporary / "baryonic_uncertainty_draws.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fieldnames = [
                "realization",
                "anchor",
                "distance_scale",
                "inclination_deg",
                "inclination_axis_ratio_scale",
                "co_spatial_unseen_baryon_fraction",
                "gas_mass_solar",
                "stellar_mass_solar",
                "total_mass_solar",
                "total_concentration",
                "total_lopsidedness",
                "total_clumpiness",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for draw in uncertainty_metadata:
                writer.writerow(
                    {
                        "realization": draw["realization"],
                        "anchor": str(draw["anchor"]).lower(),
                        "distance_scale": draw["distanceScale"],
                        "inclination_deg": draw["inclinationDeg"],
                        "inclination_axis_ratio_scale": draw[
                            "inclinationDeprojectionAxisRatioScale"
                        ],
                        "co_spatial_unseen_baryon_fraction": draw[
                            "coSpatialUnseenBaryonFraction"
                        ],
                        "gas_mass_solar": draw["massSolar"]["gas"],
                        "stellar_mass_solar": draw["massSolar"]["stars"],
                        "total_mass_solar": draw["massSolar"]["total"],
                        "total_concentration": draw["morphology"]["total"][
                            "concentration_5log_r80_r20"
                        ],
                        "total_lopsidedness": draw["morphology"]["total"][
                            "lopsidedness_180"
                        ],
                        "total_clumpiness": draw["morphology"]["total"][
                            "clumpiness_positive_highpass"
                        ],
                    }
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
            "surfaceEnsembleBundleSha256": surface_ensemble_bundle["bundleSha256"],
            "surfaceQuantileBundleSha256": surface_quantile_bundle["bundleSha256"],
            "volumeBundleSha256": None
            if volume_bundle is None
            else volume_bundle["bundleSha256"],
            "fieldVolumeBundleSha256": None
            if field_volume_bundle is None
            else field_volume_bundle["bundleSha256"],
            "volumeEnsembleBundleSha256": None
            if volume_ensemble_bundle is None
            else volume_ensemble_bundle["bundleSha256"],
            "roundtripMetrics": metrics,
            "uncertaintyEnsemble": {
                "status": "observation_conditioned_prior_not_posterior",
                "surfaceRealizations": int(surface_ensemble["total"].shape[0]),
                "verticalRealizationsPerSurface": 0
                if volume_ensemble is None
                else int(volume_ensemble["total"].shape[1]),
                "anchorRealization": 0,
                "totalMassSolarP16P50P84": np.quantile(
                    [item["massSolar"]["total"] for item in uncertainty_metadata],
                    [0.16, 0.50, 0.84],
                ).tolist(),
                "totalConcentrationP16P50P84": np.quantile(
                    [
                        item["morphology"]["total"][
                            "concentration_5log_r80_r20"
                        ]
                        for item in uncertainty_metadata
                    ],
                    [0.16, 0.50, 0.84],
                ).tolist(),
            },
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
            "surfaceEnsembleShape": list(surface_ensemble["total"].shape),
            "volumeEnsembleShape": None
            if volume_ensemble is None
            else list(volume_ensemble["total"].shape),
            "ensembleRawArrayBytes": int(
                sum(array.nbytes for array in surface_ensemble.values())
                + (
                    0
                    if volume_ensemble is None
                    else sum(array.nbytes for array in volume_ensemble.values())
                )
            ),
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
