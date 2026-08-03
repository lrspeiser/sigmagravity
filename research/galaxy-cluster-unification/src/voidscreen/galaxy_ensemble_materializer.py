"""Verify and materialize one resolved-galaxy density-ensemble realization.

The resolved-galaxy generator stores compact ensembles in astronomical units.
This module selects one named realization, verifies the complete parent archive,
converts the selected source fields to SI, and writes the ordinary array-bundle
contract consumed by every generic field solver.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .field_job import (
    _canonical_array,
    _write_deterministic_npz,
    array_content_sha256,
    canonical_sha256,
)

Array = NDArray[np.float64]

KPC_M = 3.085677581491367e19
MSUN_KG = 1.98847e30
MATERIALIZER_VERSION = "sigma-galaxy-ensemble-materializer/1"

_KINDS = {
    "surface_density_ensemble": {
        "dimensions": 2,
        "unit": "M_sun/kpc^2",
        "factor": MSUN_KG / KPC_M**2,
        "inputKeys": {
            "gas_surface_density": "gas_surface_density",
            "stellar_surface_density": "stellar_surface_density",
            "total_baryonic_surface_density": "baryon_surface_density",
        },
    },
    "volume_density_ensemble": {
        "dimensions": 3,
        "unit": "M_sun/kpc^3",
        "factor": MSUN_KG / KPC_M**3,
        "inputKeys": {
            "gas_volume_density": "gas_density",
            "stellar_volume_density": "stellar_density",
            "total_baryonic_volume_density": "baryon_density",
        },
    },
}


def _read_verified_ensemble(
    bundle_path: Path,
    archive_path: Path,
    artifact: str,
) -> tuple[dict[str, Any], dict[str, Array]]:
    specification = _KINDS.get(artifact)
    if specification is None:
        raise ValueError(f"unsupported galaxy ensemble artifact: {artifact}")
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    if bundle.get("schemaVersion") != "sigma-galaxy-density-ensemble/1":
        raise ValueError("ensemble bundle must use sigma-galaxy-density-ensemble/1")
    claimed = bundle.get("bundleSha256")
    core = {key: value for key, value in bundle.items() if key != "bundleSha256"}
    if claimed != canonical_sha256(core):
        raise ValueError("ensemble bundle manifest hash mismatch")
    geometry = bundle.get("spatialGeometry")
    if not isinstance(geometry, dict):
        raise TypeError("ensemble bundle requires spatialGeometry")
    if geometry.get("dimensions") != specification["dimensions"]:
        raise ValueError("ensemble artifact and spatial geometry dimensions disagree")
    if geometry.get("lengthUnit") != "kpc":
        raise ValueError("galaxy ensemble spatial geometry must use kpc")
    axes = bundle.get("ensembleAxes")
    expected_axes = ["surfaceRealization"]
    if specification["dimensions"] == 3:
        expected_axes.append("verticalRealization")
    if not isinstance(axes, list) or [axis.get("name") for axis in axes] != expected_axes:
        raise ValueError("ensemble axes do not match the selected artifact")
    counts = []
    for axis in axes:
        count = axis.get("count")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"ensemble axis {axis.get('name')} has an invalid count")
        counts.append(count)
    records = bundle.get("arrays")
    if not isinstance(records, list) or not records:
        raise ValueError("ensemble bundle requires array records")
    by_key = {record.get("key"): record for record in records}
    if len(by_key) != len(records):
        raise ValueError("ensemble array keys must be unique")
    if set(by_key) != set(specification["inputKeys"]):
        raise ValueError("ensemble arrays do not match the selected density artifact")
    arrays: dict[str, Array] = {}
    spatial_shape: tuple[int, ...] | None = None
    with np.load(Path(archive_path), allow_pickle=False) as archive:
        expected_npz = {str(record.get("npzKey")) for record in records}
        if set(archive.files) != expected_npz:
            raise ValueError("ensemble NPZ keys do not match its manifest")
        for public_key, record in sorted(by_key.items()):
            if record.get("rank") != "scalar_ensemble":
                raise ValueError(f"ensemble array {public_key} must have scalar_ensemble rank")
            if record.get("unit") != specification["unit"]:
                raise ValueError(f"ensemble array {public_key} has an incompatible unit")
            values = _canonical_array(archive[str(record["npzKey"])])
            expected_shape = counts + list(values.shape[len(counts) :])
            if list(values.shape) != record.get("shape") or list(values.shape) != expected_shape:
                raise ValueError(f"ensemble array {public_key} shape does not match its manifest")
            if values.ndim != len(counts) + specification["dimensions"]:
                raise ValueError(f"ensemble array {public_key} has the wrong spatial rank")
            current_spatial_shape = tuple(values.shape[len(counts) :])
            if spatial_shape is None:
                spatial_shape = current_spatial_shape
            elif current_spatial_shape != spatial_shape:
                raise ValueError("ensemble density arrays must share one spatial shape")
            if values.dtype.str != "<f8" or record.get("dtype", "<f8") != "<f8":
                raise ValueError(f"ensemble array {public_key} must use canonical float64 values")
            if int(values.size) != record.get("elementCount"):
                raise ValueError(f"ensemble array {public_key} element count mismatch")
            if array_content_sha256(values) != record.get("contentSha256"):
                raise ValueError(f"ensemble array {public_key} content hash mismatch")
            arrays[str(public_key)] = values
    return bundle, arrays


def _selection_indices(bundle: Mapping[str, Any], selection: Mapping[str, Any]) -> tuple[int, ...]:
    if not isinstance(selection, Mapping):
        raise TypeError("selection must be an object")
    allowed = {"surfaceRealization", "verticalRealization"}
    unknown = sorted(set(selection) - allowed)
    if unknown:
        raise ValueError(f"unknown realization selection keys: {', '.join(unknown)}")
    indices = []
    for axis in bundle["ensembleAxes"]:
        name = str(axis["name"])
        value = selection.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"selection.{name} must be an integer")
        if value < 0 or value >= int(axis["count"]):
            raise ValueError(f"selection.{name} is outside [0, {int(axis['count']) - 1}]")
        indices.append(value)
    if "verticalRealization" in selection and len(indices) == 1:
        raise ValueError("surface ensembles do not have a verticalRealization axis")
    return tuple(indices)


def materialize_galaxy_ensemble_realization(
    *,
    bundle_path: Path,
    archive_path: Path,
    artifact: str,
    selection: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    """Write one verified realization as a standard SI array bundle."""

    specification = _KINDS.get(artifact)
    if specification is None:
        raise ValueError(f"unsupported galaxy ensemble artifact: {artifact}")
    bundle, arrays = _read_verified_ensemble(bundle_path, archive_path, artifact)
    indices = _selection_indices(bundle, selection)
    selected: dict[str, Array] = {}
    records = []
    factor = float(specification["factor"])
    output_unit = "kg/m^2" if specification["dimensions"] == 2 else "kg/m^3"
    for source_key, output_key in specification["inputKeys"].items():
        values = _canonical_array(arrays[source_key][indices] * factor)
        selected[str(output_key)] = values
        records.append(
            {
                "key": output_key,
                "npzKey": output_key,
                "unit": output_unit,
                "rank": "scalar",
                "role": "source",
                "dtype": values.dtype.str,
                "shape": list(values.shape),
                "elementCount": int(values.size),
                "contentSha256": array_content_sha256(values),
            }
        )
    geometry = dict(bundle["spatialGeometry"])
    geometry["spacing"] = [float(value) * KPC_M for value in geometry["spacing"]]
    if "origin" in geometry:
        geometry["origin"] = [float(value) * KPC_M for value in geometry["origin"]]
    geometry["lengthUnit"] = "m"
    normalized_selection = {
        str(axis["name"]): indices[position]
        for position, axis in enumerate(bundle["ensembleAxes"])
    }
    core = {
        "schemaVersion": "sigma-array-bundle/1",
        "geometry": geometry,
        "arrays": sorted(records, key=lambda record: str(record["key"])),
        "provenance": {
            **dict(bundle["provenance"]),
            "kind": "galaxy_density_ensemble_realization",
            "uncertaintyStatus": "observation_conditioned_prior_not_posterior",
            "parentEnsembleBundleSha256": bundle["bundleSha256"],
            "ensembleArtifact": artifact,
            "realizationSelection": normalized_selection,
            "unitConversion": f"{specification['unit']} to {output_unit}",
            "materializerVersion": MATERIALIZER_VERSION,
        },
        "license": dict(bundle["license"]),
    }
    output_bundle = {**core, "bundleSha256": canonical_sha256(core)}
    target = Path(output_directory).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        _write_deterministic_npz(temporary / "arrays.npz", selected)
        (temporary / "bundle.json").write_text(
            json.dumps(output_bundle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            shutil.rmtree(temporary)
        else:
            try:
                temporary.replace(target)
            except FileExistsError:
                if not target.exists():
                    raise
                shutil.rmtree(temporary)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_bundle


def execute_materialization_request(request_path: Path) -> dict[str, Any]:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if request.get("schemaVersion") != "sigma-galaxy-ensemble-materialization/1":
        raise ValueError("request must use sigma-galaxy-ensemble-materialization/1")
    return materialize_galaxy_ensemble_realization(
        bundle_path=Path(request["bundlePath"]),
        archive_path=Path(request["archivePath"]),
        artifact=str(request["artifact"]),
        selection=request["selection"],
        output_directory=Path(request["outputDirectory"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args()
    result = execute_materialization_request(arguments.request)
    print(
        json.dumps(
            {
                "schemaVersion": "sigma-galaxy-ensemble-materialization-result/1",
                "state": "succeeded",
                "bundleSha256": result["bundleSha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schemaVersion": "sigma-galaxy-ensemble-materialization-error/1",
                    "state": "rejected_input",
                    "errorType": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            file=__import__("sys").stderr,
        )
        raise SystemExit(2) from error
