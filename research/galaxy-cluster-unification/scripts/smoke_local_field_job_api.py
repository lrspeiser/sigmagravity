"""Exercise the real asynchronous HTTP path from upload through field artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.field_job import model_sha256, write_array_bundle


def model(dimensions: int, *, axisymmetric_rotation: bool = False) -> dict[str, Any]:
    coordinate_system = (
        "axisymmetric_cylindrical"
        if axisymmetric_rotation
        else f"cartesian_{dimensions}d"
    )
    manifest = {
        "schemaVersion": "sigma-field-model/1",
        "name": (
            "Asynchronous API axisymmetric rotation field"
            if axisymmetric_rotation
            else f"Asynchronous API manufactured {dimensions}D field"
        ),
        "modelClass": "stationary_elliptic",
        "source": {
            "format": "plain_text",
            "text": (
                "laplacian(u) = forcing; acceleration = -gradient(u)"
                if axisymmetric_rotation
                else "laplacian(u) = forcing"
            ),
            "confirmedCanonical": False,
        },
        "geometry": {
            "coordinateSystem": coordinate_system,
            "dimensions": dimensions,
            "domain": {"lengthUnit": "m", "boundaryExtent": "unit hypercube"},
        },
        "fields": {
            "forcing": {
                "rank": "scalar",
                "role": "source",
                "unit": "1/s^2",
                "datasetKey": "forcing",
            },
            "u": {
                "rank": "scalar",
                "role": "solved",
                "unit": "m^2/s^2",
                "boundary": {"type": "dirichlet", "value": 0.0},
            },
        },
        "parameters": {},
        "equations": [
            {
                "id": "manufactured",
                "kind": "equality",
                "lhs": {"op": "laplacian", "args": [{"field": "u"}]},
                "rhs": {"field": "forcing"},
            }
        ],
        "observables": [
            {
                "id": "gradient",
                "target": "massive_tracers" if axisymmetric_rotation else "diagnostic",
                "rank": "vector",
                "unit": "m/s^2",
                "expression": (
                    {
                        "op": "negate",
                        "args": [
                            {"op": "gradient", "args": [{"field": "u"}]}
                        ],
                    }
                    if axisymmetric_rotation
                    else {"op": "gradient", "args": [{"field": "u"}]}
                ),
            }
        ],
        "dataRequirements": [{"key": "forcing", "rank": "scalar", "unit": "1/s^2"}],
        "solver": {
            "family": "finite_volume_elliptic",
            "relativeTolerance": 1e-10,
            "maxIterations": 8,
            "damping": 1.0,
        },
        "parameterPolicy": {"mode": "universal_fixed", "perObjectParameters": []},
    }
    manifest["source"]["confirmedCanonical"] = True
    manifest["source"]["confirmedModelSha256"] = model_sha256(manifest)
    return manifest


def metadata(
    spacing: float, dimensions: int, *, axisymmetric_rotation: bool = False
) -> dict[str, Any]:
    value = {
        "schemaVersion": "sigma-array-bundle-request/1",
        "geometry": {
            "coordinateSystem": (
                "axisymmetric_cylindrical"
                if axisymmetric_rotation
                else f"cartesian_{dimensions}d"
            ),
            "dimensions": dimensions,
            "spacing": [spacing] * dimensions,
            "lengthUnit": "m",
            "axisOrder": (
                ["r", "z"]
                if axisymmetric_rotation
                else ["x", "y", "z"][:dimensions]
            ),
            "origin": [0.0, 0.0] if axisymmetric_rotation else [0.0] * dimensions,
            "referenceFrame": (
                "analytic_axisymmetric_rotation"
                if axisymmetric_rotation
                else "manufactured_unit_hypercube"
            ),
        },
        "arrays": {
            "forcing": {
                "npzKey": "raw_forcing",
                "unit": "1/s^2",
                "rank": "scalar",
                "role": "source",
            }
        },
        "provenance": {"kind": "analytic_manufactured_solution", "citation": "API smoke fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }
    if axisymmetric_rotation:
        value["arrays"]["u_boundary"] = {
            "npzKey": "raw_boundary",
            "unit": "m^2/s^2",
            "rank": "scalar",
            "role": "boundary",
        }
    return value


def request(url: str, *, method: str = "GET", payload: Any = None, content_type: str = "application/json") -> bytes:
    data = None
    if payload is not None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    call = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(call, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned {error.code}: {detail}") from error


def request_json(url: str, *, method: str = "GET", payload: Any = None) -> dict[str, Any]:
    return json.loads(request(url, method=method, payload=payload))


def run_case(
    base: str,
    root: Path,
    dimensions: int,
    cells: int,
    *,
    axisymmetric_rotation: bool = False,
) -> dict[str, Any]:
    axis = np.linspace(0.0, 1.0, cells)
    spacing = 1.0 / (cells - 1)
    if axisymmetric_rotation:
        radius, _vertical = np.meshgrid(axis, axis, indexing="ij")
        omega = 3.0
        expected = 0.5 * omega**2 * radius**2
        forcing = np.full_like(expected, 2.0 * omega**2)
        arrays = {"raw_forcing": forcing, "raw_boundary": expected}
        bundle_name = "bundle_axisymmetric_rotation"
    else:
        coordinates = np.meshgrid(*([axis] * dimensions), indexing="ij")
        expected = np.ones([cells] * dimensions)
        for coordinate in coordinates:
            expected *= np.sin(np.pi * coordinate)
        forcing = -float(dimensions) * np.pi**2 * expected
        arrays = {"raw_forcing": forcing}
        bundle_name = f"bundle_{dimensions}d"
    bundle_directory = root / bundle_name
    bundle = write_array_bundle(
        bundle_directory,
        arrays,
        metadata(
            spacing,
            dimensions,
            axisymmetric_rotation=axisymmetric_rotation,
        ),
    )
    archive = (bundle_directory / "arrays.npz").read_bytes()
    ticket = request_json(
        f"{base}/api/v1/data-uploads",
        method="POST",
        payload={
            "schemaVersion": "sigma-data-upload-request/1",
            "inputBundle": bundle,
            "archive": {"sha256": hashlib.sha256(archive).hexdigest(), "bytes": len(archive)},
        },
    )
    ready = request_json(
        f"{base}{ticket['links']['content']}",
        method="PUT",
        payload=archive,
    )
    if ready["state"] != "ready":
        raise RuntimeError("array upload did not become ready")
    submission = request_json(
        f"{base}/api/v1/field-jobs",
        method="POST",
        payload={
            "schemaVersion": "sigma-field-job-submit/1",
            "model": model(
                dimensions, axisymmetric_rotation=axisymmetric_rotation
            ),
            "dataUploadId": ticket["id"],
            "request": {
                "schemaVersion": "sigma-field-job-request/1",
                "spacing": [spacing] * dimensions,
                "boundaryFields": {
                    "u": (
                        {"arrayKey": "u_boundary"}
                        if axisymmetric_rotation
                        else {"value": 0.0}
                    )
                },
                "requestedObservables": ["gradient"],
                "observationTargets": (
                    [
                        {
                            "schemaVersion": "sigma-observation-target/1",
                            "id": "axisymmetric-rotation",
                            "kind": "circular_speed_curve",
                            "observable": "gradient",
                            "centerM": [0.0, 0.5],
                            "radiiM": [0.125, 0.375, 0.625, 0.875],
                            "observedSpeedsMPerS": [0.375, 1.125, 1.875, 2.625],
                            "uncertaintiesMPerS": [0.01] * 4,
                            "minimumAzimuthalCoverage": 1.0,
                            "provenance": {
                                "kind": "analytic axisymmetric rotation fixture"
                            },
                            "license": {
                                "id": "CC0-1.0",
                                "redistributionAllowed": True,
                            },
                        }
                    ]
                    if axisymmetric_rotation
                    else []
                ),
                "seed": 1729 + dimensions,
            },
        },
    )
    terminal = {
        "succeeded",
        "failed",
        "failed_nonconvergence",
        "rejected_input",
        "infrastructure_failed",
        "cancelled",
    }
    job = submission
    deadline = time.monotonic() + 30.0
    while job["state"] not in terminal and time.monotonic() < deadline:
        time.sleep(0.1)
        job = request_json(f"{base}{submission['links']['self']}")
    if job["state"] != "succeeded":
        raise RuntimeError(f"{dimensions}D field job ended as {job['state']}: {job}")
    events = request_json(f"{base}{submission['links']['events']}")
    artifacts = request_json(f"{base}{submission['links']['artifacts']}")
    if job["workerSourceSha256"] != artifacts["manifest"]["worker"]["sourceSha256"]:
        raise RuntimeError("gateway and scientific worker source hashes disagree")
    downloaded: dict[str, bytes] = {}
    for artifact in artifacts["items"]:
        content = request(f"{base}{artifact['url']}")
        if (
            len(content) != artifact["bytes"]
            or hashlib.sha256(content).hexdigest() != artifact["sha256"]
        ):
            raise RuntimeError(f"downloaded artifact failed integrity: {artifact['path']}")
        downloaded[artifact["path"]] = content
    with np.load(io.BytesIO(downloaded["fields.npz"]), allow_pickle=False) as fields:
        relative_error = float(
            np.linalg.norm(fields["u"] - expected) / np.linalg.norm(expected)
        )
    if relative_error >= 0.01:
        raise RuntimeError(f"{dimensions}D field relative error exceeded acceptance: {relative_error}")
    observation_rmse = None
    observation_sampling_mode = None
    if axisymmetric_rotation:
        scores = json.loads(downloaded["observation_scores.json"])
        observation_rmse = float(scores["targets"][0]["score"]["rmseMPerS"])
        observation_sampling_mode = scores["targets"][0]["samplingMode"]
        if observation_rmse >= 1e-10:
            raise RuntimeError(
                f"axisymmetric rotation RMSE exceeded acceptance: {observation_rmse}"
            )
        if observation_sampling_mode != "axisymmetric_midplane_direct":
            raise RuntimeError("axisymmetric job used the wrong observation sampler")
    return {
        "coordinateSystem": (
            "axisymmetric_cylindrical"
            if axisymmetric_rotation
            else f"cartesian_{dimensions}d"
        ),
        "dimensions": dimensions,
        "gridShape": [cells] * dimensions,
        "uploadId": ticket["id"],
        "operationalJobId": submission["id"],
        "scientificJobId": job["scientificJobId"],
        "state": job["state"],
        "eventStates": [event["state"] for event in events["items"]],
        "artifactCount": len(artifacts["items"]),
        "allDownloadedArtifactHashesValid": True,
        "workerSourceHashAgrees": True,
        "relativeL2FieldError": relative_error,
        "observationRmseMPerS": observation_rmse,
        "observationSamplingMode": observation_sampling_mode,
        "perObjectGravityParameters": job["parameterAccounting"]["perObject"],
    }


def main() -> None:
    base = os.environ.get("SIMULATOR_URL", "http://127.0.0.1:4173").rstrip("/")
    with tempfile.TemporaryDirectory(prefix="sigma-field-api-") as directory:
        root = Path(directory)
        cases = [
            run_case(base, root, 2, 25),
            run_case(base, root, 3, 17),
            run_case(base, root, 2, 25, axisymmetric_rotation=True),
        ]
    print(
        json.dumps(
            {
                "base": base,
                "cases": cases,
                "allCasesSucceeded": all(case["state"] == "succeeded" for case in cases),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
