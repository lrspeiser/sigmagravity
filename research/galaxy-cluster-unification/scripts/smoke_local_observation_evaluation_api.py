"""Exercise the decoupled observation-evaluation API with the real workers."""

from __future__ import annotations

import hashlib
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


def request(
    url: str,
    *,
    method: str = "GET",
    payload: Any = None,
    content_type: str = "application/json",
) -> bytes:
    data = None
    if payload is not None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    call = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": content_type}
    )
    try:
        with urllib.request.urlopen(call, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned {error.code}: {detail}") from error


def request_json(url: str, *, method: str = "GET", payload: Any = None) -> dict:
    return json.loads(request(url, method=method, payload=payload))


def upload_bundle(base: str, directory: Path) -> dict:
    bundle = json.loads((directory / "bundle.json").read_text())
    archive = (directory / "arrays.npz").read_bytes()
    ticket = request_json(
        f"{base}/api/v1/data-uploads",
        method="POST",
        payload={
            "schemaVersion": "sigma-data-upload-request/1",
            "inputBundle": bundle,
            "archive": {
                "sha256": hashlib.sha256(archive).hexdigest(),
                "bytes": len(archive),
            },
        },
    )
    return request_json(
        f"{base}{ticket['links']['content']}", method="PUT", payload=archive
    )


def wait_for_job(base: str, submission: dict) -> dict:
    terminal = {
        "succeeded",
        "failed",
        "failed_nonconvergence",
        "rejected_input",
        "infrastructure_failed",
        "cancelled",
    }
    job = submission
    deadline = time.monotonic() + 30
    while job["state"] not in terminal and time.monotonic() < deadline:
        time.sleep(0.05)
        job = request_json(f"{base}{submission['links']['self']}")
    if job["state"] != "succeeded":
        raise RuntimeError(f"job ended as {job['state']}: {job}")
    return job


def model() -> dict:
    manifest = {
        "schemaVersion": "sigma-field-model/1",
        "name": "P0732 HTTP solid-body field",
        "modelClass": "stationary_elliptic",
        "source": {
            "format": "plain_text",
            "text": "laplacian(u)=forcing; acceleration=-gradient(u)",
            "confirmedCanonical": False,
        },
        "geometry": {
            "coordinateSystem": "cartesian_2d",
            "dimensions": 2,
            "domain": {"lengthUnit": "m", "boundaryExtent": "fixture box"},
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
                "boundary": {"type": "dirichlet", "value": 0},
            },
        },
        "parameters": {},
        "equations": [
            {
                "id": "poisson",
                "kind": "equality",
                "lhs": {"op": "laplacian", "args": [{"field": "u"}]},
                "rhs": {"field": "forcing"},
            }
        ],
        "observables": [
            {
                "id": "acceleration",
                "target": "massive_tracers",
                "rank": "vector",
                "unit": "m/s^2",
                "expression": {
                    "op": "negate",
                    "args": [{"op": "gradient", "args": [{"field": "u"}]}],
                },
            }
        ],
        "dataRequirements": [
            {"key": "forcing", "rank": "scalar", "unit": "1/s^2"}
        ],
        "solver": {
            "family": "finite_volume_elliptic",
            "relativeTolerance": 1e-10,
            "maxIterations": 8,
            "damping": 1,
        },
        "parameterPolicy": {"mode": "universal_fixed", "perObjectParameters": []},
    }
    manifest["source"]["confirmedCanonical"] = True
    manifest["source"]["confirmedModelSha256"] = model_sha256(manifest)
    return manifest


def main() -> None:
    base = os.environ.get("SIMULATOR_URL", "http://127.0.0.1:4173").rstrip("/")
    with tempfile.TemporaryDirectory(prefix="sigma-observation-api-") as directory:
        root = Path(directory)
        cells = 17
        spacing = 0.5
        axis = np.linspace(-4, 4, cells)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        omega = 1.25
        potential = 0.5 * omega**2 * (x**2 + y**2)
        forcing = np.full_like(potential, 2 * omega**2)
        field_bundle = root / "field_bundle"
        write_array_bundle(
            field_bundle,
            {"forcing": forcing, "boundary": potential},
            {
                "schemaVersion": "sigma-array-bundle-request/1",
                "geometry": {
                    "coordinateSystem": "cartesian_2d",
                    "dimensions": 2,
                    "spacing": [spacing, spacing],
                    "origin": [-4, -4],
                    "lengthUnit": "m",
                },
                "arrays": {
                    "forcing": {
                        "npzKey": "forcing",
                        "unit": "1/s^2",
                        "rank": "scalar",
                        "role": "source",
                    },
                    "u_boundary": {
                        "npzKey": "boundary",
                        "unit": "m^2/s^2",
                        "rank": "scalar",
                        "role": "boundary",
                    },
                },
                "provenance": {"kind": "P0732 HTTP fixture"},
                "license": {"id": "CC0-1.0", "redistributionAllowed": True},
            },
        )
        field_upload = upload_bundle(base, field_bundle)
        source_submission = request_json(
            f"{base}/api/v1/field-jobs",
            method="POST",
            payload={
                "schemaVersion": "sigma-field-job-submit/1",
                "model": model(),
                "dataUploadId": field_upload["id"],
                "request": {
                    "schemaVersion": "sigma-field-job-request/1",
                    "boundaryFields": {"u": {"arrayKey": "u_boundary"}},
                    "requestedObservables": ["acceleration"],
                },
            },
        )
        source = wait_for_job(base, source_submission)
        observation_bundle = root / "observation_bundle"
        write_array_bundle(
            observation_bundle,
            {"placeholder": np.zeros((5, 5))},
            {
                "schemaVersion": "sigma-array-bundle-request/1",
                "geometry": {
                    "coordinateSystem": "observation_table",
                    "dimensions": 2,
                    "spacing": [1, 1],
                    "lengthUnit": "1",
                },
                "arrays": {
                    "placeholder": {
                        "unit": "1",
                        "rank": "scalar",
                        "role": "observation",
                    }
                },
                "provenance": {"kind": "P0732 HTTP fixture"},
                "license": {"id": "CC0-1.0", "redistributionAllowed": True},
            },
        )
        observation_upload = upload_bundle(base, observation_bundle)
        target = {
            "schemaVersion": "sigma-observation-target/1",
            "id": "http-curve",
            "kind": "circular_speed_curve",
            "observable": "acceleration",
            "centerM": [0, 0],
            "radiiM": [0.5, 1, 2, 3],
            "observedSpeedsMPerS": [0.625, 1.25, 2.5, 3.75],
            "uncertaintiesMPerS": [0.1, 0.1, 0.1, 0.1],
            "minimumAzimuthalCoverage": 1,
            "provenance": {"kind": "P0732 HTTP fixture"},
            "license": {"id": "CC0-1.0", "redistributionAllowed": True},
        }
        payload = {
            "schemaVersion": "sigma-observation-evaluation-job-submit/1",
            "fieldJobId": source["id"],
            "dataUploadId": observation_upload["id"],
            "observationTargets": [target],
        }
        submission = request_json(
            f"{base}/api/v1/observation-evaluation-jobs",
            method="POST",
            payload=payload,
        )
        completed = wait_for_job(base, submission)
        duplicate = request_json(
            f"{base}/api/v1/observation-evaluation-jobs",
            method="POST",
            payload=payload,
        )
        artifacts = request_json(f"{base}{completed['links']['artifacts']}")
        downloaded = {}
        for artifact in artifacts["items"]:
            content = request(f"{base}{artifact['url']}")
            if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
                raise RuntimeError(f"artifact failed hash verification: {artifact['path']}")
            downloaded[artifact["path"]] = content
        resource = json.loads(downloaded["resource_log.json"])
        result = json.loads(downloaded["scientific_result.json"])
        field_jobs = request_json(f"{base}/api/v1/field-jobs")
        if resource["fieldSolverInvocations"] != 0 or len(field_jobs["items"]) != 1:
            raise RuntimeError("observation evaluation unexpectedly triggered a field solve")
        if duplicate["id"] != submission["id"] or not duplicate["duplicate"]:
            raise RuntimeError("identical observation submission did not reuse its identity")
        print(
            json.dumps(
                {
                    "sourceFieldJobId": source["id"],
                    "observationJobId": completed["id"],
                    "state": completed["state"],
                    "fieldJobCount": len(field_jobs["items"]),
                    "fieldSolverInvocationsDuringEvaluation": resource[
                        "fieldSolverInvocations"
                    ],
                    "duplicateIdentityReused": True,
                    "artifactCount": len(artifacts["items"]),
                    "allArtifactHashesValid": True,
                    "validScoredPoints": result["observationEvaluation"][
                        "validScoredPoints"
                    ],
                    "evaluationAddedGravityParameters": result[
                        "evaluationAddedGravityParameters"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
