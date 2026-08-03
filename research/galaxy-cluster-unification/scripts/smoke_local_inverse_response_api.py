"""Exercise inverse response discovery through the real local HTTP queue."""

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

from voidscreen.field_job import write_array_bundle
from voidscreen.inverse_response import convolve_stationary_response


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
        with urllib.request.urlopen(call, timeout=90) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned {error.code}: {detail}") from error


def request_json(url: str, *, method: str = "GET", payload: Any = None) -> dict[str, Any]:
    return json.loads(request(url, method=method, payload=payload))


def wait_job(base: str, submission: dict[str, Any]) -> dict[str, Any]:
    terminal = {
        "succeeded",
        "failed",
        "rejected_input",
        "infrastructure_failed",
        "cancelled",
    }
    result = submission
    deadline = time.monotonic() + 120.0
    while result["state"] not in terminal and time.monotonic() < deadline:
        time.sleep(0.1)
        result = request_json(f"{base}{submission['links']['self']}")
    if result["state"] != "succeeded":
        raise RuntimeError(f"inverse response job ended as {result['state']}: {result}")
    return result


def main() -> None:
    base = os.environ.get(
        "SIMULATOR_BASE_URL",
        os.environ.get("SIMULATOR_URL", "http://127.0.0.1:4173"),
    )
    with tempfile.TemporaryDirectory(prefix="sigma-inverse-http-") as temporary_value:
        temporary = Path(temporary_value)
        cells = 15
        axis = np.linspace(-1.0, 1.0, cells)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        kernel_axis = np.arange(-2.0, 3.0)
        kx, ky = np.meshgrid(kernel_axis, kernel_axis, indexing="ij")
        kernel = np.exp(-((kx - 0.4) ** 2 / 2.0 + (ky + 0.3) ** 2 / 3.0))
        kernel /= np.sum(kernel)
        arrays: dict[str, np.ndarray] = {}
        descriptions: dict[str, dict[str, Any]] = {}
        for index, phase in enumerate((0.2, 1.3), start=1):
            source = (
                np.exp(
                    -(
                        (x - 0.25 * index) ** 2 / 0.11
                        + (y + 0.2) ** 2 / 0.18
                    )
                )
                + 0.65
                * np.exp(
                    -(
                        (x + 0.3) ** 2 / 0.2
                        + (y - 0.2 * index) ** 2 / 0.07
                    )
                )
                + 0.1 * np.cos(3.0 * np.arctan2(y, x) + phase) ** 2
            )
            target = 1.6 * convolve_stationary_response(source, kernel, 1.0)
            uncertainty = np.full_like(target, 0.01)
            for label, values, role, scientific_role in (
                ("baryons", source, "source", "baryonic_input"),
                (
                    "response",
                    target,
                    "auxiliary",
                    "model_derived_discovery_target",
                ),
                (
                    "uncertainty",
                    uncertainty,
                    "uncertainty",
                    "nuisance_or_calibration",
                ),
            ):
                key = f"{label}_{index}"
                arrays[key] = values
                descriptions[key] = {
                    "unit": "kg/m^2",
                    "rank": "scalar",
                    "role": role,
                    "scientificRole": scientific_role,
                }
        bundle = write_array_bundle(
            temporary / "bundle",
            arrays,
            {
                "schemaVersion": "sigma-array-bundle-request/1",
                "geometry": {
                    "coordinateSystem": "cartesian_2d",
                    "dimensions": 2,
                    "spacing": [1.0, 1.0],
                    "lengthUnit": "kpc",
                    "axisOrder": ["x", "y"],
                    "referenceFrame": "synthetic_inverse_http_fixture",
                },
                "arrays": descriptions,
                "provenance": {"kind": "synthetic_injected_kernel_fixture"},
                "license": {"id": "CC0-1.0", "redistributionAllowed": True},
            },
        )
        archive = (temporary / "bundle" / "arrays.npz").read_bytes()
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
        request_json(
            f"{base}{ticket['links']['content']}", method="PUT", payload=archive
        )
        submission = request_json(
            f"{base}/api/v1/inverse-response-jobs",
            method="POST",
            payload={
                "schemaVersion": "sigma-inverse-response-job-submit/1",
                "dataUploadId": ticket["id"],
                "systems": [
                    {
                        "id": f"SYNTH-{index}",
                        "sourceKey": f"baryons_{index}",
                        "targetKey": f"response_{index}",
                        "uncertaintyKey": f"uncertainty_{index}",
                    }
                    for index in (1, 2)
                ],
                "kernel": {
                    "shape": [5, 5],
                    "ridge": 1.0e-10,
                    "smoothness": 1.0e-8,
                    "nonnegative": True,
                },
                "uncertainty": {"ensembleSize": 20, "seed": 17},
                "nullControls": {
                    "kind": "source_radial_angle_shuffle",
                    "count": 19,
                    "seed": 23,
                },
                "outputLicense": {
                    "id": "CC-BY-4.0",
                    "redistributionAllowed": True,
                },
            },
        )
        completed = wait_job(base, submission)
        artifact_response = request_json(f"{base}{submission['links']['artifacts']}")
        downloaded: dict[str, bytes] = {}
        for artifact in artifact_response["items"]:
            content = request(f"{base}{artifact['url']}")
            if len(content) != artifact["bytes"]:
                raise RuntimeError(f"artifact size mismatch: {artifact['path']}")
            if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
                raise RuntimeError(f"artifact hash mismatch: {artifact['path']}")
            downloaded[artifact["path"]] = content
        if completed["workerSourceSha256"] != artifact_response["manifest"]["worker"]["sourceSha256"]:
            raise RuntimeError("gateway and inverse worker source hashes disagree")
        result = json.loads(downloaded["scientific_result.json"])
        with np.load(io.BytesIO(downloaded["kernels.npz"])) as kernels:
            recovered = np.asarray(kernels["normalized"], dtype=float)
        cosine = float(
            np.sum(recovered * kernel)
            / (np.linalg.norm(recovered) * np.linalg.norm(kernel))
        )
        if cosine < 0.999:
            raise RuntimeError(f"injected kernel recovery cosine is {cosine}")
        print(
            json.dumps(
                {
                    "schemaVersion": "sigma-inverse-response-http-smoke/1",
                    "state": "pass",
                    "operationalJobId": submission["id"],
                    "scientificJobId": completed["scientificJobId"],
                    "artifactCount": len(artifact_response["items"]),
                    "allDownloadedArtifactHashesValid": True,
                    "workerSourceHashAgrees": True,
                    "recoveredKernelCosine": cosine,
                    "recoveredAmplitude": result["amplitude"],
                    "rSquared": result["aggregateMetrics"]["r_squared"],
                    "nullPValue": result["nullSummary"]["permutation_p_value"],
                    "signalAgainstNull": result["nullSummary"]["signal_against_null"],
                    "fittedPerSystemGravityParameters": result[
                        "parameterAccounting"
                    ]["fittedPerSystemGravityParameters"],
                    "targetRole": result["dataRoleAudit"][0]["targetRole"],
                    "heldOutRawObservationsUsed": result["dataRoleAudit"][0][
                        "heldOutRawObservationsUsed"
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
