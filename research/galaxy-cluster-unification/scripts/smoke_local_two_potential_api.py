"""Exercise the published two-potential manifest through the real local HTTP path."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.field_job import write_array_bundle

from smoke_local_field_job_api import request, request_json


def bundle_metadata(spacing: float) -> dict[str, Any]:
    return {
        "schemaVersion": "sigma-array-bundle-request/1",
        "geometry": {
            "coordinateSystem": "cartesian_3d",
            "dimensions": 3,
            "spacing": [spacing, spacing, spacing],
            "lengthUnit": "m",
            "axisOrder": ["x", "y", "z"],
            "referenceFrame": "synthetic_baryon_test",
        },
        "arrays": {
            "baryon_density": {
                "npzKey": "raw_baryon_density",
                "unit": "kg/m^3",
                "rank": "scalar",
                "role": "source",
            }
        },
        "provenance": {
            "kind": "analytic_manufactured_source",
            "citation": "two-potential API smoke fixture",
        },
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
    }


def download_verified_artifacts(base: str, artifacts: dict[str, Any]) -> dict[str, bytes]:
    downloaded: dict[str, bytes] = {}
    for artifact in artifacts["items"]:
        content = request(f"{base}{artifact['url']}")
        if len(content) != artifact["bytes"]:
            raise RuntimeError(f"artifact size mismatch: {artifact['path']}")
        if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {artifact['path']}")
        downloaded[artifact["path"]] = content
    return downloaded


def main() -> None:
    base = os.environ.get("SIMULATOR_URL", "http://127.0.0.1:4173").rstrip("/")
    model_path = ROOT / "hosted-simulator" / "examples" / "models" / "two-potential.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    cells = 9
    spacing = 0.5 * 3.085677581491367e19
    coordinates = (np.arange(cells) - cells // 2) * spacing
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    density = 2.0e-21 * np.exp(
        -(x**2 + y**2 + z**2) / (2.0 * spacing**2)
    )

    with tempfile.TemporaryDirectory(prefix="sigma-two-potential-api-") as directory:
        bundle_directory = Path(directory) / "bundle"
        bundle = write_array_bundle(
            bundle_directory,
            {"raw_baryon_density": density},
            bundle_metadata(spacing),
        )
        archive = (bundle_directory / "arrays.npz").read_bytes()
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
                "model": model,
                "dataUploadId": ticket["id"],
                "request": {
                    "schemaVersion": "sigma-field-job-request/1",
                    "spacing": [spacing, spacing, spacing],
                    "requestedObservables": [
                        "massive_tracer_acceleration",
                        "photon_lensing_acceleration",
                    ],
                    "seed": 20260802,
                },
            },
        )
        terminal_states = {
            "succeeded",
            "failed",
            "failed_nonconvergence",
            "rejected_input",
            "infrastructure_failed",
            "cancelled",
        }
        job = submission
        deadline = time.monotonic() + 30.0
        while job["state"] not in terminal_states and time.monotonic() < deadline:
            time.sleep(0.1)
            job = request_json(f"{base}{submission['links']['self']}")
        if job["state"] != "succeeded":
            raise RuntimeError(f"two-potential field job ended as {job['state']}: {job}")
        artifacts = request_json(f"{base}{submission['links']['artifacts']}")
        downloaded = download_verified_artifacts(base, artifacts)
        if job["workerSourceSha256"] != artifacts["manifest"]["worker"]["sourceSha256"]:
            raise RuntimeError("gateway and scientific worker source hashes disagree")

        with np.load(io.BytesIO(downloaded["fields.npz"]), allow_pickle=False) as fields:
            field_ratio_error = float(
                np.linalg.norm(fields["Phi"] - 1.5 * fields["Psi"])
                / np.linalg.norm(fields["Phi"])
            )
        observable_ratio_errors = []
        with np.load(
            io.BytesIO(downloaded["observables.npz"]), allow_pickle=False
        ) as observables:
            for axis_index in range(3):
                matter = observables[
                    f"massive_tracer_acceleration__axis{axis_index}"
                ]
                photons = observables[
                    f"photon_lensing_acceleration__axis{axis_index}"
                ]
                denominator = max(float(np.linalg.norm(photons)), np.finfo(float).tiny)
                observable_ratio_errors.append(
                    float(np.linalg.norm(photons - 1.25 * matter) / denominator)
                )
        if field_ratio_error >= 1e-10 or max(observable_ratio_errors) >= 1e-10:
            raise RuntimeError("two-potential analytic ratio acceptance failed")

        result = json.loads(downloaded["scientific_result.json"])
        output = {
            "base": base,
            "state": job["state"],
            "modelSha256": job["preflight"]["modelSha256"],
            "scientificJobId": job["scientificJobId"],
            "artifactCount": len(artifacts["items"]),
            "allDownloadedArtifactHashesValid": True,
            "workerSourceHashAgrees": True,
            "fieldPhiOverPsiExpected": 1.5,
            "fieldRatioRelativeError": field_ratio_error,
            "photonOverMatterExpected": 1.25,
            "observableRatioRelativeErrors": observable_ratio_errors,
            "solverFamily": result["numericalMetadata"]["solver_family"],
            "equationCount": result["numericalMetadata"]["equation_count"],
            "solvedFieldCount": result["numericalMetadata"]["solved_field_count"],
            "perObjectGravityParameters": job["parameterAccounting"]["perObject"],
        }
        print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
