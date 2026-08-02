"""Run one frozen Newtonian field manifest over three generated galaxy volumes."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.field_job import write_array_bundle

KPC_M = 3.085677581491367e19


def request(
    url: str,
    *,
    method: str = "GET",
    payload: Any = None,
    content_type: str = "application/json",
) -> bytes:
    data = None
    if payload is not None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    call = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": content_type}
    )
    try:
        with urllib.request.urlopen(call, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned {error.code}: {detail}") from error


def request_json(url: str, *, method: str = "GET", payload: Any = None) -> dict[str, Any]:
    return json.loads(request(url, method=method, payload=payload))


def wait(base: str, submission: dict[str, Any], timeout: float = 240.0) -> dict[str, Any]:
    terminal = {
        "succeeded",
        "completed_with_failures",
        "failed",
        "failed_nonconvergence",
        "rejected_input",
        "infrastructure_failed",
        "cancelled",
    }
    result = submission
    deadline = time.monotonic() + timeout
    while result["state"] not in terminal and time.monotonic() < deadline:
        time.sleep(0.1)
        result = request_json(f"{base}{submission['links']['self']}")
    if result["state"] not in {"succeeded", "completed_with_failures"}:
        raise RuntimeError(f"job ended as {result['state']}: {result}")
    return result


def download_artifacts(
    base: str, submission: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bytes]]:
    response = request_json(f"{base}{submission['links']['artifacts']}")
    downloaded: dict[str, bytes] = {}
    for artifact in response["items"]:
        content = request(f"{base}{artifact['url']}")
        if len(content) != artifact["bytes"]:
            raise RuntimeError(f"artifact size mismatch: {artifact['path']}")
        if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {artifact['path']}")
        downloaded[artifact["path"]] = content
    return response, downloaded


def upload_ddo101(base: str, temporary: Path) -> dict[str, Any]:
    source_path = (
        ROOT / "results" / "p0639_registered_baryonic_maps" / "maps" / "DDO101.npz"
    )
    with np.load(source_path) as source:
        axis = np.asarray(source["axis_kpc"], dtype=float)
        gas = np.asarray(source["gas"], dtype=float)
        stars = np.asarray(source["stars"], dtype=float)
    bundle = write_array_bundle(
        temporary / "bundle",
        {"gas": gas, "stars": stars},
        {
            "schemaVersion": "sigma-array-bundle-request/1",
            "geometry": {
                "coordinateSystem": "cartesian_2d",
                "dimensions": 2,
                "spacing": [float(axis[1] - axis[0])] * 2,
                "lengthUnit": "kpc",
                "axisOrder": ["x", "y"],
                "referenceFrame": "intrinsic_face_on_baryonic_map",
            },
            "arrays": {
                "gas_surface_density": {
                    "npzKey": "gas",
                    "unit": "M_sun/kpc^2",
                    "rank": "scalar",
                    "role": "source",
                },
                "stellar_surface_density": {
                    "npzKey": "stars",
                    "unit": "M_sun/kpc^2",
                    "rank": "scalar",
                    "role": "source",
                },
            },
            "provenance": {"kind": "P0639 registered baryonic map", "galaxy": "DDO101"},
            "license": {"id": "research-source-license", "redistributionAllowed": False},
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
    return request_json(f"{base}{ticket['links']['content']}", method="PUT", payload=archive)


def ddo101_rotation_target() -> dict[str, Any]:
    path = (
        ROOT
        / "data"
        / "raw"
        / "p0633_little_things_kinematics"
        / "stw3285_Supplementary_Data.zip"
    )
    with zipfile.ZipFile(path) as outer:
        nested = outer.read("results.zip")
    with zipfile.ZipFile(io.BytesIO(nested)) as archive:
        text = archive.read("finalrot/ddo101_onlinetab.txt").decode(
            "utf-8", errors="replace"
        )
    rows = []
    for line in text.splitlines():
        tokens = line.strip().split()
        if len(tokens) != 12:
            continue
        try:
            rows.append([float(value) for value in tokens])
        except ValueError:
            continue
    if len(rows) != 10:
        raise RuntimeError("DDO101 published rotation target changed")
    return {
        "schemaVersion": "sigma-observation-target/1",
        "id": "DDO101-published-circular-speed",
        "kind": "circular_speed_curve",
        "observable": "massive_tracer_acceleration",
        "centerM": [0.0, 0.0, 0.0],
        "planeAxes": [0, 1],
        "radiiM": [row[1] * KPC_M for row in rows],
        "observedSpeedsMPerS": [row[6] * 1000.0 for row in rows],
        "uncertaintiesMPerS": [row[7] * 1000.0 for row in rows],
        "azimuthalSamples": 128,
        "minimumAzimuthalCoverage": 1.0,
        "fittedNuisanceParameters": 0,
        "provenance": {
            "kind": "published LITTLE THINGS rotation table",
            "archiveSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "member": "results.zip/finalrot/ddo101_onlinetab.txt",
        },
        "license": {
            "id": "published-supplementary-material",
            "redistributionAllowed": False,
        },
    }


def newtonian_fixture_conformance(base: str, field_job_id: str) -> dict[str, float]:
    artifacts = request_json(f"{base}/api/v1/field-jobs/{field_job_id}/artifacts")
    record = next(
        item for item in artifacts["items"] if item["path"] == "observation_predictions.csv"
    )
    content = request(f"{base}{record['url']}")
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise RuntimeError("field observation prediction hash mismatch")
    predicted_rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    radius = np.asarray([float(row["radius_m"]) / KPC_M for row in predicted_rows])
    predicted = np.asarray(
        [float(row["predicted_speed_m_s"]) / 1000.0 for row in predicted_rows]
    )
    with (
        ROOT
        / "results"
        / "p0708_external_prediction_lock"
        / "galaxy_prediction_curves.csv"
    ).open(encoding="utf-8", newline="") as handle:
        reference_rows = [
            row
            for row in csv.DictReader(handle)
            if row["system"] == "DDO101" and row["model"] == "Newtonian_3D"
        ]
    reference_radius = np.asarray([float(row["radius_kpc"]) for row in reference_rows])
    reference_speed = np.asarray(
        [float(row["circular_speed_km_s"]) for row in reference_rows]
    )
    interpolated = np.interp(radius, reference_radius, reference_speed)
    rmse = float(np.sqrt(np.mean(np.square(predicted - interpolated))))
    normalized = rmse / float(np.sqrt(np.mean(np.square(interpolated))))
    return {"rmseKmS": rmse, "normalizedRmse": normalized}


def galaxy_job(
    base: str,
    *,
    operation: str,
    upload_id: str | None = None,
    parameters: dict[str, Any] | None = None,
    controls: dict[str, Any] | None = None,
    output_cells: int | None = None,
    seed: int,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    payload: dict[str, Any] = {
        "schemaVersion": "sigma-galaxy-job-submit/1",
        "operation": operation,
        "galaxy": "DDO101",
        "vertical": {
            "enabled": operation == "generate",
            "realizations": 1,
            "zCells": 9,
            "seed": seed,
        },
        "outputLicense": {
            "id": "research-source-license",
            "redistributionAllowed": False,
        },
    }
    if operation == "extract_roundtrip":
        payload.update(
            {
                "dataUploadId": upload_id,
                "extractionControls": {
                    "radialBins": 20,
                    "maximumFourierMode": 4,
                    "residualFeatureCountPerComponent": 24,
                },
            }
        )
    else:
        payload["parameterPackage"] = parameters
        payload["generationControls"] = controls or {}
        payload["outputGrid"] = {"cellsPerAxis": int(output_cells or 25)}
    submission = request_json(f"{base}/api/v1/galaxy-jobs", method="POST", payload=payload)
    completed = wait(base, submission)
    _, downloaded = download_artifacts(base, submission)
    return completed, downloaded


def main() -> None:
    base = os.environ.get("SIMULATOR_BASE_URL", "http://127.0.0.1:4173").rstrip("/")
    with tempfile.TemporaryDirectory(prefix="sigma-batch-http-") as temporary_value:
        upload = upload_ddo101(base, Path(temporary_value))
        _extracted, extract_artifacts = galaxy_job(
            base,
            operation="extract_roundtrip",
            upload_id=upload["id"],
            seed=301,
        )
        parameters = json.loads(extract_artifacts["parameters.json"])
        replay, _ = galaxy_job(
            base,
            operation="generate",
            parameters=parameters,
            controls={},
            output_cells=25,
            seed=302,
        )
        compact, _ = galaxy_job(
            base,
            operation="generate",
            parameters=parameters,
            controls={
                "gas": {"radialScale": 0.78},
                "stars": {"radialScale": 0.78},
            },
            output_cells=25,
            seed=303,
        )
        diffuse, _ = galaxy_job(
            base,
            operation="generate",
            parameters=parameters,
            controls={
                "gas": {"radialScale": 1.22},
                "stars": {"radialScale": 1.22},
            },
            output_cells=25,
            seed=304,
        )
        model = json.loads(
            (ROOT / "hosted-simulator" / "examples" / "models" / "newtonian-poisson.json").read_text(
                encoding="utf-8"
            )
        )
        batch_submission = request_json(
            f"{base}/api/v1/batches",
            method="POST",
            payload={
                "schemaVersion": "sigma-batch-submit/1",
                "model": model,
                "systems": [
                    {
                        "id": "DDO101-replay",
                        "galaxyJobId": replay["id"],
                        "galaxyArtifact": "field_volume_density",
                        "observationTargets": [ddo101_rotation_target()],
                    },
                    {
                        "id": "DDO101-compact",
                        "galaxyJobId": compact["id"],
                        "galaxyArtifact": "field_volume_density",
                    },
                    {
                        "id": "DDO101-diffuse",
                        "galaxyJobId": diffuse["id"],
                        "galaxyArtifact": "field_volume_density",
                    },
                ],
                "fieldRequest": {
                    "schemaVersion": "sigma-field-job-request/1",
                    "requestedObservables": ["massive_tracer_acceleration"],
                    "seed": 404,
                },
                "parameterPolicy": {
                    "mode": "published_fixed",
                    "perObjectParameters": [],
                },
            },
        )
        batch = wait(base, batch_submission)
        response, downloaded = download_artifacts(base, batch_submission)
        aggregate = json.loads(downloaded["aggregate_scores.json"])
        child_jobs = json.loads(downloaded["child_jobs.json"])
        replay_child = next(
            item for item in child_jobs["items"] if item["systemId"] == "DDO101-replay"
        )
        conformance = newtonian_fixture_conformance(base, replay_child["fieldJobId"])
        if batch["state"] != "succeeded":
            raise RuntimeError(f"batch did not fully succeed: {batch['state']}")
        if aggregate["systemCount"] != 3:
            raise RuntimeError("batch did not retain all three systems")
        if aggregate["succeededSystems"] != 3 or aggregate["convergenceFraction"] != 1.0:
            raise RuntimeError("not every system converged")
        if aggregate["maximumEquationResidual"] > 1e-7:
            raise RuntimeError(
                "batch equation residual exceeded acceptance threshold: "
                f"{aggregate['maximumEquationResidual']}"
            )
        if aggregate["perObjectGravityParameters"] != 0:
            raise RuntimeError("batch introduced per-object gravity parameters")
        if aggregate["parameterPolicy"]["mode"] != "published_fixed":
            raise RuntimeError("batch parameter policy changed")
        if aggregate["observationScoresAvailable"] is not True:
            raise RuntimeError("batch did not publish the requested rotation score")
        if aggregate["scoredObservationTargets"] != 1:
            raise RuntimeError("batch did not score exactly one declared target")
        if aggregate["validObservationPoints"] != 10:
            raise RuntimeError("batch lost published DDO101 rotation points")
        if not np.isfinite(aggregate["observationRmseMPerS"]):
            raise RuntimeError("batch observation RMSE is not finite")
        if conformance["normalizedRmse"] > 0.1:
            raise RuntimeError(
                "generic Newtonian curve does not reproduce the frozen Newtonian fixture: "
                f"{conformance}"
            )
        required = {
            "aggregate_scores.json",
            "batch.json",
            "child_jobs.json",
            "failures.csv",
            "llm_briefing.md",
            "model.json",
            "observation_predictions.csv",
            "per_galaxy.csv",
            "report.html",
            "reproduction_command.txt",
        }
        if set(downloaded) != required:
            raise RuntimeError("batch deterministic artifact set changed")
        print(
            json.dumps(
                {
                    "schemaVersion": "sigma-batch-http-smoke/1",
                    "state": "pass" if batch["state"] == "succeeded" else batch["state"],
                    "batchId": batch["id"],
                    "model": model["name"],
                    "modelSha256": aggregate["modelSha256"],
                    "parameterPolicy": aggregate["parameterPolicy"],
                    "systems": [item["systemId"] for item in child_jobs["items"]],
                    "successfulSystems": aggregate["succeededSystems"],
                    "convergenceFraction": aggregate["convergenceFraction"],
                    "maximumEquationResidual": aggregate["maximumEquationResidual"],
                    "perObjectGravityParameters": aggregate["perObjectGravityParameters"],
                    "observationScoresAvailable": aggregate["observationScoresAvailable"],
                    "scoredObservationTargets": aggregate["scoredObservationTargets"],
                    "validObservationPoints": aggregate["validObservationPoints"],
                    "observationRmseKmS": aggregate["observationRmseMPerS"] / 1000.0,
                    "observationWeightedRmseKmS": aggregate[
                        "observationInverseVarianceWeightedRmseMPerS"
                    ]
                    / 1000.0,
                    "observationReducedChiSquare": aggregate[
                        "observationReducedChiSquare"
                    ],
                    "frozenNewtonianFixtureConformance": conformance,
                    "artifactCount": len(response["items"]),
                    "allDownloadedArtifactHashesValid": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
