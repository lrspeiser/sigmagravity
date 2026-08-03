"""Run one frozen Newtonian field manifest over three generated galaxy volumes."""

from __future__ import annotations

import copy
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
    gas_sigma = np.maximum(0.15 * np.abs(gas), 1.0e3)
    stars_sigma = np.maximum(0.20 * np.abs(stars), 1.0e3)
    conditioning_mask = np.asarray((gas + stars) > 0.0, dtype=float)
    bundle = write_array_bundle(
        temporary / "bundle",
        {
            "gas": gas,
            "stars": stars,
            "gas_sigma": gas_sigma,
            "stars_sigma": stars_sigma,
            "conditioning_mask": conditioning_mask,
        },
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
                "gas_surface_density_uncertainty": {
                    "npzKey": "gas_sigma",
                    "unit": "M_sun/kpc^2",
                    "rank": "scalar",
                    "role": "uncertainty",
                },
                "stellar_surface_density_uncertainty": {
                    "npzKey": "stars_sigma",
                    "unit": "M_sun/kpc^2",
                    "rank": "scalar",
                    "role": "uncertainty",
                },
                "baryonic_conditioning_mask": {
                    "npzKey": "conditioning_mask",
                    "unit": "1",
                    "rank": "scalar",
                    "role": "mask",
                },
            },
            "provenance": {
                "kind": "P0639 registered baryonic map",
                "galaxy": "DDO101",
                "conditioningUncertaintyStatus": (
                    "commissioning fractional uncertainty fixture, not a published error map"
                ),
            },
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


def newtonian_fixture_conformance(
    base: str, observation_job_id: str
) -> dict[str, float]:
    artifacts = request_json(
        f"{base}/api/v1/observation-evaluation-jobs/{observation_job_id}/artifacts"
    )
    record = next(
        item for item in artifacts["items"] if item["path"] == "observation_predictions.csv"
    )
    content = request(f"{base}{record['url']}")
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise RuntimeError("decoupled observation prediction hash mismatch")
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
    uncertainty_ensemble: dict[str, Any] | None = None,
    vertical_enabled: bool | None = None,
    output_cells: int | None = None,
    seed: int,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    payload: dict[str, Any] = {
        "schemaVersion": "sigma-galaxy-job-submit/1",
        "operation": operation,
        "galaxy": "DDO101",
        "vertical": {
            "enabled": operation == "generate"
            if vertical_enabled is None
            else vertical_enabled,
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
    if uncertainty_ensemble is not None:
        payload["uncertaintyEnsemble"] = uncertainty_ensemble
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
        ensemble_replay, ensemble_artifacts = galaxy_job(
            base,
            operation="extract_roundtrip",
            upload_id=upload["id"],
            uncertainty_ensemble={
                "enabled": True,
                "realizations": 2,
                "seed": 902,
                "priors": {
                    "gasMassLnSigma": 0.05,
                    "stellarMassLnSigma": 0.05,
                    "gasRadialScaleLnSigma": 0.04,
                    "stellarRadialScaleLnSigma": 0.04,
                },
                "conditioning": {
                    "enabled": True,
                    "likelihood": "diagonal_gaussian_surface_density",
                    "useMask": True,
                    "minimumValidPixelsPerComponent": 25,
                    "correlationAreaPixels": 9.0,
                },
            },
            vertical_enabled=True,
            seed=302,
        )
        conditioning = json.loads(ensemble_artifacts["baryonic_conditioning.json"])
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
        batch_payload = {
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
                    {
                        "id": "DDO101-ensemble",
                        "galaxyJobId": ensemble_replay["id"],
                        "galaxyArtifact": "volume_density_ensemble",
                        "ensembleSelection": {
                            "surfaceRealizations": [0, 1],
                            "verticalRealizations": [0],
                            "maximumChildren": 2,
                        },
                        "observationTargets": [ddo101_rotation_target()],
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
            }
        batch_submission = request_json(
            f"{base}/api/v1/batches",
            method="POST",
            payload=batch_payload,
        )
        batch = wait(base, batch_submission)
        response, downloaded = download_artifacts(base, batch_submission)
        aggregate = json.loads(downloaded["aggregate_scores.json"])
        child_jobs = json.loads(downloaded["child_jobs.json"])
        replay_child = next(
            item for item in child_jobs["items"] if item["systemId"] == "DDO101-replay"
        )
        observation_job_id = replay_child["observationEvaluationJobId"]
        if not observation_job_id:
            raise RuntimeError("batch did not create a decoupled observation child")
        conformance = newtonian_fixture_conformance(base, observation_job_id)
        field_child = request_json(
            f"{base}/api/v1/field-jobs/{replay_child['fieldJobId']}"
        )
        observation_child = request_json(
            f"{base}/api/v1/observation-evaluation-jobs/{observation_job_id}"
        )
        if field_child["preflight"]["observationTargets"]:
            raise RuntimeError("batch field child still embeds observation targets")
        if observation_child["fieldJobId"] != replay_child["fieldJobId"]:
            raise RuntimeError("observation child does not reference its immutable field child")
        if observation_child["evaluationAddedGravityParameters"] != 0:
            raise RuntimeError("observation child added a gravity parameter")
        if any(
            item["observationEvaluationJobId"] is not None
            for item in child_jobs["items"]
            if item["parentSystemId"] not in {"DDO101-replay", "DDO101-ensemble"}
        ):
            raise RuntimeError("field-only systems unexpectedly created observation jobs")
        changed_payload = copy.deepcopy(batch_payload)
        changed_payload["systems"][0]["observationTargets"][0][
            "uncertaintiesMPerS"
        ] = [
            value * 1.1
            for value in changed_payload["systems"][0]["observationTargets"][0][
                "uncertaintiesMPerS"
            ]
        ]
        changed_submission = request_json(
            f"{base}/api/v1/batches", method="POST", payload=changed_payload
        )
        wait(base, changed_submission)
        _, changed_downloaded = download_artifacts(base, changed_submission)
        changed_children = json.loads(changed_downloaded["child_jobs.json"])["items"]
        changed_replay = next(
            item for item in changed_children if item["systemId"] == "DDO101-replay"
        )
        if changed_replay["fieldJobId"] != replay_child["fieldJobId"]:
            raise RuntimeError("changing observation uncertainty changed the field child")
        if (
            changed_replay["observationEvaluationJobId"]
            == replay_child["observationEvaluationJobId"]
        ):
            raise RuntimeError("changing observation uncertainty reused a stale observation child")
        duplicate_changed = request_json(
            f"{base}/api/v1/batches", method="POST", payload=changed_payload
        )
        if (
            duplicate_changed["id"] != changed_submission["id"]
            or duplicate_changed["duplicate"] is not True
        ):
            raise RuntimeError("identical composed batch did not reuse its identity")
        if batch["state"] != "succeeded":
            raise RuntimeError(f"batch did not fully succeed: {batch['state']}")
        if aggregate["systemCount"] != 5 or aggregate["parentSystemCount"] != 4:
            raise RuntimeError("batch did not retain all parent systems and ensemble children")
        if aggregate["succeededSystems"] != 5 or aggregate["convergenceFraction"] != 1.0:
            raise RuntimeError("not every system converged")
        if aggregate["maximumEquationResidual"] > 1e-7:
            raise RuntimeError(
                "batch equation residual exceeded acceptance threshold: "
                f"{aggregate['maximumEquationResidual']}"
            )
        if aggregate["perObjectGravityParameters"] != 0:
            raise RuntimeError("batch introduced per-object gravity parameters")
        if aggregate["observationAddedGravityParameters"] != 0:
            raise RuntimeError("observation evaluation introduced gravity parameters")
        if aggregate["parameterPolicy"]["mode"] != "published_fixed":
            raise RuntimeError("batch parameter policy changed")
        if aggregate["observationScoresAvailable"] is not True:
            raise RuntimeError("batch did not publish the requested rotation score")
        if aggregate["scoredObservationTargets"] != 3:
            raise RuntimeError("batch did not score the anchor and two ensemble targets")
        if aggregate["validObservationPoints"] != 30:
            raise RuntimeError("batch lost published DDO101 rotation points")
        uncertainty = aggregate["withinSystemUncertainty"]
        if (
            uncertainty["status"]
            != "baryonic_surface_likelihood_conditioned_partial_posterior"
            or uncertainty["ensembleParentCount"] != 1
            or uncertainty["ensembleRealizationCount"] != 2
            or uncertainty["surfaceLikelihoodConditionedParentCount"] != 1
            or uncertainty["degenerateConditionedParentCount"] != 1
            or uncertainty["credibleIntervalReady"] is not False
            or uncertainty["predictionQuantilePoints"] != 10
        ):
            raise RuntimeError("batch did not publish honest ensemble propagation metadata")
        if (
            conditioning["surfaceLikelihoodConditioned"] is not True
            or conditioning["verticalStructureConditioned"] is not False
            or not np.isclose(sum(conditioning["weights"]), 1.0)
            or len(set(np.round(conditioning["weights"], 14))) < 2
        ):
            raise RuntimeError("baryonic image conditioning did not produce valid weights")
        ensemble_summary = json.loads(downloaded["ensemble_summary.json"])
        if (
            len(ensemble_summary["systems"]) != 1
            or ensemble_summary["systems"][0]["realizationCount"] != 2
            or ensemble_summary["systems"][0]["metrics"]["observationRmseMPerS"]["count"]
            != 2
            or ensemble_summary["systems"][0]["weighting"][
                "surfaceLikelihoodConditioned"
            ]
            is not True
        ):
            raise RuntimeError("ensemble prediction summary is incomplete")
        prediction_quantiles = downloaded["ensemble_prediction_quantiles.csv"].decode(
            "utf-8"
        )
        if len(prediction_quantiles.strip().splitlines()) != 11:
            raise RuntimeError("weighted per-radius prediction quantiles are incomplete")
        per_realization = downloaded["per_realization.csv"].decode("utf-8")
        if "DDO101-ensemble::s000::v000" not in per_realization or "DDO101-ensemble::s001::v000" not in per_realization:
            raise RuntimeError("per-realization report lost deterministic child identities")
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
            "ensemble_prediction_quantiles.csv",
            "ensemble_summary.csv",
            "ensemble_summary.json",
            "failures.csv",
            "llm_briefing.md",
            "model.json",
            "observation_predictions.csv",
            "observation_velocity_field_predictions.csv",
            "observation_multiple_image_predictions.csv",
            "observation_multiple_image_families.csv",
            "per_galaxy.csv",
            "per_realization.csv",
            "report.html",
            "reproduction_command.txt",
        }
        if set(downloaded) != required:
            raise RuntimeError("batch deterministic artifact set changed")
        print(
            json.dumps(
                {
                    "schemaVersion": "sigma-batch-http-smoke/3",
                    "state": "pass" if batch["state"] == "succeeded" else batch["state"],
                    "batchId": batch["id"],
                    "model": model["name"],
                    "modelSha256": aggregate["modelSha256"],
                    "parameterPolicy": aggregate["parameterPolicy"],
                    "systems": [item["systemId"] for item in child_jobs["items"]],
                    "successfulSystems": aggregate["succeededSystems"],
                    "parentSystems": aggregate["parentSystemCount"],
                    "ensembleRealizations": uncertainty["ensembleRealizationCount"],
                    "baryonicConditioningStatus": conditioning["status"],
                    "baryonicConditioningWeights": conditioning["weights"],
                    "baryonicConditioningEffectiveSampleSize": conditioning[
                        "effectiveSampleSize"
                    ],
                    "degenerateConditionedParentCount": uncertainty[
                        "degenerateConditionedParentCount"
                    ],
                    "credibleIntervalReady": uncertainty["credibleIntervalReady"],
                    "weightedPredictionQuantilePoints": uncertainty[
                        "predictionQuantilePoints"
                    ],
                    "convergenceFraction": aggregate["convergenceFraction"],
                    "maximumEquationResidual": aggregate["maximumEquationResidual"],
                    "perObjectGravityParameters": aggregate["perObjectGravityParameters"],
                    "observationAddedGravityParameters": aggregate[
                        "observationAddedGravityParameters"
                    ],
                    "fieldJobId": replay_child["fieldJobId"],
                    "observationEvaluationJobId": observation_job_id,
                    "fieldChildObservationTargetCount": len(
                        field_child["preflight"]["observationTargets"]
                    ),
                    "ensembleObservationChildren": sum(
                        item["observationEvaluationJobId"] is not None
                        for item in child_jobs["items"]
                        if item["parentSystemId"] == "DDO101-ensemble"
                    ),
                    "changedObservationPreservedFieldJobId": (
                        changed_replay["fieldJobId"] == replay_child["fieldJobId"]
                    ),
                    "changedObservationChangedEvaluationJobId": (
                        changed_replay["observationEvaluationJobId"]
                        != replay_child["observationEvaluationJobId"]
                    ),
                    "duplicateComposedBatchReused": duplicate_changed["duplicate"],
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
