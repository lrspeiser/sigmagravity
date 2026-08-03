"""Exercise resolved-galaxy extraction and generation through real HTTP."""

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
        with urllib.request.urlopen(call, timeout=60) as response:
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
        "failed_nonconvergence",
        "rejected_input",
        "infrastructure_failed",
        "cancelled",
    }
    result = submission
    deadline = time.monotonic() + 90.0
    while result["state"] not in terminal and time.monotonic() < deadline:
        time.sleep(0.1)
        result = request_json(f"{base}{submission['links']['self']}")
    if result["state"] != "succeeded":
        raise RuntimeError(f"galaxy job ended as {result['state']}: {result}")
    return result


def artifacts(base: str, submission: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
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


def main() -> None:
    base = os.environ.get("SIMULATOR_BASE_URL", "http://127.0.0.1:4173")
    source_path = ROOT / "results" / "p0639_registered_baryonic_maps" / "maps" / "DDO101.npz"
    with np.load(source_path) as source:
        axis = np.asarray(source["axis_kpc"], dtype=float)
        gas = np.asarray(source["gas"], dtype=float)
        stars = np.asarray(source["stars"], dtype=float)
    with tempfile.TemporaryDirectory(prefix="sigma-galaxy-http-") as temporary_value:
        temporary = Path(temporary_value)
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
                "provenance": {
                    "kind": "P0639 registered baryonic map",
                    "galaxy": "DDO101",
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
        request_json(f"{base}{ticket['links']['content']}", method="PUT", payload=archive)
        extraction = request_json(
            f"{base}/api/v1/galaxy-jobs",
            method="POST",
            payload={
                "schemaVersion": "sigma-galaxy-job-submit/1",
                "operation": "extract_roundtrip",
                "dataUploadId": ticket["id"],
                "galaxy": "DDO101",
                "sourceObservables": {"stage": "HTTP smoke", "inclinationDeg": 51.0},
                "extractionControls": {
                    "radialBins": 24,
                    "maximumFourierMode": 4,
                    "residualFeatureCountPerComponent": 32,
                },
                "vertical": {"enabled": True, "realizations": 2, "zCells": 25, "seed": 101},
                "uncertaintyEnsemble": {
                    "enabled": True,
                    "realizations": 3,
                    "seed": 303,
                    "priors": {
                        "gasMassLnSigma": 0.08,
                        "stellarMassLnSigma": 0.12,
                        "distanceScaleLnSigma": 0.03,
                        "inclinationSigmaDeg": 2.0,
                        "warpSigmaDeg": 1.5,
                    },
                },
                "outputLicense": {
                    "id": "research-source-license",
                    "redistributionAllowed": False,
                },
            },
        )
        extracted = wait_job(base, extraction)
        extract_artifacts, extract_downloads = artifacts(base, extraction)
        if extracted["workerSourceSha256"] != extract_artifacts["manifest"]["worker"]["sourceSha256"]:
            raise RuntimeError("gateway and galaxy worker source hashes disagree")
        metrics = json.loads(extract_downloads["roundtrip_metrics.json"])
        parameters = json.loads(extract_downloads["parameters.json"])
        with np.load(io.BytesIO(extract_downloads["surface_density.npz"])) as surface:
            extracted_gas_mass_proxy = float(np.sum(surface["gas_surface_density"]))
        with np.load(io.BytesIO(extract_downloads["surface_density_ensemble.npz"])) as saved:
            surface_ensemble = saved["total_baryonic_surface_density"].copy()
        with np.load(io.BytesIO(extract_downloads["volume_density_ensemble.npz"])) as saved:
            volume_ensemble = saved["total_baryonic_volume_density"].copy()
        volume_bundle = json.loads(extract_downloads["volume_density_ensemble_bundle.json"])
        dz = float(volume_bundle["spatialGeometry"]["spacing"][2])
        projection_error = float(
            np.max(
                np.abs(
                    volume_ensemble.sum(axis=4) * dz
                    - np.broadcast_to(surface_ensemble[:, None], volume_ensemble.shape[:4])
                )
            )
            / np.max(surface_ensemble)
        )
        if projection_error > 1e-12:
            raise RuntimeError(f"ensemble projection mismatch: {projection_error}")
        generation = request_json(
            f"{base}/api/v1/galaxy-jobs",
            method="POST",
            payload={
                "schemaVersion": "sigma-galaxy-job-submit/1",
                "operation": "generate",
                "parameterPackage": parameters,
                "generationControls": {
                    "gas": {"massScale": 1.25, "radialScale": 0.85},
                    "stars": {"fourierScale": 0.5, "residualScale": 0.5},
                },
                "vertical": {"enabled": True, "realizations": 2, "zCells": 25, "seed": 202},
                "outputLicense": {
                    "id": "research-source-license",
                    "redistributionAllowed": False,
                },
            },
        )
        generated = wait_job(base, generation)
        generate_artifacts, generate_downloads = artifacts(base, generation)
        with np.load(io.BytesIO(generate_downloads["surface_density.npz"])) as surface:
            generated_gas_mass_proxy = float(np.sum(surface["gas_surface_density"]))
        mass_ratio = generated_gas_mass_proxy / extracted_gas_mass_proxy
        if not np.isclose(mass_ratio, 1.25, rtol=1e-12):
            raise RuntimeError(f"gas mass control did not transfer: {mass_ratio}")
        result = {
            "schemaVersion": "sigma-galaxy-job-http-smoke/1",
            "state": "pass",
            "realGalaxy": "DDO101",
            "extractOperationalJobId": extraction["id"],
            "extractScientificJobId": extracted["scientificJobId"],
            "generateOperationalJobId": generation["id"],
            "generateScientificJobId": generated["scientificJobId"],
            "extractArtifactCount": len(extract_artifacts["items"]),
            "generateArtifactCount": len(generate_artifacts["items"]),
            "allDownloadedArtifactHashesValid": True,
            "workerSourceHashAgrees": True,
            "roundtripTotalNormalizedL2": metrics["total"]["normalized_l2"],
            "roundtripTotalPixelCorrelation": metrics["total"]["pixel_correlation"],
            "surfaceEnsembleShape": list(surface_ensemble.shape),
            "volumeEnsembleShape": list(volume_ensemble.shape),
            "ensembleProjectionMaximumRelativeError": projection_error,
            "requestedGasMassScale": 1.25,
            "measuredGeneratedGasMassRatio": mass_ratio,
            "gravityParameters": 0,
            "velocityTargetsUsedForExtraction": False,
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
