#!/usr/bin/env python3
"""Run four formula manifests through the generic API on 13 resolved galaxies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from smoke_local_batch_api import (
    download_artifacts,
    request_json,
    wait,
)

sys.path.insert(0, str(ROOT / "src"))

from voidscreen.field_job import write_array_bundle

KPC_M = 3.085677581491367e19
DEFAULT_CONFIG = ROOT / "configs" / "p0723_formula_neutral_api_comparators.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0723_formula_neutral_api_comparators"
KINEMATICS = (
    ROOT
    / "data"
    / "raw"
    / "p0633_little_things_kinematics"
    / "stw3285_Supplementary_Data.zip"
)
FROZEN_CURVES = (
    ROOT / "results" / "p0708_external_prediction_lock" / "galaxy_prediction_curves.csv"
)
TABLE_NAMES = {
    "CVnIdwA": "cvidwa",
    "DDO47": "ddo47",
    "DDO50": "ddo50",
    "DDO52": "ddo52",
    "DDO53": "ddo53",
    "DDO87": "ddo87",
    "DDO101": "ddo101",
    "DDO126": "ddo126",
    "DDO133": "ddo133",
    "DDO210": "ddo210",
    "DDO216": "ddo216",
    "NGC1569": "ngc1569",
    "UGC8508": "ugc8508",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_numeric_rows(text: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in text.splitlines():
        tokens = line.strip().split()
        if len(tokens) != 12:
            continue
        try:
            rows.append([float(value) for value in tokens])
        except ValueError:
            continue
    return rows


def observation_targets(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    archive_sha = sha256(KINEMATICS)
    targets: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(KINEMATICS) as outer:
        nested = outer.read("results.zip")
    with zipfile.ZipFile(io.BytesIO(nested)) as archive:
        for galaxy in config["systems"]:
            member = f"finalrot/{TABLE_NAMES[galaxy]}_onlinetab.txt"
            rows = parse_numeric_rows(
                archive.read(member).decode("utf-8", errors="replace")
            )
            if len(rows) < 3:
                raise RuntimeError(f"fewer than three circular-speed rows for {galaxy}")
            controls = config["observation"]
            targets[galaxy] = {
                "schemaVersion": "sigma-observation-target/1",
                "id": f"{galaxy}-published-circular-speed",
                "kind": "circular_speed_curve",
                "observable": "massive_tracer_acceleration",
                "centerM": [0.0, 0.0, 0.0],
                "planeAxes": [0, 1],
                "radiiM": [row[1] * KPC_M for row in rows],
                "observedSpeedsMPerS": [row[6] * 1000.0 for row in rows],
                "uncertaintiesMPerS": [row[7] * 1000.0 for row in rows],
                "azimuthalSamples": int(controls["azimuthalSamples"]),
                "minimumAzimuthalCoverage": float(
                    controls["minimumAzimuthalCoverage"]
                ),
                "fittedNuisanceParameters": int(
                    controls["fittedNuisanceParameters"]
                ),
                "provenance": {
                    "kind": "published LITTLE THINGS circular-speed table",
                    "archiveSha256": archive_sha,
                    "member": f"results.zip/{member}",
                },
                "license": {
                    "id": "published-supplementary-material",
                    "redistributionAllowed": False,
                },
            }
    return targets


def upload_registered_map(base: str, temporary: Path, galaxy: str) -> dict[str, Any]:
    source_path = ROOT / "results" / "p0639_registered_baryonic_maps" / "maps" / f"{galaxy}.npz"
    with np.load(source_path) as source:
        axis = np.asarray(source["axis_kpc"], dtype=float)
        gas = np.asarray(source["gas"], dtype=float)
        stars = np.asarray(source["stars"], dtype=float)
    directory = temporary / galaxy
    bundle = write_array_bundle(
        directory,
        {"gas": gas, "stars": stars},
        {
            "schemaVersion": "sigma-array-bundle-request/1",
            "geometry": {
                "coordinateSystem": "cartesian_2d",
                "dimensions": 2,
                "spacing": [float(axis[1] - axis[0])] * 2,
                "origin": [float(axis[0]), float(axis[0])],
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
                "galaxy": galaxy,
                "sourceSha256": sha256(source_path),
            },
            "license": {
                "id": "research-source-license",
                "redistributionAllowed": False,
            },
        },
    )
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


def submit_galaxy_job(
    base: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bytes]]:
    submission = request_json(
        f"{base}/api/v1/galaxy-jobs", method="POST", payload=payload
    )
    completed = wait(base, submission, timeout=1800.0)
    _index, artifacts = download_artifacts(base, submission)
    return completed, artifacts


def build_generated_systems(
    base: str, temporary: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    systems: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    generation = config["generation"]
    for index, galaxy in enumerate(config["systems"]):
        upload = upload_registered_map(base, temporary, galaxy)
        extract, extract_artifacts = submit_galaxy_job(
            base,
            {
                "schemaVersion": "sigma-galaxy-job-submit/1",
                "operation": "extract_roundtrip",
                "galaxy": galaxy,
                "dataUploadId": upload["id"],
                "extractionControls": config["extraction"],
                "vertical": {
                    "enabled": False,
                    "realizations": 1,
                    "zCells": int(generation["zCells"]),
                    "seed": int(generation["seed"]) + index * 2,
                },
                "outputLicense": {
                    "id": "research-source-license",
                    "redistributionAllowed": False,
                },
            },
        )
        parameters = json.loads(extract_artifacts["parameters.json"])
        generated, _generated_artifacts = submit_galaxy_job(
            base,
            {
                "schemaVersion": "sigma-galaxy-job-submit/1",
                "operation": "generate",
                "galaxy": galaxy,
                "parameterPackage": parameters,
                "generationControls": {},
                "outputGrid": {"cellsPerAxis": int(generation["cellsPerAxis"])},
                "vertical": {
                    "enabled": True,
                    "realizations": int(generation["verticalRealizations"]),
                    "zCells": int(generation["zCells"]),
                    "seed": int(generation["seed"]) + index * 2 + 1,
                },
                "outputLicense": {
                    "id": "research-source-license",
                    "redistributionAllowed": False,
                },
            },
        )
        systems.append(
            {
                "id": galaxy,
                "galaxyJobId": generated["id"],
                "galaxyArtifact": "field_volume_density",
            }
        )
        jobs.append(
            {
                "galaxy": galaxy,
                "mapUploadId": upload["id"],
                "extractJobId": extract["id"],
                "extractScientificResultSha256": extract["scientificResultSha256"],
                "generateJobId": generated["id"],
                "generateScientificResultSha256": generated[
                    "scientificResultSha256"
                ],
                "parameterPackageSha256": parameters["contentSha256"],
            }
        )
    return systems, jobs


def read_csv(content: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(content.decode("utf-8"))))


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def frozen_curves() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    with FROZEN_CURVES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["system"], row["model"])].append(
                (float(row["radius_kpc"]), float(row["circular_speed_km_s"]))
            )
    result = {}
    for key, values in grouped.items():
        ordered = sorted(values)
        result[key] = (
            np.asarray([value[0] for value in ordered]),
            np.asarray([value[1] for value in ordered]),
        )
    return result


def conformance_to_frozen(
    predictions: list[dict[str, Any]], model_name: str
) -> dict[str, Any]:
    reference = frozen_curves()
    per_galaxy = []
    for galaxy in sorted({str(row["system_id"]) for row in predictions}):
        key = (galaxy, model_name)
        if key not in reference:
            continue
        rows = [
            row
            for row in predictions
            if row["system_id"] == galaxy
            and finite_float(row["predicted_speed_m_s"]) is not None
        ]
        radius = np.asarray([float(row["radius_m"]) / KPC_M for row in rows])
        predicted = np.asarray(
            [float(row["predicted_speed_m_s"]) / 1000.0 for row in rows]
        )
        source_radius, source_speed = reference[key]
        within = (radius >= source_radius.min()) & (radius <= source_radius.max())
        if not np.any(within):
            continue
        expected = np.interp(radius[within], source_radius, source_speed)
        residual = predicted[within] - expected
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        normalization = float(np.sqrt(np.mean(np.square(expected))))
        per_galaxy.append(
            {
                "galaxy": galaxy,
                "points": int(np.sum(within)),
                "rmseKmS": rmse,
                "normalizedRmse": rmse / normalization,
            }
        )
    return {
        "frozenCurveModel": model_name,
        "galaxies": len(per_galaxy),
        "equalGalaxyRmseKmS": float(
            np.sqrt(np.mean([value["rmseKmS"] ** 2 for value in per_galaxy]))
        )
        if per_galaxy
        else None,
        "equalGalaxyNormalizedRmse": float(
            np.sqrt(
                np.mean([value["normalizedRmse"] ** 2 for value in per_galaxy])
            )
        )
        if per_galaxy
        else None,
        "perGalaxy": per_galaxy,
    }


def run_model_batch(
    base: str,
    specification: dict[str, Any],
    systems: list[dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    *,
    timeout: float = 1800.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = ROOT / specification["manifest"]
    model = json.loads(manifest_path.read_text(encoding="utf-8"))
    submitted_systems = [
        {**system, "observationTargets": [targets[system["id"]]]}
        for system in systems
    ]
    submission = request_json(
        f"{base}/api/v1/batches",
        method="POST",
        payload={
            "schemaVersion": "sigma-batch-submit/1",
            "model": model,
            "systems": submitted_systems,
            "fieldRequest": {
                "schemaVersion": "sigma-field-job-request/1",
                "requestedObservables": ["massive_tracer_acceleration"],
                "seed": 20260803,
            },
            "parameterPolicy": {
                "mode": "published_fixed",
                "perObjectParameters": [],
            },
        },
    )
    completed = wait(base, submission, timeout=timeout)
    artifact_index, artifacts = download_artifacts(base, submission)
    aggregate = json.loads(artifacts["aggregate_scores.json"])
    per_galaxy = read_csv(artifacts["per_galaxy.csv"])
    predictions = read_csv(artifacts["observation_predictions.csv"])
    valid_rmse = [
        float(row["observation_rmse_m_s"]) / 1000.0
        for row in per_galaxy
        if finite_float(row.get("observation_rmse_m_s")) is not None
    ]
    conformance = (
        conformance_to_frozen(predictions, specification["frozenCurveModel"])
        if specification["frozenCurveModel"]
        else None
    )
    summary = {
        "model": specification["id"],
        "manifestPath": specification["manifest"],
        "manifestSha256": sha256(manifest_path),
        "batchId": completed["id"],
        "batchState": completed["state"],
        "batchScientificResultSha256": completed.get("scientificResultSha256"),
        "modelSha256": aggregate["modelSha256"],
        "systems": aggregate["systemCount"],
        "succeededSystems": aggregate["succeededSystems"],
        "convergenceFraction": aggregate["convergenceFraction"],
        "maximumEquationResidual": aggregate["maximumEquationResidual"],
        "scoredSystems": len(valid_rmse),
        "validObservationPoints": aggregate["validObservationPoints"],
        "equalGalaxyRmseKmS": float(np.sqrt(np.mean(np.square(valid_rmse))))
        if valid_rmse
        else None,
        "pointWeightedRmseKmS": aggregate["observationRmseMPerS"] / 1000.0
        if aggregate["observationRmseMPerS"] is not None
        else None,
        "reducedChiSquare": aggregate["observationReducedChiSquare"],
        "universalGravityParameters": aggregate["universalGravityParameters"],
        "perObjectGravityParameters": aggregate["perObjectGravityParameters"],
        "downloadedArtifactCount": len(artifact_index["items"]),
        "allDownloadedArtifactHashesValid": True,
        "frozenCurveConformance": conformance,
    }
    return summary, per_galaxy, predictions


def render_plots(
    output: Path,
    targets: dict[str, dict[str, Any]],
    model_summaries: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    axis.bar(
        [row["model"] for row in model_summaries],
        [row["equalGalaxyRmseKmS"] for row in model_summaries],
        color=["#7f8c8d", "#2980b9", "#27ae60", "#c0392b"],
    )
    axis.set_ylabel("Equal-galaxy circular-speed RMSE (km/s)")
    axis.set_title("P0723 generic API: one published-fixed manifest per batch")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "model_score_comparison.png", dpi=180)
    plt.close(figure)

    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_key[(str(row["system_id"]), str(row["model"]))].append(row)
    figure, axes = plt.subplots(4, 4, figsize=(16, 14))
    colors = ["#7f8c8d", "#2980b9", "#27ae60", "#c0392b"]
    for axis, galaxy in zip(axes.ravel(), sorted(targets), strict=False):
        target = targets[galaxy]
        axis.errorbar(
            np.asarray(target["radiiM"]) / KPC_M,
            np.asarray(target["observedSpeedsMPerS"]) / 1000.0,
            yerr=np.asarray(target["uncertaintiesMPerS"]) / 1000.0,
            fmt="o",
            color="black",
            ms=3,
            label="published",
        )
        for color, summary in zip(colors, model_summaries, strict=True):
            rows = sorted(
                by_key[(galaxy, summary["model"])], key=lambda value: int(value["point_index"])
            )
            radius = [float(row["radius_m"]) / KPC_M for row in rows]
            speed = [
                finite_float(row["predicted_speed_m_s"])
                / 1000.0
                if finite_float(row["predicted_speed_m_s"]) is not None
                else np.nan
                for row in rows
            ]
            axis.plot(radius, speed, "-o", ms=2.5, lw=1.2, color=color, label=summary["model"])
        axis.set_title(galaxy)
        axis.set_xlabel("R (kpc)")
        axis.set_ylabel("speed (km/s)")
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(targets) :]:
        axis.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.suptitle("Formula-neutral HTTP predictions on resolved galaxy replicas")
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    figure.savefig(output / "rotation_curve_atlas.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SIMULATOR_BASE_URL", "http://127.0.0.1:4173"),
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = observation_targets(config)
    all_per_galaxy: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sigma-p0723-") as temporary_value:
        systems, galaxy_jobs = build_generated_systems(
            args.base_url.rstrip("/"), Path(temporary_value), config
        )
        for specification in config["models"]:
            summary, per_galaxy, predictions = run_model_batch(
                args.base_url.rstrip("/"), specification, systems, targets
            )
            model_summaries.append(summary)
            all_per_galaxy.extend(
                [{"model": specification["id"], **row} for row in per_galaxy]
            )
            all_predictions.extend(
                [{"model": specification["id"], **row} for row in predictions]
            )

    gates = config["engineeringGates"]
    common_scored = set(config["systems"])
    for summary in model_summaries:
        model_rows = [
            row
            for row in all_per_galaxy
            if row["model"] == summary["model"]
            and finite_float(row.get("observation_rmse_m_s")) is not None
        ]
        common_scored &= {str(row["system_id"]) for row in model_rows}
    conformance_values = [
        summary["frozenCurveConformance"]["equalGalaxyNormalizedRmse"]
        for summary in model_summaries
        if summary["frozenCurveConformance"] is not None
    ]
    gate_results = {
        "required_systems": len(config["systems"]) == int(gates["requiredSystems"]),
        "minimum_scored_systems_per_model": all(
            summary["scoredSystems"] >= int(gates["minimumScoredSystemsPerModel"])
            for summary in model_summaries
        ),
        "minimum_common_scored_systems": len(common_scored)
        >= int(gates["minimumCommonScoredSystems"]),
        "all_batches_succeeded": all(
            summary["batchState"] == "succeeded" for summary in model_summaries
        ),
        "convergence_fraction": all(
            summary["convergenceFraction"]
            >= float(gates["minimumConvergenceFraction"])
            for summary in model_summaries
        ),
        "equation_residual": all(
            summary["maximumEquationResidual"]
            <= float(gates["maximumEquationResidual"])
            for summary in model_summaries
        ),
        "frozen_curve_conformance": all(
            value <= float(gates["maximumFrozenCurveNormalizedRmse"])
            for value in conformance_values
        ),
        "no_per_object_gravity_parameters": all(
            summary["perObjectGravityParameters"]
            <= int(gates["maximumPerObjectGravityParameters"])
            for summary in model_summaries
        ),
        "artifact_hashes": all(
            summary["allDownloadedArtifactHashesValid"] for summary in model_summaries
        ),
    }
    report = {
        "schemaVersion": "sigma-p0723-formula-neutral-api-comparators/1",
        "stage": config["stage"],
        "status": "pass" if all(gate_results.values()) else "fail",
        "sampleStatus": config["sampleStatus"],
        "systems": len(config["systems"]),
        "models": len(model_summaries),
        "commonScoredSystems": sorted(common_scored),
        "modelSummaries": model_summaries,
        "gateResults": gate_results,
        "failedGates": [name for name, value in gate_results.items() if not value],
        "configSha256": sha256(config_path),
        "kinematicsArchiveSha256": sha256(KINEMATICS),
        "frozenCurvesSha256": sha256(FROZEN_CURVES),
        "galaxyJobs": galaxy_jobs,
        "claimBoundary": config["claimBoundary"],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        output / "model_summary.csv",
        [
            {
                **{key: value for key, value in summary.items() if key != "frozenCurveConformance"},
                "frozenCurveEqualGalaxyRmseKmS": (
                    summary["frozenCurveConformance"]["equalGalaxyRmseKmS"]
                    if summary["frozenCurveConformance"]
                    else None
                ),
                "frozenCurveEqualGalaxyNormalizedRmse": (
                    summary["frozenCurveConformance"]["equalGalaxyNormalizedRmse"]
                    if summary["frozenCurveConformance"]
                    else None
                ),
            }
            for summary in model_summaries
        ],
    )
    write_csv(output / "per_galaxy_scores.csv", all_per_galaxy)
    write_csv(output / "point_predictions.csv", all_predictions)
    render_plots(output, targets, model_summaries, all_predictions)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
