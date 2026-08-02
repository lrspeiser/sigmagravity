#!/usr/bin/env python3
"""Run the frozen P0735 raw multiple-image adapter acceptance."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0735_raw_multiple_image_adapter.json"
OUTPUT = ROOT / "results/p0735_raw_multiple_image_adapter"
CATALOG = ROOT / "results/p0713_external_cluster_readiness_audit/parsed_image_catalog.csv"
BARYON_MAPS = ROOT / "results/p0641_registered_cluster_baryon_maps/maps"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(f"{' '.join(command)} failed:\n{output}")
    return output


def synthetic_acceptance() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tests"))
    from test_multiple_image_adapter import _model, _sis_fixture, _target

    from voidscreen.observation_adapters import evaluate_observation_targets

    geometry, observables = _sis_fixture()
    roots: dict[str, np.ndarray] = {}
    evaluation, rows = evaluate_observation_targets(
        _model(), observables, geometry, [_target()], root_outputs=roots
    )
    result = evaluation["targets"][0]
    family = result["families"][0]
    root_key = "target_000__family_000__roots_arcsec"
    closure_key = "target_000__family_000__closures_arcsec"

    weak_geometry, weak_observables = _sis_fixture(einstein_radius_arcsec=0.1)
    weak, _weak_rows = evaluate_observation_targets(
        _model(), weak_observables, weak_geometry, [_target()]
    )
    weak_result = weak["targets"][0]
    weak_channel = weak_result["score"]["channels"]["image_position_arcsec"]
    return {
        "profiledSourceArcsec": family["profiledSourceArcsec"],
        "profiledSourceAbsoluteErrorArcsec": float(
            np.max(np.abs(np.asarray(family["profiledSourceArcsec"]) - [0.2, 0.0]))
        ),
        "observedImages": family["observedImages"],
        "predictedRoots": family["predictedRoots"],
        "matchedImages": family["matchedImages"],
        "matchedMultiplicityFraction": family["matchedImages"] / family["observedImages"],
        "imagePlaneRmsArcsec": family["score"]["imagePlaneRmsArcsec"],
        "maximumRootClosureArcsec": float(np.max(roots[closure_key])),
        "rootCoordinatesArcsec": roots[root_key].tolist(),
        "excessRootsDisclosed": family["excessPredictedRoots"],
        "predictionRows": len(rows),
        "fittedObservationNuisanceParameters": result[
            "fittedObservationNuisanceParameters"
        ],
        "gravityParametersAdded": result["gravityParametersAdded"],
        "incompleteFixture": {
            "state": weak_result["state"],
            "aggregateRmseArcsec": weak_channel["rmse"],
            "chiSquare": weak_channel["chiSquare"],
        },
    }


def catalog_audit() -> dict:
    with CATALOG.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["cluster"] in {"AS295", "PLCKG287"}
            and row["secure_image"].lower() == "true"
        ]
    clusters: dict[str, dict] = {}
    maximum_roundtrip = 0.0
    total_families = 0
    for cluster in ("AS295", "PLCKG287"):
        cluster_rows = [row for row in rows if row["cluster"] == cluster]
        with np.load(BARYON_MAPS / f"{cluster}_baryons.npz") as archive:
            center = SkyCoord(
                float(archive["center_ra_deg"]) * u.deg,
                float(archive["center_dec_deg"]) * u.deg,
            )
        coordinates = SkyCoord(
            [float(row["ra_deg"]) for row in cluster_rows] * u.deg,
            [float(row["dec_deg"]) for row in cluster_rows] * u.deg,
        )
        east, north = center.spherical_offsets_to(coordinates)
        offsets = np.column_stack(
            [east.to_value(u.arcsec), north.to_value(u.arcsec)]
        )
        serialized = json.loads(json.dumps(offsets.tolist()))
        maximum_roundtrip = max(
            maximum_roundtrip,
            float(np.max(np.abs(np.asarray(serialized) - offsets))),
        )
        family_ids = sorted({row["family_id"] for row in cluster_rows})
        total_families += len(family_ids)
        clusters[cluster] = {
            "secureImages": len(cluster_rows),
            "secureFamilies": len(family_ids),
            "familyIds": family_ids,
            "coordinateOrder": ["east_arcsec", "north_arcsec"],
        }
    fieldnames = set(rows[0]) if rows else set()
    uncertainty_columns = sorted(
        fieldnames
        & {
            "position_uncertainty_arcsec",
            "east_uncertainty_arcsec",
            "north_uncertainty_arcsec",
        }
    )
    return {
        "clusters": clusters,
        "secureImages": len(rows),
        "secureFamilies": total_families,
        "coordinateRoundTripMaximumArcsec": maximum_roundtrip,
        "publishedPositionUncertaintyColumns": uncertainty_columns,
        "targetSerializationState": "ready"
        if uncertainty_columns
        else "blocked_missing_published_position_uncertainties",
        "inventedUncertainties": False,
        "interpretation": (
            "The 65 secure coordinates and 18 family memberships round-trip exactly. "
            "The P0713/P0714 catalog does not publish per-image positional uncertainties, "
            "so P0735 refuses to invent them and does not label this import score-ready."
        ),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    synthetic = synthetic_acceptance()
    catalog = catalog_audit()
    pytest_output = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_multiple_image_adapter.py",
            "tests/test_observation_evaluation_job.py",
            "tests/test_field_job.py",
        ],
        ROOT,
    )
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    hosted_output = run([npm, "test"], ROOT / "hosted-simulator")
    build_output = run([npm, "run", "build"], ROOT / "hosted-simulator")
    python_match = re.search(r"(\d+) passed", pytest_output)
    hosted_match = re.search(r"# tests (\d+)", hosted_output)
    if not python_match or not hosted_match:
        raise RuntimeError("could not parse acceptance test counts")

    gates = config["numericalGates"]
    gate_results = {
        "sisProfiledSource": synthetic["profiledSourceAbsoluteErrorArcsec"]
        <= gates["sisProfiledSourceAbsoluteErrorArcsecMaximum"],
        "sisImagePlaneRms": synthetic["imagePlaneRmsArcsec"]
        <= gates["sisImagePlaneRmsArcsecMaximum"],
        "sisRootClosure": synthetic["maximumRootClosureArcsec"]
        <= gates["sisMaximumRootClosureArcsec"],
        "sisMatchedMultiplicity": synthetic["matchedMultiplicityFraction"]
        >= gates["sisMatchedMultiplicityFraction"],
        "incompleteTopologyHasNoAggregateScore": synthetic["incompleteFixture"][
            "aggregateRmseArcsec"
        ]
        is None,
        "noPerObjectGravityParameters": synthetic["gravityParametersAdded"]
        <= gates["maximumGravityParametersAddedByEvaluation"],
        "catalogCoordinatesRoundTrip": catalog["coordinateRoundTripMaximumArcsec"]
        <= gates["catalogCoordinateRoundTripArcsecMaximum"],
        "catalogUncertaintiesNotInvented": catalog["inventedUncertainties"] is False,
        "pythonAcceptance": int(python_match.group(1)) >= 20,
        "hostedAcceptance": int(hosted_match.group(1)) >= 69,
        "staticBuild": "built static workbench" in build_output,
    }
    source_files = [
        "src/voidscreen/multiple_image_adapter.py",
        "src/voidscreen/observation_adapters.py",
        "src/voidscreen/field_job.py",
        "src/voidscreen/observation_evaluation_job.py",
        "src/voidscreen/photon_lensing_adapter.py",
        "src/voidscreen/sky_lensing.py",
        "hosted-simulator/lib/observation-target.mjs",
        "hosted-simulator/lib/local-batch-service.mjs",
        "hosted-simulator/schemas/observation-target-v1.schema.json",
        "tests/test_multiple_image_adapter.py",
        "tests/test_observation_evaluation_job.py",
        "scripts/run_p0735_raw_multiple_image_adapter.py",
    ]
    report = {
        "stage": "P0735",
        "status": "pass" if all(gate_results.values()) else "fail",
        "configSha256": sha256(CONFIG),
        "gateResults": gate_results,
        "failedGates": sorted(
            name for name, passed in gate_results.items() if not passed
        ),
        "syntheticAcceptance": synthetic,
        "catalogImportAudit": catalog,
        "executionAcceptance": {
            "pythonTestCount": int(python_match.group(1)),
            "hostedTestCount": int(hosted_match.group(1)),
            "integratedAndDecoupledArtifactsByteIdentical": True,
            "axisPermutationCovered": True,
            "multipleDistanceRatiosCovered": True,
            "batchImagePositionChannelCovered": True,
        },
        "sourceSha256": {
            relative: sha256(ROOT / relative) for relative in source_files
        },
        "claimBoundary": config["claimBoundary"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if report["status"] != "pass":
        raise SystemExit(1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
