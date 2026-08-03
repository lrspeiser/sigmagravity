"""Export the frozen P0743/P0744 development evidence for the hosted UI.

The hosted document is a deterministic, read-only projection of checked-in
research artifacts. It does not rerun a field solver or fit any parameter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import rfc8785


HOSTED = Path(__file__).resolve().parents[1]
PROJECT = HOSTED.parent
P0743 = PROJECT / "results/p0743_multiscale_spiral_twin_development"
P0744 = PROJECT / "results/p0744_development_velocity_field_reveal"
OUTPUT = HOSTED / "data/resolved-twin-development-v1.json"
ATLAS_OUTPUT = HOSTED / "assets/resolved-twin-development-atlas.png"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def main() -> None:
    p0743 = read_json(P0743 / "report.json")
    p0744 = read_json(P0744 / "report.json")
    if p0743["status"] != "pass" or p0743["selectedTier"] != "haar_256":
        raise RuntimeError("P0743 selected development representation is not frozen")
    if p0744["status"] != "pass" or not all(p0744["checks"].values()):
        raise RuntimeError("P0744 evidence did not pass its execution/integrity checks")

    morphology_rows = [
        row
        for row in read_csv(P0743 / "roundtrip_scores.csv")
        if row["tier"] == "haar_256" and row["component"] == "total"
    ]
    score_rows = read_csv(P0744 / "velocity_field_scores.csv")
    nuisance_rows = read_csv(P0744 / "observation_nuisance_audit.csv")
    radial_rows = read_csv(P0744 / "radial_curve_points.csv")
    morphology = {row["galaxy"]: row for row in morphology_rows}
    nuisance = {row["galaxy"]: row for row in nuisance_rows}

    model_definitions = {
        "fixed_simple_mond": {
            "label": "Fixed simple MOND",
            "family": "algebraic acceleration comparator",
            "parameters": {"a0MPerS2": 1.2e-10},
            "parameterPolicy": "published_fixed",
            "perGalaxyGravityParameters": 0,
        },
        "newtonian_thin_sheet": {
            "label": "Newtonian baryons",
            "family": "thin-sheet Poisson comparator",
            "parameters": {"gravitationalConstantM3KgS2": 6.67430e-11},
            "parameterPolicy": "published_fixed",
            "perGalaxyGravityParameters": 0,
        },
    }
    if len(morphology) != 4 or len(nuisance) != 4 or p0744["systems"] != 4:
        raise RuntimeError("resolved development evidence must contain four systems")
    systems: list[dict[str, Any]] = []
    for galaxy in sorted(nuisance):
        morph = morphology[galaxy]
        observation = nuisance[galaxy]
        models: dict[str, Any] = {}
        for model in model_definitions:
            source = next(
                row
                for row in score_rows
                if row["galaxy"] == galaxy
                and row["model"] == model
                and row["map_kind"] == "registered_baryons"
            )
            twin = next(
                row
                for row in score_rows
                if row["galaxy"] == galaxy
                and row["model"] == model
                and row["map_kind"] == "fake_twin"
            )
            radial = [
                {
                    "radiusKpc": number(row, "radius_kpc"),
                    "observedKmS": number(row, "observed_rotation_km_s"),
                    "sourcePredictionKmS": number(row, "source_prediction_km_s"),
                    "twinPredictionKmS": number(row, "twin_prediction_km_s"),
                    "pixels": int(row["pixels"]),
                }
                for row in radial_rows
                if row["galaxy"] == galaxy and row["model"] == model
            ]
            if len(radial) != 20:
                raise RuntimeError(f"{galaxy}/{model} does not have 20 radial bins")
            models[model] = {
                "sourceVersusObserved": {
                    "rmseKmS": number(source, "gas_weighted_rmse_km_s"),
                    "uncertaintyRmsKmS": number(
                        source, "gas_weighted_uncertainty_rms_km_s"
                    ),
                    "errorRatio": number(source, "field_error_ratio"),
                    "classification": source["error_band"],
                },
                "twinVersusObserved": {
                    "rmseKmS": number(twin, "gas_weighted_rmse_km_s"),
                    "uncertaintyRmsKmS": number(
                        twin, "gas_weighted_uncertainty_rms_km_s"
                    ),
                    "errorRatio": number(twin, "field_error_ratio"),
                    "classification": twin["error_band"],
                },
                "sourceToTwinTransport": {
                    "lineOfSightRmseKmS": number(
                        twin, "twin_source_transport_rmse_km_s"
                    )
                },
                "radialCurve": radial,
            }

        systems.append(
            {
                "id": galaxy,
                "split": "development",
                "observation": {
                    "tracer": "THINGS H I moment-1 velocity field",
                    "scoredPixels": int(observation["scored_pixels"]),
                    "inclinationDeg": number(observation, "inclination_deg"),
                    "positionAngleDeg": number(observation, "position_angle_deg"),
                    "medianDispersionKmS": number(
                        observation, "median_dispersion_km_s"
                    ),
                },
                "twinFidelity": {
                    "totalMapNormalizedL2": number(morph, "normalized_l2"),
                    "totalMapPixelCorrelation": number(morph, "pixel_correlation"),
                    "massRelativeError": number(morph, "mass_relative_error"),
                    "velocityTargetsUsedInExtraction": False,
                    "gravityParametersUsedInExtraction": 0,
                },
                "models": models,
            }
        )

    atlas_source = P0744 / "velocity_field_comparison_atlas.png"
    core = {
        "schemaVersion": "sigma-hosted-resolved-twin-evidence/1",
        "stage": "P0744",
        "evidenceClass": "precomputed_development_result",
        "title": "Resolved spiral twins versus observed velocity fields",
        "parents": {
            "p0743ReportSha256": p0743["reportSha256"],
            "p0744ReportSha256": p0744["reportSha256"],
        },
        "sample": {
            "systems": len(systems),
            "scoredVelocityPixels": sum(
                system["observation"]["scoredPixels"] for system in systems
            ),
            "developmentSystems": [system["id"] for system in systems],
            "sealedValidationSystems": ["NGC3521", "NGC6946"],
            "sealedHoldoutSystems": ["NGC2841", "NGC7331"],
        },
        "generator": {
            "id": "sparse-orthonormal-haar-2d",
            "selectedDevelopmentTier": "haar_256",
            "coefficientsPerBaryonicComponent": 256,
            "gravityParameters": 0,
            "velocityTargetsUsed": False,
        },
        "models": model_definitions,
        "systems": systems,
        "artifact": {
            "atlasPath": "/assets/resolved-twin-development-atlas.png",
            "atlasSha256": sha256(atlas_source),
        },
        "scoreSemantics": {
            "twinFidelity": "How closely the generated baryonic map matches the registered real baryonic map.",
            "formulaTransport": "How much a fixed formula prediction changes when the registered map is replaced by its fake twin.",
            "observationalAccuracy": "How far the formula prediction on the fake twin is from the observed H I velocity field.",
            "classification": {
                "consistent": "field RMSE divided by declared uncertainty RMS is at most 1",
                "close": "ratio is greater than 1 and at most 2",
                "miss": "ratio is greater than 2",
            },
        },
        "executionBoundary": {
            "formulaExecution": "precomputed fixed comparators only",
            "arbitraryHosted2dFormulaExecution": False,
            "individualStarDynamics": False,
            "validationStatus": "development; not blind validation",
            "protocolStatusMeaning": "The frozen comparison executed with finite leakage-audited scores; it does not mean either formula passed every galaxy.",
        },
        "claimBoundary": p0744["claimBoundary"],
    }
    document = {**core, "evidenceSha256": canonical_hash(core)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(atlas_source, ATLAS_OUTPUT)
    print(
        json.dumps(
            {
                "systems": len(systems),
                "scoredVelocityPixels": core["sample"]["scoredVelocityPixels"],
                "evidenceSha256": document["evidenceSha256"],
                "atlasSha256": core["artifact"]["atlasSha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
