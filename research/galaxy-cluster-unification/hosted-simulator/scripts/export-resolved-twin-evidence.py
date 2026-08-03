"""Export frozen development and validation resolved-twin evidence for the hosted UI.

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
P0745C = PROJECT / "results/p0745c_validation_multiscale_spiral_twins"
P0746 = PROJECT / "results/p0746_validation_velocity_field_reveal"
P0747 = PROJECT / "results/p0747_post_reveal_kinematic_axis_diagnostic"
OUTPUT = HOSTED / "data/resolved-twin-development-v1.json"
ATLAS_OUTPUT = HOSTED / "assets/resolved-twin-development-atlas.png"
VALIDATION_ATLAS_OUTPUT = HOSTED / "assets/resolved-twin-validation-atlas.png"
GEOMETRY_ATLAS_OUTPUT = HOSTED / "assets/resolved-twin-geometry-diagnostic-atlas.png"


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
    p0745c = read_json(P0745C / "report.json")
    p0746 = read_json(P0746 / "report.json")
    p0747 = read_json(P0747 / "report.json")
    if p0743["status"] != "pass" or p0743["selectedTier"] != "haar_256":
        raise RuntimeError("P0743 selected development representation is not frozen")
    if p0744["status"] != "pass" or not all(p0744["checks"].values()):
        raise RuntimeError("P0744 evidence did not pass its execution/integrity checks")
    if p0745c["status"] != "fail" or p0745c["selectedTier"] is not None:
        raise RuntimeError("P0745C validation-twin failure is not preserved")
    if p0746["status"] != "fail" or p0746["checks"]["maximumTwinSourcePredictionTransportRmseKmS"]:
        raise RuntimeError("P0746 frozen validation result is not preserved")
    if p0747["status"] != "pass" or not all(p0747["checks"].values()):
        raise RuntimeError("P0747 geometry diagnostic did not pass its integrity checks")

    morphology_rows = [
        row
        for row in read_csv(P0743 / "roundtrip_scores.csv")
        if row["tier"] == "haar_256" and row["component"] == "total"
    ]
    morphology_rows += [
        row
        for row in read_csv(P0745C / "roundtrip_scores.csv")
        if row["tier"] == "haar_256" and row["component"] == "total"
    ]
    score_rows = read_csv(P0744 / "velocity_field_scores.csv") + read_csv(P0746 / "velocity_field_scores.csv")
    nuisance_rows = read_csv(P0744 / "observation_nuisance_audit.csv") + read_csv(P0746 / "observation_nuisance_audit.csv")
    radial_rows = read_csv(P0744 / "radial_curve_points.csv") + read_csv(P0746 / "radial_curve_points.csv")
    geometry_rows = read_csv(P0747 / "kinematic_axis_audit.csv")
    geometry_score_rows = read_csv(P0747 / "diagnostic_velocity_field_scores.csv")
    split_by_galaxy = {
        **{galaxy: "development" for galaxy in ("NGC2403", "NGC3198", "NGC5055", "NGC7793")},
        **{galaxy: "validation" for galaxy in ("NGC3521", "NGC6946")},
    }
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
    if len(morphology) != 6 or len(nuisance) != 6 or p0744["systems"] != 4 or p0746["systems"] != 2:
        raise RuntimeError("resolved evidence must contain four development and two validation systems")
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

        system_document = {
                "id": galaxy,
                "split": split_by_galaxy[galaxy],
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
        if split_by_galaxy[galaxy] == "validation":
            geometry = next(row for row in geometry_rows if row["galaxy"] == galaxy)
            adjusted_models: dict[str, Any] = {}
            for model in model_definitions:
                adjusted_source = next(
                    row for row in geometry_score_rows
                    if row["galaxy"] == galaxy and row["model"] == model
                    and row["prediction_kind"] == "registered_baryons_kinematic_axis"
                )
                adjusted_twin = next(
                    row for row in geometry_score_rows
                    if row["galaxy"] == galaxy and row["model"] == model
                    and row["prediction_kind"] == "fake_twin_kinematic_axis"
                )
                adjusted_models[model] = {
                    "sourceVersusObserved": {
                        "rmseKmS": number(adjusted_source, "gas_weighted_rmse_km_s"),
                        "uncertaintyRmsKmS": number(adjusted_source, "gas_weighted_uncertainty_rms_km_s"),
                        "errorRatio": number(adjusted_source, "field_error_ratio"),
                        "classification": adjusted_source["error_band"],
                    },
                    "twinVersusObserved": {
                        "rmseKmS": number(adjusted_twin, "gas_weighted_rmse_km_s"),
                        "uncertaintyRmsKmS": number(adjusted_twin, "gas_weighted_uncertainty_rms_km_s"),
                        "errorRatio": number(adjusted_twin, "field_error_ratio"),
                        "classification": adjusted_twin["error_band"],
                    },
                }
            system_document["geometryDiagnostic"] = {
                "status": "post_reveal_not_blind_validation",
                "axisOffsetDeg": number(geometry, "kinematic_phase_offset_deg_in_registered_plane"),
                "firstHarmonicExplainedVarianceFraction": number(
                    geometry, "first_harmonic_explained_variance_fraction"
                ),
                "fittedObservationNuisances": 1,
                "fittedVelocityAmplitudes": 0,
                "fittedGravityParameters": 0,
                "models": adjusted_models,
            }
        systems.append(system_document)

    atlas_source = P0744 / "velocity_field_comparison_atlas.png"
    validation_atlas_source = P0746 / "velocity_field_comparison_atlas.png"
    geometry_atlas_source = P0747 / "kinematic_axis_diagnostic_atlas.png"
    core = {
        "schemaVersion": "sigma-hosted-resolved-twin-evidence/1",
        "stage": "P0747",
        "evidenceClass": "precomputed_development_and_validation_result",
        "title": "Resolved spiral twins versus observed development and validation velocity fields",
        "parents": {
            "p0743ReportSha256": p0743["reportSha256"],
            "p0744ReportSha256": p0744["reportSha256"],
            "p0745cReportSha256": p0745c["reportSha256"],
            "p0746ReportSha256": p0746["reportSha256"],
            "p0747ReportSha256": p0747["reportSha256"],
        },
        "sample": {
            "systems": len(systems),
            "scoredVelocityPixels": sum(
                system["observation"]["scoredPixels"] for system in systems
            ),
            "developmentSystems": [system["id"] for system in systems if system["split"] == "development"],
            "validationSystems": [system["id"] for system in systems if system["split"] == "validation"],
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
        "validationArtifacts": {
            "rawValidationAtlasPath": "/assets/resolved-twin-validation-atlas.png",
            "rawValidationAtlasSha256": sha256(validation_atlas_source),
            "geometryDiagnosticAtlasPath": "/assets/resolved-twin-geometry-diagnostic-atlas.png",
            "geometryDiagnosticAtlasSha256": sha256(geometry_atlas_source),
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
            "validationStatus": "four development systems plus two frozen-method validation systems; holdout remains sealed",
            "protocolStatusMeaning": "The frozen comparison executed with finite leakage-audited scores; it does not mean either formula passed every galaxy.",
        },
        "claimBoundary": p0746["claimBoundary"] + p0747["claimBoundary"],
    }
    document = {**core, "evidenceSha256": canonical_hash(core)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(atlas_source, ATLAS_OUTPUT)
    shutil.copyfile(validation_atlas_source, VALIDATION_ATLAS_OUTPUT)
    shutil.copyfile(geometry_atlas_source, GEOMETRY_ATLAS_OUTPUT)
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
