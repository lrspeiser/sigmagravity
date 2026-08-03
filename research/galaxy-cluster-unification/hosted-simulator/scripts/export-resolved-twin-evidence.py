"""Export checked-in resolved development, validation, and final holdout evidence."""

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
P0749 = PROJECT / "results/p0749_adaptive_haar_policy_development"
P0750 = PROJECT / "results/p0750_adaptive_twin_kinematic_axis_development"
P0751C = PROJECT / "results/p0751c_holdout_adaptive_haar_twins"
P0752 = PROJECT / "results/p0752_final_holdout_velocity_field_test"
OUTPUT = HOSTED / "data/resolved-twin-development-v1.json"
ASSETS = {
    "development": (
        P0744 / "velocity_field_comparison_atlas.png",
        HOSTED / "assets/resolved-twin-development-atlas.png",
    ),
    "validation": (
        P0746 / "velocity_field_comparison_atlas.png",
        HOSTED / "assets/resolved-twin-validation-atlas.png",
    ),
    "geometry": (
        P0750 / "adaptive_twin_kinematic_axis_atlas.png",
        HOSTED / "assets/resolved-twin-geometry-diagnostic-atlas.png",
    ),
    "holdout": (
        P0752 / "final_holdout_velocity_field_atlas.png",
        HOSTED / "assets/resolved-twin-holdout-atlas.png",
    ),
    "holdoutCurves": (
        P0752 / "final_holdout_radial_speed_curves.png",
        HOSTED / "assets/resolved-twin-holdout-curves.png",
    ),
}


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


def metric(row: dict[str, str]) -> dict[str, Any]:
    return {
        "rmseKmS": number(row, "gas_weighted_rmse_km_s"),
        "uncertaintyRmsKmS": number(row, "gas_weighted_uncertainty_rms_km_s"),
        "errorRatio": number(row, "field_error_ratio"),
        "classification": row["error_band"],
    }


def radial_curve(rows: list[dict[str, str]], galaxy: str, model: str) -> list[dict[str, Any]]:
    radial = [
        {
            "radiusKpc": number(row, "radius_kpc"),
            "observedKmS": number(row, "observed_rotation_km_s"),
            "sourcePredictionKmS": number(row, "source_prediction_km_s"),
            "twinPredictionKmS": number(row, "twin_prediction_km_s"),
            "pixels": int(row["pixels"]),
        }
        for row in rows
        if row["galaxy"] == galaxy and row["model"] == model
    ]
    if len(radial) != 20:
        raise RuntimeError(f"{galaxy}/{model} does not have 20 radial bins")
    return radial


def main() -> None:
    reports = {
        "p0743": read_json(P0743 / "report.json"),
        "p0744": read_json(P0744 / "report.json"),
        "p0745c": read_json(P0745C / "report.json"),
        "p0746": read_json(P0746 / "report.json"),
        "p0747": read_json(P0747 / "report.json"),
        "p0749": read_json(P0749 / "report.json"),
        "p0750": read_json(P0750 / "report.json"),
        "p0751c": read_json(P0751C / "report.json"),
        "p0752": read_json(P0752 / "report.json"),
    }
    if reports["p0743"]["status"] != "pass" or reports["p0743"]["selectedTier"] != "haar_256":
        raise RuntimeError("P0743 development representation is not frozen")
    if reports["p0744"]["status"] != "pass" or not all(reports["p0744"]["checks"].values()):
        raise RuntimeError("P0744 development evidence failed integrity checks")
    if reports["p0745c"]["status"] != "fail" or reports["p0745c"]["selectedTier"] is not None:
        raise RuntimeError("P0745C validation-twin failure is not preserved")
    if reports["p0746"]["status"] != "fail" or reports["p0746"]["checks"]["maximumTwinSourcePredictionTransportRmseKmS"]:
        raise RuntimeError("P0746 frozen validation result is not preserved")
    for stage in ("p0747", "p0749", "p0750", "p0751c", "p0752"):
        if reports[stage]["status"] != "pass" or not all(reports[stage]["checks"].values()):
            raise RuntimeError(f"{stage.upper()} did not pass its execution/integrity checks")

    legacy_morphology = [
        row for row in read_csv(P0743 / "roundtrip_scores.csv")
        if row["tier"] == "haar_256" and row["component"] == "total"
    ] + [
        row for row in read_csv(P0745C / "roundtrip_scores.csv")
        if row["tier"] == "haar_256" and row["component"] == "total"
    ]
    holdout_catalog = {
        row["galaxy"]: row for row in read_csv(P0751C / "selected_parameter_catalog.csv")
    }
    holdout_morphology = [
        row for row in read_csv(P0751C / "candidate_roundtrip_scores.csv")
        if row["component"] == "total"
        and row["tier"] == holdout_catalog[row["galaxy"]]["selected_tier"]
    ]
    morphology = {row["galaxy"]: row for row in legacy_morphology + holdout_morphology}
    legacy_scores = read_csv(P0744 / "velocity_field_scores.csv") + read_csv(P0746 / "velocity_field_scores.csv")
    holdout_scores = read_csv(P0752 / "holdout_velocity_field_scores.csv")
    legacy_nuisance = {
        row["galaxy"]: row
        for row in read_csv(P0744 / "observation_nuisance_audit.csv")
        + read_csv(P0746 / "observation_nuisance_audit.csv")
    }
    holdout_nuisance = {
        row["galaxy"]: row for row in read_csv(P0752 / "holdout_observation_nuisance_audit.csv")
    }
    legacy_radial = read_csv(P0744 / "radial_curve_points.csv") + read_csv(P0746 / "radial_curve_points.csv")
    holdout_radial = read_csv(P0752 / "holdout_radial_speed_points.csv")
    development_geometry = {
        row["galaxy"]: row for row in read_csv(P0750 / "kinematic_axis_policy_audit.csv")
    }
    development_geometry_scores = read_csv(P0750 / "adaptive_twin_velocity_field_scores.csv")
    split_by_galaxy = {
        **{galaxy: "development" for galaxy in ("NGC2403", "NGC3198", "NGC5055", "NGC7793")},
        **{galaxy: "validation" for galaxy in ("NGC3521", "NGC6946")},
        **{galaxy: "holdout" for galaxy in ("NGC2841", "NGC7331")},
    }
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

    systems: list[dict[str, Any]] = []
    for galaxy in sorted(split_by_galaxy):
        split = split_by_galaxy[galaxy]
        morph = morphology[galaxy]
        observation = holdout_nuisance[galaxy] if split == "holdout" else legacy_nuisance[galaxy]
        models: dict[str, Any] = {}
        for model in model_definitions:
            if split == "holdout":
                source = next(
                    row for row in holdout_scores
                    if row["galaxy"] == galaxy and row["model"] == model
                    and row["prediction_kind"] == "registered_baryons_kinematic_axis"
                )
                twin = next(
                    row for row in holdout_scores
                    if row["galaxy"] == galaxy and row["model"] == model
                    and row["prediction_kind"] == "adaptive_twin_kinematic_axis"
                )
                transport_key = "twin_source_transport_rmse_km_s"
                curve = radial_curve(holdout_radial, galaxy, model)
            else:
                source = next(
                    row for row in legacy_scores
                    if row["galaxy"] == galaxy and row["model"] == model
                    and row["map_kind"] == "registered_baryons"
                )
                twin = next(
                    row for row in legacy_scores
                    if row["galaxy"] == galaxy and row["model"] == model
                    and row["map_kind"] == "fake_twin"
                )
                transport_key = "twin_source_transport_rmse_km_s"
                curve = radial_curve(legacy_radial, galaxy, model)
            models[model] = {
                "sourceVersusObserved": metric(source),
                "twinVersusObserved": metric(twin),
                "sourceToTwinTransport": {
                    "lineOfSightRmseKmS": number(twin, transport_key)
                },
                "radialCurve": curve,
            }

        position_key = "image_position_angle_deg" if split == "holdout" else "position_angle_deg"
        system_document: dict[str, Any] = {
            "id": galaxy,
            "split": split,
            "scoreProtocol": (
                "preregistered_kinematic_axis_final_holdout"
                if split == "holdout" else "frozen_image_axis_development_or_validation"
            ),
            "observation": {
                "tracer": "THINGS H I moment-1 velocity field",
                "scoredPixels": int(observation["scored_pixels"]),
                "inclinationDeg": number(observation, "inclination_deg"),
                "positionAngleDeg": number(observation, position_key),
                "medianDispersionKmS": number(observation, "median_dispersion_km_s"),
            },
            "twinFidelity": {
                "totalMapNormalizedL2": number(morph, "normalized_l2"),
                "totalMapPixelCorrelation": number(morph, "pixel_correlation"),
                "massRelativeError": number(morph, "mass_relative_error"),
                "coefficientsPerBaryonicComponent": (
                    int(holdout_catalog[galaxy]["coefficients_per_component"])
                    if split == "holdout" else 256
                ),
                "velocityTargetsUsedInExtraction": False,
                "gravityParametersUsedInExtraction": 0,
            },
            "simulatorFidelityLimitKmS": 12.0 if split == "holdout" else 8.0,
            "models": models,
        }

        if split == "holdout":
            adjusted_models: dict[str, Any] = {}
            for model in model_definitions:
                raw = next(
                    row for row in holdout_scores
                    if row["galaxy"] == galaxy and row["model"] == model
                    and row["prediction_kind"] == "registered_baryons_photometric_axis"
                )
                adjusted_models[model] = {
                    "imageAxisSourceVersusObserved": metric(raw),
                    "sourceVersusObserved": models[model]["sourceVersusObserved"],
                    "twinVersusObserved": models[model]["twinVersusObserved"],
                }
            system_document["geometryDiagnostic"] = {
                "status": "preregistered_final_holdout_observation_policy",
                "axisOffsetDeg": number(observation, "kinematic_phase_offset_deg_in_registered_plane"),
                "firstHarmonicExplainedVarianceFraction": number(
                    observation, "first_harmonic_explained_variance_fraction"
                ),
                "fittedObservationNuisances": 1,
                "fittedVelocityAmplitudes": 0,
                "fittedGravityParameters": 0,
                "models": adjusted_models,
            }
        elif galaxy in development_geometry:
            geometry = development_geometry[galaxy]
            adjusted_models = {}
            for model in model_definitions:
                adjusted_source = next(
                    row for row in development_geometry_scores
                    if row["galaxy"] == galaxy and row["model"] == model
                    and row["prediction_kind"] == "registered_baryons_kinematic_axis"
                )
                adjusted_twin = next(
                    row for row in development_geometry_scores
                    if row["galaxy"] == galaxy and row["model"] == model
                    and row["prediction_kind"] == "adaptive_twin_kinematic_axis"
                )
                adjusted_models[model] = {
                    "sourceVersusObserved": metric(adjusted_source),
                    "twinVersusObserved": metric(adjusted_twin),
                }
            system_document["geometryDiagnostic"] = {
                "status": "post_reveal_method_development_not_blind_evidence",
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

    if len(systems) != 8:
        raise RuntimeError("resolved evidence must contain eight systems")
    artifact_hashes = {key: sha256(source) for key, (source, _target) in ASSETS.items()}
    core = {
        "schemaVersion": "sigma-hosted-resolved-twin-evidence/1",
        "stage": "P0752",
        "evidenceClass": "precomputed_development_validation_and_final_holdout_result",
        "title": "Resolved spiral twins versus observed velocity fields, including final holdout",
        "parents": {f"{stage}ReportSha256": report["reportSha256"] for stage, report in reports.items()},
        "sample": {
            "systems": len(systems),
            "scoredVelocityPixels": sum(system["observation"]["scoredPixels"] for system in systems),
            "developmentSystems": [system["id"] for system in systems if system["split"] == "development"],
            "validationSystems": [system["id"] for system in systems if system["split"] == "validation"],
            "holdoutSystems": [system["id"] for system in systems if system["split"] == "holdout"],
            "sealedHoldoutSystems": [],
        },
        "generator": {
            "id": "adaptive-sparse-orthonormal-haar-2d",
            "candidateCoefficientsPerBaryonicComponent": [256, 384, 512, 768, 1024],
            "selectionInputs": "baryonic morphology, mass, 3D projection replay, and compression only",
            "gravityParameters": 0,
            "velocityTargetsUsed": False,
        },
        "models": model_definitions,
        "systems": systems,
        "artifacts": {
            "developmentAtlasPath": "/assets/resolved-twin-development-atlas.png",
            "developmentAtlasSha256": artifact_hashes["development"],
            "rawValidationAtlasPath": "/assets/resolved-twin-validation-atlas.png",
            "rawValidationAtlasSha256": artifact_hashes["validation"],
            "geometryDevelopmentAtlasPath": "/assets/resolved-twin-geometry-diagnostic-atlas.png",
            "geometryDevelopmentAtlasSha256": artifact_hashes["geometry"],
            "finalHoldoutAtlasPath": "/assets/resolved-twin-holdout-atlas.png",
            "finalHoldoutAtlasSha256": artifact_hashes["holdout"],
            "finalHoldoutCurvesPath": "/assets/resolved-twin-holdout-curves.png",
            "finalHoldoutCurvesSha256": artifact_hashes["holdoutCurves"],
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
        "finalHoldoutVerdicts": reports["p0752"]["formulaVerdicts"],
        "executionBoundary": {
            "formulaExecution": "precomputed fixed comparators only",
            "arbitraryHosted2dFormulaExecution": False,
            "individualStarDynamics": False,
            "validationStatus": "four development, two validation, and two one-shot final holdout systems are complete",
            "protocolStatusMeaning": "Execution integrity passed; fixed MOND was competitive but incomplete and Newtonian baryons failed the final holdout criteria.",
        },
        "claimBoundary": reports["p0750"]["claimBoundary"] + reports["p0752"]["claimBoundary"],
    }
    document = {**core, "evidenceSha256": canonical_hash(core)}
    OUTPUT.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for source, target in ASSETS.values():
        shutil.copyfile(source, target)
    print(json.dumps({
        "systems": len(systems),
        "scoredVelocityPixels": core["sample"]["scoredVelocityPixels"],
        "evidenceSha256": document["evidenceSha256"],
        "holdoutVerdicts": core["finalHoldoutVerdicts"],
    }))


if __name__ == "__main__":
    main()
