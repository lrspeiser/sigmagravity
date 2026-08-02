#!/usr/bin/env python3
"""Run the frozen P0728 complete fine-grid AQUAL reconstruction."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_p0725_aqual_solver_robustness import (
    physics_core,
    prepare_input_bundle,
    read_json,
    write_csv,
    write_json,
)
from run_p0726_independent_nonlinear_crosscheck import (
    compare_reference,
    execute_variant,
)

from voidscreen.field_job import _worker_source_sha256, canonical_sha256, file_sha256

DEFAULT_CONFIG = ROOT / "configs" / "p0728_complete_fine_grid_aqual.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0728_complete_fine_grid_aqual"
DEFAULT_STORE = ROOT / "tmp" / "p0724-http"
DEFAULT_WORK = ROOT / "tmp" / "p0728-worker-cache"


def selected_manifest(base: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    manifest = copy.deepcopy(base)
    manifest["name"] = config.get(
        "manifestName", "AQUAL P0728 selected universal hybrid"
    )
    manifest["solver"] = copy.deepcopy(config["solver"])
    return manifest


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def score_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    residuals = np.asarray([float(row["residual_m_s"]) for row in rows], dtype=float)
    uncertainties = np.asarray(
        [float(row["uncertainty_m_s"]) for row in rows], dtype=float
    )
    inverse_variance = 1.0 / np.square(uncertainties)
    return {
        "validObservationPoints": int(residuals.size),
        "rmseMPerS": float(np.sqrt(np.mean(np.square(residuals)))),
        "inverseVarianceWeightedRmseMPerS": float(
            np.sqrt(np.sum(inverse_variance * np.square(residuals)) / np.sum(inverse_variance))
        ),
        "chiSquare": float(np.sum(np.square(residuals / uncertainties))),
    }


def aggregate_scores(per_system: list[dict[str, Any]]) -> dict[str, Any]:
    rmse = np.asarray([float(row["rmseMPerS"]) for row in per_system], dtype=float)
    all_residuals = np.concatenate(
        [np.asarray(row["residualsMPerS"], dtype=float) for row in per_system]
    )
    all_uncertainties = np.concatenate(
        [np.asarray(row["uncertaintiesMPerS"], dtype=float) for row in per_system]
    )
    inverse_variance = 1.0 / np.square(all_uncertainties)
    chi_square = float(np.sum(np.square(all_residuals / all_uncertainties)))
    return {
        "systems": len(per_system),
        "validObservationPoints": int(all_residuals.size),
        "equalGalaxyRmseKmS": float(np.sqrt(np.mean(np.square(rmse))) / 1000.0),
        "pointWeightedRmseKmS": float(
            np.sqrt(np.mean(np.square(all_residuals))) / 1000.0
        ),
        "inverseVarianceWeightedRmseKmS": float(
            np.sqrt(
                np.sum(inverse_variance * np.square(all_residuals))
                / np.sum(inverse_variance)
            )
            / 1000.0
        ),
        "chiSquare": chi_square,
        "reducedChiSquare": chi_square / all_residuals.size,
    }


def normalized_prediction_change(
    baseline: dict[int, float], candidate: dict[int, float]
) -> float:
    points = sorted(set(baseline) & set(candidate))
    if not points:
        raise ValueError("no paired prediction points")
    baseline_values = np.asarray([baseline[index] for index in points], dtype=float)
    candidate_values = np.asarray([candidate[index] for index in points], dtype=float)
    return float(
        np.sqrt(np.mean(np.square(candidate_values - baseline_values)))
        / np.sqrt(np.mean(np.square(baseline_values)))
    )


def render_model_plot(
    output: Path, summaries: list[dict[str, Any]], stage: str
) -> None:
    ordered = sorted(summaries, key=lambda row: float(row["equalGalaxyRmseKmS"]))
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.barh(
        [str(row["model"]) for row in ordered],
        [float(row["equalGalaxyRmseKmS"]) for row in ordered],
        color=["#3b82f6" if row["model"] == "aqual_simple_mu" else "#94a3b8" for row in ordered],
    )
    axis.invert_yaxis()
    axis.set_xlabel("equal-galaxy RMSE (km/s; lower is better)")
    title = (
        f"{stage} complete four-galaxy fine-grid comparison"
        if any(row["model"] == "aqual_simple_mu" for row in summaries)
        else f"{stage} eligible complete rows; incomplete AQUAL excluded"
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output / "fine_grid_model_comparison.png", dpi=180)
    plt.close(figure)


def render_reference_plot(
    output: Path, comparisons: list[dict[str, Any]], stage: str
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    systems = [str(row["system"]) for row in comparisons]
    values = [
        float(row["predictionNormalizedRmse"])
        if row["predictionNormalizedRmse"] is not None
        else np.nan
        for row in comparisons
    ]
    positions = np.arange(len(systems))
    axis.bar(positions, values, color="#7c3aed")
    valid_values = [value for value in values if np.isfinite(value)]
    axis.set_xticks(positions, systems)
    axis.set_xlim(-0.5, len(systems) - 0.5)
    axis.set_ylim(min(valid_values) / 2.0, max(valid_values) * 5.0)
    for index, row in enumerate(comparisons):
        if not row["comparable"]:
            axis.text(
                index,
                max(valid_values) * 1.5,
                "not converged",
                ha="center",
                va="center",
                rotation=90,
            )
    axis.set_yscale("log")
    axis.set_ylabel("circular-speed normalized RMS difference")
    axis.set_title(f"{stage} agreement with independent field references")
    figure.tight_layout()
    figure.savefig(output / "aqual_reference_agreement.png", dpi=180)
    plt.close(figure)


def main(
    default_config: Path = DEFAULT_CONFIG,
    default_output: Path = DEFAULT_OUTPUT,
    default_work: Path = DEFAULT_WORK,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--work", type=Path, default=default_work)
    arguments = parser.parse_args()

    config_path = arguments.config.resolve()
    output = arguments.output.resolve()
    store = arguments.store.resolve()
    work = arguments.work.resolve()
    config = read_json(config_path)
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    p0727_path = (ROOT / config["p0727Report"]).resolve()
    p0724_path = (ROOT / config["p0724Report"]).resolve()
    point_path = (ROOT / config["p0724PointPredictions"]).resolve()
    paired_path = (ROOT / config["p0724PairedSensitivity"]).resolve()
    summary_path = (ROOT / config["p0724ModelSummary"]).resolve()
    locked_files = {
        "p0727Report": (p0727_path, config["p0727ReportSha256"]),
        "p0724Report": (p0724_path, config["p0724ReportSha256"]),
        "p0724PointPredictions": (
            point_path,
            config["p0724PointPredictionsSha256"],
        ),
        "p0724PairedSensitivity": (
            paired_path,
            config["p0724PairedSensitivitySha256"],
        ),
        "p0724ModelSummary": (summary_path, config["p0724ModelSummarySha256"]),
    }
    for label, (path, expected) in locked_files.items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"{label} hash changed")
    for item in config.get("additionalLockedFiles", []):
        path = (ROOT / item["path"]).resolve()
        if file_sha256(path) != item["sha256"]:
            raise RuntimeError(f"{item['id']} hash changed")

    p0727 = read_json(p0727_path)
    qualification = config.get("p0727VariantRequirement", "selected")
    candidate_variant = config["selectedSolverVariant"]
    if p0727["status"] != "pass":
        raise RuntimeError("P0727 no longer passes")
    if qualification == "selected":
        if p0727["selectedUniversalSolverVariant"] != candidate_variant:
            raise RuntimeError("P0727 selection changed")
    elif qualification == "qualifying":
        if candidate_variant not in p0727["qualifyingUniversalVariants"]:
            raise RuntimeError("P0727 candidate is not a qualified universal variant")
    else:
        raise RuntimeError(f"unsupported P0727 variant requirement: {qualification}")
    selected_record = next(
        item
        for item in p0727["manifests"]
        if item["variant"] == candidate_variant
    )
    if selected_record["solver"] != config["solver"]:
        raise RuntimeError("selected P0727 solver controls changed")

    base = read_json(ROOT / config["baseManifest"])
    manifest = selected_manifest(base, config)
    if canonical_sha256(physics_core(manifest)) != canonical_sha256(physics_core(base)):
        raise RuntimeError("P0728 changed the physical model")

    bundles: dict[str, Path] = {}
    targets: dict[str, list[dict[str, Any]]] = {}
    for system in config["systems"]:
        reference = (ROOT / system["referenceDirectory"]).resolve()
        scientific = read_json(reference / "scientific_result.json")
        if scientific["resultSha256"] != system["referenceScientificResultSha256"]:
            raise RuntimeError(f"{system['id']} reference hash changed")
        bundle, observation_targets = prepare_input_bundle(store, work, system)
        bundles[system["id"]] = bundle
        targets[system["id"]] = observation_targets

    payloads = []
    for system in config["systems"]:
        system_id = str(system["id"])
        payloads.append(
            {
                "variant": candidate_variant,
                "system": system_id,
                "model": manifest,
                "bundleRoot": str(bundles[system_id]),
                "inputBundleSha256": system["inputBundleSha256"],
                "output": str(work / "runs" / system_id),
                "request": {
                    "schemaVersion": "sigma-field-job-request/1",
                    "requestedObservables": ["massive_tracer_acceleration"],
                    "observationTargets": targets[system_id],
                    "parameterPolicy": manifest["parameterPolicy"],
                    "seed": 20260805,
                },
            }
        )

    runs = []
    with ProcessPoolExecutor(max_workers=int(config["parallelWorkers"])) as executor:
        futures = {executor.submit(execute_variant, payload): payload for payload in payloads}
        for future in as_completed(futures):
            row = future.result()
            scientific = read_json(Path(row["output"]) / "scientific_result.json")
            metadata = scientific["numericalMetadata"]
            row["picardWarmupIterations"] = metadata["picard_warmup_iterations"]
            row["picardWarmupDamping"] = metadata["picard_warmup_damping"]
            runs.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    system_order = {item["id"]: index for index, item in enumerate(config["systems"])}
    runs.sort(key=lambda row: system_order[row["system"]])

    comparisons = compare_reference(runs, config["systems"])
    per_system = []
    prediction_rows = []
    for run in runs:
        if not run["converged"]:
            continue
        predictions = read_csv(Path(run["output"]) / "observation_predictions.csv")
        scored = score_predictions(predictions)
        residuals = [float(row["residual_m_s"]) for row in predictions]
        uncertainties = [float(row["uncertainty_m_s"]) for row in predictions]
        per_system.append(
            {
                "system": run["system"],
                **scored,
                "rmseKmS": scored["rmseMPerS"] / 1000.0,
                "residualsMPerS": residuals,
                "uncertaintiesMPerS": uncertainties,
            }
        )
        prediction_rows.extend(
            {
                "scenario": config["scenario"],
                "model": "aqual_simple_mu",
                "system_id": run["system"],
                **row,
            }
            for row in predictions
        )

    aggregate = aggregate_scores(per_system) if per_system else None
    p0724 = read_json(p0724_path)
    if p0724["stage"] != "P0724":
        raise RuntimeError("P0724 report identity changed")
    prior_summaries = read_csv(summary_path)
    fine_summaries: list[dict[str, Any]] = []
    for row in prior_summaries:
        if row["scenario"] != config["scenario"] or row["model"] == "aqual_simple_mu":
            continue
        fine_summaries.append(
            {
                "model": row["model"],
                "systems": int(row["systems"]),
                "scoredSystems": int(row["scoredSystems"]),
                "validObservationPoints": int(row["validObservationPoints"]),
                "equalGalaxyRmseKmS": float(row["equalGalaxyRmseKmS"]),
                "pointWeightedRmseKmS": float(row["pointWeightedRmseKmS"]),
                "reducedChiSquare": float(row["reducedChiSquare"]),
                "universalGravityParameters": int(row["universalGravityParameters"]),
                "perObjectGravityParameters": int(row["perObjectGravityParameters"]),
                "source": "locked P0724 complete row",
            }
        )
    complete_aggregate = (
        aggregate
        if aggregate is not None and aggregate["systems"] == len(config["systems"])
        else None
    )
    if complete_aggregate is not None:
        fine_summaries.append(
            {
                "model": "aqual_simple_mu",
                "systems": complete_aggregate["systems"],
                "scoredSystems": complete_aggregate["systems"],
                "validObservationPoints": complete_aggregate["validObservationPoints"],
                "equalGalaxyRmseKmS": complete_aggregate["equalGalaxyRmseKmS"],
                "pointWeightedRmseKmS": complete_aggregate["pointWeightedRmseKmS"],
                "reducedChiSquare": complete_aggregate["reducedChiSquare"],
                "universalGravityParameters": 2,
                "perObjectGravityParameters": 0,
                "source": (
                    "P0728 selected universal hybrid"
                    if config["stage"] == "P0728"
                    else f"{config['stage']} preregistered universal hybrid"
                ),
            }
        )
    fine_summaries.sort(key=lambda row: float(row["equalGalaxyRmseKmS"]))
    for index, row in enumerate(fine_summaries, start=1):
        row["rank"] = index

    baseline_rows = [
        row
        for row in read_csv(point_path)
        if row["scenario"] == p0724["scenarios"][0]["id"]
        and row["model"] == "aqual_simple_mu"
    ]
    baseline_by_system: dict[str, dict[int, float]] = {}
    candidate_by_system: dict[str, dict[int, float]] = {}
    for row in baseline_rows:
        baseline_by_system.setdefault(row["system_id"], {})[int(row["point_index"])] = float(
            row["predicted_speed_m_s"]
        )
    for row in prediction_rows:
        candidate_by_system.setdefault(str(row["system_id"]), {})[
            int(row["point_index"])
        ] = float(row["predicted_speed_m_s"])
    aqual_changes = [
        {
            "system": system["id"],
            "normalizedPredictionRmse": normalized_prediction_change(
                baseline_by_system[system["id"]], candidate_by_system[system["id"]]
            ),
        }
        for system in config["systems"]
        if system["id"] in candidate_by_system
    ]

    inherited_other_changes = [
        float(row["normalized_prediction_rmse"])
        for row in read_csv(paired_path)
        if row["scenario"] == config["scenario"]
        and row["model"] != "aqual_simple_mu"
        and row["normalized_prediction_rmse"]
    ]
    all_changes = np.asarray(
        inherited_other_changes
        + [float(row["normalizedPredictionRmse"]) for row in aqual_changes],
        dtype=float,
    )
    baseline_aqual = next(
        row
        for row in prior_summaries
        if row["scenario"] == p0724["scenarios"][0]["id"]
        and row["model"] == "aqual_simple_mu"
    )
    aqual_fit_change = (
        abs(
            float(complete_aggregate["equalGalaxyRmseKmS"])
            - float(baseline_aqual["equalGalaxyRmseKmS"])
        )
        / float(baseline_aqual["equalGalaxyRmseKmS"])
        if complete_aggregate is not None
        else None
    )
    other_fit_changes = []
    for model in ["newtonian_poisson", "qumond_simple_nu", "refracted_gravity_published_fixture"]:
        baseline = next(
            row
            for row in prior_summaries
            if row["scenario"] == p0724["scenarios"][0]["id"] and row["model"] == model
        )
        fine = next(row for row in prior_summaries if row["scenario"] == config["scenario"] and row["model"] == model)
        other_fit_changes.append(
            abs(float(fine["equalGalaxyRmseKmS"]) - float(baseline["equalGalaxyRmseKmS"]))
            / float(baseline["equalGalaxyRmseKmS"])
        )
    stability_limits = config["inheritedStabilityGates"]
    stability_metrics = {
        "pairedModelSystems": int(all_changes.size),
        "expectedPairedModelSystems": 16,
        "medianNormalizedPredictionRmse": float(np.median(all_changes)),
        "p90NormalizedPredictionRmse": float(np.percentile(all_changes, 90.0)),
        "aqualAggregateFitRmseRelativeChange": aqual_fit_change,
        "maximumModelAggregateFitRmseRelativeChange": (
            max(other_fit_changes + [aqual_fit_change])
            if aqual_fit_change is not None
            else None
        ),
    }
    stability_gates = {
        "complete_prediction_coverage": all_changes.size == 16,
        "median_normalized_prediction_rmse": stability_metrics[
            "medianNormalizedPredictionRmse"
        ]
        <= float(stability_limits["maximumScenarioMedianNormalizedPredictionRmse"]),
        "p90_normalized_prediction_rmse": stability_metrics[
            "p90NormalizedPredictionRmse"
        ]
        <= float(stability_limits["maximumScenarioP90NormalizedPredictionRmse"]),
        "aggregate_fit_rmse_relative_change": stability_metrics[
            "maximumModelAggregateFitRmseRelativeChange"
        ]
        is not None
        and stability_metrics["maximumModelAggregateFitRmseRelativeChange"]
        <= float(stability_limits["maximumModelAggregateFitRmseRelativeChange"]),
    }

    gates = config["engineeringGates"]
    engineering_gates = {
        "required_systems": len(runs) == int(gates["requiredSystems"]),
        "all_fields_converged": len(runs) == int(gates["requiredSystems"])
        and all(row["converged"] for row in runs),
        "equation_residual": all(
            float(row["maximumEquationResidual"]) <= float(gates["maximumEquationResidual"])
            for row in runs
        ),
        "relative_update": all(
            float(row["maximumRelativeUpdate"]) <= float(gates["maximumRelativeUpdate"])
            for row in runs
        ),
        "independent_reference_agreement": len(comparisons) == int(gates["requiredSystems"])
        and all(
            row["comparable"]
            and float(row["predictionNormalizedRmse"])
            <= float(gates["maximumReferencePredictionNormalizedRmse"])
            and float(row["potentialNormalizedRmse"])
            <= float(gates["maximumReferencePotentialNormalizedRmse"])
            and float(row["accelerationNormalizedRmse"])
            <= float(gates["maximumReferenceAccelerationNormalizedRmse"])
            for row in comparisons
        ),
        "complete_observation_scoring": aggregate is not None
        and complete_aggregate is not None
        and complete_aggregate["validObservationPoints"]
        == int(gates["requiredObservationPoints"]),
        "iteration_limit_faithful": all(
            not row["iterationLimitAdjusted"]
            and row["requestedMaximumIterations"] == row["executedMaximumIterations"]
            for row in runs
        ),
        "selected_solver_locked": selected_record["solver"] == config["solver"],
        "physics_manifest_unchanged": canonical_sha256(physics_core(manifest))
        == canonical_sha256(physics_core(base)),
        "no_per_object_gravity_parameters": manifest["parameterPolicy"][
            "perObjectParameters"
        ]
        == [],
        "locked_input_hashes_valid": True,
    }
    report = {
        "schemaVersion": f"sigma-{config['stage'].lower()}-complete-fine-grid-aqual/1",
        "stage": config["stage"],
        "status": "pass" if all(engineering_gates.values()) else "fail",
        "stabilityStatus": "stable" if all(stability_gates.values()) else "sensitive",
        "sampleStatus": config["sampleStatus"],
        "scenario": config["scenario"],
        "configSha256": file_sha256(config_path),
        "workerSourceSha256": _worker_source_sha256(),
        "modelDocumentSha256": canonical_sha256(manifest),
        "modelPhysicsSha256": canonical_sha256(physics_core(manifest)),
        "selectedSolverVariant": candidate_variant,
        "solver": config["solver"],
        "runs": runs,
        "referenceComparisons": comparisons,
        "perGalaxyScores": [
            {key: value for key, value in row.items() if key not in {"residualsMPerS", "uncertaintiesMPerS"}}
            for row in per_system
        ],
        "aqualAggregateScores": aggregate,
        "completeFineGridModelSummary": fine_summaries,
        "completeFineGridRankOrder": [row["model"] for row in fine_summaries],
        "aqualBaselinePredictionChanges": aqual_changes,
        "reconstructedFineGridSensitivity": stability_metrics,
        "engineeringGateResults": engineering_gates,
        "failedEngineeringGates": [key for key, value in engineering_gates.items() if not value],
        "stabilityGateResults": stability_gates,
        "failedStabilityGates": [key for key, value in stability_gates.items() if not value],
        "selectionPolicy": config["selectionPolicy"],
        "rethinkPolicy": config["rethinkPolicy"],
        "claimBoundary": config["claimBoundary"],
    }
    write_json(output / "report.json", report)
    write_csv(output / "run_summary.csv", runs)
    write_csv(output / "reference_comparison.csv", comparisons)
    write_csv(
        output / "per_galaxy_scores.csv",
        report["perGalaxyScores"],
    )
    write_csv(output / "point_predictions.csv", prediction_rows)
    write_csv(output / "complete_fine_grid_model_summary.csv", fine_summaries)
    write_csv(output / "aqual_baseline_prediction_changes.csv", aqual_changes)
    render_model_plot(output, fine_summaries, config["stage"])
    render_reference_plot(output, comparisons, config["stage"])
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
