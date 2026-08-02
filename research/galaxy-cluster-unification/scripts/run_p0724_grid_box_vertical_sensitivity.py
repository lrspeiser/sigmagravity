#!/usr/bin/env python3
"""Run the frozen P0724 grid, box, and vertical-prior sensitivity study."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from run_p0723_formula_neutral_api_comparators import (
    KINEMATICS,
    finite_float,
    observation_targets,
    run_model_batch,
    sha256,
    submit_galaxy_job,
    upload_registered_map,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0724_grid_box_vertical_sensitivity.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0724_grid_box_vertical_sensitivity"


def extract_parameter_packages(
    base: str,
    temporary: Path,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    packages: dict[str, dict[str, Any]] = {}
    jobs: list[dict[str, Any]] = []
    baseline = next(
        item for item in config["scenarios"] if item["id"] == config["baselineScenario"]
    )
    for index, galaxy in enumerate(config["systems"]):
        upload = upload_registered_map(base, temporary, galaxy)
        extracted, artifacts = submit_galaxy_job(
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
                    "zCells": int(baseline["zCells"]),
                    "seed": int(config["generationSeed"]) + index * 2,
                },
                "outputLicense": {
                    "id": "research-source-license",
                    "redistributionAllowed": False,
                },
            },
        )
        package = json.loads(artifacts["parameters.json"])
        packages[galaxy] = package
        jobs.append(
            {
                "operation": "extract_roundtrip",
                "galaxy": galaxy,
                "scenario": None,
                "mapUploadId": upload["id"],
                "jobId": extracted["id"],
                "scientificResultSha256": extracted["scientificResultSha256"],
                "parameterPackageSha256": package["contentSha256"],
            }
        )
    return packages, jobs


def generate_scenarios(
    base: str,
    config: dict[str, Any],
    packages: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    systems_by_scenario: dict[str, list[dict[str, Any]]] = {}
    jobs: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        scenario_id = str(scenario["id"])
        systems: list[dict[str, Any]] = []
        for index, galaxy in enumerate(config["systems"]):
            seed = (
                int(config["generationSeed"])
                + int(scenario["verticalSeedOffset"])
                + index * 2
                + 1
            )
            generated, _artifacts = submit_galaxy_job(
                base,
                {
                    "schemaVersion": "sigma-galaxy-job-submit/1",
                    "operation": "generate",
                    "galaxy": galaxy,
                    "parameterPackage": packages[galaxy],
                    "generationControls": {},
                    "outputGrid": {
                        "cellsPerAxis": int(scenario["cellsPerAxis"]),
                        "extentScale": float(scenario["extentScale"]),
                    },
                    "vertical": {
                        "enabled": True,
                        "realizations": 1,
                        "zCells": int(scenario["zCells"]),
                        "seed": seed,
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
                    "operation": "generate",
                    "galaxy": galaxy,
                    "scenario": scenario_id,
                    "jobId": generated["id"],
                    "scientificResultSha256": generated["scientificResultSha256"],
                    "parameterPackageSha256": packages[galaxy]["contentSha256"],
                    "cellsPerAxis": int(scenario["cellsPerAxis"]),
                    "zCells": int(scenario["zCells"]),
                    "extentScale": float(scenario["extentScale"]),
                    "verticalSeed": seed,
                }
            )
        systems_by_scenario[scenario_id] = systems
    return systems_by_scenario, jobs


def prediction_groups(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[int, float]]:
    grouped: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        predicted = finite_float(row.get("predicted_speed_m_s"))
        if predicted is None:
            continue
        key = (str(row["scenario"]), str(row["model"]), str(row["system_id"]))
        grouped[key][int(row["point_index"])] = predicted / 1000.0
    return grouped


def paired_prediction_sensitivity(
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_id = str(config["baselineScenario"])
    grouped = prediction_groups(predictions)
    rows: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        scenario_id = str(scenario["id"])
        if scenario_id == baseline_id:
            continue
        for model in config["models"]:
            model_id = str(model["id"])
            for galaxy in config["systems"]:
                baseline = grouped[(baseline_id, model_id, galaxy)]
                variant = grouped[(scenario_id, model_id, galaxy)]
                points = sorted(set(baseline) & set(variant))
                if not points:
                    delta_rmse = None
                    normalized = None
                else:
                    baseline_values = np.asarray([baseline[index] for index in points])
                    variant_values = np.asarray([variant[index] for index in points])
                    delta_rmse = float(np.sqrt(np.mean(np.square(variant_values - baseline_values))))
                    denominator = float(np.sqrt(np.mean(np.square(baseline_values))))
                    normalized = delta_rmse / denominator if denominator > 0.0 else None
                rows.append(
                    {
                        "scenario": scenario_id,
                        "model": model_id,
                        "system_id": galaxy,
                        "paired_points": len(points),
                        "prediction_delta_rmse_km_s": delta_rmse,
                        "normalized_prediction_rmse": normalized,
                    }
                )
    return rows


def sensitivity_summaries(
    config: dict[str, Any],
    paired: list[dict[str, Any]],
    model_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_id = str(config["baselineScenario"])
    fit_by_key = {
        (str(item["scenario"]), str(item["model"])): float(item["equalGalaxyRmseKmS"])
        for item in model_summaries
    }
    limits = config["stabilityGates"]
    results: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        scenario_id = str(scenario["id"])
        if scenario_id == baseline_id:
            continue
        values = np.asarray(
            [
                float(row["normalized_prediction_rmse"])
                for row in paired
                if row["scenario"] == scenario_id
                and row["normalized_prediction_rmse"] is not None
            ],
            dtype=float,
        )
        fit_changes = []
        for model in config["models"]:
            model_id = str(model["id"])
            baseline_fit = fit_by_key[(baseline_id, model_id)]
            variant_fit = fit_by_key[(scenario_id, model_id)]
            fit_changes.append(abs(variant_fit - baseline_fit) / baseline_fit)
        median = float(np.median(values)) if values.size else None
        p90 = float(np.percentile(values, 90.0)) if values.size else None
        maximum_fit_change = max(fit_changes) if fit_changes else None
        gates = {
            "median_normalized_prediction_rmse": median is not None
            and median
            <= float(limits["maximumScenarioMedianNormalizedPredictionRmse"]),
            "p90_normalized_prediction_rmse": p90 is not None
            and p90 <= float(limits["maximumScenarioP90NormalizedPredictionRmse"]),
            "aggregate_fit_rmse_relative_change": maximum_fit_change is not None
            and maximum_fit_change
            <= float(limits["maximumModelAggregateFitRmseRelativeChange"]),
        }
        results.append(
            {
                "scenario": scenario_id,
                "role": scenario["role"],
                "pairedModelSystems": int(values.size),
                "medianNormalizedPredictionRmse": median,
                "p90NormalizedPredictionRmse": p90,
                "maximumModelAggregateFitRmseRelativeChange": maximum_fit_change,
                "gateResults": gates,
                "status": "stable" if all(gates.values()) else "sensitive",
            }
        )
    return results


def rank_orders(
    config: dict[str, Any], model_summaries: list[dict[str, Any]]
) -> dict[str, list[str]]:
    return {
        str(scenario["id"]): [
            str(item["model"])
            for item in sorted(
                [
                    row
                    for row in model_summaries
                    if row["scenario"] == str(scenario["id"])
                ],
                key=lambda row: float(row["equalGalaxyRmseKmS"]),
            )
        ]
        for scenario in config["scenarios"]
    }


def render_plots(
    output: Path,
    config: dict[str, Any],
    paired: list[dict[str, Any]],
    model_summaries: list[dict[str, Any]],
) -> None:
    scenario_ids = [str(item["id"]) for item in config["scenarios"]]
    model_ids = [str(item["id"]) for item in config["models"]]
    colors = ["#7f8c8d", "#2980b9", "#27ae60", "#c0392b"]
    x = np.arange(len(scenario_ids), dtype=float)
    width = 0.19
    figure, axis = plt.subplots(figsize=(12.5, 6.2))
    for model_index, (model, color) in enumerate(zip(model_ids, colors, strict=True)):
        values = [
            next(
                float(row["equalGalaxyRmseKmS"])
                for row in model_summaries
                if row["scenario"] == scenario and row["model"] == model
            )
            for scenario in scenario_ids
        ]
        axis.bar(
            x + (model_index - 1.5) * width,
            values,
            width=width,
            color=color,
            label=model,
        )
    axis.set_xticks(x, scenario_ids, rotation=18, ha="right")
    axis.set_ylabel("Equal-galaxy circular-speed RMSE (km/s)")
    axis.set_title("P0724 fit sensitivity across frozen numerical scenarios")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "aggregate_fit_sensitivity.png", dpi=180)
    plt.close(figure)

    nonbaseline = [
        value for value in scenario_ids if value != str(config["baselineScenario"])
    ]
    column_keys = [
        (str(model["id"]), galaxy)
        for model in config["models"]
        for galaxy in config["systems"]
    ]
    value_by_key = {
        (str(row["scenario"]), str(row["model"]), str(row["system_id"])): float(
            row["normalized_prediction_rmse"]
        )
        for row in paired
        if row["normalized_prediction_rmse"] is not None
    }
    matrix = np.asarray(
        [
            [value_by_key.get((scenario, model, galaxy), np.nan) for model, galaxy in column_keys]
            for scenario in nonbaseline
        ]
    )
    figure, axis = plt.subplots(figsize=(15.5, 5.0))
    image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=0.25)
    axis.set_yticks(np.arange(len(nonbaseline)), nonbaseline)
    axis.set_xticks(
        np.arange(len(column_keys)),
        [f"{model}\n{galaxy}" for model, galaxy in column_keys],
        rotation=65,
        ha="right",
        fontsize=7,
    )
    axis.set_title("RMS prediction change relative to the P0724 baseline")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("normalized prediction RMSE")
    figure.tight_layout()
    figure.savefig(output / "prediction_sensitivity_heatmap.png", dpi=180)
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
    base = args.base_url.rstrip("/")
    targets = observation_targets(config)
    model_summaries: list[dict[str, Any]] = []
    all_per_galaxy: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sigma-p0724-") as temporary_value:
        packages, extraction_jobs = extract_parameter_packages(
            base, Path(temporary_value), config
        )
        systems_by_scenario, generation_jobs = generate_scenarios(base, config, packages)
        for scenario in config["scenarios"]:
            scenario_id = str(scenario["id"])
            for specification in config["models"]:
                summary, per_galaxy, predictions = run_model_batch(
                    base,
                    specification,
                    systems_by_scenario[scenario_id],
                    targets,
                    timeout=7200.0,
                )
                summary["scenario"] = scenario_id
                summary["scenarioRole"] = scenario["role"]
                model_summaries.append(summary)
                all_per_galaxy.extend(
                    [
                        {
                            "scenario": scenario_id,
                            "model": specification["id"],
                            **row,
                        }
                        for row in per_galaxy
                    ]
                )
                all_predictions.extend(
                    [
                        {
                            "scenario": scenario_id,
                            "model": specification["id"],
                            **row,
                        }
                        for row in predictions
                    ]
                )

    paired = paired_prediction_sensitivity(config, all_predictions)
    scenario_summaries = sensitivity_summaries(config, paired, model_summaries)
    engineering = config["engineeringGates"]
    expected_batches = (
        int(engineering["requiredScenarios"]) * int(engineering["requiredModels"])
    )
    engineering_gates = {
        "required_design": len(config["systems"]) == int(engineering["requiredSystems"])
        and len(config["models"]) == int(engineering["requiredModels"])
        and len(config["scenarios"]) == int(engineering["requiredScenarios"]),
        "all_batches_present": len(model_summaries) == expected_batches,
        "all_batches_succeeded": all(
            item["batchState"] == "succeeded" for item in model_summaries
        ),
        "all_systems_scored": all(
            int(item["scoredSystems"]) == len(config["systems"])
            for item in model_summaries
        ),
        "convergence_fraction": all(
            float(item["convergenceFraction"])
            >= float(engineering["minimumConvergenceFraction"])
            for item in model_summaries
        ),
        "equation_residual": all(
            float(item["maximumEquationResidual"])
            <= float(engineering["maximumEquationResidual"])
            for item in model_summaries
        ),
        "no_per_object_gravity_parameters": all(
            int(item["perObjectGravityParameters"])
            <= int(engineering["maximumPerObjectGravityParameters"])
            for item in model_summaries
        ),
        "artifact_hashes": all(
            bool(item["allDownloadedArtifactHashesValid"]) for item in model_summaries
        ),
    }
    stability_gates = {
        "scenario_median_prediction_change": all(
            item["gateResults"]["median_normalized_prediction_rmse"]
            for item in scenario_summaries
        ),
        "scenario_p90_prediction_change": all(
            item["gateResults"]["p90_normalized_prediction_rmse"]
            for item in scenario_summaries
        ),
        "model_aggregate_fit_change": all(
            item["gateResults"]["aggregate_fit_rmse_relative_change"]
            for item in scenario_summaries
        ),
    }
    numerical_pass = all(engineering_gates.values())
    stability_pass = all(stability_gates.values())
    status = (
        "stable"
        if numerical_pass and stability_pass
        else "sensitive"
        if numerical_pass
        else "numerical_failure"
    )
    report = {
        "schemaVersion": "sigma-p0724-grid-box-vertical-sensitivity/1",
        "stage": config["stage"],
        "status": status,
        "numericalStatus": "pass" if numerical_pass else "fail",
        "stabilityStatus": "pass" if stability_pass else "fail",
        "sampleStatus": config["sampleStatus"],
        "systems": config["systems"],
        "models": [item["id"] for item in config["models"]],
        "scenarios": config["scenarios"],
        "modelSummaries": model_summaries,
        "scenarioSensitivity": scenario_summaries,
        "modelRankOrderByScenario": rank_orders(config, model_summaries),
        "engineeringGateResults": engineering_gates,
        "stabilityGateResults": stability_gates,
        "failedEngineeringGates": [
            key for key, value in engineering_gates.items() if not value
        ],
        "failedStabilityGates": [
            key for key, value in stability_gates.items() if not value
        ],
        "configSha256": sha256(config_path),
        "kinematicsArchiveSha256": sha256(KINEMATICS),
        "galaxyJobs": extraction_jobs + generation_jobs,
        "claimBoundary": config["claimBoundary"],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output / "scenario_model_summary.csv", model_summaries)
    write_csv(output / "per_galaxy_scores.csv", all_per_galaxy)
    write_csv(output / "point_predictions.csv", all_predictions)
    write_csv(output / "paired_prediction_sensitivity.csv", paired)
    write_csv(output / "scenario_sensitivity.csv", scenario_summaries)
    render_plots(output, config, paired, model_summaries)
    print(json.dumps(report, indent=2))
    if not numerical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
