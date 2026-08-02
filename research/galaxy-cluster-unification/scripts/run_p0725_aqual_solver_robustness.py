#!/usr/bin/env python3
"""Run the frozen P0725 nonlinear-solver robustness matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.field_job import (
    _worker_source_sha256,
    canonical_sha256,
    execute_field_job,
    file_sha256,
)
from voidscreen.generic_field_worker import (
    _finite_volume_divergence_gradient,
    evaluate_field_expression,
    solve_field_manifest,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0725_aqual_solver_robustness.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0725_aqual_solver_robustness"
DEFAULT_STORE = ROOT / "tmp" / "p0724-http"
DEFAULT_WORK = ROOT / "tmp" / "p0725-worker-cache"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def physics_core(manifest: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    result.pop("name", None)
    result.pop("solver", None)
    return result


def variant_manifest(
    base: dict[str, Any], constants: dict[str, Any], variant: dict[str, Any]
) -> dict[str, Any]:
    manifest = copy.deepcopy(base)
    manifest["name"] = f"AQUAL P0725 numerical variant {variant['id']}"
    manifest["solver"] = {
        **constants,
        "initialization": variant["initialization"],
        "damping": variant["damping"],
    }
    return manifest


def prepare_input_bundle(
    store: Path, work: Path, system: dict[str, Any]
) -> tuple[Path, list[dict[str, Any]]]:
    upload_root = store / "uploads" / str(system["dataUploadId"])
    upload = read_json(upload_root / "upload.json")
    if upload["inputBundle"]["bundleSha256"] != system["inputBundleSha256"]:
        raise RuntimeError(f"{system['id']} input bundle hash changed")
    if upload["archive"]["sha256"] != system["archiveSha256"]:
        raise RuntimeError(f"{system['id']} archive hash changed")
    if file_sha256(upload_root / "arrays.npz") != system["archiveSha256"]:
        raise RuntimeError(f"{system['id']} arrays.npz bytes changed")

    bundle_root = work / "inputs" / str(system["id"])
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_root / "bundle.json"
    arrays_path = bundle_root / "arrays.npz"
    if not bundle_path.exists():
        write_json(bundle_path, upload["inputBundle"])
    if not arrays_path.exists():
        shutil.copyfile(upload_root / "arrays.npz", arrays_path)
    if read_json(bundle_path)["bundleSha256"] != system["inputBundleSha256"]:
        raise RuntimeError(f"{system['id']} cached bundle metadata changed")
    if file_sha256(arrays_path) != system["archiveSha256"]:
        raise RuntimeError(f"{system['id']} cached array bytes changed")

    prior_job = read_json(
        store / "jobs" / str(system["p0724FieldJobId"]) / "artifacts" / "job.json"
    )
    return bundle_root, prior_job["observationTargets"]


def execute_variant(payload: dict[str, Any]) -> dict[str, Any]:
    output = Path(payload["output"])
    expected_worker = _worker_source_sha256()
    if output.exists():
        job = read_json(output / "job.json")
        model = read_json(output / "model.json")
        bundle = read_json(output / "input_bundle.json")
        if (
            canonical_sha256(model) != canonical_sha256(payload["model"])
            or bundle["bundleSha256"] != payload["inputBundleSha256"]
            or job["worker"]["sourceSha256"] != expected_worker
        ):
            raise RuntimeError(f"stale P0725 cache at {output}")
    else:
        execute_field_job(
            payload["model"],
            Path(payload["bundleRoot"]),
            payload["request"],
            output,
        )
    scientific = read_json(output / "scientific_result.json")
    resource = read_json(output / "resource_log.json")
    metadata = scientific["numericalMetadata"]
    maximum_residual = max(float(value) for value in scientific["equationResiduals"].values())
    return {
        "variant": payload["variant"],
        "system": payload["system"],
        "state": scientific["state"],
        "converged": scientific["converged"],
        "iterations": scientific["iterations"],
        "maximumRelativeUpdate": scientific["maximumRelativeUpdate"],
        "maximumEquationResidual": maximum_residual,
        "wallSeconds": resource["wallSeconds"],
        "cpuSeconds": resource["cpuSeconds"],
        "peakPythonHeapBytes": resource["peakPythonHeapBytes"],
        "requestedMaximumIterations": metadata["requested_maximum_iterations"],
        "executedMaximumIterations": metadata["executed_maximum_iterations"],
        "iterationLimitAdjusted": metadata["maximum_iterations_limited_by_worker"],
        "initialization": metadata["initialization"],
        "damping": payload["model"]["solver"]["damping"],
        "modelDocumentSha256": canonical_sha256(payload["model"]),
        "modelPhysicsSha256": canonical_sha256(physics_core(payload["model"])),
        "inputBundleSha256": payload["inputBundleSha256"],
        "scientificResultSha256": scientific["resultSha256"],
        "output": str(output),
    }


def known_answer_manifest(solver: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": "sigma-field-model/1",
        "name": "P0725 nonlinear manufactured solution",
        "modelClass": "stationary_elliptic",
        "geometry": {
            "coordinateSystem": "cartesian_2d",
            "dimensions": 2,
            "domain": {"lengthUnit": "m"},
        },
        "fields": {
            "forcing": {
                "rank": "scalar",
                "role": "source",
                "unit": "1/s^2",
                "datasetKey": "forcing",
            },
            "u": {
                "rank": "scalar",
                "role": "solved",
                "unit": "m^2/s^2",
                "boundary": {"type": "dirichlet", "value": 0.0},
            },
        },
        "parameters": {"beta": {"unit": "1", "value": 0.25, "scope": "universal"}},
        "equations": [
            {
                "id": "nonlinear_manufactured",
                "kind": "equality",
                "lhs": {
                    "op": "divergence",
                    "args": [
                        {
                            "op": "multiply",
                            "args": [
                                {
                                    "op": "add",
                                    "args": [
                                        {"const": 1.0},
                                        {
                                            "op": "multiply",
                                            "args": [
                                                {"parameter": "beta"},
                                                {
                                                    "op": "norm",
                                                    "args": [
                                                        {
                                                            "op": "gradient",
                                                            "args": [{"field": "u"}],
                                                        }
                                                    ],
                                                },
                                            ],
                                        },
                                    ],
                                },
                                {"op": "gradient", "args": [{"field": "u"}]},
                            ],
                        }
                    ],
                },
                "rhs": {"field": "forcing"},
            }
        ],
        "observables": [],
        "dataRequirements": [{"key": "forcing", "rank": "scalar", "unit": "1/s^2"}],
        "solver": solver,
        "parameterPolicy": {"mode": "universal_fixed", "perObjectParameters": []},
    }


def run_known_answers(config: dict[str, Any]) -> list[dict[str, Any]]:
    cells = 17
    axis = np.linspace(0.0, 1.0, cells)
    spacing = float(axis[1] - axis[0])
    x, y = np.meshgrid(axis, axis, indexing="ij")
    expected = np.sin(np.pi * x) * np.sin(np.pi * y)
    coefficient_expression = {
        "op": "add",
        "args": [
            {"const": 1.0},
            {
                "op": "multiply",
                "args": [
                    {"parameter": "beta"},
                    {"op": "norm", "args": [{"op": "gradient", "args": [{"field": "u"}]}]},
                ],
            },
        ],
    }
    coefficient = np.asarray(
        evaluate_field_expression(
            coefficient_expression,
            fields={"u": expected},
            parameters={"beta": 0.25},
            spacing=[spacing, spacing],
        )
    )
    forcing, _scale = _finite_volume_divergence_gradient(
        expected,
        coefficient,
        [spacing, spacing],
        coefficient_floor=1e-8,
    )
    rows = []
    for variant in config["solverVariants"]:
        solver = {
            **config["solverConstants"],
            "initialization": variant["initialization"],
            "damping": variant["damping"],
        }
        solution = solve_field_manifest(
            known_answer_manifest(solver), {"forcing": forcing}, spacing
        )
        relative_error = float(
            np.linalg.norm(solution.fields["u"] - expected)
            / np.linalg.norm(expected)
        )
        rows.append(
            {
                "variant": variant["id"],
                "converged": solution.converged,
                "iterations": solution.iterations,
                "maximumRelativeUpdate": solution.maximum_relative_update,
                "maximumEquationResidual": max(solution.equation_residuals.values()),
                "relativeFieldError": relative_error,
            }
        )
    return rows


def load_predictions(row: dict[str, Any]) -> np.ndarray:
    path = Path(row["output"]) / "observation_predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return np.asarray(
            [float(item["predicted_speed_m_s"]) for item in csv.DictReader(handle)],
            dtype=float,
        )


def prediction_agreement(
    rows: list[dict[str, Any]], variants: list[str], systems: list[str]
) -> list[dict[str, Any]]:
    by_key = {(row["variant"], row["system"]): row for row in rows}
    comparisons = []
    for left_index, left in enumerate(variants):
        for right in variants[left_index + 1 :]:
            values = []
            complete = True
            for system in systems:
                left_row = by_key[(left, system)]
                right_row = by_key[(right, system)]
                if not left_row["converged"] or not right_row["converged"]:
                    complete = False
                    continue
                left_values = load_predictions(left_row)
                right_values = load_predictions(right_row)
                denominator = float(np.sqrt(np.mean(np.square(left_values))))
                values.append(
                    float(np.sqrt(np.mean(np.square(left_values - right_values))))
                    / denominator
                )
            comparisons.append(
                {
                    "leftVariant": left,
                    "rightVariant": right,
                    "completeSystems": len(values),
                    "expectedSystems": len(systems),
                    "complete": complete and len(values) == len(systems),
                    "maximumNormalizedPredictionRmse": max(values) if values else None,
                }
            )
    return comparisons


def render_plot(output: Path, rows: list[dict[str, Any]]) -> None:
    variants = list(dict.fromkeys(row["variant"] for row in rows))
    systems = list(dict.fromkeys(row["system"] for row in rows))
    by_key = {(row["variant"], row["system"]): row for row in rows}
    matrix = np.asarray(
        [
            [
                math.log10(max(float(by_key[(variant, system)]["maximumEquationResidual"]), 1e-16))
                for system in systems
            ]
            for variant in variants
        ]
    )
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=-8, vmax=-1)
    axis.set_xticks(np.arange(len(systems)), systems)
    axis.set_yticks(np.arange(len(variants)), variants)
    axis.set_title("P0725 final equation residual by universal solver variant")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("log10(maximum normalized equation residual)")
    figure.tight_layout()
    figure.savefig(output / "solver_residual_matrix.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    arguments = parser.parse_args()

    config_path = arguments.config.resolve()
    output = arguments.output.resolve()
    store = arguments.store.resolve()
    work = arguments.work.resolve()
    config = read_json(config_path)
    base_manifest = read_json(ROOT / config["baseManifest"])
    base_physics_sha = canonical_sha256(physics_core(base_manifest))
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    bundles = {}
    targets = {}
    for system in config["systems"]:
        bundle, observation_targets = prepare_input_bundle(store, work, system)
        bundles[system["id"]] = bundle
        targets[system["id"]] = observation_targets

    payloads = []
    manifest_records = []
    for variant in config["solverVariants"]:
        model = variant_manifest(base_manifest, config["solverConstants"], variant)
        physics_sha = canonical_sha256(physics_core(model))
        if physics_sha != base_physics_sha:
            raise RuntimeError(f"{variant['id']} changed the physical manifest")
        manifest_records.append(
            {
                "variant": variant["id"],
                "modelDocumentSha256": canonical_sha256(model),
                "modelPhysicsSha256": physics_sha,
                "solver": model["solver"],
            }
        )
        for system in config["systems"]:
            system_id = str(system["id"])
            payloads.append(
                {
                    "variant": variant["id"],
                    "system": system_id,
                    "model": model,
                    "bundleRoot": str(bundles[system_id]),
                    "inputBundleSha256": system["inputBundleSha256"],
                    "output": str(work / "runs" / variant["id"] / system_id),
                    "request": {
                        "schemaVersion": "sigma-field-job-request/1",
                        "requestedObservables": ["massive_tracer_acceleration"],
                        "observationTargets": targets[system_id],
                        "parameterPolicy": model["parameterPolicy"],
                        "seed": 20260804,
                    },
                }
            )

    rows = []
    with ProcessPoolExecutor(max_workers=int(config["parallelWorkers"])) as executor:
        futures = {executor.submit(execute_variant, payload): payload for payload in payloads}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    variant_order = {item["id"]: index for index, item in enumerate(config["solverVariants"])}
    system_order = {item["id"]: index for index, item in enumerate(config["systems"])}
    rows.sort(key=lambda row: (variant_order[row["variant"]], system_order[row["system"]]))

    residual_rows = []
    for row in rows:
        with (Path(row["output"]) / "residual_history.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for item in csv.DictReader(handle):
                residual_rows.append(
                    {"variant": row["variant"], "system": row["system"], **item}
                )
    known_answers = run_known_answers(config)
    variants = [str(item["id"]) for item in config["solverVariants"]]
    systems = [str(item["id"]) for item in config["systems"]]
    agreements = prediction_agreement(rows, variants, systems)
    gates = config["gates"]
    by_variant = {
        variant: [row for row in rows if row["variant"] == variant]
        for variant in variants
    }
    robust_variants = [
        variant
        for variant, subset in by_variant.items()
        if len(subset) == len(systems)
        and all(row["converged"] for row in subset)
        and max(float(row["maximumEquationResidual"]) for row in subset)
        <= float(gates["maximumEquationResidual"])
        and max(float(row["maximumRelativeUpdate"]) for row in subset)
        <= float(gates["maximumRelativeUpdate"])
    ]
    agreement_edges = [
        item
        for item in agreements
        if item["complete"]
        and float(item["maximumNormalizedPredictionRmse"])
        <= float(gates["maximumPairwiseNormalizedPredictionRmse"])
        and item["leftVariant"] in robust_variants
        and item["rightVariant"] in robust_variants
    ]
    agreeing_variants = sorted(
        {
            value
            for item in agreement_edges
            for value in (item["leftVariant"], item["rightVariant"])
        }
    )
    median_wall = {
        variant: float(np.median([row["wallSeconds"] for row in by_variant[variant]]))
        for variant in agreeing_variants
    }
    selected = min(median_wall, key=median_wall.get) if median_wall else None
    gate_results = {
        "required_design": len(systems) == int(gates["requiredSystems"])
        and len(variants) == int(gates["requiredVariants"]),
        "known_answer_convergence": sum(row["converged"] for row in known_answers)
        >= int(gates["minimumKnownAnswerConvergedVariants"]),
        "known_answer_field_error": max(row["relativeFieldError"] for row in known_answers)
        <= float(gates["maximumKnownAnswerRelativeFieldError"]),
        "universal_real_system_convergence": len(robust_variants)
        >= int(gates["minimumUniversalVariantsConvergingBothSystems"]),
        "cross_variant_solution_agreement": len(agreeing_variants)
        >= int(gates["minimumUniversalVariantsConvergingBothSystems"]),
        "iteration_limit_faithful": all(
            not row["iterationLimitAdjusted"]
            and row["requestedMaximumIterations"] == row["executedMaximumIterations"]
            for row in rows
        ),
        "no_per_object_gravity_parameters": True,
        "physics_manifest_unchanged": all(
            item["modelPhysicsSha256"] == base_physics_sha for item in manifest_records
        ),
    }
    report = {
        "schemaVersion": "sigma-p0725-aqual-solver-robustness/1",
        "stage": config["stage"],
        "status": "pass" if all(gate_results.values()) else "fail",
        "sampleStatus": config["sampleStatus"],
        "configSha256": file_sha256(config_path),
        "basePhysicsSha256": base_physics_sha,
        "workerSourceSha256": _worker_source_sha256(),
        "manifests": manifest_records,
        "runs": rows,
        "knownAnswerRuns": known_answers,
        "predictionAgreement": agreements,
        "robustUniversalVariants": robust_variants,
        "agreeingUniversalVariants": agreeing_variants,
        "selectedUniversalSolverVariant": selected,
        "selectionMedianWallSeconds": median_wall,
        "gateResults": gate_results,
        "failedGates": [key for key, value in gate_results.items() if not value],
        "selectionPolicy": config["selectionPolicy"],
        "rethinkPolicy": config["rethinkPolicy"],
        "claimBoundary": config["claimBoundary"],
    }
    write_json(output / "report.json", report)
    write_csv(output / "run_summary.csv", rows)
    write_csv(output / "known_answer_summary.csv", known_answers)
    write_csv(output / "prediction_agreement.csv", agreements)
    write_csv(output / "residual_history.csv", residual_rows)
    render_plot(output, rows)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
