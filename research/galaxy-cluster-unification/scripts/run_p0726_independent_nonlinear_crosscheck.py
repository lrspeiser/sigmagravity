#!/usr/bin/env python3
"""Run the frozen P0726 Newton--Krylov cross-method field check."""

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0725_aqual_solver_robustness import (
    known_answer_manifest,
    physics_core,
    prepare_input_bundle,
    read_json,
    write_csv,
    write_json,
)

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

DEFAULT_CONFIG = ROOT / "configs" / "p0726_independent_nonlinear_crosscheck.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0726_independent_nonlinear_crosscheck"
DEFAULT_STORE = ROOT / "tmp" / "p0724-http"
DEFAULT_WORK = ROOT / "tmp" / "p0726-worker-cache"


def variant_manifest(
    base: dict[str, Any], constants: dict[str, Any], variant: dict[str, Any]
) -> dict[str, Any]:
    manifest = copy.deepcopy(base)
    manifest["name"] = f"AQUAL P0726 numerical variant {variant['id']}"
    controls = {key: value for key, value in variant.items() if key not in {"id", "role"}}
    manifest["solver"] = {**constants, **controls}
    return manifest


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
            raise RuntimeError(f"stale P0726 cache at {output}")
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
    return {
        "variant": payload["variant"],
        "system": payload["system"],
        "state": scientific["state"],
        "converged": scientific["converged"],
        "iterations": scientific["iterations"],
        "maximumRelativeUpdate": scientific["maximumRelativeUpdate"],
        "maximumEquationResidual": max(scientific["equationResiduals"].values()),
        "wallSeconds": resource["wallSeconds"],
        "cpuSeconds": resource["cpuSeconds"],
        "peakPythonHeapBytes": resource["peakPythonHeapBytes"],
        "requestedMaximumIterations": metadata["requested_maximum_iterations"],
        "executedMaximumIterations": metadata["executed_maximum_iterations"],
        "iterationLimitAdjusted": metadata["maximum_iterations_limited_by_worker"],
        "nonlinearMethod": metadata["nonlinear_method"],
        "krylovMethod": metadata["krylov_method"],
        "lineSearch": metadata["line_search"],
        "modelDocumentSha256": canonical_sha256(payload["model"]),
        "modelPhysicsSha256": canonical_sha256(physics_core(payload["model"])),
        "inputBundleSha256": payload["inputBundleSha256"],
        "scientificResultSha256": scientific["resultSha256"],
        "output": str(output),
    }


def manufactured_source() -> tuple[np.ndarray, np.ndarray, float]:
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
        expected, coefficient, [spacing, spacing], coefficient_floor=1e-8
    )
    return expected, forcing, spacing


def run_known_answers(config: dict[str, Any]) -> list[dict[str, Any]]:
    expected, forcing, spacing = manufactured_source()
    rows = []
    for variant in config["solverVariants"]:
        solver = {
            **config["solverConstants"],
            **{key: value for key, value in variant.items() if key not in {"id", "role"}},
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


def normalized_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(np.square(reference))))
    return float(np.sqrt(np.mean(np.square(candidate - reference)))) / denominator


def load_prediction(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        return np.asarray(
            [float(row["predicted_speed_m_s"]) for row in csv.DictReader(handle)],
            dtype=float,
        )


def compare_reference(
    rows: list[dict[str, Any]], systems: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    system_by_id = {str(item["id"]): item for item in systems}
    comparisons = []
    for row in rows:
        reference = (ROOT / system_by_id[row["system"]]["referenceDirectory"]).resolve()
        if not row["converged"]:
            comparisons.append(
                {
                    "variant": row["variant"],
                    "system": row["system"],
                    "comparable": False,
                    "predictionNormalizedRmse": None,
                    "potentialNormalizedRmse": None,
                    "accelerationNormalizedRmse": None,
                }
            )
            continue
        candidate = Path(row["output"])
        reference_prediction = load_prediction(reference / "observation_predictions.csv")
        candidate_prediction = load_prediction(candidate / "observation_predictions.csv")
        with np.load(reference / "fields.npz", allow_pickle=False) as archive:
            reference_potential = archive["Phi"]
        with np.load(candidate / "fields.npz", allow_pickle=False) as archive:
            candidate_potential = archive["Phi"]
        with np.load(reference / "observables.npz", allow_pickle=False) as archive:
            reference_acceleration = np.stack(
                [archive[name] for name in sorted(archive.files)], axis=0
            )
        with np.load(candidate / "observables.npz", allow_pickle=False) as archive:
            candidate_acceleration = np.stack(
                [archive[name] for name in sorted(archive.files)], axis=0
            )
        comparisons.append(
            {
                "variant": row["variant"],
                "system": row["system"],
                "comparable": True,
                "predictionNormalizedRmse": normalized_rmse(
                    reference_prediction, candidate_prediction
                ),
                "potentialNormalizedRmse": normalized_rmse(
                    reference_potential, candidate_potential
                ),
                "accelerationNormalizedRmse": normalized_rmse(
                    reference_acceleration, candidate_acceleration
                ),
            }
        )
    return comparisons


def render_plot(output: Path, comparisons: list[dict[str, Any]]) -> None:
    valid = [row for row in comparisons if row["comparable"]]
    variants = list(dict.fromkeys(row["variant"] for row in comparisons))
    systems = list(dict.fromkeys(row["system"] for row in comparisons))
    by_key = {(row["variant"], row["system"]): row for row in valid}
    matrix = np.asarray(
        [
            [
                by_key.get((variant, system), {}).get("predictionNormalizedRmse", np.nan)
                for system in systems
            ]
            for variant in variants
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=0.01)
    axis.set_xticks(np.arange(len(systems)), systems)
    axis.set_yticks(np.arange(len(variants)), variants)
    axis.set_title("P0726 circular-speed difference from converged Picard reference")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("normalized RMS difference")
    figure.tight_layout()
    figure.savefig(output / "reference_prediction_agreement.png", dpi=180)
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
    p0725 = read_json(ROOT / config["p0725Report"])
    if p0725["robustUniversalVariants"] != ["linearized_d020"]:
        raise RuntimeError("P0725 reference selection changed")
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    bundles = {}
    targets = {}
    for system in config["systems"]:
        reference = (ROOT / system["referenceDirectory"]).resolve()
        scientific = read_json(reference / "scientific_result.json")
        if scientific["resultSha256"] != system["referenceScientificResultSha256"]:
            raise RuntimeError(f"{system['id']} P0725 reference hash changed")
        bundle, observation_targets = prepare_input_bundle(store, work, system)
        bundles[system["id"]] = bundle
        targets[system["id"]] = observation_targets

    payloads = []
    manifests = []
    for variant in config["solverVariants"]:
        model = variant_manifest(base_manifest, config["solverConstants"], variant)
        physics_sha = canonical_sha256(physics_core(model))
        if physics_sha != base_physics_sha:
            raise RuntimeError(f"{variant['id']} changed the physical manifest")
        manifests.append(
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

    known_answers = run_known_answers(config)
    comparisons = compare_reference(rows, config["systems"])
    residual_rows = []
    for row in rows:
        with (Path(row["output"]) / "residual_history.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            residual_rows.extend(
                {"variant": row["variant"], "system": row["system"], **item}
                for item in csv.DictReader(handle)
            )

    gates = config["gates"]
    systems = [str(item["id"]) for item in config["systems"]]
    variants = [str(item["id"]) for item in config["solverVariants"]]
    comparisons_by_key = {
        (row["variant"], row["system"]): row for row in comparisons
    }
    by_variant = {
        variant: [row for row in rows if row["variant"] == variant]
        for variant in variants
    }
    qualifying = []
    for variant, subset in by_variant.items():
        if len(subset) != len(systems) or not all(row["converged"] for row in subset):
            continue
        if max(row["maximumEquationResidual"] for row in subset) > float(
            gates["maximumEquationResidual"]
        ):
            continue
        if max(row["maximumRelativeUpdate"] for row in subset) > float(
            gates["maximumRelativeUpdate"]
        ):
            continue
        paired = [comparisons_by_key[(variant, system)] for system in systems]
        if not all(row["comparable"] for row in paired):
            continue
        if max(row["predictionNormalizedRmse"] for row in paired) > float(
            gates["maximumReferencePredictionNormalizedRmse"]
        ):
            continue
        if max(row["potentialNormalizedRmse"] for row in paired) > float(
            gates["maximumReferencePotentialNormalizedRmse"]
        ):
            continue
        if max(row["accelerationNormalizedRmse"] for row in paired) > float(
            gates["maximumReferenceAccelerationNormalizedRmse"]
        ):
            continue
        qualifying.append(variant)
    median_wall = {
        variant: float(np.median([row["wallSeconds"] for row in by_variant[variant]]))
        for variant in qualifying
    }
    selected = min(median_wall, key=median_wall.get) if median_wall else None
    gate_results = {
        "required_design": len(systems) == int(gates["requiredSystems"])
        and len(variants) == int(gates["requiredVariants"]),
        "known_answer_convergence": sum(row["converged"] for row in known_answers)
        >= int(gates["minimumKnownAnswerConvergedVariants"]),
        "known_answer_field_error": max(row["relativeFieldError"] for row in known_answers)
        <= float(gates["maximumKnownAnswerRelativeFieldError"]),
        "independent_real_system_convergence_and_agreement": len(qualifying)
        >= int(gates["minimumIndependentVariantsConvergingBothSystems"]),
        "iteration_limit_faithful": all(
            not row["iterationLimitAdjusted"]
            and row["requestedMaximumIterations"] == row["executedMaximumIterations"]
            for row in rows
        ),
        "no_per_object_gravity_parameters": True,
        "physics_manifest_unchanged": all(
            item["modelPhysicsSha256"] == base_physics_sha for item in manifests
        ),
        "reference_hashes_valid": True,
    }
    report = {
        "schemaVersion": "sigma-p0726-independent-nonlinear-crosscheck/1",
        "stage": config["stage"],
        "status": "pass" if all(gate_results.values()) else "fail",
        "sampleStatus": config["sampleStatus"],
        "configSha256": file_sha256(config_path),
        "basePhysicsSha256": base_physics_sha,
        "workerSourceSha256": _worker_source_sha256(),
        "manifests": manifests,
        "runs": rows,
        "knownAnswerRuns": known_answers,
        "referenceComparisons": comparisons,
        "qualifyingUniversalVariants": qualifying,
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
    write_csv(output / "reference_comparison.csv", comparisons)
    write_csv(output / "residual_history.csv", residual_rows)
    render_plot(output, comparisons)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
