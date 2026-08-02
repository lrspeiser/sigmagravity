#!/usr/bin/env python3
"""Run the frozen P0727 Picard/Newton hybrid cross-check."""

from __future__ import annotations

import argparse
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
    run_known_answers,
    variant_manifest,
)

from voidscreen.field_job import _worker_source_sha256, canonical_sha256, file_sha256

DEFAULT_CONFIG = ROOT / "configs" / "p0727_hybrid_nonlinear_crosscheck.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0727_hybrid_nonlinear_crosscheck"
DEFAULT_STORE = ROOT / "tmp" / "p0724-http"
DEFAULT_WORK = ROOT / "tmp" / "p0727-worker-cache"


def render_plot(output: Path, comparisons: list[dict[str, Any]]) -> None:
    variants = list(dict.fromkeys(row["variant"] for row in comparisons))
    systems = list(dict.fromkeys(row["system"] for row in comparisons))
    by_key = {
        (row["variant"], row["system"]): row
        for row in comparisons
        if row["comparable"]
    }
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
    axis.set_title("P0727 hybrid circular-speed agreement with Picard reference")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("normalized RMS difference")
    figure.tight_layout()
    figure.savefig(output / "hybrid_reference_agreement.png", dpi=180)
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
            scientific = read_json(Path(row["output"]) / "scientific_result.json")
            metadata = scientific["numericalMetadata"]
            row["picardWarmupIterations"] = metadata["picard_warmup_iterations"]
            row["picardWarmupDamping"] = metadata["picard_warmup_damping"]
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
    warmup_by_variant = {
        item["id"]: int(item["picardWarmupIterations"])
        for item in config["solverVariants"]
    }
    selected = (
        min(qualifying, key=lambda variant: warmup_by_variant[variant])
        if qualifying
        else None
    )
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
        "schemaVersion": "sigma-p0727-hybrid-nonlinear-crosscheck/1",
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
