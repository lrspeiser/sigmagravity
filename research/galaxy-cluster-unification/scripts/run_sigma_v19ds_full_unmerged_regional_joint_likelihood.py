#!/usr/bin/env python3
"""Run the frozen 494-region observation-resolved joint spectral likelihood."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_sigma_v19dp_unmerged_regional_joint_likelihood_preflight as v19dp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19ds_full_unmerged_regional_joint_likelihood.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19ds_full_unmerged_regional_joint_likelihood"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_frozen(config: dict[str, Any], config_path: Path) -> list[dict[str, str]]:
    if config["freeze_state"] != "frozen_after_terminal_v19dr_pass_before_any_new_region_fit":
        raise RuntimeError("V19DS configuration is not frozen at the V19DR boundary")
    if sha256(Path(__file__).resolve()) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DS runner changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DS parent changed after freeze: {name}")

    dr_report = load_json(ROOT / config["parents"]["v19dr_report"]["path"])
    if (
        dr_report["status"]
        != "full_256_cell_ccd7_background_archive_recovery_passed"
        or not dr_report["aggregate_pass"]
        or not dr_report["full_494_region_joint_likelihood_successor_authorized"]
        or dr_report["all_494_regions_run"]
        or dr_report["thermal_stress_or_baroclinicity_constructed"]
        or dr_report["lensing_halo_action_gravity_or_holdout_payload_opened"]
        or dr_report["gravity_formula_or_parameter_changed"]
    ):
        raise RuntimeError("V19DR no longer supplies the sealed production authorization")

    dp_config = load_json(ROOT / config["parents"]["v19dp_config"]["path"])
    for key in ("model", "clusters"):
        if config[key] != dp_config[key]:
            raise RuntimeError(f"V19DS changed the frozen V19DP {key}")
    products_path = ROOT / config["parents"]["v19dr_unified_index"]["path"]
    products = read_csv(products_path)
    expected_hash = config["parents"]["v19dr_unified_index"]["sha256"]
    if sha256(products_path) != expected_hash:
        raise RuntimeError("V19DS unified product index changed")
    if sha256(config_path) == "":  # pragma: no cover - explicit readability guard
        raise RuntimeError("V19DS configuration is unreadable")
    return products


def build_plan(
    config: dict[str, Any], products: list[dict[str, str]]
) -> list[tuple[str, int, list[dict[str, str]]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in products:
        grouped[(row["cluster"], int(row["bin_id"]))].append(row)
    plan = []
    seen_cells: set[str] = set()
    for (cluster, bin_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (int(row["obsid"]), int(row["ccd_id"])))
        for row in rows:
            if row["cell_name"] in seen_cells:
                raise RuntimeError(f"V19DS duplicate cell: {row['cell_name']}")
            seen_cells.add(row["cell_name"])
        plan.append((cluster, bin_id, rows))

    expected = config["registered_workload"]["clusters"]
    counts = Counter(cluster for cluster, _, _ in plan)
    cell_counts = Counter()
    for cluster, _, rows in plan:
        cell_counts[cluster] += len(rows)
    if len(plan) != int(config["registered_workload"]["total_regions"]):
        raise RuntimeError("V19DS total region count changed")
    if len(seen_cells) != int(config["registered_workload"]["total_cells"]):
        raise RuntimeError("V19DS unique cell count changed")
    for cluster, values in expected.items():
        if counts[cluster] != int(values["regions"]):
            raise RuntimeError(f"V19DS region count changed for {cluster}")
        if cell_counts[cluster] != int(values["cells"]):
            raise RuntimeError(f"V19DS cell count changed for {cluster}")
    return plan


def region_digest(
    config: dict[str, Any], cluster: str, bin_id: int, rows: list[dict[str, str]]
) -> str:
    return canonical_sha256(
        {
            "protocol_version": config["protocol_version"],
            "cluster": cluster,
            "bin_id": bin_id,
            "model": config["model"],
            "cells": [
                {
                    "cell_name": row["cell_name"],
                    "source_pha_sha256": row["source_pha_sha256"],
                    "background_pha_sha256": row["background_pha_sha256"],
                    "arf_sha256": row["arf_sha256"],
                    "rmf_sha256": row["rmf_sha256"],
                }
                for row in rows
            ],
        }
    )


def checkpoint_path(output: Path, cluster: str, bin_id: int) -> Path:
    return output / "checkpoints" / cluster / f"bin{bin_id}.json"


def validate_checkpoint(
    path: Path, digest: str, cluster: str, bin_id: int
) -> dict[str, Any]:
    result = load_json(path)
    if (
        result.get("input_digest") != digest
        or result.get("cluster") != cluster
        or int(result.get("bin_id", -1)) != bin_id
    ):
        raise RuntimeError(f"V19DS checkpoint changed: {path}")
    return result


def uncertainty_state(fit: dict[str, Any], config: dict[str, Any]) -> str:
    interval = fit["temperature_confidence_68_percent"]
    if interval["ordered"]:
        return "ordered_two_sided"
    temperature = float(fit["parameters"]["temperature_keV"])
    bounds = config["model"]["temperature_keV"]
    tolerance = float(config["production"]["parameter_bound_relative_tolerance"])
    if (
        abs(temperature / float(bounds["minimum"]) - 1.0) <= tolerance
        or abs(temperature / float(bounds["maximum"]) - 1.0) <= tolerance
    ):
        return "censored_at_frozen_model_bound"
    return "unresolved"


def quality_gates(fit: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    counts = [float(row["source_counts_in_fit_band"]) for row in fit["datasets"]]
    interval = fit["temperature_confidence_68_percent"]
    return {
        "finite_best_fit": all(
            math.isfinite(float(fit["parameters"][key]))
            for key in ("temperature_keV", "abundance_solar", "normalization")
        )
        and math.isfinite(float(fit["fit"]["statistic"]))
        and int(fit["fit"]["dof"]) > 0,
        "reduced_statistic_at_most_1_5": math.isfinite(
            float(fit["fit"]["reduced_statistic"])
        )
        and float(fit["fit"]["reduced_statistic"])
        <= float(config["quality_gates"]["maximum_reduced_statistic"]),
        "temperature_interval_ordered_and_precise": bool(interval["ordered"])
        and float(interval["fractional_half_width"])
        <= float(
            config["quality_gates"]["maximum_fractional_temperature_half_width"]
        ),
        "all_free_parameters_strictly_inside_bounds": bool(
            fit["all_free_parameters_strictly_inside_bounds"]
        ),
        "no_dataset_exceeds_30_percent_of_source_counts": max(counts) / sum(counts)
        <= float(config["quality_gates"]["maximum_single_dataset_count_fraction"]),
    }


def fit_one(
    config: dict[str, Any], cluster: str, bin_id: int, rows: list[dict[str, str]]
) -> dict[str, Any]:
    started = time.monotonic()
    digest = region_digest(config, cluster, bin_id, rows)
    try:
        fit = v19dp.fit_joint(config, cluster, rows, confidence=True)
        gates = quality_gates(fit, config)
        state = uncertainty_state(fit, config)
        return {
            "cluster": cluster,
            "bin_id": bin_id,
            "cells": len(rows),
            "input_digest": digest,
            "status": "fit_completed",
            "fit": fit,
            "uncertainty_state": state,
            "quality_gates": gates,
            "full_quality_pass": all(gates.values()),
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as exc:  # noqa: BLE001 - retain every regional failure
        return {
            "cluster": cluster,
            "bin_id": bin_id,
            "cells": len(rows),
            "input_digest": digest,
            "status": "fit_failed",
            "exception": f"{type(exc).__name__}: {exc}",
            "uncertainty_state": "unresolved",
            "quality_gates": {"finite_best_fit": False},
            "full_quality_pass": False,
            "elapsed_seconds": time.monotonic() - started,
        }


def summarize(
    config: dict[str, Any], rows: list[dict[str, Any]], reused: int
) -> dict[str, Any]:
    rows.sort(key=lambda row: (row["cluster"], int(row["bin_id"])))
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster"]].append(row)
    summaries = {}
    for cluster, expected in config["registered_workload"]["clusters"].items():
        cluster_rows = by_cluster[cluster]
        completed = [row for row in cluster_rows if row["status"] == "fit_completed"]
        reduced = [
            float(row["fit"]["fit"]["reduced_statistic"]) for row in completed
        ]
        temperatures = [
            float(row["fit"]["parameters"]["temperature_keV"]) for row in completed
        ]
        summaries[cluster] = {
            "expected_regions": int(expected["regions"]),
            "attempted_regions": len(cluster_rows),
            "completed_regions": len(completed),
            "failed_regions": len(cluster_rows) - len(completed),
            "response_cells": sum(int(row["cells"]) for row in cluster_rows),
            "ordered_uncertainty_regions": sum(
                row["uncertainty_state"] == "ordered_two_sided"
                for row in cluster_rows
            ),
            "censored_uncertainty_regions": sum(
                row["uncertainty_state"] == "censored_at_frozen_model_bound"
                for row in cluster_rows
            ),
            "unresolved_uncertainty_regions": sum(
                row["uncertainty_state"] == "unresolved" for row in cluster_rows
            ),
            "full_quality_pass_regions": sum(
                bool(row["full_quality_pass"]) for row in cluster_rows
            ),
            "reduced_statistic_range": [min(reduced), max(reduced)] if reduced else [],
            "temperature_keV_range": (
                [min(temperatures), max(temperatures)] if temperatures else []
            ),
        }

    minimum_quality = int(config["production_gates"]["minimum_quality_regions_per_cluster"])
    expected_regions = int(config["registered_workload"]["total_regions"])
    expected_cells = int(config["registered_workload"]["total_cells"])
    gates = {
        "all_494_registered_regions_attempted": len(rows) == expected_regions,
        "all_5082_response_cells_used_exactly_once": sum(
            int(row["cells"]) for row in rows
        )
        == expected_cells,
        "every_region_has_finite_best_fit": all(
            row["status"] == "fit_completed"
            and row["quality_gates"].get("finite_best_fit", False)
            for row in rows
        ),
        "every_region_has_explicit_usable_uncertainty_state": all(
            row["uncertainty_state"]
            in {"ordered_two_sided", "censored_at_frozen_model_bound"}
            for row in rows
        ),
        "each_cluster_has_minimum_full_quality_regions": all(
            summary["full_quality_pass_regions"] >= minimum_quality
            for summary in summaries.values()
        ),
        "region_and_cell_counts_match_each_cluster": all(
            summary["attempted_regions"] == int(expected["regions"])
            and summary["response_cells"] == int(expected["cells"])
            for cluster, expected in config["registered_workload"]["clusters"].items()
            for summary in [summaries[cluster]]
        ),
    }
    passed = all(gates.values())
    return {
        "status": (
            "full_494_region_unmerged_joint_likelihood_passed"
            if passed
            else "full_494_region_unmerged_joint_likelihood_failed_closed"
        ),
        "aggregate_pass": passed,
        "regions": rows,
        "cluster_summaries": summaries,
        "checkpoint_reused_regions": reused,
        "gates": gates,
        "i4_i5_source_only_successor_authorized": passed,
    }


def run(
    config: dict[str, Any], products: list[dict[str, str]], output: Path
) -> dict[str, Any]:
    plan = build_plan(config, products)
    results: list[dict[str, Any]] = []
    pending = []
    reused = 0
    for cluster, bin_id, rows in plan:
        digest = region_digest(config, cluster, bin_id, rows)
        checkpoint = checkpoint_path(output, cluster, bin_id)
        if checkpoint.is_file():
            results.append(validate_checkpoint(checkpoint, digest, cluster, bin_id))
            reused += 1
        else:
            pending.append((cluster, bin_id, rows))

    started = time.monotonic()
    workers = int(config["implementation"]["maximum_concurrent_regions"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fit_one, config, cluster, bin_id, rows): (cluster, bin_id)
            for cluster, bin_id, rows in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            cluster, bin_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001  # pragma: no cover
                result = {
                    "cluster": cluster,
                    "bin_id": bin_id,
                    "cells": 0,
                    "input_digest": "worker_crashed_before_digest_return",
                    "status": "fit_failed",
                    "exception": f"{type(exc).__name__}: {exc}",
                    "uncertainty_state": "unresolved",
                    "quality_gates": {"finite_best_fit": False},
                    "full_quality_pass": False,
                    "elapsed_seconds": 0.0,
                }
            write_json(checkpoint_path(output, cluster, bin_id), result)
            results.append(result)
            failures = sum(row["status"] != "fit_completed" for row in results)
            print(
                f"V19DS {reused + index}/{len(plan)}: {cluster} bin {bin_id}; "
                f"failures={failures}; elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    return summarize(config, results, reused)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        config = load_json(config_path)
        products = validate_frozen(config, config_path)
        result = run(config, products, output)
    except Exception as exc:  # noqa: BLE001 - terminal fail-closed report
        result = {
            "status": "full_494_region_unmerged_joint_likelihood_execution_failed",
            "aggregate_pass": False,
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "i4_i5_source_only_successor_authorized": False,
        }
    report = {
        "protocol_version": "SIGMA-V19DS-FULL-UNMERGED-JOINT-1.0.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "all_494_regions_run": len(result.get("regions", [])) == 494,
        "thermal_stress_or_baroclinicity_constructed": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    write_json(output / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "regions"}, indent=2))
    if not report["aggregate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
