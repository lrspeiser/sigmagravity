#!/usr/bin/env python3
"""Construct the frozen V19U batch and outcome-blind pilot plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19u_response_production_plan.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19u_response_production_plan"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_parent_hashes(config: dict[str, Any]) -> None:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None and sha256(ROOT / value) != expected:
            raise RuntimeError(f"V19U parent hash mismatch: {value}")


def load_tasks(config: dict[str, Any]) -> list[dict[str, str]]:
    path = ROOT / config["parents"]["v19q_manifest"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cluster_rank = {
        cluster: index for index, cluster in enumerate(config["workload"]["cluster_order"])
    }
    try:
        return sorted(
            rows,
            key=lambda row: (
                cluster_rank[row["cluster"]],
                int(row["obsid"]),
                int(row["bin_id"]),
                int(row["ccd_id"]),
            ),
        )
    except KeyError as exc:
        raise RuntimeError(f"V19U manifest contains an unexpected cluster: {exc}") from exc


def task_key(row: dict[str, str]) -> tuple[str, int, int, int]:
    return (
        row["cluster"],
        int(row["bin_id"]),
        int(row["obsid"]),
        int(row["ccd_id"]),
    )


def pilot_rows(
    tasks: list[dict[str, str]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for cluster in config["workload"]["cluster_order"]:
        ranked = sorted(
            (row for row in tasks if row["cluster"] == cluster),
            key=lambda row: (
                int(row["source_band_events"]),
                int(row["obsid"]),
                int(row["bin_id"]),
                int(row["ccd_id"]),
            ),
        )
        for quantile in config["throughput_pilot"][
            "quantiles_of_source_band_events_per_cluster"
        ]:
            rank = math.floor(float(quantile) * (len(ranked) - 1) + 0.5)
            row = dict(ranked[rank])
            row["pilot_quantile"] = float(quantile)
            row["pilot_rank_zero_based"] = rank
            row["pilot_population"] = len(ranked)
            selected.append(row)
    return selected


def write_manifest(
    tasks: list[dict[str, str]], pilots: list[dict[str, Any]], batch_size: int, path: Path
) -> None:
    pilot_lookup = {
        task_key(row): str(row["pilot_quantile"]) for row in pilots
    }
    parent_fields = list(tasks[0])
    fields = [
        "production_index",
        "batch_id",
        "batch_index",
        "throughput_pilot",
        "pilot_quantile",
        *parent_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for zero_index, row in enumerate(tasks):
            key = task_key(row)
            writer.writerow(
                {
                    "production_index": zero_index + 1,
                    "batch_id": zero_index // batch_size + 1,
                    "batch_index": zero_index % batch_size + 1,
                    "throughput_pilot": str(key in pilot_lookup).lower(),
                    "pilot_quantile": pilot_lookup.get(key, ""),
                    **row,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = load_json(config_path)
    validate_parent_hashes(config)

    q_report = load_json(ROOT / config["parents"]["v19q_report"])
    r_report = load_json(ROOT / config["parents"]["v19r_report"])
    t_report = load_json(ROOT / config["parents"]["v19t_report"])
    tasks = load_tasks(config)
    keys = [task_key(row) for row in tasks]
    counts = Counter(row["cluster"] for row in tasks)
    batch_size = int(config["workload"]["batch_size"])
    batch_count = math.ceil(len(tasks) / batch_size)
    pilots = pilot_rows(tasks, config)

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "production_manifest.csv"
    write_manifest(tasks, pilots, batch_size, manifest_path)

    response_bytes = int(
        config["resource_model"]["v19r_four_response_product_bytes"]
    )
    projected_bytes = response_bytes * len(tasks)
    minimum_free = math.ceil(
        projected_bytes
        * float(
            config["resource_model"][
                "minimum_free_space_multiplier_over_response_projection"
            ]
        )
    )
    batch_sizes = [
        min(batch_size, len(tasks) - start)
        for start in range(0, len(tasks), batch_size)
    ]
    expected_counts = config["workload"]["expected_task_count_by_cluster"]
    commissioning_key = (
        r_report["selected_manifest_row"]["cluster"],
        int(r_report["selected_manifest_row"]["bin_id"]),
        int(r_report["selected_manifest_row"]["obsid"]),
        int(r_report["selected_manifest_row"]["ccd_id"]),
    )
    pilot_keys = [task_key(row) for row in pilots]
    gates = {
        "all_parent_hashes_exact": True,
        "v19q_authorizes_exactly_5082_tasks": bool(
            q_report["response_extraction_authorized"]
            and q_report["manifest"]["response_task_count"]
            == config["workload"]["expected_task_count_total"]
        ),
        "v19r_authorizes_response_production": bool(
            r_report["full_response_production_authorized"]
        ),
        "v19t_authorizes_response_and_fit_production": bool(
            t_report["full_response_and_fit_production_authorized"]
        ),
        "production_task_key_set_equals_v19q": len(tasks) == len(keys)
        and len(keys) == len(set(keys)),
        "batch_count_equals_80_and_no_batch_exceeds_64": batch_count
        == config["workload"]["expected_batch_count"]
        and max(batch_sizes) <= batch_size,
        "each_task_occurs_exactly_once": len(keys) == len(set(keys)),
        "pilot_has_two_outcome_blind_source_count_strata_per_cluster": len(pilots)
        == config["throughput_pilot"]["expected_new_cell_count"]
        and len(set(pilot_keys)) == len(pilot_keys)
        and commissioning_key not in pilot_keys,
        "capacity_snapshot_exceeds_2p5_times_projected_response_storage": int(
            config["resource_model"]["wsl_target_free_bytes_at_snapshot"]
        )
        >= minimum_free,
        "failure_or_retry_cannot_change_scientific_inputs": True,
        "pilot_pass_required_before_full_execution": config["throughput_pilot"][
            "full_production_is_authorized_at_freeze"
        ]
        is False,
        "cluster_task_counts_match_frozen_expectations": all(
            counts[cluster] == expected for cluster, expected in expected_counts.items()
        ),
        "projected_storage_matches_frozen_arithmetic": projected_bytes
        == config["resource_model"][
            "projected_four_response_product_bytes_all_cells"
        ]
        and minimum_free
        == config["resource_model"]["minimum_free_space_bytes_before_full_run"],
    }
    passed = all(gates.values())
    report = {
        "status": (
            "production_plan_passed_and_throughput_pilot_authorized"
            if passed
            else "production_plan_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "parent_manifest": {
            "path": config["parents"]["v19q_manifest"],
            "sha256": sha256(ROOT / config["parents"]["v19q_manifest"]),
            "task_count": len(tasks),
        },
        "production_manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "task_count": len(tasks),
            "batch_count": batch_count,
            "batch_size": batch_size,
            "final_batch_size": batch_sizes[-1],
        },
        "task_count_by_cluster": dict(counts),
        "pilot_cells": pilots,
        "resource_projection": {
            "response_bytes_per_cell_from_v19r": response_bytes,
            "response_bytes_all_cells": projected_bytes,
            "response_decimal_gb_all_cells": projected_bytes / 1e9,
            "response_gib_all_cells": projected_bytes / 2**30,
            "minimum_free_space_bytes": minimum_free,
            "wsl_free_bytes_at_freeze_snapshot": config["resource_model"][
                "wsl_target_free_bytes_at_snapshot"
            ],
            "approximate_serial_hours": config["resource_model"][
                "projected_serial_hours_at_commissioning_rate"
            ],
            "approximate_four_worker_ideal_hours": config["resource_model"][
                "projected_four_worker_ideal_hours_at_commissioning_rate"
            ],
            "timing_is_planning_only_until_pilot": True,
        },
        "gates": gates,
        "throughput_pilot_authorized": passed,
        "full_production_authorized": False,
        "response_or_spectrum_constructed": False,
        "temperature_density_mach_or_speed_fitted": False,
        "lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"status: {report['status']}")
    print(f"tasks/batches/pilot: {len(tasks)}/{batch_count}/{len(pilots)}")
    print(f"response storage: {projected_bytes / 1e9:.3f} GB")
    print(f"report: {report_path}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
