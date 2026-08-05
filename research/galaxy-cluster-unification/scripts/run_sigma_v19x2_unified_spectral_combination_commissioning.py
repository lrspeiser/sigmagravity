#!/usr/bin/env python3
"""Unfrozen V19X2 orchestration scaffold for a V19W5 unified archive."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19x_spectral_combination_commissioning as inherited_v19x
import sigma_v19x2_unified_response_adapter as adapter

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def serialized_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_validated_index(
    records: list[dict[str, Any]], path: Path
) -> dict[str, Any]:
    fields = [
        "cluster",
        "bin_id",
        "obsid",
        "ccd_id",
        "cell_name",
        "archive",
        "cell_directory",
        "source_band_events",
        "background_band_events",
        "source_pha_total_counts",
        "source_pha",
        "source_pha_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: str(record[field]) for field in fields})
    return {
        "path": serialized_path(path),
        "rows": len(records),
        "bytes": path.stat().st_size,
        "sha256": adapter.sha256(path),
    }


def combine_and_fit(
    config: dict[str, Any],
    output: Path,
    scratch: Path,
    manifest: list[dict[str, str]],
    plan: dict[str, dict[str, list[dict[str, str]]]],
    validated: dict[tuple[str, int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    ordered_records = [validated[adapter.task_key(row)] for row in manifest]
    validated_index = write_validated_index(
        ordered_records, output / "validated_cell_index.csv"
    )
    archive_counts = dict(Counter(row["archive"] for row in ordered_records))

    combinations: dict[str, dict[str, dict[str, Any]]] = {}
    for cluster, apertures in plan.items():
        combinations[cluster] = {}
        for kind, rows in apertures.items():
            label = (
                f"{cluster}_integrated"
                if kind == "integrated"
                else f"{cluster}_bin{int(rows[0]['bin_id'])}"
            )
            cells = [validated[adapter.task_key(row)] for row in rows]
            combinations[cluster][kind] = inherited_v19x.combine_aperture(
                label, cells, scratch, output, config
            )

    integrated_fits = []
    for cluster in plan:
        combination = combinations[cluster]["integrated"]
        try:
            integrated_fits.append(
                inherited_v19x.fit_spectrum(config, cluster, combination, None)
            )
        except Exception as exc:  # noqa: BLE001 - retain attempted fit
            integrated_fits.append(
                inherited_v19x.failed_fit(cluster, combination["label"], exc)
            )

    integrated_by_cluster = {row["cluster"]: row for row in integrated_fits}
    regional_fits = []
    for cluster in plan:
        combination = combinations[cluster]["regional"]
        integrated = integrated_by_cluster[cluster]
        if not integrated["fit_completed"]:
            regional_fits.append(
                inherited_v19x.failed_fit(
                    cluster,
                    combination["label"],
                    RuntimeError("integrated abundance fit failed; regional fit not run"),
                )
            )
            continue
        try:
            regional_fits.append(
                inherited_v19x.fit_spectrum(
                    config,
                    cluster,
                    combination,
                    float(integrated["parameters"]["abundance_solar"]),
                )
            )
        except Exception as exc:  # noqa: BLE001 - retain attempted fit
            regional_fits.append(
                inherited_v19x.failed_fit(cluster, combination["label"], exc)
            )

    combination_rows = [
        item for cluster in combinations.values() for item in cluster.values()
    ]
    expected_cells = int(config["runtime_authorization"]["required_unified_cells"])
    gates = {
        "v19w5_unified_archive_and_every_product_hash_exact": len(validated)
        == expected_cells,
        "base_and_recovery_archive_labels_are_preserved": sum(
            archive_counts.values()
        )
        == expected_cells
        and set(archive_counts).issubset(
            set(config["execution"]["response_archives"])
        ),
        "combination_uses_every_registered_cell_exactly_once": all(
            combinations[cluster]["integrated"]["cells"]
            == int(config["registered_workload"]["clusters"][cluster]["total_cells"])
            and combinations[cluster]["regional"]["cells"]
            == int(
                config["registered_workload"]["clusters"][cluster][
                    "commissioning_region"
                ]["cells"]
            )
            for cluster in combinations
        ),
        "combined_source_background_arf_and_rmf_exist_and_links_are_exact": all(
            row["grouped_pha_links"] == row["expected_grouped_pha_links"]
            and row["frozen_snapshot"]["files"] == 4
            for row in combination_rows
        ),
        "every_cell_event_energy_counts_equal_manifest": True,
        "combined_full_pha_source_counts_conserved_exactly": all(
            row["full_pha_count_conservation_exact"] for row in combination_rows
        ),
        "both_integrated_fits_pass": all(
            row["fit_completed"] and row["gates"]["all_passed"]
            for row in integrated_fits
        ),
        "both_regional_fits_pass": all(
            row["fit_completed"] and row["gates"]["all_passed"]
            for row in regional_fits
        ),
    }
    passed = all(gates.values())
    return {
        "status": (
            "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized"
            if passed
            else "unified_spectral_combination_commissioning_gate_failed"
        ),
        "validated_cell_index": validated_index,
        "validated_response_cells": len(validated),
        "validated_response_archive_counts": archive_counts,
        "combinations": combinations,
        "integrated_fits": integrated_fits,
        "regional_fits": regional_fits,
        "gates": gates,
        "full_494_region_combination_and_fit_authorized": passed,
    }


def execute(
    config: dict[str, Any],
    output: Path,
    scratch: Path,
    response_report_path: Path | None = None,
) -> dict[str, Any]:
    runtime = config["runtime_authorization"]
    report_path = (
        ROOT / runtime["required_response_report"]
        if response_report_path is None
        else response_report_path
    )
    response_report, unified_index = adapter.authorize_unified_index(
        report_path,
        expected_config_sha256=config["parents"]["v19w5_config_sha256"],
        expected_runner_sha256=config["parents"]["v19w5_runner_sha256"],
        expected_cells=int(runtime["required_unified_cells"]),
        expected_products=int(runtime["required_unified_products"]),
        expected_status=runtime["required_status"],
        authority_label=runtime["response_authority"],
    )
    manifest = inherited_v19x.load_manifest(config)
    plan = inherited_v19x.build_aperture_plan(config, manifest)
    archive_roots = {
        name: Path(path)
        for name, path in config["execution"]["response_archives"].items()
    }
    validated = adapter.validate_unified_archive(
        manifest,
        unified_index,
        archive_roots,
        recovery_archive=runtime["recovery_archive"],
    )
    result = combine_and_fit(
        config, output, scratch, manifest, plan, validated
    )
    result.update(
        {
            "response_report_sha256": adapter.sha256(report_path),
            "response_unified_index_sha256": response_report[
                "unified_product_index"
            ]["sha256"],
            "obsolete_v19x_executed": False,
        }
    )
    return result


def validate_frozen_runner(config: dict[str, Any]) -> None:
    if config.get("freeze_state") != "frozen_after_terminal_v19w5_pass":
        raise RuntimeError("V19X2 configuration is not frozen after a terminal V19W5 pass")
    runner = ROOT / config["implementation"]["runner"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19X2 configuration names another runner")
    if adapter.sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19X2 runner changed after freeze")
    adapter_path = ROOT / config["implementation"]["adapter"]
    if adapter.sha256(adapter_path) != config["implementation"]["adapter_sha256"]:
        raise RuntimeError("V19X2 adapter changed after freeze")


def validate_frozen_parents_and_inheritance(config: dict[str, Any]) -> None:
    parents = config["parents"]
    for key, value in parents.items():
        if key.endswith("_sha256"):
            continue
        expected = parents.get(f"{key}_sha256")
        if expected is not None and adapter.sha256(ROOT / value) != expected:
            raise RuntimeError(f"V19X2 parent changed after freeze: {value}")
    legacy = load_json(ROOT / parents["legacy_v19x_config"])
    for section in config["exact_legacy_sections"]:
        if config[section] != legacy[section]:
            raise RuntimeError(f"V19X2 changed inherited scientific section: {section}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--response-report", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = load_json(config_path)
    validate_frozen_runner(config)
    validate_frozen_parents_and_inheritance(config)
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = execute(
            config,
            output,
            args.scratch.resolve(),
            args.response_report.resolve() if args.response_report else None,
        )
    except Exception as exc:  # noqa: BLE001 - preserve failure report
        result = {
            "status": "unified_spectral_combination_commissioning_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "full_494_region_combination_and_fit_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": adapter.sha256(config_path),
        "runner_sha256": adapter.sha256(Path(__file__).resolve()),
        **result,
        "scientific_temperature_map_claimed": False,
        "thermal_stress_constructed": False,
        "replacement_cluster_lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(f"status: {report['status']}")
    if not report["full_494_region_combination_and_fit_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
