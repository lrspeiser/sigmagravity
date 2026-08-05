#!/usr/bin/env python3
"""Checkpointed full-region successor to a passing V19X2 commissioning run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19x_spectral_combination_commissioning as inherited_v19x
import sigma_v19x2_unified_response_adapter as adapter

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_X2_STATUS = (
    "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def serialized_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_full_region_plan(
    config: dict[str, Any], manifest: list[dict[str, str]]
) -> dict[str, dict[int, list[dict[str, str]]]]:
    """Group every frozen response cell by cluster and accepted region."""

    grouped: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[tuple[str, int, int, int]] = set()
    for row in sorted(manifest, key=lambda item: int(item["production_index"])):
        key = adapter.task_key(row)
        if key in seen:
            raise RuntimeError(f"V19X3 manifest contains duplicate response cell: {key}")
        seen.add(key)
        grouped[key[0]][key[1]].append(row)

    expected_clusters = config["registered_workload"]["clusters"]
    if set(grouped) != set(expected_clusters):
        raise RuntimeError("V19X3 manifest cluster inventory changed")
    for cluster, expected in expected_clusters.items():
        if len(grouped[cluster]) != int(expected["total_regions"]):
            raise RuntimeError(f"V19X3 region count changed for {cluster}")
        cell_count = sum(len(rows) for rows in grouped[cluster].values())
        if cell_count != int(expected["total_cells"]):
            raise RuntimeError(f"V19X3 response-cell count changed for {cluster}")
    return {
        cluster: dict(sorted(regions.items()))
        for cluster, regions in sorted(grouped.items())
    }


def validate_x2_authorization(
    config: dict[str, Any], report_path: Path
) -> tuple[dict[str, Any], dict[str, float]]:
    if not report_path.is_file():
        raise RuntimeError("V19X2 commissioning report is absent")
    report = load_json(report_path)
    runtime = config["runtime_authorization"]
    if report.get("status") != AUTHORIZED_X2_STATUS:
        raise RuntimeError("V19X2 did not authorize full regional production")
    if report.get("config_sha256") != runtime["required_v19x2_config_sha256"]:
        raise RuntimeError("V19X2 report names another frozen config")
    if report.get("runner_sha256") != runtime["required_v19x2_runner_sha256"]:
        raise RuntimeError("V19X2 report names another runner")
    if not report.get("gates") or not all(report["gates"].values()):
        raise RuntimeError("V19X2 report contains a failed commissioning gate")
    if report.get("full_494_region_combination_and_fit_authorized") is not True:
        raise RuntimeError("V19X2 withheld regional-fit authorization")
    if report.get("replacement_cluster_lensing_target_opened") is not False:
        raise RuntimeError("V19X2 report opened a prohibited lensing target")
    fits = report.get("integrated_fits", [])
    by_cluster: dict[str, float] = {}
    for row in fits:
        if not row.get("fit_completed") or not row.get("gates", {}).get("all_passed"):
            raise RuntimeError("V19X2 integrated fit did not pass")
        abundance = float(row["parameters"]["abundance_solar"])
        if not math.isfinite(abundance):
            raise RuntimeError("V19X2 integrated abundance is non-finite")
        by_cluster[str(row["cluster"])] = abundance
    expected_clusters = set(config["registered_workload"]["clusters"])
    if set(by_cluster) != expected_clusters:
        raise RuntimeError("V19X2 integrated abundance inventory changed")
    return report, by_cluster


def region_input_digest(
    cluster: str,
    bin_id: int,
    cells: list[dict[str, Any]],
    abundance: float,
) -> str:
    payload = {
        "cluster": cluster,
        "bin_id": bin_id,
        "abundance_solar_fixed": abundance,
        "cells": [
            {
                "key": list(adapter.task_key(row)),
                "cell_name": row["cell_name"],
                "source_pha_sha256": row["source_pha_sha256"],
                "source_pha_total_counts": int(row["source_pha_total_counts"]),
            }
            for row in cells
        ],
    }
    return canonical_sha256(payload)


def validate_combination_checkpoint(
    record: dict[str, Any], expected_digest: str, expected_label: str
) -> dict[str, Any]:
    if record.get("input_digest") != expected_digest:
        raise RuntimeError(f"V19X3 changed checkpoint inputs for {expected_label}")
    combination = record.get("combination", {})
    if combination.get("label") != expected_label:
        raise RuntimeError(f"V19X3 changed checkpoint label for {expected_label}")
    for role in (
        "grouped_source_spectrum",
        "background_spectrum",
        "source_arf",
        "source_rmf",
    ):
        inherited_v19x.snapshot_path(combination, role)
    if (
        not combination.get("full_pha_count_conservation_exact")
        or combination.get("grouped_pha_links")
        != combination.get("expected_grouped_pha_links")
        or int(combination.get("frozen_snapshot", {}).get("files", -1)) != 4
    ):
        raise RuntimeError(f"V19X3 invalid combination checkpoint for {expected_label}")
    return combination


def validate_fit_checkpoint(
    record: dict[str, Any], expected_digest: str, combination: dict[str, Any]
) -> dict[str, Any]:
    if record.get("input_digest") != expected_digest:
        raise RuntimeError("V19X3 changed regional fit checkpoint inputs")
    fit = record.get("fit", {})
    if fit.get("label") != combination["label"]:
        raise RuntimeError("V19X3 changed regional fit checkpoint label")
    if fit.get("fit_completed"):
        source = inherited_v19x.snapshot_path(
            combination, "grouped_source_spectrum"
        )
        if fit.get("source_spectrum_sha256") != adapter.sha256(source):
            raise RuntimeError("V19X3 changed fitted source spectrum")
    return fit


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_region(
    config: dict[str, Any],
    output: Path,
    scratch: Path,
    cluster: str,
    bin_id: int,
    cells: list[dict[str, Any]],
    abundance: float,
) -> dict[str, Any]:
    label = f"{cluster}_bin{bin_id}"
    checkpoint_root = output / "checkpoints" / cluster / f"bin{bin_id}"
    combination_checkpoint = checkpoint_root / "combination.json"
    fit_checkpoint = checkpoint_root / "fit.json"
    digest = region_input_digest(cluster, bin_id, cells, abundance)

    if combination_checkpoint.is_file():
        combination = validate_combination_checkpoint(
            load_json(combination_checkpoint), digest, label
        )
        combination_reused = True
    else:
        combination = inherited_v19x.combine_aperture(
            label, cells, scratch, output, config
        )
        write_checkpoint(
            combination_checkpoint,
            {"input_digest": digest, "combination": combination},
        )
        combination_reused = False

    if fit_checkpoint.is_file():
        fit = validate_fit_checkpoint(load_json(fit_checkpoint), digest, combination)
        fit_reused = True
    else:
        try:
            fit = inherited_v19x.fit_spectrum(
                config, cluster, combination, abundance
            )
        except Exception as exc:  # noqa: BLE001 - retain terminal regional failure
            fit = inherited_v19x.failed_fit(cluster, label, exc)
        write_checkpoint(fit_checkpoint, {"input_digest": digest, "fit": fit})
        fit_reused = False
    return {
        "cluster": cluster,
        "bin_id": bin_id,
        "cells": len(cells),
        "input_digest": digest,
        "combination_reused": combination_reused,
        "fit_reused": fit_reused,
        "combination": combination,
        "fit": fit,
    }


def finite_best_fit(row: dict[str, Any]) -> bool:
    fit = row["fit"]
    if not fit.get("fit_completed"):
        return False
    parameters = fit.get("parameters", {})
    return all(
        math.isfinite(float(parameters.get(key, math.nan)))
        for key in ("temperature_keV", "abundance_solar", "normalization")
    )


def summarize_regions(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster"]].append(row)
    cluster_summaries: dict[str, dict[str, int]] = {}
    for cluster, expected in config["registered_workload"]["clusters"].items():
        cluster_rows = by_cluster[cluster]
        cluster_summaries[cluster] = {
            "expected_regions": int(expected["total_regions"]),
            "attempted_regions": len(cluster_rows),
            "finite_best_fit_regions": sum(finite_best_fit(row) for row in cluster_rows),
            "individual_quality_pass_regions": sum(
                bool(row["fit"].get("gates", {}).get("all_passed"))
                for row in cluster_rows
            ),
            "response_cells": sum(int(row["cells"]) for row in cluster_rows),
        }
    total_expected = sum(
        int(row["total_regions"])
        for row in config["registered_workload"]["clusters"].values()
    )
    registered_total = int(config["regional_gates"]["expected_total_regions"])
    total_cells = int(config["runtime_authorization"]["required_unified_cells"])
    minimum_quality = int(config["regional_gates"]["minimum_quality_passes_per_cluster"])
    gates = {
        "all_494_registered_regions_attempted": (
            len(rows) == total_expected == registered_total
        ),
        "every_unified_response_cell_used_in_one_region": sum(
            int(row["cells"]) for row in rows
        )
        == total_cells,
        "every_combination_conserved_counts_and_links": all(
            row["combination"]["full_pha_count_conservation_exact"]
            and row["combination"]["grouped_pha_links"]
            == row["combination"]["expected_grouped_pha_links"]
            and int(row["combination"]["frozen_snapshot"]["files"]) == 4
            for row in rows
        ),
        "every_region_has_finite_best_fit": all(finite_best_fit(row) for row in rows),
        "each_cluster_has_minimum_quality_passes": all(
            summary["individual_quality_pass_regions"] >= minimum_quality
            for summary in cluster_summaries.values()
        ),
        "region_and_cell_counts_match_each_cluster": all(
            summary["attempted_regions"] == summary["expected_regions"]
            and summary["response_cells"]
            == int(config["registered_workload"]["clusters"][cluster]["total_cells"])
            for cluster, summary in cluster_summaries.items()
        ),
    }
    return {
        "status": (
            "all_494_regional_spectra_and_finite_temperature_fits_passed_source_map_authorized"
            if all(gates.values())
            else "full_regional_spectral_production_gate_failed"
        ),
        "cluster_summaries": cluster_summaries,
        "gates": gates,
        "source_map_construction_authorized": all(gates.values()),
    }


def run_full_regional_production(
    config: dict[str, Any],
    output: Path,
    scratch: Path,
    plan: dict[str, dict[int, list[dict[str, str]]]],
    validated: dict[tuple[str, int, int, int], dict[str, Any]],
    abundances: dict[str, float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cluster, regions in plan.items():
        for index, (bin_id, manifest_rows) in enumerate(regions.items(), start=1):
            cells = [validated[adapter.task_key(row)] for row in manifest_rows]
            rows.append(
                process_region(
                    config,
                    output,
                    scratch,
                    cluster,
                    bin_id,
                    cells,
                    abundances[cluster],
                )
            )
            print(
                f"{cluster}: regional combination/fit {index}/{len(regions)}",
                flush=True,
            )
    return {"regions": rows, **summarize_regions(config, rows)}


def validate_frozen_runner(config: dict[str, Any]) -> None:
    if config.get("freeze_state") != "frozen_after_terminal_v19x2_pass":
        raise RuntimeError("V19X3 configuration is not frozen after a terminal V19X2 pass")
    runner = ROOT / config["implementation"]["runner"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19X3 configuration names another runner")
    if adapter.sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19X3 runner changed after freeze")


def validate_frozen_parents(config: dict[str, Any]) -> None:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None and adapter.sha256(ROOT / value) != expected:
            raise RuntimeError(f"V19X3 parent changed after freeze: {value}")


def execute(
    config: dict[str, Any], output: Path, scratch: Path
) -> dict[str, Any]:
    runtime = config["runtime_authorization"]
    x2_report_path = ROOT / runtime["required_v19x2_report"]
    x2_report, abundances = validate_x2_authorization(config, x2_report_path)
    v19w4_report, unified_index = adapter.authorize_unified_index(
        ROOT / runtime["required_v19w4_report"],
        expected_config_sha256=config["parents"]["v19w4_config_sha256"],
        expected_runner_sha256=config["parents"]["v19w4_runner_sha256"],
        expected_cells=int(runtime["required_unified_cells"]),
        expected_products=int(runtime["required_unified_products"]),
    )
    manifest = inherited_v19x.load_manifest(config)
    plan = build_full_region_plan(config, manifest)
    archives = {
        name: Path(path) for name, path in config["execution"]["response_archives"].items()
    }
    validated = adapter.validate_unified_archive(manifest, unified_index, archives)
    result = run_full_regional_production(
        config, output, scratch, plan, validated, abundances
    )
    result.update(
        {
            "v19x2_report_sha256": adapter.sha256(x2_report_path),
            "v19w4_report_sha256": adapter.sha256(
                ROOT / runtime["required_v19w4_report"]
            ),
            "v19w4_unified_index_sha256": v19w4_report["unified_product_index"][
                "sha256"
            ],
            "integrated_abundances_solar": abundances,
            "v19x2_status": x2_report["status"],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    validate_frozen_runner(config)
    validate_frozen_parents(config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = execute(config, output, args.scratch.resolve())
    except Exception as exc:  # noqa: BLE001 - retain terminal production failure
        result = {
            "status": "full_regional_spectral_production_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "source_map_construction_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": adapter.sha256(config_path),
        "runner_sha256": adapter.sha256(Path(__file__).resolve()),
        **result,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(f"status: {report['status']}")
    if not report["source_map_construction_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
