#!/usr/bin/env python3
"""Rebuild every zero-background Abell 2146 CCD7 response product."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19dq_ccd7_background_recovery_preflight as v19dq
import run_sigma_v19w2_exact_binmap_response_commissioning as v19w2
import run_sigma_v19w_full_response_production as v19w

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19dr_full_ccd7_background_archive_recovery.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19dr_full_ccd7_background_archive_recovery"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19dr-ccd7-background/v100")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def selected_name_hash(rows: list[dict[str, str]]) -> str:
    payload = "".join(
        f"{row['cluster']}|{row['bin_id']}|{row['obsid']}|{row['ccd_id']}\n"
        for row in rows
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_frozen(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256(Path(__file__).resolve()) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DR runner changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DR parent changed: {name}")
    dq_config = load_json(ROOT / config["parents"]["v19dq_config"]["path"])
    dq_report = load_json(ROOT / config["parents"]["v19dq_report"]["path"])
    if (
        dq_report["status"] != "ccd7_real_background_recovery_preflight_passed"
        or not dq_report["full_ccd7_background_archive_recovery_successor_authorized"]
        or dq_report["next_required_stage"]
        != "rebuild_and_audit_all_256_zero_background_ccd7_products"
        or dq_report["full_494_region_joint_likelihood_successor_authorized"]
    ):
        raise RuntimeError("V19DQ does not authorize the full CCD7 recovery")
    base_config = load_json(ROOT / config["parents"]["v19w_config"]["path"])
    v19w.validate_parent_hashes(base_config)
    return dq_config, base_config


def select_rows(config: dict[str, Any], base_config: dict[str, Any]) -> list[dict[str, str]]:
    rows = sorted(
        (
            dict(row)
            for row in v19w.load_manifest(base_config)
            if row["cluster"] == "ABELL2146"
            and int(row["ccd_id"]) == 7
            and int(row["obsid"]) in {10464, 10888}
        ),
        key=lambda row: int(row["production_index"]),
    )
    selection = config["selection"]
    counts = Counter(int(row["obsid"]) for row in rows)
    source_counts = [int(row["source_band_events"]) for row in rows]
    if (
        len(rows) != int(selection["expected_cells"])
        or counts != Counter({10464: 128, 10888: 128})
        or selected_name_hash(rows) != selection["cell_names_sha256"]
        or min(source_counts) != int(selection["minimum_source_band_events"])
        or max(source_counts) != int(selection["maximum_source_band_events"])
        or any(int(row["background_band_events"]) != 0 for row in rows)
    ):
        raise RuntimeError("V19DR frozen 256-cell population changed")
    return rows


def prepare_cells(
    config: dict[str, Any], dq_config: dict[str, Any], base_config: dict[str, Any],
    rows: list[dict[str, str]], scratch: Path, output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contexts = v19dq.build_contexts(dq_config, rows, scratch)
    boundary = v19dq.audit_background_boundary(dq_config, contexts, scratch)
    if not all(row["passed"] for row in boundary):
        raise RuntimeError(f"V19DR background boundary failed: {boundary}")
    prepared = []
    for index, row in enumerate(rows, start=1):
        context = contexts[(row["cluster"], int(row["obsid"]))]
        cluster, bin_id, _obsid, ccd_id = v19w.task_key(row)
        env = v19w.inherited.isolated_environment(
            os.environ,
            scratch / "pfiles_count" / str(row["production_index"]),
            scratch / "tmp_count" / str(row["production_index"]),
        )
        binmap = v19w.v19p.region_product(context["region_row"], "binmap")
        mask_record = v19w2.write_exact_bin_mask(
            binmap,
            bin_id,
            scratch / "masks" / cluster / f"bin{bin_id}.fits",
            env,
            scratch / "mask_logs" / str(row["production_index"]) / "mask.log",
        )
        background_filter = (
            f"{context['background']}[ccd_id={ccd_id}]"
            f"[sky=mask({mask_record['path']})][energy=500:7000]"
        )
        background_events = v19w.inherited.event_count(background_filter, env)
        if background_events <= 0:
            raise RuntimeError(f"V19DR zero recovered background for {v19w.cell_name(row)}")
        updated = dict(row)
        updated["background_band_events"] = str(background_events)
        prepared.append(v19w2.prepare_mask_cell(updated, context, scratch))
        if index % 16 == 0 or index == len(rows):
            write_json(
                output / "progress.json",
                {
                    "status": "v19dr_preparing_exact_background_cells",
                    "prepared_cells": index,
                    "expected_cells": len(rows),
                    "updated_utc": datetime.now(UTC).isoformat(),
                },
            )
            print(f"V19DR prepared {index}/{len(rows)}")
            sys.stdout.flush()
    return prepared, boundary


def execute_one(cell: dict[str, Any], scratch: Path) -> dict[str, Any]:
    try:
        return v19w2.execute_mask_cell(cell, scratch)
    except Exception:
        partial = scratch / "partial" / cell["token"]
        failed = scratch / "failed_attempts" / f"{cell['cell_name']}_attempt1"
        if partial.exists() and not failed.exists():
            failed.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(partial), str(failed))
        raise


def execute_cells(
    config: dict[str, Any], prepared: list[dict[str, Any]], scratch: Path, output: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    completed = []
    failures = {}
    workers = int(config["implementation"]["maximum_concurrent_cells"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute_one, cell, scratch): cell for cell in prepared}
        for index, future in enumerate(as_completed(futures), start=1):
            cell = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:  # noqa: BLE001 - retain every terminal failure
                failures[cell["cell_name"]] = f"{type(exc).__name__}: {exc}"
            write_json(
                output / "progress.json",
                {
                    "status": "v19dr_extracting_real_ccd7_backgrounds",
                    "completed_attempts": index,
                    "expected_attempts": len(prepared),
                    "passed_cells": len(completed),
                    "failed_cells": len(failures),
                    "updated_utc": datetime.now(UTC).isoformat(),
                },
            )
            if index % 8 == 0 or index == len(prepared):
                print(
                    f"V19DR extracted {index}/{len(prepared)} "
                    f"passed={len(completed)} failed={len(failures)}"
                )
                sys.stdout.flush()
    completed.sort(key=lambda row: int(row["production_index"]))
    return completed, failures


def recovery_rows(
    completed: list[dict[str, Any]], scratch: Path
) -> list[dict[str, Any]]:
    rows = []
    for report in completed:
        name = report["cell_name"]
        products = report["products"]
        rows.append(
            {
                "production_index": int(report["production_index"]),
                "cluster": report["cluster"],
                "bin_id": int(report["bin_id"]),
                "obsid": int(report["obsid"]),
                "ccd_id": int(report["ccd_id"]),
                "cell_name": name,
                "cell_directory": str(scratch / "completed" / name),
                "cell_report_sha256": sha256(
                    scratch / "completed" / name / "cell_report.json"
                ),
                "source_band_events": int(
                    report["materialized_event_subsets"]["source"][
                        "band_500_7000_rows"
                    ]
                ),
                "background_all_energy_events": int(
                    report["materialized_event_subsets"]["background"][
                        "all_energy_rows"
                    ]
                ),
                "background_band_events": int(
                    report["materialized_event_subsets"]["background"][
                        "band_500_7000_rows"
                    ]
                ),
                "blanksky_scale": float(report["blanksky_scaling"]["BKGSCALn"]),
                "effective_background_scale": float(
                    report["blanksky_scaling"]["effective_background_scale"]
                ),
                "source_pha_name": products["source_pha"]["name"],
                "source_pha_sha256": products["source_pha"]["sha256"],
                "background_pha_name": products["background_pha"]["name"],
                "background_pha_sha256": products["background_pha"]["sha256"],
                "arf_name": products["arf"]["name"],
                "arf_sha256": products["arf"]["sha256"],
                "rmf_name": products["rmf"]["name"],
                "rmf_sha256": products["rmf"]["sha256"],
                "four_product_bytes": int(report["four_product_bytes"]),
                "all_cell_gates_passed": all(report["gates"].values()),
            }
        )
    return rows


def unified_rows(
    config: dict[str, Any], recovered: list[dict[str, Any]]
) -> list[dict[str, str]]:
    parent = read_csv(ROOT / config["parents"]["v19w5_product_index"]["path"])
    replacements = {row["cell_name"]: row for row in recovered}
    fields = list(parent[0])
    result = []
    for original in parent:
        replacement = replacements.get(original["cell_name"])
        if replacement is None:
            result.append(dict(original))
            continue
        row = dict(original)
        row["archive"] = "v19dr_real_ccd7_background"
        row["cell_directory"] = str(replacement["cell_directory"])
        row["cell_report_sha256"] = str(replacement["cell_report_sha256"])
        row["four_product_bytes"] = str(replacement["four_product_bytes"])
        for key in (
            "source_pha_name",
            "source_pha_sha256",
            "background_pha_name",
            "background_pha_sha256",
            "arf_name",
            "arf_sha256",
            "rmf_name",
            "rmf_sha256",
        ):
            row[key] = str(replacement[key])
        product_dir = Path(row["cell_directory"]) / "products"
        for name_key, bytes_key in (
            ("source_pha_name", "source_pha_bytes"),
            ("background_pha_name", "background_pha_bytes"),
            ("arf_name", "arf_bytes"),
            ("rmf_name", "rmf_bytes"),
        ):
            row[bytes_key] = str((product_dir / row[name_key]).stat().st_size)
        result.append(row)
    result.sort(key=lambda row: int(row["production_index"]))
    if (
        len(result) != 5082
        or len({row["cell_name"] for row in result}) != 5082
        or len(replacements) != 256
        or sum(row["archive"] == "v19dr_real_ccd7_background" for row in result)
        != 256
        or set(result[0]) != set(fields)
    ):
        raise RuntimeError("V19DR unified product index is structurally invalid")
    return result


def run(config_path: Path, output: Path, scratch: Path) -> dict[str, Any]:
    config = load_json(config_path)
    dq_config, base_config = validate_frozen(config)
    free_bytes = shutil.disk_usage(Path(config["implementation"]["free_space_probe"])).free
    if free_bytes < int(config["implementation"]["minimum_free_bytes_at_launch"]):
        raise RuntimeError(f"V19DR free-space gate failed: {free_bytes}")
    rows = select_rows(config, base_config)
    prepared, boundary = prepare_cells(
        config, dq_config, base_config, rows, scratch, output
    )
    completed, failures = execute_cells(config, prepared, scratch, output)
    recovered = recovery_rows(completed, scratch)
    recovery_fields = list(recovered[0]) if recovered else []
    recovery_path = output / "recovery_index.csv"
    if recovered:
        write_csv(recovery_path, recovered, recovery_fields)
    product_path = output / "unified_product_index.csv"
    unified = []
    if len(recovered) == 256 and not failures:
        unified = unified_rows(config, recovered)
        write_csv(product_path, unified, list(unified[0]))

    scales_exact = all(
        abs(row["effective_background_scale"] / row["blanksky_scale"] - 1.0)
        <= 1e-6
        for row in recovered
    )
    gates = {
        "v19dq_two_cell_recovery_parent_passes": True,
        "exactly_256_frozen_ccd7_cells_selected": len(rows) == 256,
        "both_observations_contribute_128_cells": Counter(
            int(row["obsid"]) for row in recovered
        )
        == Counter({10464: 128, 10888: 128}),
        "every_recovered_background_is_nonzero": len(recovered) == 256
        and all(row["background_band_events"] > 0 for row in recovered),
        "every_cell_response_and_pha_audit_passes": len(recovered) == 256
        and all(row["all_cell_gates_passed"] for row in recovered),
        "every_particle_scale_is_exact": len(recovered) == 256 and scales_exact,
        "no_recovery_failure_remains": not failures,
        "unified_index_has_5082_unique_cells": len(unified) == 5082,
        "unified_index_replaces_exactly_256_ccd7_cells": len(unified) == 5082
        and sum(row["archive"] == "v19dr_real_ccd7_background" for row in unified)
        == 256,
    }
    passed = all(gates.values())
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "status": (
            "full_256_cell_ccd7_background_archive_recovery_passed"
            if passed
            else "full_ccd7_background_archive_recovery_failed_closed"
        ),
        "launch_free_bytes": free_bytes,
        "background_boundary": boundary,
        "selected_cells": len(rows),
        "completed_cells": len(recovered),
        "failures": failures,
        "source_band_event_range": [
            min(row["source_band_events"] for row in recovered) if recovered else None,
            max(row["source_band_events"] for row in recovered) if recovered else None,
        ],
        "background_band_event_range": [
            min(row["background_band_events"] for row in recovered)
            if recovered
            else None,
            max(row["background_band_events"] for row in recovered)
            if recovered
            else None,
        ],
        "total_background_band_events": sum(
            row["background_band_events"] for row in recovered
        ),
        "recovery_index": {
            "path": str(recovery_path.relative_to(ROOT)),
            "rows": len(recovered),
            "sha256": sha256(recovery_path) if recovery_path.is_file() else None,
        },
        "unified_product_index": {
            "path": str(product_path.relative_to(ROOT)),
            "rows": len(unified),
            "sha256": sha256(product_path) if product_path.is_file() else None,
        },
        "gates": gates,
        "aggregate_pass": passed,
        "full_494_region_joint_likelihood_successor_authorized": passed,
        "all_494_regions_run": False,
        "thermal_stress_or_baroclinicity_constructed": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "claim_boundary": config["claim_boundary"],
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    try:
        report = run(args.config.resolve(), args.output.resolve(), args.scratch.resolve())
    except Exception as exc:
        report = {
            "protocol_version": "SIGMA-V19DR-FULL-CCD7-BACKGROUND-RECOVERY-1.0.0",
            "generated_utc": datetime.now(UTC).isoformat(),
            "status": "full_ccd7_background_archive_recovery_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
            "full_494_region_joint_likelihood_successor_authorized": False,
            "all_494_regions_run": False,
            "thermal_stress_or_baroclinicity_constructed": False,
            "lensing_halo_action_gravity_or_holdout_payload_opened": False,
            "gravity_formula_or_parameter_changed": False,
        }
        write_json(args.output.resolve() / "report.json", report)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
