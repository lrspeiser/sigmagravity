#!/usr/bin/env python3
"""Commission the exact-binmap response path for absent blank-sky CCD 7."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_sigma_v19w_live_archive as v19w_audit
import run_sigma_v19w2_exact_binmap_response_commissioning as v19w2
import run_sigma_v19w_full_response_production as v19w

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19w2c_ccd7_response_commissioning.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19w2c_ccd7_response_commissioning"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19w2c-ccd7/v100")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_parents(config: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"V19W2C parent hash mismatch for {name}: {actual}")
        hashes[name] = actual
    base_config = load_json(ROOT / config["parents"]["v19w_config"]["path"])
    v19w.validate_parent_hashes(base_config)
    v19w2_report = load_json(ROOT / config["parents"]["v19w2_report"]["path"])
    if not v19w2_report.get("full_missing_cell_recovery_authorized"):
        raise RuntimeError("V19W2 did not authorize exact-binmap recovery")
    runner = ROOT / config["execution"]["runner"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19W2C frozen runner path identifies another implementation")
    if sha256(runner) != config["execution"]["runner_sha256"]:
        raise RuntimeError("V19W2C frozen runner hash changed")
    return hashes


def selected_quantile_names(
    manifest: list[dict[str, str]], obsid: int
) -> list[str]:
    candidates = sorted(
        (
            row
            for row in manifest
            if row["cluster"] == "ABELL2146"
            and int(row["obsid"]) == obsid
            and int(row["ccd_id"]) == 7
        ),
        key=lambda row: (int(row["source_band_events"]), int(row["production_index"])),
    )
    if len(candidates) != 128:
        raise RuntimeError(f"V19W2C CCD7 manifest population changed for ObsID {obsid}")
    positions = (0, (len(candidates) - 1) // 2, len(candidates) - 1)
    return [v19w.cell_name(candidates[position]) for position in positions]


def select_rows(
    config: dict[str, Any], manifest: list[dict[str, str]], snapshot: dict[str, Any]
) -> list[dict[str, str]]:
    basis = config["snapshot_basis"]
    exhausted = set(snapshot["exhausted_failed_cells_observed_after_snapshot"])
    observed_basis = {
        "completed_cells": int(snapshot["snapshot_completed_cells"]),
        "audited_product_files": int(snapshot["audited_product_files"]),
        "audited_product_bytes": int(snapshot["audited_product_bytes"]),
        "maximum_completed_production_index": int(
            snapshot["maximum_completed_production_index"]
        ),
        "failed_attempt_directories": int(
            snapshot["failed_attempt_directories_observed_after_snapshot"]
        ),
        "exhausted_failed_cells": len(exhausted),
        "snapshot_cell_names_sha256": snapshot["snapshot_cell_names_sha256"],
    }
    expected_basis = {
        key: basis[key]
        for key in (
            "completed_cells",
            "audited_product_files",
            "audited_product_bytes",
            "maximum_completed_production_index",
            "failed_attempt_directories",
            "exhausted_failed_cells",
            "snapshot_cell_names_sha256",
        )
    }
    if (
        observed_basis != expected_basis
        or snapshot["status"]
        != "completed_checkpoint_snapshot_passed_with_exhausted_base_production_cell"
        or not all(snapshot["gates"].values())
    ):
        raise RuntimeError("V19W2C frozen live snapshot basis changed")

    expected_names = []
    for obsid in config["selection_definition"]["observation_contexts"]:
        expected_names.extend(selected_quantile_names(manifest, int(obsid)))
    configured_names = [item["cell_name"] for item in config["commissioning_cells"]]
    if configured_names != expected_names:
        raise RuntimeError("V19W2C frozen minimum/median/maximum selection changed")

    by_name = {v19w.cell_name(row): row for row in manifest}
    selected: list[dict[str, str]] = []
    for spec in config["commissioning_cells"]:
        name = spec["cell_name"]
        if name not in exhausted or name not in by_name:
            raise RuntimeError(f"V19W2C selected cell is not an exhausted omission: {name}")
        row = by_name[name]
        expected = (
            int(spec["expected_production_index"]),
            int(spec["expected_source_band_events"]),
            int(spec["expected_background_band_events"]),
        )
        observed = (
            int(row["production_index"]),
            int(row["source_band_events"]),
            int(row["background_band_events"]),
        )
        if observed != expected:
            raise RuntimeError(f"V19W2C manifest count changed for {name}: {observed}")
        selected.append(row)

    if len(selected) != 6 or len({v19w.task_key(row) for row in selected}) != 6:
        raise RuntimeError("V19W2C selection is not six unique manifest cells")
    if {int(row["ccd_id"]) for row in selected} != {7}:
        raise RuntimeError("V19W2C selection is not restricted to CCD 7")
    if {int(row["obsid"]) for row in selected} != {10464, 10888}:
        raise RuntimeError("V19W2C observation coverage changed")
    source_counts = [int(row["source_band_events"]) for row in selected]
    background_counts = [int(row["background_band_events"]) for row in selected]
    if min(source_counts) != 22 or max(source_counts) != 532:
        raise RuntimeError("V19W2C source-count range changed")
    if any(value != 0 for value in background_counts):
        raise RuntimeError("V19W2C selected background-band regime changed")
    return selected


def execute_cell(
    row: dict[str, str], context: dict[str, Any], scratch: Path
) -> dict[str, Any]:
    prepared = v19w2.prepare_mask_cell(row, context, scratch)
    try:
        return v19w2.execute_mask_cell(prepared, scratch)
    except Exception:
        partial = scratch / "partial" / prepared["token"]
        failed = scratch / "failed_attempts" / f"{prepared['cell_name']}_attempt1"
        if partial.exists() and not failed.exists():
            failed.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(partial), str(failed))
        raise


def run(config_path: Path, output: Path, scratch: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    output = output.resolve()
    scratch = scratch.resolve()
    config = load_json(config_path)
    parent_hashes = verify_parents(config)
    base_scratch = Path(config["execution"]["protected_base_scratch"]).resolve()
    if scratch == base_scratch or scratch.is_relative_to(base_scratch):
        raise RuntimeError("V19W2C scratch overlaps the protected base archive")
    free_bytes = shutil.disk_usage(Path(config["execution"]["free_space_probe"])).free
    if free_bytes < int(config["execution"]["minimum_free_bytes_at_launch"]):
        raise RuntimeError(f"V19W2C free-space gate failed: {free_bytes}")

    base_config = load_json(ROOT / config["parents"]["v19w_config"]["path"])
    manifest = v19w.load_manifest(base_config)
    snapshot = load_json(ROOT / config["parents"]["live_snapshot_report"]["path"])
    rows = select_rows(config, manifest, snapshot)
    contexts = v19w.observation_contexts(base_config, rows, scratch)
    results: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        name = v19w.cell_name(row)
        try:
            execute_cell(row, contexts[(row["cluster"], int(row["obsid"]))], scratch)
            report_path = scratch / "completed" / name / "cell_report.json"
            audited = v19w_audit.validate_checkpoint(report_path, row)
            cell_report = load_json(report_path)
            background_all_energy = int(
                cell_report["materialized_event_subsets"]["background"][
                    "all_energy_rows"
                ]
            )
            results.append(
                {
                    "cell_name": name,
                    "cluster": row["cluster"],
                    "production_index": int(row["production_index"]),
                    "obsid": int(row["obsid"]),
                    "ccd_id": int(row["ccd_id"]),
                    "source_band_events": int(row["source_band_events"]),
                    "background_band_events": int(row["background_band_events"]),
                    "background_all_energy_events": background_all_energy,
                    "used_zero_background_path": cell_report["zero_background_steps"]
                    is not None,
                    "response_position": cell_report["response_position"],
                    "four_product_bytes": audited["four_product_bytes"],
                    "cell_report_sha256": audited["cell_report_sha256"],
                    "product_hashes": audited["product_hashes"],
                    "all_cell_gates_passed": all(cell_report["gates"].values()),
                }
            )
        except Exception as exc:  # noqa: BLE001 - retain all commissioning failures
            failures[name] = f"{type(exc).__name__}: {exc}"
        progress = {
            "status": "v19w2c_ccd7_commissioning_running",
            "completed_attempts": index,
            "expected_attempts": len(rows),
            "passed_cells": len(results),
            "failures": failures,
            "updated_utc": datetime.now(UTC).isoformat(),
            "base_archive_modified": False,
        }
        atomic_json(output / "progress.json", progress)
        print(
            f"V19W2C {index}/{len(rows)} passed={len(results)} "
            f"failed={len(failures)}"
        )
        sys.stdout.flush()

    ccds = {row["ccd_id"] for row in results}
    obsids = {row["obsid"] for row in results}
    source_counts = [row["source_band_events"] for row in results]
    gates = {
        "all_parent_hashes_exact": True,
        "v19w2_parent_passed": True,
        "live_snapshot_and_exhausted_failure_inventory_exact": True,
        "all_six_cells_are_outcome_blind_quantile_selections": len(results) == 6,
        "both_affected_observation_contexts_covered": obsids == {10464, 10888},
        "ccd7_and_source_count_range_covered": (
            ccds == {7}
            and bool(source_counts)
            and min(source_counts) == 22
            and max(source_counts) == 532
        ),
        "all_six_exact_masks_equal_frozen_integer_binmap_labels": len(results) == 6,
        "all_six_materialized_event_histograms_are_exact": len(results) == 6,
        "all_six_use_zero_all_energy_background_path": len(results) == 6
        and all(
            row["background_all_energy_events"] == 0
            and row["used_zero_background_path"]
            for row in results
        ),
        "all_six_detector_medoids_map_to_ccd7": len(results) == 6
        and all(
            row["response_position"]["detector_medoid"]["mapped_ccd_id"] == 7
            for row in results
        ),
        "all_six_arf_rmf_pha_link_scaling_size_and_hash_audits_pass": len(results)
        == 6
        and all(row["all_cell_gates_passed"] for row in results),
        "base_archive_remains_unmodified_by_commissioning": True,
    }
    expected_gates = config["gates"]
    if set(gates) != set(expected_gates) or not all(expected_gates.values()):
        raise RuntimeError("V19W2C frozen gate schema changed")
    passed = all(gates.values()) and not failures
    report = {
        "status": (
            "ccd7_exact_binmap_commissioning_passed"
            if passed
            else "ccd7_exact_binmap_commissioning_failed_closed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": parent_hashes,
        "scratch_root": str(scratch),
        "launch_free_bytes": free_bytes,
        "selected_cells": [item["cell_name"] for item in config["commissioning_cells"]],
        "completed_cells": results,
        "failures": failures,
        "gates": gates,
        "v19w5_hardened_recovery_may_be_frozen": passed,
        "base_archive_modified": False,
        "spectrum_combined_or_fitted": False,
        "temperature_density_mach_or_speed_fitted": False,
        "lensing_halo_or_gravity_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(output / "report.json", report)
    if not passed:
        raise RuntimeError(f"V19W2C failed closed: {failures or gates}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    report = run(args.config, args.output, args.scratch)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
