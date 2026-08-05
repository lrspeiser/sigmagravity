#!/usr/bin/env python3
"""Commission the exact-binmap recovery path across CCD and count regimes."""

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
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19w2b_cross_detector_response_commissioning.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19w2b_cross_detector_response_commissioning"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19w2b-cross-detector/v100")


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_parents(config: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"V19W2B parent hash mismatch for {name}: {actual}")
        hashes[name] = actual
    v19w_config = load_json(ROOT / config["parents"]["v19w_config"]["path"])
    v19w.validate_parent_hashes(v19w_config)
    v19w2_report = load_json(ROOT / config["parents"]["v19w2_report"]["path"])
    if not v19w2_report.get("full_missing_cell_recovery_authorized"):
        raise RuntimeError("V19W2 did not authorize broader exact-binmap recovery")
    runner = ROOT / config["execution"]["runner"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19W2B frozen runner path identifies another implementation")
    if sha256(runner) != config["execution"]["runner_sha256"]:
        raise RuntimeError("V19W2B frozen runner hash changed")
    return hashes


def snapshot_missing_names(snapshot: dict[str, Any]) -> set[str]:
    return {
        str(item["cell_name"])
        for item in snapshot["missing_manifest_tasks_at_or_before_maximum_completed_index"]
    }


def select_rows(
    config: dict[str, Any], manifest: list[dict[str, str]], snapshot: dict[str, Any]
) -> list[dict[str, str]]:
    basis = config["snapshot_basis"]
    missing_names = snapshot_missing_names(snapshot)
    if (
        int(snapshot["snapshot_completed_cells"]) != int(basis["completed_cells"])
        or int(snapshot["audited_product_files"]) != int(basis["audited_product_files"])
        or int(snapshot["maximum_completed_production_index"])
        != int(basis["maximum_completed_production_index"])
        or len(missing_names) != int(basis["missing_at_or_before_frontier"])
        or not all(snapshot["gates"].values())
    ):
        raise RuntimeError("V19W2B frozen snapshot basis changed")
    by_name = {v19w.cell_name(row): row for row in manifest}
    selected: list[dict[str, str]] = []
    for spec in config["commissioning_cells"]:
        name = spec["cell_name"]
        if name not in missing_names or name not in by_name:
            raise RuntimeError(f"V19W2B selected cell is not a frozen omission: {name}")
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
            raise RuntimeError(f"V19W2B manifest count changed for {name}: {observed}")
        selected.append(row)
    if len(selected) != 6 or len({v19w.task_key(row) for row in selected}) != 6:
        raise RuntimeError("V19W2B selection is not six unique manifest cells")
    if {int(row["ccd_id"]) for row in selected} != {0, 1, 2}:
        raise RuntimeError("V19W2B does not cover every previously uncommissioned CCD")
    if {int(row["obsid"]) for row in selected} != {4986, 5355, 5356, 5357}:
        raise RuntimeError("V19W2B does not cover the four new observation contexts")
    source_counts = [int(row["source_band_events"]) for row in selected]
    background_counts = [int(row["background_band_events"]) for row in selected]
    if min(source_counts) != 1 or max(source_counts) < 250:
        raise RuntimeError("V19W2B source-count extremes changed")
    if not any(value == 0 for value in background_counts) or not any(
        value > 0 for value in background_counts
    ):
        raise RuntimeError("V19W2B background-count regimes changed")
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
        raise RuntimeError("V19W2B scratch overlaps the protected base archive")
    free_bytes = shutil.disk_usage(Path(config["execution"]["free_space_probe"])).free
    if free_bytes < int(config["execution"]["minimum_free_bytes_at_launch"]):
        raise RuntimeError(f"V19W2B free-space gate failed: {free_bytes}")
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
            report = load_json(report_path)
            results.append(
                {
                    "cell_name": name,
                    "cluster": row["cluster"],
                    "obsid": int(row["obsid"]),
                    "ccd_id": int(row["ccd_id"]),
                    "source_band_events": int(row["source_band_events"]),
                    "background_band_events": int(row["background_band_events"]),
                    "response_position": report["response_position"],
                    "four_product_bytes": audited["four_product_bytes"],
                    "cell_report_sha256": audited["cell_report_sha256"],
                    "product_hashes": audited["product_hashes"],
                    "all_cell_gates_passed": all(report["gates"].values()),
                }
            )
        except Exception as exc:  # noqa: BLE001 - retain all commissioning failures
            failures[name] = f"{type(exc).__name__}: {exc}"
        progress = {
            "status": "v19w2b_cross_detector_commissioning_running",
            "completed_attempts": index,
            "expected_attempts": len(rows),
            "passed_cells": len(results),
            "failures": failures,
            "updated_utc": datetime.now(UTC).isoformat(),
            "base_archive_modified": False,
        }
        atomic_json(output / "progress.json", progress)
        print(f"V19W2B {index}/{len(rows)} passed={len(results)} failed={len(failures)}")
        sys.stdout.flush()

    ccds = {row["ccd_id"] for row in results}
    obsids = {row["obsid"] for row in results}
    source_counts = [row["source_band_events"] for row in results]
    background_counts = [row["background_band_events"] for row in results]
    gates = {
        "all_parent_hashes_exact": True,
        "v19w2_parent_passed": True,
        "snapshot_integrity_and_coverage_counts_exact": True,
        "all_six_cells_are_snapshot_omissions_with_exact_manifest_counts": len(results)
        == 6,
        "all_previously_uncommissioned_ccds_covered": ccds == {0, 1, 2},
        "all_four_new_observation_contexts_covered": obsids
        == {4986, 5355, 5356, 5357},
        "minimum_high_zero_and_positive_count_regimes_covered": (
            bool(source_counts)
            and min(source_counts) == 1
            and max(source_counts) >= 250
            and any(value == 0 for value in background_counts)
            and any(value > 0 for value in background_counts)
        ),
        "all_six_exact_masks_equal_frozen_integer_binmap_labels": len(results)
        == 6,
        "all_six_materialized_event_histograms_are_exact": len(results) == 6,
        "all_six_detector_medoids_map_to_declared_ccd": len(results) == 6
        and all(
            row["response_position"]["detector_medoid"]["mapped_ccd_id"]
            == row["ccd_id"]
            for row in results
        ),
        "all_six_arf_rmf_pha_link_scaling_size_and_hash_audits_pass": len(results)
        == 6
        and all(row["all_cell_gates_passed"] for row in results),
        "base_archive_remains_unmodified": True,
    }
    expected_gates = config["gates"]
    if set(gates) != set(expected_gates) or not all(expected_gates.values()):
        raise RuntimeError("V19W2B frozen gate schema changed")
    passed = all(gates.values()) and not failures
    report = {
        "status": (
            "cross_detector_exact_binmap_commissioning_passed"
            if passed
            else "cross_detector_exact_binmap_commissioning_failed_closed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": parent_hashes,
        "scratch_root": str(scratch),
        "launch_free_bytes": free_bytes,
        "selected_cells": [spec["cell_name"] for spec in config["commissioning_cells"]],
        "completed_cells": results,
        "failures": failures,
        "gates": gates,
        "v19w4_hardened_recovery_may_be_frozen": passed,
        "base_archive_modified": False,
        "spectrum_combined_or_fitted": False,
        "temperature_density_mach_or_speed_fitted": False,
        "lensing_halo_or_gravity_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(output / "report.json", report)
    if not passed:
        raise RuntimeError(f"V19W2B failed closed: {failures or gates}")
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
