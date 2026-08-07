#!/usr/bin/env python3
"""Recover the real CCD7 blank-sky background and repeat the joint preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_sigma_v19dp_unmerged_regional_joint_likelihood_preflight as v19dp
import run_sigma_v19w2_exact_binmap_response_commissioning as v19w2
import run_sigma_v19w_full_response_production as v19w

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19dq_ccd7_background_recovery_preflight.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19dq_ccd7_background_recovery_preflight"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19dq-ccd7-background/v100")


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


def validate_frozen(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    runner = Path(__file__).resolve()
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DQ runner changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DQ parent changed: {name}")
    for name, item in config["background_inputs"].items():
        for role in ("pre_reprojection", "post_reprojection"):
            path = Path(item[role]["path"])
            if sha256(path) != item[role]["sha256"]:
                raise RuntimeError(f"V19DQ {name} {role} input changed")

    parent = load_json(ROOT / config["parents"]["v19dp_report"]["path"])
    if parent["status"] != "unmerged_regional_joint_likelihood_preflight_failed":
        raise RuntimeError("V19DQ requires the terminal failed V19DP parent")
    failures = {
        row["cluster"]: [name for name, passed in row["gates"].items() if not passed]
        for row in parent["regions"]
    }
    if failures != {"BULLET": [], "ABELL2146": ["reduced_statistic_at_most_1_5"]}:
        raise RuntimeError(f"V19DP failure boundary changed: {failures}")
    abell = next(row for row in parent["regions"] if row["cluster"] == "ABELL2146")
    loo = {row["omitted_cell"]: row for row in abell["leave_one_observation_out"]}
    if (
        loo["ABELL2146_bin62_obs10464_ccd7"]["reduced_statistic"] > 1.5
        or loo["ABELL2146_bin62_obs10888_ccd7"]["reduced_statistic"]
        >= abell["primary"]["fit"]["reduced_statistic"]
    ):
        raise RuntimeError("V19DP no longer localizes the failure to CCD7")
    v19dp_config = load_json(ROOT / config["parents"]["v19dp_config"]["path"])
    return parent, v19dp_config


def build_contexts(
    config: dict[str, Any], rows: list[dict[str, str]], scratch: Path
) -> dict[tuple[str, int], dict[str, Any]]:
    base_config = load_json(ROOT / config["parents"]["v19w_config"]["path"])
    contexts = v19w.observation_contexts(base_config, rows, scratch)
    for obsid_text, item in config["background_inputs"].items():
        key = ("ABELL2146", int(obsid_text))
        contexts[key]["background"] = Path(item["pre_reprojection"]["path"])
        contexts[key]["background_geometry"] = {
            "remediation": "use the astrometry-corrected source-excluded blank sky before the lossy common-grid reprojection",
            "pre_reprojection": item["pre_reprojection"],
            "post_reprojection": item["post_reprojection"],
        }
    return contexts


def audit_background_boundary(
    config: dict[str, Any], contexts: dict[tuple[str, int], dict[str, Any]], scratch: Path
) -> list[dict[str, Any]]:
    records = []
    for obsid_text, item in config["background_inputs"].items():
        obsid = int(obsid_text)
        env = v19w.inherited.isolated_environment(
            os.environ,
            scratch / "pfiles_audit" / obsid_text,
            scratch / "tmp_audit" / obsid_text,
        )
        pre = Path(item["pre_reprojection"]["path"])
        post = Path(item["post_reprojection"]["path"])
        pre_ccd7 = v19w.inherited.event_count(f"{pre}[ccd_id=7]", env)
        post_ccd7 = v19w.inherited.event_count(f"{post}[ccd_id=7]", env)
        record = {
            "obsid": obsid,
            "pre_reprojection_ccd7_rows": pre_ccd7,
            "post_reprojection_ccd7_rows": post_ccd7,
            "expected_pre_reprojection_ccd7_rows": int(
                item["expected_pre_reprojection_ccd7_rows"]
            ),
            "expected_post_reprojection_ccd7_rows": int(
                item["expected_post_reprojection_ccd7_rows"]
            ),
            "pre_reprojection_background": str(contexts[("ABELL2146", obsid)]["background"]),
        }
        record["passed"] = (
            pre_ccd7 == record["expected_pre_reprojection_ccd7_rows"]
            and post_ccd7 == record["expected_post_reprojection_ccd7_rows"]
            and pre_ccd7 > 0
            and post_ccd7 == 0
        )
        records.append(record)
    return records


def extract_remediated_cells(
    config: dict[str, Any], scratch: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    base_config = load_json(ROOT / config["parents"]["v19w_config"]["path"])
    manifest = v19w.load_manifest(base_config)
    by_name = {v19w.cell_name(row): row for row in manifest}
    selected = []
    for item in config["registered_cells"]:
        row = dict(by_name[item["cell_name"]])
        if (
            int(row["source_band_events"]) != int(item["expected_source_band_events"])
            or int(row["background_band_events"]) != 0
            or int(row["ccd_id"]) != 7
        ):
            raise RuntimeError(f"V19DQ frozen manifest boundary changed: {item['cell_name']}")
        row["background_band_events"] = str(item["expected_background_band_events"])
        selected.append(row)
    contexts = build_contexts(config, selected, scratch)
    boundary = audit_background_boundary(config, contexts, scratch)
    if not all(row["passed"] for row in boundary):
        raise RuntimeError(f"V19DQ CCD7 background boundary failed: {boundary}")

    completed = []
    for row, item in zip(selected, config["registered_cells"], strict=True):
        key = (row["cluster"], int(row["obsid"]))
        prepared = v19w2.prepare_mask_cell(row, contexts[key], scratch)
        result = v19w2.execute_mask_cell(prepared, scratch)
        observed = result["materialized_event_subsets"]["background"]
        if (
            int(observed["all_energy_rows"])
            != int(item["expected_background_all_energy_events"])
            or int(observed["band_500_7000_rows"])
            != int(item["expected_background_band_events"])
            or result["zero_background_steps"] is not None
            or not all(result["gates"].values())
        ):
            raise RuntimeError(f"V19DQ remediated extraction failed: {item['cell_name']}")
        completed.append(result)
    return completed, selected, boundary


def remediated_product_index(
    config: dict[str, Any], completed: list[dict[str, Any]], scratch: Path
) -> list[dict[str, str]]:
    products = read_csv(ROOT / config["parents"]["v19w5_product_index"]["path"])
    by_name = {row["cell_name"]: row for row in products}
    for result in completed:
        name = result["cell_name"]
        row = by_name[name]
        directory = scratch / "completed" / name
        product_dir = directory / "products"
        row["archive"] = "v19dq_real_ccd7_background"
        row["cell_directory"] = str(directory)
        row["cell_report_sha256"] = sha256(directory / "cell_report.json")
        total = 0
        for role, filename_key, bytes_key, hash_key in (
            ("source", "source_pha_name", "source_pha_bytes", "source_pha_sha256"),
            ("background", "background_pha_name", "background_pha_bytes", "background_pha_sha256"),
            ("arf", "arf_name", "arf_bytes", "arf_sha256"),
            ("rmf", "rmf_name", "rmf_bytes", "rmf_sha256"),
        ):
            suffix = {
                "source": f"{name}.pi",
                "background": f"{name}_bkg.pi",
                "arf": f"{name}.arf",
                "rmf": f"{name}.rmf",
            }[role]
            path = product_dir / suffix
            row[filename_key] = path.name
            row[bytes_key] = str(path.stat().st_size)
            row[hash_key] = sha256(path)
            total += path.stat().st_size
        row["four_product_bytes"] = str(total)
    return products


def run(config_path: Path, output: Path, scratch: Path) -> dict[str, Any]:
    config = load_json(config_path)
    parent, v19dp_config = validate_frozen(config)
    completed, _, boundary = extract_remediated_cells(config, scratch)
    products = remediated_product_index(config, completed, scratch)
    v19dl_report, _, _ = v19dp.validate_frozen(
        v19dp_config, ROOT / config["parents"]["v19dp_runner"]["path"]
    )
    joint = v19dp.execute(v19dp_config, v19dl_report, products)
    abell = next(row for row in joint["regions"] if row["cluster"] == "ABELL2146")
    bullet = next(row for row in joint["regions"] if row["cluster"] == "BULLET")
    extraction_pass = all(all(row["gates"].values()) for row in completed)
    gates = {
        "v19dp_failure_is_only_abell_joint_goodness": True,
        "pre_reprojection_ccd7_background_is_present": True,
        "common_grid_reprojection_loss_is_reproduced": True,
        "both_registered_ccd7_background_subsets_are_nonzero_and_exact": extraction_pass,
        "both_remediated_response_archives_pass": extraction_pass,
        "bullet_joint_preflight_still_passes": bool(bullet["passed"]),
        "abell_joint_preflight_passes_after_background_recovery": bool(abell["passed"]),
        "corrected_two_cluster_joint_preflight_passes": bool(joint["aggregate_pass"]),
    }
    passed = all(gates.values())
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "status": (
            "ccd7_real_background_recovery_preflight_passed"
            if passed
            else "ccd7_real_background_recovery_preflight_failed_closed"
        ),
        "parent_v19dp_status": parent["status"],
        "background_boundary": boundary,
        "remediated_cells": completed,
        "joint_preflight": joint,
        "gates": gates,
        "aggregate_pass": passed,
        "full_ccd7_background_archive_recovery_successor_authorized": passed,
        "full_494_region_joint_likelihood_successor_authorized": False,
        "next_required_stage": (
            "rebuild_and_audit_all_254_affected_ccd7_background_products"
            if passed
            else "diagnose_registered_ccd7_background_recovery_failure"
        ),
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
            "protocol_version": "SIGMA-V19DQ-CCD7-BACKGROUND-RECOVERY-1.0.0",
            "generated_utc": datetime.now(UTC).isoformat(),
            "status": "ccd7_real_background_recovery_preflight_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
            "full_ccd7_background_archive_recovery_successor_authorized": False,
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
