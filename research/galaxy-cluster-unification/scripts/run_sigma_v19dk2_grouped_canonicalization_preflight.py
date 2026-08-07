#!/usr/bin/env python3
"""Remediate V19DK's grouped-PHA comparator alias without changing science."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_sigma_v19di_direct_ogip_writer_preflight as v19di
import run_sigma_v19dk_fits_canonicalization_preflight as v19dk

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen(config: dict[str, Any], runner: Path) -> dict[str, Any]:
    implementation = config["implementation"]
    if v19dk.sha256(runner) != implementation["runner_sha256"]:
        raise RuntimeError("V19DK2 runner changed after freeze")
    canonicalizer = ROOT / implementation["canonicalizer"]
    if v19dk.sha256(canonicalizer) != implementation["canonicalizer_sha256"]:
        raise RuntimeError("V19DK2 canonicalizer changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if v19dk.sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DK2 parent changed: {name}")
    index = ROOT / config["input"]["validated_cell_index"]
    if v19dk.sha256(index) != config["input"]["validated_cell_index_sha256"]:
        raise RuntimeError("V19DK2 validated-cell index changed")

    parent_config_path = ROOT / config["parents"]["v19dk_config"]["path"]
    parent_runner = ROOT / config["parents"]["v19dk_runner"]["path"]
    science = v19dk.validate_frozen(load_json(parent_config_path), parent_runner)
    terminal = load_json(ROOT / config["parents"]["v19dk_report"]["path"])
    if terminal["status"] != "fits_canonicalization_preflight_execution_failed":
        raise RuntimeError("V19DK terminal status changed")
    for key in (
        "spectrum_fitted",
        "temperature_or_source_invariant_opened",
        "lensing_halo_action_gravity_or_holdout_payload_opened",
        "gravity_formula_or_parameter_changed",
    ):
        if terminal[key]:
            raise RuntimeError(f"V19DK unexpectedly opened sealed payload: {key}")
    return science


def grouped_alias_compare(left: Path, right: Path) -> dict[str, Any]:
    """Expose grouped PHA counts through the legacy `_src.pi` comparator API."""
    if not left.name.endswith("_src_grp.pi") or not right.name.endswith("_src_grp.pi"):
        raise RuntimeError("V19DK2 received an unexpected grouped-PHA name")
    left_alias = left.with_name(left.name.removesuffix("_grp.pi") + ".pi")
    right_alias = right.with_name(right.name.removesuffix("_grp.pi") + ".pi")
    if not left_alias.is_file():
        raise RuntimeError(f"V19DK2 raw ungrouped source is missing: {left_alias}")
    if right_alias.exists():
        raise RuntimeError(f"V19DK2 canonical alias already exists: {right_alias}")
    shutil.copy2(right, right_alias)
    return grouped_alias_compare.original(left_alias, right_alias)


grouped_alias_compare.original = v19di.v19dg.compare_products


def execute(
    config: dict[str, Any],
    science: dict[str, Any],
    output: Path,
    scratch: Path,
) -> dict[str, Any]:
    original = v19di.v19dg.compare_products
    v19di.v19dg.compare_products = grouped_alias_compare
    try:
        result = v19dk.execute(config, science, output, scratch)
    finally:
        v19di.v19dg.compare_products = original
    passed = bool(result["aggregate_pass"])
    result["status"] = (
        "grouped_fits_canonicalization_preflight_passed_full_successor_may_be_frozen"
        if passed
        else "grouped_fits_canonicalization_preflight_failed_no_successor_authorized"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        config = load_json(config_path)
        science = validate_frozen(config, Path(__file__).resolve())
        result = execute(config, science, output, args.scratch.resolve())
    except Exception as exc:  # noqa: BLE001 - preserve terminal failure evidence
        result = {
            "status": "grouped_fits_canonicalization_preflight_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
            "full_response_commissioning_successor_authorized": False,
        }
    report = {
        "protocol_version": "SIGMA-V19DK2-GROUPED-CANONICALIZATION-PREFLIGHT-1.0.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": v19dk.sha256(config_path),
        "runner_sha256": v19dk.sha256(Path(__file__).resolve()),
        **result,
        "spectrum_fitted": False,
        "temperature_or_source_invariant_opened": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(f"status: {report['status']}")
    if not report["aggregate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
