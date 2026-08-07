#!/usr/bin/env python3
"""Repeat V19DM after restoring the frozen parent statistic explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_sigma_v19dm_minimal_thermal_mixture_diagnostic as v19dm

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen(
    config: dict[str, Any], runner: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DM2 runner changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DM2 parent changed: {name}")
    parent_config_path = ROOT / config["parents"]["v19dm_config"]["path"]
    parent_runner = ROOT / config["parents"]["v19dm_runner"]["path"]
    parent_config = load_json(parent_config_path)
    science, v19dl_report = v19dm.validate_frozen(parent_config, parent_runner)
    parent_report = load_json(ROOT / config["parents"]["v19dm_report"]["path"])
    if parent_report["status"] != (
        "minimal_thermal_mixture_diagnostic_failed_no_successor_authorized"
    ):
        raise RuntimeError("V19DM does not retain the registered terminal failure")
    if parent_report["aggregate_pass"]:
        raise RuntimeError("V19DM unexpectedly passed")
    if "ui.set_stat" in parent_runner.read_text(encoding="utf-8"):
        raise RuntimeError("V19DM statistic-initialization defect is no longer present")
    if parent_report["all_494_regions_run"]:
        raise RuntimeError("V19DM unexpectedly ran all regions")
    if parent_report["lensing_halo_action_gravity_or_holdout_payload_opened"]:
        raise RuntimeError("V19DM unexpectedly opened a sealed payload")
    expected = str(science["fit_sequence"]["statistic"])
    if expected != config["remediation"]["expected_statistic"]:
        raise RuntimeError("V19DM2 expected statistic differs from the frozen parent")
    return science, v19dl_report, parent_report


def execute(
    config: dict[str, Any], science: dict[str, Any], v19dl_report: dict[str, Any]
) -> dict[str, Any]:
    expected = config["remediation"]["expected_statistic"]
    original_clean = v19dm.ui.clean

    def clean_with_frozen_statistic() -> None:
        original_clean()
        v19dm.ui.set_stat(expected)

    v19dm.ui.clean = clean_with_frozen_statistic
    try:
        result = v19dm.execute(config, science, v19dl_report)
    finally:
        v19dm.ui.clean = original_clean
    for row in result["integrated_model_selection"]:
        row["two_temperature"]["statistic_name"] = expected
    passed = bool(result["aggregate_pass"])
    result["status"] = (
        "statistic_parity_minimal_thermal_mixture_passed_successor_may_be_frozen"
        if passed
        else "statistic_parity_minimal_thermal_mixture_failed_no_successor_authorized"
    )
    result["v19dm_execution_result_scientifically_discarded"] = True
    result["remediation_scope"] = (
        "Set the frozen chi2xspecvar statistic immediately after every Sherpa clean; "
        "all data, models, starts, bounds, gates, and decisions are unchanged."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        config = load_json(config_path)
        science, v19dl_report, parent_report = validate_frozen(
            config, Path(__file__).resolve()
        )
        result = execute(config, science, v19dl_report)
        result["discarded_v19dm_report_sha256"] = sha256(
            ROOT / config["parents"]["v19dm_report"]["path"]
        )
        result["discarded_v19dm_status"] = parent_report["status"]
    except Exception as exc:  # noqa: BLE001 - preserve terminal audit evidence
        result = {
            "status": "statistic_parity_minimal_thermal_mixture_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
            "minimal_adequate_full_regional_successor_authorized": False,
        }
    report = {
        "protocol_version": "SIGMA-V19DM2-STATISTIC-PARITY-1.0.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "all_494_regions_run": False,
        "thermal_stress_or_baroclinicity_constructed": False,
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
