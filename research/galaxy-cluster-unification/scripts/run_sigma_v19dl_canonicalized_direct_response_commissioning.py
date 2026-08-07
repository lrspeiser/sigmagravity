#!/usr/bin/env python3
"""Run full V19DJ commissioning with V19DK2 deterministic FITS snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_sigma_v19dj_direct_response_commissioning as v19dj
import run_sigma_v19dk_fits_canonicalization_preflight as v19dk
from sigma_v19dk_fits_canonical import canonicalize_fits

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen(config: dict[str, Any], runner: Path) -> dict[str, Any]:
    implementation = config["implementation"]
    if sha256(runner) != implementation["runner_sha256"]:
        raise RuntimeError("V19DL runner changed after freeze")
    canonicalizer = ROOT / implementation["canonicalizer"]
    if sha256(canonicalizer) != implementation["canonicalizer_sha256"]:
        raise RuntimeError("V19DL canonicalizer changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DL parent changed: {name}")

    v19dj_config_path = ROOT / config["parents"]["v19dj_config"]["path"]
    v19dj_runner = ROOT / config["parents"]["v19dj_runner"]["path"]
    science = v19dj.validate_frozen(load_json(v19dj_config_path), v19dj_runner)
    for section, expected in config["inherited_section_sha256"].items():
        actual = hashlib.sha256(
            json.dumps(science[section], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if actual != expected:
            raise RuntimeError(f"V19DL inherited science changed: {section}")

    preflight = load_json(ROOT / config["parents"]["v19dk2_report"]["path"])
    expected_status = (
        "grouped_fits_canonicalization_preflight_passed_full_successor_may_be_frozen"
    )
    if preflight["status"] != expected_status or not preflight["aggregate_pass"]:
        raise RuntimeError("V19DK2 preflight no longer authorizes V19DL")
    if not preflight["full_response_commissioning_successor_authorized"]:
        raise RuntimeError("V19DK2 did not authorize a full successor")
    for key in (
        "spectrum_fitted",
        "temperature_or_source_invariant_opened",
        "lensing_halo_action_gravity_or_holdout_payload_opened",
        "gravity_formula_or_parameter_changed",
    ):
        if preflight[key]:
            raise RuntimeError(f"V19DK2 unexpectedly opened sealed payload: {key}")
    return science


def execute(
    config: dict[str, Any],
    science: dict[str, Any],
    output: Path,
    scratch: Path,
) -> dict[str, Any]:
    spectra = v19dj.v19x2.inherited_v19x.inherited_spectra
    original: Callable[[Path, Path], dict[str, Any]] = spectra.copy_snapshot

    def canonical_copy_snapshot(source: Path, destination: Path) -> dict[str, Any]:
        canonicalize_fits(source, config["canonicalization"]["stable_history"])
        checksums = v19dk.checksum_gate(source)
        if not checksums["passed"]:
            raise RuntimeError(f"V19DL canonical checksum failed: {source}")
        item = original(source, destination)
        item["canonicalized_before_snapshot"] = True
        item["checksum_gate"] = checksums
        return item

    spectra.copy_snapshot = canonical_copy_snapshot
    try:
        result = v19dj.execute(config, science, output, scratch)
    finally:
        spectra.copy_snapshot = original

    combinations = [
        item for cluster in result["combinations"].values() for item in cluster.values()
    ]
    result["gates"]["every_snapshot_canonicalized_with_valid_checksums"] = all(
        product.get("canonicalized_before_snapshot", False)
        and product["checksum_gate"]["passed"]
        for combination in combinations
        for product in combination["frozen_snapshot"]["products"]
    )
    passed = all(result["gates"].values())
    result["status"] = (
        "canonicalized_direct_response_commissioning_passed_all_regions_authorized"
        if passed
        else "canonicalized_direct_response_commissioning_gate_failed"
    )
    result["full_494_region_combination_and_fit_authorized"] = passed
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
        if args.scratch.exists():
            raise RuntimeError(f"V19DL scratch must not already exist: {args.scratch}")
        args.scratch.mkdir(parents=True)
        result = execute(config, science, output, args.scratch.resolve())
    except Exception as exc:  # noqa: BLE001 - preserve terminal failure evidence
        result = {
            "status": "canonicalized_direct_response_commissioning_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "full_494_region_combination_and_fit_authorized": False,
        }
    report = {
        "protocol_version": "SIGMA-V19DL-CANONICAL-DIRECT-COMMISSIONING-1.0.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "thermal_stress_constructed": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
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
