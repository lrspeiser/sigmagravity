#!/usr/bin/env python3
"""Audit the V19X3B successor without opening terminal measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19x3b_v19w5_regional_successor_preflight.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "sigma_v19x3b_v19w5_regional_successor_preflight"
    / "report.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def execute(config: dict[str, Any]) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    hash_gates: dict[str, bool] = {}
    for section in ("parents", "implementation"):
        values = config[section]
        for key, value in values.items():
            if key.endswith("_sha256"):
                continue
            expected = values.get(f"{key}_sha256")
            if expected is None:
                continue
            path = ROOT / value
            actual = sha256(path) if path.is_file() else "absent"
            hashes[key] = actual
            hash_gates[key] = actual == expected

    x4 = load_json(ROOT / config["parents"]["v19x4_config"])
    original_x3_preserved = all(
        x4["parents"][f"v19x3_{role}_sha256"]
        == hashes[f"original_v19x3_{role}"]
        for role in ("runner", "freezer")
    )
    gates = {
        "all_parent_and_implementation_hashes_exact": bool(hash_gates)
        and all(hash_gates.values()),
        "original_hash_bound_v19x3_chain_preserved": original_x3_preserved,
        "successor_paths_are_distinct": (
            config["implementation"]["runner"]
            != config["parents"]["original_v19x3_runner"]
            and config["implementation"]["freezer"]
            != config["parents"]["original_v19x3_freezer"]
        ),
        "v19w5_authority_is_mandatory": (
            config["successor_contract"]["response_authority"] == "V19W5"
            and config["successor_contract"]["recovery_archive"]
            == "v19w5_recovery"
            and not config["authorization"]["accept_v19w4_terminal_authority"]
        ),
        "all_494_regions_and_quality_rule_unchanged": (
            config["successor_contract"]["expected_regions"] == 494
            and config["successor_contract"]["minimum_quality_passes_per_cluster"]
            == 12
            and config["successor_contract"]["reuse_inherited_engine_byte_exact"]
        ),
        "terminal_measurements_and_targets_sealed": not any(
            config["authorization"][key]
            for key in (
                "open_v19x2_terminal_report_now",
                "open_regional_temperature_results_now",
                "open_lensing_or_halo_payload",
                "change_gravity_formula_or_parameter",
                "open_holdout",
            )
        ),
        "runtime_freeze_requires_passing_v19x2": config["authorization"][
            "freeze_only_after_terminal_v19x2_pass"
        ],
    }
    return {
        "protocol_version": config["protocol_version"],
        "status": (
            "v19x3b_v19w5_regional_successor_preflight_passed"
            if all(gates.values())
            else "v19x3b_v19w5_regional_successor_preflight_failed"
        ),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "hashes": hashes,
        "hash_gates": hash_gates,
        "gates": gates,
        "terminal_v19x2_or_regional_measurement_opened": False,
        "lensing_halo_or_gravity_payload_opened": False,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    global DEFAULT_CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    DEFAULT_CONFIG = args.config.resolve()
    report = execute(load_json(DEFAULT_CONFIG))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(report["status"])
    if not all(report["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
