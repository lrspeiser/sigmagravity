#!/usr/bin/env python3
"""Audit the V19X4B gas successor before terminal regional measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19x4b_v19x3b_gas_successor_preflight.json"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "sigma_v19x4b_v19x3b_gas_successor_preflight"
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

    original = load_json(ROOT / config["parents"]["original_v19x4_config"])
    original_report = load_json(
        ROOT / config["parents"]["original_v19x4_preflight_report"]
    )
    x3b_preflight = load_json(ROOT / config["parents"]["v19x3b_preflight_report"])
    science = config["successor_contract"]
    gates = {
        "all_parent_and_implementation_hashes_exact": bool(hash_gates)
        and all(hash_gates.values()),
        "original_v19x4_math_preflight_still_passes": (
            original_report.get("status")
            == "gas_state_math_preflight_passed_awaiting_v19x3_measurements"
            and original_report.get("gates")
            and all(original_report["gates"].values())
        ),
        "v19x3b_successor_preflight_passes": (
            x3b_preflight.get("status")
            == "v19x3b_v19w5_regional_successor_preflight_passed"
            and x3b_preflight.get("gates")
            and all(x3b_preflight["gates"].values())
        ),
        "gas_posterior_rules_are_identical": all(
            science["science_section_sha256"][section]
            == hashlib.sha256(
                json.dumps(
                    original[section], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            for section in science["science_sections"]
        ),
        "full_uncertainty_and_grid_contract_retained": (
            original["posterior"]["draws"] == 4096
            and original["posterior"]["rank_correlations"] == [-0.9, 0.0, 0.9]
            and original["common_grid"]["cells_per_axis"] == 241
            and original["common_grid"]["smoothing_fwhm_kpc"] == [50.0, 100.0]
        ),
        "terminal_measurements_and_targets_sealed": not any(
            config["authorization"][key]
            for key in (
                "freeze_before_terminal_v19x3b_pass",
                "open_regional_temperature_result_now",
                "construct_observed_gas_posterior_now",
                "open_lensing_or_halo_payload",
                "change_gravity_formula_or_parameter",
                "open_holdout",
            )
        ),
        "separately_named_successor_required": (
            config["implementation"]["runner"]
            != config["parents"]["original_v19x4_runner"]
            and config["authorization"]["freeze_after_terminal_v19x3b_pass"]
        ),
    }
    return {
        "protocol_version": config["protocol_version"],
        "status": (
            "v19x4b_v19x3b_gas_successor_preflight_passed"
            if all(gates.values())
            else "v19x4b_v19x3b_gas_successor_preflight_failed"
        ),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "hashes": hashes,
        "hash_gates": hash_gates,
        "gates": gates,
        "terminal_regional_or_gas_measurement_opened": False,
        "lensing_halo_gravity_or_holdout_payload_opened": False,
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
