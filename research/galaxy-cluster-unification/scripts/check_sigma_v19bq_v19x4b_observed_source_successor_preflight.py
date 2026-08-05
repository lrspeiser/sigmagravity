#!/usr/bin/env python3
"""Audit V19BQ before terminal gas, stellar or source results exist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bq_v19x4b_observed_source_successor_preflight.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19bq_v19x4b_observed_source_successor_preflight" / "report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def execute(config: dict[str, Any], config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
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
    original = load_json(ROOT / config["parents"]["original_v19bp_config"])
    bp_preflight = load_json(ROOT / config["parents"]["original_v19bp_preflight_report"])
    x4b_preflight = load_json(ROOT / config["parents"]["v19x4b_preflight_report"])
    bmb_preflight = load_json(ROOT / config["parents"]["v19bmb_preflight_report"])
    contract = config["successor_contract"]
    gates = {
        "all_parent_and_implementation_hashes_exact": bool(hash_gates) and all(hash_gates.values()),
        "original_v19bp_preflight_still_passes": (
            bp_preflight.get("decision") == "passed_observed_source_executor_preflight_awaiting_terminal_inputs"
            and bp_preflight.get("gates") and all(bp_preflight["gates"].values())
        ),
        "v19x4b_and_v19bmb_preflights_pass": (
            x4b_preflight.get("status") == "v19x4b_v19x3b_gas_successor_preflight_passed"
            and x4b_preflight.get("gates") and all(x4b_preflight["gates"].values())
            and bmb_preflight.get("status") == "v19bmb_v19x4b_stellar_successor_preflight_passed"
            and bmb_preflight.get("gates") and all(bmb_preflight["gates"].values())
        ),
        "source_decision_sections_are_identical": all(
            contract["science_section_sha256"][section] == canonical_sha256(original[section])
            for section in contract["science_sections"]
        ),
        "full_branch_variant_and_candidate_contract_retained": (
            original["registered_inputs"]["clusters"] == ["BULLET", "ABELL2146"]
            and original["registered_inputs"]["rank_correlations"] == [-0.9, 0.0, 0.9]
            and len(original["variants"]["smoothing_fwhm_kpc"]) * len(original["variants"]["radii_kpc"]) == 6
            and original["registered_inputs"]["candidates"] == ["I4_THERMODYNAMIC_GRADIENT_STRESS", "I5_BAROCLINICITY"]
        ),
        "terminal_products_and_targets_sealed": not any(
            config["authorization"][key]
            for key in (
                "freeze_before_terminal_v19x4b_and_v19bmb_pass",
                "run_observed_source_score_now",
                "open_lensing_or_halo_payload",
                "derive_action_now",
                "change_gravity_formula_or_parameter",
                "open_holdout",
            )
        ),
        "separately_named_successor_required": (
            config["implementation"]["runner"] != config["parents"]["original_v19bp_runner"]
            and config["authorization"]["freeze_after_terminal_v19x4b_and_v19bmb_pass"]
        ),
    }
    return {
        "protocol_version": config["protocol_version"],
        "status": (
            "v19bq_observed_source_successor_preflight_passed"
            if all(gates.values()) else "v19bq_observed_source_successor_preflight_failed"
        ),
        "config_sha256": sha256(config_path),
        "hashes": hashes,
        "hash_gates": hash_gates,
        "terminal_gas_stellar_or_source_result_opened": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gates": gates,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    report = execute(load_json(config_path), config_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(report["status"])
    if not all(report["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
