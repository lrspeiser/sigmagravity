#!/usr/bin/env python3
"""Audit V19BMB before terminal V19X4B gas products exist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bmb_v19x4b_stellar_successor_preflight.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19bmb_v19x4b_stellar_successor_preflight" / "report.json"


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
    original = load_json(ROOT / config["parents"]["original_v19bm_config"])
    original_report = load_json(ROOT / config["parents"]["original_v19bm_preflight_report"])
    x4b_report = load_json(ROOT / config["parents"]["v19x4b_preflight_report"])
    contract = config["successor_contract"]
    gates = {
        "all_parent_and_implementation_hashes_exact": bool(hash_gates) and all(hash_gates.values()),
        "original_v19bm_preflight_still_passes": (
            original_report.get("decision") == "passed_stellar_control_preflight_awaiting_terminal_v19x4"
            and original_report.get("gates") and all(original_report["gates"].values())
        ),
        "v19x4b_successor_preflight_passes": (
            x4b_report.get("status") == "v19x4b_v19x3b_gas_successor_preflight_passed"
            and x4b_report.get("gates") and all(x4b_report["gates"].values())
        ),
        "stellar_science_sections_are_identical": all(
            contract["science_section_sha256"][section] == canonical_sha256(original[section])
            for section in contract["science_sections"]
        ),
        "full_draw_grid_and_filter_contract_retained": (
            original["construction"]["draws"] == 4096
            and original["construction"]["common_axis_kpc"]["cells"] == 241
            and original["construction"]["smoothing_fwhm_kpc"] == [50.0, 100.0]
            and not original["authorization"]["compare_cross_filter_amplitudes"]
            and not original["authorization"]["infer_stellar_mass"]
        ),
        "terminal_products_and_targets_sealed": not any(
            config["authorization"][key]
            for key in (
                "freeze_before_terminal_v19x4b_pass",
                "construct_observed_stellar_control_now",
                "open_source_score_now",
                "open_lensing_or_halo_payload",
                "change_gravity_formula_or_parameter",
                "open_holdout",
            )
        ),
        "separately_named_successor_required": (
            config["implementation"]["runner"] != config["parents"]["original_v19bm_runner"]
            and config["authorization"]["freeze_after_terminal_v19x4b_pass"]
        ),
    }
    return {
        "protocol_version": config["protocol_version"],
        "status": (
            "v19bmb_v19x4b_stellar_successor_preflight_passed"
            if all(gates.values()) else "v19bmb_v19x4b_stellar_successor_preflight_failed"
        ),
        "config_sha256": sha256(config_path),
        "hashes": hashes,
        "hash_gates": hash_gates,
        "gates": gates,
        "terminal_gas_or_stellar_product_opened": False,
        "source_lensing_halo_gravity_or_holdout_payload_opened": False,
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
