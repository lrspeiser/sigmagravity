#!/usr/bin/env python3
"""Prove deterministic FITS serialization on repeated direct-response controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_sigma_v19di_direct_ogip_writer_preflight as v19di
import run_sigma_v19dj_direct_response_commissioning as v19dj
from astropy.io import fits
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
        raise RuntimeError("V19DK runner changed after freeze")
    canonicalizer = ROOT / implementation["canonicalizer"]
    if sha256(canonicalizer) != implementation["canonicalizer_sha256"]:
        raise RuntimeError("V19DK canonicalizer changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DK parent changed: {name}")
    index = ROOT / config["input"]["validated_cell_index"]
    if sha256(index) != config["input"]["validated_cell_index_sha256"]:
        raise RuntimeError("V19DK validated-cell index changed")
    v19dj_config_path = ROOT / config["parents"]["v19dj_config"]["path"]
    v19dj_runner = ROOT / config["parents"]["v19dj_runner"]["path"]
    science = v19dj.validate_frozen(load_json(v19dj_config_path), v19dj_runner)
    terminal = load_json(ROOT / config["parents"]["v19dj_report"]["path"])
    if terminal["status"] != "direct_response_commissioning_execution_failed":
        raise RuntimeError("V19DJ terminal status changed")
    if terminal["thermal_stress_constructed"]:
        raise RuntimeError("V19DJ opened thermal stress before remediation")
    if terminal["lensing_halo_action_gravity_or_holdout_payload_opened"]:
        raise RuntimeError("V19DJ opened a sealed payload before remediation")
    if terminal["gravity_formula_or_parameter_changed"]:
        raise RuntimeError("V19DJ changed gravity before remediation")
    return science


def checksum_gate(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=False, checksum=True) as hdus:
        checksums = [int(hdu.verify_checksum()) for hdu in hdus]
        datasums = [int(hdu.verify_datasum()) for hdu in hdus]
    return {
        "hdu_count": len(checksums),
        "checksum_values": checksums,
        "datasum_values": datasums,
        "passed": all(value == 1 for value in checksums + datasums),
    }


def product_paths(combination: dict[str, Any]) -> dict[str, Path]:
    return {
        item["role"]: ROOT / item["relative_path"]
        for item in combination["frozen_snapshot"]["products"]
    }


def run_control_once(
    control: dict[str, Any],
    cells: list[dict[str, Any]],
    run_id: str,
    scratch: Path,
    output: Path,
    science: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    label = control["id"]
    run_scratch = scratch / label / run_id
    run_output = output / "runs" / label / run_id
    combination = v19dj.combine_aperture(
        label,
        cells,
        run_scratch,
        run_output,
        science,
        float(config["writer"]["rmf_threshold"]),
    )
    canonical = product_paths(combination)
    for path in canonical.values():
        canonicalize_fits(path, config["canonicalization"]["stable_history"])

    work = run_scratch / label
    raw_grouped = work / f"{label}_src_grp.pi"
    canonical_grouped = canonical["grouped_source_spectrum"]
    v19di.v19dg.table = v19di.materialized_table
    v19di.v19dg.numeric_column = v19di.materialized_numeric_column
    comparisons = v19di.v19dg.compare_products(raw_grouped, canonical_grouped)
    gates = v19di.product_gates(
        comparisons, float(config["tolerances"]["header_max_relative"])
    )
    structure = v19di.validate_written_response(
        canonical["source_arf"],
        canonical["source_rmf"],
        int(config["expected_structure"]["energy_rows"]),
        int(config["expected_structure"]["channels"]),
    )
    checksums = {role: checksum_gate(path) for role, path in canonical.items()}
    sherpa = v19di.sherpa_forward_fold(canonical_grouped)
    gates["canonicalization_preserves_science_arrays_exactly"] = all(
        gates[f"{role}_arrays_exact"]
        for role in ("source", "background", "arf", "rmf_matrix", "rmf_ebounds")
    )
    gates["written_response_structure_passes"] = structure["passed"]
    gates["all_hdu_checksums_and_datasums_pass"] = all(
        item["passed"] for item in checksums.values()
    )
    gates["sherpa_load_and_forward_fold_passes"] = sherpa["passed"]
    hashes = {role: sha256(path) for role, path in canonical.items()}
    return {
        "run_id": run_id,
        "cells": combination["cells"],
        "full_pha_count_conservation_exact": combination[
            "full_pha_count_conservation_exact"
        ],
        "grouped_pha_links": combination["grouped_pha_links"],
        "expected_grouped_pha_links": combination["expected_grouped_pha_links"],
        "canonical_products": {
            role: {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashes[role],
            }
            for role, path in canonical.items()
        },
        "semantic_comparisons": comparisons,
        "response_structure": structure,
        "checksums": checksums,
        "sherpa": sherpa,
        "gates": gates,
        "passed": (
            combination["full_pha_count_conservation_exact"]
            and combination["grouped_pha_links"]
            == combination["expected_grouped_pha_links"]
            and all(gates.values())
        ),
    }


def execute(
    config: dict[str, Any],
    science: dict[str, Any],
    output: Path,
    scratch: Path,
) -> dict[str, Any]:
    if (output / "runs").exists():
        raise RuntimeError("V19DK output runs already exist")
    if scratch.exists():
        raise RuntimeError(f"V19DK scratch must not already exist: {scratch}")
    scratch.mkdir(parents=True)
    index = ROOT / config["input"]["validated_cell_index"]
    rows = v19di.v19dg.read_rows(index)
    controls = []
    for control in config["controls"]:
        cells = [
            row
            for row in rows
            if row["cluster"] == control["cluster"]
            and int(row["bin_id"]) == int(control["bin_id"])
        ]
        if len(cells) != int(control["expected_cells"]):
            raise RuntimeError(f"{control['id']} cell count changed")
        runs = [
            run_control_once(
                control,
                cells,
                f"repeat_{index + 1}",
                scratch,
                output,
                science,
                config,
            )
            for index in range(int(config["canonicalization"]["independent_runs"]))
        ]
        roles = runs[0]["canonical_products"]
        byte_identical = {
            role: len(
                {
                    run["canonical_products"][role]["sha256"]
                    for run in runs
                }
            )
            == 1
            for role in roles
        }
        controls.append(
            {
                "id": control["id"],
                "cluster": control["cluster"],
                "runs": runs,
                "byte_identical_across_independent_runs": byte_identical,
                "passed": all(run["passed"] for run in runs)
                and all(byte_identical.values()),
            }
        )
    passed = all(control["passed"] for control in controls)
    return {
        "status": (
            "fits_canonicalization_preflight_passed_full_successor_may_be_frozen"
            if passed
            else "fits_canonicalization_preflight_failed_no_full_successor_authorized"
        ),
        "aggregate_pass": passed,
        "controls": controls,
        "full_response_commissioning_successor_authorized": passed,
    }


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
        runner = Path(__file__).resolve()
        science = validate_frozen(config, runner)
        result = execute(config, science, output, args.scratch.resolve())
    except Exception as exc:  # noqa: BLE001 - preserve terminal failure evidence
        result = {
            "status": "fits_canonicalization_preflight_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
            "full_response_commissioning_successor_authorized": False,
        }
    report = {
        "protocol_version": "SIGMA-V19DK-FITS-CANONICALIZATION-PREFLIGHT-1.0.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
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
