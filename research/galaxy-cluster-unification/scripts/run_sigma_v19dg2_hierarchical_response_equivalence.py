#!/usr/bin/env python3
"""Run V19DG with variable-length RMF heap data materialized in scope."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import run_sigma_v19dg_hierarchical_response_equivalence as parent
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialized_table(path: Path, extension: str) -> tuple[fits.Header, dict[str, Any]]:
    with fits.open(path, memmap=False) as hdus:
        hdu = hdus[extension]
        columns: dict[str, Any] = {}
        for name in hdu.columns.names:
            values = hdu.data[name]
            if values.dtype.kind == "O":
                columns[name] = [np.asarray(value).copy() for value in values]
            else:
                columns[name] = np.asarray(values).copy()
        return hdu.header.copy(), columns


def materialized_numeric_column(data: dict[str, Any], name: str) -> np.ndarray:
    values = data[name]
    if isinstance(values, list):
        return np.concatenate(
            [np.asarray(value, dtype=np.float64).ravel() for value in values]
        )
    return np.asarray(values, dtype=np.float64).ravel()


def run(config_path: Path, output: Path, scratch: Path) -> dict[str, Any]:
    config = parent.load_json(config_path)
    runner = Path(__file__).resolve()
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DG2 runner changed after freeze")
    parent_runner = ROOT / config["parents"]["v19dg_runner"]
    if sha256(parent_runner) != config["parents"]["v19dg_runner_sha256"]:
        raise RuntimeError("V19DG2 parent runner changed")
    parent_report = ROOT / config["parents"]["v19dg_failed_report"]
    if sha256(parent_report) != config["parents"]["v19dg_failed_report_sha256"]:
        raise RuntimeError("V19DG2 parent failure report changed")
    failure = parent.load_json(parent_report)
    if failure.get("status") != "hierarchical_response_equivalence_execution_failed":
        raise RuntimeError("V19DG2 parent did not fail in execution")
    index_path = ROOT / config["input"]["validated_cell_index"]
    if sha256(index_path) != config["input"]["validated_cell_index_sha256"]:
        raise RuntimeError("V19DG2 validated-cell index changed")

    parent.table = materialized_table
    parent.numeric_column = materialized_numeric_column
    rows = parent.read_rows(index_path)
    controls = parent.select_controls(rows, config)
    if scratch.exists():
        raise RuntimeError(f"V19DG2 scratch must not already exist: {scratch}")
    scratch.mkdir(parents=True)
    results = []
    for control_id, selected in controls.items():
        paths = [Path(row["source_pha"]) for row in selected]
        control_root = scratch / control_id
        direct_source = parent.run_combine(
            paths,
            control_root / "direct" / "combined",
            control_root / "direct" / "combined.log",
        )
        hierarchical_source, chunk_counts = parent.hierarchical_combine(
            paths,
            control_root / "hierarchical" / "combined",
            int(config["hierarchy"]["chunk_size"]),
        )
        result = {
            "id": control_id,
            "cells": len(paths),
            "chunk_counts": chunk_counts,
            "comparisons": parent.compare_products(direct_source, hierarchical_source),
        }
        result["gates"] = parent.gate_control(result, config["tolerances"])
        result["passed"] = all(result["gates"].values())
        results.append(result)
    passed = all(item["passed"] for item in results)
    return {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(runner),
        "status": (
            "hierarchical_response_equivalence_passed_successor_may_be_frozen"
            if passed
            else "hierarchical_response_equivalence_failed_no_successor_authorized"
        ),
        "controls": results,
        "aggregate_pass": passed,
        "full_combination_executed": False,
        "spectrum_fitted": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args.config.resolve(), output, args.scratch.resolve())
    except Exception as exc:  # noqa: BLE001 - preserve terminal failure evidence
        report = {
            "protocol_version": "SIGMA-V19DG2-HIERARCHICAL-RESPONSE-EQUIVALENCE-1.0.0",
            "generated_utc": datetime.now(UTC).isoformat(),
            "status": "hierarchical_response_equivalence_execution_failed",
            "exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
            "full_combination_executed": False,
            "spectrum_fitted": False,
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
