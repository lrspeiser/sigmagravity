#!/usr/bin/env python3
"""Validate a direct array response calculation against CIAO addresp."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import run_sigma_v19dg_hierarchical_response_equivalence as tools
from astropy.io import fits
from run_sigma_v19dg2_hierarchical_response_equivalence import materialized_table

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dense_rmf(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header, data = materialized_table(path, "MATRIX")
    energy_lo = np.asarray(data["ENERG_LO"], dtype=np.float64)
    energy_hi = np.asarray(data["ENERG_HI"], dtype=np.float64)
    matrix = np.zeros((energy_lo.size, int(header["DETCHANS"])), dtype=np.float64)
    first_channel = int(header.get("TLMIN4", 1))
    for row in range(energy_lo.size):
        groups = int(data["N_GRP"][row])
        starts = np.atleast_1d(data["F_CHAN"][row])[:groups].astype(int)
        lengths = np.atleast_1d(data["N_CHAN"][row])[:groups].astype(int)
        values = np.atleast_1d(data["MATRIX"][row]).astype(np.float64)
        offset = 0
        for start, length in zip(starts, lengths, strict=True):
            begin = int(start) - first_channel
            matrix[row, begin : begin + int(length)] = values[offset : offset + length]
            offset += int(length)
    return energy_lo, energy_hi, matrix


def linked_paths(source: Path) -> tuple[Path, Path, float]:
    with fits.open(source, memmap=False) as hdus:
        header = hdus["SPECTRUM"].header
        return (
            source.parent / header["ANCRFILE"],
            source.parent / header["RESPFILE"],
            float(header["EXPOSURE"]),
        )


def direct_arrays(paths: list[Path]) -> dict[str, np.ndarray | float]:
    arf_numerator = None
    rmf_numerator = None
    response_denominator = None
    energy_lo = None
    energy_hi = None
    total_exposure = 0.0
    for source in paths:
        arf_path, rmf_path, exposure = linked_paths(source)
        with fits.open(arf_path, memmap=False) as hdus:
            arf_data = hdus["SPECRESP"].data
            current_lo = np.asarray(arf_data["ENERG_LO"], dtype=np.float64)
            current_hi = np.asarray(arf_data["ENERG_HI"], dtype=np.float64)
            arf = np.asarray(arf_data["SPECRESP"], dtype=np.float64)
        rmf_lo, rmf_hi, rmf = dense_rmf(rmf_path)
        if not np.array_equal(current_lo, rmf_lo) or not np.array_equal(
            current_hi, rmf_hi
        ):
            raise RuntimeError(f"ARF/RMF energy grids differ: {source}")
        if arf_numerator is None:
            energy_lo = current_lo
            energy_hi = current_hi
            arf_numerator = np.zeros_like(arf)
            response_denominator = np.zeros_like(arf)
            rmf_numerator = np.zeros_like(rmf)
        elif not np.array_equal(energy_lo, current_lo) or not np.array_equal(
            energy_hi, current_hi
        ):
            raise RuntimeError(f"Input energy grid differs: {source}")
        weight = arf * exposure
        arf_numerator += weight
        response_denominator += weight
        rmf_numerator += rmf * weight[:, None]
        total_exposure += exposure
    if arf_numerator is None or rmf_numerator is None or response_denominator is None:
        raise RuntimeError("No response inputs")
    combined_arf = arf_numerator / total_exposure
    combined_rmf = np.divide(
        rmf_numerator,
        response_denominator[:, None],
        out=np.zeros_like(rmf_numerator),
        where=response_denominator[:, None] != 0.0,
    )
    return {
        "energy_lo": energy_lo,
        "energy_hi": energy_hi,
        "arf": combined_arf,
        "rmf": combined_rmf,
        "exposure": total_exposure,
    }


def folded_diagnostics(
    energy_lo: np.ndarray,
    energy_hi: np.ndarray,
    arf: np.ndarray,
    expected: np.ndarray,
    observed: np.ndarray,
) -> dict[str, float]:
    center = 0.5 * (energy_lo + energy_hi)
    width = energy_hi - energy_lo
    models = {
        "flat": np.ones_like(center),
        "power_law_1": np.power(np.maximum(center, 0.05), -1.0),
        "power_law_2": np.power(np.maximum(center, 0.05), -2.0),
        "thermal_5kev": np.exp(-center / 5.0) / np.maximum(center, 0.05),
    }
    results = {}
    for name, model in models.items():
        incident = model * arf * width
        expected_counts = incident @ expected
        observed_counts = incident @ observed
        denominator = max(float(np.sum(np.abs(observed_counts))), 1.0e-30)
        results[name] = float(np.sum(np.abs(expected_counts - observed_counts))) / denominator
    return results


def select_suffix(rows: list[dict[str, str]], cluster: str, count: int) -> list[Path]:
    eligible = [Path(row["source_pha"]) for row in rows if row["cluster"] == cluster]
    if len(eligible) < count:
        raise RuntimeError(f"{cluster} has only {len(eligible)} cells")
    return eligible[-count:]


def run(config_path: Path, output: Path, scratch: Path) -> dict[str, Any]:
    config = tools.load_json(config_path)
    runner = Path(__file__).resolve()
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DH runner changed after freeze")
    for name in ("v19dg2_report", "v19dg2_config"):
        parent_path = ROOT / config["parents"][name]
        expected = config["parents"][f"{name}_sha256"]
        if sha256(parent_path) != expected:
            raise RuntimeError(f"V19DH parent changed: {name}")
    index = ROOT / config["input"]["validated_cell_index"]
    if sha256(index) != config["input"]["validated_cell_index_sha256"]:
        raise RuntimeError("V19DH validated-cell index changed")
    if scratch.exists():
        raise RuntimeError(f"V19DH scratch must not already exist: {scratch}")
    scratch.mkdir(parents=True)
    rows = tools.read_rows(index)
    controls = []
    for item in config["controls"]:
        paths = select_suffix(rows, item["cluster"], int(item["suffix_cells"]))
        control_root = scratch / item["id"]
        source = tools.run_combine(
            paths,
            control_root / "ciao" / "combined",
            control_root / "ciao" / "combined.log",
        )
        calculated = direct_arrays(paths)
        official_arf_path = source.with_name("combined_src.arf")
        official_rmf_path = source.with_name("combined_src.rmf")
        with fits.open(official_arf_path, memmap=False) as hdus:
            official_arf = np.asarray(
                hdus["SPECRESP"].data["SPECRESP"], dtype=np.float64
            )
        official_lo, official_hi, official_rmf = dense_rmf(official_rmf_path)
        energy_exact = np.array_equal(calculated["energy_lo"], official_lo) and np.array_equal(
            calculated["energy_hi"], official_hi
        )
        arf_difference = np.abs(calculated["arf"] - official_arf)
        arf_relative = arf_difference / np.maximum(
            np.maximum(np.abs(calculated["arf"]), np.abs(official_arf)), 1.0e-30
        )
        rmf_difference = np.abs(calculated["rmf"] - official_rmf)
        folded = folded_diagnostics(
            calculated["energy_lo"],
            calculated["energy_hi"],
            calculated["arf"],
            calculated["rmf"],
            official_rmf,
        )
        evidence = {
            "energy_grid_exact": bool(energy_exact),
            "arf_max_relative": float(np.max(arf_relative, initial=0.0)),
            "rmf_max_absolute": float(np.max(rmf_difference, initial=0.0)),
            "rmf_elements_at_or_above_addresp_threshold": int(
                np.sum(rmf_difference >= config["reference"]["addresp_threshold"])
            ),
            "rmf_row_sum_max_absolute": float(
                np.max(
                    np.abs(
                        np.sum(calculated["rmf"], axis=1)
                        - np.sum(official_rmf, axis=1)
                    ),
                    initial=0.0,
                )
            ),
            "folded_l1_relative": folded,
        }
        thresholds = config["tolerances"]
        gates = {
            "energy_grid_exact": evidence["energy_grid_exact"],
            "arf_close": evidence["arf_max_relative"] <= thresholds["arf_max_relative"],
            "no_rmf_element_reaches_addresp_threshold": evidence[
                "rmf_elements_at_or_above_addresp_threshold"
            ]
            == 0,
            "rmf_row_sums_close": evidence["rmf_row_sum_max_absolute"]
            <= thresholds["rmf_row_sum_max_absolute"],
            "all_folded_controls_close": max(folded.values())
            <= thresholds["folded_l1_relative"],
        }
        controls.append(
            {
                "id": item["id"],
                "cluster": item["cluster"],
                "cells": len(paths),
                "evidence": evidence,
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    passed = all(item["passed"] for item in controls)
    return {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(runner),
        "status": (
            "direct_array_response_parity_passed_full_successor_may_be_frozen"
            if passed
            else "direct_array_response_parity_failed_no_successor_authorized"
        ),
        "aggregate_pass": passed,
        "controls": controls,
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
            "protocol_version": "SIGMA-V19DH-DIRECT-RESPONSE-PARITY-1.0.0",
            "generated_utc": datetime.now(UTC).isoformat(),
            "status": "direct_array_response_parity_execution_failed",
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
