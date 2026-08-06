#!/usr/bin/env python3
"""Test whether bounded hierarchical CIAO response combination is equivalent."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def isolated_environment(work: Path) -> dict[str, str]:
    env = os.environ.copy()
    pfiles = work / "pfiles"
    tmp = work / "tmp"
    pfiles.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    inherited = env.get("PFILES", "")
    system = inherited.split(";", maxsplit=1)[1] if ";" in inherited else inherited
    env["PFILES"] = f"{pfiles};{system}" if system else str(pfiles)
    env["TMPDIR"] = str(tmp)
    return env


def run_combine(paths: list[Path], outroot: Path, log: Path) -> Path:
    outroot.parent.mkdir(parents=True, exist_ok=True)
    stack = outroot.with_name(outroot.name + "_source_spectra.lis")
    stack.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
    env = isolated_environment(outroot.parent)
    command = [
        "combine_spectra",
        f"src_spectra=@{stack}",
        f"outroot={outroot}",
        "method=sum",
        "bscale_method=asca",
        "exp_origin=pha",
        "clobber=yes",
        "verbose=0",
        "mode=h",
    ]
    completed = subprocess.run(
        command,
        cwd=outroot.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"combine_spectra exited {completed.returncode}; inspect {log}"
        )
    source = outroot.with_name(outroot.name + "_src.pi")
    required = [
        source,
        outroot.with_name(outroot.name + "_bkg.pi"),
        outroot.with_name(outroot.name + "_src.arf"),
        outroot.with_name(outroot.name + "_src.rmf"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"combine_spectra omitted products: {missing}")
    return source


def hierarchical_combine(
    paths: list[Path], outroot: Path, chunk_size: int
) -> tuple[Path, list[int]]:
    chunk_sources = []
    chunk_counts = []
    for index, start in enumerate(range(0, len(paths), chunk_size)):
        chunk = paths[start : start + chunk_size]
        chunk_root = outroot.parent / "chunks" / f"chunk_{index:04d}"
        chunk_sources.append(
            run_combine(
                chunk,
                chunk_root,
                chunk_root.with_name(chunk_root.name + ".log"),
            )
        )
        chunk_counts.append(len(chunk))
    final = run_combine(
        chunk_sources,
        outroot,
        outroot.with_name(outroot.name + ".log"),
    )
    return final, chunk_counts


def table(path: Path, extension: str) -> tuple[fits.Header, Any]:
    with fits.open(path, memmap=False) as hdus:
        hdu = hdus[extension]
        return hdu.header.copy(), hdu.data.copy()


def numeric_column(data: Any, name: str) -> np.ndarray:
    values = data[name]
    if values.dtype.kind == "O":
        return np.concatenate(
            [np.asarray(value, dtype=np.float64).ravel() for value in values]
        )
    return np.asarray(values, dtype=np.float64).ravel()


def difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "same_shape": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "max_absolute": None,
            "max_relative": None,
        }
    absolute = np.abs(left - right)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0e-30)
    return {
        "same_shape": True,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "max_absolute": float(np.max(absolute, initial=0.0)),
        "max_relative": float(np.max(absolute / scale, initial=0.0)),
    }


def compare_products(direct_source: Path, hierarchical_source: Path) -> dict[str, Any]:
    def related(source: Path, suffix: str) -> Path:
        root = source.with_name(source.name.removesuffix("_src.pi"))
        return root.with_name(root.name + suffix)

    comparisons: dict[str, Any] = {}
    for role, suffix, extension, columns in (
        ("source", "_src.pi", "SPECTRUM", ["CHANNEL", "COUNTS"]),
        ("background", "_bkg.pi", "SPECTRUM", ["CHANNEL", "COUNTS"]),
        ("arf", "_src.arf", "SPECRESP", ["ENERG_LO", "ENERG_HI", "SPECRESP"]),
        (
            "rmf_matrix",
            "_src.rmf",
            "MATRIX",
            ["ENERG_LO", "ENERG_HI", "N_GRP", "F_CHAN", "N_CHAN", "MATRIX"],
        ),
        ("rmf_ebounds", "_src.rmf", "EBOUNDS", ["CHANNEL", "E_MIN", "E_MAX"]),
    ):
        left_path = related(direct_source, suffix)
        right_path = related(hierarchical_source, suffix)
        left_header, left_data = table(left_path, extension)
        right_header, right_data = table(right_path, extension)
        role_columns = {
            column: difference(
                numeric_column(left_data, column), numeric_column(right_data, column)
            )
            for column in columns
        }
        role_headers = {}
        for key in ("EXPOSURE", "BACKSCAL", "AREASCAL", "DETCHANS"):
            if key in left_header or key in right_header:
                left = left_header.get(key)
                right = right_header.get(key)
                role_headers[key] = {
                    "direct": left,
                    "hierarchical": right,
                    "equal": left == right,
                    "relative_difference": (
                        abs(float(left) - float(right))
                        / max(abs(float(left)), abs(float(right)), 1.0e-30)
                        if left is not None and right is not None
                        else None
                    ),
                }
        comparisons[role] = {
            "direct": str(left_path),
            "hierarchical": str(right_path),
            "columns": role_columns,
            "headers": role_headers,
        }
    return comparisons


def select_controls(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> dict[str, list[dict[str, str]]]:
    controls = {}
    for item in config["controls"]:
        eligible = [row for row in rows if row["cluster"] == item["cluster"]]
        if "bin_id" in item:
            eligible = [row for row in eligible if int(row["bin_id"]) == item["bin_id"]]
        if "prefix_cells" in item:
            eligible = eligible[: item["prefix_cells"]]
        if len(eligible) != item["expected_cells"]:
            raise RuntimeError(
                f"{item['id']} selected {len(eligible)} not {item['expected_cells']} cells"
            )
        controls[item["id"]] = eligible
    return controls


def gate_control(result: dict[str, Any], tolerances: dict[str, float]) -> dict[str, bool]:
    comparisons = result["comparisons"]
    source = comparisons["source"]["columns"]
    background = comparisons["background"]["columns"]
    arf = comparisons["arf"]["columns"]
    rmf = comparisons["rmf_matrix"]["columns"]
    ebounds = comparisons["rmf_ebounds"]["columns"]
    return {
        "source_channel_and_counts_exact": all(
            source[column]["same_shape"] and source[column]["max_absolute"] == 0.0
            for column in ("CHANNEL", "COUNTS")
        ),
        "background_channel_exact": background["CHANNEL"]["same_shape"]
        and background["CHANNEL"]["max_absolute"] == 0.0,
        "background_counts_close": background["COUNTS"]["same_shape"]
        and background["COUNTS"]["max_absolute"]
        <= tolerances["background_counts_max_absolute"],
        "arf_energy_grid_exact": all(
            arf[column]["same_shape"] and arf[column]["max_absolute"] == 0.0
            for column in ("ENERG_LO", "ENERG_HI")
        ),
        "arf_response_close": arf["SPECRESP"]["same_shape"]
        and arf["SPECRESP"]["max_relative"]
        <= tolerances["arf_max_relative"],
        "rmf_structure_exact": all(
            rmf[column]["same_shape"] and rmf[column]["max_absolute"] == 0.0
            for column in ("ENERG_LO", "ENERG_HI", "N_GRP", "F_CHAN", "N_CHAN")
        ),
        "rmf_matrix_close": rmf["MATRIX"]["same_shape"]
        and rmf["MATRIX"]["max_absolute"]
        <= tolerances["rmf_matrix_max_absolute"],
        "rmf_ebounds_exact": all(
            ebounds[column]["same_shape"] and ebounds[column]["max_absolute"] == 0.0
            for column in ("CHANNEL", "E_MIN", "E_MAX")
        ),
        "source_exposure_exact": comparisons["source"]["headers"]["EXPOSURE"][
            "relative_difference"
        ]
        <= tolerances["header_max_relative"],
        "arf_exposure_exact": comparisons["arf"]["headers"]["EXPOSURE"][
            "relative_difference"
        ]
        <= tolerances["header_max_relative"],
    }


def run(config_path: Path, output: Path, scratch: Path) -> dict[str, Any]:
    config = load_json(config_path)
    runner = Path(__file__).resolve()
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DG runner changed after freeze")
    index_path = ROOT / config["input"]["validated_cell_index"]
    if sha256(index_path) != config["input"]["validated_cell_index_sha256"]:
        raise RuntimeError("V19DG validated-cell index changed")
    rows = read_rows(index_path)
    controls = select_controls(rows, config)
    if scratch.exists():
        raise RuntimeError(f"V19DG scratch must not already exist: {scratch}")
    scratch.mkdir(parents=True)
    results = []
    for control_id, selected in controls.items():
        paths = [Path(row["source_pha"]) for row in selected]
        control_root = scratch / control_id
        direct_source = run_combine(
            paths,
            control_root / "direct" / "combined",
            control_root / "direct" / "combined.log",
        )
        hierarchical_source, chunk_counts = hierarchical_combine(
            paths,
            control_root / "hierarchical" / "combined",
            int(config["hierarchy"]["chunk_size"]),
        )
        result = {
            "id": control_id,
            "cells": len(paths),
            "chunk_counts": chunk_counts,
            "comparisons": compare_products(direct_source, hierarchical_source),
        }
        result["gates"] = gate_control(result, config["tolerances"])
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
            "protocol_version": "SIGMA-V19DG-HIERARCHICAL-RESPONSE-EQUIVALENCE-1.0.0",
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
