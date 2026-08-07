#!/usr/bin/env python3
"""Commission the V19DI direct-array OGIP writer on two exact controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import run_sigma_v19dg_hierarchical_response_equivalence as v19dg
from run_sigma_v19dg2_hierarchical_response_equivalence import (
    materialized_numeric_column,
    materialized_table,
)
from run_sigma_v19dh_direct_response_parity import direct_arrays, linked_paths
from sigma_v19di_direct_ogip import (
    link_pha,
    validate_written_response,
    write_arf,
    write_rmf,
)

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_no_response_combine(paths: list[Path], outroot: Path) -> Path:
    outroot.parent.mkdir(parents=True, exist_ok=True)
    stack = outroot.with_name(outroot.name + "_source_spectra.lis")
    stack.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
    env = v19dg.isolated_environment(outroot.parent / "runtime")
    command = [
        "combine_spectra",
        f"src_spectra=@{stack}",
        f"outroot={outroot}",
        "src_arfs=NONE",
        "src_rmfs=NONE",
        "bkg_arfs=NONE",
        "bkg_rmfs=NONE",
        "method=sum",
        "bscale_method=asca",
        "exp_origin=pha",
        "clobber=no",
        "verbose=1",
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
    log = outroot.with_name(outroot.name + ".log")
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"response-free combine_spectra exited {completed.returncode}; inspect {log}"
        )
    source = outroot.with_name(outroot.name + "_src.pi")
    background = outroot.with_name(outroot.name + "_bkg.pi")
    if not source.is_file() or not background.is_file():
        raise RuntimeError("response-free combine_spectra omitted a PHA product")
    return source


def sherpa_forward_fold(path: Path) -> dict[str, Any]:
    from sherpa.astro import ui

    ui.clean()
    ui.load_pha(str(path))
    ui.notice(0.5, 7.0)
    ui.set_source(ui.powlaw1d.writer_powerlaw)
    ui.set_par("writer_powerlaw.gamma", 1.8)
    ui.set_par("writer_powerlaw.ampl", 1.0e-4)
    data = ui.get_data()
    arf = ui.get_arf()
    rmf = ui.get_rmf()
    model = np.asarray(ui.get_model_plot().y, dtype=np.float64)
    mask = np.asarray(data.mask)
    evidence = {
        "noticed_bins": int(np.sum(mask)) if mask.ndim else len(data.channel),
        "arf_energy_bins": len(arf.energ_lo),
        "rmf_energy_bins": len(rmf.energ_lo),
        "rmf_channels": len(rmf.e_min),
        "model_finite": bool(np.all(np.isfinite(model))),
        "model_has_positive_prediction": bool(np.any(model > 0.0)),
        "model_sum": float(np.sum(model)),
    }
    evidence["passed"] = (
        evidence["noticed_bins"] > 0
        and evidence["arf_energy_bins"] == 1070
        and evidence["rmf_energy_bins"] == 1070
        and evidence["rmf_channels"] == 1024
        and evidence["model_finite"]
        and evidence["model_has_positive_prediction"]
    )
    ui.clean()
    return evidence


def product_gates(comparisons: dict[str, Any], header_tolerance: float) -> dict[str, bool]:
    exact_columns = {
        "source": ("CHANNEL", "COUNTS"),
        "background": ("CHANNEL", "COUNTS"),
        "arf": ("ENERG_LO", "ENERG_HI", "SPECRESP"),
        "rmf_matrix": ("ENERG_LO", "ENERG_HI", "N_GRP", "F_CHAN", "N_CHAN", "MATRIX"),
        "rmf_ebounds": ("CHANNEL", "E_MIN", "E_MAX"),
    }
    gates = {}
    for role, columns in exact_columns.items():
        gates[f"{role}_arrays_exact"] = all(
            comparisons[role]["columns"][column]["same_shape"]
            and comparisons[role]["columns"][column]["max_absolute"] == 0.0
            for column in columns
        )
    gates["source_exposure_close"] = comparisons["source"]["headers"][
        "EXPOSURE"
    ]["relative_difference"] <= header_tolerance
    gates["arf_exposure_close"] = comparisons["arf"]["headers"]["EXPOSURE"][
        "relative_difference"
    ] <= header_tolerance
    return gates


def run(config_path: Path, output: Path, scratch: Path) -> dict[str, Any]:
    config = v19dg.load_json(config_path)
    runner = Path(__file__).resolve()
    module = ROOT / config["implementation"]["writer_module"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DI runner changed after freeze")
    if sha256(module) != config["implementation"]["writer_module_sha256"]:
        raise RuntimeError("V19DI writer module changed after freeze")
    for name in ("v19dh_config", "v19dh_report"):
        parent = ROOT / config["parents"][name]
        if sha256(parent) != config["parents"][f"{name}_sha256"]:
            raise RuntimeError(f"V19DI parent changed: {name}")
    index = ROOT / config["input"]["validated_cell_index"]
    if sha256(index) != config["input"]["validated_cell_index_sha256"]:
        raise RuntimeError("V19DI validated-cell index changed")
    if scratch.exists():
        raise RuntimeError(f"V19DI scratch must not already exist: {scratch}")
    scratch.mkdir(parents=True)
    rows = v19dg.read_rows(index)
    v19dg.table = materialized_table
    v19dg.numeric_column = materialized_numeric_column
    controls = []
    for item in config["controls"]:
        selected = [
            row
            for row in rows
            if row["cluster"] == item["cluster"]
            and int(row["bin_id"]) == int(item["bin_id"])
        ]
        if len(selected) != int(item["expected_cells"]):
            raise RuntimeError(f"{item['id']} cell count changed")
        paths = [Path(row["source_pha"]) for row in selected]
        control_root = scratch / item["id"]
        official_source = v19dg.run_combine(
            paths,
            control_root / "official" / "combined",
            control_root / "official" / "combined.log",
        )
        written_source = run_no_response_combine(
            paths, control_root / "written" / "combined"
        )
        written_background = written_source.with_name("combined_bkg.pi")
        written_arf = written_source.with_name("combined_src.arf")
        written_rmf = written_source.with_name("combined_src.rmf")
        arrays = direct_arrays(paths)
        template_arf, template_rmf, _ = linked_paths(paths[0])
        write_arf(
            template_arf,
            written_arf,
            arrays["energy_lo"],
            arrays["energy_hi"],
            arrays["arf"],
            float(arrays["exposure"]),
        )
        writer = write_rmf(
            template_rmf,
            written_rmf,
            arrays["energy_lo"],
            arrays["energy_hi"],
            arrays["rmf"],
            float(arrays["exposure"]),
            float(config["writer"]["rmf_threshold"]),
        )
        link_pha(written_source, written_background, written_arf, written_rmf)
        validation = validate_written_response(
            written_arf,
            written_rmf,
            len(arrays["energy_lo"]),
            arrays["rmf"].shape[1],
        )
        comparisons = v19dg.compare_products(official_source, written_source)
        gates = product_gates(
            comparisons, float(config["tolerances"]["header_max_relative"])
        )
        sherpa = sherpa_forward_fold(written_source)
        gates["written_response_structure_passes"] = validation["passed"]
        gates["sherpa_load_and_forward_fold_passes"] = sherpa["passed"]
        controls.append(
            {
                "id": item["id"],
                "cluster": item["cluster"],
                "cells": len(paths),
                "writer": writer,
                "written_validation": validation,
                "comparisons": comparisons,
                "sherpa": sherpa,
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
        "writer_module_sha256": sha256(module),
        "status": (
            "direct_ogip_writer_preflight_passed_full_commissioning_may_be_frozen"
            if passed
            else "direct_ogip_writer_preflight_failed_no_full_commissioning_authorized"
        ),
        "aggregate_pass": passed,
        "controls": controls,
        "spectrum_fitted": False,
        "temperature_or_source_invariant_opened": False,
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
            "protocol_version": "SIGMA-V19DI-DIRECT-OGIP-WRITER-PREFLIGHT-1.0.0",
            "generated_utc": datetime.now(UTC).isoformat(),
            "status": "direct_ogip_writer_preflight_execution_failed",
            "exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
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
