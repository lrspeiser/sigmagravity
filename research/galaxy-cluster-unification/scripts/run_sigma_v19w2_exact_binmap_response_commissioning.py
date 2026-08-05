#!/usr/bin/env python3
"""Commission exact-binmap response extraction for frozen V19W failure classes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19p_exact_flux_obs_support as v19p
import run_sigma_v19r_response_commissioning as v19r
import run_sigma_v19w_full_response_production as v19w

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19w2_exact_binmap_response_commissioning.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19w2_exact_binmap_response_commissioning"
DEFAULT_SCRATCH = Path("/home/henry/sv19w2")


def sha256(path: Path) -> str:
    return v19p.sha256(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_parent_hashes(config: dict[str, Any]) -> None:
    for spec in config["parents"].values():
        path = ROOT / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise RuntimeError(f"V19W2 parent hash mismatch: {path}")
    base = load_json(ROOT / config["parents"]["v19w_config"]["path"])
    v19w.validate_parent_hashes(base)


def select_rows(config: dict[str, Any], manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    by_name = {v19w.cell_name(row): row for row in manifest}
    expected_names = [item["cell_name"] for item in config["commissioning_cells"]]
    if len(expected_names) != len(set(expected_names)):
        raise RuntimeError("V19W2 commissioning selection is not unique")
    if any(name not in by_name for name in expected_names):
        raise RuntimeError("V19W2 commissioning selection is absent from the frozen manifest")
    rows = [by_name[name] for name in expected_names]
    for spec, row in zip(config["commissioning_cells"], rows, strict=True):
        if int(row["source_band_events"]) != int(spec["expected_source_band_events"]):
            raise RuntimeError(f"V19W2 frozen source count changed: {spec['cell_name']}")
        if int(row["background_band_events"]) != int(
            spec["expected_background_band_events"]
        ):
            raise RuntimeError(f"V19W2 frozen background count changed: {spec['cell_name']}")
    return rows


def write_exact_bin_mask(
    binmap_path: Path,
    bin_id: int,
    destination: Path,
) -> dict[str, Any]:
    with fits.open(binmap_path, memmap=False) as hdus:
        source = np.asarray(hdus[0].data)
        header = hdus[0].header.copy()
    mask = (source == int(bin_id)).astype(np.uint8)
    if int(mask.sum()) <= 0:
        raise RuntimeError(f"V19W2 bin {bin_id} has an empty exact mask")
    if destination.exists():
        with fits.open(destination, memmap=False) as hdus:
            existing = np.asarray(hdus[0].data, dtype=np.uint8)
        if not np.array_equal(existing, mask):
            raise RuntimeError(f"V19W2 existing mask changed: {destination}")
        reused = True
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        fits.PrimaryHDU(data=mask, header=header).writeto(temporary, overwrite=False)
        temporary.replace(destination)
        reused = False
    return {
        "path": str(destination),
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "selected_pixels": int(mask.sum()),
        "definition": f"frozen V19M binmap == {int(bin_id)}",
        "reused": reused,
    }


def prepare_mask_cell(
    row: dict[str, str],
    context: dict[str, Any],
    scratch: Path,
) -> dict[str, Any]:
    cluster, bin_id, obsid, ccd_id = v19w.task_key(row)
    name = v19w.cell_name(row)
    token = f"c{int(row['production_index'])}"
    binmap = v19p.region_product(context["region_row"], "binmap")
    mask_record = write_exact_bin_mask(
        binmap,
        bin_id,
        scratch / "masks" / cluster / f"bin{bin_id}.fits",
    )
    exact_mask = Path(mask_record["path"])
    env = v19w.inherited.isolated_environment(
        os.environ,
        scratch / "pf" / token,
        scratch / "tmp" / token,
    )
    source_filter = (
        f"{context['science']}[ccd_id={ccd_id}]"
        f"[sky=region({context['fov']})][sky=mask({exact_mask})]"
    )
    background_filter = (
        f"{context['background']}[ccd_id={ccd_id}][sky=mask({exact_mask})]"
    )
    source_events = v19w.inherited.event_count(
        source_filter + "[energy=500:7000]", env
    )
    background_events = v19w.inherited.event_count(
        background_filter + "[energy=500:7000]", env
    )
    expected_source = int(row["source_band_events"])
    expected_background = int(row["background_band_events"])
    if (source_events, background_events) != (expected_source, expected_background):
        raise RuntimeError(
            f"V19W2 exact mask count mismatch for {name}: "
            f"{source_events}/{background_events} vs {expected_source}/{expected_background}"
        )
    return {
        "cluster": cluster,
        "bin_id": bin_id,
        "obsid": obsid,
        "ccd_id": ccd_id,
        "production_index": int(row["production_index"]),
        "cell_name": name,
        "token": token,
        "source_filter": source_filter,
        "background_filter": background_filter,
        "aspect": context["aspect"],
        "mask": context["mask"],
        "badpix": context["badpix"],
        "blanksky_scale": float(row["blanksky_scale"]),
        "preflight": {
            "source_band_events": source_events,
            "background_band_events": background_events,
        },
        "exact_bin_mask": mask_record,
    }


def execute_mask_cell(cell: dict[str, Any], scratch: Path) -> dict[str, Any]:
    name = cell["cell_name"]
    token = cell["token"]
    partial = scratch / "partial" / token
    completed = scratch / "completed" / name
    if completed.exists():
        report = load_json(completed / "cell_report.json")
        if report["cell_name"] != name or not all(report["gates"].values()):
            raise RuntimeError(f"V19W2 existing completed cell changed: {completed}")
        return {**report, "reused": True}
    if partial.exists():
        raise RuntimeError(f"V19W2 partial cell already exists: {partial}")
    products = partial / "products"
    logs = partial / "logs"
    products.mkdir(parents=True)
    outroot = products / name
    source_pha = outroot.with_suffix(".pi")
    background_pha = outroot.with_name(outroot.name + "_bkg.pi")
    arf = outroot.with_suffix(".arf")
    rmf = outroot.with_suffix(".rmf")
    command = [
        "specextract",
        f"infile={cell['source_filter']}",
        f"outroot={outroot}",
        f"bkgfile={cell['background_filter']}",
        f"asp=@{cell['aspect']}",
        f"mskfile={cell['mask']}",
        f"badpixfile={cell['badpix']}",
        "dafile=CALDB",
        "bkgresp=no",
        "weight=yes",
        "weight_rmf=yes",
        "resp_pos=CENTROID",
        "refcoord=",
        "correctpsf=no",
        "combine=no",
        "grouptype=NONE",
        "binspec=NONE",
        "bkg_grouptype=NONE",
        "bkg_binspec=NONE",
        "energy=0.3:11.0:0.01",
        "energy_wmap=500:7000",
        "binwmap=det=8",
        "binarfwmap=1",
        "parallel=no",
        "nproc=1",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    env = v19w.inherited.isolated_environment(
        os.environ,
        scratch / "pf2" / token,
        scratch / "t2" / token,
    )
    started_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    step = v19w.inherited.run_step(
        command,
        logs / "specextract.log",
        [source_pha, background_pha, arf, rmf],
        env,
    )
    scaling = v19w.inherited.verify_blanksky_scaling(
        source_pha,
        background_pha,
        float(cell["blanksky_scale"]),
        env,
    )
    source_audit = v19r.pha_channel_audit(source_pha, cell["source_filter"])
    background_audit = v19r.pha_channel_audit(
        background_pha, cell["background_filter"]
    )
    response = v19r.response_audit(arf, rmf)
    links = v19r.pha_links(source_pha, env)
    four_products = (source_pha, background_pha, arf, rmf)
    gates = {
        "exact_bin_mask_counts_match_frozen_manifest": (
            source_audit["event_rows"] >= cell["preflight"]["source_band_events"]
            and background_audit["event_rows"]
            >= cell["preflight"]["background_band_events"]
        ),
        "source_and_background_pha_channel_histograms_match_events": source_audit[
            "exact"
        ]
        and background_audit["exact"],
        "arf_is_finite_positive": response["arf_finite"]
        and response["arf_positive_bins"] > 0,
        "rmf_is_finite_nonzero": response["rmf_finite"]
        and response["rmf_nonzero_elements"] > 0,
        "pha_links_present": all(value and value.upper() != "NONE" for value in links.values()),
        "blanksky_scale_exact": scaling[
            "effective_scale_relative_error_from_BKGSCALn"
        ]
        <= 1e-6,
        "manual_refcoord_absent_and_mask_centroid_used": (
            "refcoord=" in command and "resp_pos=CENTROID" in command
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"V19W2 cell audit failed for {name}: {gates}")
    record = {
        "cell_name": name,
        "cluster": cell["cluster"],
        "bin_id": int(cell["bin_id"]),
        "obsid": int(cell["obsid"]),
        "ccd_id": int(cell["ccd_id"]),
        "production_index": int(cell["production_index"]),
        "attempt": 1,
        "start_utc": started_utc,
        "elapsed_seconds": time.perf_counter() - started,
        "preflight": cell["preflight"],
        "exact_bin_mask": cell["exact_bin_mask"],
        "response_position": {
            "refcoord": None,
            "resp_pos": "CENTROID",
            "reason": "CIAO-supported pixel-mask centroid; no manual cross-CCD coordinate",
        },
        "short_path_token": token,
        "step": step,
        "blanksky_scaling": scaling,
        "source_pha_channel_audit": source_audit,
        "background_pha_channel_audit": background_audit,
        "response_audit": response,
        "source_pha_links": links,
        "four_product_bytes": sum(path.stat().st_size for path in four_products),
        "products": {
            "source_pha": {
                "name": source_pha.name,
                "bytes": source_pha.stat().st_size,
                "sha256": sha256(source_pha),
            },
            "background_pha": {
                "name": background_pha.name,
                "bytes": background_pha.stat().st_size,
                "sha256": sha256(background_pha),
            },
            "arf": {
                "name": arf.name,
                "bytes": arf.stat().st_size,
                "sha256": sha256(arf),
            },
            "rmf": {
                "name": rmf.name,
                "bytes": rmf.stat().st_size,
                "sha256": sha256(rmf),
            },
        },
        "gates": gates,
        "reused": False,
    }
    (partial / "cell_report.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    completed.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(completed)
    return record


def run(config_path: Path, output: Path, scratch: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    runner_path = ROOT / config["implementation"]["runner"]
    if runner_path.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19W2 frozen runner path changed")
    if sha256(runner_path) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19W2 frozen runner hash changed")
    verify_parent_hashes(config)
    base_config = load_json(ROOT / config["parents"]["v19w_config"]["path"])
    manifest = v19w.load_manifest(base_config)
    rows = select_rows(config, manifest)
    contexts = v19w.observation_contexts(base_config, rows, scratch)
    prepared = [
        prepare_mask_cell(
            row,
            contexts[(row["cluster"], int(row["obsid"]))],
            scratch,
        )
        for row in rows
    ]
    records: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    workers = int(config["execution"]["maximum_concurrent_cells"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute_mask_cell, cell, scratch): cell for cell in prepared}
        for future in as_completed(futures):
            cell = futures[future]
            try:
                records[cell["cell_name"]] = future.result()
            except Exception as exc:  # noqa: BLE001 - retain every commissioning failure
                partial = scratch / "partial" / cell["token"]
                failed = scratch / "failed" / cell["cell_name"]
                if partial.exists() and not failed.exists():
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(partial), str(failed))
                failures[cell["cell_name"]] = f"{type(exc).__name__}: {exc}"
    ordered = [records[name] for name in [item["cell_name"] for item in config["commissioning_cells"]] if name in records]
    gates = {
        "all_five_failure_classes_have_a_completed_cell": len(ordered)
        == len(config["commissioning_cells"]),
        "every_exact_mask_preflight_matches_frozen_manifest": all(
            row["gates"]["exact_bin_mask_counts_match_frozen_manifest"] for row in ordered
        ),
        "every_pha_event_histogram_is_exact": all(
            row["gates"]["source_and_background_pha_channel_histograms_match_events"]
            for row in ordered
        ),
        "every_arf_rmf_link_and_blanksky_gate_passes": all(
            all(row["gates"].values()) for row in ordered
        ),
        "every_response_uses_mask_centroid_without_manual_refcoord": all(
            row["response_position"]["refcoord"] is None
            and row["response_position"]["resp_pos"] == "CENTROID"
            for row in ordered
        ),
        "no_commissioning_cell_failed": not failures,
    }
    passed = all(gates.values())
    report = {
        "protocol_version": config["protocol_version"],
        "status": (
            "exact_binmap_response_commissioning_passed_and_recovery_protocol_authorized"
            if passed
            else "exact_binmap_response_commissioning_failed_closed"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(runner_path),
        "scratch_root": str(scratch),
        "completed_cells": ordered,
        "failed_cells": failures,
        "gates": gates,
        "full_missing_cell_recovery_authorized": passed,
        "base_v19w_archive_modified": False,
        "spectrum_combined_or_fitted": False,
        "temperature_density_mach_or_speed_fitted": False,
        "lensing_halo_or_gravity_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "claim_boundary": config["claim_boundary"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"V19W2 failed closed: {failures} {gates}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    report = run(args.config, args.output.resolve(), args.scratch.resolve())
    print(args.output.resolve() / "report.json")
    print(f"status: {report['status']}")


if __name__ == "__main__":
    main()
