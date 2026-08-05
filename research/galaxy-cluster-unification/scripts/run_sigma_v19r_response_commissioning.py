#!/usr/bin/env python3
"""Commission one deterministic V19Q source/background/ARF/RMF response cell."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pycrates

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v17c_integrated_spectra as inherited
import run_sigma_v19p_exact_flux_obs_support as v19p
import run_sigma_v19q_positive_exposure_response_workload as v19q

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19r_response_commissioning.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19r_response_commissioning"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19r-response-commissioning/v100")


def resolve_input(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def validate_external_inputs(config: dict[str, Any]) -> None:
    for item in config["frozen_inputs"].values():
        path = resolve_input(item["path"])
        if not path.is_file() or v19p.sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19R frozen input mismatch: {path}")


def selected_manifest_row(config: dict[str, Any]) -> dict[str, str]:
    path = ROOT / config["parents"]["v19q_manifest"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ranked = sorted(
        rows,
        key=lambda row: (
            -int(row["source_band_events"]),
            row["cluster"],
            int(row["bin_id"]),
            int(row["obsid"]),
            int(row["ccd_id"]),
        ),
    )
    if len(ranked) < 2 or int(ranked[0]["source_band_events"]) == int(
        ranked[1]["source_band_events"]
    ):
        raise RuntimeError("V19R selected response cell is not a unique count maximum")
    selected = config["selection"]
    expected = (
        selected["cluster"],
        int(selected["bin_id"]),
        int(selected["obsid"]),
        int(selected["ccd_id"]),
        int(selected["source_band_events"]),
        int(selected["background_band_events"]),
    )
    actual = (
        ranked[0]["cluster"],
        int(ranked[0]["bin_id"]),
        int(ranked[0]["obsid"]),
        int(ranked[0]["ccd_id"]),
        int(ranked[0]["source_band_events"]),
        int(ranked[0]["background_band_events"]),
    )
    if actual != expected:
        raise RuntimeError(f"V19R deterministic selection mismatch: {actual} vs {expected}")
    return ranked[0]


def pha_channel_audit(pha: Path, event_filter: str) -> dict[str, Any]:
    events = pycrates.read_file(event_filter)
    pi = np.asarray(events.get_column("PI").values, dtype=int)
    spectrum = pycrates.read_file(str(pha))
    channel = np.asarray(spectrum.get_column("CHANNEL").values, dtype=int)
    counts = np.asarray(spectrum.get_column("COUNTS").values, dtype=int)
    event_by_channel = {int(key): int(value) for key, value in zip(*np.unique(pi, return_counts=True), strict=True)}
    pha_by_channel = {
        int(key): int(value) for key, value in zip(channel, counts, strict=True) if value
    }
    all_channels = set(event_by_channel) | set(pha_by_channel)
    deltas = {
        key: pha_by_channel.get(key, 0) - event_by_channel.get(key, 0)
        for key in all_channels
        if pha_by_channel.get(key, 0) != event_by_channel.get(key, 0)
    }
    return {
        "event_rows": int(pi.size),
        "pha_total_counts": int(np.sum(counts)),
        "nonzero_event_channels": len(event_by_channel),
        "nonzero_pha_channels": len(pha_by_channel),
        "mismatched_channel_count": len(deltas),
        "maximum_absolute_channel_delta": max(
            (abs(value) for value in deltas.values()), default=0
        ),
        "exact": not deltas and int(np.sum(counts)) == int(pi.size),
    }


def response_audit(arf: Path, rmf: Path) -> dict[str, Any]:
    arf_crate = pycrates.read_file(str(arf))
    spectral_response = np.asarray(
        arf_crate.get_column("SPECRESP").values, dtype=float
    )
    rmf_crate = pycrates.read_file(f"{rmf}[MATRIX]")
    matrix_values = rmf_crate.get_column("MATRIX").values
    arrays = [np.asarray(value, dtype=float).ravel() for value in matrix_values]
    nonempty = [value for value in arrays if value.size]
    matrix = np.concatenate(nonempty) if nonempty else np.array([], dtype=float)
    return {
        "arf_rows": int(spectral_response.size),
        "arf_finite": bool(np.isfinite(spectral_response).all()),
        "arf_positive_bins": int(np.count_nonzero(spectral_response > 0.0)),
        "arf_maximum_cm2": float(np.max(spectral_response)),
        "rmf_matrix_elements": int(matrix.size),
        "rmf_finite": bool(np.isfinite(matrix).all()),
        "rmf_nonzero_elements": int(np.count_nonzero(matrix)),
        "rmf_maximum": float(np.max(matrix)) if matrix.size else 0.0,
    }


def pha_links(source_pha: Path, env: dict[str, str]) -> dict[str, str]:
    return {
        key: inherited.command_text(["dmkeypar", str(source_pha), key, "echo+"], env)
        for key in ("BACKFILE", "ANCRFILE", "RESPFILE")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = v19p.load_json(config_path)
    v19p.validate_parent_hashes(config)
    validate_external_inputs(config)
    selected = selected_manifest_row(config)
    source_report = v19p.load_json(ROOT / config["parents"]["source_map_report"])
    region_report = v19p.load_json(ROOT / config["parents"]["v19m_report"])
    source_row = next(row for row in source_report["clusters"] if row["cluster"] == "BULLET")
    region_row = next(row for row in region_report["clusters"] if row["cluster"] == "BULLET")
    selection = config["selection"]
    obsid = int(selection["obsid"])
    bin_id = int(selection["bin_id"])
    ccd_id = int(selection["ccd_id"])
    science = resolve_input(config["frozen_inputs"]["science_event"]["path"])
    background = resolve_input(config["frozen_inputs"]["blanksky_event"]["path"])
    fov = resolve_input(config["frozen_inputs"]["exact_flux_obs_fov"]["path"])
    exposure_path = resolve_input(
        config["frozen_inputs"]["exact_flux_obs_exposure"]["path"]
    )
    region = resolve_input(config["frozen_inputs"]["spectral_region"]["path"])
    mask = resolve_input(config["frozen_inputs"]["mask"]["path"])
    badpix = resolve_input(config["frozen_inputs"]["badpix"]["path"])
    aspect = resolve_input(config["frozen_inputs"]["corrected_aspect_list"]["path"])
    scratch = args.scratch.resolve()
    work = scratch / "commissioning"
    logs = work / "logs"
    products = work / "products"
    env = inherited.isolated_environment(
        os.environ, work / "pfiles", work / "tmp"
    )
    binmap = v19p.image(v19p.region_product(region_row, "binmap")).astype(int)
    exposure = np.nan_to_num(v19p.image(exposure_path), nan=0.0)
    science_table, _, _, _ = v19q.science_assignments(
        f"{science}[sky=region({fov})]",
        binmap,
        exposure,
        source_row["grid"],
        (500, 7000),
        0.0,
    )
    positive_exposure_task_events = science_table.get((bin_id, ccd_id), 0)
    source_filter = (
        f"{science}[ccd_id={ccd_id}]"
        f"[sky=region({fov})][sky=region({region})]"
    )
    source_band_events = inherited.event_count(
        source_filter + "[energy=500:7000]", env
    )
    corrected_background = work / "background_geometry" / "acisf5356_blanksky_geometry.fits"
    background_geometry = inherited.prepare_background_geometry(
        science, background, corrected_background, env
    )
    background_filter = f"{corrected_background}[ccd_id={ccd_id}][sky=region({region})]"
    background_band_events = inherited.event_count(
        background_filter + "[energy=500:7000]", env
    )
    if (
        positive_exposure_task_events != int(selection["source_band_events"])
        or source_band_events != int(selection["source_band_events"])
        or background_band_events != int(selection["background_band_events"])
    ):
        raise RuntimeError(
            "V19R preflight count mismatch: "
            f"positive={positive_exposure_task_events}, source={source_band_events}, "
            f"background={background_band_events}"
        )
    response_reference = inherited.event_reference_coordinate(source_filter, science, env)
    source_chip = inherited.celestial_coordinate_chip(
        science,
        aspect,
        response_reference["ra_deg"],
        response_reference["dec_deg"],
        env,
    )
    background_chip = inherited.celestial_coordinate_chip(
        corrected_background,
        aspect,
        response_reference["ra_deg"],
        response_reference["dec_deg"],
        env,
    )
    response_reference["science_aspect_chip_id"] = source_chip
    response_reference["background_aspect_chip_id"] = background_chip
    if (
        response_reference["events"] != source_band_events
        or response_reference["dmcoords_chip_id"] != ccd_id
        or source_chip != ccd_id
        or background_chip != ccd_id
    ):
        raise RuntimeError(f"V19R response reference is off selected CCD: {response_reference}")
    outroot = products / "BULLET_bin390_obs5356_ccd2"
    source_pha = outroot.with_suffix(".pi")
    background_pha = outroot.with_name(outroot.name + "_bkg.pi")
    arf = outroot.with_suffix(".arf")
    rmf = outroot.with_suffix(".rmf")
    command = [
        "specextract",
        f"infile={source_filter}",
        f"outroot={outroot}",
        f"bkgfile={background_filter}",
        f"asp=@{aspect}",
        f"mskfile={mask}",
        f"badpixfile={badpix}",
        "dafile=CALDB",
        "bkgresp=no",
        "weight=yes",
        "weight_rmf=yes",
        "resp_pos=CENTROID",
        f"refcoord={response_reference['ra_deg']:.14f},{response_reference['dec_deg']:.14f}",
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
    extraction = inherited.execute_extraction_cell(
        {
            "cluster": "BULLET",
            "obsid": obsid,
            "ccd_id": ccd_id,
            "source_band_events": source_band_events,
            "background_band_events": background_band_events,
            "response_reference": response_reference,
            "source_pha": source_pha,
            "background_pha": background_pha,
            "arf": arf,
            "rmf": rmf,
            "bkgscale_value": float(selection["blanksky_scale"]),
            "translated_fov": {
                "role": "exact_flux_obs_fov",
                "path": str(fov),
                "sha256": v19p.sha256(fov),
            },
            "command": command,
            "log": logs / "specextract.log",
        },
        scratch,
        "v19r",
    )
    source_channel_audit = pha_channel_audit(source_pha, source_filter)
    background_channel_audit = pha_channel_audit(background_pha, background_filter)
    response = response_audit(arf, rmf)
    links = pha_links(source_pha, env)
    link_gate = all(value and value.upper() != "NONE" for value in links.values())
    output = args.output.resolve()
    frozen = output / "frozen_products"
    snapshot = {
        "source_pha": inherited.copy_snapshot(source_pha, frozen / source_pha.name),
        "background_pha": inherited.copy_snapshot(
            background_pha, frozen / background_pha.name
        ),
        "arf": inherited.copy_snapshot(arf, frozen / arf.name),
        "rmf": inherited.copy_snapshot(rmf, frozen / rmf.name),
        "specextract_log": inherited.copy_snapshot(
            logs / "specextract.log", frozen / "specextract.log"
        ),
    }
    gates = {
        "selected_row_is_unique_deterministic_manifest_maximum": selected["cluster"]
        == "BULLET",
        "all_frozen_input_hashes_exact": True,
        "source_preflight_count_equals_625": source_band_events == 625,
        "background_preflight_count_equals_232": background_band_events == 232,
        "positive_exposure_and_extraction_source_counts_equal": positive_exposure_task_events
        == source_band_events,
        "response_reference_maps_to_ccd_2_in_science_and_background": all(
            value == ccd_id
            for value in (
                response_reference["dmcoords_chip_id"],
                source_chip,
                background_chip,
            )
        ),
        "source_pha_background_pha_arf_and_rmf_exist_and_are_nonempty": all(
            path.is_file() and path.stat().st_size > 0
            for path in (source_pha, background_pha, arf, rmf)
        ),
        "source_and_background_pha_channel_histograms_match_events": source_channel_audit[
            "exact"
        ]
        and background_channel_audit["exact"],
        "arf_has_finite_positive_spectral_response": response["arf_finite"]
        and response["arf_positive_bins"] > 0,
        "rmf_has_finite_nonzero_matrix": response["rmf_finite"]
        and response["rmf_nonzero_elements"] > 0,
        "pha_response_and_background_links_are_present": link_gate,
        "effective_blanksky_scale_matches_frozen_scale_within_1e_6_relative": extraction[
            "blanksky_scaling"
        ]["effective_scale_relative_error_from_BKGSCALn"]
        <= 1e-6,
    }
    passed = all(gates.values())
    report = {
        "status": (
            "commissioning_response_passed_and_full_response_production_authorized"
            if passed
            else "commissioning_response_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": v19p.sha256(config_path),
        "runner_sha256": v19p.sha256(Path(__file__).resolve()),
        "selected_manifest_row": selected,
        "preflight": {
            "positive_exposure_task_events": positive_exposure_task_events,
            "source_band_events": source_band_events,
            "background_band_events": background_band_events,
        },
        "background_geometry": background_geometry,
        "response_reference": response_reference,
        "extraction": extraction,
        "source_pha_channel_audit": source_channel_audit,
        "background_pha_channel_audit": background_channel_audit,
        "response_audit": response,
        "source_pha_links": links,
        "frozen_snapshot": snapshot,
        "gates": gates,
        "full_response_production_authorized": passed,
        "temperature_density_mach_or_speed_fitted": False,
        "lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"status: {report['status']}")
    print(f"source/background: {source_band_events}/{background_band_events}")
    print(f"ARF/RMF: {response['arf_positive_bins']}/{response['rmf_nonzero_elements']}")
    print(f"report: {report_path}")
    print(f"sha256: {v19p.sha256(report_path)}")


if __name__ == "__main__":
    main()
