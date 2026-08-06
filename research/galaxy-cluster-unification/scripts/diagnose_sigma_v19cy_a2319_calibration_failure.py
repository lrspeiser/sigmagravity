#!/usr/bin/env python3
"""Diagnose the failed A2319 calibration gate without cluster-event access."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import apply_sigma_v19cy_a2319_calibration_candidates as application
import fit_sigma_v19cy_a2319_calibration_line_shape as line_shape

DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19cy_a2319_calibration_failure_diagnosis.json"
)
LINE_CONFIG = ROOT / "configs" / "sigma_v19cy_a2319_calibration_line_shape_gate.json"
BLOCK_BYTES = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    config = load_json(config_path)
    if config.get("protocol_version") != (
        "SIGMA-V19CY-A2319-CALIBRATION-FAILURE-DIAGNOSIS-1.0.0"
    ):
        raise RuntimeError("unexpected calibration-failure diagnosis protocol")
    expected_status = (
        "frozen after the calibration line-shape gate rejected all three candidates, "
        "but before any time-resolved candidate spectrum or continuous-calibration "
        "control spectrum was inspected"
    )
    if config.get("status") != expected_status:
        raise RuntimeError("calibration-failure diagnosis is not frozen")

    parents: dict[str, dict[str, Any]] = {}
    for name in (
        "line_shape_report",
        "application_report",
        "topology_report",
        "download_provenance",
    ):
        path = ROOT / config["parents"][name]
        if not path.is_file() or sha256(path) != config["parents"][f"{name}_sha256"]:
            raise RuntimeError(f"frozen diagnosis parent changed: {path}")
        parents[name] = load_json(path)

    failed = parents["line_shape_report"]
    applied = parents["application_report"]
    topology = parents["topology_report"]
    provenance = parents["download_provenance"]
    if failed.get("decision") != "stop_before_cluster_event_application":
        raise RuntimeError("parent line-shape result is not the frozen failure")
    if failed.get("cluster_sky_event_accessed") or failed.get("selected_candidate"):
        raise RuntimeError("parent line-shape boundary was not preserved")
    if not applied.get("terminal_gate_passed"):
        raise RuntimeError("candidate application parent did not pass")
    if not topology.get("topology_gate_passed"):
        raise RuntimeError("topology parent did not pass")
    if sha256(LINE_CONFIG) != failed["config_sha256"]:
        raise RuntimeError("frozen line-shape config changed")

    authorization = config["authorization"]
    for key in (
        "read_cluster_sky_event_rows",
        "apply_gain_to_cluster_sky_events",
        "fit_cluster_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed diagnosis boundary is open: {key}")
    return config, failed, applied, topology, provenance, load_json(LINE_CONFIG)


def verified_raw_path(
    raw_root: Path,
    relative: str,
    provenance_by_path: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    path = (raw_root / relative).resolve()
    if not path.is_relative_to(raw_root):
        raise RuntimeError(f"raw diagnosis path escapes root: {relative}")
    record = provenance_by_path.get(relative)
    if record is None:
        raise RuntimeError(f"raw diagnosis source absent from provenance: {relative}")
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise RuntimeError(f"raw diagnosis source size changed: {relative}")
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"raw diagnosis source hash changed: {relative}")
    return path, record


def decompress_gzip(source: Path, destination: Path) -> dict[str, Any]:
    with gzip.open(source, "rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
    return {
        "path": str(destination.relative_to(ROOT)),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def fit_energies(
    energies: np.ndarray,
    template: dict[str, np.ndarray],
    fit_config: dict[str, Any],
) -> dict[str, Any]:
    edges = np.arange(
        fit_config["energy_min_ev"],
        fit_config["energy_max_ev"] + fit_config["bin_width_ev"],
        fit_config["bin_width_ev"],
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    selected = energies[
        np.isfinite(energies)
        & (energies >= fit_config["energy_min_ev"])
        & (energies <= fit_config["energy_max_ev"])
    ]
    observed, _ = np.histogram(selected, bins=edges)
    fit = line_shape.fit_histogram(
        observed,
        centers,
        fit_config["bin_width_ev"],
        template,
        fit_config["bounds"],
    )
    if fit_config["require_convergence"] and not fit["converged"]:
        raise RuntimeError("frozen calibration diagnostic line fit did not converge")
    return fit


def read_candidate_branch(
    application_root: Path,
    applied: dict[str, Any],
    branch: str,
    candidate: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    expected = next(
        item["output"]
        for item in applied["applications"]
        if item["branch"] == branch and item["candidate"] == candidate
    )
    path = application_root / branch / f"{candidate}.evt"
    if (
        not path.is_file()
        or path.stat().st_size != expected["bytes"]
        or sha256(path) != expected["sha256"]
    ):
        raise RuntimeError(f"diagnosed candidate output changed: {path}")
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        data = hdus["EVENTS"].data
        times = np.asarray(data["TIME"], dtype=float).copy()
        energies = np.asarray(data["EPI2"], dtype=float).copy()
    return times, energies, {
        "path": str(path.relative_to(ROOT)),
        "bytes": expected["bytes"],
        "sha256": expected["sha256"],
        "rows": len(times),
    }


def time_resolved_fits(
    times: np.ndarray,
    energies: np.ndarray,
    branch: dict[str, Any],
    diagnostic: dict[str, Any],
    template: dict[str, np.ndarray],
    fit_config: dict[str, Any],
) -> dict[str, Any]:
    count = int(diagnostic["bins_per_branch"])
    edges = np.linspace(float(branch["start"]), float(branch["stop"]), count + 1)
    bins: list[dict[str, Any]] = []
    for index in range(count):
        if index + 1 == count:
            selected = (times >= edges[index]) & (times <= edges[index + 1])
        else:
            selected = (times >= edges[index]) & (times < edges[index + 1])
        fit = fit_energies(energies[selected], template, fit_config)
        minimum = diagnostic["minimum_events_in_fit_window_per_bin"]
        if fit["events_in_fit_window"] < minimum:
            raise RuntimeError(
                f"time bin has {fit['events_in_fit_window']} events, below {minimum}: "
                f"{branch['name']} quartile {index + 1}"
            )
        bins.append(
            {
                "quartile": index + 1,
                "start": float(edges[index]),
                "stop": float(edges[index + 1]),
                "midpoint": float(0.5 * (edges[index] + edges[index + 1])),
                "selected_rows": int(np.sum(selected)),
                "fit": fit,
            }
        )
    return {"total_fit": fit_energies(energies, template, fit_config), "quartiles": bins}


def ftcopy_control_command(
    config: dict[str, Any], source: Path, output: Path, branch: dict[str, Any]
) -> str:
    selection = (
        f"{application.to_wsl_path(source)}[EVENTS][TIME>={branch['start']}"
        f"&&TIME<={branch['stop']}&&PIXEL==12&&ITYPE==0]"
    )
    return (
        application.runtime_environment(config)
        + "ftcopy "
        + shlex.quote(selection)
        + " "
        + shlex.quote(application.to_wsl_path(output))
        + " copyall=yes clobber=yes history=yes"
    )


def diagnostic_flags(
    config: dict[str, Any],
    failed: dict[str, Any],
    topology: dict[str, Any],
    resolved: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    threshold = float(config["diagnostic_thresholds"]["material_centroid_change_ev"])
    candidate = config["diagnosed_candidate"]
    targets = load_json(LINE_CONFIG)["published_targets"]
    output: dict[str, Any] = {}
    for obsid in ("000101000", "000102000", "000103000"):
        branches = [branch for branch in topology["branches"] if branch["obsid"] == obsid]
        branch_rows = []
        any_slope = False
        any_curvature = False
        for branch in branches:
            result = resolved[branch["name"]]
            quartiles = result["quartiles"]
            x = np.asarray([row["midpoint"] for row in quartiles], dtype=float)
            y = np.asarray(
                [row["fit"]["centroid_shift_ev"] for row in quartiles], dtype=float
            )
            endpoint_change = float(y[-1] - y[0])
            endpoint_line = y[0] + (y[-1] - y[0]) * (x - x[0]) / (x[-1] - x[0])
            maximum_curvature = float(np.max(np.abs(y - endpoint_line)))
            slope_flag = abs(endpoint_change) >= threshold
            curvature_flag = maximum_curvature >= threshold
            any_slope = any_slope or slope_flag
            any_curvature = any_curvature or curvature_flag
            branch_rows.append(
                {
                    "branch": branch["name"],
                    "total_centroid_shift_ev": result["total_fit"]["centroid_shift_ev"],
                    "endpoint_change_ev": endpoint_change,
                    "maximum_curvature_ev": maximum_curvature,
                    "slope_flag": slope_flag,
                    "curvature_flag": curvature_flag,
                }
            )
        totals = [row["total_centroid_shift_ev"] for row in branch_rows]
        branch_range = float(max(totals) - min(totals))
        discontinuity = branch_range >= threshold
        combined = failed["fit_results"][candidate][obsid]["centroid_shift_ev"]
        target = targets[obsid]["centroid_shift_ev"]
        mismatch = float(combined - target)
        offset = (
            abs(mismatch) >= threshold
            and not any_slope
            and not any_curvature
            and not discontinuity
        )
        output[obsid] = {
            "combined_candidate_minus_published_centroid_ev": mismatch,
            "branch_total_centroid_range_ev": branch_range,
            "slope_flag": any_slope,
            "curvature_flag": any_curvature,
            "branch_discontinuity_flag": discontinuity,
            "reference_offset_flag": offset,
            "branches": branch_rows,
        }
    return output


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, failed, applied, topology, provenance, line_config = validate_inputs(
        config_path
    )
    raw_root = (ROOT / config["paths"]["raw_root"]).resolve()
    application_root = (ROOT / config["paths"]["application_scratch_root"]).resolve()
    scratch_root = (ROOT / config["paths"]["scratch_root"]).resolve()
    if scratch_root.exists():
        raise RuntimeError(f"refusing to overwrite diagnosis scratch root: {scratch_root}")
    if not scratch_root.is_relative_to((ROOT / "tmp").resolve()):
        raise RuntimeError("diagnosis scratch root must remain under repository tmp")
    scratch_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=scratch_root.name + ".installing.", dir=scratch_root.parent)
    )

    template_path = (ROOT / config["paths"]["line_template_scratch"]).resolve()
    expected_template = failed["template_copy"]
    if (
        not template_path.is_file()
        or template_path.stat().st_size != expected_template["bytes"]
        or sha256(template_path) != expected_template["sha256"]
    ):
        raise RuntimeError("frozen copied line template changed")
    template = line_shape.read_template(
        template_path, line_config["line_template"]["extension"]
    )
    fit_config = line_config["fit"]

    candidate = config["diagnosed_candidate"]
    resolved: dict[str, dict[str, Any]] = {}
    candidate_sources: list[dict[str, Any]] = []
    for branch in topology["branches"]:
        times, energies, source = read_candidate_branch(
            application_root, applied, branch["name"], candidate
        )
        resolved[branch["name"]] = time_resolved_fits(
            times,
            energies,
            branch,
            config["time_resolved_test"],
            template,
            fit_config,
        )
        candidate_sources.append({"branch": branch["name"], **source})

    provenance_by_path = {item["download_path"]: item for item in provenance["records"]}
    control = config["continuous_control"]
    obsids = sorted({branch["obsid"] for branch in topology["branches"]})
    histories: dict[str, Path] = {}
    event_sources: dict[str, Path] = {}
    source_audits: list[dict[str, Any]] = []
    history_dir = staging / "histories"
    history_dir.mkdir()
    for obsid in obsids:
        history_relative = control["history_template"].format(obsid=obsid)
        history_source, history_record = verified_raw_path(
            raw_root, history_relative, provenance_by_path
        )
        history_output = history_dir / f"xa{obsid}rsl_000_pxcal.ghf"
        decompressed = decompress_gzip(history_source, history_output)
        histories[obsid] = history_output
        source_audits.append(
            {
                "obsid": obsid,
                "kind": "continuous_calibration_history",
                "raw_path": history_relative,
                "raw_sha256": history_record["sha256"],
                "raw_bytes": history_record["bytes"],
                "decompressed": decompressed,
            }
        )
        event_relative = control["calibration_event_template"].format(obsid=obsid)
        event_source, event_record = verified_raw_path(
            raw_root, event_relative, provenance_by_path
        )
        event_sources[obsid] = event_source
        source_audits.append(
            {
                "obsid": obsid,
                "kind": "calibration_events",
                "raw_path": event_relative,
                "raw_sha256": event_record["sha256"],
                "raw_bytes": event_record["bytes"],
            }
        )

    commands: list[dict[str, Any]] = []
    control_outputs: list[dict[str, Any]] = []
    control_energies: dict[str, list[np.ndarray]] = {obsid: [] for obsid in obsids}
    distribution = config["runtime"]["wsl_distribution"]
    for branch in topology["branches"]:
        branch_dir = staging / branch["name"]
        branch_dir.mkdir()
        selected = branch_dir / "selected_pixel12_hp.evt"
        copy_result = application.run_wsl(
            distribution,
            ftcopy_control_command(config, event_sources[branch["obsid"]], selected, branch),
        )
        commands.append({"stage": "ftcopy", "branch": branch["name"], **copy_result})
        if copy_result["exit_code"] != 0 or not selected.is_file():
            raise RuntimeError(f"continuous-control ftcopy failed: {branch['name']}")
        output = branch_dir / "continuous_pxcal.evt"
        apply_result = application.run_wsl(
            distribution,
            application.rslpha2pi_command(
                config, selected, output, histories[branch["obsid"]]
            ),
            timeout=1200,
        )
        commands.append(
            {"stage": "rslpha2pi", "branch": branch["name"], **apply_result}
        )
        if apply_result["exit_code"] != 0 or not output.is_file():
            raise RuntimeError(f"continuous-control rslpha2pi failed: {branch['name']}")
        audit = application.audit_output(output)
        if (
            audit["rows"] != application.event_rows(selected)
            or audit["null_epi2"] != 0
            or audit["null_temp"] != 0
            or audit["null_pi_not_explained_by_negative_epi2"] != 0
            or audit["negative_epi2_without_null_pi"] != 0
        ):
            raise RuntimeError(f"continuous-control output failed audit: {branch['name']}")
        with fits.open(output, memmap=True, mode="readonly") as hdus:
            control_energies[branch["obsid"]].append(
                np.asarray(hdus["EVENTS"].data["EPI2"], dtype=float).copy()
            )
        control_outputs.append({"branch": branch["name"], "obsid": branch["obsid"], **audit})

    targets = line_config["published_targets"]
    control_fits: dict[str, dict[str, Any]] = {}
    control_passed = True
    for obsid in obsids:
        fit = fit_energies(np.concatenate(control_energies[obsid]), template, fit_config)
        centroid_ok = abs(fit["centroid_shift_ev"]) <= control[
            "maximum_absolute_centroid_shift_ev"
        ]
        fwhm_difference = fit["instrument_fwhm_ev"] - targets[obsid]["fwhm_ev"]
        fwhm_ok = abs(fwhm_difference) <= control[
            "maximum_absolute_fwhm_difference_from_published_ev"
        ]
        control_fits[obsid] = {
            **fit,
            "fwhm_minus_published_ev": float(fwhm_difference),
            "centroid_control_passed": centroid_ok,
            "fwhm_control_passed": fwhm_ok,
            "passed": centroid_ok and fwhm_ok,
        }
        control_passed = control_passed and centroid_ok and fwhm_ok

    flags = diagnostic_flags(config, failed, topology, resolved)
    os.replace(staging, scratch_root)
    decision = (
        "authorize_freeze_of_one_calibration_only_replacement_reconstruction_protocol"
        if control_passed
        else "stop_and_audit_line_fitter_or_continuous_gain_application"
    )
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_calibration_failure_diagnosed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "diagnosed_candidate": candidate,
        "candidate_sources": candidate_sources,
        "time_resolved_results": resolved,
        "diagnostic_flags": flags,
        "continuous_control_sources": source_audits,
        "continuous_control_outputs": control_outputs,
        "continuous_control_fits": control_fits,
        "continuous_control_passed": control_passed,
        "commands": commands,
        "decision": decision,
        "cluster_sky_event_accessed": False,
        "cluster_velocity_fit": False,
        "validation_or_holdout_accessed": False,
        "authorization": {
            "freeze_one_replacement_calibration_protocol": control_passed,
            "read_cluster_sky_event_rows": False,
            "apply_gain_to_cluster_sky_events": False,
            "fit_cluster_velocity": False,
            "access_validation_or_holdout_assets": False,
            "open_lensing_halo_or_gravity_targets": False,
            "change_gravity_formula_or_parameters": False,
            "derive_or_select_action": False,
        },
    }
    output = ROOT / config["paths"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = build_report()
    print(
        json.dumps(
            {
                "status": result["status"],
                "continuous_control_fits": result["continuous_control_fits"],
                "continuous_control_passed": result["continuous_control_passed"],
                "diagnostic_flags": result["diagnostic_flags"],
                "cluster_sky_event_accessed": result["cluster_sky_event_accessed"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
