#!/usr/bin/env python3
"""Apply validated bracketed gain interpolation to A2319 development sky rows."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
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
import validate_sigma_v19cy_a2319_common_differential_gain as closure

DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19cy_a2319_bracketed_science_calibration.json"
)
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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    if config.get("protocol_version") != (
        "SIGMA-V19CY-A2319-BRACKETED-SCIENCE-CALIBRATION-1.0.1"
    ):
        raise RuntimeError("unexpected bracketed-science protocol")
    expected_status = (
        "corrected and refrozen after version 1.0.0 stopped during header-only "
        "validation because 000102000 has the official OBJECT value Abell2319_Cor1 "
        "rather than Abell2319; no sky-event row or energy was read, selected, "
        "recalculated, summarized, or fit"
    )
    if config.get("status") != expected_status:
        raise RuntimeError("bracketed-science protocol is not frozen")
    parents: dict[str, dict[str, Any]] = {}
    for name in (
        "endpoint_report",
        "closure_report",
        "topology_report",
        "download_provenance",
    ):
        path = ROOT / config["parents"][name]
        if not path.is_file() or sha256(path) != config["parents"][f"{name}_sha256"]:
            raise RuntimeError(f"bracketed-science parent changed: {path}")
        parents[name] = load_json(path)
    endpoint = parents["endpoint_report"]
    closure_parent = parents["closure_report"]
    if endpoint.get("selection", {}).get("passed"):
        raise RuntimeError("endpoint extrapolation was unexpectedly selected")
    if not endpoint.get("interior_interpolation_evidence_preserved"):
        raise RuntimeError("interior interpolation evidence was not preserved")
    folds = {fold["segment"]: fold for fold in closure_parent["folds"]}
    if not folds[1]["comparison"]["passed"] or not folds[2]["comparison"]["passed"]:
        raise RuntimeError("required bracketed held-out folds did not pass")
    authorization = config["authorization"]
    for key in (
        "inspect_or_fit_cluster_energy_distribution",
        "define_or_fit_spatial_velocity_regions",
        "fit_cluster_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed bracketed-science boundary is open: {key}")
    return config, parents["topology_report"], parents["download_provenance"]


def fit_bracketed_models(
    anchor_obsids: list[str],
    fe55_valid: dict[str, np.ndarray],
    pxcal_valid: dict[str, np.ndarray],
) -> dict[int, dict[str, float | int]]:
    if len(anchor_obsids) != 2 or anchor_obsids[0] == anchor_obsids[1]:
        raise RuntimeError("bracketed model requires two distinct anchor observations")
    models: dict[int, dict[str, float | int]] = {}
    for pixel in range(36):
        if pixel == 12:
            models[pixel] = {
                "rows": 0,
                "time_center": 0.0,
                "differential_at_center": 0.0,
                "slope_per_second": 0.0,
            }
            continue
        time_parts: list[np.ndarray] = []
        differential_parts: list[np.ndarray] = []
        for obsid in anchor_obsids:
            selected = fe55_valid[obsid][fe55_valid[obsid]["PIXEL"] == pixel]
            times = np.asarray(selected["TIME"], dtype=float)
            temperatures = np.asarray(selected["TEMP_FIT"], dtype=float)
            common = closure.common_temperature(pxcal_valid[obsid], times)
            time_parts.append(times)
            differential_parts.append(temperatures - common)
        times = np.concatenate(time_parts)
        differential = np.concatenate(differential_parts)
        center = float(np.mean(times))
        slope, intercept = np.polyfit(times - center, differential, 1)
        if not np.isfinite([slope, intercept]).all():
            raise RuntimeError(f"non-finite bracketed model for pixel {pixel}")
        models[pixel] = {
            "rows": len(times),
            "time_center": center,
            "differential_at_center": float(intercept),
            "slope_per_second": float(slope),
        }
    return models


def verify_cleaned_header(
    path: Path,
    expected_common: dict[str, str],
    expected_object: str,
    obsid: str,
) -> dict[str, Any]:
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        header = hdus["EVENTS"].header
        values = {key: header.get(key) for key in expected_common}
        object_value = header.get("OBJECT")
        rows = int(header["NAXIS2"])
    if values != expected_common or object_value != expected_object:
        raise RuntimeError(
            f"unexpected cleaned sky header for {obsid}: {values}, OBJECT={object_value}"
        )
    return {"header": {**values, "OBJECT": object_value}, "rows": rows}


def ftcopy_command(
    config: dict[str, Any], source: Path, output: Path, branch: dict[str, Any]
) -> str:
    selection = (
        f"{application.to_wsl_path(source)}[EVENTS][TIME>={branch['start']}"
        f"&&TIME<={branch['stop']}&&PIXEL!=12]"
    )
    return (
        application.runtime_environment(config)
        + "ftcopy "
        + shlex.quote(selection)
        + " "
        + shlex.quote(application.to_wsl_path(output))
        + " copyall=yes clobber=yes history=yes"
    )


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, topology, provenance = validate_inputs(config_path)
    raw_root = (ROOT / config["paths"]["raw_root"]).resolve()
    scratch_root = (ROOT / config["paths"]["scratch_root"]).resolve()
    if scratch_root.exists():
        raise RuntimeError(f"refusing to overwrite bracketed scratch root: {scratch_root}")
    if not scratch_root.is_relative_to((ROOT / "tmp").resolve()):
        raise RuntimeError("bracketed scratch root must remain under repository tmp")
    scratch_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=scratch_root.name + ".installing.", dir=scratch_root.parent)
    )
    provenance_by_path = {item["download_path"]: item for item in provenance["records"]}
    branches = {branch["name"]: branch for branch in topology["branches"]}
    segments = {segment["segment"]: segment for segment in topology["segment_sets"]["10800"]}
    extension = config["inputs"]["gain_history_extension"]
    minimum = config["inputs"]["minimum_gain_events"]

    required_segments = sorted(
        {segment for branch in config["included_branches"] for segment in branch["anchor_segments"]}
    )
    anchor_obsids = [segments[index]["obsids"][0] for index in required_segments]
    target_obsids = sorted({branch["obsid"] for branch in config["included_branches"]})
    fe55_hdus: dict[str, list[fits.hdu.base.ExtensionHDU]] = {}
    fe55_data: dict[str, np.ndarray] = {}
    fe55_valid: dict[str, np.ndarray] = {}
    pxcal_valid: dict[str, np.ndarray] = {}
    sky_paths: dict[str, Path] = {}
    sources: list[dict[str, Any]] = []

    for obsid in sorted(set(anchor_obsids) | set(target_obsids)):
        for role, template_name in (
            ("calibration_pixel_history", "calibration_pixel_history_template"),
            ("fe55_history", "fe55_history_template"),
        ):
            if role == "fe55_history" and obsid not in anchor_obsids:
                continue
            relative = config["inputs"][template_name].format(obsid=obsid)
            path, record = closure.verified_raw_path(
                raw_root, relative, provenance_by_path
            )
            sources.append(
                {
                    "obsid": obsid,
                    "kind": role,
                    "path": relative,
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
            )
            hdus, data, valid = closure.read_gain_history(path, extension, minimum)
            if role == "fe55_history":
                fe55_hdus[obsid] = hdus
                fe55_data[obsid] = data
                fe55_valid[obsid] = valid
            else:
                pxcal_valid[obsid] = valid
    for obsid in target_obsids:
        relative = config["inputs"]["cleaned_sky_event_template"].format(obsid=obsid)
        path, record = closure.verified_raw_path(raw_root, relative, provenance_by_path)
        header_audit = verify_cleaned_header(
            path,
            config["inputs"]["required_common_input_header"],
            config["inputs"]["required_object_by_obsid"][obsid],
            obsid,
        )
        sky_paths[obsid] = path
        sources.append(
            {
                "obsid": obsid,
                "kind": "cleaned_open_sky_events",
                "path": relative,
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                **header_audit,
            }
        )

    commands: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    distribution = config["runtime"]["wsl_distribution"]
    for declared in config["included_branches"]:
        branch = branches[declared["name"]]
        if branch["obsid"] != declared["obsid"]:
            raise RuntimeError(f"declared branch ObsID changed: {branch['name']}")
        declared_anchor_obsids = [
            segments[index]["obsids"][0] for index in declared["anchor_segments"]
        ]
        branch_dir = staging / branch["name"]
        branch_dir.mkdir()
        models = fit_bracketed_models(
            declared_anchor_obsids, fe55_valid, pxcal_valid
        )
        history = branch_dir / "common_differential.ghf"
        template_obsid = declared_anchor_obsids[0]
        history_audit = closure.build_candidate_history(
            history,
            fe55_hdus[template_obsid],
            fe55_data[template_obsid],
            pxcal_valid[branch["obsid"]],
            branch,
            models,
            extension,
        )
        selected = branch_dir / "selected_cleaned_sky.evt"
        copy_result = application.run_wsl(
            distribution,
            ftcopy_command(config, sky_paths[branch["obsid"]], selected, branch),
            timeout=1200,
        )
        commands.append({"stage": "ftcopy", "branch": branch["name"], **copy_result})
        if copy_result["exit_code"] != 0 or not selected.is_file():
            raise RuntimeError(f"cleaned sky selection failed: {branch['name']}")
        selected_rows = application.event_rows(selected)
        if selected_rows <= 0:
            raise RuntimeError(f"cleaned sky branch is empty: {branch['name']}")
        output = branch_dir / "calibrated_cleaned_sky.evt"
        apply_result = application.run_wsl(
            distribution,
            application.rslpha2pi_command(config, selected, output, history),
            timeout=1200,
        )
        commands.append(
            {"stage": "rslpha2pi", "branch": branch["name"], **apply_result}
        )
        if apply_result["exit_code"] != 0 or not output.is_file():
            raise RuntimeError(f"cleaned sky calibration failed: {branch['name']}")
        output_audit = application.audit_output(output)
        passed = (
            output_audit["rows"] == selected_rows
            and output_audit["null_epi2"] == 0
            and output_audit["null_temp"] == 0
            and output_audit["null_pi_not_explained_by_negative_epi2"] == 0
            and output_audit["negative_epi2_without_null_pi"] == 0
        )
        applications.append(
            {
                "branch": branch["name"],
                "obsid": branch["obsid"],
                "anchor_segments": declared["anchor_segments"],
                "anchor_obsids": declared_anchor_obsids,
                "selected_rows": selected_rows,
                "models": {str(key): value for key, value in models.items()},
                "history": history_audit,
                "output": output_audit,
                "passed": passed,
            }
        )
    gate = (
        len(applications) == config["terminal_gate"]["required_branch_outputs"]
        and all(command["exit_code"] == 0 for command in commands)
        and all(item["passed"] for item in applications)
    )
    if not gate:
        raise RuntimeError("bracketed science calibration terminal gate failed")
    os.replace(staging, scratch_root)
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_bracketed_science_calibration_completed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "source_files": sources,
        "applications": applications,
        "commands": commands,
        "terminal_gate_passed": gate,
        "excluded_branches": config["excluded_branches"],
        "cluster_sky_event_rows_read": True,
        "cluster_sky_event_energies_recalculated": True,
        "cluster_energy_distribution_inspected_or_fit": False,
        "cluster_velocity_fit": False,
        "validation_or_holdout_accessed": False,
        "decision": "authorize_freeze_of_reduced_a2319_spectral_region_protocol",
        "authorization": {
            "freeze_reduced_a2319_spectral_region_protocol": True,
            "inspect_or_fit_cluster_energy_distribution": False,
            "fit_cluster_velocity": False,
            "access_validation_or_holdout_assets": False,
            "open_lensing_halo_or_gravity_targets": False,
            "change_gravity_formula_or_parameters": False,
            "derive_or_select_action": False,
        },
    }
    output_path = ROOT / config["paths"]["report"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = build_report()
    print(
        json.dumps(
            {
                "status": result["status"],
                "applications": result["applications"],
                "terminal_gate_passed": result["terminal_gate_passed"],
                "cluster_energy_distribution_inspected_or_fit": result[
                    "cluster_energy_distribution_inspected_or_fit"
                ],
                "cluster_velocity_fit": result["cluster_velocity_fit"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
