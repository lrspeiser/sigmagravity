#!/usr/bin/env python3
"""Test frozen nearest-anchor A2319 gain rules on endpoint calibration folds."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import apply_sigma_v19cy_a2319_calibration_candidates as application
import fit_sigma_v19cy_a2319_calibration_line_shape as line_shape
import validate_sigma_v19cy_a2319_common_differential_gain as closure

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cy_a2319_endpoint_gain_candidates.json"
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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    if config.get("protocol_version") != (
        "SIGMA-V19CY-A2319-ENDPOINT-GAIN-CANDIDATES-1.0.0"
    ):
        raise RuntimeError("unexpected endpoint-gain protocol")
    expected_status = (
        "frozen after both interior common/differential folds passed and both endpoint "
        "folds failed, but before either nearest-anchor endpoint candidate was applied "
        "to a held-out calibration event"
    )
    if config.get("status") != expected_status:
        raise RuntimeError("endpoint-gain candidates are not frozen")
    parents: dict[str, dict[str, Any]] = {}
    for name in ("closure_report", "topology_report", "download_provenance"):
        path = ROOT / config["parents"][name]
        if not path.is_file() or sha256(path) != config["parents"][f"{name}_sha256"]:
            raise RuntimeError(f"endpoint-gain parent changed: {path}")
        parents[name] = load_json(path)
    line_config_path = ROOT / config["parents"]["line_shape_config"]
    if (
        not line_config_path.is_file()
        or sha256(line_config_path)
        != config["parents"]["line_shape_config_sha256"]
    ):
        raise RuntimeError("frozen endpoint line-shape config changed")
    parent = parents["closure_report"]
    folds = {fold["segment"]: fold for fold in parent["folds"]}
    if parent.get("terminal_gate_passed") or parent.get("cluster_sky_event_accessed"):
        raise RuntimeError("unexpected parent closure state")
    if not folds[1]["comparison"]["passed"] or not folds[2]["comparison"]["passed"]:
        raise RuntimeError("parent interior interpolation evidence did not pass")
    if folds[0]["comparison"]["passed"] or folds[3]["comparison"]["passed"]:
        raise RuntimeError("parent endpoint failures are absent")
    for key in (
        "read_cluster_sky_event_rows",
        "apply_gain_to_cluster_sky_events",
        "fit_cluster_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if config["authorization"][key]:
            raise RuntimeError(f"sealed endpoint boundary is open: {key}")
    return (
        config,
        parent,
        parents["topology_report"],
        parents["download_provenance"],
        load_json(line_config_path),
    )


def fit_nearest_models(
    mode: str,
    anchor_fe55: np.ndarray,
    anchor_pxcal: np.ndarray,
) -> dict[int, dict[str, float | int]]:
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
        selected = anchor_fe55[anchor_fe55["PIXEL"] == pixel]
        times = np.asarray(selected["TIME"], dtype=float)
        temperatures = np.asarray(selected["TEMP_FIT"], dtype=float)
        common = closure.common_temperature(anchor_pxcal, times)
        differential = temperatures - common
        if len(times) < 2 or not np.isfinite(differential).all():
            raise RuntimeError(f"insufficient nearest-anchor data for pixel {pixel}")
        center = float(np.mean(times))
        if mode == "nearest_anchor_constant":
            intercept = float(np.median(differential))
            slope = 0.0
        elif mode == "nearest_anchor_linear":
            slope_value, intercept_value = np.polyfit(times - center, differential, 1)
            slope = float(slope_value)
            intercept = float(intercept_value)
        else:
            raise RuntimeError(f"unknown endpoint candidate: {mode}")
        if not np.isfinite([intercept, slope]).all():
            raise RuntimeError(f"non-finite nearest-anchor model for pixel {pixel}")
        models[pixel] = {
            "rows": len(times),
            "time_center": center,
            "differential_at_center": intercept,
            "slope_per_second": slope,
        }
    return models


def select_candidate(results: dict[str, list[dict[str, Any]]], order: list[str]) -> dict[str, Any]:
    summaries = []
    for candidate in order:
        fold_passes = [fold["comparison"]["passed"] for fold in results[candidate]]
        summaries.append(
            {
                "candidate": candidate,
                "fold_passes": fold_passes,
                "passed_both_endpoint_folds": len(fold_passes) == 2 and all(fold_passes),
            }
        )
    passing = [row for row in summaries if row["passed_both_endpoint_folds"]]
    selected = min(passing, key=lambda row: order.index(row["candidate"])) if passing else None
    return {
        "summaries": summaries,
        "passed": selected is not None,
        "selected": selected["candidate"] if selected else None,
    }


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, parent, topology, provenance, line_config = validate_inputs(config_path)
    raw_root = (ROOT / config["paths"]["raw_root"]).resolve()
    scratch_root = (ROOT / config["paths"]["scratch_root"]).resolve()
    if scratch_root.exists():
        raise RuntimeError(f"refusing to overwrite endpoint scratch root: {scratch_root}")
    if not scratch_root.is_relative_to((ROOT / "tmp").resolve()):
        raise RuntimeError("endpoint scratch root must remain under repository tmp")
    scratch_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=scratch_root.name + ".installing.", dir=scratch_root.parent)
    )
    provenance_by_path = {item["download_path"]: item for item in provenance["records"]}
    segments = {segment["segment"]: segment for segment in topology["segment_sets"]["10800"]}
    extension = config["inputs"]["gain_history_extension"]
    minimum = config["inputs"]["minimum_gain_events"]

    template_path = (ROOT / config["paths"]["line_template"]).resolve()
    if (
        not template_path.is_file()
        or sha256(template_path) != line_config["line_template"]["sha256"]
    ):
        raise RuntimeError("frozen endpoint line template changed")
    template = line_shape.read_template(
        template_path, line_config["line_template"]["extension"]
    )
    parent_folds = {fold["segment"]: fold for fold in parent["folds"]}
    candidates = [row["name"] for row in config["candidates"]]
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in candidates}
    commands: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    distribution = config["runtime"]["wsl_distribution"]

    for endpoint in config["endpoint_folds"]:
        held_segment = segments[endpoint["held_out_segment"]]
        held_obsid = endpoint["held_out_obsid"]
        anchor_obsid = endpoint["nearest_anchor_obsid"]
        fold_dir = staging / f"segment_{held_segment['segment']}_{held_obsid}"
        fold_dir.mkdir()

        verified: dict[str, tuple[Path, dict[str, Any]]] = {}
        for obsid, role, template_name in (
            (held_obsid, "held_pxcal", "calibration_pixel_history_template"),
            (held_obsid, "held_events", "filter_wheel_event_template"),
            (anchor_obsid, "anchor_fe55", "fe55_history_template"),
            (anchor_obsid, "anchor_pxcal", "calibration_pixel_history_template"),
        ):
            relative = config["inputs"][template_name].format(obsid=obsid)
            verified[role] = closure.verified_raw_path(
                raw_root, relative, provenance_by_path
            )
            _, record = verified[role]
            sources.append(
                {
                    "held_out_segment": held_segment["segment"],
                    "obsid": obsid,
                    "kind": role,
                    "path": relative,
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
            )
        anchor_hdus, anchor_data, anchor_valid = closure.read_gain_history(
            verified["anchor_fe55"][0], extension, minimum
        )
        _, _, anchor_pxcal = closure.read_gain_history(
            verified["anchor_pxcal"][0], extension, minimum
        )
        _, _, held_pxcal = closure.read_gain_history(
            verified["held_pxcal"][0], extension, minimum
        )

        selected = fold_dir / "selected_main_array_fe55_hp.evt"
        copy_result = application.run_wsl(
            distribution,
            closure.ftcopy_command(
                config, verified["held_events"][0], selected, held_segment
            ),
            timeout=1200,
        )
        commands.append(
            {
                "stage": "ftcopy",
                "held_out_segment": held_segment["segment"],
                "obsid": held_obsid,
                **copy_result,
            }
        )
        if copy_result["exit_code"] != 0 or not selected.is_file():
            raise RuntimeError(f"endpoint calibration selection failed: {held_obsid}")
        selected_rows = application.event_rows(selected)
        if selected_rows != parent_folds[held_segment["segment"]]["selected_rows"]:
            raise RuntimeError("endpoint calibration selection changed from parent closure")

        for candidate in candidates:
            candidate_dir = fold_dir / candidate
            candidate_dir.mkdir()
            models = fit_nearest_models(candidate, anchor_valid, anchor_pxcal)
            history = candidate_dir / "gain.ghf"
            history_audit = closure.build_candidate_history(
                history,
                anchor_hdus,
                anchor_data,
                held_pxcal,
                held_segment,
                models,
                extension,
            )
            output = candidate_dir / "calibrated.evt"
            apply_result = application.run_wsl(
                distribution,
                application.rslpha2pi_command(config, selected, output, history),
                timeout=2400,
            )
            commands.append(
                {
                    "stage": "rslpha2pi",
                    "held_out_segment": held_segment["segment"],
                    "obsid": held_obsid,
                    "candidate": candidate,
                    **apply_result,
                }
            )
            if apply_result["exit_code"] != 0 or not output.is_file():
                raise RuntimeError(
                    f"endpoint candidate application failed: {held_obsid} {candidate}"
                )
            output_audit = application.audit_output(output)
            if (
                output_audit["rows"] != selected_rows
                or output_audit["null_epi2"] != 0
                or output_audit["null_temp"] != 0
                or output_audit["null_pi_not_explained_by_negative_epi2"] != 0
                or output_audit["negative_epi2_without_null_pi"] != 0
            ):
                raise RuntimeError(
                    f"endpoint candidate audit failed: {held_obsid} {candidate}"
                )
            fitted = closure.fit_output(
                output, template, line_config["fit"], config["fit"]
            )
            comparison = closure.compare_fits(
                fitted,
                parent_folds[held_segment["segment"]]["fits"]["control"],
                config["terminal_gate"],
            )
            results[candidate].append(
                {
                    "held_out_segment": held_segment["segment"],
                    "held_out_obsid": held_obsid,
                    "nearest_anchor_segment": endpoint["nearest_anchor_segment"],
                    "nearest_anchor_obsid": anchor_obsid,
                    "role": endpoint["role"],
                    "selected_rows": selected_rows,
                    "models": {str(key): value for key, value in models.items()},
                    "history": history_audit,
                    "output": output_audit,
                    "fit": fitted,
                    "comparison": comparison,
                }
            )

    selection = select_candidate(results, candidates)
    os.replace(staging, scratch_root)
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_endpoint_gain_candidates_completed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "source_files": sources,
        "candidate_results": results,
        "selection": selection,
        "commands": commands,
        "interior_interpolation_evidence_preserved": True,
        "cluster_sky_event_accessed": False,
        "cluster_velocity_fit": False,
        "validation_or_holdout_accessed": False,
        "decision": (
            "authorize_freeze_of_mixed_interpolation_and_selected_endpoint_science_protocol"
            if selection["passed"]
            else "retain_interior_interpolation_and_park_endpoint_extrapolation"
        ),
        "authorization": {
            "freeze_selected_endpoint_science_protocol": selection["passed"],
            "freeze_bracketed_interpolation_science_protocol": True,
            "read_cluster_sky_event_rows": False,
            "apply_gain_to_cluster_sky_events": False,
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
                "candidate_results": {
                    candidate: [
                        {
                            "held_out_segment": fold["held_out_segment"],
                            "held_out_obsid": fold["held_out_obsid"],
                            "comparison": fold["comparison"],
                        }
                        for fold in folds
                    ]
                    for candidate, folds in result["candidate_results"].items()
                },
                "selection": result["selection"],
                "cluster_sky_event_accessed": result["cluster_sky_event_accessed"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
