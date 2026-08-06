#!/usr/bin/env python3
"""Count-only readiness audit for frozen A2319 detector regions."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import apply_sigma_v19cy_a2319_calibration_candidates as application

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cy_a2319_region_readiness.json"
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    if config.get("protocol_version") != "SIGMA-V19CY-A2319-REGION-READINESS-1.0.0":
        raise RuntimeError("unexpected region-readiness protocol")
    if config.get("status") != (
        "frozen before reading, summarizing, binning, plotting, or fitting any "
        "calibrated A2319 science energy value or distribution"
    ):
        raise RuntimeError("region-readiness protocol is not frozen")
    parent_path = ROOT / config["parents"]["calibration_report"]
    if not parent_path.is_file() or sha256(parent_path) != config["parents"][
        "calibration_report_sha256"
    ]:
        raise RuntimeError("frozen calibration parent changed")
    parent = load_json(parent_path)
    if not parent.get("terminal_gate_passed"):
        raise RuntimeError("calibration parent did not pass")
    if parent.get("cluster_energy_distribution_inspected_or_fit"):
        raise RuntimeError("calibration parent crossed the energy boundary")
    if parent.get("validation_or_holdout_accessed"):
        raise RuntimeError("calibration parent opened a sealed asset")
    authorization = config["authorization"]
    for key in (
        "read_or_summarize_any_energy_column",
        "plot_or_bin_energy_distribution",
        "generate_response_or_background",
        "fit_spectrum_or_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed readiness boundary is open: {key}")
    validate_partition(config)
    return config, parent


def validate_partition(config: dict[str, Any]) -> None:
    expected = config["pixel_partition"]["science_pixels"]
    if len(expected) != len(set(expected)):
        raise RuntimeError("science-pixel list contains duplicates")
    for name, pointing in config["pixel_partition"]["pointings"].items():
        flat = [pixel for pixels in pointing["regions"].values() for pixel in pixels]
        if sorted(flat) != sorted(expected) or len(flat) != len(set(flat)):
            raise RuntimeError(f"{name} regions do not exactly partition science pixels")


def screen_expression(config: dict[str, Any], pixels: list[int] | None = None) -> str:
    screen = config["count_only_screen"]
    value = f"(RISE_TIME+{screen['rise_time_deriv_coefficient']}*DERIV_MAX)"
    terms = [
        f"ITYPE=={screen['high_resolution_primary_itype']}",
        f"{value}>{screen['rise_time_lower_exclusive']}",
        f"{value}<{screen['rise_time_upper_exclusive']}",
        "STATUS[4]==b0",
        f"PIXEL!={screen['excluded_pixel']}",
    ]
    if pixels is not None:
        terms.append("(" + "||".join(f"PIXEL=={pixel}" for pixel in pixels) + ")")
    return "&&".join(terms)


def ftcopy_command(
    config: dict[str, Any], source: Path, output: Path, pixels: list[int] | None
) -> str:
    selection = (
        f"{application.to_wsl_path(source)}[EVENTS][{screen_expression(config, pixels)}]"
    )
    return (
        application.runtime_environment(config)
        + "ftcopy "
        + shlex.quote(selection)
        + " "
        + shlex.quote(application.to_wsl_path(output))
        + " copyall=yes clobber=yes history=yes"
    )


def event_rows(path: Path) -> int:
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        return int(hdus["EVENTS"].header["NAXIS2"])


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, parent = validate_inputs(config_path)
    calibrated_root = (ROOT / config["paths"]["calibrated_scratch_root"]).resolve()
    scratch_root = (ROOT / config["paths"]["scratch_root"]).resolve()
    if scratch_root.exists():
        raise RuntimeError(f"refusing to overwrite readiness scratch root: {scratch_root}")
    if not scratch_root.is_relative_to((ROOT / "tmp").resolve()):
        raise RuntimeError("readiness scratch root must remain under repository tmp")
    scratch_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=scratch_root.name + ".installing.", dir=scratch_root.parent)
    )
    parent_by_branch = {item["branch"]: item for item in parent["applications"]}
    commands: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str], int] = defaultdict(int)
    distribution = config["runtime"]["wsl_distribution"]

    for pointing_name, pointing in config["pixel_partition"]["pointings"].items():
        for branch_name in pointing["branches"]:
            parent_item = parent_by_branch.get(branch_name)
            if parent_item is None or parent_item["obsid"] != pointing["obsid"]:
                raise RuntimeError(f"frozen branch/pointing mismatch: {branch_name}")
            source = calibrated_root / branch_name / "calibrated_cleaned_sky.evt"
            if not source.is_file() or sha256(source) != parent_item["output"]["sha256"]:
                raise RuntimeError(f"calibrated branch output changed: {source}")
            branch_dir = staging / branch_name
            branch_dir.mkdir()
            full_output = branch_dir / "screened_full_array.evt"
            full_result = application.run_wsl(
                distribution,
                ftcopy_command(config, source, full_output, None),
                timeout=1200,
            )
            commands.append({"branch": branch_name, "region": "full_array", **full_result})
            if full_result["exit_code"] != 0 or not full_output.is_file():
                raise RuntimeError(f"full-array count-only screen failed: {branch_name}")
            full_rows = event_rows(full_output)
            region_records: list[dict[str, Any]] = []
            for region_name, pixels in pointing["regions"].items():
                output = branch_dir / f"region_{region_name}.evt"
                result = application.run_wsl(
                    distribution,
                    ftcopy_command(config, source, output, pixels),
                    timeout=1200,
                )
                commands.append({"branch": branch_name, "region": region_name, **result})
                if result["exit_code"] != 0 or not output.is_file():
                    raise RuntimeError(f"region count-only screen failed: {branch_name}/{region_name}")
                rows = event_rows(output)
                aggregate[(pointing_name, region_name)] += rows
                region_records.append(
                    {
                        "region": region_name,
                        "pixels": pixels,
                        "rows": rows,
                        "bytes": output.stat().st_size,
                        "sha256": sha256(output),
                    }
                )
            partition_rows = sum(item["rows"] for item in region_records)
            branches.append(
                {
                    "pointing": pointing_name,
                    "obsid": pointing["obsid"],
                    "branch": branch_name,
                    "input_rows": parent_item["output"]["rows"],
                    "screened_full_array_rows": full_rows,
                    "partition_rows": partition_rows,
                    "partition_exact": partition_rows == full_rows,
                    "regions": region_records,
                }
            )

    aggregate_records = [
        {"pointing": key[0], "region": key[1], "rows": rows}
        for key, rows in sorted(aggregate.items())
    ]
    gate = (
        sum(len(item["regions"]) for item in branches)
        == config["terminal_gate"]["required_branch_region_outputs"]
        and all(command["exit_code"] == 0 for command in commands)
        and all(item["partition_exact"] for item in branches)
        and all(region["rows"] > 0 for item in branches for region in item["regions"])
        and all(
            item["rows"]
            >= config["terminal_gate"]["minimum_aggregate_rows_per_detector_region"]
            for item in aggregate_records
        )
    )
    if not gate:
        raise RuntimeError("A2319 detector-region readiness gate failed")
    os.replace(staging, scratch_root)
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_detector_region_count_readiness_completed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "parent_calibration_report_sha256": sha256(
            ROOT / config["parents"]["calibration_report"]
        ),
        "branches": branches,
        "aggregate_detector_regions": aggregate_records,
        "commands": commands,
        "terminal_gate_passed": gate,
        "energy_column_or_distribution_read": False,
        "spectrum_or_velocity_fit": False,
        "validation_or_holdout_accessed": False,
        "decision": "authorize_freeze_of_a2319_detector_region_spectral_protocol",
        "authorization": {
            "freeze_detector_region_spectral_protocol": True,
            "read_or_fit_energy_distribution": False,
            "fit_velocity": False,
            "access_validation_or_holdout_assets": False,
        },
    }
    report_path = ROOT / config["paths"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = build_report()
    print(
        json.dumps(
            {
                "status": result["status"],
                "branches": result["branches"],
                "aggregate_detector_regions": result["aggregate_detector_regions"],
                "terminal_gate_passed": result["terminal_gate_passed"],
                "energy_column_or_distribution_read": result[
                    "energy_column_or_distribution_read"
                ],
                "spectrum_or_velocity_fit": result["spectrum_or_velocity_fit"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
