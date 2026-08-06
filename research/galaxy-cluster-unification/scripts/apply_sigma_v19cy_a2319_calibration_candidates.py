#!/usr/bin/env python3
"""Apply frozen A2319 gain candidates to calibration-pixel events only."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19cy_a2319_calibration_application_candidates.json"
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


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if len(drive) != 1 or not drive.isalpha():
        raise RuntimeError(f"unsupported Windows path for WSL: {resolved}")
    tail = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def run_wsl(distribution: str, command: str, timeout: int = 600) -> dict[str, Any]:
    process = subprocess.run(
        ["wsl.exe", "-d", distribution, "--", "bash", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def validate_inputs(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    if config.get("protocol_version") != (
        "SIGMA-V19CY-A2319-CALIBRATION-APPLICATION-CANDIDATES-1.0.0"
    ):
        raise RuntimeError("unexpected calibration-application protocol")
    if config.get("status") != (
        "frozen after the scalar topology gate passed, but before any gain-history array was copied or any calibration event energy was recalculated"
    ):
        raise RuntimeError("calibration-application protocol is not frozen")
    for name in ("topology_report", "environment_report", "download_provenance"):
        path = ROOT / config["parents"][name]
        if not path.is_file() or sha256(path) != config["parents"][f"{name}_sha256"]:
            raise RuntimeError(f"frozen calibration-application parent changed: {path}")
    topology = load_json(ROOT / config["parents"]["topology_report"])
    environment = load_json(ROOT / config["parents"]["environment_report"])
    if not topology.get("topology_gate_passed"):
        raise RuntimeError("gain-reconstruction topology gate did not pass")
    if not environment.get("gates", {}).get("all_runtime_commands_exited_zero"):
        raise RuntimeError("audited HEASoft runtime gate did not pass")
    authorization = config["authorization"]
    for key in (
        "inspect_or_fit_energy_distribution",
        "read_cluster_sky_event_rows",
        "fit_cluster_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed calibration-application boundary is open: {key}")
    provenance = load_json(ROOT / config["parents"]["download_provenance"])
    return config, topology, provenance


def verified_path(
    relative: str,
    raw_root: Path,
    provenance_by_path: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    path = (raw_root / relative).resolve()
    if not path.is_relative_to(raw_root):
        raise RuntimeError(f"calibration-application path escapes raw root: {relative}")
    terminal = provenance_by_path.get(relative)
    if terminal is None:
        raise RuntimeError(f"calibration-application source absent from provenance: {relative}")
    if not path.is_file() or path.stat().st_size != terminal["bytes"]:
        raise RuntimeError(f"calibration-application source size changed: {relative}")
    if sha256(path) != terminal["sha256"]:
        raise RuntimeError(f"calibration-application source hash changed: {relative}")
    return path, terminal


def candidate_residual(candidate: dict[str, str], branch: dict[str, Any], time: float) -> float:
    summary = branch["calibration_pixel_residual"]
    if candidate["name"] == "fe55_branch_only":
        return 0.0
    if candidate["name"] == "branch_median_common_mode":
        return float(summary["median"])
    if candidate["name"] == "branch_linear_common_mode":
        midpoint = 0.5 * (float(branch["start"]) + float(branch["stop"]))
        return float(summary["median"]) + float(summary["linear_slope_per_hour"]) * (
            time - midpoint
        ) / 3600.0
    raise RuntimeError(f"unknown frozen candidate: {candidate['name']}")


def fitted_temperature(branch: dict[str, Any], pixel: int, time: float) -> float:
    fit = branch["fits"][str(pixel)]
    if not fit.get("finite"):
        raise RuntimeError(f"non-finite topology fit for {branch['name']} pixel {pixel}")
    return float(fit["temperature_at_center"]) + float(fit["slope_per_second"]) * (
        time - float(fit["time_center"])
    )


def load_gain_source(path: Path, extension: str) -> tuple[list[fits.hdu.base.ExtensionHDU], np.ndarray]:
    with gzip.open(path, "rb") as stream, fits.open(
        stream, memmap=False, mode="readonly"
    ) as hdus:
        copies = [hdu.copy() for hdu in hdus]
        data = np.asarray(hdus[extension].data).copy()
    return copies, data


def nearest_record(data: np.ndarray, pixel: int, time: float) -> np.void:
    indexes = np.flatnonzero(np.asarray(data["PIXEL"], dtype=int) == pixel)
    if not indexes.size:
        raise RuntimeError(f"gain source has no row for pixel {pixel}")
    times = np.asarray(data["TIME"][indexes], dtype=float)
    return data[indexes[int(np.argmin(np.abs(times - time)))]]


def build_drift_file(
    output: Path,
    branch: dict[str, Any],
    candidate: dict[str, str],
    source_by_obsid: dict[str, tuple[list[fits.hdu.base.ExtensionHDU], np.ndarray]],
    extension: str,
) -> dict[str, Any]:
    anchors = branch["anchors"]
    start_obsid = anchors[0]["obsids"][0]
    stop_obsid = anchors[-1]["obsids"][0]
    start_hdus, start_data = source_by_obsid[start_obsid]
    _, stop_data = source_by_obsid[stop_obsid]
    dtype = start_data.dtype
    if stop_data.dtype != dtype:
        raise RuntimeError("anchor gain-history schemas differ")
    records = []
    endpoints = (float(branch["start"]), float(branch["stop"]))
    for endpoint, data in ((endpoints[0], start_data), (endpoints[1], stop_data)):
        for pixel in range(36):
            records.append(nearest_record(data, pixel, endpoint).copy())
    new_data = np.asarray(records, dtype=dtype)
    for index, endpoint in enumerate((endpoints[0], endpoints[1])):
        residual = candidate_residual(candidate, branch, endpoint)
        for pixel in range(36):
            row = index * 36 + pixel
            new_data["TIME"][row] = endpoint
            new_data["PIXEL"][row] = pixel
            new_data["TEMP_FIT"][row] = fitted_temperature(branch, pixel, endpoint) + residual
    table_index = next(
        index for index, hdu in enumerate(start_hdus) if hdu.name == extension
    )
    header = start_hdus[table_index].header.copy()
    header["TSTART"] = endpoints[0]
    header["TSTOP"] = endpoints[1]
    start_hdus[table_index] = fits.BinTableHDU(
        data=new_data,
        header=header,
        name=extension,
    )
    fits.HDUList(start_hdus).writeto(output, overwrite=False, checksum=True)
    counts = Counter(int(value) for value in new_data["PIXEL"])
    finite = bool(np.isfinite(np.asarray(new_data["TEMP_FIT"], dtype=float)).all())
    return {
        "rows": len(new_data),
        "per_pixel_rows": {str(pixel): counts[pixel] for pixel in range(36)},
        "temp_fit_finite": finite,
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
    }


def runtime_environment(config: dict[str, Any]) -> str:
    runtime = config["runtime"]
    caldb = runtime["caldb_root"]
    prefix = runtime["heasoft_prefix"]
    return (
        f"export CONDA_PREFIX={shlex.quote(prefix)}; "
        f"source {shlex.quote(prefix + '/bin/heainit.sh')} >/dev/null 2>&1; "
        f"export CALDB={shlex.quote(caldb)}; "
        f"export CALDBCONFIG={shlex.quote(caldb + '/software/tools/caldb.config')}; "
        f"export CALDBALIAS={shlex.quote(caldb + '/software/tools/alias_config.fits')}; "
    )


def ftcopy_command(config: dict[str, Any], source: Path, output: Path, branch: dict[str, Any]) -> str:
    selection = (
        f"{to_wsl_path(source)}[EVENTS][TIME>={branch['start']}&&TIME<={branch['stop']}"
        f"&&PIXEL=={config['inputs']['calibration_pixel']}"
        f"&&ITYPE=={config['inputs']['high_resolution_primary_itype']}]"
    )
    return (
        runtime_environment(config)
        + "ftcopy "
        + shlex.quote(selection)
        + " "
        + shlex.quote(to_wsl_path(output))
        + " copyall=yes clobber=yes history=yes"
    )


def rslpha2pi_command(
    config: dict[str, Any], source: Path, output: Path, drift: Path
) -> str:
    params = config["runtime"]["rslpha2pi"]
    pieces = [
        "rslpha2pi",
        f"infile={to_wsl_path(source)}",
        f"outfile={to_wsl_path(output)}",
        f"driftfile={to_wsl_path(drift)}",
    ]
    pieces.extend(f"{key}={value}" for key, value in params.items())
    return runtime_environment(config) + " ".join(shlex.quote(piece) for piece in pieces)


def event_rows(path: Path) -> int:
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        return int(hdus["EVENTS"].header["NAXIS2"])


def audit_output(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        data = hdus["EVENTS"].data
        rows = len(data)
        pi = np.asarray(data["PI"])
        epi2 = np.asarray(data["EPI2"], dtype=float)
        temp = np.asarray(data["TEMP"], dtype=float)
        pi_null = hdus["EVENTS"].header.get("TNULL" + str(data.names.index("PI") + 1))
        null_pi = int(np.sum(pi == pi_null)) if pi_null is not None else 0
        null_epi2 = int(np.sum(~np.isfinite(epi2)))
        null_temp = int(np.sum(~np.isfinite(temp)))
    return {
        "rows": rows,
        "null_pi": null_pi,
        "null_epi2": null_epi2,
        "null_temp": null_temp,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, topology, provenance = validate_inputs(config_path)
    raw_root = (ROOT / config["paths"]["raw_root"]).resolve()
    scratch_root = (ROOT / config["paths"]["scratch_root"]).resolve()
    expected_parent = (ROOT / "tmp").resolve()
    if not scratch_root.is_relative_to(expected_parent):
        raise RuntimeError("calibration scratch root must remain inside repository tmp")
    if scratch_root.exists():
        raise RuntimeError(f"refusing to overwrite calibration scratch root: {scratch_root}")
    scratch_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=scratch_root.name + ".installing.", dir=scratch_root.parent))

    provenance_by_path = {
        record["download_path"]: record for record in provenance["records"]
    }
    source_paths: dict[str, Path] = {}
    source_records: list[dict[str, Any]] = []
    source_by_obsid: dict[str, tuple[list[fits.hdu.base.ExtensionHDU], np.ndarray]] = {}
    gain_obsids = sorted(
        {
            anchor_obsid
            for branch in topology["branches"]
            for anchor in branch["anchors"]
            for anchor_obsid in anchor["obsids"]
        }
    )
    for obsid in gain_obsids:
        relative = config["inputs"]["fe55_history_template"].format(obsid=obsid)
        path, terminal = verified_path(relative, raw_root, provenance_by_path)
        source_by_obsid[obsid] = load_gain_source(
            path, config["inputs"]["gain_history_extension"]
        )
        source_records.append(
            {"obsid": obsid, "kind": "fe55_history", "path": relative, **terminal}
        )
    for obsid in sorted({branch["obsid"] for branch in topology["branches"]}):
        relative = config["inputs"]["calibration_event_template"].format(obsid=obsid)
        path, terminal = verified_path(relative, raw_root, provenance_by_path)
        source_paths[obsid] = path
        source_records.append(
            {"obsid": obsid, "kind": "calibration_events", "path": relative, **terminal}
        )

    commands: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    distribution = config["runtime"]["wsl_distribution"]
    for branch in topology["branches"]:
        branch_dir = staging / branch["name"]
        branch_dir.mkdir(parents=True)
        selected = branch_dir / "selected_pixel12_hp.evt"
        ft_result = run_wsl(
            distribution,
            ftcopy_command(config, source_paths[branch["obsid"]], selected, branch),
        )
        commands.append({"stage": "ftcopy", "branch": branch["name"], **ft_result})
        if ft_result["exit_code"] != 0 or not selected.is_file():
            raise RuntimeError(f"ftcopy failed for {branch['name']}: {ft_result['stderr']}")
        selected_rows = event_rows(selected)
        if selected_rows <= 0:
            raise RuntimeError(f"no calibration events selected for {branch['name']}")
        for candidate in config["candidates"]:
            drift = branch_dir / f"{candidate['name']}.ghf"
            drift_audit = build_drift_file(
                drift,
                branch,
                candidate,
                source_by_obsid,
                config["inputs"]["gain_history_extension"],
            )
            output = branch_dir / f"{candidate['name']}.evt"
            result = run_wsl(
                distribution,
                rslpha2pi_command(config, selected, output, drift),
                timeout=1200,
            )
            commands.append(
                {
                    "stage": "rslpha2pi",
                    "branch": branch["name"],
                    "candidate": candidate["name"],
                    **result,
                }
            )
            if result["exit_code"] != 0 or not output.is_file():
                raise RuntimeError(
                    f"rslpha2pi failed for {branch['name']} {candidate['name']}: {result['stderr']}"
                )
            output_audit = audit_output(output)
            passed = (
                drift_audit["rows"] == 72
                and all(value == 2 for value in drift_audit["per_pixel_rows"].values())
                and drift_audit["temp_fit_finite"]
                and output_audit["rows"] == selected_rows
                and output_audit["null_pi"] == 0
                and output_audit["null_epi2"] == 0
                and output_audit["null_temp"] == 0
            )
            applications.append(
                {
                    "branch": branch["name"],
                    "obsid": branch["obsid"],
                    "candidate": candidate["name"],
                    "selected_rows": selected_rows,
                    "drift": drift_audit,
                    "output": output_audit,
                    "passed": passed,
                }
            )
    expected_outputs = config["terminal_gate"]["required_candidate_branch_outputs"]
    gate = (
        len(applications) == expected_outputs
        and all(command["exit_code"] == 0 for command in commands)
        and all(item["passed"] for item in applications)
    )
    if not gate:
        raise RuntimeError("frozen calibration-application terminal gate failed")
    os.replace(staging, scratch_root)
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_calibration_application_candidates_completed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "scratch_root": config["paths"]["scratch_root"],
        "source_files": source_records,
        "applications": applications,
        "commands": commands,
        "terminal_gate_passed": gate,
        "gain_history_array_fields_copied_without_inspection": True,
        "calibration_event_energies_recalculated": True,
        "energy_distribution_inspected_or_fit": False,
        "cluster_sky_event_accessed": False,
        "cluster_velocity_fit": False,
        "validation_or_holdout_accessed": False,
        "decision": "authorize_calibration_pixel_line_shape_gate_freeze",
        "authorization": {
            "freeze_calibration_pixel_line_shape_gate": True,
            "inspect_or_fit_energy_distribution": False,
            "read_cluster_sky_event_rows": False,
            "fit_cluster_velocity": False,
            "access_validation_or_holdout_assets": False,
            "open_lensing_halo_or_gravity_targets": False,
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
                "applications": len(result["applications"]),
                "terminal_gate_passed": result["terminal_gate_passed"],
                "energy_distribution_inspected_or_fit": result[
                    "energy_distribution_inspected_or_fit"
                ],
                "cluster_sky_event_accessed": result["cluster_sky_event_accessed"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
