#!/usr/bin/env python3
"""Fit the frozen A2319 source and weighted NXB spectra with XSPEC."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import apply_sigma_v19cy_a2319_calibration_candidates as application
import generate_sigma_v19cy_a2319_response_components as components
import prepare_sigma_v19cy_a2319_response_inputs as preparation

CONFIG = ROOT / "configs/sigma_v19cy_a2319_response_aware_spectral.json"
COMPONENT_REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_components.json"
)
ARF_REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_arfs.json"
)
REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_aware_spectral.json"
)
EXPECTED_PROTOCOL = "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.4"
NXB_PARAMETER_COUNT = 56
NXB_THAWED_NORMALIZATIONS = (3, 7, 14, 20, 23, 29, 35, 41, 47, 50, 53, 56)
LIGHT_SPEED_KM_S = 299_792.458
XSPEC_TIMEOUT_SECONDS = 7_200
MARKER = "__SIGMA_XSPEC__"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_nxb_model(text: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].lower().startswith("model  2:nxb1 "):
        raise RuntimeError("unexpected public Resolve NXB model header")
    expression = lines[0].split("2:nxb1", 1)[1].strip()
    specs = lines[1:]
    if len(specs) != NXB_PARAMETER_COUNT:
        raise RuntimeError(
            f"expected {NXB_PARAMETER_COUNT} NXB parameters, found {len(specs)}"
        )
    return expression, specs


def shift_nxb_link(spec: str, offset: int) -> str:
    if not spec.startswith("="):
        return spec
    if re.search(r"nxb1:p\d+", spec) is None:
        raise RuntimeError(f"could not identify NXB link: {spec}")

    def replace(match: re.Match[str]) -> str:
        return f"nxb1:p{int(match.group(1)) + offset}"

    shifted = re.sub(r"nxb1:p(\d+)", replace, spec)
    return shifted


def nxb_model_lines(base_specs: list[str], source_group_count: int) -> list[str]:
    if len(base_specs) != NXB_PARAMETER_COUNT or source_group_count not in (1, 2):
        raise ValueError("invalid NXB model layout")
    lines: list[str] = []
    for source_index in range(source_group_count):
        offset = source_index * NXB_PARAMETER_COUNT
        lines.extend(shift_nxb_link(spec, offset) for spec in base_specs)
    for source_index in range(source_group_count):
        offset = source_index * NXB_PARAMETER_COUNT
        lines.extend(
            f"= nxb1:p{offset + local_index}"
            for local_index in range(1, NXB_PARAMETER_COUNT + 1)
        )
    return lines


def _primary_source_group(anchor_offset: int | None, *, background: bool) -> list[str]:
    if anchor_offset is None:
        specs = [
            "0.112 -1 0 0 10 10",
            "8.0 0.01 2 2 20 20",
            "0.3 0.01 0 0 2 2",
            "0.05458 0.00001 0.045 0.045 0.065 0.065",
            "200 1 0 0 1000 1000",
            "0.01 0.0001 0 0 1000000 1000000",
        ]
    else:
        specs = [f"= p{anchor_offset + index}" for index in range(1, 6)]
        specs.append("0 -1 0 0 1000000 1000000" if background else "0.01 0.0001 0 0 1000000 1000000")
    if background and anchor_offset is None:
        specs[-1] = "0 -1 0 0 1000000 1000000"
    return specs


def primary_source_model_lines(source_group_count: int) -> list[str]:
    if source_group_count not in (1, 2):
        raise ValueError("invalid source-group count")
    lines = _primary_source_group(None, background=False)
    if source_group_count == 2:
        lines.extend(_primary_source_group(0, background=False))
    for source_index in range(source_group_count):
        lines.extend(
            _primary_source_group(source_index * 6, background=True)
        )
    return lines


def _two_temperature_source_group(
    anchor_offset: int | None, *, background: bool
) -> list[str]:
    if anchor_offset is None:
        specs = [
            "0.112 -1 0 0 10 10",
            "8.0 0.01 2 2 20 20",
            "0.3 0.01 0 0 2 2",
            "0.05458 0.00001 0.045 0.045 0.065 0.065",
            "200 1 0 0 1000 1000",
            "0.008 0.0001 0 0 1000000 1000000",
            "4.0 0.01 2 2 20 20",
            "= p3",
            "= p4",
            "= p5",
            "0.002 0.0001 0 0 1000000 1000000",
        ]
    else:
        specs = [
            f"= p{anchor_offset + 1}",
            f"= p{anchor_offset + 2}",
            f"= p{anchor_offset + 3}",
            f"= p{anchor_offset + 4}",
            f"= p{anchor_offset + 5}",
            "0 -1 0 0 1000000 1000000" if background else "0.008 0.0001 0 0 1000000 1000000",
            f"= p{anchor_offset + 7}",
            f"= p{anchor_offset + 3}",
            f"= p{anchor_offset + 4}",
            f"= p{anchor_offset + 5}",
            "0 -1 0 0 1000000 1000000" if background else "0.002 0.0001 0 0 1000000 1000000",
        ]
    if background and anchor_offset is None:
        specs[5] = "0 -1 0 0 1000000 1000000"
        specs[10] = "0 -1 0 0 1000000 1000000"
    return specs


def two_temperature_source_model_lines(source_group_count: int) -> list[str]:
    if source_group_count not in (1, 2):
        raise ValueError("invalid source-group count")
    lines = _two_temperature_source_group(None, background=False)
    if source_group_count == 2:
        lines.extend(_two_temperature_source_group(0, background=False))
    for source_index in range(source_group_count):
        lines.extend(
            _two_temperature_source_group(source_index * 11, background=True)
        )
    return lines


def numeric_parameter_bounds(spec: str) -> tuple[float, float] | None:
    if spec.startswith("="):
        return None
    fields = spec.split()
    if len(fields) != 6:
        raise RuntimeError(f"invalid XSPEC parameter specification: {spec}")
    return float(fields[2]), float(fields[5])


def numeric_parameter_delta(spec: str) -> float | None:
    if spec.startswith("="):
        return None
    return float(spec.split()[1])


def nxb_free_parameter_indices(
    base_specs: list[str], source_group_count: int
) -> list[int]:
    if len(base_specs) != NXB_PARAMETER_COUNT or source_group_count not in (1, 2):
        raise ValueError("invalid NXB model layout")
    free: list[int] = []
    for source_index in range(source_group_count):
        offset = source_index * NXB_PARAMETER_COUNT
        free.extend(offset + local_index for local_index in NXB_THAWED_NORMALIZATIONS)
    return free


def nxb_numeric_parameter_indices(
    base_specs: list[str], source_group_count: int
) -> list[int]:
    if len(base_specs) != NXB_PARAMETER_COUNT or source_group_count not in (1, 2):
        raise ValueError("invalid NXB model layout")
    numeric: list[int] = []
    for source_index in range(source_group_count):
        offset = source_index * NXB_PARAMETER_COUNT
        numeric.extend(
            offset + local_index
            for local_index, spec in enumerate(base_specs, start=1)
            if not spec.startswith("=")
        )
    return numeric


def source_free_parameter_indices(source_group_count: int, two_temperature: bool) -> list[int]:
    if two_temperature:
        free = [2, 3, 4, 5, 6, 7, 11]
        if source_group_count == 2:
            free.extend([11 + 6, 11 + 11])
        return free
    free = [2, 3, 4, 5, 6]
    if source_group_count == 2:
        free.append(12)
    return free


def marker_commands(
    source_parameter_count: int,
    source_group_count: int,
    nxb_free_indices: list[int],
    redshift_index: int,
) -> list[str]:
    commands = [
        "tclout stat",
        f'puts "{MARKER} statistic $xspec_tclout"',
        "tclout dof",
        f'puts "{MARKER} dof $xspec_tclout"',
        "tclout covar",
        f'puts "{MARKER} covariance $xspec_tclout"',
        "tclout varpar",
        f'puts "{MARKER} variable_parameters $xspec_tclout"',
    ]
    for index in range(1, source_parameter_count * source_group_count * 2 + 1):
        commands.extend(
            [
                f"tclout param {index}",
                f'puts "{MARKER} source_parameter_{index} $xspec_tclout"',
            ]
        )
    for index in nxb_free_indices:
        commands.extend(
            [
                f"tclout param nxb1:{index}",
                f'puts "{MARKER} nxb_parameter_{index} $xspec_tclout"',
            ]
        )
    commands.extend(
        [
            f"error 1.0 {redshift_index}",
            f"tclout error {redshift_index}",
            f'puts "{MARKER} redshift_error $xspec_tclout"',
            f"tclout param {redshift_index}",
            f'puts "{MARKER} redshift $xspec_tclout"',
            f"tclout sigma {redshift_index}",
            f'puts "{MARKER} redshift_sigma $xspec_tclout"',
        ]
    )
    return commands


def xspec_path(config: dict[str, Any], path: Path) -> str:
    return components.tool_path(config, path)


def build_xspec_deck(
    config: dict[str, Any],
    bundle: list[dict[str, Path]],
    *,
    variant: dict[str, Any],
    nxb_expression: str,
    nxb_specs: list[str],
    log_path: Path,
    session_path: Path,
) -> tuple[str, dict[str, Any]]:
    source_group_count = len(bundle)
    if source_group_count not in (1, 2):
        raise ValueError("A2319 regions require one or two source branches")
    two_temperature = variant["name"] == "two_temperature_shared_velocity"
    source_expression = "tbabs*(bapec+bapec)" if two_temperature else "tbabs*bapec"
    source_specs = (
        two_temperature_source_model_lines(source_group_count)
        if two_temperature
        else primary_source_model_lines(source_group_count)
    )
    source_parameter_count = 11 if two_temperature else 6
    redshift_index = 4
    data_parts: list[str] = []
    for index, row in enumerate(bundle, start=1):
        data_parts.append(f"{index}:{index} {xspec_path(config, row['source_pha'])}")
    for offset, row in enumerate(bundle, start=source_group_count + 1):
        data_parts.append(f"{offset}:{offset} {xspec_path(config, row['nxb_pha'])}")
    commands = [
        "query yes",
        "chatter 10",
        f"log {xspec_path(config, log_path)}",
        "data none",
        "data " + " ".join(data_parts),
    ]
    diagonal = xspec_path(config, ROOT / config["nxb_protocol"]["diagonal_response_path"])
    for index, row in enumerate(bundle, start=1):
        commands.extend(
            [
                f"response 1:{index} {xspec_path(config, row['rmf'])}",
                f"arf 1:{index} {xspec_path(config, row['arf'])}",
                f"response 2:{index} {diagonal}",
            ]
        )
    for index in range(source_group_count + 1, 2 * source_group_count + 1):
        commands.extend(
            [
                f"response 1:{index} {diagonal}",
                f"response 2:{index} {diagonal}",
            ]
        )
    source_range = f"1-{source_group_count}"
    nxb_range = f"{source_group_count + 1}-{2 * source_group_count}"
    commands.extend(
        [
            "abund lodd",
            "xsect vern",
            "xset APECROOT " + config["fit_protocol"]["atomdb"]["xspec_apecroot"],
            f"statistic cstat {source_range}",
            f"statistic chi standard {nxb_range}",
            "method leven 1000 0.0001",
            "model " + source_expression,
            *source_specs,
            "model 2:nxb1 " + nxb_expression,
            *nxb_model_lines(nxb_specs, source_group_count),
        ]
    )
    nxb_band = config["fit_protocol"]["nxb_constraint_band_keV"]
    for index in range(1, source_group_count + 1):
        commands.append(f"ignore {index}:**")
    for index in range(source_group_count + 1, 2 * source_group_count + 1):
        commands.append(f"ignore {index}:**-{nxb_band[0]} {nxb_band[1]}-**")
    commands.append("fit")
    commands.extend(
        f"freeze nxb1:{index}"
        for index in nxb_numeric_parameter_indices(nxb_specs, source_group_count)
    )
    commands.extend(
        f"thaw nxb1:{index}"
        for index in nxb_free_parameter_indices(nxb_specs, source_group_count)
    )
    commands.append("fit")
    source_band = variant["band_keV"]
    for index in range(1, source_group_count + 1):
        commands.extend(
            [
                f"notice {index}:**",
                f"ignore {index}:**-{source_band[0]} {source_band[1]}-**",
            ]
        )
    commands.append("fit")
    nxb_free = nxb_free_parameter_indices(nxb_specs, source_group_count)
    commands.extend(
        marker_commands(
            source_parameter_count,
            source_group_count,
            nxb_free,
            redshift_index,
        )
    )
    commands.extend(
        [
            f"save all {xspec_path(config, session_path)}",
            "log none",
            "exit",
        ]
    )
    metadata = {
        "source_group_count": source_group_count,
        "source_expression": source_expression,
        "source_parameter_count_per_group": source_parameter_count,
        "source_parameter_specs": source_specs,
        "source_free_parameter_indices": source_free_parameter_indices(
            source_group_count, two_temperature
        ),
        "nxb_free_parameter_indices": nxb_free,
        "redshift_parameter_index": redshift_index,
        "source_statistic": "cstat",
        "nxb_statistic": "chi standard",
        "source_band_keV": source_band,
        "nxb_constraint_band_keV": nxb_band,
    }
    return "\n".join(commands) + "\n", metadata


def parse_markers(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if not line.startswith(MARKER + " "):
            continue
        rest = line[len(MARKER) + 1 :]
        key, separator, value = rest.partition(" ")
        if not separator or key in parsed:
            raise RuntimeError(f"malformed or duplicate XSPEC marker: {line}")
        parsed[key] = value.strip()
    return parsed


def first_float(value: str) -> float:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", value)
    if match is None:
        raise RuntimeError(f"no numeric value in XSPEC marker: {value}")
    result = float(match.group(0))
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite XSPEC marker: {value}")
    return result


def parse_error(value: str) -> tuple[float, float, str]:
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", value)
    if len(matches) < 2:
        raise RuntimeError(f"invalid XSPEC error marker: {value}")
    low, high = float(matches[0]), float(matches[1])
    status = value.split()[-1] if len(value.split()) > 2 else ""
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise RuntimeError(f"invalid XSPEC error interval: {value}")
    return low, high, status


def velocity_km_s(config: dict[str, Any], redshift: float) -> float:
    protocol = config["fit_protocol"]
    return (
        LIGHT_SPEED_KM_S
        * (redshift - protocol["bcg_redshift"])
        / (1.0 + protocol["bcg_redshift"])
        + protocol["heliocentric_correction_km_s"]
    )


def parameter_at_bound(value: float, bounds: tuple[float, float]) -> bool:
    low, high = bounds
    scale = max(abs(low), abs(high), 1.0)
    tolerance = 1.0e-7 * scale
    return value <= low + tolerance or value >= high - tolerance


def inspect_fit(
    config: dict[str, Any],
    markers: dict[str, str],
    metadata: dict[str, Any],
    nxb_specs: list[str],
) -> dict[str, Any]:
    required = {"statistic", "dof", "covariance", "variable_parameters", "redshift", "redshift_error", "redshift_sigma"}
    missing = sorted(required - set(markers))
    if missing:
        raise RuntimeError(f"missing XSPEC result markers: {missing}")
    redshift = first_float(markers["redshift"])
    low_z, high_z, error_status = parse_error(markers["redshift_error"])
    velocity = velocity_km_s(config, redshift)
    low_velocity = velocity_km_s(config, low_z)
    high_velocity = velocity_km_s(config, high_z)
    source_values: dict[str, float] = {}
    source_bound_hits: list[int] = []
    source_specs = metadata["source_parameter_specs"]
    for index, spec in enumerate(source_specs, start=1):
        key = f"source_parameter_{index}"
        if key not in markers:
            raise RuntimeError(f"missing {key}")
        value = first_float(markers[key])
        source_values[str(index)] = value
        if index in metadata["source_free_parameter_indices"]:
            bounds = numeric_parameter_bounds(spec)
            if bounds is not None and parameter_at_bound(value, bounds):
                source_bound_hits.append(index)
    nxb_values: dict[str, float] = {}
    nxb_bound_hits: list[int] = []
    for global_index in metadata["nxb_free_parameter_indices"]:
        key = f"nxb_parameter_{global_index}"
        if key not in markers:
            raise RuntimeError(f"missing {key}")
        value = first_float(markers[key])
        nxb_values[str(global_index)] = value
        local_index = (global_index - 1) % NXB_PARAMETER_COUNT
        bounds = numeric_parameter_bounds(nxb_specs[local_index])
        if bounds is not None and parameter_at_bound(value, bounds):
            nxb_bound_hits.append(global_index)
    halfwidth = max(abs(velocity - low_velocity), abs(high_velocity - velocity))
    statistic = first_float(markers["statistic"])
    dof = round(first_float(markers["dof"]))
    return {
        "statistic": statistic,
        "dof": dof,
        "redshift": redshift,
        "redshift_profile_interval": [low_z, high_z],
        "redshift_profile_status": error_status,
        "redshift_covariance_sigma": first_float(markers["redshift_sigma"]),
        "velocity_km_s": velocity,
        "velocity_profile_interval_km_s": [low_velocity, high_velocity],
        "velocity_interval_halfwidth_km_s": halfwidth,
        "source_parameters": source_values,
        "nxb_free_parameters": nxb_values,
        "source_hard_bound_hits": source_bound_hits,
        "nxb_hard_bound_hits": nxb_bound_hits,
        "no_free_parameter_at_hard_bound": not source_bound_hits and not nxb_bound_hits,
        "covariance_lower_triangle": markers["covariance"],
        "variable_parameters": markers["variable_parameters"],
    }


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG)
    if config.get("protocol_version") != EXPECTED_PROTOCOL:
        raise RuntimeError("unexpected response-aware spectral protocol")
    authorization = config["authorization"]
    if not authorization["fit_A2319_development_spectra_and_velocities"]:
        raise RuntimeError("A2319 spectral fitting is not authorized")
    if authorization["access_A3667_validation"] or authorization["access_A754_holdout"]:
        raise RuntimeError("sealed validation or holdout access is enabled")
    if authorization["open_lensing_halo_or_gravity_targets"]:
        raise RuntimeError("lensing or gravity targets are not sealed")
    components_report = load_json(COMPONENT_REPORT)
    arf_report = load_json(ARF_REPORT)
    if not components_report.get("component_gate_passed"):
        raise RuntimeError("response-component gate did not pass")
    if not arf_report.get("arf_gate_passed"):
        raise RuntimeError("ARF gate did not pass")
    if arf_report.get("component_report_sha256") != preparation.sha256(COMPONENT_REPORT):
        raise RuntimeError("ARF report belongs to different response components")
    if arf_report.get("config_sha256") != preparation.sha256(CONFIG):
        raise RuntimeError("ARF report belongs to a different fit protocol")
    return config, components_report, arf_report


def assemble_bundles(
    config: dict[str, Any], components_report: dict[str, Any], arf_report: dict[str, Any]
) -> dict[str, list[dict[str, Path]]]:
    product_root = (ROOT / config["paths"]["product_root"]).resolve()
    component_root = product_root / "response_components"
    arf_root = product_root / "response_arfs"
    component_branches = {row["branch"]: row for row in components_report["branches"]}
    arf_branches = {row["branch"]: row for row in arf_report["branches"]}
    bundles: dict[str, list[dict[str, Path]]] = {}
    for branch in config["branches"]:
        name = branch["name"]
        component_regions = {row["region"]: row for row in component_branches[name]["regions"]}
        arf_regions = {row["region"]: row for row in arf_branches[name]["regions"]}
        for region in branch["regions"]:
            component = component_regions[region]
            row = {
                "source_pha": component_root / name / region / "source.pha",
                "nxb_pha": component_root / name / region / "nxb.pha",
                "rmf": component_root / name / region / component["rmf_name"],
                "arf": arf_root / name / region / "response.arf",
            }
            expected_hashes = {
                "source_pha": component["source_pha"]["sha256"],
                "nxb_pha": component["nxb_pha"]["sha256"],
                "rmf": component["rmf"]["sha256"],
                "arf": arf_regions[region]["arf"]["sha256"],
            }
            for role, path in row.items():
                if preparation.sha256(path) != expected_hashes[role]:
                    raise RuntimeError(f"fit input changed: {path}")
            bundles.setdefault(region, []).append(row)
    expected_counts = {"a": 2, "b": 2, "d": 2, "b_prime": 1, "c_prime": 1, "d_prime": 1, "e_prime": 1}
    if {key: len(value) for key, value in bundles.items()} != expected_counts:
        raise RuntimeError("unexpected A2319 branch-to-region fit topology")
    return bundles


def xspec_command(config: dict[str, Any], work: Path, deck: Path) -> str:
    system_pfiles = config["runtime"]["pfiles"].split(";", 1)[1]
    local_pfiles = xspec_path(config, work / "pfiles")
    return (
        application.runtime_environment(config)
        + "mkdir -p "
        + shlex.quote(local_pfiles)
        + "; export PFILES="
        + shlex.quote(local_pfiles + ";" + system_pfiles)
        + "; cd "
        + shlex.quote(xspec_path(config, work))
        + "; xspec - < "
        + shlex.quote(xspec_path(config, deck))
    )


def run_fit(
    config: dict[str, Any],
    region: str,
    bundle: list[dict[str, Path]],
    variant: dict[str, Any],
    nxb_expression: str,
    nxb_specs: list[str],
    staging: Path,
) -> dict[str, Any]:
    work = staging / region / variant["name"]
    work.mkdir(parents=True)
    deck = work / "fit.xcm"
    log = work / "xspec.log"
    session = work / "best_fit.xcm"
    deck_text, metadata = build_xspec_deck(
        config,
        bundle,
        variant=variant,
        nxb_expression=nxb_expression,
        nxb_specs=nxb_specs,
        log_path=log,
        session_path=session,
    )
    deck.write_text(deck_text, encoding="utf-8")
    command = xspec_command(config, work, deck)
    record = application.run_wsl(
        config["runtime"]["wsl_distribution"], command, timeout=XSPEC_TIMEOUT_SECONDS
    )
    if record["exit_code"] != 0:
        raise RuntimeError(
            f"XSPEC failed for {region}/{variant['name']}: {record['stderr']}"
        )
    markers = parse_markers(record["stdout"])
    fit = inspect_fit(config, markers, metadata, nxb_specs)
    if not log.is_file() or not session.is_file():
        raise RuntimeError(f"XSPEC did not write terminal products for {region}/{variant['name']}")
    fit["region"] = region
    fit["variant"] = variant["name"]
    fit["metadata"] = metadata
    fit["inputs"] = [
        {
            role: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": preparation.sha256(path),
            }
            for role, path in row.items()
        }
        for row in bundle
    ]
    fit["command"] = record
    fit["deck"] = {"bytes": deck.stat().st_size, "sha256": preparation.sha256(deck)}
    fit["log"] = {"bytes": log.stat().st_size, "sha256": preparation.sha256(log)}
    fit["session"] = {"bytes": session.stat().st_size, "sha256": preparation.sha256(session)}
    fit["converged"] = (
        math.isfinite(fit["statistic"])
        and fit["dof"] > 0
        and fit["redshift_profile_status"] == "FFFFFFFFF"
        and fit["redshift_profile_interval"][0] <= fit["redshift"] <= fit["redshift_profile_interval"][1]
    )
    return fit


def intervals_overlap(left: list[float], right: list[float], multiplier: float = 2.0) -> bool:
    left_center = 0.5 * (left[0] + left[1])
    right_center = 0.5 * (right[0] + right[1])
    left_scaled = [left_center + multiplier * (left[0] - left_center), left_center + multiplier * (left[1] - left_center)]
    right_scaled = [right_center + multiplier * (right[0] - right_center), right_center + multiplier * (right[1] - right_center)]
    return max(left_scaled[0], right_scaled[0]) <= min(left_scaled[1], right_scaled[1])


def published_comparison(
    config: dict[str, Any], primary_by_region: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    published = config["published_no_ssm_benchmark"]["regions"]
    for region, benchmark in published.items():
        fit = primary_by_region[region]
        difference = fit["velocity_km_s"] - benchmark["velocity_km_s"]
        benchmark_sigma = (
            benchmark["plus_1sigma"] if difference >= 0 else benchmark["minus_1sigma"]
        )
        comparison[region] = {
            "fitted_velocity_km_s": fit["velocity_km_s"],
            "published_velocity_km_s": benchmark["velocity_km_s"],
            "difference_km_s": difference,
            "difference_over_published_directional_1sigma": difference / benchmark_sigma,
            "diagnostic_only": True,
        }
    return comparison


def generate() -> dict[str, Any]:
    config, component_report, arf_report = validate_inputs()
    bundles = assemble_bundles(config, component_report, arf_report)
    nxb_path = ROOT / config["nxb_protocol"]["empirical_model_path"]
    nxb_expression, nxb_specs = parse_nxb_model(nxb_path.read_text(encoding="utf-8"))
    product_root = (ROOT / config["paths"]["product_root"]).resolve()
    output_root = product_root / "spectral_fits"
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite spectral fits: {output_root}")
    distribution = config["runtime"]["wsl_distribution"]
    native_temp = Path(f"//wsl.localhost/{distribution}/tmp")
    staging = Path(tempfile.mkdtemp(prefix="sigma_v19cy_spectral_fits_", dir=native_temp))
    primary_band = config["fit_protocol"]["primary_band_keV"]
    variants = [
        {"name": "primary", "band_keV": primary_band},
        *config["fit_protocol"]["robustness_models"],
    ]
    fits: list[dict[str, Any]] = []
    try:
        for region, bundle in bundles.items():
            for variant in variants:
                fits.append(
                    run_fit(
                        config,
                        region,
                        bundle,
                        variant,
                        nxb_expression,
                        nxb_specs,
                        staging,
                    )
                )
        publish = Path(tempfile.mkdtemp(prefix="spectral_fits.installing.", dir=product_root))
        try:
            shutil.copytree(staging, publish, dirs_exist_ok=True)
            os.replace(publish, output_root)
        except Exception:
            shutil.rmtree(publish, ignore_errors=True)
            raise
        shutil.rmtree(staging)
    except Exception as exc:
        failure = {
            "protocol_version": "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-RESULT-1.0.0",
            "status": "response_aware_spectral_fit_failed_closed",
            "generated_utc": datetime.now(UTC).isoformat(),
            "config_sha256": preparation.sha256(CONFIG),
            "component_report_sha256": preparation.sha256(COMPONENT_REPORT),
            "arf_report_sha256": preparation.sha256(ARF_REPORT),
            "error": str(exc),
            "completed_fits": fits,
            "staging_path": str(staging),
            "terminal_gate_passed": False,
            "signed_gas_current_constructed": False,
            "validation_or_holdout_accessed": False,
        }
        REPORT.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    by_region_variant = {(row["region"], row["variant"]): row for row in fits}
    primary = [by_region_variant[(region, "primary")] for region in bundles]
    primary_by_region = {row["region"]: row for row in primary}
    robust_regions: list[str] = []
    robustness: dict[str, Any] = {}
    for region in bundles:
        base = by_region_variant[(region, "primary")]
        checks: dict[str, bool] = {}
        for name in ("narrow_fe_k", "two_temperature_shared_velocity"):
            alternative = by_region_variant[(region, name)]
            checks[name] = alternative["converged"] and (
                abs(alternative["velocity_km_s"] - base["velocity_km_s"])
                <= 150.0
                or intervals_overlap(
                    base["velocity_profile_interval_km_s"],
                    alternative["velocity_profile_interval_km_s"],
                )
            )
        robustness[region] = checks
        if all(checks.values()):
            robust_regions.append(region)
    gate = config["terminal_gate"]
    gates = {
        "response_component_gate_passed": component_report["component_gate_passed"] is True,
        "arf_gate_passed": arf_report["arf_gate_passed"] is True,
        "all_seven_primary_fits_converged": len(primary) == 7 and all(row["converged"] for row in primary),
        "no_free_parameter_at_hard_bound_in_any_fit": all(
            row["no_free_parameter_at_hard_bound"] for row in fits
        ),
        "at_least_five_primary_velocity_halfwidths_at_most_200_km_s": sum(
            row["velocity_interval_halfwidth_km_s"]
            <= gate["maximum_primary_velocity_interval_halfwidth_km_s_for_at_least_five_regions"]
            for row in primary
        )
        >= gate["minimum_regions_meeting_velocity_interval_gate"],
        "at_least_five_regions_pass_both_robustness_models": len(robust_regions)
        >= gate["minimum_robust_regions"],
    }
    report = {
        "protocol_version": "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-RESULT-1.0.0",
        "status": "response_aware_spectral_terminal_gate_passed" if all(gates.values()) else "response_aware_spectral_terminal_gate_failed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": preparation.sha256(CONFIG),
        "component_report_sha256": preparation.sha256(COMPONENT_REPORT),
        "arf_report_sha256": preparation.sha256(ARF_REPORT),
        "nxb_model_sha256": preparation.sha256(nxb_path),
        "fits": fits,
        "robustness": robustness,
        "robust_regions": robust_regions,
        "published_no_ssm_comparison": published_comparison(
            config, primary_by_region
        ),
        "gates": gates,
        "terminal_gate_passed": all(gates.values()),
        "signed_gas_current_constructed": False,
        "missing_000103_exposure_limitation": "P2 uses only 000102000 because 000103000 has no strictly bracketed gain interval.",
        "validation_or_holdout_accessed": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, sort_keys=True))
