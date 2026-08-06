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

import numpy as np
from astropy.io import fits

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
INTERFACE_FAILURE_REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_aware_spectral_interface_failure.json"
)
REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_aware_spectral.json"
)
ARTIFACT_INDEX = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_aware_spectral_artifacts.json"
)
CHECKPOINT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_aware_spectral.checkpoint.json"
)
EXPECTED_PROTOCOL = "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.7"
EXPECTED_COMPONENT_RESULT = "SIGMA-V19CY-A2319-RESPONSE-COMPONENTS-RESULT-1.0.0"
EXPECTED_ARF_RESULT = "SIGMA-V19CY-A2319-ARF-RESULT-1.0.0"
NXB_PARAMETER_COUNT = 56
NXB_THAWED_NORMALIZATIONS = (3, 7, 14, 20, 23, 29, 35, 41, 47, 50, 53, 56)
NXB_GROUPING_TYPE = "optsnmin"
NXB_GROUPING_SCALE = 3.0
LIGHT_SPEED_KM_S = 299_792.458
XSPEC_TIMEOUT_SECONDS = 7_200
MARKER = "__SIGMA_XSPEC__"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_xspec_deck(path: Path, deck: str) -> None:
    """Write a WSL-stdin deck without Windows newline translation."""
    payload = deck.encode("utf-8")
    if b"\r" in payload:
        raise RuntimeError("XSPEC stdin deck contains a carriage return")
    path.write_bytes(payload)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_checkpoint(
    staging: Path,
    nxb_grouping: list[dict[str, Any]],
    nxb_prefits: list[dict[str, Any]],
    fits: list[dict[str, Any]],
    status: str,
) -> None:
    write_json_atomic(
        CHECKPOINT,
        {
            "protocol_version": (
                "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-CHECKPOINT-1.0.0"
            ),
            "status": status,
            "updated_utc": datetime.now(UTC).isoformat(),
            "config_sha256": preparation.sha256(CONFIG),
            "component_report_sha256": preparation.sha256(COMPONENT_REPORT),
            "arf_report_sha256": preparation.sha256(ARF_REPORT),
            "staging_path": str(staging),
            "nxb_grouping": nxb_grouping,
            "nxb_prefits": nxb_prefits,
            "completed_fits": fits,
            "terminal_gate_passed": False,
            "signed_gas_current_constructed": False,
            "validation_or_holdout_accessed": False,
        },
    )


def nxb_grouping_command(
    config: dict[str, Any], original: Path, scratch: Path, grouped: Path, work: Path
) -> str:
    """Build the preregistered weighted-NXB grouping command.

    ``ftgrouppha`` accepts background-class spectra, while ``rslnxbgen`` marks
    its weighted RATE products as DERIVED.  The class is therefore changed
    only on a staging copy for tool compatibility and restored on the grouped
    product.  The immutable response-component input is never edited.
    """
    system_pfiles = config["runtime"]["pfiles"].split(";", 1)[1]
    local_pfiles = xspec_path(config, work / "pfiles")
    diagonal = ROOT / config["nxb_protocol"]["diagonal_response_path"]
    spectrum_scratch = xspec_path(config, scratch) + "[SPECTRUM]"
    spectrum_grouped = xspec_path(config, grouped) + "[SPECTRUM]"
    return (
        application.runtime_environment(config)
        + "mkdir -p "
        + shlex.quote(local_pfiles)
        + "; export PFILES="
        + shlex.quote(local_pfiles + ";" + system_pfiles)
        + "; punlearn ftcopy && ftcopy infile="
        + shlex.quote(xspec_path(config, original))
        + " outfile="
        + shlex.quote(xspec_path(config, scratch))
        + " copyall=yes clobber=no history=yes"
        + " && punlearn fthedit && fthedit infile="
        + shlex.quote(spectrum_scratch)
        + " keyword=HDUCLAS2 operation=add value=BKG"
        + " comment='temporary ftgrouppha compatibility class'"
        + " && punlearn ftgrouppha && ftgrouppha infile="
        + shlex.quote(xspec_path(config, scratch))
        + " outfile="
        + shlex.quote(xspec_path(config, grouped))
        + f" grouptype={NXB_GROUPING_TYPE} groupscale={NXB_GROUPING_SCALE}"
        + " respfile="
        + shlex.quote(xspec_path(config, diagonal))
        + " && punlearn fthedit && fthedit infile="
        + shlex.quote(spectrum_grouped)
        + " keyword=HDUCLAS2 operation=add value=DERIVED"
        + " comment='XRISM weighted NXB derived spectrum'"
        + " && punlearn fthedit && fthedit infile="
        + shlex.quote(spectrum_grouped)
        + " keyword=RESPFILE operation=add value=NONE"
        + " comment='response assigned explicitly by frozen XSPEC deck'"
    )


def summarize_nxb_groups(
    rate: np.ndarray,
    stat_err: np.ndarray,
    grouping: np.ndarray,
    energy_min: np.ndarray,
    energy_max: np.ndarray,
    band_keV: list[float],
) -> dict[str, Any]:
    """Audit the actual grouped constraints that overlap the fit band."""
    arrays = (rate, stat_err, grouping, energy_min, energy_max)
    if len({np.asarray(value).size for value in arrays}) != 1:
        raise RuntimeError("NXB grouping arrays have inconsistent lengths")
    if rate.size == 0:
        raise RuntimeError("NXB grouping arrays are empty")
    if (
        not np.all(np.isfinite(rate))
        or not np.all(np.isfinite(stat_err))
        or not np.all(np.isfinite(energy_min))
        or not np.all(np.isfinite(energy_max))
        or np.any(rate < 0)
        or np.any(stat_err < 0)
        or np.any(energy_max <= energy_min)
    ):
        raise RuntimeError("NXB grouping arrays contain invalid values")
    grouping = np.asarray(grouping, dtype=int)
    if not np.all(np.isin(grouping, (-1, 0, 1))):
        raise RuntimeError("NXB GROUPING contains an invalid flag")

    group_ids = np.empty(grouping.size, dtype=np.int64)
    current = -1
    previous = 1
    for index, flag in enumerate(grouping):
        if flag in (0, 1):
            current += 1
        elif index == 0 or previous == 0:
            raise RuntimeError("NXB GROUPING starts or continues after zero with -1")
        group_ids[index] = current
        previous = int(flag)

    low, high = map(float, band_keV)
    overlaps_band = (energy_max > low) & (energy_min < high)
    if not np.any(overlaps_band):
        raise RuntimeError("NXB grouping has no channels in its constraint band")
    overlapping_groups = np.unique(group_ids[overlaps_band])
    selected_groups: list[int] = []
    for group_id in overlapping_groups:
        selected = group_ids == group_id
        if np.min(energy_min[selected]) >= low and np.max(energy_max[selected]) <= high:
            selected_groups.append(int(group_id))
    if not selected_groups:
        raise RuntimeError("NXB grouping has no complete groups in its constraint band")
    effective_counts: list[float] = []
    widths: list[int] = []
    energy_widths: list[float] = []
    zero_variance = 0
    for group_id in selected_groups:
        selected = group_ids == group_id
        variance = float(np.sum(np.square(stat_err[selected]), dtype=np.float64))
        if not math.isfinite(variance) or variance <= 0:
            zero_variance += 1
            continue
        total_rate = float(np.sum(rate[selected], dtype=np.float64))
        effective_counts.append(total_rate * total_rate / variance)
        widths.append(int(np.count_nonzero(selected)))
        energy_widths.append(
            float(np.max(energy_max[selected]) - np.min(energy_min[selected]))
        )
    if zero_variance:
        raise RuntimeError(
            f"grouped NXB has {zero_variance} zero-variance groups in the fit band"
        )
    minimum_effective_counts = float(min(effective_counts))
    minimum_signal_to_noise = math.sqrt(minimum_effective_counts)
    if minimum_signal_to_noise < NXB_GROUPING_SCALE - 1e-6:
        raise RuntimeError(
            "grouped NXB does not meet the frozen minimum signal-to-noise"
        )
    return {
        "constraint_band_keV": [low, high],
        "groups_in_band": len(selected_groups),
        "boundary_groups_excluded_by_xspec": int(
            overlapping_groups.size - len(selected_groups)
        ),
        "zero_variance_groups_in_band": zero_variance,
        "minimum_effective_counts_in_band": minimum_effective_counts,
        "minimum_signal_to_noise_in_band": minimum_signal_to_noise,
        "median_effective_counts_in_band": float(np.median(effective_counts)),
        "maximum_channels_per_group_in_band": int(max(widths)),
        "maximum_group_width_keV_in_band": float(max(energy_widths)),
    }


def inspect_grouped_nxb(
    original: Path, grouped: Path, diagonal: Path, band_keV: list[float]
) -> dict[str, Any]:
    """Verify grouping changes only grouping metadata and removes zero variance."""
    with fits.open(original, memmap=True, mode="readonly") as hdus:
        hdus.verify("exception")
        source = hdus["SPECTRUM"]
        original_channel = np.asarray(source.data["CHANNEL"], dtype=int).copy()
        original_rate = np.asarray(source.data["RATE"], dtype=float).copy()
        original_error = np.asarray(source.data["STAT_ERR"], dtype=float).copy()
    with fits.open(grouped, memmap=True, mode="readonly") as hdus:
        hdus.verify("exception")
        spectrum = hdus["SPECTRUM"]
        names = set(spectrum.columns.names)
        if "GROUPING" not in names:
            raise RuntimeError("grouped NXB is missing GROUPING")
        header_class = str(spectrum.header.get("HDUCLAS2", "")).strip().upper()
        poiserr = bool(spectrum.header.get("POISSERR", True))
        respfile = str(spectrum.header.get("RESPFILE", "")).strip().upper()
        channel = np.asarray(spectrum.data["CHANNEL"], dtype=int).copy()
        rate = np.asarray(spectrum.data["RATE"], dtype=float).copy()
        stat_err = np.asarray(spectrum.data["STAT_ERR"], dtype=float).copy()
        grouping = np.asarray(spectrum.data["GROUPING"], dtype=int).copy()
    if header_class != "DERIVED" or poiserr or respfile != "NONE":
        raise RuntimeError(
            "grouped NXB did not retain the DERIVED, non-Poisson, explicit-response contract"
        )
    if not (
        np.array_equal(channel, original_channel)
        and np.array_equal(rate, original_rate)
        and np.array_equal(stat_err, original_error)
    ):
        raise RuntimeError("NXB grouping altered channels, rates, or statistical errors")
    with fits.open(diagonal, memmap=True, mode="readonly") as hdus:
        hdus.verify("exception")
        bounds = hdus["EBOUNDS"].data
        bound_channel = np.asarray(bounds["CHANNEL"], dtype=int)
        if not np.array_equal(bound_channel, channel):
            raise RuntimeError("NXB and diagonal-response channel grids differ")
        energy_min = np.asarray(bounds["E_MIN"], dtype=float)
        energy_max = np.asarray(bounds["E_MAX"], dtype=float)
    summary = summarize_nxb_groups(
        rate, stat_err, grouping, energy_min, energy_max, band_keV
    )
    return {
        "original": {
            "path": str(original.relative_to(ROOT)).replace("\\", "/"),
            "bytes": original.stat().st_size,
            "sha256": preparation.sha256(original),
            "channels": int(channel.size),
            "total_rate": float(np.sum(original_rate, dtype=np.float64)),
        },
        "grouped": {
            "bytes": grouped.stat().st_size,
            "sha256": preparation.sha256(grouped),
            "channels": int(channel.size),
            "total_rate": float(np.sum(rate, dtype=np.float64)),
            "hduclas2": header_class,
            "poiserr": poiserr,
            "respfile": respfile,
            "grouping_type": NXB_GROUPING_TYPE,
            "grouping_scale": NXB_GROUPING_SCALE,
        },
        "rate_and_stat_err_preserved_exactly": True,
        **summary,
    }


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


def nxb_prefit_model_lines(
    base_specs: list[str], source_group_count: int
) -> list[str]:
    """Build independent delivered-state NXB models for NXB-only groups."""
    if len(base_specs) != NXB_PARAMETER_COUNT or source_group_count not in (1, 2):
        raise ValueError("invalid NXB prefit model layout")
    lines: list[str] = []
    for source_index in range(source_group_count):
        offset = source_index * NXB_PARAMETER_COUNT
        lines.extend(shift_nxb_link(spec, offset) for spec in base_specs)
    return lines


def nxb_specs_after_prefit(
    base_specs: list[str],
    source_group_count: int,
    fitted_values: dict[str, float],
) -> list[str]:
    """Transfer NXB-only best fits into the joint fit with frozen shape."""
    if len(base_specs) != NXB_PARAMETER_COUNT or source_group_count not in (1, 2):
        raise ValueError("invalid NXB prefit transfer layout")
    lines: list[str] = []
    for source_index in range(source_group_count):
        offset = source_index * NXB_PARAMETER_COUNT
        for local_index, spec in enumerate(base_specs, start=1):
            global_index = offset + local_index
            if spec.startswith("="):
                lines.append(shift_nxb_link(spec, offset))
                continue
            if str(global_index) not in fitted_values:
                raise RuntimeError(
                    f"missing fitted NXB parameter {global_index} during transfer"
                )
            fields = spec.split()
            fields[0] = f"{fitted_values[str(global_index)]:.17g}"
            fields[1] = (
                fields[1]
                if local_index in NXB_THAWED_NORMALIZATIONS
                else "-1"
            )
            lines.append(" ".join(fields))
    return lines


def nxb_joint_model_lines(
    prefit_specs: list[str], source_group_count: int
) -> list[str]:
    """Apply transferred NXB shapes to source and NXB constraint groups."""
    expected = source_group_count * NXB_PARAMETER_COUNT
    if len(prefit_specs) != expected or source_group_count not in (1, 2):
        raise ValueError("invalid transferred NXB model layout")
    lines = list(prefit_specs)
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
    base_specs: list[str],
    source_group_count: int,
    policy: str = "normalizations_only",
) -> list[int]:
    if len(base_specs) != NXB_PARAMETER_COUNT or source_group_count not in (1, 2):
        raise ValueError("invalid NXB model layout")
    if policy == "normalizations_only":
        local_free = set(NXB_THAWED_NORMALIZATIONS)
    elif policy == "official_preserve_delivered_shapes":
        delivered_free = {
            local_index
            for local_index, spec in enumerate(base_specs, start=1)
            if not spec.startswith("=") and numeric_parameter_delta(spec) > 0
        }
        # The official second-stage recipe freezes the common scale (p1), thaws
        # the twelve listed normalizations, and explicitly leaves the delivered
        # photon-index/Au-width shape parameters free.
        local_free = (delivered_free - {1}) | set(NXB_THAWED_NORMALIZATIONS)
    else:
        raise ValueError(f"unknown NXB second-stage free policy: {policy}")
    free: list[int] = []
    for source_index in range(source_group_count):
        offset = source_index * NXB_PARAMETER_COUNT
        free.extend(offset + local_index for local_index in sorted(local_free))
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
    for index in range(1, 2 * source_group_count + 1):
        commands.extend(
            [
                f"tclout stat {index}",
                f'puts "{MARKER} statistic_spectrum_{index} $xspec_tclout"',
            ]
        )
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


def nxb_prefit_marker_commands(
    source_group_count: int, numeric_indices: list[int]
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
    for index in range(1, source_group_count + 1):
        commands.extend(
            [
                f"tclout stat {index}",
                f'puts "{MARKER} statistic_spectrum_{index} $xspec_tclout"',
            ]
        )
    for index in numeric_indices:
        commands.extend(
            [
                f"tclout param nxb1:{index}",
                f'puts "{MARKER} nxb_parameter_{index} $xspec_tclout"',
            ]
        )
    return commands


def xspec_path(config: dict[str, Any], path: Path) -> str:
    return components.tool_path(config, path)


def build_nxb_prefit_deck(
    config: dict[str, Any],
    bundle: list[dict[str, Path]],
    *,
    nxb_expression: str,
    nxb_specs: list[str],
    log_path: Path,
    session_path: Path,
) -> tuple[str, dict[str, Any]]:
    """Build an NXB-only session so no source spectrum must be hidden."""
    source_group_count = len(bundle)
    if source_group_count not in (1, 2):
        raise ValueError("A2319 regions require one or two NXB branches")
    data_parts = [
        f"{index}:{index} {xspec_path(config, row['nxb_pha'])}"
        for index, row in enumerate(bundle, start=1)
    ]
    diagonal = xspec_path(
        config, ROOT / config["nxb_protocol"]["diagonal_response_path"]
    )
    commands = [
        "query yes",
        "chatter 10",
        f"log {xspec_path(config, log_path)}",
        "data none",
        "data " + " ".join(data_parts),
    ]
    for index in range(1, source_group_count + 1):
        commands.append(f"response 1:{index} {diagonal}")
    nxb_range = f"1-{source_group_count}"
    commands.extend(
        [
            f"statistic chi standard {nxb_range}",
            "method leven 1000 0.0001",
            "model 1:nxb1 " + nxb_expression,
            *nxb_prefit_model_lines(nxb_specs, source_group_count),
        ]
    )
    nxb_band = config["fit_protocol"]["nxb_constraint_band_keV"]
    for index in range(1, source_group_count + 1):
        commands.append(f"ignore {index}:**-{nxb_band[0]} {nxb_band[1]}-**")
    commands.append("fit")
    numeric_indices = nxb_numeric_parameter_indices(nxb_specs, source_group_count)
    free_indices = nxb_free_parameter_indices(
        nxb_specs,
        source_group_count,
        config["nxb_protocol"].get(
            "second_stage_free_policy", "normalizations_only"
        ),
    )
    commands.extend(f"freeze nxb1:{index}" for index in numeric_indices)
    commands.extend(f"thaw nxb1:{index}" for index in free_indices)
    commands.append("fit")
    commands.extend(nxb_prefit_marker_commands(source_group_count, numeric_indices))
    commands.extend(
        [
            f"save all {xspec_path(config, session_path)}",
            "log none",
            "exit",
        ]
    )
    metadata = {
        "source_group_count": source_group_count,
        "nxb_numeric_parameter_indices": numeric_indices,
        "nxb_free_parameter_indices": free_indices,
        "nxb_statistic": "chi standard",
        "nxb_constraint_band_keV": nxb_band,
        "source_spectra_loaded": False,
    }
    return "\n".join(commands) + "\n", metadata


def build_xspec_deck(
    config: dict[str, Any],
    bundle: list[dict[str, Path]],
    *,
    variant: dict[str, Any],
    nxb_expression: str,
    nxb_specs: list[str],
    nxb_prefit_values: dict[str, float],
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
            *nxb_joint_model_lines(
                nxb_specs_after_prefit(
                    nxb_specs, source_group_count, nxb_prefit_values
                ),
                source_group_count,
            ),
        ]
    )
    nxb_band = config["fit_protocol"]["nxb_constraint_band_keV"]
    for index in range(source_group_count + 1, 2 * source_group_count + 1):
        commands.append(f"ignore {index}:**-{nxb_band[0]} {nxb_band[1]}-**")
    commands.extend(
        f"freeze nxb1:{index}"
        for index in nxb_numeric_parameter_indices(nxb_specs, source_group_count)
    )
    commands.extend(
        f"thaw nxb1:{index}"
        for index in nxb_free_parameter_indices(nxb_specs, source_group_count)
    )
    source_band = variant["band_keV"]
    for index in range(1, source_group_count + 1):
        commands.append(f"ignore {index}:**-{source_band[0]} {source_band[1]}-**")
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
    required = {
        "statistic",
        "dof",
        "covariance",
        "variable_parameters",
        "redshift",
        "redshift_error",
        "redshift_sigma",
        *(
            f"statistic_spectrum_{index}"
            for index in range(1, 2 * metadata["source_group_count"] + 1)
        ),
    }
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
    statistic_by_spectrum = {
        str(index): first_float(markers[f"statistic_spectrum_{index}"])
        for index in range(1, 2 * metadata["source_group_count"] + 1)
    }
    contribution_sum = float(sum(statistic_by_spectrum.values()))
    if not math.isclose(statistic, contribution_sum, rel_tol=1e-5, abs_tol=1e-3):
        raise RuntimeError(
            "per-spectrum XSPEC statistic contributions do not sum to the total"
        )
    dof = round(first_float(markers["dof"]))
    return {
        "statistic": statistic,
        "statistic_by_spectrum": statistic_by_spectrum,
        "source_cstat_contribution": float(
            sum(
                statistic_by_spectrum[str(index)]
                for index in range(1, metadata["source_group_count"] + 1)
            )
        ),
        "nxb_chi_square_contribution": float(
            sum(
                statistic_by_spectrum[str(index)]
                for index in range(
                    metadata["source_group_count"] + 1,
                    2 * metadata["source_group_count"] + 1,
                )
            )
        ),
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
    grouping_amendment = config.get("pre_fit_grouping_amendment", {})
    if grouping_amendment.get("previous_version") != (
        "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.4"
    ):
        raise RuntimeError("unexpected pre-fit grouping amendment ancestry")
    grouping = grouping_amendment.get("nxb_grouping", {})
    if (
        grouping.get("tool") != "ftgrouppha"
        or grouping.get("grouptype") != NXB_GROUPING_TYPE
        or grouping.get("groupscale") != NXB_GROUPING_SCALE
        or grouping.get("grid_channels") != 60000
        or grouping.get("constraint_band_keV") != [1.0, 17.0]
        or grouping_amendment.get("source_grouping") != "none"
    ):
        raise RuntimeError("pre-fit NXB grouping rule is not the frozen contract")
    if any(
        grouping_amendment.get(key)
        for key in (
            "source_model_or_energy_band_changed",
            "gravity_formula_or_parameters_changed",
            "source_energy_distribution_summarized_or_fit",
            "velocity_fit_performed",
            "validation_or_holdout_accessed",
        )
    ) or grouping_amendment.get("arf_generated") is not True:
        raise RuntimeError("pre-fit grouping amendment crossed a sealed boundary")
    execution_amendment = config.get("pre_fit_execution_amendment", {})
    if execution_amendment.get("previous_version") != (
        "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.5"
    ):
        raise RuntimeError("unexpected pre-fit execution amendment ancestry")
    if preparation.sha256(INTERFACE_FAILURE_REPORT) != execution_amendment.get(
        "archived_failure_report", {}
    ).get("sha256"):
        raise RuntimeError("archived XSPEC interface failure report changed")
    for key in ("checkpoint", "failed_deck", "failed_xspec_log"):
        artifact = execution_amendment.get(key, {})
        path = ROOT / artifact.get("path", "missing")
        if (
            not path.is_file()
            or path.stat().st_size != artifact.get("bytes")
            or preparation.sha256(path) != artifact.get("sha256")
        ):
            raise RuntimeError(f"archived XSPEC interface artifact changed: {key}")
    if (
        execution_amendment.get("valid_model_optimization_or_velocity_fit_completed")
        is not False
        or execution_amendment.get("source_result_used_to_change_model_band_or_parameters")
        is not False
        or execution_amendment.get("nxb_grouping_or_response_changed") is not False
        or execution_amendment.get("gravity_formula_or_parameters_changed") is not False
        or execution_amendment.get("validation_or_holdout_accessed") is not False
    ):
        raise RuntimeError("pre-fit execution amendment crossed a frozen boundary")
    nxb_session_amendment = config.get("pre_fit_nxb_session_amendment", {})
    if nxb_session_amendment.get("previous_version") != (
        "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.6"
    ):
        raise RuntimeError("unexpected NXB-session amendment ancestry")
    for key in (
        "failure_report",
        "lf_deck",
        "lf_xspec_log",
        "grouping_checkpoint_after_lf_retry",
    ):
        artifact = nxb_session_amendment.get(key, {})
        path = ROOT / artifact.get("path", "missing")
        if (
            not path.is_file()
            or path.stat().st_size != artifact.get("bytes")
            or preparation.sha256(path) != artifact.get("sha256")
        ):
            raise RuntimeError(f"archived NXB-session failure artifact changed: {key}")
    for key in ("probe_before_ignore", "probe_ignore_first_source"):
        probe = nxb_session_amendment.get(key, {})
        for role in ("deck", "log"):
            path = ROOT / probe.get(f"{role}_path", "missing")
            if (
                not path.is_file()
                or path.stat().st_size != probe.get(f"{role}_bytes")
                or preparation.sha256(path) != probe.get(f"{role}_sha256")
            ):
                raise RuntimeError(
                    f"archived NXB-session diagnostic changed: {key}/{role}"
                )
    if (
        nxb_session_amendment.get("probe_before_ignore", {}).get("exit_code") != 0
        or nxb_session_amendment.get("probe_ignore_first_source", {}).get(
            "exit_code"
        )
        != 139
        or nxb_session_amendment.get(
            "valid_model_optimization_or_velocity_fit_completed"
        )
        is not False
        or nxb_session_amendment.get("source_model_or_energy_band_changed")
        is not False
        or nxb_session_amendment.get(
            "nxb_model_grouping_band_or_response_changed"
        )
        is not False
        or nxb_session_amendment.get("gravity_formula_or_parameters_changed")
        is not False
        or nxb_session_amendment.get("validation_or_holdout_accessed") is not False
    ):
        raise RuntimeError("pre-fit NXB-session amendment crossed a frozen boundary")
    authorization = config["authorization"]
    if not authorization["fit_A2319_development_spectra_and_velocities"]:
        raise RuntimeError("A2319 spectral fitting is not authorized")
    if authorization["access_A3667_validation"] or authorization["access_A754_holdout"]:
        raise RuntimeError("sealed validation or holdout access is enabled")
    if authorization["open_lensing_halo_or_gravity_targets"]:
        raise RuntimeError("lensing or gravity targets are not sealed")
    components_report = load_json(COMPONENT_REPORT)
    arf_report = load_json(ARF_REPORT)
    if components_report.get("protocol_version") != EXPECTED_COMPONENT_RESULT:
        raise RuntimeError("unexpected response-component result protocol")
    if arf_report.get("protocol_version") != EXPECTED_ARF_RESULT:
        raise RuntimeError("unexpected ARF result protocol")
    if not components_report.get("component_gate_passed"):
        raise RuntimeError("response-component gate did not pass")
    if not arf_report.get("arf_gate_passed"):
        raise RuntimeError("ARF gate did not pass")
    for label, report in (
        ("response-component", components_report),
        ("ARF", arf_report),
    ):
        if any(
            report.get(key)
            for key in (
                "xrism_energy_distribution_summarized_or_fit",
                "velocity_fit_performed",
                "validation_or_holdout_accessed",
            )
        ):
            raise RuntimeError(f"{label} report crossed a sealed pre-fit boundary")
    if len(components_report.get("branches", [])) != 3 or sum(
        len(row.get("regions", [])) for row in components_report.get("branches", [])
    ) != 10:
        raise RuntimeError(
            "response-component report does not contain three branches and ten regions"
        )
    if not components_report.get("commands") or any(
        row.get("exit_code") != 0 for row in components_report["commands"]
    ):
        raise RuntimeError(
            "response-component report contains a failed or missing mission-tool command"
        )
    if len(arf_report.get("branches", [])) != 3 or sum(
        len(row.get("regions", [])) for row in arf_report.get("branches", [])
    ) != 10:
        raise RuntimeError("ARF report does not contain three branches and ten regions")
    if not all(
        row.get("one_raytrace_reused_within_branch") is True
        for row in arf_report["branches"]
    ):
        raise RuntimeError("ARF report did not preserve branch-scoped raytrace reuse")
    if not arf_report.get("commands") or any(
        row.get("exit_code") != 0 for row in arf_report["commands"]
    ):
        raise RuntimeError("ARF report contains a failed or missing mission-tool command")
    statistical_amendment = config.get("pre_fit_statistical_amendment", {})
    if components_report.get("config_sha256") != statistical_amendment.get(
        "component_generation_config_sha256"
    ):
        raise RuntimeError("response-component report belongs to a different protocol")
    if preparation.sha256(COMPONENT_REPORT) != statistical_amendment.get(
        "component_report_sha256"
    ):
        raise RuntimeError("response-component report hash does not match its amendment")
    if arf_report.get("component_report_sha256") != preparation.sha256(COMPONENT_REPORT):
        raise RuntimeError("ARF report belongs to different response components")
    if arf_report.get("config_sha256") != grouping_amendment.get(
        "arf_generation_config_sha256"
    ):
        raise RuntimeError("ARF report belongs to a different generation protocol")
    if preparation.sha256(ARF_REPORT) != grouping_amendment.get("arf_report_sha256"):
        raise RuntimeError("ARF report hash does not match its amendment")
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


def prepare_grouped_nxb_spectra(
    config: dict[str, Any],
    bundles: dict[str, list[dict[str, Path]]],
    staging: Path,
    reports: list[dict[str, Any]],
) -> dict[str, list[dict[str, Path]]]:
    """Create one audited optimal-S/N NXB product per branch-region pair."""
    grouped_bundles: dict[str, list[dict[str, Path]]] = {}
    diagonal = ROOT / config["nxb_protocol"]["diagonal_response_path"]
    band = config["fit_protocol"]["nxb_constraint_band_keV"]
    for region, bundle in bundles.items():
        grouped_rows: list[dict[str, Path]] = []
        for row in bundle:
            original = row["nxb_pha"]
            branch = original.parent.parent.name
            work = staging / "grouped_nxb" / branch / region
            work.mkdir(parents=True)
            scratch = work / "nxb_input_class_bkg.pha"
            grouped = work / "nxb_optsnmin3.pha"
            command = nxb_grouping_command(config, original, scratch, grouped, work)
            record = application.run_wsl(
                config["runtime"]["wsl_distribution"], command, timeout=600
            )
            if record["exit_code"] != 0:
                raise RuntimeError(
                    f"NXB grouping failed for {branch}/{region}: {record['stderr']}"
                )
            audit = inspect_grouped_nxb(original, grouped, diagonal, band)
            audit["grouped"]["path"] = str(
                Path("data/processed/sigma_v19cy_a2319_response_aware_spectral")
                / "spectral_fits"
                / grouped.relative_to(staging)
            ).replace("\\", "/")
            audit["branch"] = branch
            audit["region"] = region
            audit["command"] = record
            reports.append(audit)
            scratch.unlink()
            grouped_row = dict(row)
            grouped_row["nxb_pha"] = grouped
            grouped_rows.append(grouped_row)
        grouped_bundles[region] = grouped_rows
    if len(reports) != 10:
        raise RuntimeError(f"expected ten grouped NXB products, found {len(reports)}")
    return grouped_bundles


def nxb_grouping_gate_passed(reports: list[dict[str, Any]]) -> bool:
    return len(reports) == 10 and all(
        row.get("rate_and_stat_err_preserved_exactly") is True
        and row.get("zero_variance_groups_in_band") == 0
        and float(row.get("minimum_signal_to_noise_in_band", -math.inf))
        >= NXB_GROUPING_SCALE - 1e-6
        and row.get("grouped", {}).get("hduclas2") == "DERIVED"
        and row.get("grouped", {}).get("poiserr") is False
        and row.get("grouped", {}).get("respfile") == "NONE"
        and row.get("grouped", {}).get("grouping_type") == NXB_GROUPING_TYPE
        and row.get("grouped", {}).get("grouping_scale") == NXB_GROUPING_SCALE
        for row in reports
    )


def resume_grouped_nxb_bundles(
    bundles: dict[str, list[dict[str, Path]]],
    staging: Path,
    reports: list[dict[str, Any]],
) -> dict[str, list[dict[str, Path]]]:
    """Reattach only the exact grouped products recorded by a frozen checkpoint."""
    if not nxb_grouping_gate_passed(reports):
        raise RuntimeError("checkpoint NXB grouping gate did not pass")
    index = {(row["branch"], row["region"]): row for row in reports}
    if len(index) != 10:
        raise RuntimeError("checkpoint NXB grouping identities are not unique")
    resumed: dict[str, list[dict[str, Path]]] = {}
    used: set[tuple[str, str]] = set()
    for region, rows in bundles.items():
        resumed_rows: list[dict[str, Path]] = []
        for row in rows:
            branch = row["nxb_pha"].parent.parent.name
            key = (branch, region)
            report = index.get(key)
            if report is None:
                raise RuntimeError(f"checkpoint lacks grouped NXB product: {key}")
            grouped = staging / "grouped_nxb" / branch / region / "nxb_optsnmin3.pha"
            if (
                not grouped.is_file()
                or preparation.sha256(grouped) != report["grouped"]["sha256"]
            ):
                raise RuntimeError(f"checkpoint grouped NXB product changed: {grouped}")
            resumed_row = dict(row)
            resumed_row["nxb_pha"] = grouped
            resumed_rows.append(resumed_row)
            used.add(key)
        resumed[region] = resumed_rows
    if used != set(index):
        raise RuntimeError("checkpoint contains unused grouped NXB products")
    return resumed


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


def inspect_nxb_prefit(
    markers: dict[str, str], metadata: dict[str, Any], nxb_specs: list[str]
) -> dict[str, Any]:
    required = {"statistic", "dof", "covariance", "variable_parameters"}
    required.update(
        f"statistic_spectrum_{index}"
        for index in range(1, metadata["source_group_count"] + 1)
    )
    required.update(
        f"nxb_parameter_{index}"
        for index in metadata["nxb_numeric_parameter_indices"]
    )
    missing = sorted(required - markers.keys())
    if missing:
        raise RuntimeError(f"NXB prefit markers missing: {missing}")
    fitted_values: dict[str, float] = {}
    hard_bound_hits: list[int] = []
    for global_index in metadata["nxb_numeric_parameter_indices"]:
        value = first_float(markers[f"nxb_parameter_{global_index}"])
        fitted_values[str(global_index)] = value
        local_index = (global_index - 1) % NXB_PARAMETER_COUNT
        bounds = numeric_parameter_bounds(nxb_specs[local_index])
        if bounds is not None and parameter_at_bound(value, bounds):
            hard_bound_hits.append(global_index)
    statistic = first_float(markers["statistic"])
    dof = round(first_float(markers["dof"]))
    return {
        "statistic": statistic,
        "dof": dof,
        "statistic_by_spectrum": {
            str(index): first_float(markers[f"statistic_spectrum_{index}"])
            for index in range(1, metadata["source_group_count"] + 1)
        },
        "fitted_numeric_parameters": fitted_values,
        "hard_bound_hits": hard_bound_hits,
        "covariance_lower_triangle": markers["covariance"],
        "variable_parameters": markers["variable_parameters"],
        "converged": math.isfinite(statistic) and dof > 0,
    }


def run_nxb_prefit(
    config: dict[str, Any],
    region: str,
    bundle: list[dict[str, Path]],
    nxb_expression: str,
    nxb_specs: list[str],
    staging: Path,
) -> dict[str, Any]:
    work = staging / region / "nxb_prefit"
    work.mkdir(parents=True)
    deck = work / "prefit.xcm"
    log = work / "xspec.log"
    session = work / "best_prefit.xcm"
    deck_text, metadata = build_nxb_prefit_deck(
        config,
        bundle,
        nxb_expression=nxb_expression,
        nxb_specs=nxb_specs,
        log_path=log,
        session_path=session,
    )
    write_xspec_deck(deck, deck_text)
    command = xspec_command(config, work, deck)
    record = application.run_wsl(
        config["runtime"]["wsl_distribution"], command, timeout=XSPEC_TIMEOUT_SECONDS
    )
    if record["exit_code"] != 0:
        raise RuntimeError(f"XSPEC NXB prefit failed for {region}: {record['stderr']}")
    prefit = inspect_nxb_prefit(parse_markers(record["stdout"]), metadata, nxb_specs)
    if not prefit["converged"]:
        raise RuntimeError(f"XSPEC NXB prefit did not converge for {region}")
    if not log.is_file() or not session.is_file():
        raise RuntimeError(f"XSPEC did not write NXB prefit products for {region}")
    prefit.update(
        {
            "region": region,
            "metadata": metadata,
            "command": record,
            "deck": {
                "bytes": deck.stat().st_size,
                "sha256": preparation.sha256(deck),
            },
            "log": {
                "bytes": log.stat().st_size,
                "sha256": preparation.sha256(log),
            },
            "session": {
                "bytes": session.stat().st_size,
                "sha256": preparation.sha256(session),
            },
            "source_spectra_loaded": False,
            "source_energy_distribution_used": False,
        }
    )
    return prefit


def run_fit(
    config: dict[str, Any],
    region: str,
    bundle: list[dict[str, Path]],
    variant: dict[str, Any],
    nxb_expression: str,
    nxb_specs: list[str],
    nxb_prefit_values: dict[str, float],
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
        nxb_prefit_values=nxb_prefit_values,
        log_path=log,
        session_path=session,
    )
    write_xspec_deck(deck, deck_text)
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

    def report_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            return str(
                Path("data/processed/sigma_v19cy_a2319_response_aware_spectral")
                / "spectral_fits"
                / path.relative_to(staging)
            ).replace("\\", "/")

    fit["inputs"] = [
        {
            role: {
                "path": report_path(path),
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
    per_region: dict[str, Any] = {}
    published = config["published_no_ssm_benchmark"]["regions"]
    for region, benchmark in published.items():
        fit = primary_by_region[region]
        difference = fit["velocity_km_s"] - benchmark["velocity_km_s"]
        benchmark_sigma = (
            benchmark["plus_1sigma"] if difference >= 0 else benchmark["minus_1sigma"]
        )
        fit_sigma = fit["velocity_interval_halfwidth_km_s"]
        published_sigma = 0.5 * (
            benchmark["minus_1sigma"] + benchmark["plus_1sigma"]
        )
        per_region[region] = {
            "fitted_velocity_km_s": fit["velocity_km_s"],
            "published_velocity_km_s": benchmark["velocity_km_s"],
            "difference_km_s": difference,
            "difference_over_published_directional_1sigma": difference / benchmark_sigma,
            "fitted_profile_halfwidth_km_s": fit_sigma,
            "published_mean_1sigma_km_s": published_sigma,
            "combined_1sigma_km_s": math.hypot(fit_sigma, published_sigma),
            "diagnostic_only": True,
        }
    regions = list(published)
    fitted = np.asarray(
        [per_region[region]["fitted_velocity_km_s"] for region in regions],
        dtype=np.float64,
    )
    expected = np.asarray(
        [per_region[region]["published_velocity_km_s"] for region in regions],
        dtype=np.float64,
    )
    differences = fitted - expected
    combined_sigma = np.asarray(
        [per_region[region]["combined_1sigma_km_s"] for region in regions],
        dtype=np.float64,
    )
    weights = 1.0 / np.square(combined_sigma)

    def ordinal_rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(values.size, dtype=np.float64)
        ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
        return ranks

    fitted_ranks = ordinal_rank(fitted)
    expected_ranks = ordinal_rank(expected)
    pair_agreements = [
        np.sign(fitted[left] - fitted[right])
        == np.sign(expected[left] - expected[right])
        for left in range(fitted.size)
        for right in range(left + 1, fitted.size)
    ]
    aggregate = {
        "regions": regions,
        "unweighted_rms_difference_km_s": float(
            np.sqrt(np.mean(np.square(differences)))
        ),
        "inverse_combined_variance_weighted_rms_difference_km_s": float(
            np.sqrt(np.sum(weights * np.square(differences)) / np.sum(weights))
        ),
        "mean_difference_km_s": float(np.mean(differences)),
        "mean_absolute_difference_km_s": float(np.mean(np.abs(differences))),
        "pearson_velocity_correlation": float(np.corrcoef(fitted, expected)[0, 1]),
        "spearman_velocity_rank_correlation": float(
            np.corrcoef(fitted_ranks, expected_ranks)[0, 1]
        ),
        "pairwise_velocity_rank_agreement_fraction": float(np.mean(pair_agreements)),
        "sign_agreement_fraction": float(
            np.mean(np.sign(fitted) == np.sign(expected))
        ),
        "directional_published_1sigma_agreement_fraction": float(
            np.mean(
                [
                    abs(per_region[region]["difference_over_published_directional_1sigma"])
                    <= 1.0
                    for region in regions
                ]
            )
        ),
        "diagnostic_only": True,
    }
    return {"per_region": per_region, "aggregate": aggregate}


def finalize_existing_report_diagnostics() -> dict[str, Any]:
    """Add deterministic post-fit diagnostics without rerunning XSPEC."""
    config = load_json(CONFIG)
    report = load_json(REPORT)
    if report.get("config_sha256") != preparation.sha256(CONFIG):
        raise RuntimeError("terminal report does not match the current frozen config")
    fits = report.get("fits", [])
    primary = [row for row in fits if row.get("variant") == "primary"]
    if len(primary) != 7:
        raise RuntimeError("terminal report does not contain seven primary fits")
    primary_by_region = {row["region"]: row for row in primary}
    report["published_no_ssm_comparison"] = published_comparison(
        config, primary_by_region
    )
    report["post_fit_diagnostics_finalized_utc"] = datetime.now(UTC).isoformat()
    report["post_fit_diagnostics_only"] = True
    write_json_atomic(REPORT, report)
    return report


def index_existing_spectral_artifacts() -> dict[str, Any]:
    """Hash every installed fit artifact after the terminal run."""
    config = load_json(CONFIG)
    output_root = (ROOT / config["paths"]["product_root"] / "spectral_fits").resolve()
    if not output_root.is_dir():
        raise RuntimeError("installed spectral-fit artifact directory is missing")
    artifacts = []
    for path in sorted(candidate for candidate in output_root.rglob("*") if candidate.is_file()):
        artifacts.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": preparation.sha256(path),
            }
        )
    index = {
        "protocol_version": (
            "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-ARTIFACTS-1.0.0"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": preparation.sha256(CONFIG),
        "terminal_report_sha256": preparation.sha256(REPORT),
        "artifact_count": len(artifacts),
        "total_bytes": sum(row["bytes"] for row in artifacts),
        "artifacts": artifacts,
        "validation_or_holdout_accessed": False,
    }
    write_json_atomic(ARTIFACT_INDEX, index)
    return index


def generate() -> dict[str, Any]:
    config, component_report, arf_report = validate_inputs()
    raw_bundles = assemble_bundles(config, component_report, arf_report)
    nxb_path = ROOT / config["nxb_protocol"]["empirical_model_path"]
    nxb_expression, nxb_specs = parse_nxb_model(nxb_path.read_text(encoding="utf-8"))
    product_root = (ROOT / config["paths"]["product_root"]).resolve()
    output_root = product_root / "spectral_fits"
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite spectral fits: {output_root}")
    distribution = config["runtime"]["wsl_distribution"]
    native_temp = Path(f"//wsl.localhost/{distribution}/tmp")
    primary_band = config["fit_protocol"]["primary_band_keV"]
    variants = [
        {"name": "primary", "band_keV": primary_band},
        *config["fit_protocol"]["robustness_models"],
    ]
    fits: list[dict[str, Any]] = []
    nxb_prefits: list[dict[str, Any]] = []
    nxb_grouping: list[dict[str, Any]] = []
    resume_from_grouping = False
    if CHECKPOINT.exists():
        checkpoint = load_json(CHECKPOINT)
        amendment = config["pre_fit_execution_amendment"]
        checkpoint_contract = amendment["checkpoint"]
        if (
            preparation.sha256(CHECKPOINT) != checkpoint_contract["sha256"]
            or CHECKPOINT.stat().st_size != checkpoint_contract["bytes"]
            or checkpoint.get("protocol_version")
            != "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-CHECKPOINT-1.0.0"
            or checkpoint.get("status") != checkpoint_contract["status"]
            or checkpoint.get("config_sha256") != amendment["failed_run_config_sha256"]
            or checkpoint.get("component_report_sha256")
            != preparation.sha256(COMPONENT_REPORT)
            or checkpoint.get("arf_report_sha256") != preparation.sha256(ARF_REPORT)
            or checkpoint.get("completed_fits") != []
            or checkpoint.get("nxb_prefits", []) != []
            or checkpoint.get("validation_or_holdout_accessed") is not False
        ):
            raise RuntimeError("incomplete spectral checkpoint is not the frozen resume point")
        staging = Path(checkpoint["staging_path"])
        if not staging.is_dir():
            raise RuntimeError("frozen resume staging directory is unavailable")
        nxb_grouping = checkpoint["nxb_grouping"]
        resume_from_grouping = True
    else:
        staging = Path(
            tempfile.mkdtemp(prefix="sigma_v19cy_spectral_fits_", dir=native_temp)
        )
    try:
        if resume_from_grouping:
            bundles = resume_grouped_nxb_bundles(
                raw_bundles, staging, nxb_grouping
            )
            write_checkpoint(
                staging,
                nxb_grouping,
                nxb_prefits,
                fits,
                "nxb_grouping_resumed_after_nxb_prefit_session_fix",
            )
        else:
            bundles = prepare_grouped_nxb_spectra(
                config, raw_bundles, staging, nxb_grouping
            )
            write_checkpoint(
                staging, nxb_grouping, nxb_prefits, fits, "nxb_grouping_completed"
            )
        for region, bundle in bundles.items():
            prefit = run_nxb_prefit(
                config,
                region,
                bundle,
                nxb_expression,
                nxb_specs,
                staging,
            )
            nxb_prefits.append(prefit)
            write_checkpoint(
                staging,
                nxb_grouping,
                nxb_prefits,
                fits,
                f"nxb_prefit_completed:{region}",
            )
            for variant in variants:
                fit = run_fit(
                    config,
                    region,
                    bundle,
                    variant,
                    nxb_expression,
                    nxb_specs,
                    prefit["fitted_numeric_parameters"],
                    staging,
                )
                fits.append(fit)
                write_checkpoint(
                    staging,
                    nxb_grouping,
                    nxb_prefits,
                    fits,
                    f"fit_completed:{region}:{variant['name']}",
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
            "protocol_version": "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-RESULT-1.0.1",
            "status": "response_aware_spectral_fit_failed_closed",
            "generated_utc": datetime.now(UTC).isoformat(),
            "config_sha256": preparation.sha256(CONFIG),
            "component_report_sha256": preparation.sha256(COMPONENT_REPORT),
            "arf_report_sha256": preparation.sha256(ARF_REPORT),
            "nxb_grouping": nxb_grouping,
            "nxb_prefits": nxb_prefits,
            "error": str(exc),
            "completed_fits": fits,
            "staging_path": str(staging),
            "terminal_gate_passed": False,
            "signed_gas_current_constructed": False,
            "validation_or_holdout_accessed": False,
        }
        write_json_atomic(REPORT, failure)
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
        "all_ten_nxb_grouping_gates_passed": nxb_grouping_gate_passed(nxb_grouping),
        "all_seven_nxb_only_prefits_converged": len(nxb_prefits) == 7
        and all(row["converged"] for row in nxb_prefits),
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
        "protocol_version": "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-RESULT-1.0.1",
        "status": "response_aware_spectral_terminal_gate_passed" if all(gates.values()) else "response_aware_spectral_terminal_gate_failed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": preparation.sha256(CONFIG),
        "component_report_sha256": preparation.sha256(COMPONENT_REPORT),
        "arf_report_sha256": preparation.sha256(ARF_REPORT),
        "nxb_model_sha256": preparation.sha256(nxb_path),
        "nxb_grouping": nxb_grouping,
        "nxb_prefits": nxb_prefits,
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
    write_json_atomic(REPORT, report)
    CHECKPOINT.unlink()
    return report


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, sort_keys=True))
