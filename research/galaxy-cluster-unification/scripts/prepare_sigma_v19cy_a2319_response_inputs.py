#!/usr/bin/env python3
"""Prepare exact-GTI A2319 development events and detector regions."""

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


DEFAULT_CONFIG = ROOT / "configs/sigma_v19cy_a2319_response_aware_spectral.json"
EXPECTED_PROTOCOL = "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.1"
BLOCK_BYTES = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("protocol_version") != EXPECTED_PROTOCOL:
        raise RuntimeError("unexpected response-aware spectral protocol")
    if "corrected and refrozen" not in config.get("status", ""):
        raise RuntimeError("response-aware spectral protocol is not frozen")
    for parent in config["parents"].values():
        path = ROOT / parent["path"]
        if not path.is_file() or sha256(path) != parent["sha256"]:
            raise RuntimeError(f"frozen parent changed: {path}")
    for branch in config["branches"]:
        path = ROOT / branch["event_path"]
        if not path.is_file() or sha256(path) != branch["event_sha256"]:
            raise RuntimeError(f"frozen calibrated branch changed: {path}")
    for support in config["observation_support"].values():
        for item in support.values():
            path = ROOT / item["path"]
            if (
                not path.is_file()
                or path.stat().st_size != item["bytes"]
                or sha256(path) != item["sha256"]
            ):
                raise RuntimeError(f"frozen observation support changed: {path}")
    authorization = config["authorization"]
    for key in (
        "access_A3667_validation",
        "access_A754_holdout",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed response boundary is open: {key}")
    return config


def normalize_intervals(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    pairs = sorted(
        (float(start), float(stop))
        for start, stop in zip(starts, stops, strict=True)
        if np.isfinite(start) and np.isfinite(stop) and stop > start
    )
    merged: list[list[float]] = []
    for start, stop in pairs:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return np.asarray(merged, dtype=float).reshape((-1, 2))


def clip_intervals(intervals: np.ndarray, start: float, stop: float) -> np.ndarray:
    clipped_start = np.maximum(np.asarray(intervals[:, 0], dtype=float), start)
    clipped_stop = np.minimum(np.asarray(intervals[:, 1], dtype=float), stop)
    keep = clipped_stop > clipped_start
    return normalize_intervals(clipped_start[keep], clipped_stop[keep])


def intersect_intervals(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_intervals(left[:, 0], left[:, 1])
    right = normalize_intervals(right[:, 0], right[:, 1])
    output: list[tuple[float, float]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i, 0], right[j, 0])
        stop = min(left[i, 1], right[j, 1])
        if stop > start:
            output.append((start, stop))
        if left[i, 1] < right[j, 1]:
            i += 1
        else:
            j += 1
    return np.asarray(output, dtype=float).reshape((-1, 2))


def read_intervals(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        for name in ("GTI", "STDGTI"):
            if name in hdus:
                data = hdus[name].data
                return normalize_intervals(data["START"], data["STOP"]), hdus[name].header.copy()
    raise RuntimeError(f"no GTI or STDGTI extension: {path}")


def write_gti(path: Path, intervals: np.ndarray, template_header: fits.Header) -> None:
    columns = [
        fits.Column(name="START", format="D", unit="s", array=intervals[:, 0]),
        fits.Column(name="STOP", format="D", unit="s", array=intervals[:, 1]),
    ]
    hdu = fits.BinTableHDU.from_columns(columns, name="GTI")
    for key in (
        "TELESCOP",
        "INSTRUME",
        "TIMESYS",
        "TIMEUNIT",
        "MJDREFI",
        "MJDREFF",
        "TIMEREF",
    ):
        if key in template_header:
            hdu.header[key] = template_header[key]
    exposure = float(np.sum(intervals[:, 1] - intervals[:, 0]))
    for header in (hdu.header,):
        header["TSTART"] = float(intervals[0, 0])
        header["TSTOP"] = float(intervals[-1, 1])
        header["EXPOSURE"] = exposure
        header["ONTIME"] = exposure
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, checksum=True)


def select_event_rows(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    selected = np.zeros(len(times), dtype=bool)
    for start, stop in intervals:
        selected |= (times >= start) & (times <= stop)
    return selected


def write_corrected_event(source: Path, output: Path, intervals: np.ndarray) -> dict[str, Any]:
    with fits.open(source, memmap=True, mode="readonly") as hdus:
        events = hdus["EVENTS"]
        selected = select_event_rows(np.asarray(events.data["TIME"], dtype=float), intervals)
        new_hdus = fits.HDUList([hdu.copy() for hdu in hdus])
        new_hdus["EVENTS"].data = events.data[selected].copy()
        gti_index = new_hdus.index_of("GTI")
        gti_header = new_hdus[gti_index].header.copy()
        gti_columns = [
            fits.Column(name="START", format="D", unit="s", array=intervals[:, 0]),
            fits.Column(name="STOP", format="D", unit="s", array=intervals[:, 1]),
        ]
        replacement = fits.BinTableHDU.from_columns(gti_columns, header=gti_header, name="GTI")
        new_hdus[gti_index] = replacement
        exposure = float(np.sum(intervals[:, 1] - intervals[:, 0]))
        for hdu in (new_hdus["EVENTS"], new_hdus["GTI"]):
            hdu.header["TSTART"] = float(intervals[0, 0])
            hdu.header["TSTOP"] = float(intervals[-1, 1])
            hdu.header["EXPOSURE"] = exposure
            hdu.header["ONTIME"] = exposure
            if "LIVETIME" in hdu.header and "DEADC" in hdu.header:
                hdu.header["LIVETIME"] = exposure * float(hdu.header["DEADC"])
        new_hdus.writeto(output, checksum=True)
    with fits.open(output, memmap=True, mode="readonly") as hdus:
        event_rows = int(len(hdus["EVENTS"].data))
        output_intervals = normalize_intervals(hdus["GTI"].data["START"], hdus["GTI"].data["STOP"])
    if not np.array_equal(output_intervals, intervals):
        raise RuntimeError(f"corrected GTI changed while writing {output}")
    return {
        "rows": event_rows,
        "gti_rows": len(intervals),
        "exposure_seconds": exposure,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }


def maketime_command(config: dict[str, Any], ehk: Path, output: Path) -> str:
    expression = config["gti_protocol"]["step_2"].split("expression ", 1)[1].rstrip(".")
    return (
        application.runtime_environment(config)
        + "export PFILES="
        + shlex.quote(config["runtime"]["pfiles"])
        + "; punlearn maketime; maketime infile="
        + shlex.quote(application.to_wsl_path(ehk) + "[EHK]")
        + " outfile="
        + shlex.quote(application.to_wsl_path(output))
        + " expr="
        + shlex.quote(expression)
        + " compact=no time=TIME copykw=no clobber=yes"
    )


def compress_pixlist(pixels: list[int]) -> str:
    values = sorted(set(pixels))
    runs: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    runs.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(runs)


def validate_detector_mapping(config: dict[str, Any], event: Path) -> None:
    expected = {int(pixel): tuple(center) for pixel, center in config["detector_pixel_centers"].items()}
    with fits.open(event, memmap=True, mode="readonly") as hdus:
        data = hdus["EVENTS"].data
        for pixel, center in expected.items():
            rows = data[data["PIXEL"] == pixel]
            actual = set(zip(rows["DETX"].tolist(), rows["DETY"].tolist(), strict=True))
            if actual != {center}:
                raise RuntimeError(f"DETX/DETY mapping changed for pixel {pixel}: {actual}")


def write_regions(config: dict[str, Any], directory: Path) -> list[dict[str, Any]]:
    mapping = {int(pixel): center for pixel, center in config["detector_pixel_centers"].items()}
    records: list[dict[str, Any]] = []
    for name, pixels in config["region_pixels"].items():
        lines = ["# Region file format: DS9 version 4.1", "physical"]
        lines.extend(f"box({mapping[pixel][0]},{mapping[pixel][1]},1,1,0)" for pixel in pixels)
        path = directory / f"detector_{name}.reg"
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        records.append(
            {
                "region": name,
                "pixels": pixels,
                "pixlist": compress_pixlist(pixels),
                "path": path.name,
                "sha256": sha256(path),
            }
        )
    return records


def prepare(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = validate_config(config_path)
    product_root = (ROOT / config["paths"]["product_root"]).resolve()
    processed_root = (ROOT / "data/processed").resolve()
    if not product_root.is_relative_to(processed_root):
        raise RuntimeError("response products must remain under data/processed")
    if product_root.exists():
        raise RuntimeError(f"refusing to overwrite response product root: {product_root}")
    product_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=product_root.name + ".installing.", dir=product_root.parent))
    commands: list[dict[str, Any]] = []
    branch_records: list[dict[str, Any]] = []
    environment_gtis: dict[str, tuple[np.ndarray, fits.Header, Path]] = {}

    try:
        for obsid, support in config["observation_support"].items():
            output = staging / f"{obsid}_source_environment.gti"
            command = maketime_command(config, ROOT / support["ehk"]["path"], output)
            result = application.run_wsl(config["runtime"]["wsl_distribution"], command, timeout=1200)
            commands.append({"stage": "maketime", "obsid": obsid, **result})
            if (
                result["exit_code"] != 0
                or "could not load system parameter file" in result["stderr"]
                or not output.is_file()
            ):
                raise RuntimeError(f"source-environment maketime failed for {obsid}")
            intervals, header = read_intervals(output)
            environment_gtis[obsid] = (intervals, header, output)

        validate_detector_mapping(config, ROOT / config["branches"][0]["event_path"])
        region_records = write_regions(config, staging)

        for branch in config["branches"]:
            source = ROOT / branch["event_path"]
            parent_intervals, parent_header = read_intervals(source)
            clipped = clip_intervals(parent_intervals, branch["branch_start"], branch["branch_stop"])
            if len(clipped) != branch["clipped_parent_gti_rows"]:
                raise RuntimeError(f"clipped parent GTI row count changed: {branch['name']}")
            clipped_exposure = float(np.sum(clipped[:, 1] - clipped[:, 0]))
            if not np.isclose(clipped_exposure, branch["clipped_parent_gti_exposure_seconds"], atol=1e-6):
                raise RuntimeError(f"clipped parent GTI exposure changed: {branch['name']}")
            environment, _, _ = environment_gtis[branch["obsid"]]
            final = intersect_intervals(clipped, environment)
            if not len(final):
                raise RuntimeError(f"environment screen removed all GTI: {branch['name']}")
            branch_directory = staging / branch["name"]
            branch_directory.mkdir()
            clipped_path = branch_directory / "branch_clipped.gti"
            final_path = branch_directory / "final_analysis.gti"
            corrected_path = branch_directory / "corrected_branch.evt"
            write_gti(clipped_path, clipped, parent_header)
            write_gti(final_path, final, parent_header)
            product = write_corrected_event(source, corrected_path, final)
            if product["rows"] <= 0:
                raise RuntimeError(f"no corrected event rows: {branch['name']}")
            branch_records.append(
                {
                    "branch": branch["name"],
                    "obsid": branch["obsid"],
                    "parent_clipped": {
                        "rows": len(clipped),
                        "exposure_seconds": clipped_exposure,
                        "sha256": sha256(clipped_path),
                    },
                    "final": product,
                    "final_gti_sha256": sha256(final_path),
                    "retained_exposure_fraction": product["exposure_seconds"] / clipped_exposure,
                }
            )

        os.replace(staging, product_root)
    except Exception:
        raise

    report = {
        "protocol_version": "SIGMA-V19CY-A2319-RESPONSE-INPUT-PREPARATION-RESULT-1.0.1",
        "status": "exact_branch_gti_and_detector_region_inputs_prepared",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "branches": branch_records,
        "regions": region_records,
        "commands": commands,
        "terminal_gate_passed": (
            len(branch_records) == 3
            and len(region_records) == 7
            and all(item["final"]["rows"] > 0 for item in branch_records)
            and all(item["exit_code"] == 0 for item in commands)
            and all("could not load system parameter file" not in item["stderr"] for item in commands)
        ),
        "science_energy_distribution_summarized_or_fit": False,
        "response_or_background_generated": False,
        "velocity_fit_performed": False,
        "validation_or_holdout_accessed": False,
    }
    report_path = ROOT / config["paths"]["preparation_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, sort_keys=True))
