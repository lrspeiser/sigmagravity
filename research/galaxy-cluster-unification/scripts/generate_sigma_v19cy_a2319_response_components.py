#!/usr/bin/env python3
"""Generate frozen A2319 source, NXB, RMF, and exposure-map components."""

from __future__ import annotations

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
import prepare_sigma_v19cy_a2319_response_inputs as preparation

CONFIG = ROOT / "configs/sigma_v19cy_a2319_response_aware_spectral.json"
PREPARATION_REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_input_preparation.json"
)
CHANDRA_REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_chandra_image.json"
)
REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_components.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG)
    if config.get("protocol_version") not in {
        "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.4",
        "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.5",
        "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.6",
    }:
        raise RuntimeError("unexpected response-aware protocol")
    authorization = config["authorization"]
    if authorization["access_A3667_validation"] or authorization["access_A754_holdout"]:
        raise RuntimeError("sealed validation or holdout access is enabled")
    if authorization["open_lensing_halo_or_gravity_targets"]:
        raise RuntimeError("gravity targets are not sealed")
    for parent in config["parents"].values():
        path = ROOT / parent["path"]
        if not path.is_file() or preparation.sha256(path) != parent["sha256"]:
            raise RuntimeError(f"frozen parent changed: {path}")
    prep = load_json(PREPARATION_REPORT)
    chandra = load_json(CHANDRA_REPORT)
    if not prep.get("terminal_gate_passed") or not chandra.get("terminal_gate_passed"):
        raise RuntimeError("a required preparation gate did not pass")
    if any(
        prep.get(key)
        for key in (
            "science_energy_distribution_summarized_or_fit",
            "response_or_background_generated",
            "velocity_fit_performed",
            "validation_or_holdout_accessed",
        )
    ):
        raise RuntimeError("preparation report crossed a frozen boundary")
    prep_branches = {item["branch"]: item for item in prep["branches"]}
    product_root = (ROOT / config["paths"]["product_root"]).resolve()
    for branch in config["branches"]:
        branch_root = product_root / branch["name"]
        event = branch_root / "corrected_branch.evt"
        gti = branch_root / "final_analysis.gti"
        expected = prep_branches[branch["name"]]
        if preparation.sha256(event) != expected["final"]["sha256"]:
            raise RuntimeError(f"prepared event changed: {event}")
        if preparation.sha256(gti) != expected["final_gti_sha256"]:
            raise RuntimeError(f"prepared GTI changed: {gti}")
    image = product_root / "chandra/a2319_chandra_0p5_7p0keV_12arcmin.img"
    if preparation.sha256(image) != chandra["image"]["sha256"]:
        raise RuntimeError("frozen Chandra image changed")
    return config, prep, chandra


def runtime_prefix(config: dict[str, Any]) -> str:
    return (
        application.runtime_environment(config)
        + "export PFILES="
        + shlex.quote(config["runtime"]["pfiles"])
        + "; "
    )


def tool_path(config: dict[str, Any], path: Path) -> str:
    text = str(path.resolve())
    distribution = config["runtime"]["wsl_distribution"]
    prefix = f"\\\\wsl.localhost\\{distribution}\\"
    if text.lower().startswith(prefix.lower()):
        return "/" + text[len(prefix) :].replace("\\", "/")
    return application.to_wsl_path(path)


def pixel_clause(pixels: list[int]) -> str:
    return "(" + "||".join(f"PIXEL=={value}" for value in pixels) + ")"


def nxb_pixel_list(pixels: list[int]) -> str:
    return ",".join(str(value) for value in sorted(set(pixels)))


def event_date_obs(event: Path) -> str:
    with fits.open(event, memmap=True, mode="readonly") as hdus:
        value = hdus["EVENTS"].header.get("DATE-OBS")
    if not isinstance(value, str) or "T" not in value:
        raise RuntimeError(f"DATE-OBS is unavailable in {event}")
    return value


def ftcopy_event_command(
    config: dict[str, Any], event: Path, output: Path, selection: str
) -> str:
    infile = tool_path(config, event) + "[EVENTS][" + selection + "]"
    return (
        runtime_prefix(config)
        + "punlearn ftcopy; ftcopy infile="
        + shlex.quote(infile)
        + " outfile="
        + shlex.quote(tool_path(config, output))
        + " copyall=yes clobber=no history=yes"
    )


def extractor_command(config: dict[str, Any], event: Path, output: Path) -> str:
    return (
        runtime_prefix(config)
        + "punlearn extractor; extractor filename="
        + shlex.quote(tool_path(config, event))
        + " eventsout=NONE imgfile=NONE phafile="
        + shlex.quote(tool_path(config, output))
        + " fitsbinlc=NONE qdpfile=NONE unbinlc=NONE regionfile=NONE timefile=none"
        + " xcolf=DETX ycolf=DETY xcolh=DETX ycolh=DETY tcol=TIME ecol=PI"
        + " ccol=NONE gcol=NONE gstring=NONE events=EVENTS gti=GTI"
        + " specbin=1 wtmapb=no fullimage=no copyall=no clobber=no"
    )


def nxb_command(
    config: dict[str, Any], event: Path, ehk: Path, region_file: Path, output: Path, pixels: list[int]
) -> str:
    protocol = config["nxb_protocol"]
    return (
        runtime_prefix(config)
        + "punlearn rslnxbgen; rslnxbgen infile="
        + shlex.quote(tool_path(config, event))
        + " ehkfile="
        + shlex.quote(tool_path(config, ehk))
        + " regfile="
        + shlex.quote(tool_path(config, region_file))
        + " innxbfile="
        + shlex.quote(tool_path(config, ROOT / protocol["base_event_path"]))
        + " innxbehk="
        + shlex.quote(tool_path(config, ROOT / protocol["base_ehk_path"]))
        + " outpifile="
        + shlex.quote(tool_path(config, output))
        + " outehkfile=NONE outnxbfile=NONE outnxbehk=NONE database=LOCAL db_location="
        + shlex.quote(tool_path(config, ROOT / protocol["database_path"]))
        + " exclude_keys=NONE regmode=DET timefirst="
        + protocol["timefirst_parameter"]
        + " timelast="
        + protocol["timelast_parameter"]
        + " picol=PI sortcol="
        + shlex.quote(protocol["sortcol"])
        + " sortbin="
        + shlex.quote(protocol["sortbin"])
        + " pixels="
        + shlex.quote(preparation.compress_pixlist(pixels))
        + " expr="
        + shlex.quote(config["event_selections"]["nxb_event_expression"])
        + " cleanup=yes clobber=no"
    )


def rmf_command(
    config: dict[str, Any], event: Path, output_root: Path, pixels: list[int]
) -> str:
    protocol = config["rmf_protocol"]
    return (
        runtime_prefix(config)
        + "punlearn rslmkrmf; rslmkrmf infile="
        + shlex.quote(tool_path(config, event))
        + " outfileroot="
        + shlex.quote(tool_path(config, output_root))
        + " splitrmf=no resolist="
        + shlex.quote(protocol["resolist"])
        + " combps=no secondaries=yes regmode=DET regionfile=NONE pixlist="
        + shlex.quote(preparation.compress_pixlist(pixels))
        + " pixeltest=CENTER outrsp=no arfinfile=NONE time="
        + shlex.quote(event_date_obs(event))
        + " rmfparamfile=CALDB whichrmf="
        + shlex.quote(protocol["whichrmf"])
        + " cleanup=yes clobber=no"
    )


def expmap_command(
    config: dict[str, Any], ehk: Path, event: Path, pixel_gti: Path, output: Path
) -> str:
    protocol = config["attitude_and_arf_protocol"]["xaexpmap"]
    return (
        runtime_prefix(config)
        + "punlearn xaexpmap; xaexpmap ehkfile="
        + shlex.quote(tool_path(config, ehk))
        + " gtifile="
        + shlex.quote(tool_path(config, event) + "[GTI]")
        + " instrume=RESOLVE badimgfile=NONE pixgtifile="
        + shlex.quote(tool_path(config, pixel_gti))
        + " outfile="
        + shlex.quote(tool_path(config, output))
        + " outmaptype=EXPOSURE delta="
        + str(protocol["delta_arcmin"])
        + " numphi="
        + str(protocol["numphi"])
        + " maskcalsrc=yes clobber=no"
    )


def run_command(config: dict[str, Any], stage: str, command: str) -> dict[str, Any]:
    record = application.run_wsl(
        config["runtime"]["wsl_distribution"], command, timeout=7200
    )
    record["stage"] = stage
    return record


def require_success(record: dict[str, Any]) -> None:
    if record["exit_code"] != 0:
        raise RuntimeError(
            f"{record['stage']} failed with exit {record['exit_code']}: {record['stderr']}"
        )


def inspect_pha(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        hdus.verify("exception")
        spectrum = hdus["SPECTRUM"]
        names = set(spectrum.columns.names)
        count_name = "COUNTS" if "COUNTS" in names else "RATE"
        values = np.asarray(spectrum.data[count_name], dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise RuntimeError(f"invalid PHA values in {path}")
        exposure = float(spectrum.header.get("EXPOSURE", np.nan))
        if not np.isfinite(exposure) or exposure <= 0:
            raise RuntimeError(f"invalid PHA exposure in {path}")
        total = float(np.sum(values, dtype=np.float64))
    return {
        "bytes": path.stat().st_size,
        "sha256": preparation.sha256(path),
        "channels": int(values.size),
        "value_column": count_name,
        "total": total,
        "exposure_seconds": exposure,
    }


def inspect_event(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        hdus.verify("exception")
        rows = len(hdus["EVENTS"].data)
        gti = hdus["GTI"].data
        exposure = float(np.sum(gti["STOP"] - gti["START"], dtype=np.float64))
    if rows <= 0 or not np.isfinite(exposure) or exposure <= 0:
        raise RuntimeError(f"invalid filtered event file: {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": preparation.sha256(path),
        "rows": int(rows),
        "gti_rows": len(gti),
        "gti_exposure_seconds": exposure,
    }


def inspect_rmf(path: Path) -> dict[str, Any]:
    minimum = np.inf
    maximum = -np.inf
    elements = 0
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        hdus.verify("exception")
        matrix = hdus["MATRIX"].data
        for row in matrix["MATRIX"]:
            values = np.asarray(row, dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise RuntimeError(f"non-finite or negative RMF matrix in {path}")
            if values.size:
                minimum = min(minimum, float(np.min(values)))
                maximum = max(maximum, float(np.max(values)))
                elements += int(values.size)
        channels = len(hdus["EBOUNDS"].data)
    if elements == 0:
        raise RuntimeError(f"empty RMF matrix in {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": preparation.sha256(path),
        "matrix_rows": len(matrix),
        "matrix_elements": elements,
        "channels": int(channels),
        "minimum_matrix_value": minimum,
        "maximum_matrix_value": maximum,
    }


def inspect_expmap(path: Path) -> dict[str, Any]:
    finite_values = 0
    positive_values = 0
    extensions: list[str] = []
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        hdus.verify("exception")
        for hdu in hdus:
            extensions.append(hdu.name)
            if hdu.data is None:
                continue
            array = np.asarray(hdu.data)
            if array.dtype.fields:
                continue
            numeric = np.asarray(array, dtype=float)
            finite_values += int(np.count_nonzero(np.isfinite(numeric)))
            positive_values += int(np.count_nonzero(np.isfinite(numeric) & (numeric > 0)))
    if finite_values == 0:
        raise RuntimeError(f"exposure map has no finite numeric image: {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": preparation.sha256(path),
        "extensions": extensions,
        "finite_image_values": finite_values,
        "positive_image_values": positive_values,
    }


def generate() -> dict[str, Any]:
    config, prep, chandra = validate_inputs()
    product_root = (ROOT / config["paths"]["product_root"]).resolve()
    output_root = product_root / "response_components"
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite response components: {output_root}")
    distribution = config["runtime"]["wsl_distribution"]
    native_temp = Path(f"//wsl.localhost/{distribution}/tmp")
    if not native_temp.is_dir():
        raise RuntimeError(f"WSL-native temporary directory is unavailable: {native_temp}")
    staging = Path(
        tempfile.mkdtemp(prefix="sigma_v19cy_response_components_", dir=native_temp)
    )
    commands: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    try:
        for branch in config["branches"]:
            name = branch["name"]
            source_root = product_root / name
            event = source_root / "corrected_branch.evt"
            ehk = ROOT / config["observation_support"][branch["obsid"]]["ehk"]["path"]
            pixel_gti = (
                ROOT / config["observation_support"][branch["obsid"]]["pixel_gti"]["path"]
            )
            branch_root = staging / name
            branch_root.mkdir(parents=True)
            grade_event = branch_root / "rmf_grade_weight.evt"
            record = run_command(
                config,
                f"{name}:ftcopy_rmf_grade_weight",
                ftcopy_event_command(
                    config,
                    event,
                    grade_event,
                    config["event_selections"]["rmf_grade_weight"],
                ),
            )
            commands.append(record)
            require_success(record)
            grade_event_summary = inspect_event(grade_event)
            expmap = branch_root / "exposure_map.fits"
            command = expmap_command(config, ehk, event, pixel_gti, expmap)
            record = run_command(config, f"{name}:xaexpmap", command)
            commands.append(record)
            require_success(record)
            regions: list[dict[str, Any]] = []
            for region in branch["regions"]:
                pixels = config["region_pixels"][region]
                region_root = branch_root / region
                region_root.mkdir()
                source_event = region_root / "source_hp.evt"
                source_pha = region_root / "source.pha"
                nxb_pha = region_root / "nxb.pha"
                rmf_root = region_root / "response"
                region_file = product_root / f"detector_{region}.reg"
                source_selection = (
                    config["event_selections"]["source_hp"]
                    + "&&"
                    + pixel_clause(pixels)
                )
                record = run_command(
                    config,
                    f"{name}:{region}:ftcopy_source_hp",
                    ftcopy_event_command(config, event, source_event, source_selection),
                )
                commands.append(record)
                require_success(record)
                source_event_summary = inspect_event(source_event)
                record = run_command(
                    config,
                    f"{name}:{region}:extractor",
                    extractor_command(config, source_event, source_pha),
                )
                commands.append(record)
                require_success(record)
                record = run_command(
                    config,
                    f"{name}:{region}:rslnxbgen",
                    nxb_command(config, event, ehk, region_file, nxb_pha, pixels),
                )
                commands.append(record)
                require_success(record)
                record = run_command(
                    config,
                    f"{name}:{region}:rslmkrmf",
                    rmf_command(config, grade_event, rmf_root, pixels),
                )
                commands.append(record)
                require_success(record)
                rmfs = sorted(region_root.glob("response*.rmf"))
                if len(rmfs) != 1:
                    raise RuntimeError(f"expected one RMF for {name}/{region}, found {rmfs}")
                regions.append(
                    {
                        "region": region,
                        "pixels": pixels,
                        "source_event": source_event_summary,
                        "source_pha": inspect_pha(source_pha),
                        "nxb_pha": inspect_pha(nxb_pha),
                        "rmf_name": rmfs[0].name,
                        "rmf": inspect_rmf(rmfs[0]),
                    }
                )
            branches.append(
                {
                    "branch": name,
                    "obsid": branch["obsid"],
                    "date_obs": event_date_obs(event),
                    "rmf_grade_weight_event": grade_event_summary,
                    "expmap": inspect_expmap(expmap),
                    "regions": regions,
                }
            )
        publish_staging = Path(
            tempfile.mkdtemp(prefix="response_components.installing.", dir=product_root)
        )
        try:
            shutil.copytree(staging, publish_staging, dirs_exist_ok=True)
            os.replace(publish_staging, output_root)
        except Exception:
            shutil.rmtree(publish_staging, ignore_errors=True)
            raise
        shutil.rmtree(staging)
    except Exception as exc:
        failure = {
            "protocol_version": "SIGMA-V19CY-A2319-RESPONSE-COMPONENTS-RESULT-1.0.0",
            "status": "response_component_generation_failed_closed",
            "generated_utc": datetime.now(UTC).isoformat(),
            "config_sha256": preparation.sha256(CONFIG),
            "error": str(exc),
            "staging_path": str(staging),
            "commands": commands,
            "component_gate_passed": False,
            "arf_generated": False,
            "velocity_fit_performed": False,
            "validation_or_holdout_accessed": False,
        }
        REPORT.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    report = {
        "protocol_version": "SIGMA-V19CY-A2319-RESPONSE-COMPONENTS-RESULT-1.0.0",
        "status": "source_nxb_rmf_and_exposure_components_completed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": preparation.sha256(CONFIG),
        "preparation_report_sha256": preparation.sha256(PREPARATION_REPORT),
        "chandra_report_sha256": preparation.sha256(CHANDRA_REPORT),
        "branches": branches,
        "commands": commands,
        "component_gate_passed": (
            len(branches) == 3
            and sum(len(branch["regions"]) for branch in branches) == 10
            and all(command["exit_code"] == 0 for command in commands)
        ),
        "arf_generated": False,
        "xrism_energy_distribution_summarized_or_fit": False,
        "velocity_fit_performed": False,
        "validation_or_holdout_accessed": False,
        "preparation_status": prep["status"],
        "chandra_status": chandra["status"],
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, sort_keys=True))
