#!/usr/bin/env python3
"""Generate the frozen A2319 IMAGE-source ARFs after component completion."""

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

import generate_sigma_v19cy_a2319_response_components as components
import prepare_sigma_v19cy_a2319_response_inputs as preparation

CONFIG = ROOT / "configs/sigma_v19cy_a2319_response_aware_spectral.json"
COMPONENT_REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_components.json"
)
REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_arfs.json"
)
ARF_TIMEOUT_SECONDS = 43_200


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    config, _, _ = components.validate_inputs()
    if not COMPONENT_REPORT.is_file():
        raise RuntimeError("response-component report is unavailable")
    report = load_json(COMPONENT_REPORT)
    if report.get("protocol_version") != (
        "SIGMA-V19CY-A2319-RESPONSE-COMPONENTS-RESULT-1.0.0"
    ):
        raise RuntimeError("unexpected response-component result protocol")
    if not report.get("component_gate_passed"):
        raise RuntimeError("response-component gate did not pass")
    if report.get("arf_generated") or report.get("velocity_fit_performed"):
        raise RuntimeError("component report crossed its frozen boundary")
    if report.get("validation_or_holdout_accessed"):
        raise RuntimeError("sealed validation or holdout data were accessed")
    if report.get("config_sha256") != preparation.sha256(CONFIG):
        raise RuntimeError("response-component config hash does not match")

    product_root = (ROOT / config["paths"]["product_root"]).resolve()
    image = product_root / "chandra/a2319_chandra_0p5_7p0keV_12arcmin.img"
    if not image.is_file():
        raise RuntimeError("frozen Chandra source image is unavailable")
    report_branches = {row["branch"]: row for row in report["branches"]}
    if set(report_branches) != {row["name"] for row in config["branches"]}:
        raise RuntimeError("component branches do not match the frozen protocol")
    region_count = 0
    for branch in config["branches"]:
        name = branch["name"]
        branch_report = report_branches[name]
        branch_root = product_root / "response_components" / name
        expmap = branch_root / "exposure_map.fits"
        if preparation.sha256(expmap) != branch_report["expmap"]["sha256"]:
            raise RuntimeError(f"component exposure map changed: {expmap}")
        report_regions = {row["region"]: row for row in branch_report["regions"]}
        if set(report_regions) != set(branch["regions"]):
            raise RuntimeError(f"component regions do not match for {name}")
        for region in branch["regions"]:
            region_count += 1
            row = report_regions[region]
            region_root = branch_root / region
            for filename, summary_key in (
                ("source.pha", "source_pha"),
                ("nxb.pha", "nxb_pha"),
                (row["rmf_name"], "rmf"),
            ):
                path = region_root / filename
                if preparation.sha256(path) != row[summary_key]["sha256"]:
                    raise RuntimeError(f"component product changed: {path}")
    if region_count != 10:
        raise RuntimeError(f"expected 10 component regions, found {region_count}")
    return config, report


def arf_command(
    config: dict[str, Any],
    *,
    workdir: Path,
    raytrace: Path,
    expmap: Path,
    rmf: Path,
    image: Path,
    region_file: Path,
    output: Path,
) -> str:
    protocol = config["attitude_and_arf_protocol"]["xaarfgen"]
    return (
        components.runtime_prefix(config)
        + "cd "
        + shlex.quote(components.tool_path(config, workdir))
        + "; punlearn xaarfgen; xaarfgen xrtevtfile="
        + shlex.quote(components.tool_path(config, raytrace))
        + " source_ra="
        + str(protocol["source_ra_deg"])
        + " source_dec="
        + str(protocol["source_dec_deg"])
        + " telescop=XRISM instrume=RESOLVE teldeffile=CALDB emapfile="
        + shlex.quote(components.tool_path(config, expmap))
        + " qefile=CALDB obffile=CALDB fwfile=CALDB contamifile=CALDB"
        + " abund=1.0 cols=0.0 covfac=1.0 gatevalvefile=CALDB rmffile="
        + shlex.quote(components.tool_path(config, rmf))
        + " erange="
        + shlex.quote(protocol["erange"])
        + " onaxisffile=CALDB onaxiscfile=CALDB outfile="
        + shlex.quote(components.tool_path(config, output))
        + " regmode=DET regionfile="
        + shlex.quote(components.tool_path(config, region_file))
        + " rslgapreg=no doublesonly=no mirrorfile=CALDB obstructfile=CALDB"
        + " frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB"
        + " scatterfile=CALDB numphoton="
        + str(protocol["numphoton"])
        + " minphoton="
        + str(protocol["minphoton"])
        + " sourcetype=IMAGE imgfile="
        + shlex.quote(components.tool_path(config, image))
        + " auxtransfile=NONE seed="
        + str(protocol["seed"])
        + " cleanup=yes clobber=no chatter=2 history=yes"
    )


def inspect_arf(arf: Path, rmf: Path) -> dict[str, Any]:
    with fits.open(arf, memmap=True, mode="readonly") as arf_hdus:
        arf_hdus.verify("exception")
        response = arf_hdus["SPECRESP"].data
        lo = np.asarray(response["ENERG_LO"], dtype=float)
        hi = np.asarray(response["ENERG_HI"], dtype=float)
        values = np.asarray(response["SPECRESP"], dtype=float)
    with fits.open(rmf, memmap=True, mode="readonly") as rmf_hdus:
        matrix = rmf_hdus["MATRIX"].data
        rmf_lo = np.asarray(matrix["ENERG_LO"], dtype=float)
        rmf_hi = np.asarray(matrix["ENERG_HI"], dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise RuntimeError(f"invalid ARF response values in {arf}")
    if np.count_nonzero(values > 0) == 0:
        raise RuntimeError(f"ARF has no positive effective area: {arf}")
    grid_exact = bool(
        lo.shape == rmf_lo.shape
        and hi.shape == rmf_hi.shape
        and np.array_equal(lo, rmf_lo)
        and np.array_equal(hi, rmf_hi)
    )
    if not grid_exact:
        raise RuntimeError(f"ARF and RMF energy grids differ: {arf}, {rmf}")
    return {
        "bytes": arf.stat().st_size,
        "sha256": preparation.sha256(arf),
        "rows": int(values.size),
        "finite": True,
        "positive_bins": int(np.count_nonzero(values > 0)),
        "minimum_cm2": float(np.min(values)),
        "maximum_cm2": float(np.max(values)),
        "rmf_energy_grid_exact": grid_exact,
    }


def inspect_raytrace(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        hdus.verify("exception")
        extensions = [hdu.name for hdu in hdus]
        table_rows = {
            hdu.name: len(hdu.data)
            for hdu in hdus
            if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None
        }
    if not table_rows or max(table_rows.values()) <= 0:
        raise RuntimeError(f"raytrace contains no table events: {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": preparation.sha256(path),
        "extensions": extensions,
        "table_rows": table_rows,
    }


def run_command(config: dict[str, Any], stage: str, command: str) -> dict[str, Any]:
    record = components.application.run_wsl(
        config["runtime"]["wsl_distribution"],
        command,
        timeout=ARF_TIMEOUT_SECONDS,
    )
    record["stage"] = stage
    return record


def generate() -> dict[str, Any]:
    config, component_report = validate_inputs()
    product_root = (ROOT / config["paths"]["product_root"]).resolve()
    component_root = product_root / "response_components"
    output_root = product_root / "response_arfs"
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite response ARFs: {output_root}")
    distribution = config["runtime"]["wsl_distribution"]
    native_temp = Path(f"//wsl.localhost/{distribution}/tmp")
    if not native_temp.is_dir():
        raise RuntimeError(f"WSL-native temporary directory is unavailable: {native_temp}")
    staging = Path(tempfile.mkdtemp(prefix="sigma_v19cy_response_arfs_", dir=native_temp))
    image = product_root / "chandra/a2319_chandra_0p5_7p0keV_12arcmin.img"
    report_branches = {row["branch"]: row for row in component_report["branches"]}
    commands: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    try:
        for branch in config["branches"]:
            name = branch["name"]
            branch_stage = staging / name
            branch_stage.mkdir(parents=True)
            component_branch = component_root / name
            expmap = component_branch / "exposure_map.fits"
            raytrace = branch_stage / "image_raytrace.fits"
            report_regions = {
                row["region"]: row for row in report_branches[name]["regions"]
            }
            regions: list[dict[str, Any]] = []
            frozen_raytrace_sha: str | None = None
            for region in branch["regions"]:
                region_stage = branch_stage / region
                region_stage.mkdir()
                component_region = component_branch / region
                rmf = component_region / report_regions[region]["rmf_name"]
                detector_region = product_root / f"detector_{region}.reg"
                arf = region_stage / "response.arf"
                raytrace_existed_before = raytrace.is_file()
                raytrace_sha_before = (
                    preparation.sha256(raytrace) if raytrace_existed_before else None
                )
                command = arf_command(
                    config,
                    workdir=region_stage,
                    raytrace=raytrace,
                    expmap=expmap,
                    rmf=rmf,
                    image=image,
                    region_file=detector_region,
                    output=arf,
                )
                record = run_command(config, f"{name}:{region}:xaarfgen", command)
                commands.append(record)
                components.require_success(record)
                if not raytrace.is_file():
                    raise RuntimeError(f"xaarfgen did not create raytrace: {raytrace}")
                raytrace_sha_after = preparation.sha256(raytrace)
                if frozen_raytrace_sha is None:
                    frozen_raytrace_sha = raytrace_sha_after
                elif raytrace_sha_after != frozen_raytrace_sha:
                    raise RuntimeError(f"branch raytrace changed during reuse: {name}")
                if raytrace_sha_before is not None and raytrace_sha_after != raytrace_sha_before:
                    raise RuntimeError(f"existing branch raytrace was overwritten: {name}")
                regions.append(
                    {
                        "region": region,
                        "arf": inspect_arf(arf, rmf),
                        "rmf_sha256": preparation.sha256(rmf),
                        "raytrace_existed_before": raytrace_existed_before,
                        "raytrace_sha256_before": raytrace_sha_before,
                        "raytrace_sha256_after": raytrace_sha_after,
                    }
                )
            branches.append(
                {
                    "branch": name,
                    "raytrace": inspect_raytrace(raytrace),
                    "one_raytrace_reused_within_branch": (
                        regions[0]["raytrace_existed_before"] is False
                        and all(row["raytrace_existed_before"] for row in regions[1:])
                        and len({row["raytrace_sha256_after"] for row in regions}) == 1
                    ),
                    "regions": regions,
                }
            )
        publish_staging = Path(
            tempfile.mkdtemp(prefix="response_arfs.installing.", dir=product_root)
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
            "protocol_version": "SIGMA-V19CY-A2319-ARF-RESULT-1.0.0",
            "status": "response_arf_generation_failed_closed",
            "generated_utc": datetime.now(UTC).isoformat(),
            "config_sha256": preparation.sha256(CONFIG),
            "component_report_sha256": preparation.sha256(COMPONENT_REPORT),
            "error": str(exc),
            "staging_path": str(staging),
            "commands": commands,
            "arf_gate_passed": False,
            "xrism_energy_distribution_summarized_or_fit": False,
            "velocity_fit_performed": False,
            "validation_or_holdout_accessed": False,
        }
        REPORT.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    report = {
        "protocol_version": "SIGMA-V19CY-A2319-ARF-RESULT-1.0.0",
        "status": "image_source_arfs_and_branch_raytraces_completed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": preparation.sha256(CONFIG),
        "component_report_sha256": preparation.sha256(COMPONENT_REPORT),
        "branches": branches,
        "commands": commands,
        "arf_gate_passed": (
            len(branches) == 3
            and sum(len(row["regions"]) for row in branches) == 10
            and all(row["one_raytrace_reused_within_branch"] for row in branches)
            and all(command["exit_code"] == 0 for command in commands)
        ),
        "xrism_energy_distribution_summarized_or_fit": False,
        "velocity_fit_performed": False,
        "validation_or_holdout_accessed": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, sort_keys=True))
