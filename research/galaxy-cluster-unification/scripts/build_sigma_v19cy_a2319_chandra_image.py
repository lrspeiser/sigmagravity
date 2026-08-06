#!/usr/bin/env python3
"""Build the frozen Chandra surface-brightness image for A2319 responses."""

from __future__ import annotations

import argparse
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
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import apply_sigma_v19cy_a2319_calibration_candidates as application
import prepare_sigma_v19cy_a2319_response_inputs as preparation

DEFAULT_CONFIG = ROOT / "configs/sigma_v19cy_a2319_response_aware_spectral.json"
PREPARATION_REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_input_preparation.json"
)
REPORT = (
    ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_chandra_image.json"
)

MERGED_FLUX_IMAGE = "0.5-7.0_flux.img"
FROZEN_CHANDRA_PROTOCOL_SHA256 = (
    "5e485fd746c6346b54d939022e0e1c121bf1d9800cc237afb474c5e2b1206eeb"
)
FROZEN_PREPARATION_REPORT_SHA256 = (
    "67638b4ad8eaa440c1dc7eca80f72007eeaf27028461152f1e905b6260feabc8"
)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_inputs(config_path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    config = preparation.load_json(config_path)
    accepted_protocols = {
        "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.1",
        "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.2",
        "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.3",
        "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.4",
    }
    if config.get("protocol_version") not in accepted_protocols:
        raise RuntimeError("unexpected response-aware spectral protocol")
    if canonical_json_sha256(config["chandra_image_protocol"]) != (
        FROZEN_CHANDRA_PROTOCOL_SHA256
    ):
        raise RuntimeError("frozen Chandra image protocol changed")
    report = preparation.load_json(PREPARATION_REPORT)
    if not report.get("terminal_gate_passed"):
        raise RuntimeError("response input preparation did not pass")
    if preparation.sha256(PREPARATION_REPORT) != FROZEN_PREPARATION_REPORT_SHA256:
        raise RuntimeError("frozen response input preparation report changed")
    if report.get("config_sha256") != preparation.sha256(config_path):
        amendments = (
            config.get("pre_response_interface_amendment", {}),
            config.get("pre_nxb_interface_amendment", {}),
        )
        if config.get("protocol_version") == (
            "SIGMA-V19CY-A2319-RESPONSE-AWARE-SPECTRAL-1.0.1"
        ) or any(
            amendment.get(key)
            for amendment in amendments
            for key in (
                "scientific_inputs_changed",
                "prepared_gti_or_region_bytes_changed",
                "chandra_image_changed",
            )
        ):
            raise RuntimeError("response input preparation belongs to a different protocol")
    if any(
        report.get(key)
        for key in (
            "science_energy_distribution_summarized_or_fit",
            "response_or_background_generated",
            "velocity_fit_performed",
            "validation_or_holdout_accessed",
        )
    ):
        raise RuntimeError("response input preparation crossed a frozen boundary")
    for item in config["chandra_image_protocol"]["inputs"]:
        path = ROOT / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or preparation.sha256(path) != item["sha256"]
        ):
            raise RuntimeError(f"frozen Chandra input changed: {path}")
    return config, report


def work_to_wsl_path(path: Path, distribution: str) -> str:
    text = str(path.resolve())
    prefix = f"\\\\wsl.localhost\\{distribution}\\"
    if text.lower().startswith(prefix.lower()):
        return "/" + text[len(prefix) :].replace("\\", "/")
    return application.to_wsl_path(path)


def merge_obs_command(config: dict[str, Any], work: Path) -> str:
    protocol = config["chandra_image_protocol"]
    events = [ROOT / item["path"] for item in protocol["inputs"] if "evt2" in item["path"]]
    infiles = ",".join(application.to_wsl_path(path) for path in events)
    outroot = work_to_wsl_path(
        work / "merged", config["runtime"]["wsl_distribution"]
    ) + "/"
    ciao = config["runtime"]["ciao_prefix"] + "/init/ciao.sh"
    band = f"{protocol['energy_band_keV'][0]}:{protocol['energy_band_keV'][1]}:2.3"
    return (
        "source "
        + shlex.quote(ciao)
        + " >/dev/null 2>&1; export PFILES="
        + shlex.quote(
            "/home/henry/cxcds_param4;" + config["runtime"]["ciao_prefix"] + "/param"
        )
        + "; punlearn merge_obs; merge_obs infiles="
        + shlex.quote(infiles)
        + " outroot="
        + shlex.quote(outroot)
        + " bands="
        + shlex.quote(band)
        + " binsize=1 units=default random=7 parallel=no nproc=1 cleanup=yes clobber=no verbose=1"
    )


def crop_positive_image(source: Path, output: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    with fits.open(source, memmap=True, mode="readonly") as hdus:
        image_hdu = next(
            (hdu for hdu in hdus if hdu.data is not None and np.ndim(hdu.data) == 2),
            None,
        )
        if image_hdu is None:
            raise RuntimeError(f"no 2-D Chandra image in {source}")
        data = np.asarray(image_hdu.data, dtype=float)
        header = image_hdu.header.copy()
    wcs = WCS(header).celestial
    if not wcs.has_celestial:
        raise RuntimeError("merged Chandra image lacks celestial WCS")
    scales_arcsec = np.abs(proj_plane_pixel_scales(wcs) * 3600.0)
    if not np.isfinite(scales_arcsec).all() or np.max(scales_arcsec) / np.min(scales_arcsec) > 1.01:
        raise RuntimeError(f"invalid Chandra image pixel scales: {scales_arcsec}")
    scale_arcsec = float(np.mean(scales_arcsec))
    center = SkyCoord(protocol["crop_center_ra_deg"], protocol["crop_center_dec_deg"], unit="deg")
    center_x, center_y = wcs.world_to_pixel(center)
    width_pixels = round(protocol["crop_width_arcmin"] * 60.0 / scale_arcsec)
    width_pixels = max(width_pixels, 1)
    x0 = round(center_x - (width_pixels - 1) / 2.0)
    y0 = round(center_y - (width_pixels - 1) / 2.0)
    x1 = x0 + width_pixels
    y1 = y0 + width_pixels
    if x0 < 0 or y0 < 0 or x1 > data.shape[1] or y1 > data.shape[0]:
        raise RuntimeError("frozen 12-arcmin crop is not contained in merged Chandra image")
    cropped = data[y0:y1, x0:x1]
    nonfinite = int(np.count_nonzero(~np.isfinite(cropped)))
    negative = int(np.count_nonzero(np.isfinite(cropped) & (cropped < 0)))
    cleaned = np.where(np.isfinite(cropped) & (cropped > 0), cropped, 0.0).astype(np.float32)
    if not np.any(cleaned > 0):
        raise RuntimeError("cleaned Chandra source image has no positive pixels")
    header["NAXIS1"] = width_pixels
    header["NAXIS2"] = width_pixels
    header["CRPIX1"] = float(header["CRPIX1"]) - x0
    header["CRPIX2"] = float(header["CRPIX2"]) - y0
    header["HISTORY"] = "Sigma V19CY frozen 12 arcmin crop; non-finite and negative pixels set to zero"
    fits.PrimaryHDU(data=cleaned, header=header).writeto(output, checksum=True)
    output_wcs = WCS(header).celestial
    recovered = output_wcs.pixel_to_world((width_pixels - 1) / 2, (width_pixels - 1) / 2)
    return {
        "shape": [width_pixels, width_pixels],
        "pixel_scale_arcsec": scale_arcsec,
        "width_arcmin": width_pixels * scale_arcsec / 60.0,
        "center_ra_deg": float(recovered.ra.deg),
        "center_dec_deg": float(recovered.dec.deg),
        "positive_pixels": int(np.count_nonzero(cleaned > 0)),
        "zero_pixels": int(np.count_nonzero(cleaned == 0)),
        "replaced_nonfinite_pixels": nonfinite,
        "replaced_negative_pixels": negative,
        "sum_positive_brightness": float(np.sum(cleaned, dtype=np.float64)),
        "bytes": output.stat().st_size,
        "sha256": preparation.sha256(output),
    }


def inspect_completed_merge(merge_root: Path) -> dict[str, Any]:
    """Verify the terminal products needed to recover a detached merge_obs run."""
    expected = (
        "merged_evt.fits",
        "merged.fov",
        "3231_0.5-7.0_flux.img",
        "15187_0.5-7.0_flux.img",
        MERGED_FLUX_IMAGE,
    )
    products: list[dict[str, Any]] = []
    for name in expected:
        path = merge_root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"detached merge is incomplete: {path}")
        with fits.open(path, memmap=True, mode="readonly") as hdus:
            hdus.verify("exception")
            if name.endswith("_flux.img"):
                image_hdu = next(
                    (hdu for hdu in hdus if hdu.data is not None and np.ndim(hdu.data) == 2),
                    None,
                )
                if image_hdu is None or not WCS(image_hdu.header).celestial.has_celestial:
                    raise RuntimeError(f"recovered flux image lacks a celestial image: {path}")
        products.append(
            {"name": name, "bytes": path.stat().st_size, "sha256": preparation.sha256(path)}
        )
    return {"terminal_products_verified": True, "products": products}


def finalize_merge(
    config_path: Path,
    merge_root: Path,
    command_record: dict[str, Any],
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config, preparation_report = validate_inputs(config_path)
    product_root = (ROOT / config["paths"]["product_root"]).resolve()
    output_root = product_root / "chandra"
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite Chandra response image: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="chandra.installing.", dir=output_root.parent))
    try:
        source = merge_root / MERGED_FLUX_IMAGE
        if not source.is_file():
            raise RuntimeError(f"expected merged Chandra flux image is absent: {source}")
        output = staging / "a2319_chandra_0p5_7p0keV_12arcmin.img"
        image = crop_positive_image(source, output, config["chandra_image_protocol"])
        merge_manifest = [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": preparation.sha256(path),
            }
            for path in sorted(merge_root.glob("*"))
            if path.is_file()
        ]
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    command_completed = command_record.get("exit_code") == 0 or bool(
        recovery and recovery.get("terminal_products_verified")
    )
    report = {
        "protocol_version": "SIGMA-V19CY-A2319-CHANDRA-RESPONSE-IMAGE-RESULT-1.0.1",
        "status": "frozen_chandra_response_image_completed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": preparation.sha256(config_path),
        "preparation_report_sha256": preparation.sha256(PREPARATION_REPORT),
        "command": command_record,
        "detached_process_recovery": recovery,
        "merge_manifest": merge_manifest,
        "image": image,
        "terminal_gate_passed": (
            command_completed
            and image["positive_pixels"] > 0
            and abs(
                image["width_arcmin"]
                - config["chandra_image_protocol"]["crop_width_arcmin"]
            )
            <= image["pixel_scale_arcsec"] / 60.0
        ),
        "xrism_energy_distribution_read_or_fit": False,
        "response_or_background_generated": False,
        "velocity_fit_performed": False,
        "validation_or_holdout_accessed": False,
        "preparation_status": preparation_report["status"],
        "closed_failure_history": [
            {
                "attempt": 1,
                "failed_at": "CIAO merge_obs startup before opening either Chandra event table",
                "root_cause": "the non-interactive CIAO initialization left PFILES unset",
                "correction": "export the isolated writable CIAO parameter directory and installed system parameter directory explicitly before punlearn",
                "chandra_image_generated": False,
                "xrism_energy_distribution_read_or_fit": False,
                "response_or_background_generated": False,
                "velocity_fit_performed": False,
                "validation_or_holdout_accessed": False,
            },
            {
                "attempt": 2,
                "failed_at": "the Windows host command window ended after 120 seconds while CIAO remained active",
                "root_cause": "the outer runner timeout was shorter than the full-resolution CIAO projection runtime",
                "correction": "rerun under a long-lived host runner using WSL-native temporary storage and preserve the direct process exit code",
                "scientific_products_recovered": bool(recovery),
                "detached_temporary_products_survived": bool(recovery),
                "xrism_energy_distribution_read_or_fit": False,
                "response_or_background_generated": False,
                "velocity_fit_performed": False,
                "validation_or_holdout_accessed": False,
            },
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def recover_detached_merge(
    merge_root: Path, config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    recovery = inspect_completed_merge(merge_root)
    command_record = {
        "exit_code": None,
        "stdout": "host wrapper ended before CIAO; terminal files recovered after WSL process completion",
        "stderr": "",
        "exit_code_note": "not exposed by the detached Windows WSL process handle",
    }
    return finalize_merge(config_path, merge_root, command_record, recovery)


def build(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, _ = validate_inputs(config_path)
    distribution = config["runtime"]["wsl_distribution"]
    native_temp = Path(f"//wsl.localhost/{distribution}/tmp")
    if not native_temp.is_dir():
        raise RuntimeError(f"WSL-native temporary directory is unavailable: {native_temp}")
    with tempfile.TemporaryDirectory(
        prefix="sigma_v19cy_chandra_merge_", dir=native_temp
    ) as temporary:
        work = Path(temporary)
        command = merge_obs_command(config, work)
        command_record = application.run_wsl(
            distribution, command, timeout=7200
        )
        if command_record["exit_code"] != 0:
            raise RuntimeError(f"CIAO merge_obs failed: {command_record['stderr']}")
        return finalize_merge(config_path, work / "merged", command_record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover-merge-root", type=Path)
    args = parser.parse_args()
    result = (
        recover_detached_merge(args.recover_merge_root.resolve())
        if args.recover_merge_root
        else build()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
