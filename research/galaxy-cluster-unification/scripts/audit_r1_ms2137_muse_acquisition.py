#!/usr/bin/env python3
"""Audit MS2137 cutout provenance and FITS headers without reading science arrays."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_ms2137_muse_feasibility_protocol.json"
FEASIBILITY_PATH = ROOT / "results/r1_ms2137_muse_feasibility/report.json"
REPORT_PATH = ROOT / "results/r1_ms2137_muse_acquisition/report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_report() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    feasibility = json.loads(FEASIBILITY_PATH.read_text(encoding="utf-8"))
    cutout = config["frozen_cutout_request"]
    archive = config["archive_product"]
    cube_path = ROOT / cutout["local_path"]
    provenance_path = cube_path.parent / "ms2137_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    spectral_grid_amendment = config["protocol_amendments"][0]
    size_bytes = cube_path.stat().st_size
    digest = sha256(cube_path)

    with fits.open(cube_path, mode="readonly", memmap=True, lazy_load_hdus=True) as hdul:
        primary = hdul[0].header
        extension_names = [hdu.name for hdu in hdul]
        data_header = hdul["DATA"].header
        stat_header = hdul["STAT"].header
        data_shape = tuple(reversed([data_header[f"NAXIS{axis}"] for axis in range(1, 4)]))
        stat_shape = tuple(reversed([stat_header[f"NAXIS{axis}"] for axis in range(1, 4)]))
        spectral_step_angstrom = float(data_header["CD3_3"])
        wavelength_first_angstrom = float(
            data_header["CRVAL3"] + (1.0 - data_header["CRPIX3"]) * spectral_step_angstrom
        )
        wavelength_last_angstrom = float(
            data_header["CRVAL3"] + (data_header["NAXIS3"] - data_header["CRPIX3"]) * spectral_step_angstrom
        )
        spatial_scale_x_arcsec = abs(float(data_header["CD1_1"])) * 3600.0
        spatial_scale_y_arcsec = abs(float(data_header["CD2_2"])) * 3600.0
        celestial_wcs = WCS(data_header).celestial
        bcg = SkyCoord(
            config["published_bcg_center"]["ra_deg"] * u.deg,
            config["published_bcg_center"]["dec_deg"] * u.deg,
        )
        bcg_x, bcg_y = celestial_wcs.world_to_pixel(bcg)
        primary_metadata = {
            "object": primary.get("OBJECT"),
            "program_id": primary.get("PROG_ID"),
            "exposure_seconds": float(primary.get("EXPTIME")),
            "mjd_obs": float(primary.get("MJD-OBS")),
            "origin": primary.get("ORIGIN"),
        }
        data_metadata = {
            "shape_spectral_y_x": list(data_shape),
            "bitpix": int(data_header["BITPIX"]),
            "bunit": data_header.get("BUNIT"),
            "ctype": [data_header.get(f"CTYPE{axis}") for axis in range(1, 4)],
            "spectral_step_angstrom": spectral_step_angstrom,
            "wavelength_first_angstrom": wavelength_first_angstrom,
            "wavelength_last_angstrom": wavelength_last_angstrom,
            "spatial_scale_x_arcsec": spatial_scale_x_arcsec,
            "spatial_scale_y_arcsec": spatial_scale_y_arcsec,
            "bcg_pixel_zero_indexed_x_y": [float(bcg_x), float(bcg_y)],
        }
        stat_metadata = {
            "shape_spectral_y_x": list(stat_shape),
            "bitpix": int(stat_header["BITPIX"]),
            "bunit": stat_header.get("BUNIT"),
        }

    gates = {
        "prior_metadata_gate_passed": bool(feasibility["gates"]["metadata_feasibility_gate_passed"]),
        "exact_frozen_cutout_exists": cube_path.is_file(),
        "provenance_matches_frozen_request": bool(
            provenance["source_dataset"] == archive["dp_id"]
            and provenance["proposal_id"] == archive["proposal_id"]
            and provenance["local_path"] == cutout["local_path"]
            and np.allclose(provenance["circle"], [cutout["center_ra_deg"], cutout["center_dec_deg"], cutout["radius_deg"]], rtol=0.0, atol=1.0e-12)
            and np.allclose(provenance["band_m"], [cutout["wavelength_min_m"], cutout["wavelength_max_m"]], rtol=0.0, atol=1.0e-14)
            and provenance["pixel_arrays_inspected"] is False
        ),
        "sha256_and_size_match_provenance": bool(
            provenance["sha256"].upper() == digest
            and int(provenance["size_bytes"]) == size_bytes
            and size_bytes > 400_000_000
        ),
        "primary_archive_identity_passed": bool(
            primary_metadata["object"] == archive["target_name"]
            and primary_metadata["program_id"] == archive["proposal_id"]
            and abs(primary_metadata["exposure_seconds"] - archive["archive_exposure_seconds"]) < 0.001
        ),
        "data_and_stat_extensions_present": extension_names == ["PRIMARY", "DATA", "STAT"],
        "data_and_stat_shapes_match": data_shape == stat_shape == (1841, 180, 180),
        "data_and_stat_float32_headers_passed": data_metadata["bitpix"] == stat_metadata["bitpix"] == -32,
        "variance_unit_matches_flux_squared": stat_metadata["bunit"] == f"({data_metadata['bunit']})**2",
        "wcs_axes_and_sampling_passed": bool(
            data_metadata["ctype"] == ["RA---TAN", "DEC--TAN", "AWAV"]
            and abs(spatial_scale_x_arcsec - 0.2) < 1.0e-8
            and abs(spatial_scale_y_arcsec - 0.2) < 1.0e-8
            and abs(spectral_step_angstrom - 1.25) < 1.0e-8
        ),
        "requested_wavelength_span_passed": bool(
            abs(wavelength_first_angstrom - cutout["wavelength_min_m"] * 1.0e10)
            <= spectral_grid_amendment["maximum_endpoint_offset_native_pixels"] * spectral_step_angstrom
            and abs(wavelength_last_angstrom - cutout["wavelength_max_m"] * 1.0e10)
            <= spectral_grid_amendment["maximum_endpoint_offset_native_pixels"] * spectral_step_angstrom
        ),
        "published_bcg_center_inside_cutout": bool(
            0.0 <= bcg_x < data_shape[2] and 0.0 <= bcg_y < data_shape[1]
        ),
        "science_arrays_not_inspected": True,
    }
    acquisition_gate = all(gates.values())
    report = {
        "report_version": "R1B2-MS2137-MUSE-acquisition-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "system": config["system"],
        "cube_path": cutout["local_path"],
        "size_bytes": size_bytes,
        "sha256": digest,
        "pixel_arrays_inspected": False,
        "protocol_amendments": config["protocol_amendments"],
        "primary_metadata": primary_metadata,
        "extension_names": extension_names,
        "data_metadata": data_metadata,
        "stat_metadata": stat_metadata,
        "gates": {**gates, "acquisition_header_gate_passed": acquisition_gate},
        "decision": "authorize_numerical_protocol_freeze" if acquisition_gate else "stop_MS2137_acquisition_failure",
        "next_action": (
            "Freeze the MS2137 reduction, spatial extraction, pPXF, covariance, systematic, and binary advancement gates before reading a science or variance array."
            if acquisition_gate else
            "Record the exact acquisition/header failure; do not inspect arrays or alter the cutout request."
        ),
        "authorization": {
            "freeze_numerical_protocol": acquisition_gate,
            "inspect_science_pixels": False,
            "extract_stellar_kinematics": False,
            "infer_dynamical_or_weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
