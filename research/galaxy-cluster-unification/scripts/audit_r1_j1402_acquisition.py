#!/usr/bin/env python3
"""Verify the complete locked J1402 receipt before any model replay."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "r1_j1402_acquisition_replay_jacobian_protocol.json"
MANIFEST_PATH = ROOT / "data" / "raw" / "r1_j1402" / "acquisition_manifest.json"
REPORT_PATH = ROOT / "results" / "r1_j1402_acquisition" / "report.json"
INVENTORY_PATH = ROOT / "data" / "derived" / "r1_j1402_acquisition_inventory.csv"
CHUNK_BYTES = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def h5_summary(path: Path) -> dict:
    datasets: list[dict] = []
    groups: list[str] = []
    with h5py.File(path, "r") as handle:
        top_level_keys = list(handle.keys())

        def visitor(name: str, item) -> None:
            if isinstance(item, h5py.Dataset):
                datasets.append(
                    {
                        "path": name,
                        "shape": [int(value) for value in item.shape],
                        "dtype": str(item.dtype),
                    }
                )
            elif isinstance(item, h5py.Group):
                groups.append(name)

        handle.visititems(visitor)
    return {
        "top_level_keys": top_level_keys,
        "group_count": len(groups),
        "dataset_count": len(datasets),
        "datasets": datasets[:200],
        "dataset_inventory_truncated": len(datasets) > 200,
    }


def npy_header(path: Path) -> dict:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
    return {
        "format_version": list(version),
        "shape": [int(value) for value in shape],
        "fortran_order": bool(fortran_order),
        "dtype": str(dtype),
        "object_or_pickle_payload_not_loaded": bool(dtype.hasobject),
    }


def safe_header_value(header, key: str):
    value = header.get(key)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    protocol_digest = sha256(CONFIG_PATH)
    receipts = manifest["receipts"]
    rows: list[dict] = []
    checksum_failures: list[str] = []
    h5_products: dict[str, dict] = {}
    npy_products: dict[str, dict] = {}
    kcwi_headers: list[dict] = []

    for index, receipt in enumerate(receipts, start=1):
        path = ROOT / receipt["relative_path"]
        observed_bytes = path.stat().st_size if path.exists() else -1
        observed_sha256 = sha256(path) if path.exists() else None
        checksum_pass = bool(
            path.exists()
            and observed_bytes == int(receipt["bytes"])
            and observed_sha256 == receipt["sha256"]
        )
        if not checksum_pass:
            checksum_failures.append(receipt["identity"])
        row = {
            "receipt_index": index,
            "group": receipt["group"],
            "identity": receipt["identity"],
            "relative_path": receipt["relative_path"],
            "bytes": observed_bytes,
            "sha256": observed_sha256,
            "checksum_pass": checksum_pass,
            "content_type": receipt.get("content_type"),
            "object": None,
            "imtype": None,
            "exposure_seconds": None,
            "program": None,
            "camera": None,
            "grating": None,
            "filter": None,
            "slicer": None,
            "binning": None,
            "ampmode": None,
            "ccdspeed": None,
        }

        if path.suffix.lower() == ".h5":
            h5_products[receipt["identity"]] = h5_summary(path)
        elif path.suffix.lower() == ".npy":
            npy_products[receipt["identity"]] = npy_header(path)
        elif path.suffix.lower() == ".fits":
            header = fits.getheader(path, 0, memmap=False)
            header_row = {
                "koaid": safe_header_value(header, "KOAID"),
                "object": safe_header_value(header, "OBJECT"),
                "targname": safe_header_value(header, "TARGNAME"),
                "imtype": safe_header_value(header, "IMTYPE"),
                "koaimtyp": safe_header_value(header, "KOAIMTYP"),
                "exposure_seconds": safe_header_value(header, "ELAPTIME"),
                "program": safe_header_value(header, "PROGID"),
                "camera": safe_header_value(header, "CAMERA"),
                "grating": safe_header_value(header, "BGRATNAM"),
                "filter": safe_header_value(header, "BFILTNAM"),
                "slicer": safe_header_value(header, "IFUNAM"),
                "binning": safe_header_value(header, "BINNING"),
                "ampmode": safe_header_value(header, "AMPMODE"),
                "ccdspeed": safe_header_value(header, "CCDSPEED"),
                "waveblue": safe_header_value(header, "WAVEBLUE"),
                "wavered": safe_header_value(header, "WAVERED"),
            }
            kcwi_headers.append({"group": receipt["group"], **header_row})
            row.update({key: header_row[key] for key in row if key in header_row})
        rows.append(row)

    config_path = (
        ROOT
        / "data/raw/r1_j1402/dinos_repo/2_dolphin_modelling/settings/SDSSJ1402+6321_config.yml"
    )
    dinos_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bands = dinos_config["band"]
    mask = dinos_config["mask"]
    matrices = [np.asarray(item, dtype=float) for item in mask["transform_matrix"]]
    pixel_scales = [float(math.sqrt(abs(np.linalg.det(item)))) for item in matrices]
    matrix_determinants = [float(np.linalg.det(item)) for item in matrices]

    science = [item for item in kcwi_headers if item["group"] == "kcwi_science"]
    expected_setup = protocol["acquisition"]["KCWI"]["setup"]
    science_identity_pass = all(item["koaid"] in protocol["acquisition"]["KCWI"]["science_ids"] for item in science)
    science_exposure_pass = len(science) == 4 and all(float(item["exposure_seconds"]) == 1800.0 for item in science)
    science_setup_pass = len(science) == 4 and all(
        item["camera"] == expected_setup["camera"]
        and item["grating"] == expected_setup["grating"]
        and item["filter"] == expected_setup["filter"]
        and item["slicer"] == expected_setup["slicer"]
        and item["binning"] == expected_setup["binning"]
        and item["ampmode"] == expected_setup["ampmode"]
        and str(item["ccdspeed"]) == expected_setup["ccdspeed"]
        for item in science
    )
    h5_open_pass = len(h5_products) == 8 and all(
        item["dataset_count"] > 0 for item in h5_products.values()
    )
    coordinate_declaration_pass = bool(
        bands == ["F435W", "F555W", "F814W"]
        and mask["size"] == [120, 140, 140]
        and len(mask["ra_at_xy_0"]) == 3
        and all(abs(value) > 1e-8 for value in matrix_determinants)
    )
    group_counts = dict(sorted(Counter(item["group"] for item in receipts).items()))
    checks = {
        "protocol_checksum_matches_manifest": manifest["protocol_sha256"] == protocol_digest,
        "manifest_declares_complete": bool(manifest["complete"]),
        "exact_receipt_count_46": len(receipts) == manifest["receipt_count"] == manifest["planned_file_count"] == 46,
        "all_receipt_checksums_recomputed": not checksum_failures,
        "exact_group_counts": group_counts
        == {
            "dinos_full_output": 1,
            "dinos_github": 9,
            "kcwi_arc": 2,
            "kcwi_bias": 7,
            "kcwi_continuum_bar": 1,
            "kcwi_flat": 17,
            "kcwi_science": 4,
            "kcwi_standard_star": 5,
        },
        "all_eight_HDF5_files_open_and_contain_datasets": h5_open_pass,
        "numpy_header_audited_without_loading_external_payload": len(npy_products) == 1,
        "Dinos_three_band_coordinate_declaration_is_nondegenerate": coordinate_declaration_pass,
        "KCWI_science_identity_matches_frozen_list": science_identity_pass,
        "KCWI_four_times_1800_seconds": science_exposure_pass,
        "KCWI_science_setup_matches_frozen_setup": science_setup_pass,
    }
    gate_pass = all(checks.values())
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol["protocol_version"],
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
        "science_arrays_fitted": False,
        "summary": {
            "receipt_count": len(receipts),
            "verified_bytes": sum(int(item["bytes"]) for item in receipts),
            "group_counts": group_counts,
            "checksum_failures": checksum_failures,
        },
        "Dinos": {
            "bands": bands,
            "image_sizes_pixels": mask["size"],
            "ra_at_xy_0_arcsec": mask["ra_at_xy_0"],
            "transform_matrices": mask["transform_matrix"],
            "transform_determinants": matrix_determinants,
            "transform_pixel_scales_arcsec": pixel_scales,
            "scalar_pixel_size_field": float(dinos_config["pixel_size"]),
            "coordinate_note": "The per-band transform matrices imply about 0.05 arcsec/pixel while the separate scalar pixel_size field is 0.04. Replay must use the explicit matrices and demonstrate the coordinate mapping; the scalar field is not silently substituted.",
            "lens_models": dinos_config["model"]["lens"],
            "source_light_models": dinos_config["model"]["source_light"],
            "lens_light_models": dinos_config["model"]["lens_light"],
            "shapelet_orders": dinos_config["source_light_option"]["n_max"],
            "HDF5_products": h5_products,
            "numpy_products": npy_products,
        },
        "KCWI": {
            "science_headers": science,
            "all_header_rows": kcwi_headers,
            "calibration_readout_note": protocol["acquisition"]["KCWI"]["calibration_compatibility"],
        },
        "checks": checks,
        "gate_pass": gate_pass,
        "decision": "acquisition_integrity_gate_pass_proceed_to_Dinos_coordinate_audit"
        if gate_pass
        else "stop_J1402_acquisition_integrity_failure",
        "authorization": {
            "inspect_locked_Dinos_arrays_and_configuration": gate_pass,
            "replay_Dinos_likelihood": False,
            "fit_lens_response": False,
            "reduce_KCWI": False,
            "count_toward_ten_system_target": False,
            "infer_gravity_response": False,
            "authorize_R2": False,
        },
    }

    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INVENTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(report["decision"])


if __name__ == "__main__":
    main()
