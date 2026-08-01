#!/usr/bin/env python3
"""Audit the frozen RX J2129 XMM+HST next-stage inputs without pixel access."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.heasarc import Heasarc


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_rxj2129_strict_observable_next_stage.json"
REPORT = ROOT / "results/r1_rxj2129_strict_observable_feasibility/report.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    return value


def build_report() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    center = SkyCoord(config["center_ra_deg"] * u.deg, config["center_dec_deg"] * u.deg)
    heasarc = Heasarc()
    table = heasarc.query_region(center, catalog=config["xmm_metadata_gate"]["catalog"], radius=12 * u.arcmin)
    located = heasarc.locate_data(table)
    expected = config["xmm_metadata_gate"]["required_obsid"]
    selected = [row for row in table if str(row["obsid"]) == expected]
    xmm_unique = len(table) == 1 and len(selected) == 1 and len(located) == 1
    row = selected[0]
    xgate = config["xmm_metadata_gate"]
    xmm_gate = bool(
        xmm_unique
        and str(row["status"]) == xgate["required_status"]
        and int(row["duration"]) >= xgate["minimum_gross_duration_seconds"]
        and str(row["data_in_heasarc"]) == xgate["required_data_in_heasarc"]
        and int(located[0]["content_length"]) <= xgate["maximum_located_data_bytes"]
    )

    hst_rows = []
    hst_gate = True
    for relative in config["hst_astrometric_covariance_gate"]["local_images"]:
        path = ROOT / relative
        with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
            header = hdul[0].header
            ext_header = hdul[1].header if len(hdul) > 1 else header
        finite_shape = int(ext_header.get("NAXIS1", 0)) > 0 and int(ext_header.get("NAXIS2", 0)) > 0
        passed = bool(path.exists() and path.stat().st_size > 0 and finite_shape)
        hst_gate &= passed
        hst_rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "instrument": header.get("INSTRUME"),
            "filter": header.get("FILTER") or ext_header.get("FILTER"),
            "shape": [int(ext_header.get("NAXIS2", 0)), int(ext_header.get("NAXIS1", 0))],
            "passed": passed,
        })

    ledger = pd.read_csv(ROOT / config["hst_astrometric_covariance_gate"]["source_ledger"])
    spec = ledger.loc[ledger["likelihood_included"].eq(True)]
    inner = spec.loc[spec["image_id"].astype(str).isin(config["hst_astrometric_covariance_gate"]["inner_images_required"])]
    ledger_gate = bool(
        len(spec) == config["hst_astrometric_covariance_gate"]["spectroscopic_images_required"]
        and spec["source_family"].nunique() == config["hst_astrometric_covariance_gate"]["spectroscopic_families_required"]
        and set(inner["image_id"].astype(str)) == set(config["hst_astrometric_covariance_gate"]["inner_images_required"])
        and inner["source_family"].nunique() == 3
    )
    gate = bool(xmm_gate and hst_gate and ledger_gate)
    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "XMM_pixels_downloaded_or_inspected": False,
        "HST_arc_pixels_measured": False,
        "xmm_metadata": {key: scalar(row[key]) for key in table.colnames},
        "xmm_located_data": {key: scalar(located[0][key]) for key in located.colnames},
        "hst_header_audit": hst_rows,
        "lens_ledger": {
            "spectroscopic_images": int(len(spec)),
            "spectroscopic_families": int(spec["source_family"].nunique()),
            "inner_image_ids": sorted(inner["image_id"].astype(str).tolist()),
            "inner_families": int(inner["source_family"].nunique()),
        },
        "gates": {
            "one_exact_public_XMM_observation_passed": xmm_unique,
            "XMM_duration_status_and_size_passed": xmm_gate,
            "two_local_HST_header_and_checksum_inputs_passed": hst_gate,
            "spectroscopic_lens_coordinate_ledger_passed": ledger_gate,
            "R1B3_P1_feasibility_gate_passed": gate,
        },
        "decision": "authorize_exact_XMM_acquisition_and_separate_HST_centroid_execution_freeze" if gate else "stop_RXJ2129_R1B3_feasibility_failure",
        "next_action": "Download only XMM ObsID 0093030201 with provenance; inventory ODF/PPS and freeze exact SAS/CCF, flare, background, spectral-annulus, and HST centroid-execution protocols before pixel access." if gate else "Preserve the failed metadata gate and run the same availability screen on MACS J1206.",
        "authorization": {
            "download_exact_XMM_observation": gate,
            "inspect_XMM_pixels": False,
            "measure_HST_arc_pixels": False,
            "infer_gas_profile": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
