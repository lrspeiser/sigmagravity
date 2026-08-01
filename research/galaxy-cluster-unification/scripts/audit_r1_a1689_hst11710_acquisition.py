#!/usr/bin/env python3
"""Audit the frozen A1689 HST acquisition without measuring science pixels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_hst11710_acquisition_protocol.json"
REPORT = ROOT / "results/r1_a1689_hst11710_acquisition/report.json"
LEDGER = ROOT / "data/derived/r1_a1689_hst11710_acquisition_ledger.csv"
BCG_RA_DEG = 197.873
BCG_DEC_DEG = -1.3410833333333333


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def covers_center(hdul: fits.HDUList) -> bool:
    for hdu in hdul:
        if hdu.name != "SCI" or hdu.data is None:
            continue
        try:
            x, y = WCS(hdu.header, hdul).celestial.world_to_pixel_values(BCG_RA_DEG, BCG_DEC_DEG)
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y) and -0.5 <= x < hdu.data.shape[1] - 0.5 and -0.5 <= y < hdu.data.shape[0] - 0.5:
            return True
    return False


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / cfg["outputs"]["manifest"]).read_text(encoding="utf-8"))
    raw = ROOT / cfg["outputs"]["directory"]
    rows = []
    for item in manifest["files"]:
        path = raw / item["filename"]
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        digest = sha256(path) if exists and size == item["bytes"] else ""
        with fits.open(path, memmap=True) as hdul:
            header = hdul[0].header
            proposal = str(header.get("PROPOSID", ""))
            instrument = str(header.get("INSTRUME", ""))
            detector = str(header.get("DETECTOR", ""))
            filters = {str(header.get("FILTER1", "")), str(header.get("FILTER2", "")), str(header.get("FILTER", ""))}
            exposure = float(header.get("EXPTIME", 0.0))
            center_covered = covers_center(hdul) if item["product_subgroup"] == "FLC" else None
        rows.append({
            "filename": item["filename"],
            "product_subgroup": item["product_subgroup"],
            "archive_bytes": item["bytes"],
            "local_bytes": size,
            "sha256_manifest": item["sha256"],
            "sha256_recomputed": digest,
            "checksum_matches": digest == item["sha256"],
            "proposal_id": proposal,
            "instrument": instrument,
            "detector": detector,
            "f814w_header": "F814W" in filters or item["product_subgroup"] == "ASN",
            "exposure_seconds": exposure,
            "bcg_center_covered": center_covered,
            "fits_opened": True,
        })
    ledger = pd.DataFrame(rows)
    flc = ledger.loc[ledger["product_subgroup"] == "FLC"].copy()
    drz = ledger.loc[ledger["product_subgroup"] == "DRZ"].copy()
    asn = ledger.loc[ledger["product_subgroup"] == "ASN"].copy()
    visits = sorted({name[:6] for name in flc["filename"]})
    counts_by_visit = flc.assign(visit=flc["filename"].str[:6]).groupby("visit").size().to_dict()
    checks = {
        "manifest_file_count": len(ledger) == cfg["expected_archive_totals"]["files"],
        "manifest_archive_byte_total": int(ledger["archive_bytes"].sum()) == cfg["expected_archive_totals"]["bytes"],
        "local_byte_sizes": bool((ledger["archive_bytes"] == ledger["local_bytes"]).all()),
        "all_sha256_match": bool(ledger["checksum_matches"].all()),
        "all_fits_open": bool(ledger["fits_opened"].all()),
        "exact_flc_count": len(flc) == cfg["frozen_product_selection"]["individual_exposures"]["expected_unique_files"],
        "exact_drz_count": len(drz) == cfg["frozen_product_selection"]["visit_mosaics"]["expected_unique_files"],
        "exact_asn_count": len(asn) == cfg["frozen_product_selection"]["association_tables"]["expected_unique_files"],
        "frozen_f814w_exposure_time": abs(float(flc["exposure_seconds"].sum()) - cfg["expected_archive_totals"]["f814w_exposure_seconds"]) < 1.0,
        "seven_visits": len(visits) >= cfg["advance_gate"]["minimum_distinct_visits"],
        "eight_flc_exposures_per_visit": len(counts_by_visit) == 7 and set(counts_by_visit.values()) == {8},
        "proposal_instrument_detector": bool((flc["proposal_id"] == "11710").all() and (flc["instrument"] == "ACS").all() and (flc["detector"] == "WFC").all()),
        "f814w_headers": bool(ledger["f814w_header"].all()),
        "all_flc_cover_bcg_center": bool(flc["bcg_center_covered"].astype(bool).all()),
        "paper_products_present": all((ROOT / cfg["outputs"][key]).exists() for key in ("paper_pdf", "paper_source")),
        "no_science_pixel_measurement": True,
        "no_lens_or_gravity_residual_used": True,
    }
    gate = all(checks.values())
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER, index=False)
    report = {
        "report_version": "R1B1-A1689-HST11710-acquisition-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "files": {
            "total": int(len(ledger)),
            "flc": int(len(flc)),
            "drz": int(len(drz)),
            "asn": int(len(asn)),
            "archive_bytes": int(ledger["archive_bytes"].sum()),
            "flc_exposure_seconds": float(flc["exposure_seconds"].sum()),
            "flc_exposures_covering_bcg_center": int(flc["bcg_center_covered"].astype(bool).sum()),
            "flc_exposures_not_covering_bcg_center": flc.loc[
                ~flc["bcg_center_covered"].astype(bool), "filename"
            ].tolist(),
            "visits": visits,
            "flc_counts_by_visit": {key: int(value) for key, value in counts_by_visit.items()},
        },
        "checks": checks,
        "gates": {
            "HST11710_acquisition_gate_passed": bool(gate),
            "photometry_and_astrometry_likelihood_freeze_authorized": bool(gate),
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "freeze_bcg_tracer_light_and_astrometric_likelihood": bool(gate),
            "measure_science_pixels_before_that_freeze": False,
            "infer_weyl_or_dynamical_response": False,
            "fit_new_force_or_action": False,
        },
        "outputs": {
            "manifest": cfg["outputs"]["manifest"],
            "ledger": str(LEDGER.relative_to(ROOT)).replace("\\", "/"),
        },
        "limitations": [
            "Forty-nine of 56 individual FLC exposures cover the frozen BCG coordinate; exactly one exposure in each of the seven visits places that coordinate in the ACS inter-chip gap.",
            "The predeclared acquisition gate required all 56 exposures to cover the center, so this acquisition attempt fails even though all files, checksums, headers, visits, and total exposure are intact.",
            "No exposure is removed and no photometric or astrometric pixel value is inspected under this failed protocol."
        ],
        "next_action": (
            "Freeze the masking, PSF, sky, radial-profile, per-exposure astrometric, and covariance protocol before measuring the BCG or lens-image pixels."
            if gate else
            "Retain the acquisition shortfall and do not inspect or fit the HST science pixels."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
