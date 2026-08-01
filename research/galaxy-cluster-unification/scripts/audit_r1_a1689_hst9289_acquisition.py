#!/usr/bin/env python3
"""Audit frozen A1689 HST-9289 headers and WCS footprints only."""

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
CONFIG = ROOT / "configs/r1_a1689_hst9289_acquisition_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def covers(hdul: fits.HDUList, ra_deg: float, dec_deg: float) -> bool:
    for hdu in hdul:
        if hdu.name != "SCI" or hdu.data is None:
            continue
        try:
            x, y = WCS(hdu.header, hdul).celestial.world_to_pixel_values(ra_deg, dec_deg)
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y) and -0.5 <= x < hdu.data.shape[1] - 0.5 and -0.5 <= y < hdu.data.shape[0] - 0.5:
            return True
    return False


def science_filter(header: fits.Header) -> str:
    values = {str(header.get(key, "")).upper() for key in ("FILTER", "FILTER1", "FILTER2")}
    for name in ("F475W", "F625W", "F775W", "F850LP"):
        if name in values:
            return name
    return ""


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw = ROOT / cfg["outputs"]["directory"]
    manifest = json.loads((ROOT / cfg["outputs"]["manifest"]).read_text(encoding="utf-8"))
    lens = pd.read_csv(ROOT / cfg["frozen_geometry_gate"]["lens_image_source"])
    selected_lens = lens.loc[
        lens["family_independent_redshift_anchor"].astype(bool)
        & lens["inside_frozen_dynamics_support"].astype(bool)
    ].sort_values("radius_kpc")
    if len(selected_lens) != cfg["frozen_geometry_gate"]["expected_selected_lens_images"]:
        raise RuntimeError("Frozen six-image lens geometry selection changed")

    rows = []
    for item in manifest["files"]:
        path = raw / item["filename"]
        size = path.stat().st_size if path.exists() else 0
        digest = sha256(path) if size == item["bytes"] else ""
        with fits.open(path, memmap=True) as hdul:
            header = hdul[0].header
            subgroup = item["product_subgroup"]
            row = {
                "filename": item["filename"],
                "product_subgroup": subgroup,
                "archive_bytes": item["bytes"],
                "local_bytes": size,
                "sha256_manifest": item["sha256"],
                "sha256_recomputed": digest,
                "checksum_matches": digest == item["sha256"],
                "proposal_id": str(header.get("PROPOSID", "")),
                "instrument": str(header.get("INSTRUME", "")),
                "detector": str(header.get("DETECTOR", "")),
                "filter": science_filter(header),
                "exposure_seconds": float(header.get("EXPTIME", 0.0)),
                "bcg_center_covered": None,
                "fits_opened": True,
            }
            if subgroup == "FLC":
                center = cfg["frozen_geometry_gate"]
                row["bcg_center_covered"] = covers(hdul, center["bcg_center_ra_deg"], center["bcg_center_dec_deg"])
                for image in selected_lens.itertuples(index=False):
                    row[f"covers_{image.image_id}"] = covers(hdul, image.ra_deg, image.dec_deg)
        rows.append(row)
    ledger = pd.DataFrame(rows)
    flc = ledger.loc[ledger["product_subgroup"] == "FLC"].copy()
    drc = ledger.loc[ledger["product_subgroup"] == "DRC"].copy()
    asn = ledger.loc[ledger["product_subgroup"] == "ASN"].copy()
    filters = cfg["selection_basis"]["filters"]
    exposure_by_filter = flc.groupby("filter")["exposure_seconds"].sum().to_dict()
    center_by_filter = flc.loc[flc["bcg_center_covered"].astype(bool)].groupby("filter").size().to_dict()
    lens_rows = []
    for image in selected_lens.itertuples(index=False):
        covered = flc.loc[flc[f"covers_{image.image_id}"].astype(bool)]
        lens_rows.append({
            "image_id": image.image_id,
            "family_id": image.family_id,
            "radius_kpc": image.radius_kpc,
            "covering_flc_exposures": int(len(covered)),
            "covering_filters": int(covered["filter"].nunique()),
            "filters": " ".join(sorted(covered["filter"].unique())),
        })
    lens_coverage = pd.DataFrame(lens_rows)
    geometry = cfg["frozen_geometry_gate"]
    expected_exposure = cfg["expected_archive_totals"]["exposure_seconds_by_filter"]
    checks = {
        "manifest_file_count": len(ledger) == cfg["expected_archive_totals"]["files"],
        "manifest_archive_byte_total": int(ledger["archive_bytes"].sum()) == cfg["expected_archive_totals"]["bytes"],
        "local_byte_sizes": bool((ledger["archive_bytes"] == ledger["local_bytes"]).all()),
        "all_sha256_match": bool(ledger["checksum_matches"].all()),
        "all_fits_open": bool(ledger["fits_opened"].all()),
        "exact_flc_count": len(flc) == cfg["frozen_product_selection"]["individual_exposures"]["expected_unique_files"],
        "exact_drc_count": len(drc) == cfg["frozen_product_selection"]["aggregate_filter_visit_mosaics"]["expected_unique_files"],
        "exact_asn_count": len(asn) == cfg["frozen_product_selection"]["association_tables"]["expected_unique_files"],
        "four_science_filters": set(flc["filter"]) == set(filters),
        "frozen_exposure_by_filter": all(abs(exposure_by_filter.get(name, 0.0) - expected_exposure[name]) < 1.0 for name in filters),
        "proposal_instrument_detector": bool((flc["proposal_id"] == "9289").all() and (flc["instrument"] == "ACS").all() and (flc["detector"] == "WFC").all()),
        "bcg_total_coverage_fraction": float(flc["bcg_center_covered"].astype(bool).mean()) >= geometry["minimum_fraction_of_all_flc_exposures_covering_bcg_center"],
        "bcg_minimum_coverage_each_filter": all(center_by_filter.get(name, 0) >= geometry["minimum_flc_exposures_per_filter_covering_bcg_center"] for name in filters),
        "each_lens_image_minimum_exposure_coverage": bool((lens_coverage["covering_flc_exposures"] >= geometry["minimum_flc_exposures_covering_each_lens_image"]).all()),
        "each_lens_image_minimum_filter_coverage": bool((lens_coverage["covering_filters"] >= geometry["minimum_distinct_filters_covering_each_lens_image"]).all()),
        "no_science_pixel_measurement": True,
        "no_lens_or_gravity_residual_used": True,
    }
    gate = all(checks.values())
    ledger_path = ROOT / cfg["outputs"]["audit_ledger"]
    lens_path = ROOT / cfg["outputs"]["lens_coverage_ledger"]
    report_path = ROOT / cfg["outputs"]["audit_report"]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ledger_path, index=False)
    lens_coverage.to_csv(lens_path, index=False)
    report = {
        "report_version": "R1B1-A1689-HST9289-acquisition-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "files": {
            "total": int(len(ledger)), "flc": int(len(flc)), "drc": int(len(drc)), "asn": int(len(asn)),
            "archive_bytes": int(ledger["archive_bytes"].sum()),
            "flc_exposure_seconds_by_filter": {name: float(exposure_by_filter.get(name, 0.0)) for name in filters},
        },
        "geometry": {
            "flc_exposures_covering_bcg_center": int(flc["bcg_center_covered"].astype(bool).sum()),
            "flc_total": int(len(flc)),
            "bcg_center_coverage_fraction": float(flc["bcg_center_covered"].astype(bool).mean()),
            "bcg_covering_exposures_by_filter": {name: int(center_by_filter.get(name, 0)) for name in filters},
            "lens_images": lens_coverage.to_dict(orient="records"),
        },
        "checks": checks,
        "gates": {
            "HST9289_geometry_acquisition_gate_passed": bool(gate),
            "photometry_astrometry_covariance_protocol_freeze_authorized": bool(gate),
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "freeze_photometry_astrometry_and_covariance_protocol": bool(gate),
            "measure_science_pixels_before_that_freeze": False,
            "infer_weyl_or_dynamical_response": False,
            "fit_new_force_or_action": False,
        },
        "outputs": {
            "manifest": cfg["outputs"]["manifest"],
            "audit_ledger": cfg["outputs"]["audit_ledger"],
            "lens_coverage_ledger": cfg["outputs"]["lens_coverage_ledger"],
        },
        "next_action": (
            "Freeze the common-grid, empirical-PSF, source-mask, sky, BCG-profile, per-exposure lens-centroid, and covariance model before reading the selected science pixels."
            if gate else
            "Retain the second HST acquisition shortfall and do not fit photometry or lens positions under this route."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
