#!/usr/bin/env python3
"""Audit the immutable RX J2129 point-source mask and advance only to X2b2."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DERIVED = PROJECT / "data/derived/r1_rxj2129_xmm_x2"
MASK_MANIFEST_PATH = DERIVED / "point_source_mask_manifest.json"
REDUCTION_MANIFEST_PATH = PROJECT / "data/derived/r1_rxj2129_xmm_reduction_manifest.json"
REPORT_PATH = PROJECT / "results/r1_rxj2129_xmm_event_processing/report.json"


def main() -> None:
    mask_manifest = json.loads(MASK_MANIFEST_PATH.read_text())
    reduction_manifest = json.loads(REDUCTION_MANIFEST_PATH.read_text())
    catalog_path = PROJECT / mask_manifest["final_catalog"]
    region_path = PROJECT / mask_manifest["region_mask"]
    ledger_path = PROJECT / mask_manifest["PSF"]["ledger"]
    with catalog_path.open() as handle:
        catalog = list(csv.DictReader(handle))
    with ledger_path.open() as handle:
        psf_rows = list(csv.DictReader(handle))
    region_circles = [
        line
        for line in region_path.read_text().splitlines()
        if line.startswith("-circle(")
    ]
    catalog_gate = (
        len(catalog) == mask_manifest["PSF"]["source_count"]
        and len(region_circles) == len(catalog)
        and all(row["PSF_mask_status"] == "frozen" for row in catalog)
        and all(
            15.0 <= float(row["mask_radius_arcsec"]) <= 60.0
            and math.isfinite(float(row["maximum_local_r80_arcsec"]))
            for row in catalog
        )
    )
    psf_gate = (
        len(psf_rows) == 9 * len(catalog)
        and all(
            math.isfinite(float(row["r80_arcsec"]))
            and float(row["r80_arcsec"]) > 0
            for row in psf_rows
        )
    )
    x2b1_pass = (
        mask_manifest["gates"]["all_three_emldetect_catalog_gates_passed"]
        and mask_manifest["gates"]["frozen_catalog_filter_and_merge_completed"]
        and catalog_gate
        and psf_gate
    )
    generated = datetime.now(timezone.utc).isoformat()
    reduction_manifest["generated_utc"] = generated
    reduction_manifest["X2b1_point_source_mask"] = {
        "manifest": str(MASK_MANIFEST_PATH.relative_to(PROJECT)),
        "catalog": mask_manifest["final_catalog"],
        "region_mask": mask_manifest["region_mask"],
        "source_count": len(catalog),
        "PSF_evaluations": len(psf_rows),
        "mask_radius_arcsec_range": mask_manifest["mask_radius_arcsec_range"],
        "manual_edits": False,
        "gate_passed": x2b1_pass,
    }
    reduction_manifest["gates"]["R1B3_XMM_X2b1_point_source_mask_gate_passed"] = x2b1_pass
    reduction_manifest["gates"]["R1B3_XMM_X2_flare_background_gate_passed"] = False
    reduction_manifest["gates"]["R1B3_XMM_X3_gas_likelihood_gate_passed"] = False
    REDUCTION_MANIFEST_PATH.write_text(json.dumps(reduction_manifest, indent=2) + "\n")

    report = {
        "report_version": "R1B3-RXJ2129-XMM-event-processing-X2b1-0.3",
        "generated_utc": generated,
        "stage": "X2b1_immutable_point_source_mask",
        "status": "pass" if x2b1_pass else "fail",
        "outcome": (
            f"X2b1 passed: {len(catalog)} merged sources and {len(psf_rows)} calibrated PSF evaluations define one immutable mask."
            if x2b1_pass
            else "X2b1 failed; background tasks remain unauthorized."
        ),
        "source_count": len(catalog),
        "PSF_evaluations": len(psf_rows),
        "mask_radius_arcsec_range": mask_manifest["mask_radius_arcsec_range"],
        "gates": reduction_manifest["gates"],
        "authorization": {
            "run_frozen_X2b2_detector_corner_FWC_background_audit": x2b1_pass,
            "claim_full_X2_pass": False,
            "fit_temperature_or_density": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not x2b1_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
