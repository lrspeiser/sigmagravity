from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_point_source_catalog_and_immutable_mask_pass() -> None:
    manifest = json.loads(
        (ROOT / "data/derived/r1_rxj2129_xmm_x2/point_source_mask_manifest.json").read_text()
    )
    with (ROOT / "data/derived/r1_rxj2129_xmm_x2/point_source_catalog.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert manifest["gates"]["all_three_emldetect_catalog_gates_passed"] is True
    assert manifest["gates"]["frozen_catalog_filter_and_merge_completed"] is True
    assert manifest["gates"]["all_PSF_radii_completed"] is True
    assert manifest["gates"]["immutable_point_source_mask_frozen"] is True
    assert manifest["gates"]["X2b1_gate_passed"] is True
    assert manifest["gates"]["full_X2_gate_passed"] is False
    assert rows
    assert all(float(row["maximum_detection_likelihood"]) >= 10.0 for row in rows)
    assert all(float(row["maximum_fitted_extent_arcsec"]) <= 6.0 for row in rows)
    assert all(row["PSF_mask_status"] == "frozen" for row in rows)
    assert all(15.0 <= float(row["mask_radius_arcsec"]) <= 60.0 for row in rows)
    assert manifest["PSF"]["evaluations"] == 9 * len(rows)
