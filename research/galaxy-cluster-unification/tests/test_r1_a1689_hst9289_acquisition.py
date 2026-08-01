from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_hst9289_gate_is_frozen_check_conjunction() -> None:
    report = json.loads((ROOT / "results/r1_a1689_hst9289_acquisition/report.json").read_text())
    expected = all(report["checks"].values())
    assert report["files"]["total"] == 66
    assert report["files"]["flc"] == 40
    assert report["files"]["drc"] == 6
    assert report["files"]["asn"] == 20
    assert len(report["geometry"]["lens_images"]) == 6
    assert report["gates"]["HST9289_geometry_acquisition_gate_passed"] is expected
    assert report["gates"]["photometry_astrometry_covariance_protocol_freeze_authorized"] is expected
    assert report["authorization"]["freeze_photometry_astrometry_and_covariance_protocol"] is expected
    assert report["gates"]["gravity_response_fit_authorized"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False


def test_a1689_hst9289_ledgers_are_complete() -> None:
    ledger = pd.read_csv(ROOT / "data/derived/r1_a1689_hst9289_acquisition_ledger.csv")
    lens = pd.read_csv(ROOT / "data/derived/r1_a1689_hst9289_lens_coverage_ledger.csv")
    assert len(ledger) == 66
    assert set(ledger["product_subgroup"]) == {"FLC", "DRC", "ASN"}
    assert ledger["checksum_matches"].all()
    assert len(lens) == 6
