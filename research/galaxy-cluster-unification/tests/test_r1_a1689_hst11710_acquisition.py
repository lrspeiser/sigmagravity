from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_hst11710_acquisition_gate() -> None:
    report = json.loads((ROOT / "results/r1_a1689_hst11710_acquisition/report.json").read_text())
    assert report["files"]["total"] == 70
    assert report["files"]["flc"] == 56
    assert report["files"]["drz"] == 7
    assert report["files"]["asn"] == 7
    assert report["files"]["flc_exposures_covering_bcg_center"] == 49
    assert len(report["files"]["flc_exposures_not_covering_bcg_center"]) == 7
    assert report["files"]["flc_counts_by_visit"] == {f"jb2g0{i}": 8 for i in range(1, 8)}
    assert report["checks"]["all_flc_cover_bcg_center"] is False
    assert all(value for key, value in report["checks"].items() if key != "all_flc_cover_bcg_center")
    assert report["gates"]["HST11710_acquisition_gate_passed"] is False
    assert report["gates"]["photometry_and_astrometry_likelihood_freeze_authorized"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False


def test_a1689_hst11710_acquisition_ledger() -> None:
    ledger = pd.read_csv(ROOT / "data/derived/r1_a1689_hst11710_acquisition_ledger.csv")
    assert len(ledger) == 70
    assert set(ledger["product_subgroup"]) == {"FLC", "DRZ", "ASN"}
    assert ledger["checksum_matches"].all()
