from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_e325_immutable_receipt_and_array_integrity_pass() -> None:
    report = json.loads(
        (ROOT / "results/r1_e325_acquisition/report.json").read_text()
    )
    inventory = pd.read_csv(ROOT / report["receipt"]["output"])

    assert report["selection_blind"] is True
    assert report["gravity_residuals_inspected"] is False
    assert report["receipt"]["expected_products"] == 5
    assert report["receipt"]["received_products"] == 5
    assert report["receipt"]["received_bytes"] == 1496617920
    assert report["receipt"]["all_sizes_passed"] is True
    assert report["receipt"]["all_hashes_passed"] is True
    assert report["hst_integrity"]["passed"] is True
    assert report["hst_integrity"]["exposure_seconds_by_filter"] == {
        "F475W": 4800.0,
        "F814W": 18882.0,
    }
    assert report["hst_integrity"]["minimum_central_positive_weight_fraction"] == 1.0
    assert report["muse_integrity"]["passed"] is True
    assert report["muse_integrity"]["shape"] == "681x60x60"
    assert report["muse_integrity"]["science_finite_fraction"] == 1.0
    assert report["muse_integrity"]["variance_positive_fraction"] == 1.0
    assert report["gates"]["complete_acquisition_gate_passed"] is True
    assert report["gates"]["rank_three_candidate_admission_passed"] is False
    assert len(inventory) == 5
    assert inventory["size_pass"].all()
    assert inventory["hash_pass"].all()
    assert report["authorization"]["implement_frozen_image_level_jacobian"] is True
    assert report["authorization"]["count_toward_ten_system_target"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
