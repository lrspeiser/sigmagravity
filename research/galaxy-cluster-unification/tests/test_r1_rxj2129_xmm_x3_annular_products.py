from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_x3_annular_products_pass_frozen_adequacy_gate() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_xmm_x3_annular_products/report.json").read_text()
    )
    manifest = json.loads(
        (
            ROOT / "data/derived/r1_rxj2129_xmm_x3_annular_product_manifest.json"
        ).read_text()
    )
    with (
        ROOT / "data/derived/r1_rxj2129_xmm_x3_annular_count_ledger.csv"
    ).open(newline="") as handle:
        ledger = list(csv.DictReader(handle))

    assert report["stage"] == "X3_annular_count_response_adequacy"
    assert report["report_version"].endswith("0.2")
    assert manifest["manifest_version"].endswith("0.2")
    assert report["status"] == "pass"
    assert len(report["passing_annuli"]) >= report["minimum_accepted_annuli"]
    assert report["passing_annulus_combined_net_counts"] >= report[
        "minimum_total_net_counts"
    ]
    assert len(ledger) == 6
    for annulus_id in report["passing_annuli"]:
        result = manifest["annuli"][annulus_id]
        assert result["all_products_passed"] is True
        assert result["combined_net_counts"] > 0
        assert result["combined_signal_to_noise"] >= report[
            "minimum_signal_to_noise_each_annulus"
        ]
        for instrument in ("MOS2", "pn"):
            audit = result["instrument_results"][instrument]["product_audit"]
            assert audit["passed"] is True
            assert audit["source_mask_row_declaration_valid"] is True
            assert audit["source_removal_mode_valid"] is True
            assert all(mask["passed"] for mask in audit["source_masks"].values())
            assert audit["pn_quadrant_declaration_valid"] is True
            assert audit["pn_badpixel_resolution_valid"] is True
            assert audit["products"]["source_spectrum"]["passed"] is True
            assert audit["products"]["QPB_spectrum"]["passed"] is True
            assert audit["products"]["RMF"]["passed"] is True
            assert audit["products"]["ARF"]["passed"] is True
    assert report["authorization"]["freeze_XMM_specific_gas_likelihood_protocol"] is True
    assert report["authorization"]["fit_temperature_or_density"] is False
    assert report["authorization"]["infer_gas_mass"] is False
    assert report["authorization"]["infer_dynamical_or_Weyl_response"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
