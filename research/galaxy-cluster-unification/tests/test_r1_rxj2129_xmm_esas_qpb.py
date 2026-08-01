from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_qpb_audit_retains_exactly_the_two_strict_passing_instruments() -> None:
    report = json.loads(
        (
            ROOT
            / "results/r1_rxj2129_xmm_event_processing/qpb_background_audit.json"
        ).read_text()
    )
    assert report["stage"] == "X2b2_FWC_corner_subgate"
    assert report["minimum_instrument_gate_passed"] is True
    assert report["passing_instruments"] == ["MOS2", "pn"]
    assert report["excluded_at_FWC_corner_subgate"] == ["MOS1"]
    assert report["instrument_results"]["MOS1"]["sectors"]["5"]["passed"] is False
    assert report["instrument_results"]["MOS1"]["sectors"]["5"][
        "FWC_corner_scale"
    ] < 0.5
    for instrument in ("MOS2", "pn"):
        result = report["instrument_results"][instrument]
        assert result["passed"] is True
        assert result["product_audit"]["passed"] is True
        assert all(sector["passed"] for sector in result["sectors"].values())
    assert report["authorization"]["run_frozen_local_outer_annulus_transfer_audit"] is True
    assert report["authorization"]["claim_full_X2_pass"] is False
