from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_outer_annulus_transfer_gate_passes_for_mos2_and_pn() -> None:
    report = json.loads(
        (
            ROOT
            / "results/r1_rxj2129_xmm_event_processing/outer_annulus_transfer_audit.json"
        ).read_text()
    )
    assert report["stage"] == "X2b2_local_outer_annulus_transfer_subgate"
    assert report["frozen_region"]["kpc"] == [650.0, 900.0]
    assert report["passing_instruments"] == ["MOS2", "pn"]
    assert report["full_X2b2_background_gate_passed"] is True
    assert report["instrument_results"]["MOS2"]["observed_counts"] == 149.0
    assert report["instrument_results"]["MOS2"]["outer_annulus_transfer_scale"] == 1.3134244395380283
    assert report["instrument_results"]["pn"]["observed_counts"] == 390.88
    assert report["instrument_results"]["pn"]["outer_annulus_transfer_scale"] == 0.7419486890802747
    assert all(
        report["instrument_results"][name]["posterior"][
            "posterior_bound_rule_passed"
        ]
        for name in ("MOS2", "pn")
    )
    assert report["authorization"]["construct_X3_annular_count_response_products"] is True
    assert report["authorization"]["fit_temperature_or_density"] is False
