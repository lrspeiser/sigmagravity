from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_X2a_flare_gate_passed_without_authorizing_gas_fit() -> None:
    diagnostics = json.loads(
        (ROOT / "data/derived/r1_rxj2129_xmm_x2/flare_diagnostics.json").read_text()
    )
    manifest = json.loads(
        (ROOT / "data/derived/r1_rxj2129_xmm_reduction_manifest.json").read_text()
    )
    report = json.loads(
        (ROOT / "results/r1_rxj2129_xmm_event_processing/report.json").read_text()
    )
    assert set(diagnostics["instruments"]) == {"MOS1", "MOS2", "pn"}
    for label, instrument in diagnostics["instruments"].items():
        assert instrument["retained_bin_live_seconds"] >= 15000
        assert instrument["retained_fraction_of_eligible_bin_live_seconds"] >= 0.25
        assert len(instrument["iterations"]) <= 10
        assert instrument["iterations"][-1]["input_bins"] == instrument["iterations"][-1]["retained_bins"]
        assert instrument["final_rate_limit_counts_per_second"] <= (
            0.4 if label == "pn" else 0.35
        )
    assert manifest["gates"]["R1B3_XMM_X1_calibration_gate_passed"] is True
    assert manifest["gates"]["R1B3_XMM_X2a_flare_exposure_gate_passed"] is True
    assert manifest["gates"]["R1B3_XMM_X2_flare_background_gate_passed"] is True
    assert manifest["gates"]["R1B3_XMM_X3_gas_likelihood_gate_passed"] is False
    assert report["authorization"]["construct_X3_annular_count_response_products"] is True
    assert report["authorization"]["fit_temperature_or_density"] is False
