from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_xmm_protocol_records_OOT_semantics_correction() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_xmm_event_processing_protocol.json").read_text()
    )
    assert protocol["protocol_version"].endswith("0.4")
    oot_correction = protocol["correction_log"][-3]
    bin_phase_correction = protocol["correction_log"][-2]
    live_fraction_correction = protocol["correction_log"][-1]
    assert oot_correction["data_driven"] is False
    assert oot_correction["thresholds_or_science_selection_changed"] is False
    assert "withoutoftime=no" in oot_correction["correction"]
    assert bin_phase_correction["data_driven"] is False
    assert bin_phase_correction["thresholds_or_science_selection_changed"] is False
    assert live_fraction_correction["data_driven"] is False
    assert live_fraction_correction["thresholds_or_science_selection_changed"] is False
    assert protocol["flare_filter"]["time_bin_alignment"]["common_timemin"] == 152241400
    assert protocol["flare_filter"]["time_bin_seconds"] == 100
    assert protocol["flare_filter"]["fixed_rate_ceiling_counts_per_second"] == {
        "MOS1": 0.35,
        "MOS2": 0.35,
        "pn": 0.4,
    }


def test_rxj2129_xmm_X1_calibration_gate_passed() -> None:
    manifest = json.loads(
        (ROOT / "data/derived/r1_rxj2129_xmm_reduction_manifest.json").read_text()
    )
    report = json.loads(
        (ROOT / "results/r1_rxj2129_xmm_event_processing/report.json").read_text()
    )
    assert manifest["event_arrays_read_during_X1_audit"] is False
    assert set(manifest["products"]) == {"MOS1", "MOS2", "pn", "pn_OOT"}
    assert all(product["gate_passed"] for product in manifest["products"].values())
    assert manifest["products"]["pn"]["sha256"] != manifest["products"]["pn_OOT"]["sha256"]
    assert manifest["gates"]["R1B3_XMM_X1_calibration_gate_passed"] is True
    assert manifest["gates"]["R1B3_XMM_X2_flare_background_gate_passed"] is True
    assert manifest["gates"]["R1B3_XMM_X3_gas_likelihood_gate_passed"] is False
    assert report["status"] == "pass"
    assert report["authorization"].get(
        "fit_gas_profile",
        report["authorization"].get("fit_temperature_or_density"),
    ) is False
