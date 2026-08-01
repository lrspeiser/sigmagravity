from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_x4_response_interface_passes_without_authorizing_a_fit() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_xmm_x4_response_interface/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "pass"
    assert set(report["instruments"]) == {"MOS2", "pn"}
    for instrument in report["instruments"].values():
        assert instrument["status"] == "pass"
        assert instrument["detector_map"]["coverage_gate_passed"] is True
        assert instrument["detector_map"]["a01_map_pixel_count"] >= 301
        assert instrument["rmf"]["valid"] is True
        assert instrument["common_energy_grid"] is True
        assert all(item["valid"] for item in instrument["arfs"].values())
        assert all(item["valid"] for item in instrument["logs"].values())
        assert instrument["science_band_coupling"]["cross_to_direct_integrated_ratio"] > 0
        assert instrument["science_band_coupling"]["central_to_direct_integrated_ratio"] > 0
    assert report["authorization"]["scale_interface_to_full_X4_product_set"] is True
    assert report["authorization"]["fit_temperature_or_density"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
