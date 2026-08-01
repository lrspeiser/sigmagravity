from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_x4_map_resolution_converges_for_both_detectors() -> None:
    report = json.loads(
        (
            ROOT / "results/r1_rxj2129_xmm_x4_map_resolution_convergence/report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "pass"
    assert report["report_version"] == "R1B3-RXJ2129-XMM-X4-map-resolution-0.2"
    assert report["baseline_requested_dimension"] == 920
    assert report["refined_requested_dimension"] == 1302
    assert report["completion_marker_present"] is True
    assert set(report["instruments"]) == {"MOS2", "pn"}
    for instrument in report["instruments"].values():
        assert instrument["status"] == "pass"
        assert all(item["valid"] for item in instrument["maps"].values())
        assert all(
            item["pixel_size_gate_passed"] for item in instrument["maps"].values()
        )
        assert instrument["common_energy_grid"] is True
        assert instrument["comparison_passed"] is True
        assert all(instrument["completion_markers"].values())
        for name, threshold in instrument["thresholds"].items():
            assert instrument["differences"][name] <= threshold
    assert report["authorization"]["construct_full_X4_at_baseline_resolution"] is True
    assert report["authorization"]["fit_temperature_or_density"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
