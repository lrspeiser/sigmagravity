from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nominal_x4_map_baseline_is_preserved_as_a_failed_gate() -> None:
    report = json.loads(
        (
            ROOT
            / "results/r1_rxj2129_xmm_x4_map_resolution_convergence_v0_1/report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "fail"
    assert report["gates"]["X4_map_resolution_convergence_passed"] is False
    assert report["authorization"]["construct_full_X4_at_baseline_resolution"] is False
    for instrument in report["instruments"].values():
        baseline = instrument["maps"]["baseline"]
        assert baseline["pixel_size_detector_units"] > 80.0
        assert baseline["pixel_size_gate_passed"] is False
        assert baseline["valid"] is False
