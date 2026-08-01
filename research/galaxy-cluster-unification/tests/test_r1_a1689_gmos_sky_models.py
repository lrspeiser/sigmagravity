from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_all_frozen_sky_variants_gate() -> None:
    report = json.loads((ROOT / "results/r1_a1689_gmos_sky_models/report.json").read_text())
    assert len(report["products"]) == 4
    assert all(len(row["variants"]) == 3 for row in report["products"])
    assert all(row["all_variants_passed"] for row in report["products"])
    assert report["gates"]["P2d_all_frozen_sky_variants_gate_passed"] is True
    assert report["authorization"]["register_and_combine_baseline_exposures_for_coverage_audit"] is True
    assert report["authorization"]["fit_stellar_kinematics"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
