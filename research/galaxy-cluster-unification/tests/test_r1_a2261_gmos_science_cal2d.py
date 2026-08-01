from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a2261_individual_calibrated_2d_gate() -> None:
    report = json.loads((ROOT / "results/r1_a2261_gmos_science_cal2d/report.json").read_text())
    assert len(report["products"]) == 4
    assert all(row["passed"] for row in report["products"])
    assert all(row["cosmic_ray_flagged_pixels"] > 0 for row in report["products"])
    assert all(row["history_matches_frozen_recipe"] for row in report["products"])
    assert all(row["exact_frozen_calibration_provenance"] for row in report["products"])
    assert all(row["forbidden_sky_stack_or_extraction_primitives_absent"] for row in report["products"])
    assert report["gates"]["P2b_individual_calibrated_2d_gate_passed"] is True
    assert report["authorization"]["fit_frozen_continuum_centroid_and_sky_models"] is True
    assert report["authorization"]["fit_stellar_kinematics"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
