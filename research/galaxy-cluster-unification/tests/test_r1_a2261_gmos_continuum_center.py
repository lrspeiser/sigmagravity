from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a2261_continuum_center_failure_is_binding() -> None:
    report = json.loads((ROOT / "results/r1_a2261_gmos_continuum_center/report.json").read_text())
    assert report["continuum_window_angstrom"] == [4800.0, 5400.0]
    assert len(report["individual_fits"]) == 4
    assert report["individual_center_range_arcsec"] > report["maximum_allowed_center_range_arcsec"]
    assert report["gates"]["P2c_continuum_centroid_range_gate_passed"] is False
    assert report["authorization"]["execute_frozen_sky_window_variants"] is False
    assert report["authorization"]["fit_stellar_kinematics"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
