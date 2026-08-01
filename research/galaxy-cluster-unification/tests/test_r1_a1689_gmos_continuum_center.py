from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_continuum_only_center_gate() -> None:
    report = json.loads((ROOT / "results/r1_a1689_gmos_continuum_center/report.json").read_text())
    assert len(report["individual_fits"]) == 4
    assert report["individual_center_range_arcsec"] <= report["maximum_allowed_center_range_arcsec"]
    assert report["gates"]["P2c_continuum_centroid_range_gate_passed"] is True
    assert report["authorization"]["execute_frozen_sky_window_variants"] is True
    assert report["authorization"]["fit_stellar_kinematics"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
