from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_baseline_ppxf_retains_all_outcomes() -> None:
    report = json.loads((ROOT / "results/r1_a1689_gmos_baseline_ppxf/report.json").read_text())
    assert len(report["signed_bin_fits"]) == 9
    assert len(report["opposite_side_pairs"]) == 4
    assert report["successful_signed_bins"] >= 7
    assert report["gates"]["P3c_baseline_ppxf_minimum_fits_gate_passed"] is True
    assert report["authorization"]["run_frozen_200_replicate_covariance_and_systematic_grid"] is True
    assert report["gates"]["gravity_response_fit_authorized"] is False
