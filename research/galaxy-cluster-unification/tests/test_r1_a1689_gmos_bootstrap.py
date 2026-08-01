from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_joint_bootstrap_gate() -> None:
    report = json.loads((ROOT / "results/r1_a1689_gmos_bootstrap/report.json").read_text())
    assert report["requested_replicates"] == 200
    assert report["complete_nine_bin_replicates"] >= 180
    assert report["gates"]["P3d_bootstrap_covariance_gate_passed"] is True
    assert report["authorization"]["run_frozen_systematic_grid_and_final_covariance_assembly"] is True
    assert report["gates"]["gravity_response_fit_authorized"] is False
