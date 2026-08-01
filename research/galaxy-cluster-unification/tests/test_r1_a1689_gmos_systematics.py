from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_frozen_systematic_grid_gate() -> None:
    report = json.loads((ROOT / "results/r1_a1689_gmos_systematics/report.json").read_text())
    assert report["requested_grid_runs"] == 27
    assert report["complete_grid_runs"] == 27
    assert report["failed_bin_fits"] == 0
    assert report["baseline_reproduction_max_fractional_sigma_difference"] > 1e-6
    assert report["gates"]["baseline_reproduction_passed"] is False
    assert max(report["maximum_absolute_sigma_shift_fraction_by_signed_bin"].values()) > 0.1
    assert report["gates"]["P3e_systematic_shift_gate_passed"] is False
    assert report["authorization"]["assemble_final_signed_and_symmetrized_covariance"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
