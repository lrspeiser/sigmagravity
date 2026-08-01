from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_wavelength_dependent_lsf_gate() -> None:
    report = json.loads((ROOT / "results/r1_a1689_gmos_lsf/report.json").read_text())
    assert report["arc_count"] == 3
    assert len(report["wavelength_bins"]) == 5
    assert all(row["accepted_lines"] >= 3 for row in report["wavelength_bins"])
    assert report["gates"]["P3b_wavelength_dependent_lsf_gate_passed"] is True
    assert report["authorization"]["forward_convolve_xsl_and_run_baseline_ppxf"] is True
    assert report["gates"]["gravity_response_fit_authorized"] is False
