from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_signed_spectrum_snr_gate() -> None:
    report = json.loads((ROOT / "results/r1_a1689_gmos_signed_spectra/report.json").read_text())
    assert len(report["signed_bins"]) == 9
    assert report["passing_signed_bins"] >= 7
    assert report["gates"]["P3a_signed_spectrum_snr_gate_passed"] is True
    assert report["authorization"]["run_frozen_baseline_ppxf_on_retained_signed_bins"] is True
    assert report["gates"]["gravity_response_fit_authorized"] is False
