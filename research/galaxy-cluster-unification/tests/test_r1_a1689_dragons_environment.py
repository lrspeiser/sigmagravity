from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_dragons_environment_and_bpm_gate() -> None:
    report = json.loads((ROOT / "results/r1_a1689_dragons_environment/report.json").read_text())
    assert report["runtime"]["packages"]["dragons"] == "4.2.2"
    assert report["runtime"]["packages"]["ppxf"] == "9.4.8"
    assert report["bpm_checksum_passed"] is True
    assert all(report["recognition_gates"].values())
    assert len(report["files"]["science"]) == 4
    assert len(report["files"]["flats"]) == 4
    assert len(report["files"]["arcs"]) == 3
    assert len(report["files"]["biases"]) == 10
    assert report["gates"]["P1_environment_and_bpm_gate_passed"] is True
    assert report["authorization"]["execute_frozen_P2_calibration_reduction"] is True
    assert report["gates"]["gravity_response_fit_authorized"] is False
