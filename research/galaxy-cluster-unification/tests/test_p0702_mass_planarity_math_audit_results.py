from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_p0702_frozen_report_passes_every_math_gate_without_unsealing() -> None:
    report = json.loads(
        (ROOT / "results/p0702_mass_planarity_math_audit/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "pass"
    assert report["all_math_gates_pass"] is True
    assert report["candidate_advanced_to_spent_joint_screen"] is True
    assert all(report["gate_results"].values())
    assert report["failed_gates"] == []
    assert report["sealed_P0633_kinematics_opened"] is False
    assert report["sealed_P0640_lensing_constraints_opened"] is False


def test_p0702_identifies_sheet_without_calling_filament_a_sheet() -> None:
    report = json.loads(
        (ROOT / "results/p0702_mass_planarity_math_audit/report.json").read_text(
            encoding="utf-8"
        )
    )
    values = report["metrics"]["synthetic_planarity"]
    assert values["sheet"] > 0.95
    assert values["filament"] < 0.05
    assert values["ball"] < 1e-10
    assert report["metrics"]["spent_DDO154"]["planarity"] > 0.99
    assert report["metrics"]["spent_RXJ2129"]["planarity"] < 0.10
