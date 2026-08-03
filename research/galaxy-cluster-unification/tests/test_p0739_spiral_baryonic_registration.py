from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0739_spiral_baryonic_registration_development.json"
RESULT = ROOT / "results/p0739_spiral_baryonic_registration_development"


def test_frozen_registration_uses_only_development_baryonic_inputs() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "frozen_before_development_image_array_opening"
    assert config["eligibleSplits"] == ["development"]
    assert len(config["systems"]) == 4
    assert "THINGS moment 1 array" in config["forbiddenInputs"]
    assert "THINGS moment 2 array" in config["forbiddenInputs"]
    assert "SPARC observed rotation speeds or errors" in config["forbiddenInputs"]
    assert config["processing"]["gravityParameters"] == 0


def test_result_fails_only_the_predeclared_coverage_gate() -> None:
    report = json.loads((RESULT / "report.json").read_text(encoding="utf-8"))
    failed = {name for name, passed in report["checks"].items() if not passed}
    assert report["status"] == "fail"
    assert failed == {"minimumFiniteFootprintFractionInsideHiR995"}
    assert report["validationArraysOpened"] == 0
    assert report["holdoutArraysOpened"] == 0
    assert report["velocityOrDispersionArraysOpened"] == 0
    assert report["gravityParameters"] == 0


def test_failure_is_localized_to_missing_stellar_footprint() -> None:
    audit = pd.read_csv(RESULT / "map_audit.csv")
    coverage = audit.set_index("galaxy")["finite_footprint_fraction_inside_hi_r995"]
    assert coverage["NGC2403"] < 0.90
    assert coverage["NGC5055"] < 0.90
    assert coverage["NGC3198"] >= 0.90
    assert coverage["NGC7793"] >= 0.90
    assert audit["gas_mass_relative_error"].max() <= 1.0e-10
    assert audit["stellar_mass_relative_error"].max() <= 1.0e-10
    assert audit["velocity_arrays_opened"].sum() == 0
    assert audit["gravity_parameters"].sum() == 0

