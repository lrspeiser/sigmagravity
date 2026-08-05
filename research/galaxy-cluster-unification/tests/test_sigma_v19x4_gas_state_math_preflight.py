from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_sigma_v19x4_gas_state_math_preflight.py"
CONFIG = ROOT / "configs" / "sigma_v19x4_gas_state_math_preflight.json"
REPORT = ROOT / "results" / "sigma_v19x4_gas_state_math_preflight" / "report.json"


def load_checker():
    spec = importlib.util.spec_from_file_location("sigma_v19x4_checker", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_preflight_passes_without_opening_targets() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = load_checker().execute(config)
    assert all(report["gates"].values())
    assert report["status"].endswith("awaiting_v19x3_measurements")
    assert not report["observed_regional_spectra_opened"]
    assert not report["gravity_theory_tested"]


def test_report_records_material_prospective_correction() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    correction = report["historical_error"]
    assert correction["surface_density_understatement_fraction"] == pytest.approx(
        1.0 / 6.0
    )
    assert correction["corrected_surface_density_increase_fraction"] == pytest.approx(
        0.2
    )
    assert all(report["parent_hash_checks"].values())


def test_all_494_region_geometries_are_admitted() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    counts = {
        name: row["inventory"]["accepted_regions"]
        for name, row in report["clusters"].items()
    }
    assert counts == {"BULLET": 366, "ABELL2146": 128}
    assert sum(counts.values()) == 494
