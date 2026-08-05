from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19bk_source_observability_admission.json"
REPORT = ROOT / "results" / "sigma_v19bk_source_observability_admission" / "report.json"
RUNNER = ROOT / "scripts" / "check_sigma_v19bk_source_observability_admission.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19bk", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observability_audit_passes_and_restricts_eligibility() -> None:
    report = load_runner().build_report(CONFIG)
    assert all(report["gates"].values())
    assert report["eligible_source_ids_after_observability_audit"] == [
        "I4_THERMODYNAMIC_GRADIENT_STRESS",
        "I5_BAROCLINICITY",
    ]
    assert not report["observed_v19x4_gas_posterior_opened"]
    assert not report["invariant_score_computed"]


def test_unmeasured_vectors_and_absolute_stellar_mass_are_not_invented() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    matrix = {row["id"]: row for row in config["observability_matrix"]}
    for name in (
        "I1_COMPONENT_OVERLAP",
        "I2_RELATIVE_CURRENT",
        "I3_ANISOTROPIC_STRESS",
        "I6_CAUSAL_RELAXATION_RATE",
    ):
        assert matrix[name]["status"] == "withheld"
        assert not matrix[name]["eligible_as_new_source"]
    assert not config["authorization"]["infer_absolute_stellar_mass"]
    assert not config["authorization"]["impute_transverse_velocity"]


def test_frozen_report_matches_current_evidence() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"].startswith("passed_source_observability")
    assert all(report["gates"].values())
    assert all(
        not row["transverse_current_hdu_present"]
        for row in report["collisionless_map_evidence"].values()
    )
