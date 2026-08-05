from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19bl_source_invariant_scoring_protocol.json"
REPORT = ROOT / "results" / "sigma_v19bl_source_invariant_scoring_protocol" / "report.json"
RUNNER = ROOT / "scripts" / "check_sigma_v19bl_source_invariant_scoring_protocol.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19bl", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_preflight_passes_without_opening_outcomes() -> None:
    report = load_runner().build_report(CONFIG)
    assert all(report["gates"].values())
    assert not report["observed_v19x4_gas_posterior_opened"]
    assert not report["source_invariant_score_computed"]
    assert not report["lensing_or_halo_payload_opened"]
    assert not report["action_or_gravity_parameter_selected"]


def test_protocol_requires_full_cluster_branch_scale_and_radius_transfer() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registered = config["registered_inputs"]
    gates = config["robustness_and_decision_gates"]
    assert registered["clusters"] == ["BULLET", "ABELL2146"]
    assert registered["temperature_normalization_rank_correlations"] == [-0.9, 0.0, 0.9]
    assert registered["smoothing_fwhm_kpc"] == [50.0, 100.0]
    assert [*registered["radius_robustness_kpc"], registered["primary_radius_kpc"]] == [
        250.0,
        500.0,
        350.0,
    ]
    assert gates["all_three_rank_correlation_branches_must_pass"]
    assert gates["both_clusters_must_pass"]


def test_frozen_report_matches_current_math_contract() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"].startswith("passed_source_invariant_scoring_math")
    assert all(report["gates"].values())
