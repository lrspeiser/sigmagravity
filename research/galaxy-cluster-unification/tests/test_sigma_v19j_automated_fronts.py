from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19j_automated_front_implementation.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19j_automated_fronts.py"
REPORT = ROOT / "results" / "sigma_v19j_automated_fronts" / "report.json"
AUDIT = ROOT / "results" / "sigma_v19j_automated_fronts" / "visual_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("sigma_v19j_test", RUNNER)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def test_v19j_is_frozen_and_content_addressed_before_science_read() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["integrity"]["science_array_read_at_freeze"] is False
    assert config["integrity"]["gradient_score_or_ridge_known_at_freeze"] is False
    assert config["sample"]["same_algorithm_and_thresholds"] is True
    assert config["sample"]["lensing_targets_sealed"] is True
    assert config["poisson_step_score"]["minimum_single_scale_score_sigma"] == 5.0
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        import sigma_v19f_chandra_common as common

        common.validate_parent_hashes(config)
    finally:
        sys.path.remove(scripts)


def test_v19j_synthetic_fixtures_pass() -> None:
    module = load_runner()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = module.synthetic_fixtures(config)
    assert result["uniform_field_passed"] is True
    assert result["curved_step_passed"] is True
    assert result["masked_edge_passed"] is True
    assert result["curved_step_nearest_radius_error_kpc"] <= 8.0


def test_v19j_formal_pass_is_overridden_by_required_visual_audit() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert report["status"] == "both_clusters_passed_frozen_v19j_automated_front_map_gate"
    assert report["failed_clusters"] == []
    assert audit["parent_report_sha256"] == sha256(REPORT)
    assert audit["machine_result_preserved"]["edited_after_audit"] is False
    assert audit["audit_findings"]["BULLET"]["passed"] is False
    assert audit["audit_findings"]["ABELL2146"]["passed"] is False
    assert audit["scientific_threshold_changed"] is False
    assert audit["published_front_coordinate_used"] is False
    assert audit["lensing_target_opened"] is False
    assert "do not run profile fitting" in audit["decision"]
    assert report["profile_fit_run"] is False
    assert report["parametric_bootstrap_run"] is False
    assert report["shock_classification_run"] is False
    for cluster in report["clusters"]:
        for product in cluster["products"]:
            path = ROOT / product["path"]
            assert path.stat().st_size == product["bytes"]
            assert sha256(path) == product["sha256"]
    for product in audit["inspected_products"]:
        assert sha256(ROOT / product["path"]) == product["sha256"]
