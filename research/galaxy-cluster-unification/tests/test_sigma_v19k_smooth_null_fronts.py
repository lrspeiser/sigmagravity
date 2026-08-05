from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19k_smooth_null_front_likelihood.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19k_smooth_null_fronts.py"
FIXTURE_REPORT = ROOT / "results" / "sigma_v19k_smooth_null_fronts" / "fixture_failure.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("sigma_v19k_test", RUNNER)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def test_v19k_is_frozen_before_fixture_or_science_outcomes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["integrity"]["v19k_science_array_read_at_freeze"] is False
    assert config["integrity"]["v19k_seed_or_profile_known_at_freeze"] is False
    assert config["integrity"]["v19k_fixture_outcome_known_at_freeze"] is False
    assert config["sample"]["same_algorithm_thresholds_models_and_optimizer"] is True
    assert config["sample"]["lensing_targets_sealed"] is True
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        import sigma_v19f_chandra_common as common

        common.validate_parent_hashes(config)
    finally:
        sys.path.remove(scripts)


def synthetic_profile(step: bool) -> dict:
    centers = np.arange(-195.0, 200.0, 10.0)
    exposure = np.full(40, 1e10)
    source_rate = np.exp(-16.0 + 0.8 * centers / 200.0 + 0.3 * (centers / 200.0) ** 2)
    if step:
        source_rate = source_rate * np.where(centers >= 0.0, 4.0, 1.0)
    counts = np.rint(exposure * source_rate)
    return {
        "valid": True,
        "centers_kpc": centers,
        "counts": counts,
        "background": np.zeros(40),
        "background_variance": np.zeros(40),
        "exposure": exposure,
        "valid_pixels": np.ones(40, dtype=int),
        "positive": np.ones(40, dtype=bool),
    }


def test_v19k_likelihood_rejects_smooth_curvature_and_recovers_step() -> None:
    module = load_runner()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))["poisson_models"]
    smooth = module.fit_profile(synthetic_profile(False), config)
    step = module.fit_profile(synthetic_profile(True), config)
    assert smooth["success"] is True
    assert smooth["step_score_sigma"] < 5.0
    assert step["success"] is True
    assert step["step_score_sigma"] >= 5.0
    assert math.isclose(step["density_compression"], 2.0, rel_tol=0.05)


def test_v19k_fixture_failure_is_fail_closed_before_science() -> None:
    report = json.loads(FIXTURE_REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "mandatory_pre_science_fixture_failed"
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    assert report["failed_fixtures"] == ["step"]
    assert report["mandatory_pre_science_fixtures"]["linear"]["passing_seed_count"] == 0
    assert report["mandatory_pre_science_fixtures"]["step"]["passing_seed_count"] == 64
    assert report["mandatory_pre_science_fixtures"]["step"]["retained_arc_count"] == 0
    assert report["science_cluster_array_read_by_v19k"] is False
    assert report["science_seed_profile_or_likelihood_evaluated"] is False
    assert report["scientific_threshold_changed"] is False
    assert report["lensing_target_opened"] is False
