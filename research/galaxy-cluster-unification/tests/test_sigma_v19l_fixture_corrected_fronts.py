from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19l_fixture_corrected_front_likelihood.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19l_fixture_corrected_fronts.py"
FAILURE = (
    ROOT / "results" / "sigma_v19l_fixture_corrected_fronts" / "fixture_failure.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("sigma_v19l_test", RUNNER)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def synthetic_profile(step: bool) -> dict:
    centers = np.arange(-195.0, 200.0, 10.0)
    tau = centers / 200.0
    exposure = np.full(40, 1e10)
    source_rate = np.exp(-16.0 + 0.8 * tau + 0.3 * tau**2 - 0.2 * tau**3 + 0.1 * tau**4)
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


def test_v19l_is_frozen_and_parent_hashes_are_valid() -> None:
    frozen = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert frozen["integrity"]["v19l_fixture_outcome_known_at_freeze"] is False
    assert frozen["integrity"]["v19l_cluster_array_read_at_freeze"] is False
    assert frozen["mandatory_pre_science_fixtures"]["failure_rule"].startswith(
        "stop before cluster science"
    )
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        import sigma_v19f_chandra_common as common

        common.validate_parent_hashes(frozen)
    finally:
        sys.path.remove(scripts)


def test_v19l_quartic_null_rejects_smooth_profile_and_recovers_step() -> None:
    module = load_runner()
    frozen = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = module.resolved_config(frozen)
    v19k, _ = module.modules()
    smooth = module.fit_profile(synthetic_profile(False), config, v19k)
    step = module.fit_profile(synthetic_profile(True), config, v19k)
    assert smooth["success"] is True
    assert smooth["step_score_sigma"] < 5.0
    assert step["success"] is True
    assert step["step_score_sigma"] >= 5.0
    assert math.isclose(step["density_compression"], 2.0, rel_tol=0.05)


def test_v19l_failure_is_fail_closed_before_science() -> None:
    report = json.loads(FAILURE.read_text(encoding="utf-8"))
    fixtures = report["mandatory_pre_science_fixtures"]
    assert report["status"] == "mandatory_pre_science_fixture_failure"
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    assert fixtures["uniform"]["passing_seed_count"] == 0
    assert fixtures["linear"]["passing_seed_count"] == 0
    assert fixtures["radial"]["passing_seed_count"] == 0
    assert fixtures["step"]["passing_seed_count"] == 64
    assert fixtures["step"]["retained_arc_count"] == 0
    assert fixtures["step"]["passed"] is False
    assert report["cluster_science_array_read"] is False
    assert report["science_seed_fitted"] is False
    assert report["science_arc_linked"] is False
    assert report["lensing_target_opened"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
