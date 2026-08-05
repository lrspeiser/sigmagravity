from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19i_continuous_member_field.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19i_continuous_member_field.py"
REPORT = ROOT / "results" / "sigma_v19i_continuous_member_field" / "report.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def load_runner():
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("sigma_v19i_test", RUNNER)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def test_v19i_parent_chain_and_claim_boundary_are_fail_closed() -> None:
    config = load_config()
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        import sigma_v19f_chandra_common as common

        common.validate_parent_hashes(config)
    finally:
        sys.path.remove(scripts)
    assert config["sample"]["lensing_targets_sealed"] is True
    assert config["sample"]["published_subcluster_labels_validation_only"] is True
    assert config["continuous_fields"]["mass_weighting"].startswith("none")
    assert "not a physical baryonic mass-current" in config["continuous_fields"][
        "frame_dragging_claim_boundary"
    ]
    assert config["advance_rule"]["gravity_formula_selection_authorized"] is False
    assert config["advance_rule"]["lensing_target_access_authorized"] is False


def test_v19i_bandwidth_tie_rule_prefers_the_larger_bandwidth() -> None:
    module = load_runner()
    x = np.asarray([-1.0, 1.0])
    y = np.zeros(2)
    selected, scores = module.select_bandwidth(x, y, [50.0, 50.0])
    assert np.all(np.isfinite(scores))
    assert selected == 50.0


def test_v19i_topology_distinguishes_one_and_two_modes() -> None:
    module = load_runner()
    grid = np.arange(-500.0, 505.0, 5.0)
    mode_config = {
        "minimum_peak_to_saddle_density_ratio": 1.5,
        "minimum_peak_density_fraction_of_global": 0.1,
        "minimum_pair_separation_kpc": 100.0,
        "maximum_pair_separation_kpc": 1200.0,
    }
    y = np.zeros(60)

    one_x = np.zeros(60)
    one_density = module.density_grid(one_x, y, grid, grid, 50.0)
    one_modes = module.persistent_modes(one_density, grid, grid, 50.0, mode_config)
    assert len(one_modes) == 1
    assert module.select_primary_pair(one_modes, mode_config) == []

    two_x = np.concatenate([np.full(30, -180.0), np.full(30, 180.0)])
    two_density = module.density_grid(two_x, y, grid, grid, 50.0)
    two_modes = module.persistent_modes(two_density, grid, grid, 50.0, mode_config)
    pair = module.select_primary_pair(two_modes, mode_config)
    assert len(two_modes) == 2
    assert len(pair) == 2
    assert {row["x_kpc"] for row in pair} == {-180.0, 180.0}


def test_v19i_real_gate_records_one_pass_and_one_failure() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "frozen_v19i_continuous_member_field_gate_failed"
    assert report["failed_clusters"] == ["BULLET"]
    assert report["published_subcluster_labels_used_for_selection"] is False
    assert report["mass_current_claimed"] is False
    assert report["lensing_target_opened"] is False
    assert report["gravity_formula_selected"] is False
    assert report["gravity_parameter_changed"] is False
    clusters = {row["cluster"]: row for row in report["clusters"]}
    bullet = clusters["BULLET"]
    abell = clusters["ABELL2146"]
    assert bullet["primary_pair"] == []
    assert bullet["bootstrap"]["accepted_draws"] == 2000
    assert bullet["bootstrap"]["failed_draws"] == 0
    assert not all(bullet["gates"].values())
    assert abell["primary_pair_separation_kpc"] == 520.0240379059414
    assert [
        row["recovery_fraction_of_requested"]
        for row in abell["bootstrap"]["primary_mode_summaries"]
    ] == [0.899, 0.7125]
    assert all(abell["gates"].values())
    for cluster in clusters.values():
        draws_path = ROOT / cluster["bootstrap"]["draws_file"]
        assert (
            hashlib.sha256(draws_path.read_bytes()).hexdigest()
            == cluster["bootstrap"]["draws_sha256"]
        )
        for product in cluster["products"]:
            path = ROOT / product["path"]
            assert path.stat().st_size == product["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == product["sha256"]
