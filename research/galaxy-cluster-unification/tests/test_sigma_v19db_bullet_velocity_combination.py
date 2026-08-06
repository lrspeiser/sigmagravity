from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19db_bullet_velocity_combination.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19db_bullet_velocity_combination.py"
OUTPUT = ROOT / "results" / "sigma_v19db_bullet_velocity_combination"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19db_combiner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_authorization_is_bullet_primary_combination_only() -> None:
    payload = config()
    auth = payload["authorization"]
    assert auth["combine_bullet_primary_8000_regions"] is True
    assert auth["combine_bullet_robustness_or_obsid554"] is False
    assert auth["open_abell2146"] is False
    assert auth["fit_temperature_abundance_redshift_or_velocity"] is False
    assert auth["open_lensing_halo_or_gravity_payload"] is False
    assert auth["derive_or_change_action_or_gravity_constants"] is False


def test_hierarchy_and_forward_fold_gate_are_frozen() -> None:
    payload = config()
    hierarchy = payload["hierarchy"]
    assert hierarchy["partition_key"] == "obsid"
    assert hierarchy["bscale_method"] == "asca"
    assert hierarchy["exp_origin"] == "pha"
    assert hierarchy["intermediate_rmf_threshold"] == 0.0
    assert hierarchy["final_rmf_threshold"] == 1e-6
    assert payload["gates"]["direct_hierarchical_pilot_forward_fold_relative_l1_at_most"] == 1e-8


def test_payload_blind_plan_partitions_every_primary_cell() -> None:
    runner = load_runner()
    payload = config()
    parents = runner.validate_frozen(payload)
    plan = runner.build_plan(payload, parents)
    assert len(plan) == 43
    bins = [bin_id for region in plan for bin_id in region["member_bin_ids"]]
    cells = [cell["cell_name"] for region in plan for cell in region["cells"]]
    assert len(bins) == len(set(bins)) == 366
    assert len(cells) == len(set(cells)) == 3483
    assert {cell["obsid"] for region in plan for cell in region["cells"]} == set(payload["workload"]["obsids"])


def test_runner_contains_no_spectral_fit_engine() -> None:
    source = RUNNER.read_text(encoding="utf-8").lower()
    for forbidden in ("sherpa", "xspec", "xsapec", "xsapec", "fit_spectrum("):
        assert forbidden not in source


def test_terminal_combination_is_current_and_exact() -> None:
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "bullet_primary_velocity_region_combination_passed"
    assert report["runner_sha256"] == sha256(RUNNER)
    assert report["config_sha256"] == sha256(CONFIG)
    assert all(report["gates"].values())
    assert all(report["equivalence_pilot"]["gates"].values())
    assert max(
        item["relative_l1_difference"]
        for item in report["equivalence_pilot"]["forward_folds"].values()
    ) <= 1e-8
    assert len(report["regions"]) == 43
    assert sum(item["expected_full_pha_source_counts"] for item in report["regions"]) == 674283
    assert sum(item["combined_full_pha_source_counts"] for item in report["regions"]) == 674283
    products = [product for region in report["regions"] for product in region["products"]]
    assert len(products) == 172
    for product in products:
        path = ROOT / product["path"]
        assert path.stat().st_size == product["bytes"]
        assert sha256(path) == product["sha256"]
