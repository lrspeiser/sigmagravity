from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19db_bullet_velocity_combination.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19db_bullet_velocity_combination.py"


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
