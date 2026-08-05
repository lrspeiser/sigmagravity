from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19w2c_ccd7_response_commissioning.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19w2c_ccd7_response_commissioning.py"
REPORT = ROOT / "results" / "sigma_v19w2c_ccd7_response_commissioning" / "report.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19w2c", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules.setdefault("pycrates", types.ModuleType("pycrates"))
SPEC.loader.exec_module(MODULE)


def test_selection_is_frozen_minimum_median_maximum_for_both_obsids():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    base_config = MODULE.load_json(ROOT / config["parents"]["v19w_config"]["path"])
    manifest = MODULE.v19w.load_manifest(base_config)
    expected = []
    for obsid in (10464, 10888):
        expected.extend(MODULE.selected_quantile_names(manifest, obsid))
    assert expected == [row["cell_name"] for row in config["commissioning_cells"]]
    source = [row["expected_source_band_events"] for row in config["commissioning_cells"]]
    assert min(source) == 22
    assert max(source) == 532
    assert {row["expected_background_band_events"] for row in config["commissioning_cells"]} == {0}


def test_every_selected_cell_is_an_exhausted_snapshot_failure():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    base_config = MODULE.load_json(ROOT / config["parents"]["v19w_config"]["path"])
    manifest = MODULE.v19w.load_manifest(base_config)
    snapshot = MODULE.load_json(ROOT / config["parents"]["live_snapshot_report"]["path"])
    rows = MODULE.select_rows(config, manifest, snapshot)
    assert len(rows) == 6
    assert {int(row["ccd_id"]) for row in rows} == {7}
    assert {int(row["obsid"]) for row in rows} == {10464, 10888}


def test_protocol_is_single_attempt_and_disjoint_from_the_base():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    execution = config["execution"]
    assert execution["maximum_concurrent_cells"] == 1
    assert execution["attempts_per_commissioning_cell"] == 1
    assert execution["base_archive_mutation"] is False
    assert execution["scratch_root"] != execution["protected_base_scratch"]
    assert PurePosixPath(execution["free_space_probe"]).is_absolute()
    claim = " ".join(config["claim_boundary"]).lower()
    assert "no spectrum" in claim
    assert "no lensing" in claim
    assert "no gravity formula" in claim


def test_parent_and_runner_hashes_are_exact():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    MODULE.verify_parents(config)
    assert config["execution"]["runner"] == SCRIPT.relative_to(ROOT).as_posix()
    assert MODULE.sha256(SCRIPT) == config["execution"]["runner_sha256"]


def test_completed_report_passes_every_ccd7_gate():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "ccd7_exact_binmap_commissioning_passed"
    assert len(report["completed_cells"]) == 6
    assert all(report["gates"].values())
    assert all(row["ccd_id"] == 7 for row in report["completed_cells"])
    assert all(row["background_all_energy_events"] == 0 for row in report["completed_cells"])
    assert all(row["used_zero_background_path"] for row in report["completed_cells"])
    assert report["v19w5_hardened_recovery_may_be_frozen"]
    assert not report["base_archive_modified"]
    assert not report["spectrum_combined_or_fitted"]
    assert not report["lensing_halo_or_gravity_payload_opened"]
    assert not report["gravity_formula_or_parameter_changed"]
