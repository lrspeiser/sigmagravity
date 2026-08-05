from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19w2b_cross_detector_response_commissioning.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19w2b_cross_detector_response_commissioning.py"
REPORT = ROOT / "results" / "sigma_v19w2b_cross_detector_response_commissioning" / "report.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19w2b", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules.setdefault("pycrates", types.ModuleType("pycrates"))
SPEC.loader.exec_module(MODULE)


def test_selection_closes_the_uncommissioned_ccd_and_context_axes():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cells = config["commissioning_cells"]
    assert len(cells) == 6
    parsed = [name["cell_name"].rsplit("_", maxsplit=2) for name in cells]
    assert {int(parts[-1].removeprefix("ccd")) for parts in parsed} == {0, 1, 2}
    assert {int(parts[-2].removeprefix("obs")) for parts in parsed} == {
        4986,
        5355,
        5356,
        5357,
    }
    source = [int(item["expected_source_band_events"]) for item in cells]
    background = [int(item["expected_background_band_events"]) for item in cells]
    assert min(source) == 1
    assert max(source) >= 250
    assert 0 in background
    assert any(value > 0 for value in background)


def test_every_selected_cell_is_an_exact_frozen_snapshot_omission():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    base_config = MODULE.load_json(ROOT / config["parents"]["v19w_config"]["path"])
    manifest = MODULE.v19w.load_manifest(base_config)
    snapshot = MODULE.load_json(ROOT / config["parents"]["live_snapshot_report"]["path"])
    rows = MODULE.select_rows(config, manifest, snapshot)
    assert len(rows) == 6
    assert [MODULE.v19w.cell_name(row) for row in rows] == [
        item["cell_name"] for item in config["commissioning_cells"]
    ]


def test_protocol_is_single_worker_and_cannot_modify_base_or_science():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    execution = config["execution"]
    assert execution["maximum_concurrent_cells"] == 1
    assert execution["base_archive_mutation"] is False
    assert execution["protected_base_scratch"] != execution["scratch_root"]
    assert PurePosixPath(execution["free_space_probe"]).is_absolute()
    assert config["failure_rule"].startswith("Retain the failed scratch cell")
    claim = " ".join(config["claim_boundary"]).lower()
    assert "no spectrum" in claim
    assert "no lensing" in claim
    assert "no gravity formula" in claim


def test_parent_and_runner_hashes_are_exact():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    MODULE.verify_parents(config)
    assert config["execution"]["runner"] == SCRIPT.relative_to(ROOT).as_posix()
    assert MODULE.sha256(SCRIPT) == config["execution"]["runner_sha256"]


def test_completed_report_passes_every_cross_detector_gate():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "cross_detector_exact_binmap_commissioning_passed"
    assert len(report["completed_cells"]) == 6
    assert all(report["gates"].values())
    assert report["v19w4_hardened_recovery_may_be_frozen"]
    assert not report["base_archive_modified"]
    assert not report["spectrum_combined_or_fitted"]
    assert not report["lensing_halo_or_gravity_payload_opened"]
    assert not report["gravity_formula_or_parameter_changed"]
