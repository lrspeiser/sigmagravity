from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19w5_ccd7_hardened_response_recovery.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19w5_ccd7_hardened_response_recovery.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19w5", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules.setdefault("pycrates", types.ModuleType("pycrates"))
SPEC.loader.exec_module(MODULE)


def test_frozen_parent_chain_and_ccd7_gate_pass():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes, v19w3_config, ccd7 = MODULE.verify_parents(config)
    assert set(hashes) == set(config["parents"])
    assert not v19w3_config["base_terminal_gate"]["may_launch_while_base_process_runs"]
    assert v19w3_config["base_terminal_gate"]["require_no_process_command_containing"]
    assert ccd7["status"] == "ccd7_exact_binmap_commissioning_passed"
    assert len(ccd7["completed_cells"]) == 6
    assert all(ccd7["gates"].values())


def test_manifest_detector_coverage_is_complete_and_exact():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    coverage = config["manifest_detector_coverage"]
    assert coverage["ccd_ids"] == [0, 1, 2, 3, 7]
    assert sum(coverage["cell_counts"].values()) == 5082
    assert coverage["cell_counts"]["7"] == 256
    ccd7 = config["ccd7_absent_background_gate"]
    assert ccd7["required_ccd_ids"] == [7]
    assert ccd7["required_observation_contexts"] == [10464, 10888]


def test_recovery_cannot_start_early_or_overlap_the_base():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    terminal = config["inherited_terminal_gate"]
    execution = config["execution"]
    assert terminal["require_base_process_absent"]
    assert terminal["require_frozen_full_interval_final_report"]
    assert not terminal["may_launch_while_base_process_runs"]
    assert execution["base_scratch"] != execution["recovery_scratch"]
    assert PurePosixPath(execution["free_space_probe"]).is_absolute()
    assert execution["maximum_concurrent_cells"] == 2


def test_recovery_archive_rows_are_relabelled(tmp_path: Path):
    path = tmp_path / "index.csv"
    rows = [
        {"archive": "base_v19w", "cell_name": "base"},
        {"archive": "v19w3_recovery", "cell_name": "recovered"},
    ]
    MODULE.relabel_recovery_rows(rows, path)
    with path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert [row["archive"] for row in written] == ["base_v19w", "v19w5_recovery"]


def test_final_gate_schema_and_claim_boundary_are_strict():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert all(config["required_gates"].values())
    assert config["required_gates"]["ccd7_absent_background_commissioning_passes"]
    assert config["final_audit"]["required_unique_manifest_cells"] == 5082
    assert config["final_audit"]["required_product_files"] == 20328
    assert config["integrity"]["v19w4_executed"] is False
    assert config["integrity"]["v19w5_recovery_product_existed_at_freeze"] is False
    claim = " ".join(config["claim_boundary"]).lower()
    assert "does not combine" in claim
    assert "no gas temperature" in claim
    assert "gravity formula enters recovery selection" in claim
