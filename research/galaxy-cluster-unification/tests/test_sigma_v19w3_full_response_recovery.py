from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19w3_full_response_recovery.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19w3_full_response_recovery.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19w3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules.setdefault("pycrates", types.ModuleType("pycrates"))
SPEC.loader.exec_module(MODULE)


def test_recovery_workload_is_manifest_minus_independently_valid_base():
    rows = [
        {"cluster": "A", "bin_id": "1", "obsid": "2", "ccd_id": "3"},
        {"cluster": "A", "bin_id": "4", "obsid": "5", "ccd_id": "6"},
    ]
    valid = {("A", 1, 2, 3): (Path("report.json"), {})}
    assert MODULE.missing_manifest_rows(rows, valid) == [rows[1]]


def test_terminal_gate_fails_before_missing_report(tmp_path: Path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="final report is absent"):
        MODULE.validate_base_terminal_state(
            config,
            proc_root=tmp_path / "empty_proc",
            report_path=tmp_path / "missing.json",
        )


def test_live_process_detector_reads_proc_cmdline(tmp_path: Path):
    proc = tmp_path / "proc"
    (proc / "17").mkdir(parents=True)
    (proc / "17" / "cmdline").write_bytes(
        b"python\x00/path/run_sigma_v19w_full_response_production.py\x00"
    )
    (proc / "18").mkdir()
    (proc / "18" / "cmdline").write_bytes(b"python\x00other.py\x00")
    assert MODULE.running_base_processes(proc) == [17]


def test_protocol_preserves_base_and_requires_full_unified_audit():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert not config["base_terminal_gate"]["may_launch_while_base_process_runs"]
    assert config["recovery_definition"]["base_archive_mutation"] is False
    assert config["recovery_definition"]["attempts_per_recovery_cell"] == 1
    assert config["final_audit"]["required_unique_manifest_cells"] == 5082
    assert config["final_audit"]["required_product_files"] == 20328
    assert not config["final_audit"]["original_v19x_authorized_here"]


def test_v19w2_pass_is_frozen_parent_and_runner_hash_is_exact():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    MODULE.verify_static_parents(config)
    report = json.loads(
        (ROOT / config["parents"]["v19w2_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert report["full_missing_cell_recovery_authorized"]
    assert not report["gravity_formula_or_parameter_changed"]
    assert config["execution"]["runner"] == SCRIPT.relative_to(ROOT).as_posix()
    assert MODULE.sha256(SCRIPT) == config["execution"]["runner_sha256"]
