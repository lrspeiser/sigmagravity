from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19w4_hardened_response_recovery.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19w4_hardened_response_recovery.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19w4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules.setdefault("pycrates", types.ModuleType("pycrates"))
SPEC.loader.exec_module(MODULE)


def test_cross_detector_pass_is_a_frozen_required_parent():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    _, _ = MODULE.verify_parents(config)
    report = MODULE.load_json(ROOT / config["parents"]["v19w2b_report"]["path"])
    assert report["status"] == config["cross_detector_gate"]["required_status"]
    assert all(report["gates"].values())
    assert report["v19w4_hardened_recovery_may_be_frozen"]


def test_protected_tree_snapshot_detects_content_and_path_changes(tmp_path: Path):
    base = tmp_path / "base"
    product = base / "completed" / "cell" / "products" / "source.pi"
    product.parent.mkdir(parents=True)
    product.write_bytes(b"alpha")
    before = MODULE.protected_tree_snapshot(base, ["completed", "failed_attempts"])
    product.write_bytes(b"bravo")
    after_content = MODULE.protected_tree_snapshot(base, ["completed", "failed_attempts"])
    assert before["files"] == after_content["files"] == 1
    assert before["bytes"] == after_content["bytes"] == 5
    assert before["path_size_content_sha256"] != after_content["path_size_content_sha256"]
    moved = product.with_name("renamed.pi")
    product.rename(moved)
    after_path = MODULE.protected_tree_snapshot(base, ["completed", "failed_attempts"])
    assert after_content["path_size_content_sha256"] != after_path["path_size_content_sha256"]


def test_inventory_digest_detects_checkpoint_hash_changes():
    key = ("A", 1, 2, 3)
    record = {
        "cell_report_sha256": "a" * 64,
        "four_product_bytes": 4,
        "product_hashes": {role: "b" * 64 for role in MODULE.PRODUCT_ROLES},
    }
    valid = {key: (Path("report.json"), record)}
    before = MODULE.inventory_digest(valid, {})
    changed = json.loads(json.dumps(record))
    changed["product_hashes"]["arf"] = "c" * 64
    after = MODULE.inventory_digest({key: (Path("report.json"), changed)}, {})
    assert before["inventory_sha256"] != after["inventory_sha256"]


def test_recovery_rows_are_relabelled_for_the_actual_archive(tmp_path: Path):
    rows = [
        {"archive": "base_v19w", "cell_name": "base"},
        {"archive": "v19w3_recovery", "cell_name": "recovered"},
    ]
    path = tmp_path / "index.csv"
    MODULE.relabel_recovery_rows(rows, path)
    assert [row["archive"] for row in rows] == ["base_v19w", "v19w4_recovery"]
    assert "v19w4_recovery" in path.read_text(encoding="utf-8")


def test_protocol_requires_double_5082_cell_audit_and_zero_base_mutation():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert not config["inherited_terminal_gate"]["may_launch_while_base_process_runs"]
    assert config["recovery_definition"]["attempts_per_recovery_cell"] == 1
    assert not config["protected_base_audit"]["base_archive_mutation"]
    assert config["final_audit"]["required_unique_manifest_cells"] == 5082
    assert config["final_audit"]["required_product_files"] == 20328
    assert config["final_audit"]["reopen_index_and_revalidate_every_checkpoint_product_and_index_hash"]
    assert not config["final_audit"]["original_v19x_authorized_here"]


def test_runner_hash_and_gate_schema_are_exact():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["execution"]["runner"] == SCRIPT.relative_to(ROOT).as_posix()
    assert MODULE.sha256(SCRIPT) == config["execution"]["runner_sha256"]
    assert len(config["required_gates"]) == 11
    assert all(config["required_gates"].values())
