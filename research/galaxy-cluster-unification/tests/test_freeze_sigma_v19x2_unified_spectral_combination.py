from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import freeze_sigma_v19x2_unified_spectral_combination as freezer


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def legacy_fixture() -> dict:
    return {
        "parents": {"v19u_manifest": "manifest.csv", "v19u_manifest_sha256": "1" * 64},
        "execution": {
            "combination_concurrency": 1,
            "reason_for_serial_combination": "frozen serial rule",
        },
        "registered_workload": {"clusters": {"A": {"total_cells": 5082}}},
        "combination": {"method": "sum", "group_after_combination": {"minimum_counts": 25}},
        "fit_sequence": {"model": "xstbabs * xsapec", "fit_energy_keV": [0.5, 7.0]},
        "gates": {"all_free_parameters_strictly_inside_bounds": True},
    }


def terminal_fixture(root: Path, config: Path, runner: Path) -> tuple[Path, Path]:
    index = root / "results" / "unified.csv"
    index.parent.mkdir(parents=True)
    index.write_text("fixture\n", encoding="utf-8")
    report = {
        "status": freezer.adapter.V19W5_AUTHORIZED_STATUS,
        "config_sha256": freezer.adapter.sha256(config),
        "runner_sha256": freezer.adapter.sha256(runner),
        "gates": {"all": True},
        "unified_cells": 5082,
        "unified_product_files": 20328,
        "base_v19w_archive_modified": False,
        "original_v19x_authorized": False,
        "v19x_successor_configuration_may_be_frozen": True,
        "unified_product_index": {
            "path": "results/unified.csv",
            "rows": 5082,
            "bytes": index.stat().st_size,
            "sha256": freezer.adapter.sha256(index),
        },
    }
    report_path = root / "results" / "report.json"
    write_json(report_path, report)
    return report_path, index


def test_freezer_refuses_absent_terminal_report(tmp_path: Path) -> None:
    paths = [
        tmp_path / name
        for name in ("legacy.json", "w5.json", "w5.py", "x2.py", "adapter.py")
    ]
    for path in paths:
        path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="terminal authorization report is absent"):
        freezer.freeze_config(*paths[:3], tmp_path / "absent.json", *paths[3:], tmp_path / "out.json")


def test_freezer_copies_scientific_rules_exactly_after_terminal_pass(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    legacy = root / "configs" / "legacy.json"
    w5_config = root / "configs" / "w5.json"
    w5_runner = root / "scripts" / "w5.py"
    successor_runner = root / "scripts" / "x2.py"
    adapter_path = root / "scripts" / "adapter.py"
    freezer_path = root / "scripts" / "freezer.py"
    manifest = root / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("fixture\n", encoding="utf-8")
    payload = legacy_fixture()
    payload["parents"]["v19u_manifest_sha256"] = freezer.adapter.sha256(manifest)
    write_json(legacy, payload)
    write_json(w5_config, {"protocol": "w5"})
    w5_runner.parent.mkdir(parents=True, exist_ok=True)
    w5_runner.write_text("# w5", encoding="utf-8")
    successor_runner.write_text("# x2", encoding="utf-8")
    adapter_path.write_text("# adapter", encoding="utf-8")
    freezer_path.write_text("# freezer", encoding="utf-8")
    report, index = terminal_fixture(root, w5_config, w5_runner)
    output = root / "configs" / "x2.json"
    monkeypatch.setattr(freezer, "ROOT", root)
    monkeypatch.setattr(freezer, "__file__", str(freezer_path))
    monkeypatch.setattr(freezer.successor, "ROOT", root)
    monkeypatch.setattr(freezer.successor, "validate_frozen_runner", lambda _config: None)
    config = freezer.freeze_config(
        legacy,
        w5_config,
        w5_runner,
        report,
        successor_runner,
        adapter_path,
        output,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == config
    assert config["freeze_state"] == "frozen_after_terminal_v19w5_pass"
    assert config["runtime_authorization"]["required_unified_cells"] == 5082
    assert config["runtime_authorization"]["required_unified_products"] == 20328
    assert config["parents"]["v19w5_unified_index_sha256"] == freezer.adapter.sha256(index)
    assert config["runtime_authorization"]["recovery_archive"] == "v19w5_recovery"
    assert config["integrity"]["obsolete_v19x_authorized"] is False
    for section in freezer.EXACT_LEGACY_SECTIONS:
        assert config[section] == payload[section]
        assert config["inherited_section_sha256"][section] == freezer.canonical_sha256(
            payload[section]
        )
    freezer.successor.validate_frozen_parents_and_inheritance(config)
