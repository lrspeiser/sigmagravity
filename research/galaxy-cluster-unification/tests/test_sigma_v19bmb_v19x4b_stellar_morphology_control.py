from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_sigma_v19bmb_v19x4b_stellar_successor_preflight as checker
import freeze_sigma_v19bmb_v19x4b_stellar_morphology_control as freezer
import run_sigma_v19bmb_v19x4b_stellar_morphology_control as runner

CONFIG = ROOT / "configs" / "sigma_v19bmb_v19x4b_stellar_successor_preflight.json"
REPORT = ROOT / "results" / "sigma_v19bmb_v19x4b_stellar_successor_preflight" / "report.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_frozen_preflight_report_is_current_and_target_sealed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(REPORT.read_text(encoding="utf-8"))
    rebuilt = checker.execute(config, CONFIG)
    assert frozen == rebuilt
    assert all(frozen["gates"].values())
    assert all(frozen["hash_gates"].values())
    assert not frozen["terminal_gas_or_stellar_product_opened"]
    assert not frozen["source_lensing_halo_gravity_or_holdout_payload_opened"]


def test_runner_rejects_target_opened_x4b_report(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    config_path = root / "configs" / "x4b.json"
    runner_path = root / "scripts" / "x4b.py"
    report_path = root / "results" / "x4b.json"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("# runner\n", encoding="utf-8")
    write_json(config_path, {"fixture": True})
    products = []
    for index in range(12):
        path = root / "products" / f"{index}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]))
        products.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": 1,
                "sha256": runner.sha256(path),
            }
        )
    write_json(
        report_path,
        {
            "status": runner.AUTHORIZED_X4B_STATUS,
            "config_sha256": runner.sha256(config_path),
            "runner_sha256": runner.sha256(runner_path),
            "source_invariant_scoring_authorized": True,
            "gates": {"all": True},
            "lensing_or_halo_payload_opened": False,
            "products": products,
        },
    )
    config = {
        "parents": {
            "v19x4b_config": {"path": "configs/x4b.json", "sha256": runner.sha256(config_path)},
            "v19x4b_runner": {"path": "scripts/x4b.py", "sha256": runner.sha256(runner_path)},
        }
    }
    monkeypatch.setattr(runner, "ROOT", root)
    assert runner.validate_x4b_report(config, report_path)["source_invariant_scoring_authorized"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["lensing_or_halo_payload_opened"] = True
    write_json(report_path, report)
    with pytest.raises(RuntimeError, match="target-sealed"):
        runner.validate_x4b_report(config, report_path)


def test_freezer_copies_stellar_science_sections_exactly(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    original_path = root / "configs" / "bm.json"
    original_runner = root / "scripts" / "bm.py"
    stellar_module = root / "src" / "stellar.py"
    x4b_config = root / "configs" / "x4b.json"
    x4b_runner = root / "scripts" / "x4b.py"
    x4b_freezer = root / "scripts" / "x4b_freezer.py"
    x4b_report = root / "results" / "x4b.json"
    successor = root / "scripts" / "bmb.py"
    freezer_path = root / "scripts" / "bmb_freezer.py"
    output = root / "configs" / "bmb.json"
    for path in (original_runner, stellar_module, x4b_runner, x4b_freezer, successor, freezer_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {path.name}\n", encoding="utf-8")
    original = json.loads((ROOT / "configs" / "sigma_v19bm_stellar_morphology_control.json").read_text(encoding="utf-8"))
    original["parents"] = {}
    write_json(original_path, original)
    rel = lambda path: path.relative_to(root).as_posix()
    write_json(
        x4b_config,
        {
            "freeze_state": "frozen_after_terminal_v19x3b_pass",
            "implementation": {
                "runner": rel(x4b_runner), "runner_sha256": runner.sha256(x4b_runner),
                "freezer": rel(x4b_freezer), "freezer_sha256": runner.sha256(x4b_freezer),
            },
        },
    )
    write_json(
        x4b_report,
        {
            "status": runner.AUTHORIZED_X4B_STATUS,
            "config_sha256": runner.sha256(x4b_config),
            "runner_sha256": runner.sha256(x4b_runner),
            "source_invariant_scoring_authorized": True,
            "gates": {"all": True},
            "lensing_or_halo_payload_opened": False,
            "products": [{} for _ in range(12)],
        },
    )
    monkeypatch.setattr(freezer, "ROOT", root)
    monkeypatch.setattr(freezer, "__file__", str(freezer_path))
    monkeypatch.setattr(freezer.successor, "ROOT", root)
    monkeypatch.setattr(freezer.successor, "__file__", str(successor))
    config = freezer.freeze_config(
        original_path, original_runner, stellar_module, x4b_config, x4b_runner,
        x4b_freezer, x4b_report, successor, output,
    )
    for section in freezer.SCIENCE_SECTIONS:
        assert config[section] == original[section]
    assert json.loads(output.read_text(encoding="utf-8")) == config
