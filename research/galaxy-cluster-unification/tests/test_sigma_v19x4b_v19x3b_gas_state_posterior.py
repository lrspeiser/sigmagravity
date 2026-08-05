from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_sigma_v19x4b_v19x3b_gas_successor_preflight as checker
import freeze_sigma_v19x4b_v19x3b_gas_state_posterior as freezer
import run_sigma_v19x4b_v19x3b_gas_state_posterior as runner

ORIGINAL = ROOT / "configs" / "sigma_v19x4_gas_state_math_preflight.json"
CONFIG = ROOT / "configs" / "sigma_v19x4b_v19x3b_gas_successor_preflight.json"
REPORT = (
    ROOT
    / "results"
    / "sigma_v19x4b_v19x3b_gas_successor_preflight"
    / "report.json"
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_frozen_preflight_report_is_current_and_target_sealed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(REPORT.read_text(encoding="utf-8"))
    rebuilt = checker.execute(config)

    assert frozen == rebuilt
    assert all(frozen["gates"].values())
    assert all(frozen["hash_gates"].values())
    assert not frozen["terminal_regional_or_gas_measurement_opened"]
    assert not frozen["lensing_halo_gravity_or_holdout_payload_opened"]


def build_fixture(root: Path) -> dict[str, Path]:
    paths = {
        "original": root / "configs" / "original_x4.json",
        "x3b_config": root / "configs" / "x3b.json",
        "x3b_runner": root / "scripts" / "x3b.py",
        "x3b_freezer": root / "scripts" / "x3b_freezer.py",
        "x3b_report": root / "results" / "x3b.json",
        "inherited_x4": root / "scripts" / "inherited_x4.py",
        "posterior": root / "src" / "posterior.py",
        "successor": root / "scripts" / "x4b.py",
        "freezer": root / "scripts" / "x4b_freezer.py",
        "output": root / "configs" / "x4b.json",
    }
    for name in (
        "x3b_runner",
        "x3b_freezer",
        "inherited_x4",
        "posterior",
        "successor",
        "freezer",
    ):
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_text(f"# {name}\n", encoding="utf-8")
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    original["parents"] = {}
    write_json(paths["original"], original)
    rel = lambda path: path.relative_to(root).as_posix()
    x3b_config = {
        "freeze_state": freezer.v19x3b.FROZEN_STATE,
        "runtime_authorization": {"response_authority": "V19W5"},
        "implementation": {
            "runner": rel(paths["x3b_runner"]),
            "runner_sha256": freezer.adapter.sha256(paths["x3b_runner"]),
            "freezer": rel(paths["x3b_freezer"]),
            "freezer_sha256": freezer.adapter.sha256(paths["x3b_freezer"]),
        },
    }
    write_json(paths["x3b_config"], x3b_config)
    regions = [
        {"cluster": "BULLET", "bin_id": index} for index in range(366)
    ] + [{"cluster": "ABELL2146", "bin_id": index} for index in range(128)]
    write_json(
        paths["x3b_report"],
        {
            "status": runner.AUTHORIZED_X3B_STATUS,
            "config_sha256": freezer.adapter.sha256(paths["x3b_config"]),
            "runner_sha256": freezer.adapter.sha256(paths["x3b_runner"]),
            "gates": {"all": True},
            "source_map_construction_authorized": True,
            "lensing_or_halo_payload_opened": False,
            "regions": regions,
        },
    )
    return paths


def test_freezer_copies_every_gas_science_section_exactly(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    paths = build_fixture(root)
    monkeypatch.setattr(freezer, "ROOT", root)
    monkeypatch.setattr(freezer, "__file__", str(paths["freezer"]))
    monkeypatch.setattr(freezer.successor, "ROOT", root)
    monkeypatch.setattr(freezer.successor, "__file__", str(paths["successor"]))
    config = freezer.freeze_config(
        paths["original"],
        paths["x3b_config"],
        paths["x3b_runner"],
        paths["x3b_freezer"],
        paths["x3b_report"],
        paths["inherited_x4"],
        paths["posterior"],
        paths["successor"],
        paths["output"],
    )
    original = json.loads(paths["original"].read_text(encoding="utf-8"))

    assert config["freeze_state"] == runner.FROZEN_STATE
    assert config["runtime_authorization"]["v19w5_authority_inherited"]
    for section in freezer.SCIENCE_SECTIONS:
        assert config[section] == original[section]
        assert config["exact_science_section_sha256"][section] == (
            freezer.canonical_sha256(original[section])
        )
    assert json.loads(paths["output"].read_text(encoding="utf-8")) == config


def test_freezer_rejects_incomplete_regional_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    paths = build_fixture(root)
    report = json.loads(paths["x3b_report"].read_text(encoding="utf-8"))
    report["regions"].pop()
    write_json(paths["x3b_report"], report)
    monkeypatch.setattr(freezer, "ROOT", root)

    with pytest.raises(RuntimeError, match="regional inventory changed"):
        freezer.validate_terminal_x3b(
            paths["x3b_config"],
            paths["x3b_runner"],
            paths["x3b_freezer"],
            paths["x3b_report"],
        )


def test_runner_requires_hash_bound_target_sealed_v19x3b(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    paths = build_fixture(root)
    v19m = root / "results" / "v19m.json"
    source = root / "results" / "source.json"
    write_json(v19m, {"fixture": "v19m"})
    write_json(source, {"fixture": "source"})
    rel = lambda path: path.relative_to(root).as_posix()
    config = {
        "freeze_state": runner.FROZEN_STATE,
        "parents": {
            "v19x3b_config": rel(paths["x3b_config"]),
            "v19x3b_config_sha256": runner.sha256(paths["x3b_config"]),
            "v19x3b_runner": rel(paths["x3b_runner"]),
            "v19x3b_runner_sha256": runner.sha256(paths["x3b_runner"]),
            "v19x3b_report": rel(paths["x3b_report"]),
            "v19x3b_report_sha256": runner.sha256(paths["x3b_report"]),
            "v19m_region_report": rel(v19m),
            "v19m_region_report_sha256": runner.sha256(v19m),
            "source_map_report": rel(source),
            "source_map_report_sha256": runner.sha256(source),
        },
        "implementation": {
            "runner": rel(paths["successor"]),
            "runner_sha256": runner.sha256(paths["successor"]),
            "inherited_v19x4_runner": rel(paths["inherited_x4"]),
            "inherited_v19x4_runner_sha256": runner.sha256(paths["inherited_x4"]),
            "posterior_module": rel(paths["posterior"]),
            "posterior_module_sha256": runner.sha256(paths["posterior"]),
        },
    }
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "__file__", str(paths["successor"]))
    x3b, loaded_v19m, loaded_source = runner.validate_preconditions(
        config, paths["x3b_config"], paths["x3b_report"]
    )
    assert x3b["source_map_construction_authorized"]
    assert loaded_v19m == {"fixture": "v19m"}
    assert loaded_source == {"fixture": "source"}

    report = json.loads(paths["x3b_report"].read_text(encoding="utf-8"))
    report["lensing_or_halo_payload_opened"] = True
    write_json(paths["x3b_report"], report)
    config["parents"]["v19x3b_report_sha256"] = runner.sha256(paths["x3b_report"])
    with pytest.raises(RuntimeError, match="opened a prohibited target"):
        runner.validate_preconditions(
            config, paths["x3b_config"], paths["x3b_report"]
        )
