from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_sigma_v19x3b_v19w5_regional_successor_preflight as checker
import freeze_sigma_v19x3b_v19w5_full_regional_spectral_production as freezer
import run_sigma_v19x3b_v19w5_full_regional_spectral_production as runner

CONFIG = ROOT / "configs" / "sigma_v19x3b_v19w5_regional_successor_preflight.json"
REPORT = (
    ROOT
    / "results"
    / "sigma_v19x3b_v19w5_regional_successor_preflight"
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
    assert not frozen["terminal_v19x2_or_regional_measurement_opened"]
    assert not frozen["lensing_halo_or_gravity_payload_opened"]


def build_terminal_fixture(root: Path) -> dict[str, Path]:
    paths = {
        "x2_config": root / "configs" / "x2.json",
        "x2_runner": root / "scripts" / "x2.py",
        "x2_report": root / "results" / "x2.json",
        "v19h": root / "configs" / "v19h.json",
        "legacy_x3": root / "scripts" / "legacy_x3.py",
        "successor": root / "scripts" / "x3b.py",
        "freezer": root / "scripts" / "freezer.py",
        "adapter": root / "scripts" / "adapter.py",
        "w5_config": root / "configs" / "w5.json",
        "w5_runner": root / "scripts" / "w5.py",
        "w5_report": root / "results" / "w5.json",
        "w5_index": root / "results" / "w5_index.csv",
        "output": root / "configs" / "x3b.json",
    }
    for name in (
        "x2_runner",
        "legacy_x3",
        "successor",
        "freezer",
        "adapter",
        "w5_runner",
    ):
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_text(f"# {name}\n", encoding="utf-8")
    write_json(paths["w5_config"], {"protocol": "w5"})
    write_json(paths["w5_report"], {"status": "fixture"})
    paths["w5_index"].write_text("fixture\n", encoding="utf-8")
    write_json(
        paths["v19h"],
        {
            "adaptive_thermodynamics": {
                "fit_gates": {"minimum_passing_regions_per_cluster": 12}
            }
        },
    )
    rel = lambda path: path.relative_to(root).as_posix()
    x2_config = {
        "freeze_state": "frozen_after_terminal_v19w5_pass",
        "parents": {
            "v19w5_config": rel(paths["w5_config"]),
            "v19w5_config_sha256": freezer.adapter.sha256(paths["w5_config"]),
            "v19w5_runner": rel(paths["w5_runner"]),
            "v19w5_runner_sha256": freezer.adapter.sha256(paths["w5_runner"]),
            "v19w5_report": rel(paths["w5_report"]),
            "v19w5_report_sha256": freezer.adapter.sha256(paths["w5_report"]),
            "v19w5_unified_index": rel(paths["w5_index"]),
            "v19w5_unified_index_sha256": freezer.adapter.sha256(
                paths["w5_index"]
            ),
        },
        "runtime_authorization": {
            "response_authority": "V19W5",
            "required_response_report": rel(paths["w5_report"]),
            "required_status": freezer.adapter.V19W5_AUTHORIZED_STATUS,
            "recovery_archive": "v19w5_recovery",
            "required_unified_cells": 5082,
            "required_unified_products": 20328,
        },
        "execution": {
            "response_archives": {
                "base_v19w": "/base",
                "v19w5_recovery": "/recovery",
            }
        },
        "registered_workload": {
            "clusters": {
                "BULLET": {"total_regions": 366, "total_cells": 3812},
                "ABELL2146": {"total_regions": 128, "total_cells": 1270},
            }
        },
        "combination": {"method": "sum"},
        "fit_sequence": {"model": "xstbabs * xsapec"},
        "gates": {"both_integrated_reduced_statistics_at_most": 1.5},
        "implementation": {
            "runner": rel(paths["x2_runner"]),
            "runner_sha256": freezer.adapter.sha256(paths["x2_runner"]),
            "adapter": rel(paths["adapter"]),
            "adapter_sha256": freezer.adapter.sha256(paths["adapter"]),
        },
    }
    write_json(paths["x2_config"], x2_config)
    write_json(
        paths["x2_report"],
        {
            "status": runner.AUTHORIZED_X2_STATUS,
            "config_sha256": freezer.adapter.sha256(paths["x2_config"]),
            "runner_sha256": freezer.adapter.sha256(paths["x2_runner"]),
            "gates": {"all": True},
            "full_494_region_combination_and_fit_authorized": True,
            "replacement_cluster_lensing_target_opened": False,
            "integrated_fits": [
                {
                    "cluster": cluster,
                    "fit_completed": True,
                    "gates": {"all_passed": True},
                    "parameters": {"abundance_solar": abundance},
                }
                for cluster, abundance in (("BULLET", 0.3), ("ABELL2146", 0.2))
            ],
        },
    )
    return paths


def test_freezer_refuses_absent_v19x2_report(tmp_path: Path) -> None:
    paths = build_terminal_fixture(tmp_path / "root")
    paths["x2_report"].unlink()
    with pytest.raises(RuntimeError, match="freeze parent is absent"):
        freezer.freeze_config(
            paths["x2_config"],
            paths["x2_runner"],
            paths["x2_report"],
            paths["v19h"],
            paths["legacy_x3"],
            paths["successor"],
            paths["output"],
        )


def test_freezer_inherits_494_regions_and_v19w5_authority(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    paths = build_terminal_fixture(root)
    monkeypatch.setattr(freezer, "ROOT", root)
    monkeypatch.setattr(freezer, "__file__", str(paths["freezer"]))
    monkeypatch.setattr(freezer.successor, "ROOT", root)
    monkeypatch.setattr(freezer.successor, "__file__", str(paths["successor"]))
    config = freezer.freeze_config(
        paths["x2_config"],
        paths["x2_runner"],
        paths["x2_report"],
        paths["v19h"],
        paths["legacy_x3"],
        paths["successor"],
        paths["output"],
    )

    assert config["freeze_state"] == runner.FROZEN_STATE
    assert config["runtime_authorization"]["response_authority"] == "V19W5"
    assert config["runtime_authorization"]["recovery_archive"] == "v19w5_recovery"
    assert config["regional_gates"]["expected_total_regions"] == 494
    assert config["regional_gates"]["minimum_quality_passes_per_cluster"] == 12
    assert config["implementation"]["inherited_v19x3_runner_sha256"] == (
        freezer.adapter.sha256(paths["legacy_x3"])
    )
    assert json.loads(paths["output"].read_text(encoding="utf-8")) == config


def test_freezer_rejects_a_v19w4_authority(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    paths = build_terminal_fixture(root)
    config = json.loads(paths["x2_config"].read_text(encoding="utf-8"))
    config["runtime_authorization"]["response_authority"] = "V19W4"
    write_json(paths["x2_config"], config)
    report = json.loads(paths["x2_report"].read_text(encoding="utf-8"))
    report["config_sha256"] = freezer.adapter.sha256(paths["x2_config"])
    write_json(paths["x2_report"], report)
    monkeypatch.setattr(freezer, "ROOT", root)

    with pytest.raises(RuntimeError, match="does not preserve V19W5 authority"):
        freezer.validate_terminal_x2(
            paths["x2_config"], paths["x2_runner"], paths["x2_report"]
        )


def test_runner_propagates_explicit_v19w5_authority(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    x2_report = root / "results" / "x2.json"
    response_report = root / "results" / "w5.json"
    for path in (x2_report, response_report):
        write_json(path, {})
    config = {
        "parents": {
            "v19w5_config_sha256": "a" * 64,
            "v19w5_runner_sha256": "b" * 64,
        },
        "runtime_authorization": {
            "required_v19x2_report": "results/x2.json",
            "required_v19x2_config_sha256": "c" * 64,
            "required_v19x2_runner_sha256": "d" * 64,
            "required_response_report": "results/w5.json",
            "required_response_status": runner.adapter.V19W5_AUTHORIZED_STATUS,
            "response_authority": "V19W5",
            "recovery_archive": "v19w5_recovery",
            "required_unified_cells": 1,
            "required_unified_products": 4,
        },
        "execution": {
            "response_archives": {
                "base_v19w": str(root / "base"),
                "v19w5_recovery": str(root / "recovery"),
            }
        },
        "implementation": {"inherited_v19x3_runner_sha256": "e" * 64},
    }
    authorization: dict = {}
    validation: dict = {}
    index = root / "results" / "index.csv"
    index.write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(
        runner.inherited_v19x3,
        "validate_x2_authorization",
        lambda *_args: (
            {"status": runner.AUTHORIZED_X2_STATUS},
            {"BULLET": 0.3, "ABELL2146": 0.2},
        ),
    )

    def authorize(*_args, **kwargs):
        authorization.update(kwargs)
        return {"unified_product_index": {"sha256": "f" * 64}}, index

    def validate(*_args, **kwargs):
        validation.update(kwargs)
        return {("BULLET", 1, 2, 3): {"fixture": True}}

    monkeypatch.setattr(runner.adapter, "authorize_unified_index", authorize)
    monkeypatch.setattr(runner.adapter, "validate_unified_archive", validate)
    monkeypatch.setattr(
        runner.inherited_v19x3.inherited_v19x,
        "load_manifest",
        lambda _config: [{"fixture": "manifest"}],
    )
    monkeypatch.setattr(
        runner.inherited_v19x3,
        "build_full_region_plan",
        lambda *_args: {"fixture": "plan"},
    )
    monkeypatch.setattr(
        runner.inherited_v19x3,
        "run_full_regional_production",
        lambda *_args: {
            "status": "fixture_pass",
            "gates": {"all": True},
            "source_map_construction_authorized": True,
        },
    )
    result = runner.execute(config, root / "output", root / "scratch")

    assert authorization["expected_status"] == runner.adapter.V19W5_AUTHORIZED_STATUS
    assert authorization["authority_label"] == "V19W5"
    assert validation["recovery_archive"] == "v19w5_recovery"
    assert result["source_map_construction_authorized"]
