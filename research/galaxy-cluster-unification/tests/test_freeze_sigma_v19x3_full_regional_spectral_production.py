from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import freeze_sigma_v19x3_full_regional_spectral_production as freezer


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_freezer_refuses_absent_v19x2_report(tmp_path: Path) -> None:
    x2_config = tmp_path / "x2.json"
    x2_runner = tmp_path / "x2.py"
    v19h = tmp_path / "v19h.json"
    successor = tmp_path / "x3.py"
    for path in (x2_config, x2_runner, v19h, successor):
        path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="freeze parent is absent"):
        freezer.freeze_config(
            x2_config,
            x2_runner,
            tmp_path / "absent.json",
            v19h,
            successor,
            tmp_path / "out.json",
        )


def test_freezer_inherits_494_regions_and_v19h_quality_rule(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    x2_config_path = root / "configs" / "x2.json"
    x2_runner = root / "scripts" / "x2.py"
    successor = root / "scripts" / "x3.py"
    adapter_path = root / "scripts" / "adapter.py"
    freezer_path = root / "scripts" / "freezer.py"
    v19h_path = root / "configs" / "v19h.json"
    for path, content in (
        (x2_runner, "# x2"),
        (successor, "# x3"),
        (adapter_path, "# adapter"),
        (freezer_path, "# freezer"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    x2_config = {
        "freeze_state": "frozen_after_terminal_v19w4_pass",
        "parents": {},
        "runtime_authorization": {
            "required_v19w4_report": "results/w4.json",
            "required_unified_cells": 5082,
            "required_unified_products": 20328,
        },
        "execution": {
            "response_archives": {
                "base_v19w": "/base",
                "v19w4_recovery": "/recovery",
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
            "runner": "scripts/x2.py",
            "runner_sha256": freezer.adapter.sha256(x2_runner),
            "adapter": "scripts/adapter.py",
            "adapter_sha256": freezer.adapter.sha256(adapter_path),
        },
    }
    write_json(x2_config_path, x2_config)
    x2_report = {
        "status": freezer.successor.AUTHORIZED_X2_STATUS,
        "config_sha256": freezer.adapter.sha256(x2_config_path),
        "runner_sha256": freezer.adapter.sha256(x2_runner),
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
    }
    x2_report_path = root / "results" / "x2.json"
    write_json(x2_report_path, x2_report)
    write_json(
        v19h_path,
        {
            "adaptive_thermodynamics": {
                "fit_gates": {"minimum_passing_regions_per_cluster": 12}
            }
        },
    )
    output = root / "configs" / "x3.json"
    monkeypatch.setattr(freezer, "ROOT", root)
    monkeypatch.setattr(freezer, "__file__", str(freezer_path))
    monkeypatch.setattr(freezer.successor, "ROOT", root)
    monkeypatch.setattr(freezer.successor, "__file__", str(successor))
    config = freezer.freeze_config(
        x2_config_path,
        x2_runner,
        x2_report_path,
        v19h_path,
        successor,
        output,
    )
    assert config["freeze_state"] == "frozen_after_terminal_v19x2_pass"
    assert config["regional_gates"]["expected_total_regions"] == 494
    assert config["runtime_authorization"]["required_completed_cells"] == 5082
    assert config["regional_gates"]["minimum_quality_passes_per_cluster"] == 12
    assert config["regional_gates"][
        "every_region_requires_finite_temperature_abundance_and_normalization_best_fit"
    ]
    assert not config["authorization"]["open_lensing_or_halo_payload"]
    assert json.loads(output.read_text(encoding="utf-8")) == config
