from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def payload() -> dict:
    return json.loads((ROOT / "results" / "sigma_v19cn_fine_wmap_final_recovery" / "report.json").read_text(encoding="utf-8"))


def test_v19cn_full_recovery_and_source_chain_pass() -> None:
    report = payload()
    assert report["status"] == "fine_wmap_edge_recovery_and_target_sealed_source_chain_complete"
    assert report["decision"] == "run_frozen_v19bs_disposition_next"
    assert all(report["gate_results"].values())


def test_v19cn_changes_only_wmap_resolution_and_passes_original_cell_gates() -> None:
    edge = payload()["edge_recovery"]
    assert len(edge["command_changes"]) == 1
    assert edge["command_changes"][0]["before"] == "binwmap=det=8"
    assert edge["command_changes"][0]["after"] == "binwmap=det=1"
    assert all(edge["original_gates"].values())
    assert edge["failed_attempt2_preserved"]


def test_v19cn_full_archive_counts_and_no_physics_access() -> None:
    report = payload()
    assert report["v19w5_summary"]["unified_cells"] == 5082
    assert report["v19w5_summary"]["unified_product_files"] == 20328
    boundary = report["authorization_boundary"]
    assert not boundary["v19bs_run_here"]
    assert not boundary["action_derived"]
    assert not boundary["lensing_halo_gravity_or_holdout_opened"]
    assert not boundary["solar_optimized"]
