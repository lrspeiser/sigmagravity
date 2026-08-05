from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_sigma_v19bt_blind_cluster_source_readiness.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19bt_passes_every_preflight_gate() -> None:
    report = MODULE.build_report()
    assert report["decision"] == "passed_source_imaging_preflight_not_holdout_admission"
    assert all(report["gate_results"].values())


def test_v19bt_has_six_balanced_direct_source_systems() -> None:
    report = MODULE.build_report()
    summary = report["source_imaging_preflight"]
    assert summary["shortlist_systems"] == 8
    assert summary["direct_HST_F160W_plus_Chandra_systems"] == 6
    assert summary["direct_relaxed_side"] == 3
    assert summary["direct_disturbed_side"] == 3
    assert summary["direct_mass_span_ratio"] >= 5.0
    assert report["source_image_HEAD_audit"]["urls_checked"] == 28
    assert report["source_image_HEAD_audit"]["http_200_or_206"] == 28


def test_v19bt_does_not_claim_complete_baryons_or_admission() -> None:
    report = MODULE.build_report()
    summary = report["source_imaging_preflight"]
    assert summary["complete_baryon_models"] == 0
    assert summary["admitted_holdouts"] == 0
    assert not summary["final_six_selected"]
    assert all(not row["complete_baryon_model_ready"] for row in report["systems"])


def test_v19bt_keeps_raw_lensing_and_gravity_targets_unused() -> None:
    report = MODULE.build_report()
    boundary = report["access_boundary_audit"]
    assert boundary["temporary_mixed_manuscript_container_removed"]
    assert not boundary["raw_lens_coordinate_values_ingested"]
    assert not boundary["lens_map_downloaded"]
    assert not boundary["gravity_formula_scored"]
    authorization = report["authorization_audit"]
    assert not authorization["open_raw_lensing_coordinates"]
    assert not authorization["download_lens_maps"]
    assert not authorization["select_final_six"]


def test_v19bt_reserves_are_explicit() -> None:
    report = MODULE.build_report()
    assert set(report["source_imaging_preflight"]["reserve_systems"]) == {
        "SDSS_J1002+2031",
        "SDSS_J1226+2149",
    }


def test_v19bt_committed_report_matches_rebuild() -> None:
    expected = MODULE.build_report()
    path = ROOT / "results" / "sigma_v19bt_blind_cluster_source_readiness" / "report.json"
    actual = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected
