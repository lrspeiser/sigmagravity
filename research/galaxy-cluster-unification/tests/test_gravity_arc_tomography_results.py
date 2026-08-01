from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gravity_arc_tomography"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_frozen_protocol_and_input_coverage() -> None:
    protocol_path = ROOT / "configs" / "gravity_arc_tomography_protocol.json"
    protocol = read_json(protocol_path)
    report = read_json(RESULTS / "report.json")
    audit = read_json(ROOT / "results" / "gravity_arc_tomography_input_audit" / "report.json")

    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert protocol["protocol_version"] == "GRAVITY-ARC-TOMOGRAPHY-0.1.0"
    assert protocol["status"] == "frozen_before_catalog_to_kappa_spatial_correlation"
    assert report["protocol_sha256"] == digest
    assert audit["status"] == "completed_without_inspecting_catalog_to_kappa_spatial_correlation"
    assert audit["totals"] == {
        "catalog_rows": 7_883,
        "usable_f160_galaxies": 2_275,
        "hard_photoz_members": 360,
        "lensing_realizations": 300,
    }


def test_completed_result_coverage_and_inverse_conservation() -> None:
    report = read_json(RESULTS / "report.json")
    assert report["report_version"] == "GRAVITY-ARC-TOMOGRAPHY-0.1.0"
    assert report["coverage"] == {
        "clusters": 3,
        "sources": 1_423,
        "inverse_entropy_scales": 3,
        "forward_candidates": 1_571,
        "forward_cluster_scores": 4_713,
        "hard_source_forward_cluster_scores": 4_713,
        "leave_one_cluster_out_folds": 3,
        "validation_kappa_realizations_per_fold": 100,
    }
    assert len(read_csv(RESULTS / "forward_grid.csv")) == 4_713
    assert len(read_csv(RESULTS / "hard_source_forward_grid.csv")) == 4_713
    assert len(read_csv(RESULTS / "fold_results.csv")) == 21
    assert len(read_csv(RESULTS / "validation_uncertainty.csv")) == 21

    inverse = report["inverse_primary"]
    assert len(inverse) == 3
    assert all(50.0 < row["median_path_kpc"] < 100.0 for row in inverse)
    assert all(row["source_marginal_max_error"] < 1e-8 for row in inverse)
    assert all(row["target_marginal_max_error"] < 1e-8 for row in inverse)
    assert report["useful_spatial_law_gate_passed"] is False


def test_hard_member_clue_is_universal_but_not_a_gate_pass() -> None:
    report = read_json(RESULTS / "report.json")
    winners = [
        row
        for row in report["sensitivity_overall_winners"]
        if row["sensitivity"] == "hard_photoz_members"
    ]
    assert len(winners) == 3
    assert {row["candidate_id"] for row in winners} == {"C0351"}
    assert {row["family"] for row in winners} == {"center_return"}
    for row in winners:
        assert row["fraction"] == 0.5
        assert row["return_scale_kpc"] == 250.0
        assert row["exponent"] == -0.5
        assert row["width_kpc"] == 50.0
        assert row["landing_mode"] == "endpoint"

    rows = read_csv(RESULTS / "sensitivity_fold_results.csv")
    hard = [row for row in rows if row["sensitivity"] == "hard_photoz_members"]
    by_key = {
        (row["validation_system"], row["selection_scope"], row["family"]): row
        for row in hard
    }
    beats_local = []
    beats_central = []
    for winner in winners:
        system = winner["validation_system"]
        local = by_key[(system, "within_family", "local_gaussian")]
        central = by_key[(system, "within_family", "central_halo_null")]
        beats_local.append(winner["validation_JS"] < float(local["validation_JS"]))
        beats_central.append(winner["validation_JS"] < float(central["validation_JS"]))

    assert beats_local == [True, True, True]
    assert beats_central == [True, False, True]


def test_unsubtracted_target_prefers_only_null_families() -> None:
    report = read_json(RESULTS / "report.json")
    winners = [
        row
        for row in report["sensitivity_overall_winners"]
        if row["sensitivity"] == "unsubtracted_positive_kappa"
    ]
    assert [row["family"] for row in winners] == [
        "central_halo_null",
        "local_gaussian",
        "central_halo_null",
    ]
