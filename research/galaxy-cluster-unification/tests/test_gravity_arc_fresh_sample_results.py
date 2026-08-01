from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gravity_arc_fresh_sample"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_protocols_and_target_blind_audit_are_locked() -> None:
    acquisition_path = ROOT / "configs" / "gravity_arc_fresh_sample_protocol.json"
    analysis_path = ROOT / "configs" / "gravity_arc_fresh_analysis_protocol.json"
    acquisition = read_json(acquisition_path)
    analysis = read_json(analysis_path)
    provenance = read_json(
        ROOT / "data" / "raw" / "relics_gravity_arc_fresh_sample" / "provenance.json"
    )
    audit = read_json(
        ROOT / "results" / "gravity_arc_fresh_sample_input_audit" / "report.json"
    )
    report = read_json(RESULTS / "report.json")

    assert acquisition["status"] == "frozen_before_download_or_fresh_map_spatial_inspection"
    assert len(acquisition["systems"]) == 10
    assert len(acquisition["locked_candidates"]) == 15
    assert provenance["protocol_sha256"] == digest(acquisition_path)
    assert provenance["files"] == 1_030
    assert provenance["bytes"] > 6_000_000_000
    assert analysis["status"] == "frozen_after_geometry_audit_before_fresh_kappa_pixel_read"
    assert analysis["acquisition_protocol_sha256"] == digest(acquisition_path)
    assert report["analysis_protocol_sha256"] == digest(analysis_path)
    assert audit["status"] == "completed_without_inspecting_fresh_kappa_pixel_values"
    assert audit["coverage_gate_passed"] is True
    assert audit["totals"] == {
        "systems": 10,
        "catalog_rows": 46_917,
        "usable_f160_galaxies_300kpc": 4_281,
        "hard_photoz_members_300kpc": 832,
        "lenstool_range_maps": 1_000,
        "glafic_best_maps": 10,
    }


def test_locked_confirmation_is_complete_and_fails_its_gates() -> None:
    report = read_json(RESULTS / "report.json")
    assert report["coverage"] == {
        "fresh_clusters": 10,
        "hard_photoz_sources": 832,
        "locked_candidates": 15,
        "lenstool_range_maps": 1_000,
        "glafic_best_maps": 10,
        "primary_cluster_candidate_scores": 150,
        "all_target_candidate_scores": 450,
    }
    assert len(read_csv(RESULTS / "scores.csv")) == 450
    assert len(read_csv(RESULTS / "lenstool_uncertainty.csv")) == 150
    assert len(read_csv(RESULTS / "locked_comparisons.csv")) == 30
    assert len(read_csv(RESULTS / "method_disagreement.csv")) == 10
    assert report["gates"]["confirmation_gate_passed"] is False
    assert report["gates"]["method_robustness_gate_passed"] is False
    values = report["gates"]["confirmation_values"]
    assert values["median_improvement_over_LOCAL75"] == pytest.approx(0.0436618266)
    assert values["median_improvement_over_CENTRAL100"] == pytest.approx(-0.0266425340)
    assert values["clusters_better_than_LOCAL75"] == 6
    assert values["clusters_better_than_CENTRAL100"] == 4


def test_geometry_dominates_fraction_and_wider_endpoint_is_exploratory_clue() -> None:
    impacts = {row["candidate_id"]: row for row in read_csv(RESULTS / "variant_impacts.csv")}
    for candidate in ["ISOTROPIC", "NEIGHBOR", "EXTERNAL"]:
        assert float(impacts[candidate]["lenstool_median_delta_JS"]) > 0.0
        assert float(impacts[candidate]["glafic_median_delta_JS"]) > 0.0
    wider = impacts["W060"]
    assert float(wider["lenstool_median_delta_JS"]) < 0.0
    assert float(wider["glafic_median_delta_JS"]) < 0.0
    assert float(wider["lenstool_win_fraction"]) == 0.8
    assert float(wider["glafic_win_fraction"]) == 0.8

    ranking = read_csv(RESULTS / "parameter_impact_ranking.csv")
    assert [row["parameter"] for row in ranking] == [
        "direction",
        "width_kpc",
        "landing_mode",
        "return_scale_kpc",
        "exponent",
        "fraction",
    ]
    assert float(ranking[0]["lenstool_median_JS_span"]) > 50 * float(
        ranking[-1]["lenstool_median_JS_span"]
    )


def test_post_confirmation_driver_search_reports_multiplicity() -> None:
    report = read_json(RESULTS / "driver_report.json")
    assert report["systems"] == 10
    assert report["correlations_tested"] == 117
    assert report["fdr_discoveries_q_le_0_05"] == 0
    top = report["top_correlations"][0]
    assert top["feature"] == "radial_concentration_r50_over_r80"
    assert top["response"] == "lenstool_arc_over_local"
    assert top["spearman_rho"] > 0.85
    assert top["jackknife_same_sign_fraction"] == 1.0
    assert top["benjamini_hochberg_q"] > 0.05
