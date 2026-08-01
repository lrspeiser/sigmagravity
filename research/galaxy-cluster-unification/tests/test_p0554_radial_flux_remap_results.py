import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_constant_radial_remap_protocol_and_outputs_are_complete():
    protocol = load_json("configs/p0554_radial_flux_remap_protocol.json")
    report = load_json("results/p0554_radial_flux_remap/report.json")
    scores = pd.read_csv(ROOT / "results/p0554_radial_flux_remap/candidate_scores.csv")
    assert protocol["status"] == "frozen_before_any_radial-remap_score"
    assert protocol["grid"]["candidate_count"] == 49
    assert report["coverage"] == {
        "candidates": 49,
        "SPARC_galaxies": 131,
        "SPARC_discovery_galaxies": 91,
        "SPARC_formula_holdout_galaxies": 40,
        "CLASH_systems": 20,
        "CLASH_discovery_systems": 13,
        "CLASH_formula_holdout_systems": 7,
        "RXJ1347_pair_heldout_images": 3,
    }
    assert len(scores) == 49
    assert report["selected"]["candidate_id"] == "f001_l080"
    assert report["transfer_gate"]["passed"] is False


def test_constant_remap_exposes_direction_conflict_and_one_effective_coordinate():
    report = load_json("results/p0554_radial_flux_remap/report.json")
    selected = report["selected"]
    assert selected["discovery_gains"]["galaxy"] > 0.0
    assert selected["discovery_gains"]["cluster"] > 0.0
    assert selected["formula_holdout_gains"]["cluster"] < 0.0
    assert selected["RXJ1347_raw_gain"] < 0.0
    assert report["universal_findings"]["all_nonparent_three_domain_discovery_improvers"] == 0
    fits = report["universal_findings"]["two_grid_coordinates_collapse_to_effective_log_shift"]
    assert all(row["R_squared"] > 0.998 for row in fits.values())


def test_multicluster_raw_screen_has_no_all_domain_candidate():
    report = load_json("results/p0554_radial_flux_remap_multicluster_raw/report.json")
    scores = pd.read_csv(ROOT / "results/p0554_radial_flux_remap_multicluster_raw/system_scores.csv")
    assert report["coverage"] == {"candidates": 49, "systems": 5, "heldout_images": 18}
    assert len(scores) == 49 * 5
    assert report["counts"]["galaxy_derived_cluster_RXJ1347_and_five_raw_improvers"] == 0
    preferences = {row["system_label"]: row["preferred_direction"] for row in report["system_direction_preferences"]}
    assert preferences["MACS0329"] == "inward"
    assert preferences["RXJ2129"] == "outward"
    assert report["verdict"]["universal_radial_remap_supported"] is False


def test_forensic_potential_lead_does_not_pass_cross_domain_gate():
    report = load_json("results/p0554_radial_flux_remap_forensics/report.json")
    assert report["coverage"] == {
        "galaxies": 131,
        "derived_clusters": 20,
        "raw_clusters": 6,
        "tested_correlations": 24,
    }
    cluster = [
        row
        for row in report["FDR_significant_features"]
        if row["domain"] == "derived_cluster" and row["feature"] == "median_potential_depth"
    ][0]
    assert cluster["spearman_rho"] > 0.5
    assert cluster["fdr_q_value"] < 0.05
    assert report["same_direction_FDR_features_in_both_large_domains"] == []
    assert report["verdict"]["predeclared_invariant_promotion_gate_passed"] is False


def test_potential_transition_fails_discovery_and_transfer_gates():
    report = load_json("results/p0554_potential_transition_remap/report.json")
    scores = pd.read_csv(ROOT / "results/p0554_potential_transition_remap/scalar_scores.csv")
    assert report["coverage"] == {"amplitudes": 19, "raw_systems": 6, "raw_heldout_images": 21}
    assert len(scores) == 19
    assert report["counts"]["nonparent_discovery_joint_improvers"] == 0
    assert report["selected"]["formula_holdout_gains"]["galaxy"] < 0.0
    assert report["selected"]["formula_holdout_gains"]["cluster"] < 0.0
    assert report["transfer_gate_passed"] is False
    assert report["verdict"]["no_formula_promoted"] is True
