import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0554_mass_extent_screen_factorial"


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_factorial_protocol_is_frozen_and_complete():
    protocol = load_json("configs/p0554_mass_extent_screen_factorial_protocol.json")
    report = load_json("results/p0554_mass_extent_screen_factorial/report.json")
    scores = pd.read_csv(RESULTS / "candidate_scores.csv")
    raw = pd.read_csv(RESULTS / "raw_system_scores.csv")
    assert protocol["status"] == "frozen_after-individual-sensitivity-before-any-three-coordinate-score"
    assert protocol["grid"]["candidates"] == 64
    assert len(scores) == 64
    assert len(raw) == 64 * 6
    assert report["coverage"] == {
        "candidates": 64,
        "SPARC_galaxies": 131,
        "CLASH_systems": 20,
        "raw_discovery_systems": 3,
        "raw_holdout_systems": 3,
        "raw_heldout_images": 21,
    }


def test_no_compensated_variant_improves_all_discovery_domains():
    report = load_json("results/p0554_mass_extent_screen_factorial/report.json")
    selected = report["selected"]
    assert selected["candidate_id"] == "m00_e02_n100"
    assert all(value < 0.0 for value in selected["discovery_gains"].values())
    assert report["counts"]["nonparent_positive_all_discovery_domains"] == 0
    assert report["counts"]["nonparent_positive_all_discovery_and_holdout_domains"] == 0
    assert report["transfer_gate_passed"] is False
    assert report["verdict"]["no_formula_promoted"] is True


def test_selected_extent_change_recovers_topology_but_not_accuracy():
    report = load_json("results/p0554_mass_extent_screen_factorial/report.json")
    selected = report["selected"]
    assert selected["raw_holdout_recovered_parent_incomplete"] == "MACS1931"
    assert selected["raw_holdout_complete_systems"] == 3
    assert selected["raw_holdout_converged_roots"] == 14
    assert selected["formula_holdout_gains"]["raw_matched"] < 0.0
    assert selected["solar_pass"] is True


def test_mass_scaling_is_high_impact_but_reverses_between_partitions():
    scores = pd.read_csv(RESULTS / "candidate_scores.csv").set_index("candidate_id")
    parent = scores.loc["m00_e00_n100"]
    high = scores.loc["m03_e00_n100"]
    assert high.discovery_galaxy_gain > parent.discovery_galaxy_gain
    assert high.discovery_cluster_gain < parent.discovery_cluster_gain
    assert high.holdout_cluster_gain > parent.holdout_cluster_gain
    assert high.discovery_raw_gain < -0.10
    assert high.holdout_matched_gain_vs_parent < -10.0


def test_main_mass_effect_dominates_pair_compensation_in_raw_discovery():
    impacts = pd.read_csv(RESULTS / "factor_impacts.csv")
    block = impacts[impacts.metric.eq("discovery_raw_gain")].set_index("effect")
    mass = block.loc["mass_radius_delta", "span"]
    interactions = block[block.effect_type.eq("pair_interaction")].span.max()
    assert np.isclose(mass, 0.10002449012929368)
    assert mass > 7.0 * interactions
