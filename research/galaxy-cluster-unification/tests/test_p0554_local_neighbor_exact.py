import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0554_local_neighbor_exact_protocol.json"
RESULTS = ROOT / "results/p0554_local_neighbor_exact"


def load_report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_protocol_was_frozen_after_screen_before_exact_scores():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert protocol["status"].startswith("frozen_")
    assert "before_any_local_soft_200_geometry_refit_or_global_root_score" in protocol["status"]
    assert protocol["candidate"]["local_mix"] == 1.0
    assert protocol["candidate"]["softening_kpc"] == 200.0
    assert protocol["candidate"]["new_per_cluster_gravity_parameters"] == 0
    assert protocol["evaluation"]["optimization_starts_per_cluster"] == 8


def test_candidate_coverage_and_root_closure():
    report = load_report()
    assert report["coverage"] == {
        "candidate_geometry_fits": 5,
        "candidate_optimization_starts": 40,
        "candidate_formula_family_searches": 27,
        "candidate_accepted_global_roots": 79,
        "comparison_variants": 3,
        "systems": 5,
        "source_families": 27,
        "published_images": 77,
    }
    roots = pd.read_csv(RESULTS / "candidate_global_roots.csv")
    assert len(roots) == 79
    assert roots.closure_arcsec.max() < 1e-6
    assert len(pd.read_csv(RESULTS / "candidate_family_summary.csv")) == 27
    assert len(pd.read_csv(RESULTS / "candidate_assignments.csv")) == 77
    assert len(pd.read_csv(RESULTS / "candidate_heldout_predictions.csv")) == 18


def test_local_neighbor_beats_global_but_misses_one_percent_gate():
    position = load_report()["primary_common_family_position_comparison"]
    assert position["common_complete_families"] == 15
    assert np.isclose(position["eta_000_equal_family_RMS_arcsec"], 6.938756715534171)
    assert np.isclose(position["global_centroid_equal_family_RMS_arcsec"], 6.910175579035084)
    assert np.isclose(position["local_soft_200_equal_family_RMS_arcsec"], 6.88705309694174)
    assert np.isclose(position["local_improvement_fraction_vs_eta0"], 0.007451424039220167)
    assert np.isclose(position["local_improvement_fraction_vs_global"], 0.0033461497220845793)
    assert not load_report()["gate_results"]["assignment_improvement_vs_eta0"]
    assert load_report()["gate_results"]["beats_global_centroid"]


def test_exact_heldout_improves_in_all_four_transfer_clusters():
    report = load_report()
    assert report["primary_exact_heldout_systems_comparable"] == [
        "RXJ2129", "MACS0329", "MACS0429", "MACS1115"
    ]
    assert report["primary_exact_heldout_systems_improved"] == 4
    systems = pd.read_csv(RESULTS / "comparison_system_scores.csv")
    pivot = systems.pivot_table(
        index="system_label", columns="variant_id", values="heldout_RMS_arcsec"
    )
    for label in ("RXJ2129", "MACS0329", "MACS0429", "MACS1115"):
        assert pivot.loc[label, "local_soft_200"] < pivot.loc[label, "eta_000"]


def test_all_27_family_root_counts_are_identical():
    report = load_report()
    assert report["changed_family_root_counts"] == []
    summary = pd.read_csv(RESULTS / "variant_summary.csv").set_index("variant_id")
    columns = [
        "families_missing_multiplicity",
        "families_exact_multiplicity",
        "families_demagnified_only_surplus",
        "families_potentially_observable_surplus",
        "potentially_observable_surplus_roots",
        "heldout_roots_converged",
    ]
    expected = [8, 12, 1, 6, 7, 17]
    for variant in ("eta_000", "global_centroid", "local_soft_200"):
        assert summary.loc[variant, columns].astype(int).tolist() == expected
    assert report["primary_topology_changes_vs_eta0"] == {
        "additional_primary_observable_surplus_roots": 0,
        "additional_primary_missing_families": 0,
        "lost_primary_observed_seed_heldout_roots": 0,
    }


def test_result_is_surviving_direction_not_promoted_formula():
    report = load_report()
    assert report["gate_results"] == {
        "assignment_improvement_vs_eta0": False,
        "heldout_systems_improved": True,
        "beats_global_centroid": True,
        "no_additional_observable_surplus": True,
        "no_additional_missing_families": True,
        "no_lost_heldout_roots": True,
    }
    assert report["verdict"] == {
        "strong_exact_survival": False,
        "local_neighbor_beats_global_centroid_after_exact_refit": True,
        "no_formula_promoted": True,
    }
    geometry = pd.read_csv(RESULTS / "candidate_geometry.csv")
    assert int(geometry.geometry_at_boundary.astype(bool).sum()) == 4
