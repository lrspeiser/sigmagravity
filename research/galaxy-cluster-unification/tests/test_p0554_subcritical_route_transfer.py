import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p0554_subcritical_route_transfer_protocol.json"
RESULTS = ROOT / "results" / "p0554_subcritical_route_transfer"


def load_report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_protocol_was_frozen_before_transfer_scores():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert protocol["status"].startswith("frozen_")
    assert "before_any_other_cluster_eta030_refit_or_global_root_score" in protocol["status"]
    assert protocol["formula"]["universal_eta"] == 0.30
    assert protocol["formula"]["eta_parameters_fit"] == 0
    assert protocol["formula"]["new_per_cluster_gravity_parameters"] == 0
    assert protocol["evaluation"]["ordinary_geometry_parameters_refit_per_variant_cluster"] == 6
    assert protocol["evaluation"]["optimization_starts_per_variant_cluster"] == 8


def test_coverage_and_numerical_root_closure():
    report = load_report()
    assert report["report_version"] == "P0554-SUBCRITICAL-ROUTE-TRANSFER-RESULTS-0.1.1"
    assert report["coverage"] == {
        "variants": 2,
        "systems": 5,
        "geometry_fits": 10,
        "optimization_starts": 80,
        "source_families": 27,
        "formula_family_searches": 54,
        "published_images": 77,
        "accepted_global_roots": 158,
    }
    roots = pd.read_csv(RESULTS / "global_roots.csv")
    assert len(roots) == 158
    assert roots.closure_arcsec.max() < 1.0e-6
    assert len(pd.read_csv(RESULTS / "family_summary.csv")) == 54
    assert len(pd.read_csv(RESULTS / "assignments.csv")) == 154
    assert len(pd.read_csv(RESULTS / "heldout_predictions.csv")) == 36


def test_frozen_protocol_label_erratum_is_explicit():
    audit = load_report()["protocol_metadata_audit"]
    assert not audit["labels_match"]
    assert audit["actual_raw_context_labels"] == [
        "RXJ2129", "MACS0329", "MACS0429", "MACS1115", "MACS1931"
    ]
    assert "frozen protocol was not rewritten" in audit["disposition"]


def test_eta030_has_no_family_topology_change():
    report = load_report()
    assert report["changed_family_root_counts"] == []
    variants = pd.read_csv(RESULTS / "variant_summary.csv").set_index("variant_id")
    columns = [
        "families_missing_multiplicity",
        "families_exact_multiplicity",
        "families_demagnified_only_surplus",
        "families_potentially_observable_surplus",
        "potentially_observable_surplus_roots",
        "heldout_roots_converged",
    ]
    assert variants.loc["eta_000", columns].astype(int).tolist() == [8, 12, 1, 6, 7, 17]
    assert variants.loc["eta_030", columns].astype(int).tolist() == [8, 12, 1, 6, 7, 17]
    assert report["primary_topology_changes"] == {
        "additional_potentially_observable_surplus_roots": 0,
        "additional_missing_multiplicity_families": 0,
        "lost_observed_seed_heldout_roots": 0,
    }


def test_primary_transfer_is_small_and_inconsistent():
    report = load_report()
    primary = report["primary_paired_position_comparison"]
    assert primary["common_complete_families"] == 15
    assert np.isclose(primary["eta_000_equal_family_RMS_arcsec"], 6.938756715534171)
    assert np.isclose(primary["eta_030_equal_family_RMS_arcsec"], 6.910175579035084)
    assert np.isclose(primary["eta_030_improvement_fraction"], 0.004119057299573736)
    assert primary["systems_improved"] == 2
    comparisons = pd.read_csv(RESULTS / "paired_position_comparisons.csv").set_index("system_label")
    assert comparisons.loc["MACS0329", "eta_030_improvement_fraction"] > 0
    assert comparisons.loc["MACS1115", "eta_030_improvement_fraction"] > 0
    assert comparisons.loc["MACS0429", "eta_030_improvement_fraction"] < 0
    assert comparisons.loc["RXJ2129", "eta_030_improvement_fraction"] < 0


def test_frozen_strong_gate_fails_but_weak_topology_safe_gate_passes():
    report = load_report()
    assert report["gate_results"] == {
        "aggregate_position_improvement": False,
        "systems_improved": False,
        "no_additional_observable_surplus": True,
        "no_additional_missing_families": True,
        "no_lost_heldout_roots": True,
    }
    assert report["verdict"] == {
        "strong_transfer": False,
        "weak_topology_safe_transfer": True,
        "eta030_universal_formula_promoted": False,
    }


def test_fit_pairing_and_geometry_boundary_counts():
    systems = pd.read_csv(RESULTS / "system_scores.csv")
    assert systems.groupby("variant_id").heldout_roots_converged.sum().to_dict() == {
        "eta_000": 17,
        "eta_030": 17,
    }
    assert systems.groupby("variant_id").geometry_at_boundary.sum().astype(int).to_dict() == {
        "eta_000": 4,
        "eta_030": 4,
    }
    assert systems.route_curl_RMS.max() < 1.0e-12
