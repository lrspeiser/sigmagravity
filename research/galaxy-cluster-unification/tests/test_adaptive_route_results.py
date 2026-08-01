import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_report(name):
    return json.loads((ROOT / "results" / name / "report.json").read_text(encoding="utf-8"))


def test_adaptive_map_kernel_passed_only_its_map_transfer_gate():
    report = read_report("adaptive_route_kernel")
    assert report["LOOCV_gate_passed"] is True
    assert report["LOOCV_values"]["clusters_better_than_C0351"] == 9
    assert report["all_cluster_selected_candidate"]["candidate_id"] == "A0279"


def test_unit_strength_raw_translation_failed_rxj2129():
    report = read_report("adaptive_route_raw_rxj2129")
    primary = next(row for row in report["scores"] if row["variant"] == "A0279_primary")
    scalar = next(row for row in report["scores"] if row["variant"] == "scalar_baseline")
    assert primary["heldout_RMS_arcsec"] > 5.0 * scalar["heldout_RMS_arcsec"]
    assert report["gates"]["improvement_pass"] is False
    assert report["gates"]["absolute_RMS_pass"] is False


def test_rxj2129_training_selected_bridge_did_not_transfer_to_holdout():
    report = read_report("adaptive_route_amplitude_bridge")
    assert np.isclose(report["selected_training_only_fraction_power"], 0.5)
    selected = next(row for row in report["scores"] if row["variant"] == "selected_bridge")
    assert selected["fractional_heldout_improvement_vs_scalar"] < 0.0
    assert report["gates"]["improvement_pass"] is False


def test_multicluster_route_recovers_root_but_fails_accuracy_and_matched_gain():
    report = read_report("adaptive_route_multicluster_raw")
    matched = report["matched_primary_vs_scalar_all_four"]
    gates = report["gate_audit"]
    assert matched["all_requested_systems_comparable"] is False
    assert matched["matched_complete_systems"] == 3
    assert matched["fractional_improvement"] < 0.01
    assert gates["all_heldout_roots_pass"] is True
    assert gates["absolute_equal_system_RMS_pass"] is False
    assert gates["validation_to_compact_halo_pass"] is False
    assert gates["all_gates_pass"] is False
