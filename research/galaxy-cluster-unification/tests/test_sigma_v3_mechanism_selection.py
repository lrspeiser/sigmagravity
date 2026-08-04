from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_no_mechanism_is_prematurely_frozen() -> None:
    output = ROOT / "results" / "sigma_v3_mechanism_selection"
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    matrix = pd.read_csv(output / "mechanism_matrix.csv")
    assert report["ready_to_freeze_mechanisms"] == []
    assert not matrix.hard_gate_pass.any()
    assert not report["selection"]["action_frozen"]
    assert (
        report["selection"]["preferred_derivation_target"]
        == "degenerate_baryon_forced_tidal_geometry"
    )


def test_naive_localized_pair_has_one_negative_kinetic_eigenvalue() -> None:
    report = json.loads(
        (
            ROOT / "results" / "sigma_v3_mechanism_selection" / "report.json"
        ).read_text(encoding="utf-8")
    )
    audit = report["naive_localization_audit"]
    assert np.allclose(audit["eigenvalues"], [-1.0, 1.0])
    assert not audit["positive_definite"]


def test_current_only_vector_is_velocity_suppressed() -> None:
    report = json.loads(
        (
            ROOT / "results" / "sigma_v3_mechanism_selection" / "report.json"
        ).read_text(encoding="utf-8")
    )
    audit = report["baryon_current_scaling_audit"]
    assert audit["generous_cluster_speed_over_c"] < 0.006
    assert audit["minimum_linear_coupling_for_order_unity_response"] > 190.0
    assert audit["minimum_quadratic_coupling_for_order_unity_response"] > 39000.0


def test_stop_rule_counts_two_action_level_topology_failures() -> None:
    report = json.loads(
        (
            ROOT / "results" / "sigma_v3_mechanism_selection" / "report.json"
        ).read_text(encoding="utf-8")
    )
    failures = report["completed_action_level_raw_topology_failures"]
    assert len(failures) == 2
    assert all(not row["raw_cluster_pass"] for row in failures)
    assert report["completed_same_gate_failure_count"] == 2
    assert report["remaining_failures_before_mandatory_rethink"] == 1
