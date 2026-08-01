import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0611_frozen_dual_misalignment_raw_transfer"


def test_frozen_gate_produces_distinct_activations_without_retuning():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    directions = pd.read_csv(OUTPUT / "direction_audits.csv").set_index("system_label")
    assert report["coverage"]["systems"] == 2
    assert report["coverage"]["member_sources"] == 124
    assert directions.loc["A383", "candidate_gate_H"] < 1.0e-6
    assert 0.5 < directions.loc["MS2137", "candidate_gate_H"] < 0.7
    assert report["interpretation"]["per_cluster_gravity_retuning"] is False


def test_prospective_transfer_fails_every_advance_gate():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    gate = report["gate_audit"]
    assert gate["all_gates_pass"] is False
    assert gate["all_training_and_heldout_roots_each_system_pass"] is False
    assert gate["activation_response_ordering_pass"] is False
    assert gate["both_systems_not_worse_pass"] is False
    assert gate["equal_system_heldout_improvement_pass"] is False
    assert gate["equal_system_absolute_RMS_pass"] is False
    assert report["interpretation"]["P0610_gate_transfers_to_both_systems"] is False


def test_high_activation_system_has_no_material_valid_response():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    responses = {row["system_label"]: row for row in report["responses"]}
    scores = pd.read_csv(OUTPUT / "system_scores.csv").set_index(
        ["system_label", "variant_id"]
    )
    assert responses["MS2137"]["complete_pair"] is False
    assert responses["MS2137"]["heldout_pair_complete"] is True
    assert abs(responses["MS2137"]["heldout_only_diagnostic_improvement_fraction"]) < 0.001
    assert scores.loc[("MS2137", "P0599_no_route"), "training_roots_converged"] == 7
    assert scores.loc[("MS2137", "P0610_gated_gas_route"), "training_roots_converged"] == 7
    assert scores.loc[("MS2137", "P0599_no_route"), "training_images"] == 8


def test_near_zero_activation_system_is_an_optimizer_null_control():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    response = next(row for row in report["responses"] if row["system_label"] == "A383")
    assert response["complete_pair"] is True
    assert -0.01 < response["heldout_improvement_fraction"] < 0.0
