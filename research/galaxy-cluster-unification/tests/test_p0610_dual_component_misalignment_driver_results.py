import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0610_dual_component_misalignment_driver"


def test_candidate_gate_is_sharp_only_for_macs0429():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    table = pd.read_csv(OUTPUT / "driver_table.csv").set_index("system_label")
    assert report["coverage"]["systems_with_direction_maps"] == 5
    assert report["coverage"]["systems_with_finite_raw_response"] == 4
    assert report["candidate_gate"]["largest_activation_system"] == "MACS0429"
    assert table.loc["MACS0429", "candidate_gate_H"] > 0.90
    assert table.drop(index="MACS0429").candidate_gate_H.max() < 0.20


def test_same_data_pattern_is_not_mislabeled_as_validation():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    correlations = report["correlations"]
    interpretation = report["interpretation"]
    assert correlations["Pearson_r"] > 0.90
    assert correlations["Spearman_p"] > 0.05
    assert interpretation["same_data_evidence"] is False
    assert interpretation["candidate_for_fresh_predeclared_gate"] is True
    assert interpretation["single_outlier_dominates"] is True


def test_leave_one_out_audit_exposes_high_leverage_system():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    jackknife = pd.read_csv(OUTPUT / "jackknife.csv").set_index("omitted_system")
    assert report["correlations"]["minimum_leave_one_out_Pearson_r"] < 0.30
    assert jackknife.loc["MACS0429", "Pearson_r"] < 0.30
    assert jackknife.drop(index="MACS0429").Pearson_r.min() > 0.90
