import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0612_cross_stage_parameter_impact"


def test_atlas_covers_cluster_galaxy_and_topology_metrics():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    observations = pd.read_csv(OUTPUT / "observations.csv")
    assert report["coverage"]["stages"] >= 14
    assert report["coverage"]["impact_observations"] == len(observations)
    assert {"cluster_raw_lensing", "cluster_reconstructed_lensing", "galaxy_rotation"}.issubset(
        set(observations.domain)
    )
    assert {"performance", "topology", "topology_risk"}.issubset(
        set(observations.metric_type)
    )


def test_normalization_is_within_metric_and_bounded():
    observations = pd.read_csv(OUTPUT / "observations.csv")
    assert observations.normalized_leverage.between(0.0, 1.0).all()
    grouped = observations.groupby(["stage_id", "domain", "observable", "metric"])
    assert (grouped.normalized_leverage.max() == 1.0).all()


def test_fraction_is_explosive_but_not_mislabeled_as_success():
    observations = pd.read_csv(OUTPUT / "observations.csv")
    p0606 = observations[
        observations.stage_id.eq("P0606") & observations.coordinate.eq("fraction_max")
    ]
    assert set(p0606.metric_type) == {"performance", "topology_risk"}
    assert (p0606.normalized_leverage == 1.0).all()
    assert p0606.transfer_outcome.str.contains("zero_route_selected").all()


def test_width_and_path_are_cross_domain_but_gate_transfer_failed():
    family = pd.read_csv(OUTPUT / "family_summary.csv").set_index("coordinate_family")
    transfer = pd.read_csv(OUTPUT / "transfer_evidence.csv")
    assert family.loc["spatial_width", "domain_count"] >= 3
    assert family.loc["path_length", "domain_count"] >= 3
    assert transfer.loc[transfer.stage_id.eq("P0611"), "passes"].iloc[0] in [False, 0]


def test_report_does_not_promote_a_formula():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert "sensitivity" in report["normalization"]["claim_limit"].lower()
    assert any("High impact includes harmful" in item for item in report["claim_limits"])
    assert report["next_test"]["formula_family"].startswith("bounded endpoint")
