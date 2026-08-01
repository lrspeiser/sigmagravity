import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "solar_screened_improver_forensics_protocol.json"
OUTPUT = ROOT / "results" / "solar_screened_improver_forensics"
REPORT = OUTPUT / "report.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_forensic_sample_and_protocol_integrity() -> None:
    report = _load(REPORT)
    assert report["protocol"]["sha256"] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert report["sample"]["galaxies"] == 131
    assert report["sample"]["primary_improvers"] == 41
    assert report["sample"]["primary_worseners"] == 90
    assert report["sample"]["features_in_inventory"] == 396


def test_exhaustive_declared_scenario_counts_are_preserved() -> None:
    report = _load(REPORT)
    assert report["sample"]["numeric_univariate_tests"] == 303
    assert report["sample"]["categorical_tests"] == 8
    assert report["sample"]["one_feature_rules_scanned"] == 53_522
    assert report["sample"]["pairwise_rules_scanned"] == 156_967


def test_feature_family_ablation_separates_mechanism_from_environment() -> None:
    report = _load(REPORT)
    rows = {
        (row["task"], row["feature_set"], row["model"]): row
        for row in report["cross_validation"]["aggregate"]
    }
    environment = rows[("classification", "environment_only", "random_forest")]
    mechanism = rows[("classification", "mechanistic_inner_fit_only", "random_forest")]
    combined = rows[("classification", "core_plus_mechanistic", "random_forest")]
    assert 0.48 < environment["aggregate_repeated_OOF_ROC_AUC"] < 0.54
    assert mechanism["aggregate_repeated_OOF_ROC_AUC"] > 0.80
    assert combined["aggregate_repeated_OOF_ROC_AUC"] > 0.80


def test_mechanism_diagnostics_preserve_outer_residual_pattern() -> None:
    report = _load(REPORT)
    rows = {row["group"]: row for row in report["outcome_mechanism_diagnostics"]}
    assert rows["improvers"][
        "mean_outer_RAR_residual_predicted_minus_observed_km_s"
    ] > 10.0
    assert abs(
        rows["improvers"][
            "mean_outer_screened_residual_predicted_minus_observed_km_s"
        ]
    ) < 2.0
    assert rows["worseners"][
        "mean_outer_screened_residual_predicted_minus_observed_km_s"
    ] < -17.0
    assert rows["improvers"]["tail_helps_at_same_screened_nuisance_fraction"] == 1.0


def test_complete_improver_roster_and_reproducibility_artifacts_exist() -> None:
    with (OUTPUT / "complete_improver_roster.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 41
    assert len({row["galaxy"] for row in rows}) == 41
    assert rows[0]["galaxy"] == "UGC07577"
    for name in (
        "galaxy_outcomes.csv",
        "galaxy_features.csv",
        "feature_manifest.csv",
        "univariate_numeric_tests.csv",
        "categorical_tests.csv",
        "one_feature_subgroup_scan.csv",
        "pairwise_subgroup_scan.csv",
        "cross_validation_aggregate.csv",
        "cross_validation_summary.csv",
        "cross_validated_predictions.csv",
        "core_logistic_permutations.csv",
        "top_rule_bootstrap_intervals.csv",
        "robustness_scenarios.csv",
        "leave_one_out_influence.csv",
        "forensic_summary.png",
    ):
        assert (OUTPUT / name).stat().st_size > 0
