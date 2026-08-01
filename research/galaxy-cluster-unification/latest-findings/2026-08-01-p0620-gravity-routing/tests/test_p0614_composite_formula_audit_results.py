import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0614_composite_formula_audit"


def test_composite_uses_one_equation_and_zero_object_gravity_parameters():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["composite_equation"]["per_object_gravity_parameters"] == 0
    assert report["parameter_accounting"]["one_parameter_theory"] is False
    assert report["coverage"]["SPARC_galaxies"] == 131
    assert report["coverage"]["raw_factorial_clusters"] == 4


def test_scalar_parent_is_near_galaxy_comparators_but_endpoint_fails_raw_transfer():
    scorecard = pd.read_csv(OUTPUT / "scorecard.csv")
    rar = scorecard[scorecard.comparator.eq("fixed RAR")].iloc[0]
    transfer = scorecard[scorecard.domain.eq("RXJ2129 route transfer")].iloc[0]
    assert 1.0 < rar.ratio_to_comparator < 1.5
    assert transfer.ratio_to_comparator > 10.0


def test_solar_pass_does_not_hide_cluster_failure():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    gates = report["gates"]
    assert gates["Solar_all_proxies_pass"] is True
    assert gates["raw_four_cluster_roots_pass"] is True
    assert gates["raw_validation_near_compact_halo_pass"] is False
    assert gates["RXJ2129_route_transfer_improvement_pass"] is False
    assert gates["composite_unification_pass"] is False
    assert report["interpretation"]["formula_promoted"] is False
