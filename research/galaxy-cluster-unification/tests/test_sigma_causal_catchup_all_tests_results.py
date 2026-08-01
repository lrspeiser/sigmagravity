import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sigma_causal_catchup_all_tests"


def load_report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_all_existing_static_rows_are_exactly_invariant():
    report = load_report()
    coverage = report["coverage"]["static_invariance"]

    assert coverage["SPARC"] == {"rows": 3034, "systems": 131}
    assert coverage["CLASH"] == {"rows": 84, "systems": 20}
    assert coverage["BCG"] == {"rows": 44, "systems": 44}
    assert coverage["RXJ2129_raw_lensing"] == {"rows": 66, "systems": 22}
    assert report["static_test"]["maximum_absolute_prediction_change"] == 0.0
    assert report["gate_audit"]["static_invariance_pass"] is True


def test_causal_completion_repairs_the_scalar_cone_but_is_not_validated():
    report = load_report()
    audit = report["causal_characteristic_test"]
    gates = report["gate_audit"]

    assert audit["maximum_characteristic_speed_over_c_across_scan"] == pytest.approx(1.0)
    assert audit["universal_selected_delta"] == pytest.approx(100.0)
    assert audit["selected_maximum_response_parameter"] < 0.1
    assert gates["causal_characteristics_pass"] is True
    assert gates["mathematical_audit_pass"] is True
    assert gates["full_covariant_action_derived"] is False
    assert gates["time_dependent_observational_validation_available"] is False
    assert gates["theory_validated_pass"] is False


def test_safe_delta_has_small_galaxy_and_cluster_response():
    selected = load_report()["causal_characteristic_test"]["selected_by_domain"]

    assert selected["SPARC_outer"]["median_response_enhancement"] == pytest.approx(
        1.0001048148
    )
    assert selected["cluster"]["median_response_enhancement"] == pytest.approx(
        1.0038610988
    )
    assert selected["cluster"]["maximum_response_enhancement"] == pytest.approx(
        1.0136902366
    )


def test_large_delta_scan_exposes_resonant_breakdown():
    scan = pd.read_csv(RESULTS / "delta_scan.csv")
    assert set(scan["delta"]) == {
        0.0,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
    }
    assert (scan.loc[scan["delta"].eq(1.0e6), "fraction_at_or_above_resonance"] > 0).any()
    characteristics = pd.read_csv(RESULTS / "characteristics.csv")
    assert len(characteristics) == (968 + 72 + 44) * 8
