import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_report() -> dict:
    return json.loads(
        (
            ROOT
            / "results/p0581_locked_endpoint_exact_root/report.json"
        ).read_text(encoding="utf-8")
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p0581_protocol_and_exact_root_coverage_are_frozen():
    result = load_report()
    protocol = ROOT / result["protocol"]["path"]
    assert result["protocol"]["sha256"] == sha256(protocol)
    assert result["coverage"] == {
        "clusters": 4,
        "members": 282,
        "training_images": 44,
        "heldout_images": 11,
        "sensitivity_fields": 44,
    }


def test_p0581_primary_fails_transfer_with_different_missing_root_than_scalar():
    result = load_report()
    primary = result["exact_aggregate"]["K0338_primary"]
    scalar = result["exact_aggregate"]["scalar_baseline"]
    assert primary["complete_systems"] == 3
    assert scalar["complete_systems"] == 3
    systems = {row["system_label"]: row for row in result["per_system"]}
    assert systems["MACS0329"]["scalar_all_roots"]
    assert not systems["MACS0329"]["primary_all_roots"]
    assert not systems["MACS1931"]["scalar_all_roots"]
    assert systems["MACS1931"]["primary_all_roots"]
    assert result["systems_improved_or_root_recovered"] == 2


def test_p0581_matched_complete_systems_are_slightly_worse_than_scalar():
    matched = load_report()["matched_primary_vs_scalar_all_four"]
    assert matched["matched_labels"] == ["MACS0429", "MACS1115"]
    assert not matched["all_requested_systems_comparable"]
    assert matched["reference_RMS_arcsec"] == pytest.approx(20.305658089727142)
    assert matched["candidate_RMS_arcsec"] == pytest.approx(20.448608476635993)
    assert matched["fractional_improvement"] == pytest.approx(-0.0070399287862121795)


def test_p0581_lower_contrast_caps_restore_all_exact_roots_posthoc():
    impacts = {
        row["parameter"]: row for row in load_report()["root_topology_impacts"]
    }
    contrast = impacts["contrast_cap"]
    assert contrast["heldout_converged_roots_by_level"] == "5.0:11+10.0:11+20.0:10"
    assert contrast["complete_systems_by_level"] == "5.0:4+10.0:4+20.0:3"
    assert contrast["best_root_count_levels"] == "5.0+10.0"
    assert contrast["heldout_impact_span_arcsec"] < 0.035


def test_p0581_gate_has_largest_root_topology_span_and_length_has_interior_peak():
    impacts = {
        row["parameter"]: row for row in load_report()["root_topology_impacts"]
    }
    assert impacts["gate_mode"]["converged_root_span"] == 3
    assert impacts["gate_mode"]["complete_system_span"] == 2
    length = impacts["return_length_over_R80"]
    assert length["heldout_converged_roots_by_level"] == "0.3:8+0.36:10+0.42:8"
    assert length["best_root_count_levels"] == "0.36"


def test_p0581_field_audits_pass_but_all_performance_gates_fail():
    result = load_report()
    audit = result["field_audit"]
    assert audit["maximum_route_map_normalization_error"] < 1e-12
    assert audit["maximum_annular_convergence_mean_fraction"] < 1e-12
    assert audit["maximum_normalized_curl_RMS"] < 1e-12
    assert result["comparators"]["primary_validation_to_compact_halo_ratio"] > 1.9
    gates = result["gate_audit"]
    assert gates["route_map_normalization_pass"]
    assert gates["annular_monopole_pass"]
    assert gates["curl_free_pass"]
    assert gates["solar_axisymmetric_zero_monopole_pass"]
    assert not gates["all_gates_pass"]
