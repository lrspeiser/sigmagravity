import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_multicluster_protocol_keeps_gravity_settings_locked():
    protocol = load("configs/unbounded_running_multicluster_raw_protocol.json")
    assert protocol["status"] == "frozen_before_multicluster_raw_scores"
    assert len(protocol["systems"]) == 6
    for specification in protocol["models"].values():
        assert specification["gravity_parameters_fit_to_cluster"] == 0
    assert protocol["photon_and_environment_closure"]["gravitational_slip"] == 0.0
    assert protocol["photon_and_environment_closure"]["lensing_multiplier"] == 1.0


def test_multicluster_raw_result_is_a_failure_with_explicit_claim_boundary():
    report = load("results/unbounded_running_multicluster_raw/report.json")
    assert len(report["unseen_raw_observable_systems"]) == 4
    assert report["verdict"]["survivors"] == []
    for model in ("curvature_power_p2", "curvature_additive_alpha10"):
        audit = report["gate_audit"][model]
        assert audit["all_roots_converged"]
        assert audit["beats_simple_MOND"]
        assert not audit["absolute_RMS_pass"]
        assert not audit["within_compact_halo_ratio_gate"]
        assert audit["cutoff_robustness_pass"]
    assert any("not an external-system validation" in item for item in report["claim_boundary"])


def test_postfailure_amplitude_diagnostic_does_not_create_a_survivor():
    report = load("results/unbounded_running_multicluster_failure_diagnostic/report.json")
    assert not report["verdict"]["any_normalization_rescue"]
    assert not report["verdict"]["any_amplitude_universal"]
    assert len(report["verdict"]["compact_halo_adequate_systems"]) == 2
