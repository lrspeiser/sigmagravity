import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_member_tidal_result_is_complete_and_failed():
    report = json.loads(
        (ROOT / "results/member_tidal_metric/report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "complete"
    assert report["selection"]["selected_t"] == 0.0
    assert report["validation"]["selected_tensor"]["all_roots_converged"]
    assert report["gate_audit"]["all_gates_pass"] is False
    assert report["verdict"]["member_tidal_metric_survives"] is False
    assert report["verdict"]["full_gas_inclusive_tensor_test_completed"] is False


def test_member_tidal_numerics_pass_but_halo_comparator_wins():
    report = json.loads(
        (ROOT / "results/member_tidal_metric/report.json").read_text(encoding="utf-8")
    )
    assert report["gate_audit"]["edge_Q_pass"]
    assert report["gate_audit"]["curl_pass"]
    tensor = report["validation"]["selected_tensor"]["equal_system_radial_RMS_arcsec"]
    halo = report["comparators"]["compact_halo_validation_RMS_arcsec"]
    assert tensor > halo
    assert report["randomization_control"]["degenerate_because_selected_t_zero"]


def test_post_result_nonzero_diagnostic_cannot_rescue_primary():
    diagnostic = json.loads(
        (ROOT / "results/member_tidal_metric/nonzero_transfer_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["status"] == "complete_post_result_nonqualifying"
    assert diagnostic["validation"]["-0.6"]["all_roots_converged"]
    assert diagnostic["validation"]["-0.6"]["equal_system_radial_RMS_arcsec"] > diagnostic[
        "comparators"
    ]["compact_halo_RMS_arcsec"]
    assert diagnostic["validation"]["0.9"]["all_roots_converged"] is False
    assert diagnostic["validation"]["1.2"]["all_roots_converged"] is False
