import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "metric_slip_raw_lensing" / "report.json"


def test_metric_slip_report_reproduces_the_frozen_verdict():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    protocol_path = ROOT / report["protocol"]["path"]
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()

    assert protocol_hash == report["protocol"]["sha256"]
    assert report["selection"]["selected_slip_s"] == 5.0
    assert report["selection"]["extra_force_lensing_to_dynamics_ratio"] == 3.5
    assert report["selection"]["selection_training_aggregate"][
        "all_roots_converged"
    ]

    validation = report["cross_cluster_validation"]
    selected = validation["selected_slip"]["equal_system_radial_RMS_arcsec"]
    zero = validation["zero_slip"]["equal_system_radial_RMS_arcsec"]
    halo = report["comparators"]["compact_halo_validation"][
        "equal_system_radial_RMS_arcsec"
    ]
    assert selected < zero
    assert selected > halo

    gates = report["gate_audit"]
    assert gates["selection_all_roots_pass"]
    assert gates["far_tail_robustness_pass"]
    assert gates["Solar_System_eta_pass"]
    assert not gates["cross_cluster_compact_halo_ratio_pass"]
    assert not gates["cross_cluster_absolute_RMS_pass"]
    assert not gates["all_gates_pass"]
    assert not report["verdict"]["universal_metric_slip_survives"]
