import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_spatial_vector_protocol_is_universal_and_frozen_before_scores():
    protocol = load("configs/unbounded_running_spatial_vector_protocol.json")
    assert protocol["status"] == "frozen_before_spatial_vector_scores"
    assert len(protocol["systems"]) == 4
    assert protocol["spatial_vector_grid"]["object_specific_gravity_parameters"] == 0
    assert protocol["spatial_vector_grid"]["mass_fractions"] == [
        0.0,
        0.025,
        0.05,
        0.1,
        0.2,
        0.4,
        0.8,
    ]


def test_spatial_vector_predictive_result_has_no_survivor():
    report = load("results/unbounded_running_spatial_vector/report.json")
    assert report["verdict"]["survivors"] == []
    assert len(report["spatial_variants"]) == 4
    for name, block in report["spatial_variants"].items():
        aggregate = block["aggregate_heldout"]
        assert aggregate["all_roots_converged"]
        assert aggregate["equal_system_radial_RMS_arcsec"] > 18.0
        assert report["gate_audit"][name]["compact_halo_RMS_ratio"] > 2.0
        assert not report["gate_audit"][name]["all_gates_pass"]


def test_spatial_vector_circularization_and_oracle_claim_boundary():
    audit = pd.read_csv(
        ROOT / "results/unbounded_running_spatial_vector/member_audit.csv"
    )
    assert len(audit) == 24
    assert audit.maximum_independent_circular_mean_residual_arcsec.max() < 0.007
    oracle = load("results/unbounded_running_spatial_vector_oracle/report.json")
    assert not oracle["verdict"]["predictive_survivor_created"]
    assert any("not predictions" in item for item in oracle["claim_boundary"])
    for block in oracle["best_universal_oracle"].values():
        assert block["all_roots_converged"]
        assert block["equal_system_radial_RMS_arcsec"] > 18.0


def test_common_200_kpc_control_does_not_rescue_aperture_mismatch():
    protocol = load(
        "configs/unbounded_running_spatial_vector_common200_control.json"
    )
    assert protocol["parent_protocol"].endswith("spatial_vector_protocol.json")
    assert protocol["overrides"]["baryonic_profile"][
        "normalization_aperture_kpc"
    ] == 200.0
    report = load("results/unbounded_running_spatial_vector_common200/report.json")
    assert report["verdict"]["survivors"] == []
    best = min(
        block["aggregate_heldout"]["equal_system_radial_RMS_arcsec"]
        for block in report["spatial_variants"].values()
        if block["aggregate_heldout"]["all_roots_converged"]
    )
    assert 18.20 < best < 18.22
    audit = pd.read_csv(
        ROOT / "results/unbounded_running_spatial_vector_common200/member_audit.csv"
    )
    assert set(audit.normalization_radius_kpc) == {200.0}
    assert audit.profile_maximum_radius_kpc.max() == 600.0
