import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_report() -> dict:
    return json.loads(
        (
            ROOT / "results/p0582_smooth_endpoint_saturation/report.json"
        ).read_text(encoding="utf-8")
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p0582_protocol_and_saturation_grid_are_frozen():
    result = load_report()
    protocol = ROOT / result["protocol"]["path"]
    assert result["protocol"]["sha256"] == sha256(protocol)
    assert result["coverage"] == {
        "clusters": 4,
        "contrast_modes": 4,
        "nominal_caps": 6,
        "variants": 24,
        "cluster_fields": 96,
        "heldout_images_per_variant": 11,
    }


def test_p0582_finds_five_complete_diagnostic_variants():
    result = load_report()
    assert result["all_four_complete_variants"] == 5
    winner = result["diagnostic_winner"]
    assert winner["variant"] == "hard_A5p0"
    assert winner["heldout_converged_roots"] == 11
    assert winner["equal_complete_system_RMS_arcsec"] == pytest.approx(
        19.040150303223694
    )


def test_p0582_tanh_twenty_is_the_only_complete_smooth_form():
    rows = {row["variant"]: row for row in load_report()["summary_grid"]}
    assert rows["tanh_A20p0"]["all_four_complete"]
    assert rows["tanh_A20p0"]["heldout_converged_roots"] == 11
    assert rows["tanh_A20p0"]["equal_complete_system_RMS_arcsec"] == pytest.approx(
        19.159352387520663
    )
    complete_smooth = [
        row
        for row in rows.values()
        if row["contrast_mode"] != "hard" and row["all_four_complete"]
    ]
    assert [row["variant"] for row in complete_smooth] == ["tanh_A20p0"]


def test_p0582_same_nominal_scale_gives_four_different_root_counts():
    cap20 = {
        row["contrast_mode"]: row for row in load_report()["cap20_mode_comparison"]
    }
    assert cap20["tanh"]["heldout_converged_roots"] == 11
    assert cap20["hard"]["heldout_converged_roots"] == 10
    assert cap20["exponential"]["heldout_converged_roots"] == 9
    assert cap20["rational"]["heldout_converged_roots"] == 8


def test_p0582_response_window_is_imposed_by_opposite_cluster_limits():
    windows = {
        row["system_label"]: row for row in load_report()["field_response_windows"]
    }
    assert windows["MACS0329"]["complete_variants"] == 23
    assert windows["MACS0329"]["correction_RMS_root_spearman"] < 0.0
    assert windows["MACS1931"]["complete_variants"] == 6
    assert windows["MACS1931"]["correction_RMS_root_spearman"] == pytest.approx(
        0.7878725900388478
    )
    assert windows["MACS0429"]["complete_variants"] == 24
    assert windows["MACS1115"]["complete_variants"] == 24


def test_p0582_field_is_conservative_but_nominal_cap_is_not_final_weight_bound():
    result = load_report()
    audit = result["field_audit"]
    assert audit["maximum_annular_convergence_mean_fraction"] < 1e-12
    assert audit["maximum_normalized_curl_RMS"] < 1e-12
    assert audit["maximum_postnormalization_light_weight"] > 700.0
    gates = result["gate_audit"]
    assert gates["annular_monopole_pass"]
    assert gates["curl_free_pass"]
    assert gates["solar_axisymmetric_zero_monopole_pass"]
    assert gates["all_diagnostic_gates_pass"]
