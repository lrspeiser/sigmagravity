import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/p0557_baryon_proxy_tidal"


def load_report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_frozen_protocol_has_universal_proxy_factorial():
    protocol = json.loads(
        (ROOT / "configs/p0557_baryon_proxy_tidal_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["status"].startswith("frozen_before_any_")
    assert len(protocol["morphology_variants"]) == 9
    assert len(protocol["tensor_operators"]) == 2
    assert protocol["tensor_coupling"]["grid"] == [
        -0.9,
        -0.6,
        -0.3,
        0.0,
        0.3,
        0.6,
        0.9,
    ]
    assert protocol["weak_field_equation"]["per_cluster_gravity_parameters"] == 0
    for variant in protocol["morphology_variants"]:
        assert np.isclose(
            variant["member_fraction"]
            + variant["star_fraction"]
            + variant["gas_fraction"],
            1.0,
        )


def test_all_predeclared_fixed_source_combinations_were_scored():
    report = load_report()
    screen = pd.read_csv(RESULTS / "screen_scores.csv")
    assert report["coverage"]["fixed_source_screen_combinations"] == 108
    assert len(screen) == 9 * 2 * 6
    assert screen.iloc[0].variant_id == "gas_sqrt"
    assert screen.iloc[0].operator_id == "contrast"
    assert np.isclose(screen.iloc[0].tensor_t, 0.3)
    shortlist = screen[screen.shortlisted_for_exact_selection.astype(bool)]
    assert len(shortlist) == 4
    assert set(shortlist.operator_id) == {"contrast"}
    assert set(shortlist.tensor_t) == {0.3}


def test_exact_selection_preserves_zero_control_and_root_gate():
    exact = pd.read_csv(RESULTS / "exact_selection_scores.csv")
    aggregate = exact[exact.row_type.eq("aggregate")].set_index("variant_id")
    assert "zero" in aggregate.index
    assert bool(aggregate.loc["zero", "all_training_and_heldout_roots"])
    assert not bool(aggregate.loc["gas_linear", "all_training_and_heldout_roots"])
    report = load_report()["exact_selection"]
    assert report["winner"]["variant_id"] == "star75_gas25"
    assert report["winner"]["operator_id"] == "contrast"
    assert np.isclose(report["winner"]["tensor_t"], 0.3)
    assert np.isclose(
        report["selected_improvement_fraction_vs_zero"],
        0.0004250082803611832,
    )


def test_validation_signal_is_small_and_cluster_specific():
    scores = pd.read_csv(RESULTS / "validation_scores.csv")
    zero = scores[(scores.row_type == "system") & (scores.variant_id == "zero")].set_index(
        "system_label"
    )
    selected = scores[
        (scores.row_type == "system") & (scores.variant_id == "star75_gas25")
    ].set_index("system_label")
    assert selected.loc["MACS1115", "heldout_exact_RMS_arcsec"] > zero.loc[
        "MACS1115", "heldout_exact_RMS_arcsec"
    ]
    assert selected.loc["MACS1931", "heldout_exact_RMS_arcsec"] < zero.loc[
        "MACS1931", "heldout_exact_RMS_arcsec"
    ]
    report = load_report()
    assert report["validation"]["selected_all_roots_converged"]
    assert np.isclose(
        report["validation"]["improvement_fraction"], 0.002147744409137342
    )
    assert np.isclose(
        report["comparators"]["selected_to_compact_halo_RMS_ratio"],
        1.8412655309131447,
    )


def test_full_radial_tensor_is_far_stronger_than_contrast_tensor():
    audits = pd.read_csv(RESULTS / "tensor_audits.csv")
    screen = audits[audits.stage.eq("fixed_source_screen")]
    full = screen[screen.operator_id.eq("full")]
    contrast = screen[screen.operator_id.eq("contrast")]
    assert full.correction_RMS_arcsec_at_distance_ratio_one.min() > 12.0
    assert contrast.correction_RMS_arcsec_at_distance_ratio_one.max() < 0.7
    assert full.correction_RMS_arcsec_at_distance_ratio_one.min() > 17.0 * contrast.correction_RMS_arcsec_at_distance_ratio_one.max()
    assert audits.normalized_curl_RMS.max() < 1.0e-12


def test_primary_softening_is_best_but_result_is_not_promoted():
    sensitivity = pd.read_csv(RESULTS / "softening_sensitivity.csv").set_index(
        "softening_kpc"
    )
    assert set(sensitivity.index) == {10.0, 20.0, 30.0}
    assert sensitivity.validation_fixed_fit_local_RMS_arcsec.idxmin() == 20.0
    report = load_report()
    assert not report["gate_audit"]["validation_RMS_improvement_pass"]
    assert not report["gate_audit"]["validation_to_compact_halo_pass"]
    assert report["gate_audit"]["edge_gate_pass"]
    assert report["gate_audit"]["curl_gate_pass"]
    assert not report["verdict"]["all_empirical_gates_pass"]
    assert not report["verdict"]["formula_promoted"]
    assert report["verdict"]["gas_mass_map_still_required"]


def test_galaxy_and_solar_controls_are_explicitly_unchanged_by_scope():
    control = load_report()["galaxy_and_solar_control"]
    assert control["formula_change"] == 0.0
    assert np.isclose(control["fixed_RAR_galaxy_outer_RMSE_km_s"], 10.348465773189679)
    assert control["maximum_abs_eta_minus_one_limb_to_Saturn"] == 0.0
