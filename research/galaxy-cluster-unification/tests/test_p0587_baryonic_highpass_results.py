import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_report():
    return json.loads(
        (ROOT / "results/p0587_baryonic_highpass_metric/report.json").read_text(
            encoding="utf-8"
        )
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p0587_protocol_and_coverage_are_frozen():
    result = load_report()
    protocol = ROOT / result["protocol"]["path"]
    assert result["protocol"]["sha256"] == sha256(protocol)
    assert result["coverage"] == {
        "clusters": 4,
        "screen_candidates": 17,
        "screen_system_fields": 68,
        "exact_models": 3,
        "exact_system_fits": 12,
    }


def test_p0587_highpass_retains_small_gain_but_worsens_raw_metric():
    exact = load_report()["exact"]
    assert exact["zero_RMS_arcsec"] == pytest.approx(17.873967444043615)
    assert exact["raw_metric_RMS_arcsec"] == pytest.approx(17.63266639478282)
    assert exact["highpass_primary_RMS_arcsec"] == pytest.approx(
        17.673671353642888
    )
    assert exact["highpass_improvement_vs_zero_fraction"] > 0.011
    assert exact["highpass_improvement_vs_raw_fraction"] < 0.0
    assert exact["highpass_all_training_roots"]
    assert exact["highpass_all_heldout_roots"]


def test_p0587_grid_affine_removal_does_not_remove_image_sampled_affinity():
    result = load_report()
    audits = result["numerical"]["primary_field_audits"]
    macs1115 = next(row for row in audits if row["system_label"] == "MACS1115")
    assert macs1115["baryon_grid_affine_R2_before"] < 0.02
    assert macs1115["baryon_grid_affine_R2_after"] < 2.0e-9
    assert macs1115["affine_vector_R2_on_images"] > 0.991
    assert not result["gates"]["affine_audit_pass"]


def test_p0587_exact_system_pattern_matches_raw_branch():
    exact = pd.read_csv(ROOT / "results/p0587_baryonic_highpass_metric/exact_scores.csv")
    systems = exact[exact.row_type.eq("system")].pivot(
        index="system_label", columns="model_id", values="heldout_exact_RMS_arcsec"
    )
    assert systems.loc["MACS0329", "highpass_primary"] < systems.loc["MACS0329", "zero"]
    assert systems.loc["MACS1931", "highpass_primary"] < systems.loc["MACS1931", "zero"]
    assert systems.loc["MACS0429", "highpass_primary"] > systems.loc["MACS0429", "zero"]
    assert systems.loc["MACS1115", "highpass_primary"] > systems.loc["MACS1115", "zero"]
    assert not load_report()["gates"]["primary_all_four_systems_improve"]


def test_p0587_parameter_spans_are_tiny_and_formula_is_not_promoted():
    impacts = {row["coordinate"]: row for row in load_report()["parameter_impacts"]}
    assert impacts["aperture_r80_fraction"]["main_effect_span_arcsec"] < 0.019
    assert impacts["mode"]["main_effect_span_arcsec"] < 0.014
    assert impacts["removal_fraction"]["main_effect_span_arcsec"] < 0.0043
    gates = load_report()["gates"]
    assert gates["primary_all_roots"]
    assert gates["curl_pass"]
    assert gates["positive_metric_pass"]
    assert not gates["primary_improvement_vs_raw_pass"]
    assert not gates["compact_halo_ratio_pass"]
    assert not gates["formula_promoted"]
