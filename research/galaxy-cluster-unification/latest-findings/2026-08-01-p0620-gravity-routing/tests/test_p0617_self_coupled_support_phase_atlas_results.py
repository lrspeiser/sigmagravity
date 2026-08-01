import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0617_self_coupled_support_phase_atlas"


def test_protocol_was_frozen_and_has_no_fitted_gravity_parameter():
    protocol = json.loads(
        (ROOT / "configs/p0617_self_coupled_support_phase_atlas_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_after_P0616_before_support_phase_scores"
    assert report["coverage"]["new_fitted_gravity_parameters"] == 0
    assert report["coverage"]["support_phase_variants"] == 19


def test_all_variants_have_all_five_system_scores():
    scores = pd.read_csv(OUTPUT / "scores.csv")
    assert len(scores) == 100
    assert scores.system_label.nunique() == 5
    assert scores.variant_id.nunique() == 20
    assert scores.groupby("variant_id").size().eq(5).all()


def test_response_and_impact_accounting_are_complete():
    responses = pd.read_csv(OUTPUT / "variant_responses.csv")
    impacts = pd.read_csv(OUTPUT / "family_impacts.csv")
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert len(responses) == 19
    assert set(impacts.family) == {"baseline", "width", "return_length", "center_crossing", "joint"}
    assert report["interpretation"]["formula_promoted"] is False
    assert "all_diagnostic_gates_pass" in report["gates"]


def test_fixed_route_alignment_explains_the_response_sign_without_becoming_a_fit():
    alignment = pd.read_csv(OUTPUT / "residual_alignment.csv")
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert len(alignment) == 5
    assert (
        report["interpretation"][
            "residual_alignment_sign_matches_all_five_responses"
        ]
        is True
    )
    assert (
        (alignment.weighted_alignment_cosine < 0.0)
        == (alignment.fixed_route_improvement > 0.0)
    ).all()
