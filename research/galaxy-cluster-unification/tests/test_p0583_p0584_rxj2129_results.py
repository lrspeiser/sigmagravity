import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def report(name: str) -> dict:
    return json.loads(
        (ROOT / "results" / name / "report.json").read_text(encoding="utf-8")
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p0583_protocol_and_rxj2129_coverage_are_frozen():
    result = report("p0583_tanh_endpoint_rxj2129")
    protocol = ROOT / result["protocol"]["path"]
    assert result["protocol"]["sha256"] == sha256(protocol)
    assert result["coverage"] == {
        "hard_photometric_sources": 51,
        "training_images": 15,
        "heldout_images": 7,
        "variants": 4,
    }


def test_p0583_tanh_candidate_keeps_roots_but_fails_image_placement():
    result = report("p0583_tanh_endpoint_rxj2129")
    scores = {row["variant"]: row for row in result["scores"]}
    scalar = scores["scalar_baseline"]
    tanh = scores["K0338_tanh20_candidate"]
    assert scalar["heldout_RMS_arcsec"] == pytest.approx(1.256170108342776)
    assert tanh["heldout_converged_roots"] == 7
    assert tanh["heldout_RMS_arcsec"] == pytest.approx(14.129878355246074)
    assert tanh["fractional_improvement_vs_scalar"] < -10.0
    assert tanh["geometry_at_boundary"]
    assert not result["gate_audit"]["all_gates_pass"]


def test_p0583_saturation_shape_does_not_rescue_rxj2129():
    scores = {
        row["variant"]: row
        for row in report("p0583_tanh_endpoint_rxj2129")["scores"]
    }
    hard5 = scores["K0338_hard5_control"]["heldout_RMS_arcsec"]
    tanh20 = scores["K0338_tanh20_candidate"]["heldout_RMS_arcsec"]
    assert abs(hard5 - tanh20) < 0.006
    assert scores["K0338_hard20_parent"]["heldout_RMS_arcsec"] > 15.2


def test_p0583b_zero_wins_signed_screen_and_positive_route_is_misaligned():
    result = report("p0583b_signed_endpoint_amplitude")
    assert result["best_amplitude"]["epsilon"] == 0.0
    assert result["best_negative_amplitude"]["heldout_RMS_arcsec"] > 1.6
    assert result["best_positive_amplitude"]["heldout_converged_roots"] == 6
    alignment = result["local_directional_alignment"]
    assert alignment["images_with_both_signed_roots"] == 6
    assert alignment["images_improved_to_first_order_by_positive_route"] == 2
    assert alignment["image_lost_at_positive_0p025"] == "2c"


def test_p0584_original_rule_crosses_center_with_nearly_half_the_source_weight():
    result = report("p0584_no_overshoot_endpoint")
    audits = {row["travel_mode"]: row for row in result["travel_audits"]}
    assert audits["constant"]["sources_crossing_center"] == 15
    assert audits["constant"]["source_weight_crossing_center"] == pytest.approx(
        0.47798865348073166
    )
    assert audits["tanh_no_cross"]["source_weight_crossing_center"] == 0.0
    assert audits["rational_no_cross"]["source_weight_crossing_center"] == 0.0


def test_p0584_tanh_no_cross_is_best_but_only_at_tiny_posthoc_amplitude():
    result = report("p0584_no_overshoot_endpoint")
    winner = result["winner_including_zero"]
    assert winner["travel_mode"] == "tanh_no_cross"
    assert winner["epsilon"] == 0.005
    assert winner["heldout_RMS_arcsec"] == pytest.approx(1.2430077939002733)
    improvement = (1.2561701083427703 - winner["heldout_RMS_arcsec"]) / 1.2561701083427703
    assert 0.01 < improvement < 0.011
    rows = {
        (row["travel_mode"], row["epsilon"]): row for row in result["scores"]
    }
    assert not rows[("tanh_no_cross", 0.01)]["heldout_all_roots"]
    assert rows[("hard_no_cross", 0.01)]["heldout_RMS_arcsec"] > 1.256
