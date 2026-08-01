import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ABS_CONFIG = ROOT / "configs" / "p0554_macs1931_companion_audit_protocol.json"
REL_CONFIG = ROOT / "configs" / "p0554_macs1931_relative_companion_protocol.json"
ABS_RESULTS = ROOT / "results" / "p0554_macs1931_companion_audit"
REL_RESULTS = ROOT / "results" / "p0554_macs1931_relative_companion"


def report(directory):
    return json.loads((directory / "report.json").read_text(encoding="utf-8"))


def test_absolute_and_relative_protocols_were_frozen_before_their_pixel_tests():
    absolute = json.loads(ABS_CONFIG.read_text(encoding="utf-8"))
    relative = json.loads(REL_CONFIG.read_text(encoding="utf-8"))
    assert absolute["status"].startswith("frozen_")
    assert "before_any_hst_pixel" in absolute["status"]
    assert relative["status"].startswith("frozen_")
    assert "before_any_anchor_registered_companion_pixel_inspection" in relative["status"]
    assert absolute["selection"]["formula_parameters_fit"] == 0
    assert absolute["selection"]["geometry_parameters_refit"] == 0
    assert relative["prediction"]["translation_parameters_fit"] == 0
    assert relative["prediction"]["rotation_or_scale_allowed"] is False


def test_absolute_audit_coverage_and_signed_pair_predictions():
    result = report(ABS_RESULTS)
    assert result["coverage"] == {
        "five_root_variants": 11,
        "near_2c_roots": 22,
        "companion_predictions": 11,
        "position_groups": 13,
        "published_MACS1931_images": 19,
        "published_family2_images": 3,
    }
    predictions = result["model_predictions"]
    assert predictions["opposite_parity_pairs"] == 11
    assert predictions["same_parity_pairs"] == 0
    assert np.allclose(
        predictions["companion_to_anchor_flux_ratio_range"],
        [0.9171598179167316, 1.2509463971636319],
    )
    assert result["reference_image_2c"]["formal_snr"] > 100.0


def test_uncorrected_absolute_positions_are_all_formally_blank_and_uncatalogued():
    result = report(ABS_RESULTS)
    assert result["catalog_audit"]["companion_predictions_matching_published_family2"] == 0
    assert result["formal_photometry"]["companion_predictions_with_valid_weight"] == 11
    assert result["formal_photometry"]["companion_predictions_with_formal_source"] == 0
    assert result["formal_photometry"]["formal_blank_rejection_candidates_before_visual_audit"] == 11
    assert result["verdict"]["published_catalog_confirms_extra_family2_image"] is False
    assert result["verdict"]["no_formula_promoted"] is True


def test_anchor_registration_yields_five_distinct_test_locations():
    result = report(REL_RESULTS)
    assert result["status"] == "complete_with_manual_visual_audit"
    assert result["coverage"] == {
        "variants": 11,
        "registered_position_groups": 5,
        "published_family2_images": 3,
    }
    predictions = result["registered_predictions"]
    assert np.allclose(
        predictions["pair_separation_range_arcsec"],
        [1.7815587280200649, 10.859260038890895],
    )
    assert predictions["catalogued_family2_matches"] == 0
    assert predictions["formal_sources_at_registered_positions"] == 2
    assert predictions["formal_blanks"] == 9


def test_registered_pairs_are_opposite_parity_and_near_equal_brightness():
    audit = pd.read_csv(REL_RESULTS / "registered_companion_audit.csv")
    assert len(audit) == 11
    assert audit.pair_parities.eq("negative;positive").all()
    assert audit.predicted_companion_to_anchor_flux_ratio.between(0.91, 1.26).all()
    indexed = audit.set_index("variant_id")
    assert np.isclose(indexed.loc["combined_parent", "pair_separation_arcsec"], 1.7815587280200649)
    assert np.isclose(indexed.loc["combined_power_240", "pair_separation_arcsec"], 10.859260038890895)
    assert indexed.catalogued_family2_match.astype(bool).sum() == 0


def test_manual_visual_audit_finds_no_plausible_centered_counterimage():
    result = report(REL_RESULTS)
    assert result["visual_annotation"]["counts"] == {
        "clean_blank": 10,
        "neighbor_contaminated_nonmatching": 1,
    }
    assert result["verdict"] == {
        "published_catalog_confirms_registered_companion": False,
        "all_registered_positions_formally_blank": False,
        "visual_audit_complete": True,
        "plausible_centered_uncatalogued_counterimages": 0,
        "variants_with_clean_blank_companion": 10,
        "variants_with_contaminated_inconclusive_position": 1,
        "registered_extra_pair_supported_by_f160w": False,
        "no_formula_promoted": True,
    }
    audit = pd.read_csv(REL_RESULTS / "registered_companion_audit.csv")
    assert audit.visual_classification.eq("clean_blank").sum() == 10
    contaminated = audit[audit.visual_classification.eq("neighbor_contaminated_nonmatching")]
    assert contaminated.variant_id.tolist() == ["combined_lens_099"]


def test_visual_annotation_covers_every_variant_once():
    visual = json.loads((REL_RESULTS / "visual_assessment.json").read_text(encoding="utf-8"))
    annotations = pd.DataFrame(visual["annotations"])
    audit = pd.read_csv(REL_RESULTS / "registered_companion_audit.csv")
    assert len(annotations) == 11
    assert annotations.variant_id.is_unique
    assert set(annotations.variant_id) == set(audit.variant_id)
